from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_phase39_confirmatory_grouped_cv import (  # noqa: E402
    build_event_folds,
    make_outer_split,
    split_assignment_sha256,
)
from src.data.dataset_v2 import CorrectedEarthquakeDataset  # noqa: E402
from src.data.loaders_v2 import get_data_loaders_v2  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    aggregate_event_predictions,
    summarize_predictions,
)
from src.models.model import PINNModel  # noqa: E402
from src.training.checkpointing import atomic_torch_save  # noqa: E402
from src.training.loss_stf_rate_v2 import moment_magnitude_from_rate  # noqa: E402
from src.training.train import (  # noqa: E402
    _batch_event_sample_weights,
    _build_stf_rate_criterion,
    _prepare_v2_batch,
    _training_event_balance_weights,
)
from src.utils.config_v2 import validate_config_on_startup  # noqa: E402
from src.utils.device import configure_runtime  # noqa: E402
from src.utils.provenance import (  # noqa: E402
    current_git_commit,
    git_is_dirty,
    sha256_file,
)


DEFAULT_EXPERIMENT_CONFIG = (
    PROJECT_ROOT / "configs" / "experiments" / "phase39_causal_direct.yaml"
)
SOURCE_STEPS = 200
EXPECTED_PHASE39_PARAMETER_COUNT = 1_010_850


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prefix_horizon_for_step(
    step: int,
    *,
    seed: int,
    minimum: int,
    maximum: int,
    multiplier: int,
) -> int:
    if step < 0 or minimum < 1 or maximum < minimum:
        raise ValueError("invalid prefix horizon schedule")
    span = maximum - minimum + 1
    return minimum + ((seed + multiplier * step) % span)


def ordered_prefix_horizons_for_step(
    step: int,
    *,
    seed: int,
    minimum: int,
    maximum: int,
    multiplier: int,
    gap_sec: int,
) -> tuple[int, int]:
    if gap_sec < 1 or maximum - gap_sec < minimum:
        raise ValueError("prefix pair gap does not fit the horizon range")
    early = prefix_horizon_for_step(
        step,
        seed=seed,
        minimum=minimum,
        maximum=maximum - gap_sec,
        multiplier=multiplier,
    )
    return early, early + gap_sec


def causal_curriculum_scale(
    epoch: int,
    *,
    start_epoch: int,
    full_epoch: int,
) -> float:
    if epoch < 1 or start_epoch < 0 or full_epoch <= start_epoch:
        raise ValueError("invalid causal curriculum schedule")
    if epoch <= start_epoch:
        return 0.0
    if epoch >= full_epoch:
        return 1.0
    return float(epoch - start_epoch) / float(full_epoch - start_epoch)


def error_descent_loss(
    prefix_mw: torch.Tensor,
    endpoint_mw: torch.Tensor,
    target_mw: torch.Tensor,
    *,
    slack_mw: float,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if slack_mw < 0.0 or not math.isfinite(slack_mw):
        raise ValueError("slack_mw must be finite and nonnegative")
    prefix_error = torch.abs(prefix_mw.reshape(-1) - target_mw.reshape(-1))
    endpoint_error = torch.abs(endpoint_mw.reshape(-1) - target_mw.reshape(-1))
    violation = F.relu(endpoint_error - prefix_error.detach() - float(slack_mw))
    per_sample = violation.square()
    if sample_weights is None:
        loss = per_sample.mean()
    else:
        weights = sample_weights.reshape(-1).to(
            device=per_sample.device,
            dtype=per_sample.dtype,
        )
        if weights.shape != per_sample.shape:
            raise ValueError("sample_weights must match the magnitude batch")
        loss = (weights * per_sample).mean()
    return loss, {
        "prefix_abs_error_mw": float(prefix_error.mean().detach().cpu()),
        "endpoint_abs_error_mw": float(endpoint_error.mean().detach().cpu()),
        "violation_fraction": float((violation > 0.0).float().mean().detach().cpu()),
        "mean_violation_mw": float(violation.mean().detach().cpu()),
    }


def event_aggregated_magnitude_loss(
    predicted_mw: torch.Tensor,
    target_mw: torch.Tensor,
    events: Sequence[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted = predicted_mw.reshape(-1)
    target = target_mw.reshape(-1).to(
        device=predicted.device,
        dtype=predicted.dtype,
    )
    if predicted.shape != target.shape or len(events) != predicted.numel():
        raise ValueError("event magnitude inputs must have matching sample counts")
    grouped: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        grouped.setdefault(str(event), []).append(index)
    if not grouped:
        raise ValueError("event magnitude loss requires at least one event")
    event_errors: list[torch.Tensor] = []
    for indices in grouped.values():
        index = torch.as_tensor(indices, device=predicted.device, dtype=torch.long)
        event_errors.append(
            torch.abs(predicted[index].mean() - target[index].mean())
        )
    errors = torch.stack(event_errors)
    return errors.mean(), {
        "event_count": float(len(grouped)),
        "event_mae_mw": float(errors.mean().detach().cpu()),
    }


def direct_causal_objective(
    prefix_science_loss: torch.Tensor,
    endpoint_science_loss: torch.Tensor,
    descent_loss: torch.Tensor,
    *,
    prefix_weight: float,
    endpoint_weight: float,
    descent_weight: float,
) -> torch.Tensor:
    return (
        float(prefix_weight) * prefix_science_loss
        + float(endpoint_weight) * endpoint_science_loss
        + float(descent_weight) * descent_loss
    )


def moment_scale_from_delta_mw(delta_mw: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(delta_mw).all()):
        raise ValueError("delta_mw must be finite")
    return torch.pow(
        delta_mw.new_tensor(10.0),
        1.5 * delta_mw,
    )


def augment_prepared_with_moment_scaling(
    prepared: Any,
    sample_weights: torch.Tensor | None,
    *,
    delta_mw: torch.Tensor,
    augmentation_weight: float,
) -> tuple[Any, torch.Tensor]:
    weight = float(augmentation_weight)
    if weight <= 0.0 or not math.isfinite(weight):
        raise ValueError("augmentation_weight must be finite and positive")
    delta = delta_mw.reshape(-1).to(
        device=prepared.true_mag.device,
        dtype=prepared.true_mag.dtype,
    )
    if delta.shape != prepared.true_mag.reshape(-1).shape:
        raise ValueError("delta_mw must match the prepared batch size")
    scale = moment_scale_from_delta_mw(delta)

    def scaled(value: torch.Tensor) -> torch.Tensor:
        aligned = scale.reshape(
            (scale.shape[0],) + (1,) * (value.ndim - 1)
        ).to(device=value.device, dtype=value.dtype)
        return value * aligned

    def duplicated(value: torch.Tensor) -> torch.Tensor:
        return torch.cat((value, value), dim=0)

    augmented = replace(
        prepared,
        radial=torch.cat((prepared.radial, scaled(prepared.radial)), dim=0),
        model_input=torch.cat(
            (prepared.model_input, scaled(prepared.model_input)), dim=0
        ),
        source_distance_m=duplicated(prepared.source_distance_m),
        theta_deg=duplicated(prepared.theta_deg),
        phi_slip_deg=duplicated(prepared.phi_slip_deg),
        source_dt_sec=duplicated(prepared.source_dt_sec),
        observation_dt_sec=duplicated(prepared.observation_dt_sec),
        waveform_valid_mask=duplicated(prepared.waveform_valid_mask),
        stf_true=torch.cat((prepared.stf_true, scaled(prepared.stf_true)), dim=0),
        has_stf=duplicated(prepared.has_stf),
        true_mag=torch.cat((prepared.true_mag, prepared.true_mag + delta), dim=0),
        metadata=duplicated(prepared.metadata),
    )
    base_weights = (
        torch.ones_like(prepared.true_mag, dtype=prepared.radial.dtype)
        if sample_weights is None
        else sample_weights.reshape(-1).to(
            device=prepared.radial.device,
            dtype=prepared.radial.dtype,
        )
    )
    if base_weights.shape != prepared.true_mag.reshape(-1).shape:
        raise ValueError("sample_weights must match the prepared batch size")
    normalizer = 2.0 / (1.0 + weight)
    combined_weights = torch.cat(
        (normalizer * base_weights, normalizer * weight * base_weights),
        dim=0,
    )
    return augmented, combined_weights


def initialize_direct_model(
    model: torch.nn.Module,
    source_state: Mapping[str, torch.Tensor],
    *,
    initialization: str,
) -> None:
    mode = str(initialization).strip().lower()
    if mode == "scratch":
        return
    if mode == "phase39_checkpoint":
        model.load_state_dict(source_state, strict=True)
        return
    raise ValueError(f"unsupported direct initialization: {initialization!r}")


def configure_trainable_scope(
    model: torch.nn.Module,
    *,
    scope: str,
) -> list[torch.nn.Parameter]:
    normalized = str(scope).strip().lower()
    if normalized not in {"all", "moment_head"}:
        raise ValueError(f"unsupported trainable scope: {scope!r}")
    selected: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        trainable = normalized == "all" or name.startswith("log10_moment_head.")
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
    if not selected:
        raise ValueError(f"trainable scope selected no parameters: {scope!r}")
    return selected


def _science_loss(
    model: PINNModel,
    criterion: Any,
    prepared: Any,
    *,
    horizon_sec: int,
    sample_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    encoded = model(
        prepared.model_input[:, :, :horizon_sec],
        meta=prepared.metadata,
    )
    loss, parts = criterion(
        encoded,
        pred_catalog_mw=None,
        radial_obs=prepared.radial[:, :, :horizon_sec],
        source_distance_m=prepared.source_distance_m,
        theta_deg=prepared.theta_deg,
        phi_slip_deg=prepared.phi_slip_deg,
        source_dt_sec=prepared.source_dt_sec,
        observation_dt_sec=prepared.observation_dt_sec,
        waveform_valid_mask=prepared.waveform_valid_mask[:, :horizon_sec],
        stf_true=prepared.stf_true,
        has_stf=prepared.has_stf,
        true_mag=prepared.true_mag,
        sample_weights=sample_weights,
    )
    rate = criterion._decode_rate(encoded)
    mw = moment_magnitude_from_rate(rate, prepared.source_dt_sec)
    return loss, parts, mw


def _station_rows(
    batch: Mapping[str, Any],
    predictions: torch.Tensor,
    *,
    method: str,
    horizon_sec: int,
) -> list[dict[str, Any]]:
    predicted = predictions.reshape(-1).detach().cpu().numpy()
    catalogs = batch["magnitude_catalog"].reshape(-1).detach().cpu().numpy()
    native = batch["mw_stf_native"].reshape(-1).detach().cpu().numpy()
    return [
        {
            "method": method,
            "observation_horizon_sec": int(horizon_sec),
            "event": str(batch["event"][index]),
            "station": str(batch["station"][index]),
            "mw_pred": float(predicted[index]),
            "mw_catalog": float(catalogs[index]),
            "mw_stf_native": float(native[index]),
        }
        for index in range(len(predicted))
    ]


def evaluate_horizons(
    model: PINNModel,
    criterion: Any,
    config: dict[str, Any],
    loader: Any,
    *,
    method: str,
    horizons: Sequence[int],
    station_horizons: set[int] | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    rows_by_horizon = {int(horizon): [] for horizon in horizons}
    original_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                prepared = _prepare_v2_batch(batch, config, device)
                for horizon in horizons:
                    encoded = model(
                        prepared.model_input[:, :, : int(horizon)],
                        meta=prepared.metadata,
                    )
                    rate = criterion._decode_rate(encoded)
                    mw = moment_magnitude_from_rate(rate, prepared.source_dt_sec)
                    rows_by_horizon[int(horizon)].extend(
                        _station_rows(
                            batch,
                            mw,
                            method=method,
                            horizon_sec=int(horizon),
                        )
                    )
    finally:
        model.train(original_training)

    horizon_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    retained_station_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        station_rows = rows_by_horizon[int(horizon)]
        events = aggregate_event_predictions(
            station_rows,
            reference_key="mw_catalog",
        )
        event_rows.extend(
            {
                "method": method,
                "observation_horizon_sec": int(horizon),
                **row,
            }
            for row in events
        )
        horizon_rows.append(
            {
                "method": method,
                "observation_horizon_sec": int(horizon),
                **summarize_predictions(
                    station_rows,
                    events,
                    reference_key="mw_catalog",
                ),
            }
        )
        if station_horizons is not None and int(horizon) in station_horizons:
            retained_station_rows.extend(station_rows)
    return {
        "horizon_rows": horizon_rows,
        "event_rows": event_rows,
        "station_rows": retained_station_rows,
    }


def _metric_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(row["observation_horizon_sec"]): row for row in rows}


def summarize_trajectory(
    horizon_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    *,
    endpoint_band_tolerance_mw: float,
) -> dict[str, Any]:
    metrics = sorted(
        horizon_rows,
        key=lambda row: int(row["observation_horizon_sec"]),
    )
    horizons = np.asarray(
        [int(row["observation_horizon_sec"]) for row in metrics],
        dtype=np.float64,
    )
    maes = np.asarray([float(row["event_mae"]) for row in metrics])
    endpoint = float(maes[-1])
    upper = endpoint + float(endpoint_band_tolerance_mw)
    stable: int | None = None
    for index, horizon in enumerate(horizons.astype(int)):
        if bool(np.all(maes[index:] <= upper)):
            stable = int(horizon)
            break
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in event_rows:
        grouped.setdefault(str(row["event"]), []).append(
            (int(row["observation_horizon_sec"]), float(row["mw_pred_median"]))
        )
    late_steps: list[float] = []
    for values in grouped.values():
        predictions = [value for horizon, value in sorted(values) if horizon >= 120]
        late_steps.extend(abs(float(value)) for value in np.diff(predictions))
    return {
        "endpoint_event_mae_mw": endpoint,
        "minimum_event_mae_mw": float(np.min(maes)),
        "minimum_event_mae_horizon_sec": int(horizons[int(np.argmin(maes))]),
        "mae_linear_slope_mw_per_sec": float(np.polyfit(horizons, maes, 1)[0]),
        "mae_decreasing_step_fraction": float(np.mean(np.diff(maes) <= 0.0)),
        "stable_horizon_within_endpoint_band_sec": stable,
        "endpoint_band_upper_mw": upper,
        "post120_event_abs_step_p95_mw": (
            float(np.percentile(late_steps, 95)) if late_steps else 0.0
        ),
    }


def _load_source(
    experiment: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    summary_path = Path(str(experiment["source_run_summary"]))
    summary = _read_json(summary_path)
    fold = int(experiment["fold"])
    seed = int(experiment["seed"])
    required_summary = {
        "status": "complete",
        "arm": "phase39",
        "fold": fold,
        "seed": seed,
    }
    for key, expected in required_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"source run {key} changed: {summary.get(key)!r}")
    checkpoint_path = Path(str(summary["best_model_path"]))
    config_path = Path(str(summary["config_snapshot_path"]))
    if sha256_file(checkpoint_path) != str(summary["checkpoint_sha256"]):
        raise ValueError("source Phase39 checkpoint hash changed")
    if sha256_file(config_path) != str(summary["config_snapshot_sha256"]):
        raise ValueError("source Phase39 config hash changed")
    config = _read_yaml(config_path)
    validate_config_on_startup(config)
    loss = config["training"]["stf_rate_loss"]
    required_loss = {
        "lambda_synth": 0.5,
        "synth_polarity_mode": "global_invariant",
        "radiation_coefficient_contract": "glehman_scalar",
    }
    for key, expected in required_loss.items():
        if loss.get(key) != expected:
            raise ValueError(f"source Phase39 loss contract changed: {key}")

    samples = [
        dict(sample)
        for sample in CorrectedEarthquakeDataset(copy.deepcopy(config)).samples
    ]
    fold_manifest = build_event_folds(samples)
    split = make_outer_split(samples, fold_manifest, outer_fold=fold)
    assignment = split_assignment_sha256(samples, split)
    if assignment != str(summary["split_assignment_sha256"]):
        raise ValueError("source grouped split changed")
    train_loader, validation_loader, test_loader, split_manifest = (
        get_data_loaders_v2(config, explicit_split=split)
    )
    if split_manifest["assignment_sha256"] != assignment:
        raise ValueError("loader split assignment changed")
    del test_loader
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return {
        "summary_path": summary_path,
        "summary": summary,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "config": config,
        "state": state,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "split_manifest": split_manifest,
    }


def run_experiment(
    *,
    experiment_config_path: Path,
    output_root: Path,
    smoke: bool,
    device: torch.device,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    experiment = _read_yaml(experiment_config_path)
    training = experiment["training"]
    evaluation = experiment["evaluation"]
    seed = int(experiment["seed"])
    configure_runtime(seed, device)
    source = _load_source(experiment, device=device)
    config = source["config"]
    synth_weight = float(training["lambda_synth"])
    if synth_weight <= 0.0:
        raise ValueError("the primary direct experiment requires lambda_synth > 0")
    event_magnitude_weight = float(training.get("event_magnitude_weight", 0.0))
    if event_magnitude_weight < 0.0 or not math.isfinite(event_magnitude_weight):
        raise ValueError("event_magnitude_weight must be finite and nonnegative")
    moment_scale_weight = float(
        training.get("moment_scale_augmentation_weight", 0.0)
    )
    if moment_scale_weight < 0.0 or not math.isfinite(moment_scale_weight):
        raise ValueError(
            "moment_scale_augmentation_weight must be finite and nonnegative"
        )
    moment_scale_min_mw = float(
        training.get("moment_scale_delta_min_mw", -0.75)
    )
    moment_scale_max_mw = float(
        training.get("moment_scale_delta_max_mw", 0.5)
    )
    if moment_scale_weight > 0.0:
        if (
            not math.isfinite(moment_scale_min_mw)
            or not math.isfinite(moment_scale_max_mw)
            or moment_scale_min_mw >= moment_scale_max_mw
        ):
            raise ValueError("invalid moment-scale augmentation range")
        if event_magnitude_weight > 0.0:
            raise ValueError(
                "moment-scale augmentation is incompatible with the rejected "
                "batch event-magnitude auxiliary loss"
            )
    prefix_pair_gap_sec = int(training.get("prefix_pair_gap_sec", 0))
    if prefix_pair_gap_sec < 0:
        raise ValueError("prefix_pair_gap_sec must be nonnegative")
    curriculum_start_epoch = int(
        training.get("prefix_curriculum_start_epoch", 0)
    )
    curriculum_full_epoch = int(
        training.get("prefix_curriculum_full_epoch", 1)
    )
    causal_curriculum_scale(
        1,
        start_epoch=curriculum_start_epoch,
        full_epoch=curriculum_full_epoch,
    )
    ema_start_epoch = int(training.get("ema_start_epoch", 0))
    ema_decay = float(training.get("ema_decay", 0.0))
    if ema_start_epoch < 0:
        raise ValueError("ema_start_epoch must be nonnegative")
    if not 0.0 <= ema_decay < 1.0 or not math.isfinite(ema_decay):
        raise ValueError("ema_decay must be finite in [0, 1)")
    ema_enabled = ema_start_epoch > 0 and ema_decay > 0.0
    effective_ema_start_epoch = 1 if smoke and ema_enabled else ema_start_epoch
    config["training"]["stf_rate_loss"]["lambda_synth"] = synth_weight
    criterion = _build_stf_rate_criterion(config, device)

    model = PINNModel(config).to(device)
    initialization = str(training.get("initialization", "phase39_checkpoint"))
    initialize_direct_model(
        model,
        source["state"],
        initialization=initialization,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PHASE39_PARAMETER_COUNT:
        raise ValueError(
            "direct model parameter count differs from Phase39: "
            f"{parameter_count} != {EXPECTED_PHASE39_PARAMETER_COUNT}"
        )
    trainable_scope = str(training.get("trainable_scope", "all"))
    trainable_parameters = configure_trainable_scope(
        model,
        scope=trainable_scope,
    )
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    ema_model = (
        torch.optim.swa_utils.AveragedModel(
            model,
            avg_fn=lambda averaged, current, _: (
                ema_decay * averaged + (1.0 - ema_decay) * current
            ),
            use_buffers=True,
        )
        if ema_enabled
        else None
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(training.get("scheduler_T0", 15)),
        T_mult=int(training.get("scheduler_T_mult", 2)),
        eta_min=float(training.get("scheduler_eta_min", 1.0e-6)),
    )
    warmup_epochs = int(training.get("warmup_epochs", 5))
    event_weights = _training_event_balance_weights(
        config,
        source["train_loader"],
    )
    anchor_horizons = tuple(int(value) for value in evaluation["anchor_horizons_sec"])
    max_eval_batches = 2 if smoke else None
    baseline_model = PINNModel(config).to(device)
    baseline_model.load_state_dict(source["state"], strict=True)
    baseline = evaluate_horizons(
        baseline_model,
        criterion,
        config,
        source["validation_loader"],
        method="phase39",
        horizons=anchor_horizons,
        max_batches=max_eval_batches,
    )
    baseline_lookup = _metric_lookup(baseline["horizon_rows"])
    baseline_endpoint = float(baseline_lookup[SOURCE_STEPS]["event_mae"])

    protocol = {
        "method": "direct causal-prefix Phase39",
        "architecture_changed": False,
        "adapter_used": False,
        "teacher_used": False,
        "recurrent_state_used": False,
        "handcrafted_features_used": False,
        "total_parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "initialization": initialization,
        "trainable_scope": trainable_scope,
        "source_checkpoint": str(source["checkpoint_path"]),
        "source_checkpoint_sha256": sha256_file(source["checkpoint_path"]),
        "source_config": str(source["config_path"]),
        "source_config_sha256": sha256_file(source["config_path"]),
        "split_assignment_sha256": source["split_manifest"]["assignment_sha256"],
        "fold": int(experiment["fold"]),
        "seed": seed,
        "lambda_synth": synth_weight,
        "event_magnitude_weight": event_magnitude_weight,
        "moment_scale_augmentation_weight": moment_scale_weight,
        "moment_scale_delta_min_mw": moment_scale_min_mw,
        "moment_scale_delta_max_mw": moment_scale_max_mw,
        "prefix_pair_gap_sec": prefix_pair_gap_sec,
        "prefix_curriculum_start_epoch": curriculum_start_epoch,
        "prefix_curriculum_full_epoch": curriculum_full_epoch,
        "ema_enabled": ema_enabled,
        "ema_start_epoch": ema_start_epoch,
        "effective_ema_start_epoch": effective_ema_start_epoch,
        "ema_decay": ema_decay,
        "synth_polarity_mode": criterion.synth_polarity_mode,
        "radiation_coefficient_contract": criterion.radiation_coefficient_contract,
        "test_split_iterated": False,
        "external_events_loaded": False,
        "git_commit": current_git_commit(PROJECT_ROOT),
        "git_dirty": git_is_dirty(PROJECT_ROOT),
        "device": str(device),
        "smoke": bool(smoke),
        "experiment_config": experiment,
    }
    _write_json(output_root / "protocol.json", protocol)
    _write_json(output_root / "split.json", source["split_manifest"])
    _write_json(output_root / "baseline_anchor_metrics.json", baseline)
    (output_root / "experiment_config.yaml").write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    best_path = output_root / "best_model.pth"
    if initialization.strip().lower() == "phase39_checkpoint":
        atomic_torch_save(dict(source["state"]), best_path)
        best_epoch = 0
        best_endpoint = baseline_endpoint
    else:
        best_epoch = -1
        best_endpoint = math.inf
    patience = 0
    global_step = 0
    epoch_rows: list[dict[str, Any]] = []
    epochs = 1 if smoke else int(training["epochs"])
    max_train_batches = 2 if smoke else None

    for epoch in range(1, epochs + 1):
        curriculum_scale = causal_curriculum_scale(
            epoch,
            start_epoch=curriculum_start_epoch,
            full_epoch=curriculum_full_epoch,
        )
        effective_prefix_weight = (
            float(training["prefix_science_weight"]) * curriculum_scale
        )
        effective_descent_weight = (
            float(training["error_descent_weight"]) * curriculum_scale
        )
        model.train()
        totals = {
            "total": 0.0,
            "prefix": 0.0,
            "endpoint": 0.0,
            "synth": 0.0,
            "descent": 0.0,
            "event_magnitude": 0.0,
            "violation": 0.0,
        }
        seen = 0
        for batch_index, batch in enumerate(source["train_loader"]):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            prepared = _prepare_v2_batch(batch, config, device)
            weights = _batch_event_sample_weights(
                batch,
                event_weights,
                reference=prepared.radial,
            )
            if moment_scale_weight > 0.0:
                delta_mw = torch.empty_like(prepared.true_mag).uniform_(
                    moment_scale_min_mw,
                    moment_scale_max_mw,
                )
                prepared, weights = augment_prepared_with_moment_scaling(
                    prepared,
                    weights,
                    delta_mw=delta_mw,
                    augmentation_weight=moment_scale_weight,
                )
            optimizer.zero_grad(set_to_none=True)
            if curriculum_scale <= 0.0:
                early_mw = None
                late_mw = None
                prefix_loss = prepared.radial.new_zeros(())
                prefix_synth = 0.0
            elif prefix_pair_gap_sec > 0:
                early_horizon, late_horizon = ordered_prefix_horizons_for_step(
                    global_step,
                    seed=seed,
                    minimum=int(training["minimum_prefix_horizon_sec"]),
                    maximum=int(training["maximum_prefix_horizon_sec"]),
                    multiplier=int(training["horizon_cycle_multiplier"]),
                    gap_sec=prefix_pair_gap_sec,
                )
                early_loss, early_parts, early_mw = _science_loss(
                    model,
                    criterion,
                    prepared,
                    horizon_sec=early_horizon,
                    sample_weights=weights,
                )
                late_loss, late_parts, late_mw = _science_loss(
                    model,
                    criterion,
                    prepared,
                    horizon_sec=late_horizon,
                    sample_weights=weights,
                )
                prefix_loss = 0.5 * (early_loss + late_loss)
                prefix_synth = 0.5 * (
                    float(early_parts["L_synth"])
                    + float(late_parts["L_synth"])
                )
            else:
                early_horizon = prefix_horizon_for_step(
                    global_step,
                    seed=seed,
                    minimum=int(training["minimum_prefix_horizon_sec"]),
                    maximum=int(training["maximum_prefix_horizon_sec"]),
                    multiplier=int(training["horizon_cycle_multiplier"]),
                )
                late_horizon = early_horizon
                early_loss, early_parts, early_mw = _science_loss(
                    model,
                    criterion,
                    prepared,
                    horizon_sec=early_horizon,
                    sample_weights=weights,
                )
                late_mw = early_mw
                prefix_loss = early_loss
                prefix_synth = float(early_parts["L_synth"])
            endpoint_loss, endpoint_parts, endpoint_mw = _science_loss(
                model,
                criterion,
                prepared,
                horizon_sec=SOURCE_STEPS,
                sample_weights=weights,
            )
            if curriculum_scale <= 0.0:
                descent_loss = endpoint_mw.new_zeros(())
                descent_violation_fraction = 0.0
            elif prefix_pair_gap_sec > 0:
                assert early_mw is not None and late_mw is not None
                late_descent, late_descent_metrics = error_descent_loss(
                    early_mw,
                    late_mw,
                    prepared.true_mag,
                    slack_mw=float(training["error_descent_slack_mw"]),
                    sample_weights=weights,
                )
                endpoint_descent, endpoint_descent_metrics = error_descent_loss(
                    late_mw,
                    endpoint_mw,
                    prepared.true_mag,
                    slack_mw=float(training["error_descent_slack_mw"]),
                    sample_weights=weights,
                )
                descent_loss = 0.5 * (late_descent + endpoint_descent)
                descent_violation_fraction = 0.5 * (
                    late_descent_metrics["violation_fraction"]
                    + endpoint_descent_metrics["violation_fraction"]
                )
            else:
                assert early_mw is not None
                descent_loss, descent_metrics = error_descent_loss(
                    early_mw,
                    endpoint_mw,
                    prepared.true_mag,
                    slack_mw=float(training["error_descent_slack_mw"]),
                    sample_weights=weights,
                )
                descent_violation_fraction = descent_metrics["violation_fraction"]
            if event_magnitude_weight > 0.0:
                if early_mw is None or late_mw is None:
                    raise ValueError(
                        "event magnitude loss requires active causal prefixes"
                    )
                early_event_loss, _ = event_aggregated_magnitude_loss(
                    early_mw,
                    prepared.true_mag,
                    batch["event"],
                )
                late_event_loss, _ = event_aggregated_magnitude_loss(
                    late_mw,
                    prepared.true_mag,
                    batch["event"],
                )
                prefix_event_loss = 0.5 * (
                    early_event_loss + late_event_loss
                )
                endpoint_event_loss, _ = event_aggregated_magnitude_loss(
                    endpoint_mw,
                    prepared.true_mag,
                    batch["event"],
                )
            else:
                prefix_event_loss = endpoint_mw.new_zeros(())
                endpoint_event_loss = endpoint_mw.new_zeros(())
            total_loss = direct_causal_objective(
                prefix_loss,
                endpoint_loss,
                descent_loss,
                prefix_weight=effective_prefix_weight,
                endpoint_weight=float(training["endpoint_science_weight"]),
                descent_weight=effective_descent_weight,
            )
            total_loss = total_loss + event_magnitude_weight * (
                effective_prefix_weight * prefix_event_loss
                + float(training["endpoint_science_weight"])
                * endpoint_event_loss
            )
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("direct causal loss became non-finite")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(training["grad_clip_norm"]),
            )
            optimizer.step()
            if (
                ema_model is not None
                and epoch >= effective_ema_start_epoch
            ):
                ema_model.update_parameters(model)

            batch_size = int(prepared.radial.shape[0])
            seen += batch_size
            totals["total"] += float(total_loss.detach().cpu()) * batch_size
            totals["prefix"] += float(prefix_loss.detach().cpu()) * batch_size
            totals["endpoint"] += float(endpoint_loss.detach().cpu()) * batch_size
            mean_synth = float(endpoint_parts["L_synth"])
            if curriculum_scale > 0.0:
                mean_synth = 0.5 * (prefix_synth + mean_synth)
            totals["synth"] += mean_synth * batch_size
            totals["descent"] += float(descent_loss.detach().cpu()) * batch_size
            totals["event_magnitude"] += 0.5 * (
                float(prefix_event_loss.detach().cpu())
                + float(endpoint_event_loss.detach().cpu())
            ) * batch_size
            totals["violation"] += descent_violation_fraction * batch_size
            global_step += 1
        if seen == 0:
            raise ValueError("training loader produced no samples")

        validation_model = (
            ema_model
            if ema_model is not None and int(ema_model.n_averaged.item()) > 0
            else model
        )
        validation_weight_source = (
            "ema" if validation_model is ema_model else "raw"
        )
        validation = evaluate_horizons(
            validation_model,
            criterion,
            config,
            source["validation_loader"],
            method="direct",
            horizons=anchor_horizons,
            max_batches=max_eval_batches,
        )
        lookup = _metric_lookup(validation["horizon_rows"])
        endpoint_mae = float(lookup[SOURCE_STEPS]["event_mae"])
        improved = endpoint_mae < best_endpoint - 1.0e-6
        if improved:
            best_endpoint = endpoint_mae
            best_epoch = epoch
            patience = 0
            selected_state = (
                validation_model.module.state_dict()
                if isinstance(
                    validation_model,
                    torch.optim.swa_utils.AveragedModel,
                )
                else validation_model.state_dict()
            )
            atomic_torch_save(dict(selected_state), best_path)
        else:
            patience += 1
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_total_loss": totals["total"] / seen,
            "train_prefix_science_loss": totals["prefix"] / seen,
            "train_endpoint_science_loss": totals["endpoint"] / seen,
            "train_mean_synth_loss": totals["synth"] / seen,
            "train_error_descent_loss": totals["descent"] / seen,
            "train_mean_event_magnitude_loss": (
                totals["event_magnitude"] / seen
            ),
            "causal_curriculum_scale": curriculum_scale,
            "effective_prefix_science_weight": effective_prefix_weight,
            "effective_error_descent_weight": effective_descent_weight,
            "validation_weight_source": validation_weight_source,
            "ema_updates": (
                int(ema_model.n_averaged.item())
                if ema_model is not None
                else 0
            ),
            "train_descent_violation_fraction": totals["violation"] / seen,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_endpoint_event_mae": endpoint_mae,
            "best_validation_endpoint_event_mae": best_endpoint,
            "best_epoch": best_epoch,
            "improved": improved,
            **{
                f"validation_event_mae_{horizon:03d}s": float(
                    lookup[horizon]["event_mae"]
                )
                for horizon in anchor_horizons
            },
        }
        epoch_rows.append(row)
        _write_csv(output_root / "training_epoch_metrics.csv", epoch_rows)
        print(
            f"epoch={epoch:03d} loss={row['train_total_loss']:.6f} "
            f"endpoint={endpoint_mae:.6f} best={best_endpoint:.6f} "
            f"mae60={row['validation_event_mae_060s']:.4f} "
            f"mae120={row['validation_event_mae_120s']:.4f}"
        )
        if epoch >= warmup_epochs:
            scheduler.step()
        if not smoke and patience >= int(training["early_stop_patience"]):
            break

    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=True),
        strict=True,
    )
    full_horizons = tuple(
        range(
            int(evaluation["full_horizon_start_sec"]),
            int(evaluation["full_horizon_end_sec"]) + 1,
        )
    )
    if smoke:
        full_horizons = anchor_horizons
    final_direct = evaluate_horizons(
        model,
        criterion,
        config,
        source["validation_loader"],
        method="direct",
        horizons=full_horizons,
        station_horizons=set(anchor_horizons),
        max_batches=max_eval_batches,
    )
    baseline_model = PINNModel(config).to(device)
    baseline_model.load_state_dict(source["state"], strict=True)
    final_baseline = evaluate_horizons(
        baseline_model,
        criterion,
        config,
        source["validation_loader"],
        method="phase39",
        horizons=full_horizons,
        station_horizons=set(anchor_horizons),
        max_batches=max_eval_batches,
    )
    horizon_rows = final_baseline["horizon_rows"] + final_direct["horizon_rows"]
    event_rows = final_baseline["event_rows"] + final_direct["event_rows"]
    station_rows = final_baseline["station_rows"] + final_direct["station_rows"]
    _write_csv(output_root / "validation_horizon_metrics.csv", horizon_rows)
    _write_csv(output_root / "validation_event_predictions.csv", event_rows)
    _write_csv(output_root / "validation_anchor_station_predictions.csv", station_rows)

    direct_summary = summarize_trajectory(
        final_direct["horizon_rows"],
        final_direct["event_rows"],
        endpoint_band_tolerance_mw=float(evaluation["endpoint_band_tolerance_mw"]),
    )
    baseline_summary = summarize_trajectory(
        final_baseline["horizon_rows"],
        final_baseline["event_rows"],
        endpoint_band_tolerance_mw=float(evaluation["endpoint_band_tolerance_mw"]),
    )
    target = float(evaluation["endpoint_target_mw"])
    stretch = float(evaluation["endpoint_stretch_target_mw"])
    summary = {
        "status": "complete",
        "best_epoch": best_epoch,
        "baseline_endpoint_event_mae_mw": baseline_summary["endpoint_event_mae_mw"],
        "direct_endpoint_event_mae_mw": direct_summary["endpoint_event_mae_mw"],
        "endpoint_improvement_mw": (
            baseline_summary["endpoint_event_mae_mw"]
            - direct_summary["endpoint_event_mae_mw"]
        ),
        "endpoint_target_mw": target,
        "endpoint_target_passed": direct_summary["endpoint_event_mae_mw"] < target,
        "endpoint_stretch_target_mw": stretch,
        "endpoint_stretch_target_passed": (
            direct_summary["endpoint_event_mae_mw"] < stretch
        ),
        "lambda_synth": synth_weight,
        "event_magnitude_weight": event_magnitude_weight,
        "moment_scale_augmentation_weight": moment_scale_weight,
        "moment_scale_delta_min_mw": moment_scale_min_mw,
        "moment_scale_delta_max_mw": moment_scale_max_mw,
        "prefix_pair_gap_sec": prefix_pair_gap_sec,
        "prefix_curriculum_start_epoch": curriculum_start_epoch,
        "prefix_curriculum_full_epoch": curriculum_full_epoch,
        "ema_enabled": ema_enabled,
        "ema_start_epoch": ema_start_epoch,
        "ema_decay": ema_decay,
        "initialization": initialization,
        "trainable_scope": trainable_scope,
        "trainable_parameter_count": trainable_parameter_count,
        "architecture_changed": False,
        "test_split_iterated": False,
        "external_events_loaded": False,
        "trajectory": {
            "phase39": baseline_summary,
            "direct": direct_summary,
        },
        "checkpoint_path": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the unchanged Phase39 model on true causal prefixes using only "
            "the original synth science loss and one error-descent term."
        )
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root or (
        Path("/home/lihe/PINN_Mag/runs") / f"phase39-causal-direct-{timestamp}"
    )
    summary = run_experiment(
        experiment_config_path=args.experiment_config.resolve(),
        output_root=output_root.resolve(),
        smoke=bool(args.smoke),
        device=device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
