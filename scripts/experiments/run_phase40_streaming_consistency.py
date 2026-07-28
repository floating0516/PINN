from __future__ import annotations

import argparse
import copy
import csv
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

from src.data.loaders_v2 import get_data_loaders_v2  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    aggregate_event_predictions,
    summarize_predictions,
)
from src.models.model import PINNModel  # noqa: E402
from src.training.checkpointing import atomic_torch_save  # noqa: E402
from src.training.loss_stf_rate_v2 import (  # noqa: E402
    moment_magnitude_from_rate,
)
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


PHASE39_MODEL_DIR = Path(
    "/home/lihe/PINN_Mag/runs/"
    "phase39-manuscript-stf-glehman-scalar-20260726T111256Z-7fbb7a7/"
    "train/candidate/seed_42/models/20260726_192953"
)
PHASE39_CHECKPOINT = PHASE39_MODEL_DIR / "best_model.pth"
PHASE39_CONFIG = PHASE39_MODEL_DIR / "config.yaml"
PHASE39_SPLIT = PHASE39_MODEL_DIR / "split.json"

EXPECTED_SHA256 = {
    "checkpoint": "73500f365a58b248204d02333716f31674435927e9fc1c7d55a1453786b406f7",
    "config": "a05181166c7f40cae755ffbfbd0f4adfdb6a83703a89299677fa2be5f8ff1966",
    "split": "0ca718d5966004a768f0c648dc6dc2211b55a031819db6abeac8a111e75788c4",
}
EXPECTED_SPLIT_ASSIGNMENT_SHA256 = (
    "5ac2e07ed186dce737a3592694632775b7bbf603bf922a4a74fa6b86a3d5c240"
)

HEAD_TRAINABLE_PREFIXES = ("shape_head.", "log10_moment_head.")
LAST_TRANSFORMER_TRAINABLE_PREFIXES = (
    "transformer.layers.2.",
    "post_transformer_norm.",
    *HEAD_TRAINABLE_PREFIXES,
)
TRAINABLE_SCOPE_PREFIXES = {
    "heads": HEAD_TRAINABLE_PREFIXES,
    "last_transformer": LAST_TRANSFORMER_TRAINABLE_PREFIXES,
}
EXPECTED_SCOPE_PARAMETER_COUNTS = {
    "heads": 8_322,
    "last_transformer": 141_058,
}

# Backward-compatible names for the frozen Phase40 head-only protocol.
TRAINABLE_PREFIXES = HEAD_TRAINABLE_PREFIXES
EXPECTED_TRAINABLE_PARAMETER_COUNT = EXPECTED_SCOPE_PARAMETER_COUNTS["heads"]
EPOCHS = 20
LEARNING_RATE = 1.0e-5
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP_NORM = 1.0
LAMBDA_STREAM_CONSISTENCY = 0.1
CONSISTENCY_HUBER_BETA = 0.05
MIN_PREFIX_HORIZON = 20
MAX_PREFIX_HORIZON = 199
HORIZON_CYCLE_MULTIPLIER = 73
HORIZON_CYCLE_OFFSET = 42
VALIDATION_HORIZONS = tuple(range(179, 201))

BASELINE_METRICS = {
    "endpoint_event_mae": 0.11433351834615071,
    "endpoint_station_mae": 0.13169988409265296,
    "late_event_abs_step_p95_mw": 0.02548871040344237,
    "late_station_abs_step_p95_mw": 0.05132026672363278,
    "late_confirmed_cumulative_log10_l1_p95": 0.06841289103031142,
}
GATES = {
    "endpoint_event_mae_max": BASELINE_METRICS["endpoint_event_mae"] + 0.005,
    "endpoint_station_mae_max": BASELINE_METRICS["endpoint_station_mae"] + 0.005,
    "late_event_abs_step_p95_mw_max": (
        0.8 * BASELINE_METRICS["late_event_abs_step_p95_mw"]
    ),
    "late_station_abs_step_p95_mw_max": BASELINE_METRICS[
        "late_station_abs_step_p95_mw"
    ],
    "late_confirmed_cumulative_log10_l1_p95_max": (
        0.8
        * BASELINE_METRICS[
            "late_confirmed_cumulative_log10_l1_p95"
        ]
    ),
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for source in rows:
            writer.writerow(
                {
                    key: "" if value is None else value
                    for key, value in source.items()
                }
            )


def validate_source_artifacts() -> dict[str, str]:
    paths = {
        "checkpoint": PHASE39_CHECKPOINT,
        "config": PHASE39_CONFIG,
        "split": PHASE39_SPLIT,
    }
    actual: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing Phase39 {name}: {path}")
        actual[name] = sha256_file(path)
        if actual[name] != EXPECTED_SHA256[name]:
            raise ValueError(
                f"Phase39 {name} SHA-256 changed: {actual[name]} != "
                f"{EXPECTED_SHA256[name]}"
            )
    return actual


def load_frozen_config() -> dict[str, Any]:
    with PHASE39_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config_on_startup(config)
    training = config["training"]
    loss = training["stf_rate_loss"]
    expected = {
        "seed": (int(training["random_seed"]), 42),
        "sample_rate": (float(config["dataset"]["sample_rate_hz"]), 1.0),
        "duration": (
            float(config["dataset"]["waveform"]["duration_sec"]),
            200.0,
        ),
        "lambda_MSE": (float(loss["lambda_MSE"]), 1.0),
        "lambda_synth": (float(loss["lambda_synth"]), 0.5),
        "lambda_mag": (float(loss["lambda_mag"]), 1.0),
        "lambda_shape": (float(loss["lambda_shape"]), 0.0),
    }
    changed = [name for name, values in expected.items() if values[0] != values[1]]
    if changed:
        raise ValueError("frozen Phase39 config changed: " + ", ".join(changed))
    if tuple(config["model"]["input_components"]) != ("radial",):
        raise ValueError("Phase40 requires the frozen R-only model")
    if config["model"]["stf_output_parameterization"] != "moment_shape_factorized":
        raise ValueError("Phase40 requires the factorized STF head")
    if loss["synth_polarity_mode"] != "global_invariant":
        raise ValueError("Phase40 requires global-invariant synthesis")
    if loss["radiation_coefficient_contract"] != "glehman_scalar":
        raise ValueError("Phase40 requires Glehman scalar radiation")
    return config


def freeze_trainable_scope(
    model: PINNModel,
    trainable_scope: str,
) -> tuple[list[torch.nn.Parameter], int]:
    if trainable_scope not in TRAINABLE_SCOPE_PREFIXES:
        raise ValueError(f"unknown trainable scope: {trainable_scope}")
    prefixes = TRAINABLE_SCOPE_PREFIXES[trainable_scope]
    trainable: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(prefixes)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(parameter)
    count = sum(parameter.numel() for parameter in trainable)
    expected_count = EXPECTED_SCOPE_PARAMETER_COUNTS[trainable_scope]
    if count != expected_count:
        raise ValueError(
            f"{trainable_scope} trainable parameter count changed: "
            f"{count} != {expected_count}"
        )
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith(prefixes)
    ):
        raise RuntimeError("a parameter outside the declared scope remains trainable")
    return trainable, count


def freeze_factorized_heads(model: PINNModel) -> tuple[list[torch.nn.Parameter], int]:
    return freeze_trainable_scope(model, "heads")


def horizon_for_step(step: int) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    span = MAX_PREFIX_HORIZON - MIN_PREFIX_HORIZON + 1
    return MIN_PREFIX_HORIZON + (
        (HORIZON_CYCLE_OFFSET + HORIZON_CYCLE_MULTIPLIER * step) % span
    )


def conservative_s_supported_steps(
    source_distance_m: torch.Tensor,
    *,
    previous_horizon_sec: int,
    beta_m_per_s: float,
    source_steps: int = 200,
) -> torch.Tensor:
    if previous_horizon_sec < 1:
        raise ValueError("previous_horizon_sec must be positive")
    if not math.isfinite(beta_m_per_s) or beta_m_per_s <= 0.0:
        raise ValueError("beta_m_per_s must be positive and finite")
    delay = source_distance_m.reshape(-1) / float(beta_m_per_s)
    visible = torch.clamp(float(previous_horizon_sec) - delay, min=0.0)
    return torch.floor(visible + 1.0e-12).to(torch.long).clamp(
        min=0,
        max=int(source_steps),
    )


def cumulative_log_consistency_from_rate(
    previous_rate_nm_per_s: torch.Tensor,
    current_rate_nm_per_s: torch.Tensor,
    *,
    source_distance_m: torch.Tensor,
    source_dt_sec: torch.Tensor,
    previous_horizon_sec: int,
    beta_m_per_s: float,
    huber_beta: float = CONSISTENCY_HUBER_BETA,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    previous = previous_rate_nm_per_s
    current = current_rate_nm_per_s
    if previous.ndim != 2 or current.shape != previous.shape:
        raise ValueError("paired STF rates must have shape (batch, source_time)")
    if not math.isfinite(huber_beta) or huber_beta <= 0.0:
        raise ValueError("huber_beta must be positive and finite")
    batch_size, source_steps = previous.shape
    distance = source_distance_m.reshape(-1)
    dt = source_dt_sec.reshape(-1)
    if distance.shape != (batch_size,) or dt.shape != (batch_size,):
        raise ValueError("distance and source_dt_sec must contain one value per sample")
    if bool(torch.any(dt <= 0.0)):
        raise ValueError("source_dt_sec must be positive")
    if sample_weights is not None:
        weights = sample_weights.reshape(-1).to(
            device=previous.device,
            dtype=previous.dtype,
        )
        if weights.shape != (batch_size,) or bool(torch.any(weights <= 0.0)):
            raise ValueError("sample_weights must be positive with shape (batch,)")
    else:
        weights = None

    steps = conservative_s_supported_steps(
        distance,
        previous_horizon_sec=previous_horizon_sec,
        beta_m_per_s=beta_m_per_s,
        source_steps=source_steps,
    )
    source_index = torch.arange(source_steps, device=previous.device).unsqueeze(0)
    mask = source_index < steps.unsqueeze(1)
    comparable = steps > 0
    dt_column = dt.unsqueeze(1)
    previous_cumulative = torch.cumsum(
        torch.clamp(previous, min=0.0) * dt_column,
        dim=1,
    ).clamp_min(1.0e10)
    current_cumulative = torch.cumsum(
        torch.clamp(current, min=0.0) * dt_column,
        dim=1,
    ).clamp_min(1.0e10)
    delta = torch.log10(current_cumulative) - torch.log10(previous_cumulative)
    per_bin = F.smooth_l1_loss(
        delta,
        torch.zeros_like(delta),
        reduction="none",
        beta=float(huber_beta),
    )
    per_sample = (per_bin * mask).sum(dim=1) / steps.clamp_min(1)
    if not bool(comparable.any()):
        zero = (previous.sum() + current.sum()) * 0.0
        return zero, {
            "comparable_count": 0,
            "mean_abs_log10_revision": 0.0,
            "downward_fraction": 0.0,
        }
    selected = per_sample[comparable]
    if weights is None:
        loss = selected.mean()
    else:
        selected_weights = weights[comparable]
        loss = torch.sum(selected_weights * selected) / selected_weights.sum()

    with torch.no_grad():
        absolute_log = (
            torch.abs(delta) * mask
        ).sum(dim=1) / steps.clamp_min(1)
        previous_confirmed = (
            torch.clamp(previous, min=0.0) * dt_column * mask
        ).sum(dim=1)
        current_confirmed = (
            torch.clamp(current, min=0.0) * dt_column * mask
        ).sum(dim=1)
        mean_abs_log = float(absolute_log[comparable].mean().detach().cpu())
        downward = float(
            (current_confirmed[comparable] < previous_confirmed[comparable])
            .float()
            .mean()
            .detach()
            .cpu()
        )
    return loss, {
        "comparable_count": int(comparable.sum().detach().cpu()),
        "mean_abs_log10_revision": mean_abs_log,
        "downward_fraction": downward,
    }


def _absolute_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.abs(np.asarray(values, dtype=np.float64))
    if array.size == 0:
        return {
            "count": 0,
            "median": float("nan"),
            "p95": float("nan"),
            "maximum": float("nan"),
        }
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def evaluate_validation_streaming(
    model: PINNModel,
    config: dict[str, Any],
    validation_loader: Any,
    criterion: Any,
    *,
    max_batches: int | None = None,
) -> dict[str, Any]:
    parameter = next(model.parameters())
    device = parameter.device
    beta_m_per_s = float(config["physics"]["beta"])
    horizons = VALIDATION_HORIZONS
    mw_by_horizon: dict[int, list[float]] = {horizon: [] for horizon in horizons}
    events: list[str] = []
    catalogs: list[float] = []
    station_steps: list[float] = []
    confirmed_log_revisions: list[float] = []
    confirmed_downward: list[bool] = []
    original_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                prepared = _prepare_v2_batch(batch, config, device)
                events.extend([str(value) for value in batch["event"]])
                catalogs.extend(
                    [float(value) for value in batch["magnitude_catalog"].view(-1)]
                )
                previous_rate: torch.Tensor | None = None
                previous_mw: torch.Tensor | None = None
                for horizon in horizons:
                    encoded = model(
                        prepared.model_input[:, :, :horizon],
                        meta=prepared.metadata,
                    )
                    rate = criterion._decode_rate(encoded)
                    mw = moment_magnitude_from_rate(rate, prepared.source_dt_sec)
                    mw_by_horizon[horizon].extend(
                        [float(value) for value in mw.detach().cpu()]
                    )
                    if previous_rate is not None and previous_mw is not None:
                        station_steps.extend(
                            [float(value) for value in (mw - previous_mw).cpu()]
                        )
                        steps = conservative_s_supported_steps(
                            prepared.source_distance_m,
                            previous_horizon_sec=horizon - 1,
                            beta_m_per_s=beta_m_per_s,
                            source_steps=rate.shape[1],
                        )
                        source_index = torch.arange(
                            rate.shape[1], device=device
                        ).unsqueeze(0)
                        mask = source_index < steps.unsqueeze(1)
                        comparable = steps > 0
                        dt = prepared.source_dt_sec.reshape(-1, 1)
                        previous_cumulative = torch.cumsum(
                            previous_rate * dt, dim=1
                        ).clamp_min(1.0e10)
                        current_cumulative = torch.cumsum(
                            rate * dt, dim=1
                        ).clamp_min(1.0e10)
                        absolute_log = (
                            torch.abs(
                                torch.log10(current_cumulative)
                                - torch.log10(previous_cumulative)
                            )
                            * mask
                        ).sum(dim=1) / steps.clamp_min(1)
                        confirmed_log_revisions.extend(
                            [
                                float(value)
                                for value in absolute_log[comparable].cpu()
                            ]
                        )
                        previous_confirmed = (
                            previous_rate * dt * mask
                        ).sum(dim=1)
                        current_confirmed = (rate * dt * mask).sum(dim=1)
                        confirmed_downward.extend(
                            [
                                bool(value)
                                for value in (
                                    current_confirmed[comparable]
                                    < previous_confirmed[comparable]
                                ).cpu()
                            ]
                        )
                    previous_rate = rate
                    previous_mw = mw
    finally:
        model.train(original_training)

    if not events:
        raise ValueError("validation loader produced no samples")
    event_predictions: dict[int, dict[str, float]] = {}
    endpoint_station_rows: list[dict[str, Any]] = []
    endpoint_event_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        station_rows = [
            {
                "event": events[index],
                "mw_pred": mw_by_horizon[horizon][index],
                "mw_catalog": catalogs[index],
            }
            for index in range(len(events))
        ]
        event_rows = aggregate_event_predictions(
            station_rows,
            reference_key="mw_catalog",
        )
        event_predictions[horizon] = {
            str(row["event"]): float(row["mw_pred_median"])
            for row in event_rows
        }
        if horizon == 200:
            endpoint_station_rows = station_rows
            endpoint_event_rows = event_rows
    endpoint = summarize_predictions(
        endpoint_station_rows,
        endpoint_event_rows,
        reference_key="mw_catalog",
    )
    event_steps = [
        event_predictions[horizon][event]
        - event_predictions[horizon - 1][event]
        for horizon in horizons[1:]
        for event in sorted(event_predictions[horizon])
    ]
    station_summary = _absolute_summary(station_steps)
    event_summary = _absolute_summary(event_steps)
    confirmed_summary = _absolute_summary(confirmed_log_revisions)
    return {
        "validation_station_count": len(events),
        "validation_event_count": len(set(events)),
        "endpoint_event_mae": float(endpoint["event_mae"]),
        "endpoint_station_mae": float(endpoint["station_mae"]),
        "late_event_abs_step_median_mw": event_summary["median"],
        "late_event_abs_step_p95_mw": event_summary["p95"],
        "late_event_abs_step_max_mw": event_summary["maximum"],
        "late_station_abs_step_median_mw": station_summary["median"],
        "late_station_abs_step_p95_mw": station_summary["p95"],
        "late_station_abs_step_max_mw": station_summary["maximum"],
        "late_confirmed_cumulative_log10_l1_median": confirmed_summary["median"],
        "late_confirmed_cumulative_log10_l1_p95": confirmed_summary["p95"],
        "late_confirmed_cumulative_log10_l1_max": confirmed_summary["maximum"],
        "late_confirmed_downward_fraction": (
            float(np.mean(confirmed_downward))
            if confirmed_downward
            else float("nan")
        ),
        "late_confirmed_comparison_count": len(confirmed_downward),
    }


def validation_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    endpoint_event_pass = (
        float(metrics["endpoint_event_mae"])
        <= GATES["endpoint_event_mae_max"]
    )
    endpoint_station_pass = (
        float(metrics["endpoint_station_mae"])
        <= GATES["endpoint_station_mae_max"]
    )
    ratios = {
        "late_event_p95_ratio": (
            float(metrics["late_event_abs_step_p95_mw"])
            / GATES["late_event_abs_step_p95_mw_max"]
        ),
        "late_station_p95_ratio": (
            float(metrics["late_station_abs_step_p95_mw"])
            / GATES["late_station_abs_step_p95_mw_max"]
        ),
        "late_confirmed_log_p95_ratio": (
            float(metrics["late_confirmed_cumulative_log10_l1_p95"])
            / GATES["late_confirmed_cumulative_log10_l1_p95_max"]
        ),
    }
    selection_score = max(ratios.values())
    endpoint_preserved = endpoint_event_pass and endpoint_station_pass
    stability_passed = selection_score <= 1.0
    return {
        "endpoint_event_passed": endpoint_event_pass,
        "endpoint_station_passed": endpoint_station_pass,
        "endpoint_preserved": endpoint_preserved,
        **ratios,
        "selection_score": selection_score,
        "stability_passed": stability_passed,
        "passed": endpoint_preserved and stability_passed,
    }


def _assert_formal_baseline(metrics: Mapping[str, Any]) -> None:
    comparisons = {
        "endpoint_event_mae": BASELINE_METRICS["endpoint_event_mae"],
        "endpoint_station_mae": BASELINE_METRICS["endpoint_station_mae"],
        "late_event_abs_step_p95_mw": BASELINE_METRICS[
            "late_event_abs_step_p95_mw"
        ],
        "late_station_abs_step_p95_mw": BASELINE_METRICS[
            "late_station_abs_step_p95_mw"
        ],
        "late_confirmed_cumulative_log10_l1_p95": BASELINE_METRICS[
            "late_confirmed_cumulative_log10_l1_p95"
        ],
    }
    for name, expected in comparisons.items():
        actual = float(metrics[name])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=2.0e-7):
            raise ValueError(
                f"formal Phase39 validation baseline changed for {name}: "
                f"{actual} != {expected}"
            )


def _protocol_payload(trainable_scope: str) -> dict[str, Any]:
    prefixes = TRAINABLE_SCOPE_PREFIXES[trainable_scope]
    return {
        "source_model": "Phase39 Glehman scalar + global invariant, seed42",
        "source_checkpoint": str(PHASE39_CHECKPOINT),
        "trainable_scope": trainable_scope,
        "trainable_parameter_prefixes": list(prefixes),
        "expected_trainable_parameter_count": (
            EXPECTED_SCOPE_PARAMETER_COUNTS[trainable_scope]
        ),
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "lambda_stream_consistency": LAMBDA_STREAM_CONSISTENCY,
        "consistency_coordinate": "cumulative_log10_moment",
        "consistency_huber_beta": CONSISTENCY_HUBER_BETA,
        "consistency_gradient": "symmetric paired-prefix",
        "minimum_prefix_horizon": MIN_PREFIX_HORIZON,
        "maximum_prefix_horizon": MAX_PREFIX_HORIZON,
        "horizon_cycle_multiplier": HORIZON_CYCLE_MULTIPLIER,
        "horizon_cycle_offset": HORIZON_CYCLE_OFFSET,
        "validation_horizons": list(VALIDATION_HORIZONS),
        "baseline_metrics": dict(BASELINE_METRICS),
        "gates": dict(GATES),
        "checkpoint_selection": (
            "among endpoint-preserving epochs, minimize the worst normalized "
            "late event/station/confirmed-history p95 gate ratio"
        ),
        "hidden_data": (
            "internal test, external development events, and grouped test are not iterated"
        ),
    }


def _validate_output_root(path: Path, *, smoke: bool) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output root must be new or empty: {path}")
    if "smoke" in path.name.lower() and not smoke:
        raise ValueError("formal output root must not be named as smoke")


def run_phase40(
    *,
    output_root: Path,
    smoke: bool,
    device: torch.device,
    trainable_scope: str = "heads",
) -> dict[str, Any]:
    _validate_output_root(output_root, smoke=smoke)
    output_root.mkdir(parents=True, exist_ok=True)
    source_hashes = validate_source_artifacts()
    config = load_frozen_config()
    configure_runtime(42, device)

    train_loader, validation_loader, test_loader, split_manifest = (
        get_data_loaders_v2(config)
    )
    if split_manifest["assignment_sha256"] != EXPECTED_SPLIT_ASSIGNMENT_SHA256:
        raise ValueError("Phase39 train/validation/test assignment changed")
    if len(train_loader.dataset) != 1788 or len(validation_loader.dataset) != 385:
        raise ValueError("Phase39 train/validation record counts changed")
    # The test loader is deliberately never iterated.
    del test_loader

    model = PINNModel(config).to(device)
    source_state = torch.load(
        PHASE39_CHECKPOINT,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(source_state, strict=True)
    trainable_parameters, trainable_count = freeze_trainable_scope(
        model,
        trainable_scope,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = _build_stf_rate_criterion(config, device)
    event_weights = _training_event_balance_weights(config, train_loader)

    protocol = _protocol_payload(trainable_scope)
    _write_json(output_root / "protocol.json", protocol)
    config_snapshot = copy.deepcopy(config)
    config_snapshot["phase40_streaming_consistency"] = protocol
    with (output_root / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config_snapshot,
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
    _write_json(output_root / "split.json", split_manifest)

    baseline_metrics = evaluate_validation_streaming(
        model,
        config,
        validation_loader,
        criterion,
        max_batches=2 if smoke else None,
    )
    if not smoke:
        _assert_formal_baseline(baseline_metrics)
    _write_json(output_root / "baseline_validation_metrics.json", baseline_metrics)

    best_metrics = dict(baseline_metrics)
    best_gate = validation_gate(best_metrics) if not smoke else None
    best_score = (
        float(best_gate["selection_score"])
        if best_gate is not None
        else float("inf")
    )
    best_epoch = 0
    atomic_torch_save(dict(model.state_dict()), output_root / "best_model.pth")

    epochs = 1 if smoke else EPOCHS
    max_train_batches = 2 if smoke else None
    rows: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_seen = 0
        total_loss_sum = 0.0
        base_loss_sum = 0.0
        consistency_loss_sum = 0.0
        comparable_sum = 0
        train_revision_sum = 0.0
        train_downward_weighted_sum = 0.0

        for batch_index, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            prepared = _prepare_v2_batch(batch, config, device)
            sample_weights = _batch_event_sample_weights(
                batch,
                event_weights,
                reference=prepared.radial,
            )
            optimizer.zero_grad(set_to_none=True)

            model.train()
            full_encoded = model(
                prepared.model_input,
                meta=prepared.metadata,
            )
            base_loss, base_parts = criterion(
                full_encoded,
                pred_catalog_mw=None,
                radial_obs=prepared.radial,
                source_distance_m=prepared.source_distance_m,
                theta_deg=prepared.theta_deg,
                phi_slip_deg=prepared.phi_slip_deg,
                source_dt_sec=prepared.source_dt_sec,
                observation_dt_sec=prepared.observation_dt_sec,
                waveform_valid_mask=prepared.waveform_valid_mask,
                stf_true=prepared.stf_true,
                has_stf=prepared.has_stf,
                true_mag=prepared.true_mag,
                sample_weights=sample_weights,
            )

            horizon = horizon_for_step(global_step)
            model.eval()
            previous_encoded = model(
                prepared.model_input[:, :, :horizon],
                meta=prepared.metadata,
            )
            current_encoded = model(
                prepared.model_input[:, :, : horizon + 1],
                meta=prepared.metadata,
            )
            previous_rate = criterion._decode_rate(previous_encoded)
            current_rate = criterion._decode_rate(current_encoded)
            consistency_loss, consistency_metrics = (
                cumulative_log_consistency_from_rate(
                    previous_rate,
                    current_rate,
                    source_distance_m=prepared.source_distance_m,
                    source_dt_sec=prepared.source_dt_sec,
                    previous_horizon_sec=horizon,
                    beta_m_per_s=float(config["physics"]["beta"]),
                    sample_weights=sample_weights,
                )
            )
            model.train()
            total_loss = base_loss + LAMBDA_STREAM_CONSISTENCY * consistency_loss
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("Phase40 training loss became non-finite")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=GRAD_CLIP_NORM,
            )
            optimizer.step()

            batch_size = int(prepared.radial.shape[0])
            train_seen += batch_size
            total_loss_sum += float(total_loss.detach().cpu()) * batch_size
            base_loss_sum += float(base_loss.detach().cpu()) * batch_size
            consistency_loss_sum += (
                float(consistency_loss.detach().cpu()) * batch_size
            )
            comparable = int(consistency_metrics["comparable_count"])
            comparable_sum += comparable
            train_revision_sum += (
                float(consistency_metrics["mean_abs_log10_revision"])
                * comparable
            )
            train_downward_weighted_sum += (
                float(consistency_metrics["downward_fraction"])
                * comparable
            )
            global_step += 1

        if train_seen == 0:
            raise ValueError("training loader produced no samples")
        validation_metrics = evaluate_validation_streaming(
            model,
            config,
            validation_loader,
            criterion,
            max_batches=2 if smoke else None,
        )
        gate = validation_gate(validation_metrics) if not smoke else None
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_total_loss": total_loss_sum / train_seen,
            "train_phase39_base_loss": base_loss_sum / train_seen,
            "train_stream_consistency_loss": consistency_loss_sum / train_seen,
            "train_consistency_comparable_count": comparable_sum,
            "train_mean_abs_log10_revision": (
                train_revision_sum / comparable_sum if comparable_sum else 0.0
            ),
            "train_confirmed_downward_fraction": (
                train_downward_weighted_sum / comparable_sum
                if comparable_sum
                else 0.0
            ),
            **validation_metrics,
            "endpoint_preserved": (
                None if gate is None else bool(gate["endpoint_preserved"])
            ),
            "selection_score": (
                None if gate is None else float(gate["selection_score"])
            ),
            "validation_gate_passed": (
                None if gate is None else bool(gate["passed"])
            ),
        }
        rows.append(row)
        if gate is not None and bool(gate["endpoint_preserved"]):
            score = float(gate["selection_score"])
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_metrics = dict(validation_metrics)
                best_gate = dict(gate)
                atomic_torch_save(
                    dict(model.state_dict()),
                    output_root / "best_model.pth",
                )
        atomic_torch_save(dict(model.state_dict()), output_root / "last_model.pth")
        _write_json(output_root / "epoch_metrics.json", rows)
        _write_csv(
            output_root / "epoch_metrics.csv",
            rows,
            fieldnames=tuple(rows[0]),
        )
        print(
            f"epoch={epoch}/{epochs} "
            f"base={row['train_phase39_base_loss']:.6f} "
            f"stream={row['train_stream_consistency_loss']:.6f} "
            f"val_event={row['endpoint_event_mae']:.6f} "
            f"late_event_p95={row['late_event_abs_step_p95_mw']:.6f} "
            f"late_confirmed_p95="
            f"{row['late_confirmed_cumulative_log10_l1_p95']:.6f} "
            f"score={row['selection_score']}",
            flush=True,
        )

    if smoke:
        atomic_torch_save(dict(model.state_dict()), output_root / "best_model.pth")
        selected_metrics = rows[-1]
        selected_gate = None
        passed = True
        status = "smoke_complete"
        selected_epoch = 1
    else:
        selected_metrics = best_metrics
        selected_gate = best_gate
        selected_epoch = best_epoch
        passed = bool(best_gate and best_gate["passed"] and best_epoch > 0)
        status = "validation_gate_passed" if passed else "validation_gate_failed"

    best_checkpoint_path = output_root / "best_model.pth"
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "git_dirty": git_is_dirty(PROJECT_ROOT),
        "device": str(device),
        "trainable_scope": trainable_scope,
        "source_artifact_sha256": source_hashes,
        "source_checkpoint": str(PHASE39_CHECKPOINT),
        "source_split_assignment_sha256": split_manifest["assignment_sha256"],
        "internal_test_iterated": False,
        "external_data_loaded": False,
        "grouped_test_loaded": False,
    }
    _write_json(output_root / "provenance.json", provenance)
    summary = {
        "status": status,
        "passed": passed,
        "smoke": smoke,
        "source_model": "Phase39 Glehman scalar + global invariant, seed42",
        "trainable_scope": trainable_scope,
        "trainable_parameter_count": trainable_count,
        "selected_epoch": selected_epoch,
        "best_checkpoint": {
            "path": str(best_checkpoint_path),
            "sha256": sha256_file(best_checkpoint_path),
        },
        "baseline_metrics": baseline_metrics,
        "selected_metrics": selected_metrics,
        "selected_gate": selected_gate,
        "protocol": protocol,
        "provenance": provenance,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a frozen scope of the Phase39 model with an "
            "S-supported paired-prefix consistency objective."
        )
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="New or empty output directory.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument(
        "--trainable-scope",
        choices=tuple(TRAINABLE_SCOPE_PREFIXES),
        default="heads",
        help=(
            "Parameter scope to fine-tune. The default reproduces Phase40; "
            "last_transformer is the frozen Phase41 follow-up."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run two training/validation batches for one epoch without formal gates.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    output_root = args.output_root.resolve()
    summary = run_phase40(
        output_root=output_root,
        smoke=bool(args.smoke),
        device=torch.device(args.device),
        trainable_scope=str(args.trainable_scope),
    )
    print(json.dumps({
        "output_root": str(output_root),
        "status": summary["status"],
        "passed": summary["passed"],
        "selected_epoch": summary["selected_epoch"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
