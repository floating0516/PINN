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

from scripts.experiments.run_phase40_streaming_consistency import (  # noqa: E402
    conservative_s_supported_steps,
    cumulative_log_consistency_from_rate,
)
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


PHASE39_RUN_ROOT = Path(
    "/home/lihe/PINN_Mag/runs/"
    "phase39-manuscript-stf-glehman-scalar-20260726T111256Z-7fbb7a7/"
    "train/candidate"
)
SOURCE_TIMESTAMPS = {
    17: "20260726_192532",
    42: "20260726_192953",
    73: "20260726_193303",
}
SOURCE_SHA256 = {
    17: {
        "checkpoint": "e796fe77f60ad158f7e2103ed281eb0ffb579ea4c7fa81d61dfe5d30d099e03a",
        "config": "600e18549a4b56565b648faf1a4108faf8c746e96b1da882ed9354d558b6eb84",
        "split": "a053f5f82e1a349d3999b1cceff5282dc384418ca5f1a14d0e58e6bab0fd9ee2",
    },
    42: {
        "checkpoint": "73500f365a58b248204d02333716f31674435927e9fc1c7d55a1453786b406f7",
        "config": "a05181166c7f40cae755ffbfbd0f4adfdb6a83703a89299677fa2be5f8ff1966",
        "split": "0ca718d5966004a768f0c648dc6dc2211b55a031819db6abeac8a111e75788c4",
    },
    73: {
        "checkpoint": "e1954134400a86198542a289a7d14c3143399107615587585b8926206128e8c9",
        "config": "8607bcedded9678817cd9cddf18834af99ee2083c41c169c379eb79ad1cfde91",
        "split": "890dd8408beb23bf6403005e3250c24e17b9a3586346f7d6cdf1df6be51ad635",
    },
}
SOURCE_SPLIT_ASSIGNMENT_SHA256 = {
    17: "fa5c5d1cd3bdb3e8a775140a9bea4adce885b151eff717627d5e8ab82fd4e9a8",
    42: "5ac2e07ed186dce737a3592694632775b7bbf603bf922a4a74fa6b86a3d5c240",
    73: "786d029482fc8c6c8000b939380f7d4f6fdab2cc0dfe39a09fa982d1d9049548",
}
SOURCE_VALIDATION_EVENT_MAE = {
    17: 0.12392580509185791,
    42: 0.11433351834615071,
    73: 0.17061189810434976,
}

SEEDS = (17, 42, 73)
EXPECTED_PARAMETER_COUNT = 1_010_850
EXPECTED_RECORD_COUNTS = {
    "train": 1788,
    "validation": 385,
    "test": 385,
}

EPOCHS = 20
LEARNING_RATE = 1.0e-5
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP_NORM = 1.0
FULL_SCIENCE_WEIGHT = 0.5
PREFIX_SCIENCE_WEIGHT = 0.5
MW_STEP_WEIGHT = 0.2
MW_STEP_HUBER_BETA = 0.02
HISTORY_STEP_WEIGHT = 0.05
HISTORY_HUBER_BETA = 0.05
CONSISTENCY_START_HORIZON = 60
MIN_PREFIX_HORIZON = 20
MAX_PREFIX_HORIZON = 200
HORIZON_CYCLE_MULTIPLIER = 73

ACCURACY_HORIZONS = (20, 40, 60, 80, 100, 120, 140, 160, 180, 200)
LATE_HORIZONS = tuple(range(179, 201))
VALIDATION_HORIZONS = tuple(sorted(set((*ACCURACY_HORIZONS, *LATE_HORIZONS))))

ENDPOINT_MARGIN_MW = 0.005
STREAMING_ACCURACY_RATIO = 0.95
LATE_EVENT_STEP_RATIO = 0.80
LATE_STATION_STEP_RATIO = 1.00
LATE_HISTORY_STEP_RATIO = 0.80


def _source_paths(seed: int) -> dict[str, Path]:
    if seed not in SOURCE_TIMESTAMPS:
        raise ValueError(f"unsupported Phase47 seed: {seed}")
    model_dir = (
        PHASE39_RUN_ROOT
        / f"seed_{seed}"
        / "models"
        / SOURCE_TIMESTAMPS[seed]
    )
    return {
        "checkpoint": model_dir / "best_model.pth",
        "config": model_dir / "config.yaml",
        "split": model_dir / "split.json",
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


def _validate_output_root(path: Path, *, smoke: bool) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output root must be new or empty: {path}")
    if "smoke" in path.name.lower() and not smoke:
        raise ValueError("formal output root must not be named as smoke")


def validate_source_artifacts(seed: int) -> dict[str, str]:
    paths = _source_paths(seed)
    actual: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing Phase39 seed{seed} {name}: {path}")
        actual[name] = sha256_file(path)
        expected = SOURCE_SHA256[seed][name]
        if actual[name] != expected:
            raise ValueError(
                f"Phase39 seed{seed} {name} SHA-256 changed: "
                f"{actual[name]} != {expected}"
            )
    return actual


def load_seed_config(seed: int) -> dict[str, Any]:
    with _source_paths(seed)["config"].open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config_on_startup(config)
    training = config["training"]
    loss = training["stf_rate_loss"]
    expected = {
        "seed": (int(training["random_seed"]), seed),
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
        raise ValueError(
            f"frozen Phase39 seed{seed} config changed: " + ", ".join(changed)
        )
    if tuple(config["model"]["input_components"]) != ("radial",):
        raise ValueError("Phase47 requires the frozen R-only model")
    if config["model"]["stf_output_parameterization"] != "moment_shape_factorized":
        raise ValueError("Phase47 requires the factorized STF head")
    if loss["synth_polarity_mode"] != "global_invariant":
        raise ValueError("Phase47 requires global-invariant synthesis")
    if loss["radiation_coefficient_contract"] != "glehman_scalar":
        raise ValueError("Phase47 requires Glehman scalar radiation")
    return config


def horizon_for_step(step: int, seed: int) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if seed not in SEEDS:
        raise ValueError(f"unsupported Phase47 seed: {seed}")
    span = MAX_PREFIX_HORIZON - MIN_PREFIX_HORIZON + 1
    return MIN_PREFIX_HORIZON + (
        (seed + HORIZON_CYCLE_MULTIPLIER * step) % span
    )


def late_consistency_weight(horizon: int) -> float:
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError("horizon must be an integer")
    if horizon < MIN_PREFIX_HORIZON or horizon > MAX_PREFIX_HORIZON:
        raise ValueError("horizon is outside the Phase47 prefix range")
    if horizon <= CONSISTENCY_START_HORIZON:
        return 0.0
    span = MAX_PREFIX_HORIZON - CONSISTENCY_START_HORIZON
    progress = (horizon - CONSISTENCY_START_HORIZON) / span
    return float(progress * progress)


def mw_step_consistency(
    previous_mw: torch.Tensor,
    current_mw: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    huber_beta: float = MW_STEP_HUBER_BETA,
) -> tuple[torch.Tensor, dict[str, float]]:
    previous = previous_mw.reshape(-1)
    current = current_mw.reshape(-1)
    if previous.shape != current.shape or previous.numel() < 1:
        raise ValueError("paired Mw tensors must have the same nonempty shape")
    if not math.isfinite(huber_beta) or huber_beta <= 0.0:
        raise ValueError("huber_beta must be positive and finite")
    delta = current - previous
    per_sample = F.smooth_l1_loss(
        delta,
        torch.zeros_like(delta),
        reduction="none",
        beta=float(huber_beta),
    )
    if sample_weights is None:
        loss = per_sample.mean()
    else:
        weights = sample_weights.reshape(-1).to(
            device=current.device,
            dtype=current.dtype,
        )
        if weights.shape != current.shape:
            raise ValueError("sample_weights must match the paired Mw shape")
        if bool(torch.any(~torch.isfinite(weights))) or bool(torch.any(weights <= 0.0)):
            raise ValueError("sample_weights must be finite and positive")
        loss = (weights * per_sample).mean()
    return loss, {
        "mean_abs_mw_step": float(torch.abs(delta).mean().detach().cpu()),
        "downward_fraction": float((delta < 0.0).float().mean().detach().cpu()),
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
    mw_by_horizon: dict[int, list[float]] = {
        horizon: [] for horizon in VALIDATION_HORIZONS
    }
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
                previous_horizon: int | None = None
                for horizon in VALIDATION_HORIZONS:
                    encoded = model(
                        prepared.model_input[:, :, :horizon],
                        meta=prepared.metadata,
                    )
                    rate = criterion._decode_rate(encoded)
                    mw = moment_magnitude_from_rate(rate, prepared.source_dt_sec)
                    mw_by_horizon[horizon].extend(
                        [float(value) for value in mw.detach().cpu()]
                    )
                    consecutive_late = (
                        previous_horizon is not None
                        and horizon == previous_horizon + 1
                        and horizon >= 180
                    )
                    if consecutive_late and previous_rate is not None and previous_mw is not None:
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
                    previous_horizon = horizon
    finally:
        model.train(original_training)

    if not events:
        raise ValueError("validation loader produced no samples")
    event_predictions: dict[int, dict[str, float]] = {}
    horizon_event_mae: dict[int, float] = {}
    endpoint_station_rows: list[dict[str, Any]] = []
    endpoint_event_rows: list[dict[str, Any]] = []
    for horizon in VALIDATION_HORIZONS:
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
        horizon_event_mae[horizon] = float(
            np.mean(
                [
                    abs(float(row["mw_pred_median"]) - float(row["mw_catalog"]))
                    for row in event_rows
                ]
            )
        )
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
        for horizon in range(180, 201)
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
        "streaming_event_mae_mean": float(
            np.mean([horizon_event_mae[value] for value in ACCURACY_HORIZONS])
        ),
        "streaming_event_mae_by_horizon": {
            str(value): horizon_event_mae[value] for value in ACCURACY_HORIZONS
        },
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


def validation_gate(
    metrics: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    targets = {
        "endpoint_event_mae": (
            float(baseline["endpoint_event_mae"]) + ENDPOINT_MARGIN_MW
        ),
        "endpoint_station_mae": (
            float(baseline["endpoint_station_mae"]) + ENDPOINT_MARGIN_MW
        ),
        "streaming_event_mae_mean": (
            STREAMING_ACCURACY_RATIO
            * float(baseline["streaming_event_mae_mean"])
        ),
        "late_event_abs_step_p95_mw": (
            LATE_EVENT_STEP_RATIO
            * float(baseline["late_event_abs_step_p95_mw"])
        ),
        "late_station_abs_step_p95_mw": (
            LATE_STATION_STEP_RATIO
            * float(baseline["late_station_abs_step_p95_mw"])
        ),
        "late_confirmed_cumulative_log10_l1_p95": (
            LATE_HISTORY_STEP_RATIO
            * float(baseline["late_confirmed_cumulative_log10_l1_p95"])
        ),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in targets.values()):
        raise ValueError("validation targets must be positive and finite")
    ratios = {
        f"{name}_ratio": float(metrics[name]) / target
        for name, target in targets.items()
    }
    endpoint_preserved = (
        ratios["endpoint_event_mae_ratio"] <= 1.0
        and ratios["endpoint_station_mae_ratio"] <= 1.0
    )
    selection_score = max(ratios.values())
    return {
        "targets": targets,
        **ratios,
        "endpoint_preserved": endpoint_preserved,
        "streaming_accuracy_passed": (
            ratios["streaming_event_mae_mean_ratio"] <= 1.0
        ),
        "stability_passed": all(
            ratios[name] <= 1.0
            for name in (
                "late_event_abs_step_p95_mw_ratio",
                "late_station_abs_step_p95_mw_ratio",
                "late_confirmed_cumulative_log10_l1_p95_ratio",
            )
        ),
        "selection_score": selection_score,
        "passed": endpoint_preserved and selection_score <= 1.0,
    }


def _protocol_payload() -> dict[str, Any]:
    return {
        "source_model": "Phase39 Glehman scalar + global invariant",
        "seeds": list(SEEDS),
        "architecture_changed": False,
        "adapter_used": False,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "trainable_scope": "all existing Phase39 parameters",
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "full_science_weight": FULL_SCIENCE_WEIGHT,
        "prefix_science_weight": PREFIX_SCIENCE_WEIGHT,
        "prefix_contract": "true variable-length B x 1 x h radial input",
        "minimum_prefix_horizon": MIN_PREFIX_HORIZON,
        "maximum_prefix_horizon": MAX_PREFIX_HORIZON,
        "horizon_cycle_multiplier": HORIZON_CYCLE_MULTIPLIER,
        "previous_prefix_teacher": "same model, eval mode, stop-gradient",
        "mw_step_weight": MW_STEP_WEIGHT,
        "mw_step_huber_beta": MW_STEP_HUBER_BETA,
        "confirmed_history_weight": HISTORY_STEP_WEIGHT,
        "confirmed_history_huber_beta": HISTORY_HUBER_BETA,
        "consistency_start_horizon": CONSISTENCY_START_HORIZON,
        "consistency_time_weight": "quadratic from 0 at 60 s to 1 at 200 s",
        "validation_horizons": list(VALIDATION_HORIZONS),
        "accuracy_horizons": list(ACCURACY_HORIZONS),
        "gates": {
            "endpoint_margin_mw": ENDPOINT_MARGIN_MW,
            "streaming_accuracy_ratio": STREAMING_ACCURACY_RATIO,
            "late_event_step_ratio": LATE_EVENT_STEP_RATIO,
            "late_station_step_ratio": LATE_STATION_STEP_RATIO,
            "late_history_step_ratio": LATE_HISTORY_STEP_RATIO,
        },
        "hidden_data": (
            "internal test, external development events, and grouped test are not iterated"
        ),
    }


def _science_loss(
    model: PINNModel,
    criterion: Any,
    prepared: Any,
    *,
    horizon: int,
    sample_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    encoded = model(
        prepared.model_input[:, :, :horizon],
        meta=prepared.metadata,
    )
    loss, parts = criterion(
        encoded,
        pred_catalog_mw=None,
        radial_obs=prepared.radial[:, :, :horizon],
        source_distance_m=prepared.source_distance_m,
        theta_deg=prepared.theta_deg,
        phi_slip_deg=prepared.phi_slip_deg,
        source_dt_sec=prepared.source_dt_sec,
        observation_dt_sec=prepared.observation_dt_sec,
        waveform_valid_mask=prepared.waveform_valid_mask[:, :horizon],
        stf_true=prepared.stf_true,
        has_stf=prepared.has_stf,
        true_mag=prepared.true_mag,
        sample_weights=sample_weights,
    )
    rate = criterion._decode_rate(encoded)
    mw = moment_magnitude_from_rate(rate, prepared.source_dt_sec)
    return loss, parts, rate, mw


def _fixed_train_objective(
    model: PINNModel,
    config: dict[str, Any],
    train_loader: Any,
    criterion: Any,
    *,
    seed: int,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    device = next(model.parameters()).device
    event_weights = _training_event_balance_weights(config, train_loader)
    totals = {
        "total": 0.0,
        "full": 0.0,
        "prefix": 0.0,
        "mw_step": 0.0,
        "history_step": 0.0,
    }
    seen = 0
    original_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(train_loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                prepared = _prepare_v2_batch(batch, config, device)
                sample_weights = _batch_event_sample_weights(
                    batch,
                    event_weights,
                    reference=prepared.radial,
                )
                full_loss, _, _, _ = _science_loss(
                    model,
                    criterion,
                    prepared,
                    horizon=MAX_PREFIX_HORIZON,
                    sample_weights=sample_weights,
                )
                horizon = horizon_for_step(batch_index, seed)
                previous_horizon = max(1, horizon - 1)
                previous_encoded = model(
                    prepared.model_input[:, :, :previous_horizon],
                    meta=prepared.metadata,
                )
                previous_rate = criterion._decode_rate(previous_encoded)
                previous_mw = moment_magnitude_from_rate(
                    previous_rate,
                    prepared.source_dt_sec,
                )
                prefix_loss, _, current_rate, current_mw = _science_loss(
                    model,
                    criterion,
                    prepared,
                    horizon=horizon,
                    sample_weights=sample_weights,
                )
                mw_loss, _ = mw_step_consistency(
                    previous_mw,
                    current_mw,
                    sample_weights=sample_weights,
                )
                history_loss, _ = cumulative_log_consistency_from_rate(
                    previous_rate,
                    current_rate,
                    source_distance_m=prepared.source_distance_m,
                    source_dt_sec=prepared.source_dt_sec,
                    previous_horizon_sec=previous_horizon,
                    beta_m_per_s=float(config["physics"]["beta"]),
                    huber_beta=HISTORY_HUBER_BETA,
                    sample_weights=sample_weights,
                )
                time_weight = late_consistency_weight(horizon)
                total = (
                    FULL_SCIENCE_WEIGHT * full_loss
                    + PREFIX_SCIENCE_WEIGHT * prefix_loss
                    + time_weight
                    * (
                        MW_STEP_WEIGHT * mw_loss
                        + HISTORY_STEP_WEIGHT * history_loss
                    )
                )
                batch_size = int(prepared.radial.shape[0])
                seen += batch_size
                totals["total"] += float(total.cpu()) * batch_size
                totals["full"] += float(full_loss.cpu()) * batch_size
                totals["prefix"] += float(prefix_loss.cpu()) * batch_size
                totals["mw_step"] += float(mw_loss.cpu()) * batch_size
                totals["history_step"] += float(history_loss.cpu()) * batch_size
    finally:
        model.train(original_training)
    if seen == 0:
        raise ValueError("training loader produced no fixed-objective samples")
    return {
        "sample_count": seen,
        "fixed_total_loss": totals["total"] / seen,
        "fixed_full_science_loss": totals["full"] / seen,
        "fixed_prefix_science_loss": totals["prefix"] / seen,
        "fixed_mw_step_loss": totals["mw_step"] / seen,
        "fixed_history_step_loss": totals["history_step"] / seen,
    }


def run_seed(
    *,
    seed: int,
    output_root: Path,
    smoke: bool,
    device: torch.device,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    source_hashes = validate_source_artifacts(seed)
    config = load_seed_config(seed)
    configure_runtime(seed, device)

    train_loader, validation_loader, test_loader, split_manifest = (
        get_data_loaders_v2(config)
    )
    if split_manifest["assignment_sha256"] != SOURCE_SPLIT_ASSIGNMENT_SHA256[seed]:
        raise ValueError(f"Phase39 seed{seed} split assignment changed")
    counts = {
        "train": len(train_loader.dataset),
        "validation": len(validation_loader.dataset),
        "test": len(test_loader.dataset),
    }
    if counts != EXPECTED_RECORD_COUNTS:
        raise ValueError(f"Phase39 seed{seed} record counts changed: {counts}")
    del test_loader

    model = PINNModel(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"Phase39 parameter count changed: {parameter_count} "
            f"!= {EXPECTED_PARAMETER_COUNT}"
        )
    source_state = torch.load(
        _source_paths(seed)["checkpoint"],
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(source_state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = _build_stf_rate_criterion(config, device)
    event_weights = _training_event_balance_weights(config, train_loader)

    protocol = _protocol_payload()
    _write_json(output_root / "protocol.json", protocol)
    config_snapshot = copy.deepcopy(config)
    config_snapshot["phase47_direct_streaming_retrain"] = protocol
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
    if not smoke and not math.isclose(
        float(baseline_metrics["endpoint_event_mae"]),
        SOURCE_VALIDATION_EVENT_MAE[seed],
        rel_tol=0.0,
        abs_tol=2.0e-7,
    ):
        raise ValueError(
            f"Phase39 seed{seed} validation endpoint changed: "
            f"{baseline_metrics['endpoint_event_mae']}"
        )
    _write_json(
        output_root / "baseline_validation_metrics.json",
        baseline_metrics,
    )

    baseline_gate = validation_gate(baseline_metrics, baseline_metrics)
    best_metrics = dict(baseline_metrics)
    best_gate = dict(baseline_gate)
    best_score = float(baseline_gate["selection_score"])
    best_epoch = 0
    atomic_torch_save(dict(model.state_dict()), output_root / "best_model.pth")

    epochs = 1 if smoke else EPOCHS
    max_train_batches = 2 if smoke else None
    rows: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_seen = 0
        sums = {
            "total": 0.0,
            "full": 0.0,
            "prefix": 0.0,
            "mw_step": 0.0,
            "history_step": 0.0,
            "time_weight": 0.0,
            "abs_mw_step": 0.0,
            "mw_downward": 0.0,
            "abs_history_step": 0.0,
            "history_downward": 0.0,
        }
        comparable_history = 0

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
            full_loss, _, _, _ = _science_loss(
                model,
                criterion,
                prepared,
                horizon=MAX_PREFIX_HORIZON,
                sample_weights=sample_weights,
            )
            (FULL_SCIENCE_WEIGHT * full_loss).backward()

            horizon = horizon_for_step(global_step, seed)
            previous_horizon = max(1, horizon - 1)
            model.eval()
            with torch.no_grad():
                previous_encoded = model(
                    prepared.model_input[:, :, :previous_horizon],
                    meta=prepared.metadata,
                )
                previous_rate = criterion._decode_rate(previous_encoded)
                previous_mw = moment_magnitude_from_rate(
                    previous_rate,
                    prepared.source_dt_sec,
                )

            model.train()
            prefix_loss, _, current_rate, current_mw = _science_loss(
                model,
                criterion,
                prepared,
                horizon=horizon,
                sample_weights=sample_weights,
            )
            mw_loss, mw_metrics = mw_step_consistency(
                previous_mw,
                current_mw,
                sample_weights=sample_weights,
            )
            history_loss, history_metrics = cumulative_log_consistency_from_rate(
                previous_rate,
                current_rate,
                source_distance_m=prepared.source_distance_m,
                source_dt_sec=prepared.source_dt_sec,
                previous_horizon_sec=previous_horizon,
                beta_m_per_s=float(config["physics"]["beta"]),
                huber_beta=HISTORY_HUBER_BETA,
                sample_weights=sample_weights,
            )
            time_weight = late_consistency_weight(horizon)
            remaining_loss = (
                PREFIX_SCIENCE_WEIGHT * prefix_loss
                + time_weight
                * (
                    MW_STEP_WEIGHT * mw_loss
                    + HISTORY_STEP_WEIGHT * history_loss
                )
            )
            if not bool(torch.isfinite(remaining_loss)):
                raise FloatingPointError("Phase47 training loss became non-finite")
            remaining_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP_NORM,
            )
            optimizer.step()

            total_loss = FULL_SCIENCE_WEIGHT * full_loss + remaining_loss
            batch_size = int(prepared.radial.shape[0])
            train_seen += batch_size
            sums["total"] += float(total_loss.detach().cpu()) * batch_size
            sums["full"] += float(full_loss.detach().cpu()) * batch_size
            sums["prefix"] += float(prefix_loss.detach().cpu()) * batch_size
            sums["mw_step"] += float(mw_loss.detach().cpu()) * batch_size
            sums["history_step"] += float(history_loss.detach().cpu()) * batch_size
            sums["time_weight"] += time_weight * batch_size
            sums["abs_mw_step"] += mw_metrics["mean_abs_mw_step"] * batch_size
            sums["mw_downward"] += mw_metrics["downward_fraction"] * batch_size
            comparable = int(history_metrics["comparable_count"])
            comparable_history += comparable
            sums["abs_history_step"] += (
                float(history_metrics["mean_abs_log10_revision"]) * comparable
            )
            sums["history_downward"] += (
                float(history_metrics["downward_fraction"]) * comparable
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
        gate = validation_gate(validation_metrics, baseline_metrics)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_online_total_loss": sums["total"] / train_seen,
            "train_online_full_science_loss": sums["full"] / train_seen,
            "train_online_prefix_science_loss": sums["prefix"] / train_seen,
            "train_online_mw_step_loss": sums["mw_step"] / train_seen,
            "train_online_history_step_loss": sums["history_step"] / train_seen,
            "train_mean_time_weight": sums["time_weight"] / train_seen,
            "train_mean_abs_mw_step": sums["abs_mw_step"] / train_seen,
            "train_mw_downward_fraction": sums["mw_downward"] / train_seen,
            "train_mean_abs_history_log10_step": (
                sums["abs_history_step"] / comparable_history
                if comparable_history
                else 0.0
            ),
            "train_history_downward_fraction": (
                sums["history_downward"] / comparable_history
                if comparable_history
                else 0.0
            ),
            **{
                key: value
                for key, value in validation_metrics.items()
                if key != "streaming_event_mae_by_horizon"
            },
            "endpoint_preserved": bool(gate["endpoint_preserved"]),
            "selection_score": float(gate["selection_score"]),
            "validation_gate_passed": bool(gate["passed"]),
        }
        rows.append(row)
        if bool(gate["endpoint_preserved"]):
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
            f"seed={seed} epoch={epoch}/{epochs} "
            f"full={row['train_online_full_science_loss']:.6f} "
            f"prefix={row['train_online_prefix_science_loss']:.6f} "
            f"val_event={row['endpoint_event_mae']:.6f} "
            f"stream_mae={row['streaming_event_mae_mean']:.6f} "
            f"late_p95={row['late_event_abs_step_p95_mw']:.6f} "
            f"score={row['selection_score']:.6f}",
            flush=True,
        )

    if smoke:
        atomic_torch_save(dict(model.state_dict()), output_root / "best_model.pth")
        best_epoch = 1
        best_metrics = dict(validation_metrics)
        best_gate = dict(gate)

    best_checkpoint = output_root / "best_model.pth"
    model.load_state_dict(
        torch.load(best_checkpoint, map_location=device, weights_only=True),
        strict=True,
    )
    fixed_train = _fixed_train_objective(
        model,
        config,
        train_loader,
        criterion,
        seed=seed,
        max_batches=2 if smoke else None,
    )
    _write_json(output_root / "selected_fixed_train_objective.json", fixed_train)

    passed = bool(smoke or (best_epoch > 0 and best_gate["passed"]))
    status = (
        "smoke_complete"
        if smoke
        else "validation_gate_passed"
        if passed
        else "validation_gate_failed"
    )
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "git_dirty": git_is_dirty(PROJECT_ROOT),
        "device": str(device),
        "source_artifact_sha256": source_hashes,
        "source_checkpoint": str(_source_paths(seed)["checkpoint"]),
        "source_split_assignment_sha256": split_manifest["assignment_sha256"],
        "architecture_changed": False,
        "adapter_used": False,
        "internal_test_iterated": False,
        "external_data_loaded": False,
        "grouped_test_loaded": False,
    }
    _write_json(output_root / "provenance.json", provenance)
    summary = {
        "status": status,
        "passed": passed,
        "smoke": smoke,
        "seed": seed,
        "source_model": "Phase39 Glehman scalar + global invariant",
        "parameter_count": parameter_count,
        "selected_epoch": best_epoch,
        "best_checkpoint": {
            "path": str(best_checkpoint),
            "sha256": sha256_file(best_checkpoint),
        },
        "baseline_metrics": baseline_metrics,
        "selected_metrics": best_metrics,
        "selected_gate": best_gate,
        "selected_fixed_train_objective": fixed_train,
        "protocol": protocol,
        "provenance": provenance,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def run_campaign(
    *,
    output_root: Path,
    smoke: bool,
    device: torch.device,
) -> dict[str, Any]:
    _validate_output_root(output_root, smoke=smoke)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = _protocol_payload()
    _write_json(output_root / "protocol.json", protocol)
    selected_seeds = (42,) if smoke else SEEDS
    summaries = [
        run_seed(
            seed=seed,
            output_root=output_root / f"seed_{seed}",
            smoke=smoke,
            device=device,
        )
        for seed in selected_seeds
    ]
    passing = [summary for summary in summaries if bool(summary["passed"])]
    selected = (
        min(
            passing,
            key=lambda summary: (
                float(summary["selected_gate"]["selection_score"]),
                float(summary["selected_metrics"]["endpoint_event_mae"]),
                int(summary["seed"]),
            ),
        )
        if passing
        else None
    )
    status = (
        "smoke_complete"
        if smoke
        else "validation_gate_passed"
        if selected is not None
        else "validation_gate_failed"
    )
    campaign = {
        "status": status,
        "passed": bool(selected is not None),
        "smoke": smoke,
        "selected_seed": None if selected is None else int(selected["seed"]),
        "selected_checkpoint": (
            None if selected is None else selected["best_checkpoint"]
        ),
        "seed_summaries": {
            str(summary["seed"]): {
                "status": summary["status"],
                "passed": summary["passed"],
                "selected_epoch": summary["selected_epoch"],
                "selection_score": summary["selected_gate"]["selection_score"],
                "endpoint_event_mae": summary["selected_metrics"][
                    "endpoint_event_mae"
                ],
                "streaming_event_mae_mean": summary["selected_metrics"][
                    "streaming_event_mae_mean"
                ],
                "late_event_abs_step_p95_mw": summary["selected_metrics"][
                    "late_event_abs_step_p95_mw"
                ],
                "checkpoint": summary["best_checkpoint"],
            }
            for summary in summaries
        },
        "protocol": protocol,
        "hidden_data": {
            "internal_test_iterated": False,
            "external_data_loaded": False,
            "grouped_test_loaded": False,
        },
    }
    _write_json(output_root / "campaign_summary.json", campaign)
    return campaign


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the unchanged Phase39 model directly on variable-length "
            "streaming prefixes; no adapter or new model module is used."
        )
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="New or empty campaign output directory.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run seed42 for one epoch with two train/validation batches.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    output_root = args.output_root.resolve()
    summary = run_campaign(
        output_root=output_root,
        smoke=bool(args.smoke),
        device=torch.device(args.device),
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "status": summary["status"],
                "passed": summary["passed"],
                "selected_seed": summary["selected_seed"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
