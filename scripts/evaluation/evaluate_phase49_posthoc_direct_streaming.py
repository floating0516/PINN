#!/usr/bin/env python3
"""Post-hoc eight-event replay for frozen Phase39/47/48 direct models."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.evaluate_phase39_second_by_second import (  # noqa: E402
    CHECKPOINT_PATH as PHASE39_CHECKPOINT,
    EXTERNAL_EVENT_ROOT,
    EXPECTED_EVENT_COUNT,
    EXPECTED_STATION_COUNT,
    LABELS_PATH,
    _git_commit,
    hash_external_inputs,
    load_endpoint_reference,
    load_label_contract,
    load_phase39_config,
    validate_frozen_artifacts,
)
from scripts.evaluation.evaluate_phase39_streaming_replay import (  # noqa: E402
    build_raw_streaming_records,
)
from scripts.evaluation.evaluate_phase45_posthoc_streaming import (  # noqa: E402
    ENDPOINT_PREDICTION_TOLERANCE_MW,
    HORIZONS,
    LATE_HORIZONS,
    PROCESSING_DELAY_SEC,
    SOURCE_DT_SEC,
    TARGET_ABS_ERROR_MW,
    _endpoint_external_gate,
    _error_summary,
    _late_metrics,
    _mw_cube,
    _suffix_stable_horizon,
    _validate_new_directory,
    generate_external_raw_rates,
)
from scripts.experiments.run_phase47_direct_streaming_retrain import (  # noqa: E402
    EXPECTED_PARAMETER_COUNT,
    load_seed_config,
    validate_source_artifacts,
)
from src.models.model import PINNModel  # noqa: E402
from src.utils.provenance import sha256_file  # noqa: E402


RUNS_ROOT = PROJECT_ROOT.parent.parent / "runs"
RECOVERY_ROOT = RUNS_ROOT / "phase49-posthoc-recovery-exact-20260728T164549Z"
PHASE47_RECOVERY_ROOT = RECOVERY_ROOT / "phase47_seed73_epoch19"
PHASE48_RECOVERY_ROOT = RECOVERY_ROOT / "phase48_seed73_epoch188"
PHASE47_ORIGINAL_ROOT = (
    RUNS_ROOT
    / "phase47-direct-phase39-streaming-20260728T141807Z-31a3271"
    / "seed_73"
)
PHASE48_ORIGINAL_ROOT = (
    RUNS_ROOT
    / "phase48-joint-phase39-streaming-20260728T143620Z-3ff2449"
    / "seed_73"
)

EXPECTED_PHASE39_SHA256 = (
    "73500f365a58b248204d02333716f31674435927e9fc1c7d55a1453786b406f7"
)
EXPECTED_PHASE47_RECOVERY_SHA256 = (
    "045388c621467249b8bb5efea081fd015e478e7e62ddffca85be247f9620ad17"
)
EXPECTED_PHASE48_RECOVERY_SHA256 = (
    "8be2470122aaa6e81b379c9a73f9fefbece2e660b0ac9885978c9f8c7d98dbee"
)
STRICT_RECOVERY_METRIC_TOLERANCE = 5.0e-4
STRICT_RECOVERY_SCORE_TOLERANCE = 1.0e-2
RECOVERY_METRIC_KEYS = (
    "endpoint_event_mae",
    "endpoint_station_mae",
    "streaming_event_mae_mean",
    "late_event_abs_step_p95_mw",
    "late_station_abs_step_p95_mw",
    "late_confirmed_cumulative_log10_l1_p95",
)
DEFAULT_BATCH_SIZE = EXPECTED_STATION_COUNT


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    label: str
    checkpoint: Path
    expected_sha256: str
    seed: int
    epoch: int
    original_metrics: Path | None = None
    recovered_metrics: Path | None = None
    recovery_metric_tolerance: float = STRICT_RECOVERY_METRIC_TOLERANCE
    recovery_score_tolerance: float = STRICT_RECOVERY_SCORE_TOLERANCE
    recovery_role: str = "strict_reproduction"


MODEL_SPECS = (
    ModelSpec(
        slug="phase39",
        label="Phase39 seed42",
        checkpoint=PHASE39_CHECKPOINT,
        expected_sha256=EXPECTED_PHASE39_SHA256,
        seed=42,
        epoch=200,
    ),
    ModelSpec(
        slug="phase47",
        label="Phase47 seed73 epoch19",
        checkpoint=PHASE47_RECOVERY_ROOT / "last_model.pth",
        expected_sha256=EXPECTED_PHASE47_RECOVERY_SHA256,
        seed=73,
        epoch=19,
        original_metrics=PHASE47_ORIGINAL_ROOT / "epoch_metrics.json",
        recovered_metrics=PHASE47_RECOVERY_ROOT / "epoch_metrics.json",
    ),
    ModelSpec(
        slug="phase48",
        label="Phase48 seed73 epoch188",
        checkpoint=PHASE48_RECOVERY_ROOT / "last_model.pth",
        expected_sha256=EXPECTED_PHASE48_RECOVERY_SHA256,
        seed=73,
        epoch=188,
        original_metrics=PHASE48_ORIGINAL_ROOT / "epoch_metrics.json",
        recovered_metrics=PHASE48_RECOVERY_ROOT / "epoch_metrics.json",
        recovery_metric_tolerance=5.0e-3,
        recovery_score_tolerance=3.0e-2,
        recovery_role="user_authorized_approximate_reconstruction",
    ),
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    names = tuple(fieldnames or (tuple(rows[0]) if rows else ()))
    if not names:
        raise ValueError(f"cannot infer fields for empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {name: "" if row.get(name) is None else row.get(name) for name in names}
            )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _recovery_row(spec: ModelSpec, path: Path) -> Mapping[str, Any]:
    rows = _read_json(path)
    if not isinstance(rows, list) or len(rows) < spec.epoch:
        raise ValueError(f"{spec.slug} metrics do not contain epoch {spec.epoch}")
    row = rows[spec.epoch - 1]
    if int(row["epoch"]) != spec.epoch:
        raise ValueError(f"{spec.slug} target metric row moved")
    return row


def compare_recovery_metrics(
    spec: ModelSpec,
    original: Mapping[str, Any],
    recovered: Mapping[str, Any],
) -> dict[str, Any]:
    differences = {
        key: abs(float(recovered[key]) - float(original[key]))
        for key in RECOVERY_METRIC_KEYS
    }
    score_difference = abs(
        float(recovered["selection_score"]) - float(original["selection_score"])
    )
    max_metric_difference = max(differences.values(), default=0.0)
    strict_passed = bool(
        max_metric_difference <= STRICT_RECOVERY_METRIC_TOLERANCE
        and score_difference <= STRICT_RECOVERY_SCORE_TOLERANCE
    )
    if max_metric_difference > spec.recovery_metric_tolerance:
        raise ValueError(
            f"{spec.slug} recovery metric difference {max_metric_difference:.9g} "
            f"exceeds {spec.recovery_metric_tolerance:.9g}"
        )
    if score_difference > spec.recovery_score_tolerance:
        raise ValueError(
            f"{spec.slug} recovery score difference {score_difference:.9g} "
            f"exceeds {spec.recovery_score_tolerance:.9g}"
        )
    return {
        "epoch": spec.epoch,
        "role": spec.recovery_role,
        "strict_reproduction_passed": strict_passed,
        "metric_abs_differences": differences,
        "max_metric_abs_difference": max_metric_difference,
        "selection_score_abs_difference": score_difference,
        "strict_metric_tolerance": STRICT_RECOVERY_METRIC_TOLERANCE,
        "strict_selection_score_tolerance": STRICT_RECOVERY_SCORE_TOLERANCE,
        "accepted_metric_tolerance": spec.recovery_metric_tolerance,
        "accepted_selection_score_tolerance": spec.recovery_score_tolerance,
        "original": {
            key: float(original[key])
            for key in (*RECOVERY_METRIC_KEYS, "selection_score")
        },
        "recovered": {
            key: float(recovered[key])
            for key in (*RECOVERY_METRIC_KEYS, "selection_score")
        },
    }


def validate_model_artifacts() -> dict[str, Any]:
    frozen_external = validate_frozen_artifacts()
    phase39_seed73 = validate_source_artifacts(73)
    results: dict[str, Any] = {}
    for spec in MODEL_SPECS:
        if not spec.expected_sha256:
            raise ValueError(f"{spec.slug} checkpoint hash has not been frozen")
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(f"missing {spec.slug} checkpoint: {spec.checkpoint}")
        checkpoint_sha = sha256_file(spec.checkpoint)
        if checkpoint_sha != spec.expected_sha256:
            raise ValueError(
                f"{spec.slug} checkpoint changed: {checkpoint_sha} != "
                f"{spec.expected_sha256}"
            )
        recovery = None
        if spec.original_metrics is not None and spec.recovered_metrics is not None:
            original = _recovery_row(spec, spec.original_metrics)
            recovered = _recovery_row(spec, spec.recovered_metrics)
            recovery = compare_recovery_metrics(spec, original, recovered)
        results[spec.slug] = {
            "label": spec.label,
            "seed": spec.seed,
            "epoch": spec.epoch,
            "checkpoint_sha256": checkpoint_sha,
            "recovery": recovery,
        }
    return {
        "models": results,
        "phase39_frozen_external": frozen_external,
        "phase39_seed73_source": phase39_seed73,
    }


def _inference_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": config["dataset"],
        "model": config["model"],
        "physics": config["physics"],
    }


def validate_config_compatibility(
    phase39_config: Mapping[str, Any],
    direct_config: Mapping[str, Any],
) -> None:
    if _inference_contract(phase39_config) != _inference_contract(direct_config):
        raise ValueError("Phase39 seed42 and seed73 inference contracts differ")


def load_direct_model(
    config: Mapping[str, Any],
    *,
    checkpoint: Path,
    device: torch.device,
) -> PINNModel:
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = PINNModel(dict(config)).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(f"model parameter count changed: {parameter_count}")
    model.eval()
    return model


def _percentile_abs(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.abs(np.asarray(values, dtype=np.float64)), percentile))


def build_multimodel_tables(
    *,
    rates_by_model: Mapping[str, np.ndarray],
    available_mask: np.ndarray,
    events: Sequence[str],
    stations: Sequence[str],
    catalogs: np.ndarray,
    source_dt_sec: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    model_order = [spec.slug for spec in MODEL_SPECS]
    if set(rates_by_model) != set(model_order):
        raise ValueError("rate cube model set changed")
    mask = np.asarray(available_mask, dtype=bool)
    catalogs_array = np.asarray(catalogs, dtype=np.float64)
    mw_by_model = {
        slug: _mw_cube(np.asarray(rates_by_model[slug]), source_dt_sec)
        for slug in model_order
    }
    expected_shape = (len(HORIZONS), len(events))
    if mask.shape != expected_shape:
        raise ValueError(f"availability shape changed: {mask.shape} != {expected_shape}")
    if any(cube.shape != expected_shape for cube in mw_by_model.values()):
        raise ValueError("Mw cube shape changed")

    event_names = sorted(set(str(value) for value in events))
    event_indices = {
        event: np.asarray(
            [index for index, value in enumerate(events) if str(value) == event],
            dtype=np.int64,
        )
        for event in event_names
    }
    station_rows: list[dict[str, Any]] = []
    endpoint_station_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    previous_event: dict[tuple[str, str], float] = {}

    for horizon_index, horizon in enumerate(HORIZONS):
        available = np.flatnonzero(mask[horizon_index])
        for slug in model_order:
            cube = mw_by_model[slug]
            station_errors = cube[horizon_index, available] - catalogs_array[available]
            station_steps: list[float] = []
            for index in available:
                delta = None
                if horizon_index > 0 and mask[horizon_index - 1, index]:
                    delta = float(cube[horizon_index, index] - cube[horizon_index - 1, index])
                    station_steps.append(delta)
                row = {
                    "model": slug,
                    "event": str(events[index]),
                    "station": str(stations[index]),
                    "observation_horizon_sec": int(horizon),
                    "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                    "mw_catalog": float(catalogs_array[index]),
                    "mw_pred": float(cube[horizon_index, index]),
                    "error": float(cube[horizon_index, index] - catalogs_array[index]),
                    "abs_error": abs(float(cube[horizon_index, index] - catalogs_array[index])),
                    "delta_mw": delta,
                }
                station_rows.append(row)
                if horizon == HORIZONS[-1]:
                    endpoint_station_rows.append(row)

            current_event_rows: list[dict[str, Any]] = []
            event_steps: list[float] = []
            for event in event_names:
                indices = event_indices[event]
                current = indices[mask[horizon_index, indices]]
                if current.size == 0:
                    continue
                event_catalogs = catalogs_array[current]
                if not np.allclose(event_catalogs, event_catalogs[0], rtol=0.0, atol=1e-7):
                    raise ValueError(f"catalog magnitude differs within event {event}")
                prediction = float(np.median(cube[horizon_index, current]))
                previous = previous_event.get((slug, event))
                delta = None if previous is None else prediction - previous
                if delta is not None:
                    event_steps.append(delta)
                event_row = {
                    "model": slug,
                    "event": event,
                    "observation_horizon_sec": int(horizon),
                    "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                    "mw_catalog": float(event_catalogs[0]),
                    "mw_pred_median": prediction,
                    "error": prediction - float(event_catalogs[0]),
                    "abs_error": abs(prediction - float(event_catalogs[0])),
                    "delta_mw": delta,
                    "station_count": int(current.size),
                }
                current_event_rows.append(event_row)
                previous_event[(slug, event)] = prediction
            event_rows.extend(current_event_rows)

            station_summary = _error_summary(station_errors)
            event_summary = _error_summary([row["error"] for row in current_event_rows])
            horizon_rows.append(
                {
                    "model": slug,
                    "observation_horizon_sec": int(horizon),
                    "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                    "available_station_count": int(available.size),
                    "event_count": len(current_event_rows),
                    "station_mae": station_summary["mae"],
                    "station_rmse": station_summary["rmse"],
                    "station_bias": station_summary["bias"],
                    "event_mae": event_summary["mae"],
                    "event_rmse": event_summary["rmse"],
                    "event_bias": event_summary["bias"],
                    "event_abs_step_p95": _percentile_abs(event_steps, 95),
                    "station_abs_step_p95": _percentile_abs(station_steps, 95),
                }
            )

    convergence_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[(str(row["model"]), str(row["event"]))].append(row)
    for slug in model_order:
        for event in event_names:
            rows = sorted(
                grouped[(slug, event)],
                key=lambda row: int(row["observation_horizon_sec"]),
            )
            final = rows[-1]
            stable = _suffix_stable_horizon(rows, error_key="abs_error")
            convergence_rows.append(
                {
                    "model": slug,
                    "event": event,
                    "mw_catalog": float(final["mw_catalog"]),
                    "final_mw": float(final["mw_pred_median"]),
                    "final_abs_error": float(final["abs_error"]),
                    "stable_observation_sec": stable,
                    "stable_release_sec": (
                        None if stable is None else stable + PROCESSING_DELAY_SEC
                    ),
                    "station_count_200s": int(final["station_count"]),
                }
            )

    endpoint_event_rows: list[dict[str, Any]] = []
    endpoint_lookup = {
        (str(row["model"]), str(row["event"])): row
        for row in event_rows
        if int(row["observation_horizon_sec"]) == HORIZONS[-1]
    }
    for event in event_names:
        baseline = endpoint_lookup[("phase39", event)]
        row: dict[str, Any] = {
            "event": event,
            "mw_catalog": float(baseline["mw_catalog"]),
            "station_count_200s": int(baseline["station_count"]),
        }
        for slug in model_order:
            value = endpoint_lookup[(slug, event)]
            row[f"{slug}_mw"] = float(value["mw_pred_median"])
            row[f"{slug}_abs_error"] = float(value["abs_error"])
            if slug != "phase39":
                row[f"{slug}_abs_error_change_vs_phase39"] = (
                    float(value["abs_error"]) - float(baseline["abs_error"])
                )
        endpoint_event_rows.append(row)

    return {
        "station_rows": station_rows,
        "endpoint_station_rows": endpoint_station_rows,
        "event_rows": event_rows,
        "horizon_rows": horizon_rows,
        "convergence_rows": convergence_rows,
        "endpoint_event_rows": endpoint_event_rows,
    }


def _phase39_endpoint_gate(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    event_rows = [
        {
            "event": row["event"],
            "observation_horizon_sec": row["observation_horizon_sec"],
            "raw_mw_pred_median": row["mw_pred_median"],
            "raw_abs_error": row["abs_error"],
        }
        for row in tables["event_rows"]
        if row["model"] == "phase39"
    ]
    station_rows = [
        {
            "event": row["event"],
            "station": row["station"],
            "raw_mw_pred": row["mw_pred"],
        }
        for row in tables["endpoint_station_rows"]
        if row["model"] == "phase39"
    ]
    return _endpoint_external_gate(event_rows, station_rows, reference)


def _endpoint_metrics(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    slug: str,
) -> dict[str, float]:
    row = next(
        item
        for item in tables["horizon_rows"]
        if item["model"] == slug
        and int(item["observation_horizon_sec"]) == HORIZONS[-1]
    )
    return {
        "event_mae": float(row["event_mae"]),
        "event_rmse": float(row["event_rmse"]),
        "event_bias": float(row["event_bias"]),
        "station_mae": float(row["station_mae"]),
        "station_rmse": float(row["station_rmse"]),
        "station_bias": float(row["station_bias"]),
    }


def _convergence_summary(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    slug: str,
) -> dict[str, Any]:
    rows = [row for row in tables["convergence_rows"] if row["model"] == slug]
    stable = [
        int(row["stable_observation_sec"])
        for row in rows
        if row["stable_observation_sec"] is not None
    ]
    return {
        "stable_event_count": len(stable),
        "event_count": len(rows),
        "median_stable_observation_sec": (
            None if not stable else float(np.median(np.asarray(stable)))
        ),
        "max_stable_observation_sec": None if not stable else max(stable),
    }


def _write_outputs(
    output_dir: Path,
    *,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    summary: Mapping[str, Any],
) -> None:
    _write_csv(output_dir / "station_predictions.csv", tables["station_rows"])
    _write_csv(
        output_dir / "endpoint_station_predictions.csv",
        tables["endpoint_station_rows"],
    )
    _write_csv(output_dir / "event_predictions.csv", tables["event_rows"])
    _write_csv(output_dir / "horizon_metrics.csv", tables["horizon_rows"])
    _write_csv(output_dir / "event_convergence.csv", tables["convergence_rows"])
    _write_csv(
        output_dir / "endpoint_event_comparison.csv",
        tables["endpoint_event_rows"],
    )
    _write_json(output_dir / "summary.json", summary)


def evaluate_external(
    *,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    _validate_new_directory(output_dir)
    artifact_validation = validate_model_artifacts()
    phase39_config = load_phase39_config()
    direct_config = load_seed_config(73)
    validate_config_compatibility(phase39_config, direct_config)

    # External data are opened only after candidates, hashes, and recovery rows pass.
    external_input_hashes = hash_external_inputs(EXTERNAL_EVENT_ROOT)
    labels = load_label_contract(LABELS_PATH)
    endpoint_reference = load_endpoint_reference()
    records = build_raw_streaming_records(
        config=phase39_config,
        event_root=EXTERNAL_EVENT_ROOT,
        labels_by_dir=labels,
        expected_station_keys=set(endpoint_reference["station_predictions"]),
    )
    if len(records) != EXPECTED_STATION_COUNT:
        raise ValueError("external station count changed")
    if len({record.event for record in records}) != EXPECTED_EVENT_COUNT:
        raise ValueError("external event count changed")

    output_dir.mkdir(parents=True, exist_ok=False)
    rates_by_model: dict[str, np.ndarray] = {}
    input_gates: dict[str, Any] = {}
    common_mask: np.ndarray | None = None
    for spec in MODEL_SPECS:
        config = phase39_config if spec.slug == "phase39" else direct_config
        model = load_direct_model(config, checkpoint=spec.checkpoint, device=device)
        rates, available_mask, input_gate = generate_external_raw_rates(
            config=dict(config),
            records=records,
            model=model,
            output_path=output_dir / f"{spec.slug}_rates.npy",
            batch_size=batch_size,
        )
        if common_mask is None:
            common_mask = np.asarray(available_mask, dtype=bool)
        elif not np.array_equal(common_mask, available_mask):
            raise ValueError(f"{spec.slug} external availability mask changed")
        rates_by_model[spec.slug] = rates
        input_gates[spec.slug] = input_gate
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if common_mask is None:
        raise RuntimeError("no model rates were generated")
    events = [record.event for record in records]
    stations = [record.station for record in records]
    catalogs = np.asarray(
        [record.magnitude_catalog for record in records],
        dtype=np.float32,
    )
    source_distance = np.asarray(
        [record.source_distance_m for record in records],
        dtype=np.float32,
    )
    source_dt = np.full(len(records), SOURCE_DT_SEC, dtype=np.float32)
    tables = build_multimodel_tables(
        rates_by_model=rates_by_model,
        available_mask=common_mask,
        events=events,
        stations=stations,
        catalogs=catalogs,
        source_dt_sec=source_dt,
    )
    endpoint_gate = _phase39_endpoint_gate(tables, endpoint_reference)

    late_start = HORIZONS.index(LATE_HORIZONS[0])
    late_stop = HORIZONS.index(LATE_HORIZONS[-1]) + 1
    if not bool(np.all(common_mask[late_start:late_stop])):
        raise ValueError("external late-horizon cohort is incomplete")
    models: dict[str, Any] = {}
    for spec in MODEL_SPECS:
        late = _late_metrics(
            rates_by_model[spec.slug],
            events=events,
            catalogs=catalogs,
            source_distance_m=source_distance,
            source_dt_sec=source_dt,
            beta_m_per_s=float(phase39_config["physics"]["beta"]),
        )
        endpoint = _endpoint_metrics(tables, spec.slug)
        late.update(
            {
                "endpoint_event_mae": endpoint["event_mae"],
                "endpoint_station_mae": endpoint["station_mae"],
            }
        )
        models[spec.slug] = {
            "label": spec.label,
            "seed": spec.seed,
            "epoch": spec.epoch,
            "endpoint": endpoint,
            "late": late,
            "convergence": _convergence_summary(tables, spec.slug),
            "input_gate": input_gates[spec.slug],
            "rates_sha256": sha256_file(output_dir / f"{spec.slug}_rates.npy"),
        }

    endpoint_events = list(tables["endpoint_event_rows"])
    for slug in ("phase47", "phase48"):
        models[slug]["endpoint_improved_event_count_vs_phase39"] = sum(
            float(row[f"{slug}_abs_error_change_vs_phase39"]) < 0.0
            for row in endpoint_events
        )

    summary = {
        "status": "complete",
        "evaluation_role": "development_validation_posthoc",
        "candidate_selection": (
            "Phase47 seed73 epoch19 and Phase48 seed73 epoch188 were frozen "
            "from validation before external replay; external results were not "
            "used to select a seed or checkpoint."
        ),
        "phase48_reconstruction_limit": (
            "The original Phase48 epoch188 checkpoint was not retained. The "
            "evaluated checkpoint is a fixed numerically nearby CUDA replay. It "
            "failed the original strict reproduction tolerance and is accepted "
            "only for this user-authorized post-hoc diagnostic."
        ),
        "interpretation_limit": (
            "The eight events were excluded from training but repeatedly used "
            "during development. Results are not an unbiased blind test or final "
            "proof of unseen-event generalization."
        ),
        "event_count": EXPECTED_EVENT_COUNT,
        "station_count": EXPECTED_STATION_COUNT,
        "observation_horizons_sec": [int(value) for value in HORIZONS],
        "processing_delay_sec": PROCESSING_DELAY_SEC,
        "models": models,
        "phase39_endpoint_reproduction_gate": endpoint_gate,
        "artifact_validation": artifact_validation,
        "external_input_sha256": external_input_hashes,
        "external_data_loaded": True,
        "internal_test_loaded": False,
        "grouped_test_loaded": False,
        "provenance": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "device": str(device),
            "batch_size": batch_size,
            "phase39_endpoint_tolerance_mw": ENDPOINT_PREDICTION_TOLERANCE_MW,
        },
    }
    _write_outputs(output_dir, tables=tables, summary=summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay frozen Phase39/47/48 direct models on the repeatedly used "
            "eight-event development-validation cohort."
        )
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    summary = evaluate_external(
        output_dir=args.output_dir.resolve(),
        device=_resolve_device(args.device),
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
