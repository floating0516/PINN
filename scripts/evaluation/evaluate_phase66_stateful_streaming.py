#!/usr/bin/env python3
"""Evaluate the frozen Phase66 model on internal test and eight-event replay."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
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
    EXPECTED_PHASE39_SHA256,
    EXPECTED_TEST_COUNT,
    SOURCE_DT_SEC,
    SOURCE_STEPS,
    TARGET_ABS_ERROR_MW,
    _contiguous_segments,
    _endpoint_external_gate,
    _late_metrics,
    _load_internal_samples,
    _mw_cube,
    _resolve_device,
    _validate_new_directory,
    build_comparison_tables,
    build_convergence_rows,
)
from scripts.experiments.run_phase40_streaming_consistency import (  # noqa: E402
    EXPECTED_SPLIT_ASSIGNMENT_SHA256,
    load_frozen_config,
    validate_source_artifacts,
)
from scripts.experiments.run_phase43_streaming_adapter import (  # noqa: E402
    HORIZONS,
    PROCESSING_DELAY_SEC,
    load_cache,
)
from scripts.experiments.run_phase50_stateful_incremental_model import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    EXPECTED_TOTAL_PARAMETER_COUNT,
    LOSS_WEIGHTS,
    VALIDATION_GATES,
    phase50_config,
)
from src.models.model import PINNModel  # noqa: E402
from src.utils.device import configure_runtime  # noqa: E402
from src.utils.provenance import sha256_file  # noqa: E402


PROJECT_HOME = PROJECT_ROOT.parent.parent
PHASE66_ROOT = (
    PROJECT_HOME
    / "runs"
    / "phase66-stronger-overshoot-stateful-20260730T0330Z-9ee21b2"
)
PHASE66_CAMPAIGN_SUMMARY = PHASE66_ROOT / "campaign_summary.json"
PHASE66_SEED_SUMMARY = PHASE66_ROOT / "seed_17" / "summary.json"
PHASE66_EPOCH_METRICS = PHASE66_ROOT / "seed_17" / "epoch_metrics.csv"
PHASE66_CHECKPOINT = PHASE66_ROOT / "seed_17" / "closest_model.pth"
PHASE46_RAW_ROOT = (
    PROJECT_HOME
    / "runs"
    / "phase46-phase45-posthoc-20260728T131632Z-20d35a9"
)
DEFAULT_INTERNAL_RAW_RATES = PHASE46_RAW_ROOT / "internal" / "raw_rates.npy"
DEFAULT_EXTERNAL_RAW_RATES = PHASE46_RAW_ROOT / "external-cpu" / "raw_rates.npy"

EXPECTED_CHECKPOINT_SHA256 = (
    "490d3d7c9948d887e3e284393138223770e62dc217234792224c9639b1d9911e"
)
EXPECTED_INTERNAL_RAW_SHA256 = (
    "666f2e6a3fbf64a2e0889ddf5494f0f8bfa805b3893f6e1863f98f00907fd788"
)
EXPECTED_EXTERNAL_RAW_SHA256 = (
    "7142c39df4e7fac3a9b3757996e7aaa56ffdc73ed623b7567e1aed87729a0ead"
)
EXPECTED_SEED = 17
EXPECTED_EPOCH = 26
DEFAULT_BATCH_SIZE = 64


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
                    key: "" if source.get(key) is None else source.get(key)
                    for key in fieldnames
                }
            )


def validate_phase66_artifacts() -> dict[str, Any]:
    for path in (
        PHASE66_CAMPAIGN_SUMMARY,
        PHASE66_SEED_SUMMARY,
        PHASE66_EPOCH_METRICS,
        PHASE66_CHECKPOINT,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing Phase66 artifact: {path}")

    source_hashes = validate_source_artifacts()
    if source_hashes["checkpoint"] != EXPECTED_PHASE39_SHA256:
        raise ValueError("Phase39 source checkpoint hash changed")
    checkpoint_sha = sha256_file(PHASE66_CHECKPOINT)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Phase66 closest checkpoint hash changed")

    campaign = _read_json(PHASE66_CAMPAIGN_SUMMARY)
    if campaign.get("status") != "validation_gate_failed":
        raise ValueError("Phase66 campaign status changed")
    if campaign.get("passed") is not False:
        raise ValueError("Phase66 unexpectedly passed the frozen gate")
    if campaign.get("selected_seed") is not None:
        raise ValueError("Phase66 formal seed selection must remain empty")
    for key in (
        "internal_test_iterated",
        "external_data_loaded",
        "grouped_test_loaded",
    ):
        if campaign.get(key) is not False:
            raise ValueError(f"Phase66 hidden-data flag changed: {key}")

    seed_summary = _read_json(PHASE66_SEED_SUMMARY)
    if int(seed_summary.get("seed", -1)) != EXPECTED_SEED:
        raise ValueError("Phase66 frozen seed changed")
    if int(seed_summary.get("closest_epoch", -1)) != EXPECTED_EPOCH:
        raise ValueError("Phase66 frozen closest epoch changed")
    if seed_summary["closest_model"]["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Phase66 seed summary points to another checkpoint")
    protocol = seed_summary["protocol"]
    if protocol.get("external_adapter") is not False:
        raise ValueError("Phase66 must remain a model-internal transition")
    if protocol.get("loss_weights") != LOSS_WEIGHTS:
        raise ValueError("Phase66 training loss weights changed")
    return {
        "campaign": campaign,
        "seed_summary": seed_summary,
        "phase39": source_hashes,
        "checkpoint_sha256": checkpoint_sha,
        "campaign_summary_sha256": sha256_file(PHASE66_CAMPAIGN_SUMMARY),
        "seed_summary_sha256": sha256_file(PHASE66_SEED_SUMMARY),
        "epoch_metrics_sha256": sha256_file(PHASE66_EPOCH_METRICS),
    }


def load_phase66_model(*, device: torch.device) -> PINNModel:
    validate_phase66_artifacts()
    configure_runtime(EXPECTED_SEED, device)
    model = PINNModel(phase50_config()).to(device)
    state = torch.load(PHASE66_CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_TOTAL_PARAMETER_COUNT:
        raise ValueError("Phase66 model parameter count changed")
    model.eval()
    return model


def _load_frozen_rates(
    path: Path,
    *,
    expected_sha256: str,
    expected_station_count: int,
) -> tuple[np.ndarray, str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen Phase39 rate cube: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise ValueError(f"frozen Phase39 rate cube hash changed: {path}")
    rates = np.load(path, mmap_mode="r")
    expected_shape = (len(HORIZONS), expected_station_count, SOURCE_STEPS)
    if rates.shape != expected_shape or rates.dtype != np.float32:
        raise ValueError(
            f"frozen Phase39 rate cube changed: {rates.shape}/{rates.dtype}"
        )
    return rates, actual_sha


def extract_training_objective(output_dir: Path) -> dict[str, Any]:
    _validate_new_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = validate_phase66_artifacts()
    protocol = artifacts["seed_summary"]["protocol"]
    weights = {
        name: float(protocol["loss_weights"][name]) for name in LOSS_WEIGHTS
    }
    normalizers = {
        name: float(protocol["loss_normalizers"][name]) for name in LOSS_WEIGHTS
    }

    rows: list[dict[str, Any]] = []
    for raw in _read_csv(PHASE66_EPOCH_METRICS):
        event_ratio = (
            float(raw["endpoint_event_mae"])
            / VALIDATION_GATES["endpoint_event_mae_max"]
        )
        station_ratio = (
            float(raw["endpoint_station_mae"])
            / VALIDATION_GATES["endpoint_station_mae_max"]
        )
        streaming_ratio = float(raw["selection_score"])
        row: dict[str, Any] = {
            "epoch": int(raw["epoch"]),
            "train_total_normalized_loss": float(
                raw["train_total_normalized_loss"]
            ),
            "validation_endpoint_event_mae": float(raw["endpoint_event_mae"]),
            "validation_endpoint_station_mae": float(
                raw["endpoint_station_mae"]
            ),
            "validation_streaming_selection_score": streaming_ratio,
            "closest_rank_max": max(event_ratio, station_ratio, streaming_ratio),
            "closest_rank_sum": event_ratio + station_ratio + streaming_ratio,
            "closest_checkpoint": int(raw["epoch"]) == EXPECTED_EPOCH,
        }
        contribution_sum = 0.0
        for name in weights:
            raw_value = float(raw[f"train_{name}"])
            contribution = weights[name] * raw_value / normalizers[name]
            row[f"train_{name}_raw"] = raw_value
            row[f"train_{name}_weighted_normalized"] = contribution
            contribution_sum += contribution
        row["component_contribution_sum"] = contribution_sum
        if not math.isclose(
            contribution_sum,
            row["train_total_normalized_loss"],
            rel_tol=0.0,
            abs_tol=1.0e-5,
        ):
            raise ValueError(
                f"training objective reconstruction failed at epoch {row['epoch']}"
            )
        rows.append(row)

    closest = next(row for row in rows if row["closest_checkpoint"])
    minimum_total = min(rows, key=lambda row: row["train_total_normalized_loss"])
    minimum_rank = min(
        rows,
        key=lambda row: (row["closest_rank_max"], row["closest_rank_sum"]),
    )
    if int(minimum_rank["epoch"]) != EXPECTED_EPOCH:
        raise ValueError("reconstructed Phase66 closest checkpoint changed")
    payload = {
        "status": "complete",
        "evaluation_role": "saved_training_trace",
        "candidate": "Phase66 seed17 epoch26 frozen closest checkpoint",
        "epoch_count": len(rows),
        "loss_weights": weights,
        "loss_normalizers": normalizers,
        "closest_epoch": EXPECTED_EPOCH,
        "closest_epoch_metrics": closest,
        "minimum_training_total": {
            "epoch": int(minimum_total["epoch"]),
            "value": float(minimum_total["train_total_normalized_loss"]),
        },
        "minimum_closest_rank": {
            "epoch": int(minimum_rank["epoch"]),
            "max_ratio": float(minimum_rank["closest_rank_max"]),
            "sum_ratio": float(minimum_rank["closest_rank_sum"]),
        },
        "artifact_sha256": {
            "phase66_checkpoint": artifacts["checkpoint_sha256"],
            "phase66_campaign_summary": artifacts["campaign_summary_sha256"],
            "phase66_seed_summary": artifacts["seed_summary_sha256"],
            "phase66_epoch_metrics": artifacts["epoch_metrics_sha256"],
        },
    }
    _write_csv(
        output_dir / "training_loss_by_epoch.csv",
        rows,
        fieldnames=tuple(rows[0]),
    )
    _write_json(output_dir / "summary.json", payload)
    return payload


def apply_phase66_to_rates(
    *,
    model: PINNModel,
    raw_rates: np.ndarray,
    available_mask: np.ndarray,
    source_distance_m: np.ndarray,
    source_dt_sec: np.ndarray,
    beta_m_per_s: float,
    output_path: Path,
    batch_size: int,
) -> tuple[np.memmap, dict[str, float]]:
    cube = np.asarray(raw_rates)
    mask = np.asarray(available_mask, dtype=bool)
    if cube.shape[:2] != mask.shape or cube.shape[2] != SOURCE_STEPS:
        raise ValueError("raw rates and availability shape mismatch")
    if cube.shape[0] != len(HORIZONS):
        raise ValueError("Phase66 rate cube horizon count changed")
    station_count = cube.shape[1]
    distances = np.asarray(source_distance_m, dtype=np.float32)
    source_dt = np.asarray(source_dt_sec, dtype=np.float32)
    if distances.shape != (station_count,) or source_dt.shape != (station_count,):
        raise ValueError("Phase66 metadata station count changed")

    segment_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for station_index in range(station_count):
        available_indices = np.flatnonzero(mask[:, station_index])
        for segment in _contiguous_segments(available_indices):
            segment_groups[segment].append(station_index)

    parameter = next(model.parameters())
    device = parameter.device
    stateful = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=cube.shape,
    )
    stateful[:] = np.nan
    gate_sum = 0.0
    gate_count = 0
    with torch.no_grad():
        for (start_index, stop_index), stations in sorted(segment_groups.items()):
            segment_horizons = HORIZONS[start_index:stop_index]
            for start in range(0, len(stations), batch_size):
                batch_stations = stations[start : start + batch_size]
                selected_raw = np.take(
                    cube[start_index:stop_index],
                    batch_stations,
                    axis=1,
                ).transpose(1, 0, 2)
                raw_tensor = torch.as_tensor(
                    np.asarray(selected_raw).copy(),
                    device=device,
                    dtype=parameter.dtype,
                )
                states, _, _, gates = model.stream_sequence_from_rates(
                    raw_tensor,
                    horizons_sec=segment_horizons,
                    source_distance_m=torch.as_tensor(
                        distances[batch_stations],
                        device=device,
                        dtype=parameter.dtype,
                    ),
                    source_dt_sec=torch.as_tensor(
                        source_dt[batch_stations],
                        device=device,
                        dtype=parameter.dtype,
                    ),
                    beta_m_per_s=beta_m_per_s,
                )
                state_values = states.detach().cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
                for local_index, station_index in enumerate(batch_stations):
                    stateful[start_index:stop_index, station_index] = state_values[
                        local_index
                    ]
                gate_sum += float(gates.sum().cpu())
                gate_count += int(gates.numel())
            stateful.flush()

    finite_mask = np.all(np.isfinite(stateful), axis=2)
    if not np.array_equal(finite_mask, mask):
        raise ValueError("Phase66 availability differs from raw availability")
    if np.any(np.asarray(stateful)[mask] < 0.0):
        raise ValueError("Phase66 produced a negative STF rate")
    return stateful, {"mean_retention_gate": gate_sum / max(gate_count, 1)}


def _streaming_metrics(
    rates: np.ndarray,
    *,
    available_mask: np.ndarray,
    events: Sequence[str],
    catalogs: np.ndarray,
    source_distance_m: np.ndarray,
    source_dt_sec: np.ndarray,
    beta_m_per_s: float,
) -> dict[str, Any]:
    metrics = _late_metrics(
        rates,
        events=events,
        catalogs=catalogs,
        source_distance_m=source_distance_m,
        source_dt_sec=source_dt_sec,
        beta_m_per_s=beta_m_per_s,
    )
    mask = np.asarray(available_mask, dtype=bool)
    after_60 = np.asarray(HORIZONS) >= 60
    if not bool(np.all(mask[after_60])):
        raise ValueError("Phase66 trajectory metrics require a fixed post-60 cohort")
    station_mw = _mw_cube(rates, source_dt_sec)
    event_array = np.asarray([str(event) for event in events])
    event_mw = np.stack(
        [
            np.median(station_mw[:, event_array == event], axis=1)
            for event in sorted(set(event_array))
        ]
    )
    event_steps = np.diff(event_mw, axis=1)
    event_downward = np.maximum(-event_steps, 0.0)
    event_after_60 = event_mw[:, after_60]
    event_peak_to_final = np.max(event_after_60, axis=1) - event_after_60[:, -1]
    station_after_60 = station_mw[after_60].T
    station_peak_to_final = (
        np.max(station_after_60, axis=1) - station_after_60[:, -1]
    )
    metrics.update(
        {
            "event_downward_step_p95_mw": float(
                np.percentile(event_downward, 95)
            ),
            "event_downward_step_max_mw": float(np.max(event_downward)),
            "event_downward_fraction": float(np.mean(event_steps < 0.0)),
            "event_peak_to_final_p95_mw": float(
                np.percentile(event_peak_to_final, 95)
            ),
            "event_peak_to_final_mean_mw": float(
                np.mean(event_peak_to_final)
            ),
            "station_peak_to_final_p95_mw": float(
                np.percentile(station_peak_to_final, 95)
            ),
            "event_start_to_end_increase_fraction": float(
                np.mean(event_mw[:, -1] >= event_mw[:, 0])
            ),
        }
    )
    return metrics


def _rename_phase66(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key.replace("adapted", "phase66"): value for key, value in row.items()
    }


def _trajectory_diagnostics(
    event_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[str(row["event"])].append(row)
    output: list[dict[str, Any]] = []
    for event in sorted(grouped):
        rows = sorted(
            grouped[event],
            key=lambda row: int(row["observation_horizon_sec"]),
        )
        post60 = [row for row in rows if int(row["observation_horizon_sec"]) >= 60]
        raw_values = np.asarray(
            [float(row["raw_mw_pred_median"]) for row in post60]
        )
        phase66_values = np.asarray(
            [float(row["adapted_mw_pred_median"]) for row in post60]
        )
        raw_steps = np.diff(raw_values)
        phase66_steps = np.diff(phase66_values)
        final = rows[-1]
        output.append(
            {
                "event": event,
                "mw_catalog": float(final["mw_catalog"]),
                "raw_max_down_step_after_60_mw": float(
                    np.max(np.maximum(-raw_steps, 0.0))
                ),
                "phase66_max_down_step_after_60_mw": float(
                    np.max(np.maximum(-phase66_steps, 0.0))
                ),
                "raw_peak_to_final_drop_after_60_mw": float(
                    np.max(raw_values) - raw_values[-1]
                ),
                "phase66_peak_to_final_drop_after_60_mw": float(
                    np.max(phase66_values) - phase66_values[-1]
                ),
                "raw_final_abs_error": float(final["raw_abs_error"]),
                "phase66_final_abs_error": float(final["adapted_abs_error"]),
            }
        )
    return output


def _write_comparison_outputs(
    *,
    output_dir: Path,
    tables: Mapping[str, Any],
    summary: Mapping[str, Any],
    include_station_trajectories: bool,
) -> None:
    station_rows = [_rename_phase66(row) for row in tables["station_rows"]]
    endpoint_station_rows = [
        _rename_phase66(row) for row in tables["endpoint_station_rows"]
    ]
    event_rows = [_rename_phase66(row) for row in tables["event_rows"]]
    horizon_rows = [_rename_phase66(row) for row in tables["horizon_rows"]]
    convergence_rows = [
        _rename_phase66(row) for row in build_convergence_rows(tables["event_rows"])
    ]
    trajectory_rows = _trajectory_diagnostics(tables["event_rows"])
    station_fields = (
        "event",
        "station",
        "observation_horizon_sec",
        "release_time_sec",
        "mw_catalog",
        "raw_mw_pred",
        "phase66_mw_pred",
        "raw_error",
        "phase66_error",
        "raw_delta_mw",
        "phase66_delta_mw",
    )
    if include_station_trajectories:
        _write_csv(
            output_dir / "station_predictions.csv",
            station_rows,
            fieldnames=station_fields,
        )
    _write_csv(
        output_dir / "endpoint_station_predictions.csv",
        endpoint_station_rows,
        fieldnames=station_fields,
    )
    _write_csv(
        output_dir / "event_predictions.csv",
        event_rows,
        fieldnames=(
            "event",
            "observation_horizon_sec",
            "release_time_sec",
            "mw_catalog",
            "raw_mw_pred_median",
            "phase66_mw_pred_median",
            "raw_error",
            "phase66_error",
            "raw_abs_error",
            "phase66_abs_error",
            "raw_delta_mw",
            "phase66_delta_mw",
            "station_count",
        ),
    )
    _write_csv(
        output_dir / "horizon_metrics.csv",
        horizon_rows,
        fieldnames=tuple(horizon_rows[0]),
    )
    _write_csv(
        output_dir / "event_convergence.csv",
        convergence_rows,
        fieldnames=tuple(convergence_rows[0]),
    )
    _write_csv(
        output_dir / "trajectory_diagnostics.csv",
        trajectory_rows,
        fieldnames=tuple(trajectory_rows[0]),
    )
    _write_json(output_dir / "summary.json", dict(summary))


def evaluate_internal(
    *,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    raw_rates_path: Path = DEFAULT_INTERNAL_RAW_RATES,
) -> dict[str, Any]:
    _validate_new_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = validate_phase66_artifacts()
    data_config = load_frozen_config()
    samples, dataset_indices, split_manifest = _load_internal_samples(
        data_config,
        split_name="test",
    )
    if len(samples) != EXPECTED_TEST_COUNT:
        raise ValueError("Phase66 internal test count changed")
    raw_rates, raw_sha = _load_frozen_rates(
        raw_rates_path,
        expected_sha256=EXPECTED_INTERNAL_RAW_SHA256,
        expected_station_count=EXPECTED_TEST_COUNT,
    )
    available_mask = np.all(np.isfinite(raw_rates), axis=2)
    if not bool(np.all(available_mask)):
        raise ValueError("internal frozen raw cube must be complete")
    model = load_phase66_model(device=device)
    source_distance = np.asarray(
        [float(sample["source_distance_m"]) for sample in samples],
        dtype=np.float32,
    )
    source_dt = np.asarray(
        [float(sample["stf_dt_sec"]) for sample in samples],
        dtype=np.float32,
    )
    phase66_rates, transition = apply_phase66_to_rates(
        model=model,
        raw_rates=raw_rates,
        available_mask=available_mask,
        source_distance_m=source_distance,
        source_dt_sec=source_dt,
        beta_m_per_s=float(data_config["physics"]["beta"]),
        output_path=output_dir / "phase66_rates.npy",
        batch_size=batch_size,
    )
    events = [str(sample["event"]) for sample in samples]
    stations = [str(sample["station"]) for sample in samples]
    catalogs = np.asarray(
        [float(sample["magnitude_catalog"]) for sample in samples],
        dtype=np.float32,
    )
    tables = build_comparison_tables(
        raw_rates=raw_rates,
        adapted_rates=phase66_rates,
        available_mask=available_mask,
        events=events,
        stations=stations,
        catalogs=catalogs,
        source_dt_sec=source_dt,
        include_station_trajectories=False,
    )
    raw_metrics = _streaming_metrics(
        raw_rates,
        available_mask=available_mask,
        events=events,
        catalogs=catalogs,
        source_distance_m=source_distance,
        source_dt_sec=source_dt,
        beta_m_per_s=float(data_config["physics"]["beta"]),
    )
    phase66_metrics = _streaming_metrics(
        phase66_rates,
        available_mask=available_mask,
        events=events,
        catalogs=catalogs,
        source_distance_m=source_distance,
        source_dt_sec=source_dt,
        beta_m_per_s=float(data_config["physics"]["beta"]),
    )
    summary = {
        "status": "complete",
        "candidate": "Phase66 seed17 epoch26 frozen closest checkpoint",
        "evaluation_role": "opened_internal_test_once_user_override",
        "split": "test",
        "split_semantics": "within_event_station",
        "station_count": len(samples),
        "event_count": len(set(events)),
        "dataset_indices": dataset_indices,
        "split_assignment_sha256": split_manifest["assignment_sha256"],
        "processing_delay_sec": PROCESSING_DELAY_SEC,
        "transition_diagnostics": transition,
        "raw_metrics": raw_metrics,
        "phase66_metrics": phase66_metrics,
        "formal_validation_gate_passed": False,
        "validation_event_mae_miss_mw": 0.008906,
        "internal_test_iterated": True,
        "external_data_loaded": False,
        "grouped_test_loaded": False,
        "artifact_sha256": {
            "phase39_checkpoint": EXPECTED_PHASE39_SHA256,
            "phase66_checkpoint": artifacts["checkpoint_sha256"],
            "source_raw_rates": raw_sha,
            "phase66_rates": sha256_file(output_dir / "phase66_rates.npy"),
        },
        "provenance": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "device": str(device),
            "batch_size": batch_size,
        },
    }
    if split_manifest["assignment_sha256"] != EXPECTED_SPLIT_ASSIGNMENT_SHA256:
        raise ValueError("Phase66 internal split assignment changed")
    _write_comparison_outputs(
        output_dir=output_dir,
        tables=tables,
        summary=summary,
        include_station_trajectories=False,
    )
    return summary


def evaluate_external(
    *,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    raw_rates_path: Path = DEFAULT_EXTERNAL_RAW_RATES,
) -> dict[str, Any]:
    _validate_new_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = validate_phase66_artifacts()
    external_artifact_hashes = validate_frozen_artifacts()
    external_input_hashes = hash_external_inputs(EXTERNAL_EVENT_ROOT)
    data_config = load_phase39_config()
    labels = load_label_contract(LABELS_PATH)
    endpoint_reference = load_endpoint_reference()
    records = build_raw_streaming_records(
        config=data_config,
        event_root=EXTERNAL_EVENT_ROOT,
        labels_by_dir=labels,
        expected_station_keys=set(endpoint_reference["station_predictions"]),
    )
    if len(records) != EXPECTED_STATION_COUNT:
        raise ValueError("Phase66 external station count changed")
    if len(set(record.event for record in records)) != EXPECTED_EVENT_COUNT:
        raise ValueError("Phase66 external event count changed")
    raw_rates, raw_sha = _load_frozen_rates(
        raw_rates_path,
        expected_sha256=EXPECTED_EXTERNAL_RAW_SHA256,
        expected_station_count=EXPECTED_STATION_COUNT,
    )
    available_mask = np.all(np.isfinite(raw_rates), axis=2)
    if not bool(np.all(available_mask)):
        raise ValueError("external frozen raw cube must be complete")
    model = load_phase66_model(device=device)
    source_distance = np.asarray(
        [record.source_distance_m for record in records],
        dtype=np.float32,
    )
    source_dt = np.full(len(records), SOURCE_DT_SEC, dtype=np.float32)
    phase66_rates, transition = apply_phase66_to_rates(
        model=model,
        raw_rates=raw_rates,
        available_mask=available_mask,
        source_distance_m=source_distance,
        source_dt_sec=source_dt,
        beta_m_per_s=float(data_config["physics"]["beta"]),
        output_path=output_dir / "phase66_rates.npy",
        batch_size=batch_size,
    )
    events = [record.event for record in records]
    stations = [record.station for record in records]
    catalogs = np.asarray(
        [record.magnitude_catalog for record in records],
        dtype=np.float32,
    )
    tables = build_comparison_tables(
        raw_rates=raw_rates,
        adapted_rates=phase66_rates,
        available_mask=available_mask,
        events=events,
        stations=stations,
        catalogs=catalogs,
        source_dt_sec=source_dt,
        include_station_trajectories=True,
    )
    endpoint_gate = _endpoint_external_gate(
        tables["event_rows"],
        tables["endpoint_station_rows"],
        endpoint_reference,
    )
    if (
        endpoint_gate["max_station_prediction_abs_diff_mw"]
        > ENDPOINT_PREDICTION_TOLERANCE_MW
    ):
        raise ValueError("Phase39 external endpoint reproduction failed")
    raw_metrics = _streaming_metrics(
        raw_rates,
        available_mask=available_mask,
        events=events,
        catalogs=catalogs,
        source_distance_m=source_distance,
        source_dt_sec=source_dt,
        beta_m_per_s=float(data_config["physics"]["beta"]),
    )
    phase66_metrics = _streaming_metrics(
        phase66_rates,
        available_mask=available_mask,
        events=events,
        catalogs=catalogs,
        source_distance_m=source_distance,
        source_dt_sec=source_dt,
        beta_m_per_s=float(data_config["physics"]["beta"]),
    )
    summary = {
        "status": "complete",
        "candidate": "Phase66 seed17 epoch26 frozen closest checkpoint",
        "evaluation_role": "development_validation_reporting_only",
        "event_count": len(set(events)),
        "station_count": len(records),
        "processing_delay_sec": PROCESSING_DELAY_SEC,
        "endpoint_raw_reproduction_gate": endpoint_gate,
        "transition_diagnostics": transition,
        "raw_metrics": raw_metrics,
        "phase66_metrics": phase66_metrics,
        "formal_validation_gate_passed": False,
        "internal_test_iterated": False,
        "external_data_loaded": True,
        "grouped_test_loaded": False,
        "artifact_sha256": {
            "phase39_checkpoint": EXPECTED_PHASE39_SHA256,
            "phase66_checkpoint": artifacts["checkpoint_sha256"],
            "external_frozen_artifacts": external_artifact_hashes,
            "external_inputs": external_input_hashes,
            "source_raw_rates": raw_sha,
            "phase66_rates": sha256_file(output_dir / "phase66_rates.npy"),
        },
        "provenance": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "device": str(device),
            "batch_size": batch_size,
        },
    }
    _write_comparison_outputs(
        output_dir=output_dir,
        tables=tables,
        summary=summary,
        include_station_trajectories=True,
    )
    return summary


def evaluate_smoke(
    *,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    _validate_new_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(DEFAULT_CACHE_ROOT)
    validation = np.flatnonzero(cache.arrays["split_code"] == 1)[:4]
    raw = np.asarray(cache.raw_rates[validation]).transpose(1, 0, 2)
    mask = np.ones(raw.shape[:2], dtype=bool)
    model = load_phase66_model(device=device)
    rates, transition = apply_phase66_to_rates(
        model=model,
        raw_rates=raw,
        available_mask=mask,
        source_distance_m=np.asarray(
            cache.arrays["source_distance_m"][validation], dtype=np.float32
        ),
        source_dt_sec=np.asarray(
            cache.arrays["source_dt_sec"][validation], dtype=np.float32
        ),
        beta_m_per_s=float(phase50_config()["physics"]["beta"]),
        output_path=output_dir / "phase66_rates.npy",
        batch_size=min(batch_size, len(validation)),
    )
    summary = {
        "status": "smoke_complete",
        "evaluation_role": "validation_smoke_no_hidden_data",
        "station_count": len(validation),
        "transition_diagnostics": transition,
        "all_finite": bool(np.all(np.isfinite(rates))),
        "all_nonnegative": bool(np.all(np.asarray(rates) >= 0.0)),
        "internal_test_iterated": False,
        "external_data_loaded": False,
        "grouped_test_loaded": False,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def run_all(
    *,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    internal_raw_rates: Path,
    external_raw_rates: Path,
) -> dict[str, Any]:
    _validate_new_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training = extract_training_objective(output_dir / "training")
    internal = evaluate_internal(
        output_dir=output_dir / "internal",
        device=device,
        batch_size=batch_size,
        raw_rates_path=internal_raw_rates,
    )
    external = evaluate_external(
        output_dir=output_dir / "external",
        device=device,
        batch_size=batch_size,
        raw_rates_path=external_raw_rates,
    )
    summary = {
        "status": "complete",
        "evaluation_role": "frozen_internal_and_development_reporting",
        "candidate": "Phase66 seed17 epoch26 frozen closest checkpoint",
        "training_status": training["status"],
        "internal_status": internal["status"],
        "external_status": external["status"],
        "internal_test_iterated": True,
        "external_data_loaded": True,
        "grouped_test_loaded": False,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen Phase66 seed17 epoch26 model-internal stateful "
            "checkpoint on training traces, internal test, and eight events."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("training", "smoke", "internal", "external", "all"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--internal-raw-rates",
        type=Path,
        default=DEFAULT_INTERNAL_RAW_RATES,
    )
    parser.add_argument(
        "--external-raw-rates",
        type=Path,
        default=DEFAULT_EXTERNAL_RAW_RATES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    output_dir = args.output_dir.resolve()
    if args.stage == "training":
        summary = extract_training_objective(output_dir)
    else:
        device = _resolve_device(args.device)
        if args.stage == "smoke":
            summary = evaluate_smoke(
                output_dir=output_dir,
                device=device,
                batch_size=args.batch_size,
            )
        elif args.stage == "internal":
            summary = evaluate_internal(
                output_dir=output_dir,
                device=device,
                batch_size=args.batch_size,
                raw_rates_path=args.internal_raw_rates.resolve(),
            )
        elif args.stage == "external":
            summary = evaluate_external(
                output_dir=output_dir,
                device=device,
                batch_size=args.batch_size,
                raw_rates_path=args.external_raw_rates.resolve(),
            )
        else:
            summary = run_all(
                output_dir=output_dir,
                device=device,
                batch_size=args.batch_size,
                internal_raw_rates=args.internal_raw_rates.resolve(),
                external_raw_rates=args.external_raw_rates.resolve(),
            )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "status": summary["status"],
                "evaluation_role": summary["evaluation_role"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
