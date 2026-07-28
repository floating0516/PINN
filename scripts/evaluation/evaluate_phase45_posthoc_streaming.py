#!/usr/bin/env python3
"""Post-hoc train/test/external evaluation for the frozen Phase45 adapter."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

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
    load_model,
    load_phase39_config,
    validate_frozen_artifacts,
)
from scripts.evaluation.evaluate_phase39_streaming_replay import (  # noqa: E402
    RawStreamingStation,
    _metadata_for_records,
    build_raw_streaming_records,
    decode_stf_rate,
)
from scripts.experiments.run_phase40_streaming_consistency import (  # noqa: E402
    EXPECTED_SPLIT_ASSIGNMENT_SHA256,
    PHASE39_CHECKPOINT,
    load_frozen_config,
    validate_source_artifacts,
)
from scripts.experiments.run_phase43_streaming_adapter import (  # noqa: E402
    HORIZONS,
    LATE_HORIZONS,
    PROCESSING_DELAY_SEC,
    _late_metrics_from_rates,
    _metadata_tensor,
    _preprocess_prefix,
    _raw_record_map,
)
from src.data.loaders_v2 import get_data_loaders_v2  # noqa: E402
from src.data.waveform import preprocess_waveform  # noqa: E402
from src.models.model import PINNModel  # noqa: E402
from src.models.streaming_stf_adapter import StreamingSTFAdapter  # noqa: E402
from src.utils.config_v2 import stf_m_ref_from_config  # noqa: E402
from src.utils.device import configure_runtime  # noqa: E402
from src.utils.provenance import sha256_file  # noqa: E402


PROJECT_HOME = PROJECT_ROOT.parent.parent
PHASE45_ROOT = (
    PROJECT_HOME
    / "runs"
    / "phase45-streaming-adapter-20260728T122754Z-7bc63eb"
)
PHASE45_SUMMARY = PHASE45_ROOT / "summary.json"
PHASE45_EPOCH_METRICS = PHASE45_ROOT / "seed_42" / "epoch_metrics.csv"
ADAPTER_CHECKPOINT = PHASE45_ROOT / "seed_42" / "best_adapter.pth"
EXPECTED_ADAPTER_SHA256 = (
    "6f623cfebf74f155996cb7bc01604e561b7577d3a5e2f22f060ee540e0164e8b"
)
EXPECTED_PHASE39_SHA256 = (
    "73500f365a58b248204d02333716f31674435927e9fc1c7d55a1453786b406f7"
)
EXPECTED_TEST_COUNT = 385
EXPECTED_SELECTED_EPOCH = 27
EXPECTED_SELECTED_SCORE = 1.0171009413740808
SOURCE_STEPS = 200
SOURCE_DT_SEC = 1.0
TARGET_ABS_ERROR_MW = 0.15
ENDPOINT_INPUT_TOLERANCE_M = 1.0e-12
ENDPOINT_PREDICTION_TOLERANCE_MW = 5.0e-6
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _validate_new_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output directory must be new or empty: {path}")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(requested)


def validate_phase45_artifacts() -> dict[str, Any]:
    for path in (PHASE45_SUMMARY, PHASE45_EPOCH_METRICS, ADAPTER_CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f"missing Phase45 artifact: {path}")
    source_hashes = validate_source_artifacts()
    if source_hashes["checkpoint"] != EXPECTED_PHASE39_SHA256:
        raise ValueError("Phase39 checkpoint hash changed")
    adapter_sha = sha256_file(ADAPTER_CHECKPOINT)
    if adapter_sha != EXPECTED_ADAPTER_SHA256:
        raise ValueError("Phase45 adapter checkpoint hash changed")

    summary = _read_json(PHASE45_SUMMARY)
    if summary.get("status") != "validation_gate_failed":
        raise ValueError("Phase45 status changed")
    if summary.get("passed") is not False:
        raise ValueError("Phase45 unexpectedly passed")
    if summary.get("selected_seed") is not None:
        raise ValueError("Phase45 formal selection must remain empty")
    seed42 = next(
        item for item in summary["seed_summaries"] if int(item["seed"]) == 42
    )
    if int(seed42["selected_epoch"]) != EXPECTED_SELECTED_EPOCH:
        raise ValueError("Phase45 audit checkpoint epoch changed")
    score = float(seed42["selected_gate"]["selection_score"])
    if not math.isclose(score, EXPECTED_SELECTED_SCORE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Phase45 audit checkpoint score changed")
    if seed42["best_adapter"]["sha256"] != EXPECTED_ADAPTER_SHA256:
        raise ValueError("Phase45 summary points to a different adapter")
    if seed42.get("phase39_weights_trained") is not False:
        raise ValueError("Phase45 summary says Phase39 weights were trained")
    return {
        "phase39": source_hashes,
        "phase45_summary_sha256": sha256_file(PHASE45_SUMMARY),
        "phase45_epoch_metrics_sha256": sha256_file(PHASE45_EPOCH_METRICS),
        "adapter_checkpoint_sha256": adapter_sha,
        "phase45_summary": summary,
    }


def load_adapter(
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> StreamingSTFAdapter:
    adapter = StreamingSTFAdapter(
        stf_m_ref=stf_m_ref_from_config(dict(config))
    ).to(device)
    state = torch.load(ADAPTER_CHECKPOINT, map_location=device, weights_only=True)
    adapter.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in adapter.parameters()) != 489:
        raise ValueError("Phase45 adapter parameter count changed")
    adapter.eval()
    return adapter


def extract_training_objective(output_dir: Path) -> dict[str, Any]:
    _validate_new_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = validate_phase45_artifacts()
    summary = artifacts["phase45_summary"]
    weights = {
        name: float(value)
        for name, value in summary["protocol"]["loss_weights"].items()
    }
    normalizers = {
        name: float(value)
        for name, value in summary["protocol"]["loss_normalizers"].items()
    }
    raw_rows = _read_csv(PHASE45_EPOCH_METRICS)
    rows: list[dict[str, Any]] = []
    component_names = tuple(weights)
    for raw in raw_rows:
        row: dict[str, Any] = {
            "epoch": int(raw["epoch"]),
            "train_total_normalized_loss": float(
                raw["train_total_normalized_loss"]
            ),
            "validation_selection_score": float(raw["selection_score"]),
            "validation_endpoint_event_mae": float(raw["endpoint_event_mae"]),
            "validation_endpoint_station_mae": float(raw["endpoint_station_mae"]),
            "validation_late_event_step_p95_mw": float(
                raw["late_event_abs_step_p95_mw"]
            ),
            "validation_late_confirmed_history_p95": float(
                raw["late_confirmed_cumulative_log10_l1_p95"]
            ),
            "selected_checkpoint": int(raw["epoch"]) == EXPECTED_SELECTED_EPOCH,
        }
        contribution_sum = 0.0
        for name in component_names:
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
            raise ValueError(f"training objective reconstruction failed at epoch {row['epoch']}")
        rows.append(row)

    selected = next(row for row in rows if row["selected_checkpoint"])
    minimum_total = min(rows, key=lambda row: row["train_total_normalized_loss"])
    minimum_score = min(rows, key=lambda row: row["validation_selection_score"])
    payload = {
        "status": "complete",
        "evaluation_role": "saved_training_trace",
        "candidate": "Phase45 seed42 epoch27, audit-only",
        "epoch_count": len(rows),
        "loss_weights": weights,
        "loss_normalizers": normalizers,
        "selected_epoch": EXPECTED_SELECTED_EPOCH,
        "selected_epoch_metrics": selected,
        "minimum_training_total": {
            "epoch": int(minimum_total["epoch"]),
            "value": float(minimum_total["train_total_normalized_loss"]),
        },
        "minimum_validation_selection_score": {
            "epoch": int(minimum_score["epoch"]),
            "value": float(minimum_score["validation_selection_score"]),
        },
        "artifact_sha256": {
            "phase45_summary": artifacts["phase45_summary_sha256"],
            "phase45_epoch_metrics": artifacts["phase45_epoch_metrics_sha256"],
            "adapter_checkpoint": artifacts["adapter_checkpoint_sha256"],
        },
    }
    fieldnames = tuple(rows[0])
    _write_csv(output_dir / "training_loss_by_epoch.csv", rows, fieldnames=fieldnames)
    _write_json(output_dir / "summary.json", payload)
    return payload


def _load_internal_samples(
    config: dict[str, Any],
    *,
    split_name: str,
    limit: int | None = None,
) -> tuple[list[Mapping[str, Any]], list[int], dict[str, Any]]:
    train_loader, validation_loader, test_loader, split_manifest = get_data_loaders_v2(
        config
    )
    if split_manifest["assignment_sha256"] != EXPECTED_SPLIT_ASSIGNMENT_SHA256:
        raise ValueError("Phase39 split assignment changed")
    loader_by_name = {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
    }
    loader = loader_by_name[split_name]
    indices = [int(value) for value in loader.dataset.indices]
    full_dataset = loader.dataset.dataset
    if split_name == "test" and len(indices) != EXPECTED_TEST_COUNT:
        raise ValueError("Phase39 internal test count changed")
    if limit is not None:
        indices = indices[:limit]
    samples = [full_dataset.samples[index] for index in indices]
    del train_loader, validation_loader, test_loader
    return samples, indices, split_manifest


def _load_phase39_internal_model(
    config: dict[str, Any],
    *,
    device: torch.device,
) -> PINNModel:
    configure_runtime(42, device)
    model = PINNModel(config).to(device)
    state = torch.load(PHASE39_CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def generate_internal_raw_rates(
    *,
    config: dict[str, Any],
    samples: Sequence[Mapping[str, Any]],
    model: PINNModel,
    output_path: Path,
    batch_size: int,
) -> tuple[np.memmap, dict[str, Any]]:
    parameter = next(model.parameters())
    device = parameter.device
    raw_by_key = _raw_record_map(config, selected_samples=samples)
    records = [
        raw_by_key[(str(sample["event"]), str(sample["station"]))]
        for sample in samples
    ]
    metadata = _metadata_tensor(
        samples,
        config,
        device=device,
        dtype=parameter.dtype,
    )
    raw_rates = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(HORIZONS), len(samples), SOURCE_STEPS),
    )
    endpoint_max_diff = 0.0
    endpoint_mask_mismatches = 0
    endpoint_baseline_mismatches = 0
    stf_m_ref = stf_m_ref_from_config(config)
    with torch.no_grad():
        for horizon_index, horizon in enumerate(HORIZONS):
            prefixes = np.empty((len(samples), horizon), dtype=np.float32)
            for index, (record, sample) in enumerate(zip(records, samples, strict=True)):
                values, mask, baseline_source = _preprocess_prefix(
                    record,
                    horizon_sec=horizon,
                )
                prefixes[index] = values
                if horizon == SOURCE_STEPS:
                    endpoint_max_diff = max(
                        endpoint_max_diff,
                        float(
                            np.max(
                                np.abs(
                                    values
                                    - np.asarray(sample["radial"], dtype=np.float32)
                                )
                            )
                        ),
                    )
                    endpoint_mask_mismatches += int(
                        not np.array_equal(
                            mask,
                            np.asarray(sample["waveform_valid_mask"], dtype=bool),
                        )
                    )
                    endpoint_baseline_mismatches += int(
                        baseline_source != str(sample["baseline_source"])
                    )
            for start in range(0, len(samples), batch_size):
                stop = min(start + batch_size, len(samples))
                model_input = torch.as_tensor(
                    prefixes[start:stop],
                    device=device,
                    dtype=parameter.dtype,
                ).unsqueeze(1)
                encoded = model(model_input, meta=metadata[start:stop])
                rates = decode_stf_rate(encoded, stf_m_ref=stf_m_ref)
                raw_rates[horizon_index, start:stop] = (
                    rates.detach().cpu().numpy().astype(np.float32, copy=False)
                )
            raw_rates.flush()
            if horizon == HORIZONS[0] or horizon % 20 == 0 or horizon == 200:
                print(
                    f"internal raw horizon={horizon}/200 records={len(samples)}",
                    flush=True,
                )
    if endpoint_max_diff != 0.0:
        raise ValueError(f"internal h=200 input changed by {endpoint_max_diff:.9g} m")
    if endpoint_mask_mismatches or endpoint_baseline_mismatches:
        raise ValueError("internal h=200 preprocessing metadata changed")
    return raw_rates, {
        "station_count": len(samples),
        "max_abs_radial_input_diff_m": endpoint_max_diff,
        "valid_mask_mismatch_count": endpoint_mask_mismatches,
        "baseline_source_mismatch_count": endpoint_baseline_mismatches,
    }


def _external_prefix(
    record: RawStreamingStation,
    *,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    issue_time_sec = float(horizon + PROCESSING_DELAY_SEC)
    raw_time = np.asarray(record.raw_time_sec, dtype=np.float64)
    raw_values = np.asarray(record.raw_radial_m, dtype=np.float64)
    available = (
        np.isfinite(raw_time)
        & np.isfinite(raw_values)
        & (raw_time <= issue_time_sec)
    )
    if int(np.count_nonzero(available)) < 2:
        raise ValueError("waveform has fewer than two finite samples at issue time")
    processed = preprocess_waveform(
        raw_time[available],
        raw_values[available],
        units="m",
        p_arrival_sec=record.p_arrival_sec,
        config=record.waveform_config,
    )
    values = np.asarray(processed.values_m[:horizon], dtype=np.float32)
    mask = np.asarray(processed.valid_mask[:horizon], dtype=bool)
    if values.shape != (horizon,) or mask.shape != values.shape:
        raise ValueError("external streaming prefix shape changed")
    if np.any(values[~mask] != 0.0):
        raise ValueError("external invalid prefix slots must remain zero")
    return values, mask, str(processed.baseline_source)


def generate_external_raw_rates(
    *,
    config: dict[str, Any],
    records: Sequence[RawStreamingStation],
    model: PINNModel,
    output_path: Path,
    batch_size: int,
) -> tuple[np.memmap, np.ndarray, dict[str, Any]]:
    parameter = next(model.parameters())
    device = parameter.device
    metadata = _metadata_for_records(
        records,
        config,
        device=device,
        dtype=parameter.dtype,
    )
    raw_rates = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(HORIZONS), len(records), SOURCE_STEPS),
    )
    raw_rates[:] = np.nan
    available_mask = np.zeros((len(HORIZONS), len(records)), dtype=bool)
    unavailable_reason_counts: dict[str, int] = defaultdict(int)
    endpoint_max_diff = 0.0
    endpoint_mask_mismatches = 0
    endpoint_baseline_mismatches = 0
    stf_m_ref = stf_m_ref_from_config(config)
    with torch.no_grad():
        for horizon_index, horizon in enumerate(HORIZONS):
            prefixes: dict[int, tuple[np.ndarray, np.ndarray, str]] = {}
            for station_index, record in enumerate(records):
                try:
                    prefixes[station_index] = _external_prefix(
                        record,
                        horizon=horizon,
                    )
                except ValueError as exc:
                    reason = (
                        "insufficient_baseline"
                        if "baseline" in str(exc).lower()
                        else "insufficient_raw_samples"
                        if "fewer than two" in str(exc).lower()
                        else "invalid_waveform"
                    )
                    unavailable_reason_counts[reason] += 1
            available_indices = sorted(prefixes)
            for start in range(0, len(available_indices), batch_size):
                batch_indices = available_indices[start : start + batch_size]
                model_input = torch.as_tensor(
                    np.stack([prefixes[index][0] for index in batch_indices]),
                    device=device,
                    dtype=parameter.dtype,
                ).unsqueeze(1)
                selected = torch.as_tensor(
                    batch_indices,
                    device=device,
                    dtype=torch.long,
                )
                encoded = model(model_input, meta=metadata.index_select(0, selected))
                rates = decode_stf_rate(encoded, stf_m_ref=stf_m_ref)
                values = rates.detach().cpu().numpy().astype(np.float32, copy=False)
                for local_index, station_index in enumerate(batch_indices):
                    raw_rates[horizon_index, station_index] = values[local_index]
                    available_mask[horizon_index, station_index] = True
                    if horizon == SOURCE_STEPS:
                        prefix, mask, baseline_source = prefixes[station_index]
                        record = records[station_index]
                        endpoint_max_diff = max(
                            endpoint_max_diff,
                            float(np.max(np.abs(prefix - record.endpoint_radial_m))),
                        )
                        endpoint_mask_mismatches += int(
                            not np.array_equal(mask, record.endpoint_valid_mask)
                        )
                        endpoint_baseline_mismatches += int(
                            baseline_source != record.endpoint_baseline_source
                        )
            raw_rates.flush()
            if horizon == HORIZONS[0] or horizon % 20 == 0 or horizon == 200:
                print(
                    f"external raw horizon={horizon}/200 "
                    f"available={len(available_indices)}/{len(records)}",
                    flush=True,
                )
    if not bool(np.all(available_mask[-1])):
        raise ValueError("external h=200 did not produce every station")
    if endpoint_max_diff > ENDPOINT_INPUT_TOLERANCE_M:
        raise ValueError(
            f"external h=200 input changed by {endpoint_max_diff:.9g} m"
        )
    if endpoint_mask_mismatches or endpoint_baseline_mismatches:
        raise ValueError("external h=200 preprocessing metadata changed")
    return raw_rates, available_mask, {
        "station_count": len(records),
        "max_abs_radial_input_diff_m": endpoint_max_diff,
        "valid_mask_mismatch_count": endpoint_mask_mismatches,
        "baseline_source_mismatch_count": endpoint_baseline_mismatches,
        "unavailable_reason_counts": dict(sorted(unavailable_reason_counts.items())),
    }


def _contiguous_segments(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = start
    for value in indices[1:]:
        current = int(value)
        if current != previous + 1:
            segments.append((start, previous + 1))
            start = current
        previous = current
    segments.append((start, previous + 1))
    return segments


def apply_adapter_to_rates(
    *,
    adapter: StreamingSTFAdapter,
    raw_rates: np.ndarray,
    available_mask: np.ndarray,
    source_distance_m: np.ndarray,
    source_dt_sec: np.ndarray,
    beta_m_per_s: float,
    output_path: Path,
    batch_size: int,
) -> np.memmap:
    cube = np.asarray(raw_rates)
    mask = np.asarray(available_mask, dtype=bool)
    if cube.shape[:2] != mask.shape or cube.shape[2] != SOURCE_STEPS:
        raise ValueError("raw rates and availability shape mismatch")
    if cube.shape[0] != len(HORIZONS):
        raise ValueError("adapter rate cube horizon count changed")
    station_count = cube.shape[1]
    distances = np.asarray(source_distance_m, dtype=np.float32)
    source_dt = np.asarray(source_dt_sec, dtype=np.float32)
    if distances.shape != (station_count,) or source_dt.shape != (station_count,):
        raise ValueError("adapter metadata station count changed")

    segment_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for station_index in range(station_count):
        available_indices = np.flatnonzero(mask[:, station_index])
        for segment in _contiguous_segments(available_indices):
            segment_groups[segment].append(station_index)

    parameter = next(adapter.parameters())
    device = parameter.device
    adapted = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=cube.shape,
    )
    adapted[:] = np.nan
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
                states, _ = adapter(
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
                    adapted[start_index:stop_index, station_index] = state_values[
                        local_index
                    ]
            adapted.flush()
    finite_mask = np.all(np.isfinite(adapted), axis=2)
    if not np.array_equal(finite_mask, mask):
        raise ValueError("adapter availability differs from raw availability")
    if np.any(np.asarray(adapted)[mask] < 0.0):
        raise ValueError("adapter produced a negative STF rate")
    return adapted


def _error_summary(errors: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
        }
    return {
        "count": int(values.size),
        "mae": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "bias": float(np.mean(values)),
    }


def _mw_cube(rates: np.ndarray, source_dt_sec: np.ndarray) -> np.ndarray:
    cube = np.asarray(rates, dtype=np.float64)
    dt = np.asarray(source_dt_sec, dtype=np.float64).reshape(1, -1, 1)
    moment = np.sum(np.maximum(cube, 0.0) * dt, axis=2)
    return (2.0 / 3.0) * (np.log10(np.maximum(moment, 1.0e10)) - 9.1)


def build_comparison_tables(
    *,
    raw_rates: np.ndarray,
    adapted_rates: np.ndarray,
    available_mask: np.ndarray,
    events: Sequence[str],
    stations: Sequence[str],
    catalogs: np.ndarray,
    source_dt_sec: np.ndarray,
    include_station_trajectories: bool,
) -> dict[str, Any]:
    mask = np.asarray(available_mask, dtype=bool)
    raw_mw = _mw_cube(raw_rates, source_dt_sec)
    adapted_mw = _mw_cube(adapted_rates, source_dt_sec)
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
    previous_event_raw: dict[str, float] = {}
    previous_event_adapted: dict[str, float] = {}

    for horizon_index, horizon in enumerate(HORIZONS):
        available = np.flatnonzero(mask[horizon_index])
        raw_station_errors = raw_mw[horizon_index, available] - catalogs[available]
        adapted_station_errors = (
            adapted_mw[horizon_index, available] - catalogs[available]
        )
        raw_station_steps: list[float] = []
        adapted_station_steps: list[float] = []
        for index in available:
            previous_available = horizon_index > 0 and mask[horizon_index - 1, index]
            raw_delta = (
                float(raw_mw[horizon_index, index] - raw_mw[horizon_index - 1, index])
                if previous_available
                else None
            )
            adapted_delta = (
                float(
                    adapted_mw[horizon_index, index]
                    - adapted_mw[horizon_index - 1, index]
                )
                if previous_available
                else None
            )
            if raw_delta is not None:
                raw_station_steps.append(raw_delta)
                adapted_station_steps.append(float(adapted_delta))
            row = {
                "event": str(events[index]),
                "station": str(stations[index]),
                "observation_horizon_sec": int(horizon),
                "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                "mw_catalog": float(catalogs[index]),
                "raw_mw_pred": float(raw_mw[horizon_index, index]),
                "adapted_mw_pred": float(adapted_mw[horizon_index, index]),
                "raw_error": float(raw_station_errors[np.where(available == index)[0][0]]),
                "adapted_error": float(
                    adapted_station_errors[np.where(available == index)[0][0]]
                ),
                "raw_delta_mw": raw_delta,
                "adapted_delta_mw": adapted_delta,
            }
            if include_station_trajectories:
                station_rows.append(row)
            if horizon == 200:
                endpoint_station_rows.append(row)

        current_event_rows: list[dict[str, Any]] = []
        raw_event_steps: list[float] = []
        adapted_event_steps: list[float] = []
        for event in event_names:
            indices = event_indices[event]
            current = indices[mask[horizon_index, indices]]
            if current.size == 0:
                continue
            event_catalogs = catalogs[current]
            if not np.allclose(event_catalogs, event_catalogs[0], rtol=0.0, atol=1e-7):
                raise ValueError(f"catalog magnitude differs within event {event}")
            raw_prediction = float(np.median(raw_mw[horizon_index, current]))
            adapted_prediction = float(np.median(adapted_mw[horizon_index, current]))
            previous_raw = previous_event_raw.get(event)
            previous_adapted = previous_event_adapted.get(event)
            raw_delta = None if previous_raw is None else raw_prediction - previous_raw
            adapted_delta = (
                None
                if previous_adapted is None
                else adapted_prediction - previous_adapted
            )
            if raw_delta is not None:
                raw_event_steps.append(raw_delta)
                adapted_event_steps.append(float(adapted_delta))
            current_event_rows.append(
                {
                    "event": event,
                    "observation_horizon_sec": int(horizon),
                    "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                    "mw_catalog": float(event_catalogs[0]),
                    "raw_mw_pred_median": raw_prediction,
                    "adapted_mw_pred_median": adapted_prediction,
                    "raw_error": raw_prediction - float(event_catalogs[0]),
                    "adapted_error": adapted_prediction - float(event_catalogs[0]),
                    "raw_abs_error": abs(raw_prediction - float(event_catalogs[0])),
                    "adapted_abs_error": abs(
                        adapted_prediction - float(event_catalogs[0])
                    ),
                    "raw_delta_mw": raw_delta,
                    "adapted_delta_mw": adapted_delta,
                    "station_count": int(current.size),
                }
            )
            previous_event_raw[event] = raw_prediction
            previous_event_adapted[event] = adapted_prediction
        event_rows.extend(current_event_rows)
        raw_event_errors = [float(row["raw_error"]) for row in current_event_rows]
        adapted_event_errors = [
            float(row["adapted_error"]) for row in current_event_rows
        ]
        raw_station_summary = _error_summary(raw_station_errors)
        adapted_station_summary = _error_summary(adapted_station_errors)
        raw_event_summary = _error_summary(raw_event_errors)
        adapted_event_summary = _error_summary(adapted_event_errors)
        horizon_rows.append(
            {
                "observation_horizon_sec": int(horizon),
                "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                "available_station_count": int(available.size),
                "event_count": len(current_event_rows),
                "raw_station_mae": raw_station_summary["mae"],
                "adapted_station_mae": adapted_station_summary["mae"],
                "raw_station_rmse": raw_station_summary["rmse"],
                "adapted_station_rmse": adapted_station_summary["rmse"],
                "raw_station_bias": raw_station_summary["bias"],
                "adapted_station_bias": adapted_station_summary["bias"],
                "raw_event_mae": raw_event_summary["mae"],
                "adapted_event_mae": adapted_event_summary["mae"],
                "raw_event_rmse": raw_event_summary["rmse"],
                "adapted_event_rmse": adapted_event_summary["rmse"],
                "raw_event_bias": raw_event_summary["bias"],
                "adapted_event_bias": adapted_event_summary["bias"],
                "raw_event_abs_step_p95": (
                    float(np.percentile(np.abs(raw_event_steps), 95))
                    if raw_event_steps
                    else None
                ),
                "adapted_event_abs_step_p95": (
                    float(np.percentile(np.abs(adapted_event_steps), 95))
                    if adapted_event_steps
                    else None
                ),
                "raw_station_abs_step_p95": (
                    float(np.percentile(np.abs(raw_station_steps), 95))
                    if raw_station_steps
                    else None
                ),
                "adapted_station_abs_step_p95": (
                    float(np.percentile(np.abs(adapted_station_steps), 95))
                    if adapted_station_steps
                    else None
                ),
            }
        )
    return {
        "station_rows": station_rows,
        "endpoint_station_rows": endpoint_station_rows,
        "event_rows": event_rows,
        "horizon_rows": horizon_rows,
        "raw_mw": raw_mw,
        "adapted_mw": adapted_mw,
    }


def _suffix_stable_horizon(
    rows: Sequence[Mapping[str, Any]],
    *,
    error_key: str,
) -> int | None:
    ordered = sorted(rows, key=lambda row: int(row["observation_horizon_sec"]))
    for index, row in enumerate(ordered):
        if all(float(item[error_key]) <= TARGET_ABS_ERROR_MW for item in ordered[index:]):
            return int(row["observation_horizon_sec"])
    return None


def build_convergence_rows(event_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[str(row["event"])].append(row)
    output: list[dict[str, Any]] = []
    for event in sorted(grouped):
        rows = sorted(
            grouped[event],
            key=lambda row: int(row["observation_horizon_sec"]),
        )
        final = rows[-1]
        raw_stable = _suffix_stable_horizon(rows, error_key="raw_abs_error")
        adapted_stable = _suffix_stable_horizon(
            rows,
            error_key="adapted_abs_error",
        )
        output.append(
            {
                "event": event,
                "mw_catalog": float(final["mw_catalog"]),
                "raw_final_mw": float(final["raw_mw_pred_median"]),
                "adapted_final_mw": float(final["adapted_mw_pred_median"]),
                "raw_final_abs_error": float(final["raw_abs_error"]),
                "adapted_final_abs_error": float(final["adapted_abs_error"]),
                "raw_stable_observation_sec": raw_stable,
                "raw_stable_release_sec": (
                    None if raw_stable is None else raw_stable + PROCESSING_DELAY_SEC
                ),
                "adapted_stable_observation_sec": adapted_stable,
                "adapted_stable_release_sec": (
                    None
                    if adapted_stable is None
                    else adapted_stable + PROCESSING_DELAY_SEC
                ),
                "station_count_200s": int(final["station_count"]),
            }
        )
    return output


def _endpoint_external_gate(
    event_rows: Sequence[Mapping[str, Any]],
    endpoint_station_rows: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    actual_stations = {
        (str(row["event"]), str(row["station"])): float(row["raw_mw_pred"])
        for row in endpoint_station_rows
    }
    expected_stations = dict(reference["station_predictions"])
    if set(actual_stations) != set(expected_stations):
        raise ValueError("external endpoint station keys changed")
    station_diffs = {
        key: abs(actual_stations[key] - float(expected_stations[key]))
        for key in actual_stations
    }
    final_events = {
        str(row["event"]): float(row["raw_mw_pred_median"])
        for row in event_rows
        if int(row["observation_horizon_sec"]) == 200
    }
    expected_events = dict(reference["event_predictions"])
    if set(final_events) != set(expected_events):
        raise ValueError("external endpoint event keys changed")
    event_diffs = {
        event: abs(final_events[event] - float(expected_events[event]))
        for event in final_events
    }
    max_station = max(station_diffs.values(), default=0.0)
    max_event = max(event_diffs.values(), default=0.0)
    if max_station > ENDPOINT_PREDICTION_TOLERANCE_MW:
        raise ValueError("external raw station endpoint no longer reproduces Phase39")
    if max_event > ENDPOINT_PREDICTION_TOLERANCE_MW:
        raise ValueError("external raw event endpoint no longer reproduces Phase39")
    raw_event_mae = float(
        np.mean(
            [
                float(row["raw_abs_error"])
                for row in event_rows
                if int(row["observation_horizon_sec"]) == 200
            ]
        )
    )
    if abs(raw_event_mae - float(reference["event_mae"])) > ENDPOINT_PREDICTION_TOLERANCE_MW:
        raise ValueError("external raw endpoint Event MAE changed")
    return {
        "tolerance_mw": ENDPOINT_PREDICTION_TOLERANCE_MW,
        "max_station_prediction_abs_diff_mw": max_station,
        "max_event_median_abs_diff_mw": max_event,
        "raw_event_mae": raw_event_mae,
    }


def _late_metrics(
    rates: np.ndarray,
    *,
    events: Sequence[str],
    catalogs: np.ndarray,
    source_distance_m: np.ndarray,
    source_dt_sec: np.ndarray,
    beta_m_per_s: float,
) -> dict[str, Any]:
    start = HORIZONS.index(LATE_HORIZONS[0])
    stop = HORIZONS.index(LATE_HORIZONS[-1]) + 1
    late = np.asarray(rates[start:stop]).transpose(1, 0, 2)
    return _late_metrics_from_rates(
        late,
        events=events,
        catalogs=catalogs,
        source_distance_m=source_distance_m,
        source_dt_sec=source_dt_sec,
        beta_m_per_s=beta_m_per_s,
    )


def _write_comparison_outputs(
    *,
    output_dir: Path,
    tables: Mapping[str, Any],
    summary: Mapping[str, Any],
    include_station_trajectories: bool,
) -> None:
    station_fields = (
        "event",
        "station",
        "observation_horizon_sec",
        "release_time_sec",
        "mw_catalog",
        "raw_mw_pred",
        "adapted_mw_pred",
        "raw_error",
        "adapted_error",
        "raw_delta_mw",
        "adapted_delta_mw",
    )
    if include_station_trajectories:
        _write_csv(
            output_dir / "station_predictions.csv",
            tables["station_rows"],
            fieldnames=station_fields,
        )
    _write_csv(
        output_dir / "endpoint_station_predictions.csv",
        tables["endpoint_station_rows"],
        fieldnames=station_fields,
    )
    _write_csv(
        output_dir / "event_predictions.csv",
        tables["event_rows"],
        fieldnames=(
            "event",
            "observation_horizon_sec",
            "release_time_sec",
            "mw_catalog",
            "raw_mw_pred_median",
            "adapted_mw_pred_median",
            "raw_error",
            "adapted_error",
            "raw_abs_error",
            "adapted_abs_error",
            "raw_delta_mw",
            "adapted_delta_mw",
            "station_count",
        ),
    )
    _write_csv(
        output_dir / "horizon_metrics.csv",
        tables["horizon_rows"],
        fieldnames=tuple(tables["horizon_rows"][0]),
    )
    convergence = build_convergence_rows(tables["event_rows"])
    _write_csv(
        output_dir / "event_convergence.csv",
        convergence,
        fieldnames=tuple(convergence[0]),
    )
    _write_json(output_dir / "summary.json", dict(summary))


def evaluate_internal(
    *,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    split_name: str,
    limit: int | None = None,
) -> dict[str, Any]:
    _validate_new_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = validate_phase45_artifacts()
    config = load_frozen_config()
    samples, dataset_indices, split_manifest = _load_internal_samples(
        config,
        split_name=split_name,
        limit=limit,
    )
    model = _load_phase39_internal_model(config, device=device)
    adapter = load_adapter(config, device=device)
    raw_rates, input_gate = generate_internal_raw_rates(
        config=config,
        samples=samples,
        model=model,
        output_path=output_dir / "raw_rates.npy",
        batch_size=batch_size,
    )
    available_mask = np.ones(raw_rates.shape[:2], dtype=bool)
    source_distance = np.asarray(
        [float(sample["source_distance_m"]) for sample in samples],
        dtype=np.float32,
    )
    source_dt = np.asarray(
        [float(sample["stf_dt_sec"]) for sample in samples],
        dtype=np.float32,
    )
    adapted_rates = apply_adapter_to_rates(
        adapter=adapter,
        raw_rates=raw_rates,
        available_mask=available_mask,
        source_distance_m=source_distance,
        source_dt_sec=source_dt,
        beta_m_per_s=float(config["physics"]["beta"]),
        output_path=output_dir / "adapted_rates.npy",
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
        adapted_rates=adapted_rates,
        available_mask=available_mask,
        events=events,
        stations=stations,
        catalogs=catalogs,
        source_dt_sec=source_dt,
        include_station_trajectories=False,
    )
    raw_late = _late_metrics(
        raw_rates,
        events=events,
        catalogs=catalogs,
        source_distance_m=source_distance,
        source_dt_sec=source_dt,
        beta_m_per_s=float(config["physics"]["beta"]),
    )
    adapted_late = _late_metrics(
        adapted_rates,
        events=events,
        catalogs=catalogs,
        source_distance_m=source_distance,
        source_dt_sec=source_dt,
        beta_m_per_s=float(config["physics"]["beta"]),
    )
    stage_status = "smoke_complete" if limit is not None else "complete"
    summary = {
        "status": stage_status,
        "candidate": "Phase45 seed42 epoch27, audit-only",
        "evaluation_role": (
            "opened_internal_test_posthoc"
            if split_name == "test"
            else "validation_smoke_no_hidden_data"
        ),
        "split": split_name,
        "station_count": len(samples),
        "event_count": len(set(events)),
        "dataset_indices": dataset_indices,
        "split_assignment_sha256": split_manifest["assignment_sha256"],
        "processing_delay_sec": PROCESSING_DELAY_SEC,
        "endpoint_input_gate": input_gate,
        "raw_metrics": raw_late,
        "adapted_metrics": adapted_late,
        "phase39_weights_trained": False,
        "internal_test_iterated": split_name == "test",
        "external_data_loaded": False,
        "grouped_test_loaded": False,
        "artifact_sha256": {
            "phase39_checkpoint": EXPECTED_PHASE39_SHA256,
            "adapter_checkpoint": EXPECTED_ADAPTER_SHA256,
            "phase45_summary": artifacts["phase45_summary_sha256"],
            "raw_rates": sha256_file(output_dir / "raw_rates.npy"),
            "adapted_rates": sha256_file(output_dir / "adapted_rates.npy"),
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
        include_station_trajectories=False,
    )
    return summary


def evaluate_external(
    *,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    _validate_new_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = validate_phase45_artifacts()
    external_artifact_hashes = validate_frozen_artifacts()
    external_input_hashes = hash_external_inputs(EXTERNAL_EVENT_ROOT)
    config = load_phase39_config()
    labels = load_label_contract(LABELS_PATH)
    endpoint_reference = load_endpoint_reference()
    records = build_raw_streaming_records(
        config=config,
        event_root=EXTERNAL_EVENT_ROOT,
        labels_by_dir=labels,
        expected_station_keys=set(endpoint_reference["station_predictions"]),
    )
    if len(records) != EXPECTED_STATION_COUNT:
        raise ValueError("external station count changed")
    model = load_model(config, device=device)
    adapter = load_adapter(config, device=device)
    raw_rates, available_mask, input_gate = generate_external_raw_rates(
        config=config,
        records=records,
        model=model,
        output_path=output_dir / "raw_rates.npy",
        batch_size=batch_size,
    )
    source_distance = np.asarray(
        [record.source_distance_m for record in records],
        dtype=np.float32,
    )
    source_dt = np.full(len(records), SOURCE_DT_SEC, dtype=np.float32)
    adapted_rates = apply_adapter_to_rates(
        adapter=adapter,
        raw_rates=raw_rates,
        available_mask=available_mask,
        source_distance_m=source_distance,
        source_dt_sec=source_dt,
        beta_m_per_s=float(config["physics"]["beta"]),
        output_path=output_dir / "adapted_rates.npy",
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
        adapted_rates=adapted_rates,
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
    late_mask = available_mask[
        HORIZONS.index(LATE_HORIZONS[0]) : HORIZONS.index(LATE_HORIZONS[-1]) + 1
    ]
    if not bool(np.all(late_mask)):
        raise ValueError("external late-horizon cohort is incomplete")
    raw_late = _late_metrics(
        raw_rates,
        events=events,
        catalogs=catalogs,
        source_distance_m=source_distance,
        source_dt_sec=source_dt,
        beta_m_per_s=float(config["physics"]["beta"]),
    )
    adapted_late = _late_metrics(
        adapted_rates,
        events=events,
        catalogs=catalogs,
        source_distance_m=source_distance,
        source_dt_sec=source_dt,
        beta_m_per_s=float(config["physics"]["beta"]),
    )
    summary = {
        "status": "complete",
        "candidate": "Phase45 seed42 epoch27, audit-only",
        "evaluation_role": "development_validation_posthoc",
        "event_count": len(set(events)),
        "station_count": len(records),
        "processing_delay_sec": PROCESSING_DELAY_SEC,
        "endpoint_input_gate": input_gate,
        "endpoint_raw_reproduction_gate": endpoint_gate,
        "raw_metrics": raw_late,
        "adapted_metrics": adapted_late,
        "phase39_weights_trained": False,
        "internal_test_iterated": False,
        "external_data_loaded": True,
        "grouped_test_loaded": False,
        "artifact_sha256": {
            "phase39_checkpoint": EXPECTED_PHASE39_SHA256,
            "adapter_checkpoint": EXPECTED_ADAPTER_SHA256,
            "phase45_summary": artifacts["phase45_summary_sha256"],
            "external_frozen_artifacts": external_artifact_hashes,
            "external_inputs": external_input_hashes,
            "raw_rates": sha256_file(output_dir / "raw_rates.npy"),
            "adapted_rates": sha256_file(output_dir / "adapted_rates.npy"),
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen Phase45 seed42 epoch27 adapter on saved "
            "training traces, internal test, or the eight external events."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("training", "smoke", "internal", "external"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
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
            summary = evaluate_internal(
                output_dir=output_dir,
                device=device,
                batch_size=args.batch_size,
                split_name="validation",
                limit=4,
            )
        elif args.stage == "internal":
            summary = evaluate_internal(
                output_dir=output_dir,
                device=device,
                batch_size=args.batch_size,
                split_name="test",
            )
        else:
            summary = evaluate_external(
                output_dir=output_dir,
                device=device,
                batch_size=args.batch_size,
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
