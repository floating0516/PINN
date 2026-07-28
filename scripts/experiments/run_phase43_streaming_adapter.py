from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_phase40_streaming_consistency import (  # noqa: E402
    BASELINE_METRICS,
    EXPECTED_SPLIT_ASSIGNMENT_SHA256,
    GATES,
    PHASE39_CHECKPOINT,
    _absolute_summary,
    _assert_formal_baseline,
    _build_stf_rate_criterion,
    _training_event_balance_weights,
    _write_csv,
    _write_json,
    load_frozen_config,
    validate_source_artifacts,
    validation_gate,
)
from src.data.loaders_v2 import get_data_loaders_v2  # noqa: E402
from src.data.metadata import (  # noqa: E402
    build_metadata_tensor,
    metadata_distance_from_config,
)
from src.data.records_v2 import (  # noqa: E402
    NormalizedStationRecord,
    _iter_normalized_station_records,
)
from src.data.sample_builder import rotate_horizontal_to_rt  # noqa: E402
from src.data.waveform import (  # noqa: E402
    WaveformConfig,
    preprocess_waveform,
    waveform_config_from_v2,
)
from src.evaluation.metrics import (  # noqa: E402
    aggregate_event_predictions,
    summarize_predictions,
)
from src.models.model import PINNModel  # noqa: E402
from src.models.streaming_stf_adapter import StreamingSTFAdapter  # noqa: E402
from src.training.checkpointing import atomic_torch_save  # noqa: E402
from src.training.loss_stf_rate_v2 import (  # noqa: E402
    moment_magnitude_from_rate,
)
from src.utils.config_v2 import stf_m_ref_from_config  # noqa: E402
from src.utils.device import configure_runtime  # noqa: E402
from src.utils.provenance import (  # noqa: E402
    current_git_commit,
    git_is_dirty,
    sha256_file,
)


CACHE_VERSION = 1
HORIZONS = tuple(range(20, 201))
LATE_HORIZONS = tuple(range(179, 201))
PROCESSING_DELAY_SEC = 6.0
ADAPTER_SEEDS = (17, 42, 73)
EXPECTED_TRAIN_COUNT = 1_788
EXPECTED_VALIDATION_COUNT = 385
EXPECTED_EVENT_COUNT = 30
EXPECTED_ADAPTER_PARAMETER_COUNT = 489

BATCH_SIZE = 32
CACHE_INFERENCE_BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP_NORM = 1.0
STEP_HUBER_BETA_MW = 0.02
CONFIRMED_HUBER_BETA_LOG10 = 0.05
OBJECTIVE_PROFILES = {
    "phase43_balanced": {
        "endpoint_science": 2.0,
        "sequence_target": 1.0,
        "endpoint_teacher": 1.0,
        "late_step": 0.1,
        "confirmed_history": 0.1,
    },
    "phase44_stability10": {
        "endpoint_science": 2.0,
        "sequence_target": 1.0,
        "endpoint_teacher": 1.0,
        "late_step": 1.0,
        "confirmed_history": 1.0,
    },
    "phase45_stability20": {
        "endpoint_science": 2.0,
        "sequence_target": 1.0,
        "endpoint_teacher": 1.0,
        "late_step": 2.0,
        "confirmed_history": 2.0,
    },
}
ACTIVE_OBJECTIVE_PROFILE = "phase43_balanced"
LOSS_WEIGHTS = dict(OBJECTIVE_PROFILES[ACTIVE_OBJECTIVE_PROFILE])
NORMALIZER_BATCHES = 8
NORMALIZER_FLOORS = {
    "endpoint_science": 1.0e-8,
    "sequence_target": 1.0e-8,
    "endpoint_teacher": 1.0e-4,
    "late_step": 1.0e-8,
    "confirmed_history": 1.0e-8,
}
CACHE_BASELINE_TOLERANCE_MW = 5.0e-7


@dataclass(frozen=True)
class RawInternalStation:
    event: str
    station: str
    raw_time_sec: np.ndarray
    raw_radial: np.ndarray
    waveform_config: WaveformConfig
    p_arrival_sec: float


@dataclass
class CacheBundle:
    root: Path
    raw_rates: np.ndarray
    arrays: Mapping[str, np.ndarray]
    records: list[dict[str, Any]]
    manifest: dict[str, Any]


def select_objective_profile(name: str) -> None:
    global ACTIVE_OBJECTIVE_PROFILE, LOSS_WEIGHTS
    if name not in OBJECTIVE_PROFILES:
        raise ValueError(f"unknown Phase43/44 objective profile: {name}")
    ACTIVE_OBJECTIVE_PROFILE = name
    LOSS_WEIGHTS = dict(OBJECTIVE_PROFILES[name])


def encode_rate(rate_nm_per_s: torch.Tensor, *, stf_m_ref: float) -> torch.Tensor:
    if not math.isfinite(stf_m_ref) or stf_m_ref <= 0.0:
        raise ValueError("stf_m_ref must be positive and finite")
    if bool(torch.any(rate_nm_per_s < 0.0)):
        raise ValueError("physical STF rate must be nonnegative")
    return torch.log10(1.0 + rate_nm_per_s / float(stf_m_ref))


def _validate_new_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output directory must be new or empty: {path}")


def _selected_dataset_context(
    config: dict[str, Any],
) -> tuple[Any, list[int], list[int], dict[str, Any], dict[str, float] | None]:
    train_loader, validation_loader, test_loader, split_manifest = get_data_loaders_v2(
        config
    )
    if split_manifest["assignment_sha256"] != EXPECTED_SPLIT_ASSIGNMENT_SHA256:
        raise ValueError("Phase39 train/validation/test assignment changed")
    train_indices = [int(value) for value in train_loader.dataset.indices]
    validation_indices = [int(value) for value in validation_loader.dataset.indices]
    if len(train_indices) != EXPECTED_TRAIN_COUNT:
        raise ValueError("Phase39 training count changed")
    if len(validation_indices) != EXPECTED_VALIDATION_COUNT:
        raise ValueError("Phase39 validation count changed")
    full_dataset = train_loader.dataset.dataset
    event_weights = _training_event_balance_weights(config, train_loader)
    # The test loader is intentionally never iterated or indexed.
    del test_loader
    return (
        full_dataset,
        train_indices,
        validation_indices,
        split_manifest,
        event_weights,
    )


def _raw_record_map(
    config: dict[str, Any],
    *,
    selected_samples: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], RawInternalStation]:
    sample_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for sample in selected_samples:
        key = (str(sample["event"]), str(sample["station"]))
        if key in sample_by_key:
            raise ValueError(f"duplicate selected station identity: {key}")
        sample_by_key[key] = sample

    base_waveform_config = waveform_config_from_v2(config)
    replay_waveform_config = replace(base_waveform_config, min_valid_fraction=0.0)
    records: dict[tuple[str, str], RawInternalStation] = {}
    with np.load(config["paths"]["data_path"], allow_pickle=True) as data:
        for record in _iter_normalized_station_records(data):
            key = (record.event, record.station)
            sample = sample_by_key.get(key)
            if sample is None:
                continue
            if key in records:
                raise ValueError(f"duplicate raw station identity: {key}")
            raw_time = np.asarray(record.time_sec, dtype=np.float64)
            if record.origin_sec is not None:
                raw_time = raw_time - float(record.origin_sec)
            raw_radial, _ = rotate_horizontal_to_rt(
                np.asarray(record.east, dtype=np.float64),
                np.asarray(record.north, dtype=np.float64),
                float(sample["azimuth_deg"]),
            )
            records[key] = RawInternalStation(
                event=key[0],
                station=key[1],
                raw_time_sec=raw_time,
                raw_radial=np.asarray(raw_radial, dtype=np.float64),
                waveform_config=replay_waveform_config,
                p_arrival_sec=(
                    float(sample["source_distance_m"])
                    / float(config["physics"]["alpha"])
                ),
            )
    if set(records) != set(sample_by_key):
        missing = sorted(set(sample_by_key) - set(records))
        extra = sorted(set(records) - set(sample_by_key))
        raise ValueError(
            "raw train/validation cohort mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return records


def _preprocess_prefix(
    record: RawInternalStation,
    *,
    horizon_sec: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    issue_time_sec = float(horizon_sec) + PROCESSING_DELAY_SEC
    finite = (
        np.isfinite(record.raw_time_sec)
        & np.isfinite(record.raw_radial)
        & (record.raw_time_sec <= issue_time_sec)
    )
    if int(np.count_nonzero(finite)) < 2:
        raise ValueError("fewer than two finite raw samples at issue time")
    processed = preprocess_waveform(
        record.raw_time_sec[finite],
        record.raw_radial[finite],
        units="mm",
        p_arrival_sec=record.p_arrival_sec,
        config=record.waveform_config,
    )
    values = np.asarray(processed.values_m[:horizon_sec], dtype=np.float32)
    mask = np.asarray(processed.valid_mask[:horizon_sec], dtype=bool)
    if values.shape != (horizon_sec,) or mask.shape != values.shape:
        raise ValueError("causal prefix shape changed")
    if np.any(values[~mask] != 0.0):
        raise ValueError("invalid causal prefix samples must remain zero")
    return values, mask, str(processed.baseline_source)


def _metadata_tensor(
    samples: Sequence[Mapping[str, Any]],
    config: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    source_distance = torch.tensor(
        [float(sample["source_distance_m"]) for sample in samples],
        device=device,
        dtype=dtype,
    )
    epicentral_distance = torch.tensor(
        [float(sample["epicentral_distance_m"]) for sample in samples],
        device=device,
        dtype=dtype,
    )
    theta = torch.tensor(
        [float(sample["theta_deg"]) for sample in samples],
        device=device,
        dtype=dtype,
    )
    azimuth = torch.tensor(
        [float(sample["azimuth_deg"]) for sample in samples],
        device=device,
        dtype=dtype,
    )
    distance = metadata_distance_from_config(
        config,
        source_distance_m=source_distance,
        epicentral_distance_m=epicentral_distance,
    )
    return build_metadata_tensor(distance, theta, azimuth)


def _late_metrics_from_rates(
    rates: np.ndarray,
    *,
    events: Sequence[str],
    catalogs: np.ndarray,
    source_distance_m: np.ndarray,
    source_dt_sec: np.ndarray,
    beta_m_per_s: float,
) -> dict[str, Any]:
    cube = np.asarray(rates, dtype=np.float64)
    if cube.ndim != 3 or cube.shape[1:] != (len(LATE_HORIZONS), 200):
        raise ValueError("late rate cube must have shape (station, 22, 200)")
    if cube.shape[0] != len(events):
        raise ValueError("late rate station count changed")
    dt = np.asarray(source_dt_sec, dtype=np.float64).reshape(-1, 1, 1)
    moments = np.sum(np.maximum(cube, 0.0) * dt, axis=2)
    mw = (2.0 / 3.0) * (np.log10(np.maximum(moments, 1.0e10)) - 9.1)
    station_steps = (mw[:, 1:] - mw[:, :-1]).reshape(-1)

    event_predictions: list[dict[str, float]] = []
    endpoint_station_rows: list[dict[str, Any]] = []
    endpoint_event_rows: list[dict[str, Any]] = []
    for horizon_index, horizon in enumerate(LATE_HORIZONS):
        station_rows = [
            {
                "event": str(events[index]),
                "mw_pred": float(mw[index, horizon_index]),
                "mw_catalog": float(catalogs[index]),
            }
            for index in range(len(events))
        ]
        event_rows = aggregate_event_predictions(
            station_rows,
            reference_key="mw_catalog",
        )
        event_predictions.append(
            {
                str(row["event"]): float(row["mw_pred_median"])
                for row in event_rows
            }
        )
        if horizon == 200:
            endpoint_station_rows = station_rows
            endpoint_event_rows = event_rows
    event_steps = [
        event_predictions[index][event] - event_predictions[index - 1][event]
        for index in range(1, len(LATE_HORIZONS))
        for event in sorted(event_predictions[index])
    ]

    confirmed_log: list[float] = []
    confirmed_downward: list[bool] = []
    source_index = np.arange(200, dtype=np.int64)
    distances = np.asarray(source_distance_m, dtype=np.float64)
    sample_dt = np.asarray(source_dt_sec, dtype=np.float64)
    for transition_index, current_horizon in enumerate(LATE_HORIZONS[1:], start=1):
        previous_horizon = current_horizon - 1
        supported = np.floor(
            np.maximum(0.0, previous_horizon - distances / beta_m_per_s)
            + 1.0e-12
        ).astype(np.int64)
        supported = np.clip(supported, 0, 200)
        for station_index, steps in enumerate(supported):
            if steps <= 0:
                continue
            previous = cube[station_index, transition_index - 1]
            current = cube[station_index, transition_index]
            previous_cumulative = np.maximum(
                np.cumsum(previous * sample_dt[station_index]),
                1.0e10,
            )
            current_cumulative = np.maximum(
                np.cumsum(current * sample_dt[station_index]),
                1.0e10,
            )
            mask = source_index < steps
            confirmed_log.append(
                float(
                    np.mean(
                        np.abs(
                            np.log10(current_cumulative[mask])
                            - np.log10(previous_cumulative[mask])
                        )
                    )
                )
            )
            confirmed_downward.append(
                bool(
                    np.sum(current[mask] * sample_dt[station_index])
                    < np.sum(previous[mask] * sample_dt[station_index])
                )
            )

    endpoint = summarize_predictions(
        endpoint_station_rows,
        endpoint_event_rows,
        reference_key="mw_catalog",
    )
    station_summary = _absolute_summary(station_steps)
    event_summary = _absolute_summary(event_steps)
    confirmed_summary = _absolute_summary(confirmed_log)
    return {
        "validation_station_count": int(endpoint["station_count"]),
        "validation_event_count": int(endpoint["event_count"]),
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


def _assert_cache_baseline(metrics: Mapping[str, Any]) -> None:
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
        if not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=CACHE_BASELINE_TOLERANCE_MW,
        ):
            raise ValueError(
                f"cached Phase39 validation baseline changed for {name}: "
                f"{actual} != {expected}"
            )


def build_prefix_cache(
    *,
    cache_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    _validate_new_directory(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    source_hashes = validate_source_artifacts()
    config = load_frozen_config()
    configure_runtime(42, device)
    (
        full_dataset,
        train_indices,
        validation_indices,
        split_manifest,
        event_weights,
    ) = _selected_dataset_context(config)
    selected_indices = train_indices + validation_indices
    selected_samples = [full_dataset.samples[index] for index in selected_indices]
    raw_by_key = _raw_record_map(config, selected_samples=selected_samples)
    raw_records = [
        raw_by_key[(str(sample["event"]), str(sample["station"]))]
        for sample in selected_samples
    ]

    model = PINNModel(config).to(device)
    source_state = torch.load(PHASE39_CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(source_state, strict=True)
    model.eval()
    criterion = _build_stf_rate_criterion(config, device)
    parameter = next(model.parameters())
    metadata = _metadata_tensor(
        selected_samples,
        config,
        device=device,
        dtype=parameter.dtype,
    )

    raw_rate_path = cache_root / "raw_rates.npy"
    raw_rates = np.lib.format.open_memmap(
        raw_rate_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(selected_samples), len(HORIZONS), 200),
    )
    endpoint_max_input_diff = 0.0
    endpoint_mask_mismatches = 0
    endpoint_baseline_mismatches = 0
    with torch.no_grad():
        for horizon_index, horizon in enumerate(HORIZONS):
            prefixes = np.empty((len(raw_records), horizon), dtype=np.float32)
            for record_index, (record, sample) in enumerate(
                zip(raw_records, selected_samples, strict=True)
            ):
                values, mask, baseline_source = _preprocess_prefix(
                    record,
                    horizon_sec=horizon,
                )
                prefixes[record_index] = values
                if horizon == 200:
                    endpoint_max_input_diff = max(
                        endpoint_max_input_diff,
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
            for start in range(0, len(selected_samples), CACHE_INFERENCE_BATCH_SIZE):
                stop = min(start + CACHE_INFERENCE_BATCH_SIZE, len(selected_samples))
                model_input = torch.as_tensor(
                    prefixes[start:stop],
                    device=device,
                    dtype=parameter.dtype,
                ).unsqueeze(1)
                encoded = model(model_input, meta=metadata[start:stop])
                rates = criterion._decode_rate(encoded)
                raw_rates[start:stop, horizon_index] = (
                    rates.detach().cpu().numpy().astype(np.float32, copy=False)
                )
            raw_rates.flush()
            if horizon == HORIZONS[0] or horizon % 10 == 0 or horizon == 200:
                print(
                    f"cache horizon={horizon}/200 records={len(selected_samples)}",
                    flush=True,
                )

    if endpoint_max_input_diff != 0.0:
        raise ValueError(
            f"h=200 causal cache input changed by {endpoint_max_input_diff:.9g} m"
        )
    if endpoint_mask_mismatches or endpoint_baseline_mismatches:
        raise ValueError("h=200 causal preprocessing metadata changed")

    split_code = np.concatenate(
        (
            np.zeros(len(train_indices), dtype=np.int8),
            np.ones(len(validation_indices), dtype=np.int8),
        )
    )
    sample_weights = np.ones(len(selected_samples), dtype=np.float32)
    if event_weights is not None:
        for index, sample in enumerate(selected_samples[: len(train_indices)]):
            sample_weights[index] = float(event_weights[str(sample["event"])])
    arrays = {
        "dataset_index": np.asarray(selected_indices, dtype=np.int32),
        "split_code": split_code,
        "magnitude_catalog": np.asarray(
            [float(sample["magnitude_catalog"]) for sample in selected_samples],
            dtype=np.float32,
        ),
        "stf": np.stack(
            [np.asarray(sample["stf"], dtype=np.float32) for sample in selected_samples]
        ),
        "stf_log": np.stack(
            [
                np.asarray(sample["stf_log"], dtype=np.float32)
                for sample in selected_samples
            ]
        ),
        "radial": np.stack(
            [
                np.asarray(sample["radial"], dtype=np.float32)
                for sample in selected_samples
            ]
        ),
        "waveform_valid_mask": np.stack(
            [
                np.asarray(sample["waveform_valid_mask"], dtype=bool)
                for sample in selected_samples
            ]
        ),
        "source_distance_m": np.asarray(
            [float(sample["source_distance_m"]) for sample in selected_samples],
            dtype=np.float32,
        ),
        "theta_deg": np.asarray(
            [float(sample["theta_deg"]) for sample in selected_samples],
            dtype=np.float32,
        ),
        "phi_slip_deg": np.asarray(
            [float(sample["phi_slip_deg"]) for sample in selected_samples],
            dtype=np.float32,
        ),
        "source_dt_sec": np.asarray(
            [float(sample["stf_dt_sec"]) for sample in selected_samples],
            dtype=np.float32,
        ),
        "observation_dt_sec": np.asarray(
            [float(sample["waveform_dt_sec"]) for sample in selected_samples],
            dtype=np.float32,
        ),
        "sample_weight": sample_weights,
    }
    np.savez(cache_root / "arrays.npz", **arrays)
    records_payload = [
        {
            "cache_index": index,
            "dataset_index": int(selected_indices[index]),
            "split": "train" if index < len(train_indices) else "validation",
            "event": str(sample["event"]),
            "station": str(sample["station"]),
        }
        for index, sample in enumerate(selected_samples)
    ]
    _write_json(cache_root / "records.json", records_payload)

    validation_slice = slice(len(train_indices), len(selected_samples))
    validation_late = np.asarray(
        raw_rates[
            validation_slice,
            HORIZONS.index(LATE_HORIZONS[0]) : HORIZONS.index(LATE_HORIZONS[-1]) + 1,
        ],
        dtype=np.float32,
    )
    raw_metrics = _late_metrics_from_rates(
        validation_late,
        events=[
            str(sample["event"])
            for sample in selected_samples[len(train_indices) :]
        ],
        catalogs=arrays["magnitude_catalog"][validation_slice],
        source_distance_m=arrays["source_distance_m"][validation_slice],
        source_dt_sec=arrays["source_dt_sec"][validation_slice],
        beta_m_per_s=float(config["physics"]["beta"]),
    )
    _assert_cache_baseline(raw_metrics)
    raw_rates.flush()
    del raw_rates
    manifest = {
        "cache_version": CACHE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "git_dirty": git_is_dirty(PROJECT_ROOT),
        "device": str(device),
        "source_artifact_sha256": source_hashes,
        "source_checkpoint": str(PHASE39_CHECKPOINT),
        "split_assignment_sha256": split_manifest["assignment_sha256"],
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "horizons": list(HORIZONS),
        "processing_delay_sec": PROCESSING_DELAY_SEC,
        "raw_rate_shape": [len(selected_samples), len(HORIZONS), 200],
        "endpoint_max_input_diff_m": endpoint_max_input_diff,
        "endpoint_mask_mismatch_count": endpoint_mask_mismatches,
        "endpoint_baseline_mismatch_count": endpoint_baseline_mismatches,
        "raw_validation_metrics": raw_metrics,
        "internal_test_iterated": False,
        "external_data_loaded": False,
        "grouped_test_loaded": False,
    }
    manifest["raw_rates_sha256"] = sha256_file(raw_rate_path)
    manifest["arrays_sha256"] = sha256_file(cache_root / "arrays.npz")
    _write_json(cache_root / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "cache_complete",
                "cache_root": str(cache_root),
                "raw_rates_sha256": manifest["raw_rates_sha256"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return manifest


def finalize_existing_cache(
    *,
    cache_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    manifest_path = cache_root / "manifest.json"
    raw_rate_path = cache_root / "raw_rates.npy"
    arrays_path = cache_root / "arrays.npz"
    records_path = cache_root / "records.json"
    if manifest_path.exists():
        raise ValueError("Phase43 cache manifest already exists")
    for path in (raw_rate_path, arrays_path, records_path):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete Phase43 cache artifact: {path}")
    source_hashes = validate_source_artifacts()
    config = load_frozen_config()
    (
        full_dataset,
        train_indices,
        validation_indices,
        split_manifest,
        _,
    ) = _selected_dataset_context(config)
    selected_indices = train_indices + validation_indices
    selected_samples = [full_dataset.samples[index] for index in selected_indices]
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if len(records) != len(selected_samples):
        raise ValueError("partial Phase43 cache record count changed")
    for index, sample in enumerate(selected_samples):
        expected = (str(sample["event"]), str(sample["station"]))
        actual = (str(records[index]["event"]), str(records[index]["station"]))
        if actual != expected:
            raise ValueError(f"partial Phase43 cache record order changed at {index}")

    arrays_file = np.load(arrays_path, allow_pickle=False)
    arrays = {name: arrays_file[name] for name in arrays_file.files}
    arrays_file.close()
    raw_rates = np.load(raw_rate_path, mmap_mode="r")
    expected_shape = (len(selected_samples), len(HORIZONS), 200)
    if raw_rates.shape != expected_shape or raw_rates.dtype != np.float32:
        raise ValueError("partial Phase43 raw-rate cache shape changed")

    raw_by_key = _raw_record_map(config, selected_samples=selected_samples)
    endpoint_max_input_diff = 0.0
    endpoint_mask_mismatches = 0
    endpoint_baseline_mismatches = 0
    for sample in selected_samples:
        key = (str(sample["event"]), str(sample["station"]))
        values, mask, baseline_source = _preprocess_prefix(
            raw_by_key[key],
            horizon_sec=200,
        )
        endpoint_max_input_diff = max(
            endpoint_max_input_diff,
            float(
                np.max(
                    np.abs(values - np.asarray(sample["radial"], dtype=np.float32))
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
    if endpoint_max_input_diff != 0.0:
        raise ValueError("partial Phase43 h=200 input no longer matches Phase39")
    if endpoint_mask_mismatches or endpoint_baseline_mismatches:
        raise ValueError("partial Phase43 h=200 preprocessing metadata changed")

    validation_slice = slice(len(train_indices), len(selected_samples))
    validation_late = np.asarray(
        raw_rates[
            validation_slice,
            HORIZONS.index(LATE_HORIZONS[0]) : HORIZONS.index(LATE_HORIZONS[-1]) + 1,
        ],
        dtype=np.float32,
    )
    raw_metrics = _late_metrics_from_rates(
        validation_late,
        events=[str(sample["event"]) for sample in selected_samples[len(train_indices) :]],
        catalogs=np.asarray(arrays["magnitude_catalog"][validation_slice]),
        source_distance_m=np.asarray(arrays["source_distance_m"][validation_slice]),
        source_dt_sec=np.asarray(arrays["source_dt_sec"][validation_slice]),
        beta_m_per_s=float(config["physics"]["beta"]),
    )
    _assert_cache_baseline(raw_metrics)
    manifest = {
        "cache_version": CACHE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "git_dirty": git_is_dirty(PROJECT_ROOT),
        "device": str(device),
        "source_artifact_sha256": source_hashes,
        "source_checkpoint": str(PHASE39_CHECKPOINT),
        "split_assignment_sha256": split_manifest["assignment_sha256"],
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "horizons": list(HORIZONS),
        "processing_delay_sec": PROCESSING_DELAY_SEC,
        "raw_rate_shape": list(expected_shape),
        "endpoint_max_input_diff_m": endpoint_max_input_diff,
        "endpoint_mask_mismatch_count": endpoint_mask_mismatches,
        "endpoint_baseline_mismatch_count": endpoint_baseline_mismatches,
        "cache_baseline_tolerance_mw": CACHE_BASELINE_TOLERANCE_MW,
        "raw_validation_metrics": raw_metrics,
        "internal_test_iterated": False,
        "external_data_loaded": False,
        "grouped_test_loaded": False,
        "recovered_after_completed_generation": True,
    }
    manifest["raw_rates_sha256"] = sha256_file(raw_rate_path)
    manifest["arrays_sha256"] = sha256_file(arrays_path)
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "cache_finalized",
                "cache_root": str(cache_root),
                "raw_rates_sha256": manifest["raw_rates_sha256"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return manifest


def load_cache(cache_root: Path) -> CacheBundle:
    manifest_path = cache_root / "manifest.json"
    arrays_path = cache_root / "arrays.npz"
    records_path = cache_root / "records.json"
    raw_rate_path = cache_root / "raw_rates.npy"
    for path in (manifest_path, arrays_path, records_path, raw_rate_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing Phase43 cache artifact: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["cache_version"]) != CACHE_VERSION:
        raise ValueError("Phase43 cache version changed")
    if manifest["split_assignment_sha256"] != EXPECTED_SPLIT_ASSIGNMENT_SHA256:
        raise ValueError("Phase43 cache split assignment changed")
    if int(manifest["train_count"]) != EXPECTED_TRAIN_COUNT:
        raise ValueError("Phase43 cache train count changed")
    if int(manifest["validation_count"]) != EXPECTED_VALIDATION_COUNT:
        raise ValueError("Phase43 cache validation count changed")
    if tuple(int(value) for value in manifest["horizons"]) != HORIZONS:
        raise ValueError("Phase43 cache horizons changed")
    if sha256_file(raw_rate_path) != manifest["raw_rates_sha256"]:
        raise ValueError("Phase43 raw-rate cache hash changed")
    if sha256_file(arrays_path) != manifest["arrays_sha256"]:
        raise ValueError("Phase43 cache arrays hash changed")
    arrays_file = np.load(arrays_path, allow_pickle=False)
    arrays = {name: arrays_file[name] for name in arrays_file.files}
    arrays_file.close()
    records = json.loads(records_path.read_text(encoding="utf-8"))
    raw_rates = np.load(raw_rate_path, mmap_mode="r")
    expected_shape = (
        EXPECTED_TRAIN_COUNT + EXPECTED_VALIDATION_COUNT,
        len(HORIZONS),
        200,
    )
    if raw_rates.shape != expected_shape or raw_rates.dtype != np.float32:
        raise ValueError("Phase43 raw-rate cache shape or dtype changed")
    if len(records) != expected_shape[0]:
        raise ValueError("Phase43 cache record count changed")
    _assert_cache_baseline(manifest["raw_validation_metrics"])
    return CacheBundle(
        root=cache_root,
        raw_rates=raw_rates,
        arrays=arrays,
        records=records,
        manifest=manifest,
    )


def _tensor_batch(
    cache: CacheBundle,
    indices: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {
        "raw_rate": torch.as_tensor(
            np.asarray(cache.raw_rates[indices]).copy(),
            device=device,
            dtype=torch.float32,
        )
    }
    dtypes = {
        "waveform_valid_mask": torch.bool,
    }
    for name in (
        "magnitude_catalog",
        "stf",
        "stf_log",
        "radial",
        "waveform_valid_mask",
        "source_distance_m",
        "theta_deg",
        "phi_slip_deg",
        "source_dt_sec",
        "observation_dt_sec",
        "sample_weight",
    ):
        result[name] = torch.as_tensor(
            np.asarray(cache.arrays[name][indices]).copy(),
            device=device,
            dtype=dtypes.get(name, torch.float32),
        )
    return result


def _weighted_sample_mean(
    per_sample: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    if per_sample.ndim != 1 or sample_weights.shape != per_sample.shape:
        raise ValueError("weighted values and sample weights must have shape (batch,)")
    return torch.mean(per_sample * sample_weights)


def adapter_loss_components(
    adapter: StreamingSTFAdapter,
    batch: Mapping[str, torch.Tensor],
    *,
    criterion: Any,
    config: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    raw_rate = batch["raw_rate"]
    source_distance = batch["source_distance_m"]
    source_dt = batch["source_dt_sec"]
    sample_weights = batch["sample_weight"]
    stf_m_ref = stf_m_ref_from_config(dict(config))
    beta_m_per_s = float(config["physics"]["beta"])
    states, gates = adapter(
        raw_rate,
        horizons_sec=HORIZONS,
        source_distance_m=source_distance,
        source_dt_sec=source_dt,
        beta_m_per_s=beta_m_per_s,
    )
    encoded_states = encode_rate(states, stf_m_ref=stf_m_ref)

    endpoint_science, endpoint_parts = criterion(
        encoded_states[:, -1],
        pred_catalog_mw=None,
        radial_obs=batch["radial"],
        source_distance_m=source_distance,
        theta_deg=batch["theta_deg"],
        phi_slip_deg=batch["phi_slip_deg"],
        source_dt_sec=source_dt,
        observation_dt_sec=batch["observation_dt_sec"],
        waveform_valid_mask=batch["waveform_valid_mask"],
        stf_true=batch["stf"],
        has_stf=torch.ones_like(batch["magnitude_catalog"], dtype=torch.bool),
        true_mag=batch["magnitude_catalog"],
        sample_weights=sample_weights,
    )

    horizons = torch.as_tensor(
        HORIZONS,
        device=raw_rate.device,
        dtype=raw_rate.dtype,
    )
    supported_steps = torch.floor(
        torch.clamp(
            horizons.reshape(1, -1)
            - source_distance.reshape(-1, 1) / beta_m_per_s,
            min=0.0,
        )
        + 1.0e-12
    ).clamp(max=states.shape[2])
    information_weight = 0.05 + 0.95 * supported_steps / float(states.shape[2])
    target_encoded = batch["stf_log"].unsqueeze(1)
    mse_by_horizon = (encoded_states - target_encoded).pow(2).mean(dim=2)
    state_mw = moment_magnitude_from_rate(
        states.reshape(-1, states.shape[2]),
        source_dt.reshape(-1, 1).expand(-1, states.shape[1]).reshape(-1),
    ).reshape(states.shape[0], states.shape[1])
    magnitude_by_horizon = torch.abs(
        state_mw - batch["magnitude_catalog"].reshape(-1, 1)
    )
    sequence_per_sample = torch.sum(
        information_weight * (mse_by_horizon + magnitude_by_horizon),
        dim=1,
    ) / information_weight.sum(dim=1).clamp_min(1.0e-12)
    sequence_target = _weighted_sample_mean(sequence_per_sample, sample_weights)

    raw_endpoint_encoded = encode_rate(raw_rate[:, -1], stf_m_ref=stf_m_ref)
    teacher_per_sample = (
        encoded_states[:, -1] - raw_endpoint_encoded
    ).pow(2).mean(dim=1)
    endpoint_teacher = _weighted_sample_mean(teacher_per_sample, sample_weights)

    late_start = HORIZONS.index(LATE_HORIZONS[0])
    late_states = states[:, late_start:]
    late_mw = state_mw[:, late_start:]
    late_delta = late_mw[:, 1:] - late_mw[:, :-1]
    step_per_sample = F.smooth_l1_loss(
        late_delta,
        torch.zeros_like(late_delta),
        beta=STEP_HUBER_BETA_MW,
        reduction="none",
    ).mean(dim=1)
    late_step = _weighted_sample_mean(step_per_sample, sample_weights)

    dt = source_dt.reshape(-1, 1, 1)
    cumulative_log = torch.log10(
        torch.cumsum(late_states * dt, dim=2).clamp_min(1.0e10)
    )
    confirmed_delta = cumulative_log[:, 1:] - cumulative_log[:, :-1]
    previous_horizons = torch.as_tensor(
        LATE_HORIZONS[:-1],
        device=raw_rate.device,
        dtype=raw_rate.dtype,
    )
    confirmed_steps = torch.floor(
        torch.clamp(
            previous_horizons.reshape(1, -1)
            - source_distance.reshape(-1, 1) / beta_m_per_s,
            min=0.0,
        )
        + 1.0e-12
    ).to(torch.long).clamp(max=states.shape[2])
    source_index = torch.arange(
        states.shape[2],
        device=raw_rate.device,
    ).reshape(1, 1, -1)
    confirmed_mask = source_index < confirmed_steps.unsqueeze(2)
    confirmed_per_bin = F.smooth_l1_loss(
        confirmed_delta,
        torch.zeros_like(confirmed_delta),
        beta=CONFIRMED_HUBER_BETA_LOG10,
        reduction="none",
    )
    transition_loss = (
        (confirmed_per_bin * confirmed_mask).sum(dim=2)
        / confirmed_steps.clamp_min(1)
    )
    comparable = confirmed_steps > 0
    confirmed_per_sample = (
        (transition_loss * comparable).sum(dim=1)
        / comparable.sum(dim=1).clamp_min(1)
    )
    confirmed_history = _weighted_sample_mean(
        confirmed_per_sample,
        sample_weights,
    )

    components = {
        "endpoint_science": endpoint_science,
        "sequence_target": sequence_target,
        "endpoint_teacher": endpoint_teacher,
        "late_step": late_step,
        "confirmed_history": confirmed_history,
    }
    diagnostics = {
        "mean_gate": float(gates.detach().mean().cpu()),
        "late_mean_gate": float(gates[:, late_start:].detach().mean().cpu()),
        "endpoint_L_MSE": float(endpoint_parts["L_MSE"]),
        "endpoint_L_synth": float(endpoint_parts["L_synth"]),
        "endpoint_L_mag": float(endpoint_parts["L_mag"]),
    }
    return components, diagnostics


def normalized_adapter_loss(
    components: Mapping[str, torch.Tensor],
    normalizers: Mapping[str, float],
) -> torch.Tensor:
    missing_components = set(LOSS_WEIGHTS) - set(components)
    missing_normalizers = set(LOSS_WEIGHTS) - set(normalizers)
    if missing_components or missing_normalizers:
        raise ValueError(
            "missing normalized adapter loss inputs: "
            f"components={sorted(missing_components)}, "
            f"normalizers={sorted(missing_normalizers)}"
        )
    total: torch.Tensor | None = None
    for name, weight in LOSS_WEIGHTS.items():
        normalizer = float(normalizers[name])
        if not math.isfinite(normalizer) or normalizer <= 0.0:
            raise ValueError(f"invalid loss normalizer for {name}")
        term = float(weight) * components[name] / normalizer
        total = term if total is None else total + term
    if total is None:  # pragma: no cover - LOSS_WEIGHTS is frozen nonempty
        raise AssertionError("adapter objective is empty")
    return total


def _adapter_parameter_count(adapter: StreamingSTFAdapter) -> int:
    count = sum(parameter.numel() for parameter in adapter.parameters())
    if count != EXPECTED_ADAPTER_PARAMETER_COUNT:
        raise ValueError(
            f"Phase43 adapter parameter count changed: "
            f"{count} != {EXPECTED_ADAPTER_PARAMETER_COUNT}"
        )
    return count


def audit_loss_scales(
    *,
    cache_root: Path,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    _validate_new_directory(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_root)
    config = load_frozen_config()
    configure_runtime(42, device)
    torch.manual_seed(42)
    adapter = StreamingSTFAdapter(stf_m_ref=stf_m_ref_from_config(config)).to(device)
    _adapter_parameter_count(adapter)
    criterion = _build_stf_rate_criterion(config, device)
    train_indices = np.flatnonzero(cache.arrays["split_code"] == 0)
    component_sums = {name: 0.0 for name in LOSS_WEIGHTS}
    sample_count = 0
    first_batch: dict[str, torch.Tensor] | None = None
    for batch_index, start in enumerate(range(0, len(train_indices), BATCH_SIZE)):
        if batch_index >= NORMALIZER_BATCHES:
            break
        indices = train_indices[start : start + BATCH_SIZE]
        batch = _tensor_batch(cache, indices, device=device)
        if first_batch is None:
            first_batch = batch
        with torch.no_grad():
            components, _ = adapter_loss_components(
                adapter,
                batch,
                criterion=criterion,
                config=config,
            )
        batch_size = len(indices)
        sample_count += batch_size
        for name, value in components.items():
            component_sums[name] += float(value.detach().cpu()) * batch_size
    if sample_count == 0 or first_batch is None:
        raise ValueError("training cache produced no normalizer samples")
    normalizers = {
        name: max(component_sums[name] / sample_count, NORMALIZER_FLOORS[name])
        for name in LOSS_WEIGHTS
    }

    components, diagnostics = adapter_loss_components(
        adapter,
        first_batch,
        criterion=criterion,
        config=config,
    )
    parameters = tuple(adapter.parameters())
    gradient_norms: dict[str, float] = {}
    for component_index, name in enumerate(LOSS_WEIGHTS):
        gradients = torch.autograd.grad(
            components[name] / normalizers[name],
            parameters,
            retain_graph=component_index < len(LOSS_WEIGHTS) - 1,
            allow_unused=True,
        )
        squared = sum(
            float(torch.sum(gradient.detach().pow(2)).cpu())
            for gradient in gradients
            if gradient is not None
        )
        gradient_norms[name] = math.sqrt(squared)

    payload = {
        "status": "loss_scale_audit_complete",
        "objective_profile": ACTIVE_OBJECTIVE_PROFILE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(cache_root),
        "cache_raw_rates_sha256": cache.manifest["raw_rates_sha256"],
        "cache_arrays_sha256": cache.manifest["arrays_sha256"],
        "normalizer_batches": NORMALIZER_BATCHES,
        "normalizer_sample_count": sample_count,
        "normalizers": normalizers,
        "loss_weights": dict(LOSS_WEIGHTS),
        "normalized_gradient_norms_first_batch": gradient_norms,
        "initial_diagnostics_first_batch": diagnostics,
        "internal_test_iterated": False,
        "external_data_loaded": False,
        "grouped_test_loaded": False,
    }
    _write_json(output_root / "normalizers.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def load_normalizers(
    path: Path,
    *,
    cache: CacheBundle,
) -> tuple[dict[str, float], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing Phase43 normalizer artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "loss_scale_audit_complete":
        raise ValueError("Phase43 loss-scale audit is incomplete")
    if payload.get("objective_profile") != ACTIVE_OBJECTIVE_PROFILE:
        raise ValueError("Phase43 loss-scale objective profile changed")
    if payload["cache_raw_rates_sha256"] != cache.manifest["raw_rates_sha256"]:
        raise ValueError("Phase43 normalizers use a different raw-rate cache")
    if payload["cache_arrays_sha256"] != cache.manifest["arrays_sha256"]:
        raise ValueError("Phase43 normalizers use different cache arrays")
    if payload["loss_weights"] != LOSS_WEIGHTS:
        raise ValueError("Phase43 normalized objective weights changed")
    normalizers = {name: float(payload["normalizers"][name]) for name in LOSS_WEIGHTS}
    if any(not math.isfinite(value) or value <= 0.0 for value in normalizers.values()):
        raise ValueError("Phase43 loss normalizers must be positive and finite")
    return normalizers, payload


def evaluate_adapter(
    adapter: StreamingSTFAdapter,
    cache: CacheBundle,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    validation_indices = np.flatnonzero(cache.arrays["split_code"] == 1)
    if len(validation_indices) != EXPECTED_VALIDATION_COUNT:
        raise ValueError("Phase43 validation cache count changed")
    late_start = HORIZONS.index(LATE_HORIZONS[0])
    selected_indices: list[int] = []
    late_rates: list[np.ndarray] = []
    gate_sum = 0.0
    late_gate_sum = 0.0
    gate_count = 0
    late_gate_count = 0
    adapter.eval()
    with torch.no_grad():
        for batch_index, start in enumerate(
            range(0, len(validation_indices), BATCH_SIZE)
        ):
            if max_batches is not None and batch_index >= max_batches:
                break
            indices = validation_indices[start : start + BATCH_SIZE]
            batch = _tensor_batch(cache, indices, device=device)
            states, gates = adapter(
                batch["raw_rate"],
                horizons_sec=HORIZONS,
                source_distance_m=batch["source_distance_m"],
                source_dt_sec=batch["source_dt_sec"],
                beta_m_per_s=float(config["physics"]["beta"]),
            )
            late_rates.append(
                states[:, late_start:].detach().cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
            )
            selected_indices.extend(int(value) for value in indices)
            gate_sum += float(gates.detach().sum().cpu())
            gate_count += int(gates.numel())
            late_gate = gates[:, late_start:]
            late_gate_sum += float(late_gate.detach().sum().cpu())
            late_gate_count += int(late_gate.numel())
    if not late_rates:
        raise ValueError("Phase43 validation cache produced no batches")
    rate_cube = np.concatenate(late_rates, axis=0)
    selected = np.asarray(selected_indices, dtype=np.int64)
    metrics = _late_metrics_from_rates(
        rate_cube,
        events=[str(cache.records[index]["event"]) for index in selected],
        catalogs=np.asarray(cache.arrays["magnitude_catalog"][selected]),
        source_distance_m=np.asarray(cache.arrays["source_distance_m"][selected]),
        source_dt_sec=np.asarray(cache.arrays["source_dt_sec"][selected]),
        beta_m_per_s=float(config["physics"]["beta"]),
    )
    return metrics, {
        "validation_mean_gate": gate_sum / max(gate_count, 1),
        "validation_late_mean_gate": late_gate_sum / max(late_gate_count, 1),
    }


def _seed_protocol(
    *,
    cache: CacheBundle,
    normalizer_path: Path,
    normalizers: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "source_model": "Phase39 Glehman scalar + global invariant, seed42",
        "source_checkpoint": str(PHASE39_CHECKPOINT),
        "source_checkpoint_sha256": cache.manifest["source_artifact_sha256"][
            "checkpoint"
        ],
        "adapter_class": "StreamingSTFAdapter",
        "adapter_parameter_count": EXPECTED_ADAPTER_PARAMETER_COUNT,
        "adapter_seeds": list(ADAPTER_SEEDS),
        "objective_profile": ACTIVE_OBJECTIVE_PROFILE,
        "horizons": list(HORIZONS),
        "processing_delay_sec": PROCESSING_DELAY_SEC,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "step_huber_beta_mw": STEP_HUBER_BETA_MW,
        "confirmed_huber_beta_log10": CONFIRMED_HUBER_BETA_LOG10,
        "loss_weights": dict(LOSS_WEIGHTS),
        "loss_normalizers": dict(normalizers),
        "loss_normalizer_path": str(normalizer_path),
        "loss_normalizer_sha256": sha256_file(normalizer_path),
        "cache_root": str(cache.root),
        "cache_raw_rates_sha256": cache.manifest["raw_rates_sha256"],
        "cache_arrays_sha256": cache.manifest["arrays_sha256"],
        "validation_gates": dict(GATES),
        "selection": (
            "within each seed choose the endpoint-preserving epoch with the "
            "lowest worst normalized stability ratio; then choose one seed by "
            "the same validation score; never ensemble"
        ),
        "hidden_data": (
            "internal test, external development events, and grouped test are not iterated"
        ),
    }


def train_adapter_seed(
    *,
    seed: int,
    seed_root: Path,
    cache: CacheBundle,
    normalizers: Mapping[str, float],
    config: dict[str, Any],
    device: torch.device,
    smoke: bool,
) -> dict[str, Any]:
    seed_root.mkdir(parents=True, exist_ok=True)
    configure_runtime(seed, device)
    torch.manual_seed(seed)
    adapter = StreamingSTFAdapter(stf_m_ref=stf_m_ref_from_config(config)).to(device)
    parameter_count = _adapter_parameter_count(adapter)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in adapter.state_dict().items()
    }
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = _build_stf_rate_criterion(config, device)
    train_indices = np.flatnonzero(cache.arrays["split_code"] == 0)
    if len(train_indices) != EXPECTED_TRAIN_COUNT:
        raise ValueError("Phase43 training cache count changed")

    baseline_metrics, baseline_gate_metrics = evaluate_adapter(
        adapter,
        cache,
        config=config,
        device=device,
        max_batches=2 if smoke else None,
    )
    best_metrics = dict(baseline_metrics)
    best_gate = None if smoke else validation_gate(best_metrics)
    best_score = (
        float("inf")
        if best_gate is None
        else float(best_gate["selection_score"])
    )
    best_epoch = 0
    atomic_torch_save(dict(adapter.state_dict()), seed_root / "best_adapter.pth")

    rows: list[dict[str, Any]] = []
    epoch_count = 1 if smoke else EPOCHS
    max_train_batches = 2 if smoke else None
    for epoch in range(1, epoch_count + 1):
        adapter.train()
        generator = np.random.default_rng(seed * 10_000 + epoch)
        shuffled = generator.permutation(train_indices)
        seen = 0
        total_sum = 0.0
        component_sums = {name: 0.0 for name in LOSS_WEIGHTS}
        diagnostic_sums = {
            "mean_gate": 0.0,
            "late_mean_gate": 0.0,
            "endpoint_L_MSE": 0.0,
            "endpoint_L_synth": 0.0,
            "endpoint_L_mag": 0.0,
        }
        for batch_index, start in enumerate(range(0, len(shuffled), BATCH_SIZE)):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            indices = shuffled[start : start + BATCH_SIZE]
            batch = _tensor_batch(cache, indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            components, diagnostics = adapter_loss_components(
                adapter,
                batch,
                criterion=criterion,
                config=config,
            )
            total = normalized_adapter_loss(components, normalizers)
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("Phase43 training objective became non-finite")
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                adapter.parameters(),
                max_norm=GRAD_CLIP_NORM,
            )
            optimizer.step()
            batch_size = len(indices)
            seen += batch_size
            total_sum += float(total.detach().cpu()) * batch_size
            for name, value in components.items():
                component_sums[name] += float(value.detach().cpu()) * batch_size
            for name, value in diagnostics.items():
                diagnostic_sums[name] += float(value) * batch_size
        if seen == 0:
            raise ValueError("Phase43 training cache produced no samples")

        validation_metrics, validation_gate_metrics = evaluate_adapter(
            adapter,
            cache,
            config=config,
            device=device,
            max_batches=2 if smoke else None,
        )
        gate = None if smoke else validation_gate(validation_metrics)
        row = {
            "epoch": epoch,
            "train_total_normalized_loss": total_sum / seen,
            **{
                f"train_{name}": component_sums[name] / seen
                for name in LOSS_WEIGHTS
            },
            **{
                f"train_{name}": diagnostic_sums[name] / seen
                for name in diagnostic_sums
            },
            **validation_metrics,
            **validation_gate_metrics,
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
                    dict(adapter.state_dict()),
                    seed_root / "best_adapter.pth",
                )
        atomic_torch_save(dict(adapter.state_dict()), seed_root / "last_adapter.pth")
        _write_json(seed_root / "epoch_metrics.json", rows)
        _write_csv(
            seed_root / "epoch_metrics.csv",
            rows,
            fieldnames=tuple(rows[0]),
        )
        print(
            f"seed={seed} epoch={epoch}/{epoch_count} "
            f"train={row['train_total_normalized_loss']:.6f} "
            f"val_event={row['endpoint_event_mae']:.6f} "
            f"late_event_p95={row['late_event_abs_step_p95_mw']:.6f} "
            f"confirmed_p95="
            f"{row['late_confirmed_cumulative_log10_l1_p95']:.6f} "
            f"score={row['selection_score']}",
            flush=True,
        )

    if smoke:
        atomic_torch_save(dict(adapter.state_dict()), seed_root / "best_adapter.pth")
        selected_epoch = 1
        selected_metrics = rows[-1]
        selected_gate = None
        passed = True
        status = "smoke_complete"
    else:
        selected_epoch = best_epoch
        selected_metrics = best_metrics
        selected_gate = best_gate
        passed = bool(best_gate and best_gate["passed"] and best_epoch > 0)
        status = "validation_gate_passed" if passed else "validation_gate_failed"

    changed_tensors = [
        name
        for name, value in adapter.state_dict().items()
        if not torch.equal(value.detach().cpu(), initial_state[name])
    ]
    if not changed_tensors:
        raise RuntimeError("Phase43 adapter training changed no tensors")
    summary = {
        "seed": seed,
        "status": status,
        "passed": passed,
        "smoke": smoke,
        "adapter_parameter_count": parameter_count,
        "changed_adapter_tensor_count": len(changed_tensors),
        "changed_adapter_tensors": changed_tensors,
        "selected_epoch": selected_epoch,
        "best_adapter": {
            "path": str(seed_root / "best_adapter.pth"),
            "sha256": sha256_file(seed_root / "best_adapter.pth"),
        },
        "baseline_adapter_metrics": baseline_metrics,
        "baseline_adapter_gate_metrics": baseline_gate_metrics,
        "selected_metrics": selected_metrics,
        "selected_gate": selected_gate,
        "phase39_checkpoint_sha256": cache.manifest["source_artifact_sha256"][
            "checkpoint"
        ],
        "phase39_weights_trained": False,
        "internal_test_iterated": False,
        "external_data_loaded": False,
        "grouped_test_loaded": False,
    }
    _write_json(seed_root / "summary.json", summary)
    return summary


def run_training(
    *,
    cache_root: Path,
    normalizer_path: Path,
    output_root: Path,
    device: torch.device,
    smoke: bool,
) -> dict[str, Any]:
    _validate_new_directory(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_root)
    normalizers, normalizer_payload = load_normalizers(
        normalizer_path,
        cache=cache,
    )
    config = load_frozen_config()
    protocol = _seed_protocol(
        cache=cache,
        normalizer_path=normalizer_path,
        normalizers=normalizers,
    )
    protocol["smoke"] = smoke
    _write_json(output_root / "protocol.json", protocol)
    seeds = (42,) if smoke else ADAPTER_SEEDS
    seed_summaries = [
        train_adapter_seed(
            seed=seed,
            seed_root=output_root / f"seed_{seed}",
            cache=cache,
            normalizers=normalizers,
            config=config,
            device=device,
            smoke=smoke,
        )
        for seed in seeds
    ]

    if smoke:
        selected = seed_summaries[0]
    else:
        passing = [summary for summary in seed_summaries if summary["passed"]]
        selected = (
            min(
                passing,
                key=lambda item: float(item["selected_gate"]["selection_score"]),
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
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "git_dirty": git_is_dirty(PROJECT_ROOT),
        "device": str(device),
        "cache_manifest": str(cache.root / "manifest.json"),
        "normalizer_artifact": str(normalizer_path),
        "normalizer_artifact_sha256": sha256_file(normalizer_path),
        "phase39_checkpoint_sha256": cache.manifest["source_artifact_sha256"][
            "checkpoint"
        ],
        "phase39_weights_trained": False,
        "internal_test_iterated": False,
        "external_data_loaded": False,
        "grouped_test_loaded": False,
    }
    summary = {
        "status": status,
        "passed": True if smoke else selected is not None,
        "smoke": smoke,
        "selected_seed": None if selected is None else int(selected["seed"]),
        "selected_epoch": None if selected is None else int(selected["selected_epoch"]),
        "selected_summary": selected,
        "seed_summaries": seed_summaries,
        "protocol": protocol,
        "normalizer_audit": normalizer_payload,
        "provenance": provenance,
    }
    _write_json(output_root / "provenance.json", provenance)
    _write_json(output_root / "summary.json", summary)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "status": status,
                "passed": summary["passed"],
                "selected_seed": summary["selected_seed"],
                "selected_epoch": summary["selected_epoch"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen Phase39 raw-prefix cache and train the Phase43 "
            "causal streaming STF adapter."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("cache", "finalize-cache", "audit", "smoke", "train"),
        required=True,
    )
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--normalizers", type=Path)
    parser.add_argument(
        "--objective-profile",
        choices=tuple(OBJECTIVE_PROFILES),
        default="phase43_balanced",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    select_objective_profile(str(args.objective_profile))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    cache_root = args.cache_root.resolve()
    if args.stage == "cache":
        build_prefix_cache(cache_root=cache_root, device=device)
        return 0
    if args.stage == "finalize-cache":
        finalize_existing_cache(cache_root=cache_root, device=device)
        return 0
    if args.output_root is None:
        raise SystemExit(f"--output-root is required for stage {args.stage}")
    output_root = args.output_root.resolve()
    if args.stage == "audit":
        audit_loss_scales(
            cache_root=cache_root,
            output_root=output_root,
            device=device,
        )
        return 0
    if args.normalizers is None:
        raise SystemExit(f"--normalizers is required for stage {args.stage}")
    run_training(
        cache_root=cache_root,
        normalizer_path=args.normalizers.resolve(),
        output_root=output_root,
        device=device,
        smoke=args.stage == "smoke",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
