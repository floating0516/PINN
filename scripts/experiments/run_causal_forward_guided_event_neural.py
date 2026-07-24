from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_corrected_matrix import EXTERNAL_EVENT_NAMES  # noqa: E402
from src.data.external_records import record_from_external_bundle  # noqa: E402
from src.data.loaders_v2 import get_data_loaders_v2  # noqa: E402
from src.data.sample_builder import SampleRejected, build_station_sample  # noqa: E402
from src.data.waveform import waveform_config_from_v2  # noqa: E402
from src.evaluation.evaluate_unseen import load_event_bundle  # noqa: E402
from src.models.causal_event_magnitude import (  # noqa: E402
    CausalRadialStationObservation,
    build_causal_event_snapshot,
    causal_event_feature_names,
    causal_running_peak_cm,
    select_single_seed,
)
from src.models.causal_forward_guided import (  # noqa: E402
    CausalForwardGuidedEventNet,
    CausalForwardGuidedSpec,
)
from src.training.loss_stf_rate_v2 import (  # noqa: E402
    CausalEventSTFRateWaveformLossV2,
)
from src.utils.config_v2 import stf_m_ref_from_config, validate_config_v2  # noqa: E402
from src.utils.device import configure_runtime  # noqa: E402
from src.utils.provenance import (  # noqa: E402
    current_git_commit,
    git_is_dirty,
    sha256_file,
)


REQUIRED_SEEDS = (17, 42, 73)
SPLIT_NAMES = ("train", "validation", "test")
REFERENCE_DOI = "10.1029/2025JB033222"


@dataclass(frozen=True)
class StationTrace:
    observation: CausalRadialStationObservation
    radial_m: np.ndarray
    valid_mask: np.ndarray
    metadata: np.ndarray
    source_distance_m: float
    theta_deg: float
    phi_slip_deg: float
    observation_dt_sec: float


@dataclass(frozen=True)
class EventTrace:
    event: str
    magnitude: float
    stations: tuple[StationTrace, ...]
    stf_rate: np.ndarray | None
    stf_encoded: np.ndarray | None


@dataclass(frozen=True)
class SnapshotExample:
    event: str
    horizon_step: int
    horizon_sec: float
    online_features: np.ndarray
    active_station_count: int
    used_stations: tuple[str, ...]
    observed_steps: int
    radial_m: np.ndarray
    valid_mask: np.ndarray
    metadata: np.ndarray
    source_distance_m: np.ndarray
    theta_deg: np.ndarray
    phi_slip_deg: np.ndarray
    observation_dt_sec: np.ndarray
    station_mask: np.ndarray
    magnitude: float
    stf_rate: np.ndarray | None
    stf_encoded: np.ndarray | None


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize an empty CSV")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().replace("\r\n", "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _prepare_output_root(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    if not path.is_dir():
        raise FileExistsError(f"output root is not a directory: {path}")
    unexpected = [
        item.name
        for item in path.iterdir()
        if item.name != "console.log" or not item.is_file()
    ]
    if unexpected:
        raise FileExistsError(
            f"output root is not empty: {path}; unexpected={sorted(unexpected)}"
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _runtime_base_config(
    base_config_path: Path,
    *,
    seed: int,
    loss_weights: Mapping[str, float],
) -> dict[str, Any]:
    config = _load_yaml(base_config_path)
    config["dataset"]["radial_peak_min_cm"] = 0.0
    config["training"]["random_seed"] = int(seed)
    config["training"]["event_balanced_sampling"] = False
    config["training"]["num_workers"] = 0
    config["training"]["stf_rate_loss"].update(
        {
            "lambda_MSE": float(loss_weights["lambda_MSE"]),
            "lambda_synth": float(loss_weights["lambda_synth"]),
            "lambda_mag": float(loss_weights["lambda_mag"]),
            "lambda_shape": float(loss_weights["lambda_shape"]),
            "include_intermediate_field": False,
            "radiation_pattern_mode": "full",
        }
    )
    config.get("paths", {}).pop("dataset_manifest_path", None)
    validate_config_v2(config)
    return config


def _expected_causal_latency(config: Mapping[str, Any]) -> int:
    dataset = config["dataset"]
    sample_rate = float(dataset["sample_rate_hz"])
    interpolation = int(
        math.ceil(
            float(dataset["waveform"]["max_interpolation_gap_sec"])
            * sample_rate
        )
    )
    filter_config = dataset["filter"]
    if str(filter_config["type"]) == "none":
        filter_half_window = 0
    elif str(filter_config["type"]) == "lowpass":
        filter_half_window = (int(filter_config["num_taps"]) - 1) // 2
    else:
        raise ValueError("unsupported waveform filter for causal replay")
    return interpolation + filter_half_window


def _validate_experiment_config(payload: Mapping[str, Any]) -> None:
    if payload.get("method") != "causal_forward_guided_event_neural_v2":
        raise ValueError("unexpected experiment method")
    seeds = tuple(int(value) for value in payload["training"]["seeds"])
    if seeds != REQUIRED_SEEDS:
        raise ValueError(f"training seeds must be {REQUIRED_SEEDS}")
    selection = payload["selection"]
    if selection.get("primary_metric") != "validation_online_mae":
        raise ValueError("primary seed metric must be validation_online_mae")
    if selection.get("tie_break_metric") != "validation_final_mae":
        raise ValueError("seed tie-break must be validation_final_mae")
    if bool(selection.get("use_ensemble", True)):
        raise ValueError("formal result forbids seed ensembling")
    loss = payload["loss"]
    expected = {
        "lambda_MSE": 1.0,
        "lambda_mag": 1.0,
        "lambda_shape": 0.1,
    }
    for name, value in expected.items():
        if not math.isclose(float(loss[name]), value, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"{name} must preserve the original weight {value}")
    synth = float(loss["lambda_synth"])
    if synth not in {0.0, 0.5}:
        raise ValueError("lambda_synth must be 0.5 or the exact 0.0 ablation")
    if str(payload["reference"]["doi"]) != REFERENCE_DOI:
        raise ValueError("forward-model reference DOI is not frozen")
    horizons = tuple(int(value) for value in payload["training"]["horizons_sec"])
    if not horizons or horizons != tuple(sorted(set(horizons))):
        raise ValueError("training horizons must be sorted and unique")
    if horizons[-1] != int(payload["target"]["final_horizon_sec"]):
        raise ValueError("training horizons must include the final horizon")


def _positive_scale(values: np.ndarray) -> np.ndarray:
    scale = np.asarray(values, dtype=np.float64).std(axis=0)
    scale = np.asarray(scale)
    scale[scale < 1.0e-6] = 1.0
    return scale.astype(np.float32)


def _station_metadata(sample: Mapping[str, Any]) -> np.ndarray:
    distance = max(float(sample["source_distance_m"]), 1.0)
    theta = math.radians(float(sample["theta_deg"]))
    azimuth = math.radians(float(sample["azimuth_deg"]))
    return np.asarray(
        [
            math.log(distance),
            math.sin(theta),
            math.cos(theta),
            math.sin(azimuth),
            math.cos(azimuth),
        ],
        dtype=np.float32,
    )


def _station_trace(
    sample: Mapping[str, Any],
    *,
    spec: CausalForwardGuidedSpec,
) -> StationTrace:
    radial = np.asarray(sample["radial"], dtype=np.float32)
    valid_mask = np.asarray(sample["waveform_valid_mask"], dtype=bool)
    if radial.shape != (spec.total_steps,) or valid_mask.shape != radial.shape:
        raise ValueError("station waveform does not match the causal duration")
    p_arrival = float(sample.get("p_arrival_sec", float("nan")))
    if not math.isfinite(p_arrival):
        raise ValueError("station sample lacks a finite P arrival")
    observation = CausalRadialStationObservation(
        event=str(sample["event"]),
        station=str(sample["station"]),
        running_peak_cm=causal_running_peak_cm(
            radial,
            valid_mask,
            latency_samples=spec.causal_latency_samples,
        ),
        source_distance_km=float(sample["source_distance_m"]) / 1000.0,
        p_arrival_sec=p_arrival,
        magnitude=float(sample["magnitude_catalog"]),
    )
    return StationTrace(
        observation=observation,
        radial_m=radial.copy(),
        valid_mask=valid_mask.copy(),
        metadata=_station_metadata(sample),
        source_distance_m=float(sample["source_distance_m"]),
        theta_deg=float(sample["theta_deg"]),
        phi_slip_deg=float(sample["phi_slip_deg"]),
        observation_dt_sec=float(sample["waveform_dt_sec"]),
    )


def _subset_base_and_samples(loader: Any) -> tuple[Any, list[dict[str, Any]]]:
    subset = loader.dataset
    if not hasattr(subset, "indices") or not hasattr(subset, "dataset"):
        raise TypeError("runner requires a deterministic Subset loader")
    base = subset.dataset
    if not hasattr(base, "samples") or not hasattr(base, "_event_stf"):
        raise TypeError("runner requires CorrectedEarthquakeDataset")
    samples = [base.samples[int(index)] for index in subset.indices]
    return base, samples


def _event_traces_from_loader(
    loader: Any,
    *,
    spec: CausalForwardGuidedSpec,
    stf_m_ref: float,
) -> dict[str, EventTrace]:
    base, samples = _subset_base_and_samples(loader)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["event"])].append(sample)
    events: dict[str, EventTrace] = {}
    for event in sorted(grouped):
        event_samples = grouped[event]
        magnitudes = {float(sample["magnitude_catalog"]) for sample in event_samples}
        if len(magnitudes) != 1:
            raise ValueError(f"event magnitude is inconsistent: {event}")
        match = base._event_stf(event)
        if match is None:
            raise ValueError(f"event lacks the required SCARDEC STF: {event}")
        processed, _ = match
        rate = np.asarray(processed.rate_nm_per_s, dtype=np.float32)
        if rate.shape != (spec.total_steps,):
            raise ValueError(f"origin STF duration mismatch: {event}")
        encoded = np.log10(1.0 + np.maximum(rate, 0.0) / stf_m_ref).astype(
            np.float32
        )
        events[event] = EventTrace(
            event=event,
            magnitude=next(iter(magnitudes)),
            stations=tuple(
                _station_trace(sample, spec=spec) for sample in event_samples
            ),
            stf_rate=rate.copy(),
            stf_encoded=encoded,
        )
    return events


def _snapshot_example(
    event: EventTrace,
    *,
    horizon_step: int,
    spec: CausalForwardGuidedSpec,
) -> SnapshotExample | None:
    observed_steps = int(horizon_step) - int(spec.causal_latency_samples)
    if observed_steps < 1:
        return None
    snapshot = build_causal_event_snapshot(
        [station.observation for station in event.stations],
        horizon_step=horizon_step,
        spec=spec.event_spec,
    )
    if snapshot is None:
        return None
    by_station = {
        station.observation.station: station for station in event.stations
    }
    selected = [by_station[name] for name in snapshot.used_stations]
    top_k = int(spec.top_k)
    steps = int(spec.total_steps)
    radial = np.zeros((top_k, steps), dtype=np.float32)
    valid = np.zeros((top_k, steps), dtype=bool)
    metadata = np.zeros((top_k, 5), dtype=np.float32)
    distance = np.ones(top_k, dtype=np.float32)
    theta = np.zeros(top_k, dtype=np.float32)
    phi = np.zeros(top_k, dtype=np.float32)
    observation_dt = np.ones(top_k, dtype=np.float32)
    station_mask = np.zeros(top_k, dtype=bool)
    for index, station in enumerate(selected):
        radial[index] = station.radial_m
        valid[index] = station.valid_mask
        metadata[index] = station.metadata
        distance[index] = station.source_distance_m
        theta[index] = station.theta_deg
        phi[index] = station.phi_slip_deg
        observation_dt[index] = station.observation_dt_sec
        station_mask[index] = True
    return SnapshotExample(
        event=event.event,
        horizon_step=int(horizon_step),
        horizon_sec=float(snapshot.horizon_sec),
        online_features=np.asarray(snapshot.features, dtype=np.float32),
        active_station_count=int(snapshot.active_station_count),
        used_stations=tuple(snapshot.used_stations),
        observed_steps=observed_steps,
        radial_m=radial,
        valid_mask=valid,
        metadata=metadata,
        source_distance_m=distance,
        theta_deg=theta,
        phi_slip_deg=phi,
        observation_dt_sec=observation_dt,
        station_mask=station_mask,
        magnitude=float(event.magnitude),
        stf_rate=(None if event.stf_rate is None else event.stf_rate.copy()),
        stf_encoded=(
            None if event.stf_encoded is None else event.stf_encoded.copy()
        ),
    )


def _build_examples(
    events: Mapping[str, EventTrace],
    *,
    horizons: Iterable[int],
    spec: CausalForwardGuidedSpec,
) -> list[SnapshotExample]:
    rows: list[SnapshotExample] = []
    horizon_values = tuple(int(value) for value in horizons)
    for event_name in sorted(events):
        event = events[event_name]
        for horizon in horizon_values:
            example = _snapshot_example(event, horizon_step=horizon, spec=spec)
            if example is not None:
                rows.append(example)
    if not rows:
        raise ValueError("snapshot set is empty")
    return rows


def _collate_examples(
    examples: Sequence[SnapshotExample],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty batch")

    def tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
        values = np.asarray([getattr(example, name) for example in examples])
        return torch.as_tensor(values, dtype=dtype, device=device)

    batch = {
        "online_features": tensor("online_features", torch.float32),
        "observed_steps": tensor("observed_steps", torch.long),
        "radial_m": tensor("radial_m", torch.float32),
        "valid_mask": tensor("valid_mask", torch.bool),
        "metadata": tensor("metadata", torch.float32),
        "source_distance_m": tensor("source_distance_m", torch.float32),
        "theta_deg": tensor("theta_deg", torch.float32),
        "phi_slip_deg": tensor("phi_slip_deg", torch.float32),
        "observation_dt_sec": tensor("observation_dt_sec", torch.float32),
        "station_mask": tensor("station_mask", torch.bool),
        "magnitude": tensor("magnitude", torch.float32),
    }
    if all(example.stf_rate is not None for example in examples):
        batch["stf_rate"] = tensor("stf_rate", torch.float32)
        batch["stf_encoded"] = tensor("stf_encoded", torch.float32)
    return batch


def _released_valid_mask(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    time_steps = batch["radial_m"].shape[-1]
    time_index = torch.arange(time_steps, device=batch["radial_m"].device)
    released = time_index.reshape(1, 1, -1) < batch["observed_steps"].reshape(
        -1, 1, 1
    )
    return (
        batch["valid_mask"]
        & released
        & batch["station_mask"].unsqueeze(-1)
    )


def _predict_batch(
    model: CausalForwardGuidedEventNet,
    batch: Mapping[str, torch.Tensor],
) -> Any:
    return model(
        radial_m=batch["radial_m"],
        waveform_valid_mask=batch["valid_mask"],
        station_metadata=batch["metadata"],
        station_mask=batch["station_mask"],
        observed_steps=batch["observed_steps"],
        online_features=batch["online_features"],
    )


def _loss_batch(
    criterion: CausalEventSTFRateWaveformLossV2,
    prediction: Any,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    if "stf_rate" not in batch or "stf_encoded" not in batch:
        raise ValueError("training batch lacks an STF reference")
    return criterion(
        prediction.stf_encoded,
        pred_catalog_mw=prediction.catalog_mw,
        rate_ref_encoded=batch["stf_encoded"],
        rate_ref_physical=batch["stf_rate"],
        true_mag=batch["magnitude"],
        radial_obs=batch["radial_m"],
        source_distance_m=batch["source_distance_m"],
        theta_deg=batch["theta_deg"],
        phi_slip_deg=batch["phi_slip_deg"],
        source_dt_sec=torch.ones_like(batch["magnitude"]),
        observation_dt_sec=batch["observation_dt_sec"],
        waveform_valid_mask=_released_valid_mask(batch),
        station_mask=batch["station_mask"],
    )


def _batch_slices(
    row_count: int,
    batch_size: int,
    *,
    order: Sequence[int] | None = None,
) -> Iterable[list[int]]:
    indices = list(range(row_count)) if order is None else list(order)
    for start in range(0, row_count, batch_size):
        yield indices[start : start + batch_size]


def _predict_examples(
    model: CausalForwardGuidedEventNet,
    examples: Sequence[SnapshotExample],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    with torch.no_grad():
        for indices in _batch_slices(len(examples), batch_size):
            batch = _collate_examples([examples[index] for index in indices], device=device)
            output = _predict_batch(model, batch)
            predictions.append(output.catalog_mw.detach().cpu().numpy())
            anchors.append(output.anchor_mw.detach().cpu().numpy())
            residuals.append(output.magnitude_residual.detach().cpu().numpy())
    result = tuple(
        np.concatenate(parts).astype(np.float64)
        for parts in (predictions, anchors, residuals)
    )
    if any(values.shape != (len(examples),) for values in result):
        raise RuntimeError("prediction output shape mismatch")
    if not all(np.isfinite(values).all() for values in result):
        raise FloatingPointError("model produced non-finite predictions")
    return result


def _evaluate_with_loss(
    model: CausalForwardGuidedEventNet,
    criterion: CausalEventSTFRateWaveformLossV2,
    examples: Sequence[SnapshotExample],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    model.eval()
    predictions: list[np.ndarray] = []
    totals: dict[str, float] = defaultdict(float)
    count = 0
    with torch.no_grad():
        for indices in _batch_slices(len(examples), batch_size):
            selected = [examples[index] for index in indices]
            batch = _collate_examples(selected, device=device)
            output = _predict_batch(model, batch)
            _, metrics = _loss_batch(criterion, output, batch)
            predictions.append(output.catalog_mw.detach().cpu().numpy())
            for name, value in metrics.items():
                totals[name] += float(value) * len(selected)
            count += len(selected)
    return (
        np.concatenate(predictions).astype(np.float64),
        {name: value / count for name, value in totals.items()},
    )


def _online_metrics(
    examples: Sequence[SnapshotExample],
    predictions: np.ndarray,
) -> dict[str, Any]:
    if predictions.shape != (len(examples),):
        raise ValueError("online predictions do not match examples")
    grouped: dict[str, list[float]] = defaultdict(list)
    for example, prediction in zip(examples, predictions):
        grouped[example.event].append(float(prediction) - example.magnitude)
    return {
        "event_count": len(grouped),
        "snapshot_count": len(examples),
        "event_equal_online_mae": float(
            np.mean([np.mean(np.abs(values)) for values in grouped.values()])
        ),
        "event_equal_online_rmse": float(
            np.sqrt(np.mean([np.mean(np.square(values)) for values in grouped.values()]))
        ),
        "event_equal_online_bias": float(
            np.mean([np.mean(values) for values in grouped.values()])
        ),
    }


def _basic_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    if predictions.shape != targets.shape or predictions.size < 1:
        raise ValueError("metric arrays must be equal and non-empty")
    errors = predictions.astype(np.float64) - targets.astype(np.float64)
    return {
        "event_count": int(errors.size),
        "event_mae": float(np.mean(np.abs(errors))),
        "event_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "event_bias": float(np.mean(errors)),
    }


def _final_indices(
    examples: Sequence[SnapshotExample],
    total_steps: int,
) -> np.ndarray:
    indices = np.asarray(
        [
            index
            for index, example in enumerate(examples)
            if example.horizon_step == total_steps
        ],
        dtype=np.int64,
    )
    if indices.size < 2:
        raise ValueError("final snapshot set contains fewer than two events")
    return indices


def _prediction_rows(
    examples: Sequence[SnapshotExample],
    predictions: np.ndarray,
    anchors: np.ndarray,
    residuals: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example, prediction, anchor, residual in zip(
        examples, predictions, anchors, residuals
    ):
        error = float(prediction) - example.magnitude
        rows.append(
            {
                "event": example.event,
                "horizon_sec": example.horizon_sec,
                "mw_pred": float(prediction),
                "mw_reference": example.magnitude,
                "error": error,
                "abs_error": abs(error),
                "anchor_mw": float(anchor),
                "neural_residual_mw": float(residual),
                "active_station_count": example.active_station_count,
                "used_station_count": len(example.used_stations),
                "used_stations": "|".join(example.used_stations),
            }
        )
    return rows


def _horizon_metric_rows(
    examples: Sequence[SnapshotExample],
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[example.horizon_step].append(index)
    rows: list[dict[str, Any]] = []
    for horizon in sorted(grouped):
        indices = np.asarray(grouped[horizon], dtype=np.int64)
        targets = np.asarray(
            [examples[index].magnitude for index in indices], dtype=np.float64
        )
        rows.append(
            {
                "horizon_sec": examples[int(indices[0])].horizon_sec,
                **_basic_metrics(predictions[indices], targets),
            }
        )
    return rows


def _normalization_payload(
    train_events: Mapping[str, EventTrace],
    train_examples: Sequence[SnapshotExample],
    *,
    spec: CausalForwardGuidedSpec,
) -> dict[str, Any]:
    final = [
        example for example in train_examples if example.horizon_step == spec.total_steps
    ]
    if len(final) != len(train_events):
        raise ValueError("training set lacks one final snapshot per event")
    online = np.asarray(
        [example.online_features for example in train_examples], dtype=np.float32
    )
    anchors = np.asarray(
        [
            example.online_features[: spec.anchor_feature_count]
            for example in final
        ],
        dtype=np.float32,
    )
    metadata = np.concatenate(
        [
            np.asarray([station.metadata for station in event.stations], dtype=np.float32)
            for event in train_events.values()
        ],
        axis=0,
    )
    targets = np.asarray([example.magnitude for example in final], dtype=np.float32)
    stf_encoded = np.asarray(
        [event.stf_encoded for event in train_events.values()], dtype=np.float32
    )
    target_scale = float(targets.std())
    if target_scale < 1.0e-6:
        raise ValueError("training event targets have no variation")
    return {
        "anchor_feature_mean": anchors.mean(axis=0),
        "anchor_feature_scale": _positive_scale(anchors),
        "online_feature_mean": online.mean(axis=0),
        "online_feature_scale": _positive_scale(online),
        "metadata_mean": metadata.mean(axis=0),
        "metadata_scale": _positive_scale(metadata),
        "target_mean": float(targets.mean()),
        "target_scale": target_scale,
        "stf_encoded_mean": stf_encoded.mean(axis=0),
    }


def _train_anchor(
    model: CausalForwardGuidedEventNet,
    train_examples: Sequence[SnapshotExample],
    validation_examples: Sequence[SnapshotExample],
    *,
    training: Mapping[str, Any],
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]]]:
    train_final = [
        row for row in train_examples if row.horizon_step == model.spec.total_steps
    ]
    validation_final = [
        row
        for row in validation_examples
        if row.horizon_step == model.spec.total_steps
    ]
    train_x = torch.as_tensor(
        np.asarray([row.online_features for row in train_final]),
        dtype=torch.float32,
        device=device,
    )
    train_y = torch.as_tensor(
        np.asarray([row.magnitude for row in train_final]),
        dtype=torch.float32,
        device=device,
    )
    train_y_std = (train_y - model.target_mean) / model.target_scale
    validation_x = torch.as_tensor(
        np.asarray([row.online_features for row in validation_final]),
        dtype=torch.float32,
        device=device,
    )
    validation_y = np.asarray(
        [row.magnitude for row in validation_final], dtype=np.float64
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.anchor_branch.parameters():
        parameter.requires_grad_(True)
    epochs = int(training["anchor_epochs"])
    learning_rate = float(training["anchor_learning_rate"])
    optimizer = torch.optim.Adam(model.anchor_branch.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.01,
    )
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    logs: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        predicted_std = model.standardized_anchor(train_x)
        fit_loss = F.mse_loss(predicted_std, train_y_std)
        penalty = (
            float(training["anchor_l2_alpha"])
            / len(train_final)
            * model.anchor_branch.weight.square().sum()
        )
        loss = fit_loss + penalty
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.anchor_branch.parameters(),
            float(training["gradient_clip_norm"]),
        )
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError("anchor training became non-finite")
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            validation_prediction = (
                model.target_mean
                + model.target_scale * model.standardized_anchor(validation_x)
            ).detach().cpu().numpy()
        validation_mae = float(np.mean(np.abs(validation_prediction - validation_y)))
        logs.append(
            {
                "phase": "anchor",
                "epoch": epoch,
                "train_total_loss": float(loss.detach().cpu()),
                "train_L_MSE": "",
                "train_L_synth": "",
                "train_L_mag": float(fit_loss.detach().cpu()),
                "train_L_shape": "",
                "validation_total_loss": "",
                "validation_L_MSE": "",
                "validation_L_synth": "",
                "validation_L_mag": "",
                "validation_L_shape": "",
                "validation_online_mae": validation_mae,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_mae < best_mae:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.anchor_branch.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("anchor training produced no checkpoint")
    model.anchor_branch.load_state_dict(best_state, strict=True)
    return best_epoch, best_mae, logs


def _train_deep_branch(
    model: CausalForwardGuidedEventNet,
    criterion: CausalEventSTFRateWaveformLossV2,
    train_examples: Sequence[SnapshotExample],
    validation_examples: Sequence[SnapshotExample],
    *,
    seed: int,
    training: Mapping[str, Any],
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]]]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in model.anchor_branch.parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    epochs = int(training["deep_epochs"])
    learning_rate = float(training["deep_learning_rate"])
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=float(training["deep_weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.02,
    )
    batch_size = int(training["batch_size"])
    best_mae = float("inf")
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    logs: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_examples)))
        random.Random(seed * 10_000 + epoch).shuffle(order)
        totals: dict[str, float] = defaultdict(float)
        count = 0
        for indices in _batch_slices(len(train_examples), batch_size, order=order):
            selected = [train_examples[index] for index in indices]
            batch = _collate_examples(selected, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = _predict_batch(model, batch)
            loss, metrics = _loss_batch(criterion, prediction, batch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                float(training["gradient_clip_norm"]),
            )
            if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
                raise FloatingPointError("deep training became non-finite")
            optimizer.step()
            for name, value in metrics.items():
                totals[name] += float(value) * len(selected)
            count += len(selected)
        train_metrics = {name: value / count for name, value in totals.items()}
        validation_predictions, validation_metrics = _evaluate_with_loss(
            model,
            criterion,
            validation_examples,
            batch_size=batch_size,
            device=device,
        )
        validation_mae = float(
            _online_metrics(validation_examples, validation_predictions)[
                "event_equal_online_mae"
            ]
        )
        validation_loss = float(validation_metrics["L_total"])
        if not math.isfinite(validation_mae) or not math.isfinite(validation_loss):
            raise FloatingPointError(
                "deep validation became non-finite: "
                f"mae={validation_mae}, metrics={validation_metrics}"
            )
        logs.append(
            {
                "phase": "deep",
                "epoch": epoch,
                "train_total_loss": train_metrics["L_total"],
                "train_L_MSE": train_metrics["L_MSE"],
                "train_L_synth": train_metrics["L_synth"],
                "train_L_mag": train_metrics["L_mag"],
                "train_L_shape": train_metrics["L_shape"],
                "validation_total_loss": validation_metrics["L_total"],
                "validation_L_MSE": validation_metrics["L_MSE"],
                "validation_L_synth": validation_metrics["L_synth"],
                "validation_L_mag": validation_metrics["L_mag"],
                "validation_L_shape": validation_metrics["L_shape"],
                "validation_online_mae": validation_mae,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        candidate = (validation_mae, validation_loss, epoch)
        incumbent = (best_mae, best_loss, best_epoch)
        if candidate < incumbent:
            best_mae = validation_mae
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        scheduler.step()
    if best_state is None:
        raise RuntimeError("deep training produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    return best_epoch, best_mae, logs


def _load_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> CausalForwardGuidedEventNet:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("model_type") != "causal_forward_guided_event_neural_v2":
        raise ValueError("unexpected checkpoint model type")
    spec = CausalForwardGuidedSpec.from_dict(payload["spec"])
    state = payload["model_state_dict"]
    model = CausalForwardGuidedEventNet(
        anchor_feature_mean=state["anchor_feature_mean"],
        anchor_feature_scale=state["anchor_feature_scale"],
        online_feature_mean=state["online_feature_mean"],
        online_feature_scale=state["online_feature_scale"],
        metadata_mean=state["metadata_mean"],
        metadata_scale=state["metadata_scale"],
        target_mean=float(state["target_mean"]),
        target_scale=float(state["target_scale"]),
        stf_encoded_mean=state["stf_encoded_mean"],
        spec=spec,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _train_seed(
    *,
    seed: int,
    split_events: Mapping[str, Mapping[str, EventTrace]],
    split_manifest: Mapping[str, Any],
    spec: CausalForwardGuidedSpec,
    runtime_config: dict[str, Any],
    training: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    configure_runtime(seed, device)
    random.seed(seed)
    np.random.seed(seed)
    horizons = tuple(int(value) for value in training["horizons_sec"])
    split_training_examples = {
        name: _build_examples(
            split_events[name],
            horizons=horizons,
            spec=spec,
        )
        for name in SPLIT_NAMES
    }
    normalization = _normalization_payload(
        split_events["train"],
        split_training_examples["train"],
        spec=spec,
    )
    model = CausalForwardGuidedEventNet(
        **normalization,
        spec=spec,
    ).to(device)
    criterion = CausalEventSTFRateWaveformLossV2(runtime_config).to(device)
    anchor_epoch, anchor_mae, anchor_logs = _train_anchor(
        model,
        split_training_examples["train"],
        split_training_examples["validation"],
        training=training,
        device=device,
    )
    deep_epoch, sampled_validation_mae, deep_logs = _train_deep_branch(
        model,
        criterion,
        split_training_examples["train"],
        split_training_examples["validation"],
        seed=seed,
        training=training,
        device=device,
    )

    split_online_metrics: dict[str, dict[str, Any]] = {}
    split_final_metrics: dict[str, dict[str, Any]] = {}
    split_residual: dict[str, dict[str, float]] = {}
    for name in ("validation", "test"):
        examples = _build_examples(
            split_events[name],
            horizons=range(1, spec.total_steps + 1),
            spec=spec,
        )
        predictions, anchors, residuals = _predict_examples(
            model,
            examples,
            batch_size=int(training["evaluation_batch_size"]),
            device=device,
        )
        split_online_metrics[name] = _online_metrics(examples, predictions)
        final_indices = _final_indices(examples, spec.total_steps)
        targets = np.asarray(
            [examples[index].magnitude for index in final_indices],
            dtype=np.float64,
        )
        split_final_metrics[name] = _basic_metrics(
            predictions[final_indices], targets
        )
        split_residual[name] = {
            "mean_abs_mw": float(np.mean(np.abs(residuals))),
            "maximum_abs_mw": float(np.max(np.abs(residuals))),
            "final_mean_abs_mw": float(np.mean(np.abs(residuals[final_indices]))),
        }
        prediction_rows = _prediction_rows(
            examples, predictions, anchors, residuals
        )
        _atomic_write(
            output_dir / f"{name}_online_predictions.csv",
            _csv_bytes(prediction_rows),
        )
        _atomic_write(
            output_dir / f"{name}_horizon_metrics.csv",
            _csv_bytes(_horizon_metric_rows(examples, predictions)),
        )

    summary = {
        "seed": seed,
        "best_anchor_epoch": anchor_epoch,
        "best_anchor_validation_final_mae": anchor_mae,
        "best_deep_epoch": deep_epoch,
        "sampled_validation_online_mae": sampled_validation_mae,
        "validation_online_mae": split_online_metrics["validation"][
            "event_equal_online_mae"
        ],
        "validation_final_mae": split_final_metrics["validation"]["event_mae"],
        "split_online_metrics": split_online_metrics,
        "split_final_metrics": split_final_metrics,
        "split_neural_residual": split_residual,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_deep_parameter_count": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if not name.startswith("anchor_branch.")
        ),
        "split_assignment_sha256": split_manifest["assignment_sha256"],
    }
    checkpoint = {
        "model_type": "causal_forward_guided_event_neural_v2",
        "model_state_dict": model.state_dict(),
        "spec": spec.to_dict(),
        "feature_names": list(causal_event_feature_names(spec.event_spec)),
        "loss_weights": {
            name: float(getattr(criterion, name))
            for name in ("lambda_MSE", "lambda_synth", "lambda_mag", "lambda_shape")
        },
        "reference_doi": REFERENCE_DOI,
        "summary": summary,
    }
    _atomic_torch_save(output_dir / "best_model.pth", checkpoint)
    _atomic_write(
        output_dir / "training_log.csv",
        _csv_bytes(anchor_logs + deep_logs),
    )
    _atomic_write(output_dir / "split.json", _json_bytes(dict(split_manifest)))
    _atomic_write(output_dir / "summary.json", _json_bytes(summary))
    return summary


def _external_event_traces(
    *,
    event_root: Path,
    base_config: Mapping[str, Any],
    spec: CausalForwardGuidedSpec,
) -> tuple[dict[str, EventTrace], dict[str, int]]:
    waveform_config = waveform_config_from_v2(dict(base_config))
    alpha = float(base_config["physics"]["alpha"])
    grouped: dict[str, list[StationTrace]] = defaultdict(list)
    magnitudes: dict[str, float] = {}
    rejected = 0
    for directory_name in EXTERNAL_EVENT_NAMES:
        bundle = load_event_bundle(event_root / directory_name)
        magnitudes[str(bundle.event_name)] = float(bundle.magnitude)
        for station in bundle.stations:
            try:
                sample = build_station_sample(
                    record_from_external_bundle(bundle, station),
                    units="m",
                    waveform_config=waveform_config,
                    alpha_m_per_s=alpha,
                    radial_peak_min_cm=0.0,
                )
            except SampleRejected:
                rejected += 1
                continue
            sample["p_arrival_sec"] = float(sample["source_distance_m"]) / alpha
            grouped[str(bundle.event_name)].append(
                _station_trace(sample, spec=spec)
            )
    events = {
        event: EventTrace(
            event=event,
            magnitude=magnitudes[event],
            stations=tuple(stations),
            stf_rate=None,
            stf_encoded=None,
        )
        for event, stations in sorted(grouped.items())
    }
    return events, {
        "accepted_station_count": sum(len(event.stations) for event in events.values()),
        "rejected_station_count": rejected,
        "event_count": len(events),
    }


def _station_pool_sha256(split_manifest: Mapping[str, Any]) -> str:
    keys = sorted(
        str(key)
        for split_name in SPLIT_NAMES
        for key in split_manifest["sample_keys"][split_name]
    )
    payload = json.dumps(
        keys,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _external_input_hashes(event_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for directory_name in EXTERNAL_EVENT_NAMES:
        for filename in ("event.json", "stations.csv", "waveforms.csv.gz"):
            path = event_root / directory_name / filename
            if not path.is_file():
                raise FileNotFoundError(f"external input is missing: {path}")
            hashes[f"{directory_name}/{filename}"] = sha256_file(path)
    return hashes


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"console.log", "summary.json", "COMPLETE"}
    }


def run(
    *,
    config_path: Path,
    output_root: Path,
    device: torch.device,
    smoke: bool = False,
) -> dict[str, Any]:
    experiment = _load_yaml(config_path)
    _validate_experiment_config(experiment)
    _prepare_output_root(output_root)
    _atomic_write(output_root / "config.yaml", config_path.read_bytes())
    base_config_path = Path(experiment["base_config"]).resolve()
    event_root = Path(experiment["external_event_root"]).resolve()
    if not base_config_path.is_file():
        raise FileNotFoundError(f"base config not found: {base_config_path}")
    if not event_root.is_dir():
        raise FileNotFoundError(f"external event root not found: {event_root}")
    spec = CausalForwardGuidedSpec.from_dict(experiment["event_model"])
    loss_weights = dict(experiment["loss"])
    base_config = _runtime_base_config(
        base_config_path,
        seed=REQUIRED_SEEDS[0],
        loss_weights=loss_weights,
    )
    if spec.causal_latency_samples != _expected_causal_latency(base_config):
        raise ValueError("configured causal latency is not conservative")
    if spec.total_steps != int(experiment["target"]["final_horizon_sec"]):
        raise ValueError("model duration differs from the final horizon")
    expected_events = int(experiment["audit"]["expected_event_count"])
    expected_stations = int(experiment["audit"]["expected_station_count"])
    training = dict(experiment["training"])
    seeds = REQUIRED_SEEDS[:1] if smoke else REQUIRED_SEEDS
    if smoke:
        training["anchor_epochs"] = min(3, int(training["anchor_epochs"]))
        training["deep_epochs"] = min(2, int(training["deep_epochs"]))
        training["horizons_sec"] = [60, spec.total_steps]

    seed_summaries: dict[int, dict[str, Any]] = {}
    station_pool_sha256: str | None = None
    for seed in seeds:
        runtime = _runtime_base_config(
            base_config_path,
            seed=seed,
            loss_weights=loss_weights,
        )
        train_loader, validation_loader, test_loader, split_manifest = (
            get_data_loaders_v2(runtime)
        )
        total_stations = sum(
            int(split_manifest[f"{name}_record_count"]) for name in SPLIT_NAMES
        )
        if total_stations != expected_stations:
            raise ValueError(
                f"causal station pool has {total_stations}; expected {expected_stations}"
            )
        if len(split_manifest["train_events"]) != expected_events:
            raise ValueError("causal event count differs from the audit contract")
        current_pool = _station_pool_sha256(split_manifest)
        if station_pool_sha256 is None:
            station_pool_sha256 = current_pool
        elif current_pool != station_pool_sha256:
            raise RuntimeError("causal station pool changed between seeds")
        loaders = {
            "train": train_loader,
            "validation": validation_loader,
            "test": test_loader,
        }
        split_events = {
            name: _event_traces_from_loader(
                loaders[name],
                spec=spec,
                stf_m_ref=stf_m_ref_from_config(runtime),
            )
            for name in SPLIT_NAMES
        }
        seed_dir = output_root / f"seed_{seed}"
        seed_summaries[seed] = _train_seed(
            seed=seed,
            split_events=split_events,
            split_manifest=split_manifest,
            spec=spec,
            runtime_config=runtime,
            training=training,
            output_dir=seed_dir,
            device=device,
        )
        del split_events, loaders, train_loader, validation_loader, test_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected_seed = select_single_seed(seed_summaries)
    selection = {
        "selected_seed": selected_seed,
        "selection_metric": "validation_online_mae",
        "tie_break_metric": "validation_final_mae",
        "ensemble_used": False,
        "candidates": {
            str(seed): {
                "validation_online_mae": seed_summaries[seed][
                    "validation_online_mae"
                ],
                "validation_final_mae": seed_summaries[seed][
                    "validation_final_mae"
                ],
            }
            for seed in seeds
        },
    }
    _atomic_write(output_root / "selection.json", _json_bytes(selection))

    external_summary: dict[str, Any] | None = None
    external_input_sha256: dict[str, str] | None = None
    if not smoke:
        checkpoint_path = output_root / f"seed_{selected_seed}" / "best_model.pth"
        selected_model = _load_checkpoint(checkpoint_path, device=device)
        external_input_sha256 = _external_input_hashes(event_root)
        external_events, external_audit = _external_event_traces(
            event_root=event_root,
            base_config=base_config,
            spec=spec,
        )
        if external_audit != {
            "accepted_station_count": int(
                experiment["audit"]["expected_external_station_count"]
            ),
            "rejected_station_count": int(
                experiment["audit"]["expected_external_rejected_station_count"]
            ),
            "event_count": int(experiment["audit"]["expected_external_event_count"]),
        }:
            raise ValueError(f"external causal data audit mismatch: {external_audit}")
        examples = _build_examples(
            external_events,
            horizons=range(1, spec.total_steps + 1),
            spec=spec,
        )
        predictions, anchors, residuals = _predict_examples(
            selected_model,
            examples,
            batch_size=int(training["evaluation_batch_size"]),
            device=device,
        )
        online_metrics = _online_metrics(examples, predictions)
        final_indices = _final_indices(examples, spec.total_steps)
        targets = np.asarray(
            [examples[index].magnitude for index in final_indices],
            dtype=np.float64,
        )
        final_metrics = _basic_metrics(predictions[final_indices], targets)
        rows = _prediction_rows(examples, predictions, anchors, residuals)
        _atomic_write(
            output_root / "external_online_predictions.csv", _csv_bytes(rows)
        )
        _atomic_write(
            output_root / "external_horizon_metrics.csv",
            _csv_bytes(_horizon_metric_rows(examples, predictions)),
        )
        _atomic_write(
            output_root / "external_final_event_predictions.csv",
            _csv_bytes([rows[int(index)] for index in final_indices]),
        )
        external_summary = {
            "audit": external_audit,
            "online_metrics": online_metrics,
            "final_metrics": final_metrics,
            "neural_residual": {
                "mean_abs_mw": float(np.mean(np.abs(residuals))),
                "maximum_abs_mw": float(np.max(np.abs(residuals))),
                "final_mean_abs_mw": float(
                    np.mean(np.abs(residuals[final_indices]))
                ),
            },
            "target_passed": float(final_metrics["event_mae"])
            <= float(experiment["target"]["final_event_mae_maximum"]),
        }

    summary = {
        "status": "smoke_complete" if smoke else "complete",
        "method": "causal_forward_guided_event_neural_v2",
        "framework": "pytorch",
        "deep_learning": True,
        "input_components": ["R"],
        "uses_tcn": True,
        "uses_transformer": True,
        "uses_shared_event_stf": True,
        "uses_original_four_term_loss": True,
        "uses_absolute_forward_delays": True,
        "forward_reference_doi": REFERENCE_DOI,
        "uses_future_waveform": False,
        "uses_final_peak_for_station_selection": False,
        "uses_ensemble": False,
        "external_loaded_after_seed_selection": not smoke,
        "created_at_utc": _utc_now_iso(),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "git_dirty": git_is_dirty(PROJECT_ROOT),
        "device": str(device),
        "base_config_sha256": sha256_file(base_config_path),
        "source_data_sha256": sha256_file(base_config["paths"]["data_path"]),
        "station_pool_sha256": station_pool_sha256,
        "external_input_sha256": external_input_sha256,
        "experiment_config_sha256": sha256_file(config_path),
        "spec": spec.to_dict(),
        "loss": {name: float(value) for name, value in loss_weights.items()},
        "selection": selection,
        "seed_summaries": {str(seed): seed_summaries[seed] for seed in seeds},
        "external": external_summary,
    }
    summary["artifact_sha256"] = _artifact_hashes(output_root)
    _atomic_write(output_root / "summary.json", _json_bytes(summary))
    _atomic_write(output_root / "COMPLETE", b"\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a causal forward-guided R-only event neural model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / "causal_forward_guided_event_neural.yaml"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    summary = run(
        config_path=args.config.resolve(),
        output_root=args.output_dir.resolve(),
        device=torch.device(args.device),
        smoke=bool(args.smoke),
    )
    selected = summary["selection"]["selected_seed"]
    if summary["external"] is None:
        print(f"causal forward-guided smoke complete; selected seed {selected}")
    else:
        print(
            "causal forward-guided run complete; "
            f"selected seed {selected}; "
            f"external final MAE={summary['external']['final_metrics']['event_mae']:.6f}"
        )


if __name__ == "__main__":
    main()
