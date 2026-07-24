from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
    CausalEventSnapshot,
    CausalRadialEventNet,
    CausalRadialEventSpec,
    CausalRadialStationObservation,
    build_causal_event_snapshot,
    causal_event_feature_names,
    causal_running_peak_cm,
    select_single_seed,
)
from src.utils.config_v2 import validate_config_v2  # noqa: E402
from src.utils.device import configure_runtime  # noqa: E402
from src.utils.provenance import (  # noqa: E402
    current_git_commit,
    git_is_dirty,
    sha256_file,
)


REQUIRED_SEEDS = (17, 42, 73)
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SnapshotSet:
    rows: tuple[CausalEventSnapshot, ...]
    features: np.ndarray
    targets: np.ndarray


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize an empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(rows[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


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


def _validate_experiment_config(payload: Mapping[str, Any]) -> None:
    if payload.get("method") != "causal_radial_event_neural_v1":
        raise ValueError("method must be causal_radial_event_neural_v1")
    seeds = tuple(int(value) for value in payload["training"]["head_seeds"])
    if seeds != REQUIRED_SEEDS:
        raise ValueError(f"head_seeds must be {REQUIRED_SEEDS}")
    selection = payload["selection"]
    if selection.get("primary_metric") != "validation_online_mae":
        raise ValueError("primary seed metric must be validation_online_mae")
    if selection.get("tie_break_metric") != "validation_final_mae":
        raise ValueError("seed tie-break must be validation_final_mae")
    if bool(selection.get("use_ensemble", True)):
        raise ValueError("causal formal result forbids seed ensembling")
    if int(payload["target"]["final_horizon_sec"]) != 200:
        raise ValueError("formal final horizon must be 200 seconds")


def _runtime_base_config(
    base_config_path: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    config = _load_yaml(base_config_path)
    config["dataset"]["radial_peak_min_cm"] = 0.0
    config["training"]["random_seed"] = int(seed)
    config["training"]["event_balanced_sampling"] = False
    config["training"]["num_workers"] = 0
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
        taps = int(filter_config["num_taps"])
        filter_half_window = (taps - 1) // 2
    else:
        raise ValueError("unsupported waveform filter for causal replay")
    return interpolation + filter_half_window


def _sample_to_observation(
    sample: Mapping[str, Any],
    *,
    spec: CausalRadialEventSpec,
) -> CausalRadialStationObservation:
    radial = np.asarray(sample["radial"], dtype=np.float64)
    valid_mask = np.asarray(sample["waveform_valid_mask"], dtype=bool)
    if radial.shape != (spec.total_steps,) or valid_mask.shape != radial.shape:
        raise ValueError("station waveform does not match causal event window")
    p_arrival = float(sample.get("p_arrival_sec", float("nan")))
    if not math.isfinite(p_arrival):
        raise ValueError("station sample lacks a finite P arrival")
    return CausalRadialStationObservation(
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


def _subset_samples(loader: Any) -> list[dict[str, Any]]:
    subset = loader.dataset
    if not hasattr(subset, "indices") or not hasattr(subset, "dataset"):
        raise TypeError("causal runner requires a deterministic Subset loader")
    base = subset.dataset
    if not hasattr(base, "samples"):
        raise TypeError("causal runner requires CorrectedEarthquakeDataset")
    return [base.samples[int(index)] for index in subset.indices]


def _group_observations(
    observations: Iterable[CausalRadialStationObservation],
) -> dict[str, list[CausalRadialStationObservation]]:
    grouped: dict[str, list[CausalRadialStationObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.event].append(observation)
    return dict(grouped)


def _build_snapshot_set(
    observations: Sequence[CausalRadialStationObservation],
    *,
    spec: CausalRadialEventSpec,
) -> SnapshotSet:
    rows: list[CausalEventSnapshot] = []
    grouped = _group_observations(observations)
    for event in sorted(grouped):
        event_observations = grouped[event]
        for horizon_step in range(1, spec.total_steps + 1):
            snapshot = build_causal_event_snapshot(
                event_observations,
                horizon_step=horizon_step,
                spec=spec,
            )
            if snapshot is not None:
                rows.append(snapshot)
    if not rows or any(row.magnitude is None for row in rows):
        raise ValueError("causal snapshot set has no labeled predictions")
    features = np.asarray([row.features for row in rows], dtype=np.float32)
    targets = np.asarray([row.magnitude for row in rows], dtype=np.float32)
    if features.shape != (len(rows), spec.feature_count):
        raise ValueError("causal snapshot feature matrix has the wrong shape")
    return SnapshotSet(tuple(rows), features, targets)


def _final_indices(rows: Sequence[CausalEventSnapshot], total_steps: int) -> np.ndarray:
    indices = np.asarray(
        [index for index, row in enumerate(rows) if row.horizon_step == total_steps],
        dtype=np.int64,
    )
    if indices.size < 2:
        raise ValueError("final snapshot set contains fewer than two events")
    return indices


def _positive_scale(values: np.ndarray) -> np.ndarray:
    scale = values.std(axis=0)
    scale[scale < 1.0e-6] = 1.0
    return scale


def _event_equal_weights(rows: Sequence[CausalEventSnapshot]) -> np.ndarray:
    counts = Counter(row.event for row in rows)
    weights = np.asarray([1.0 / counts[row.event] for row in rows], dtype=np.float32)
    weights *= len(weights) / float(weights.sum())
    return weights


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


def _online_metrics(
    rows: Sequence[CausalEventSnapshot],
    predictions: np.ndarray,
) -> dict[str, Any]:
    if len(rows) != predictions.size:
        raise ValueError("online predictions do not match snapshots")
    errors_by_event: dict[str, list[float]] = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        if row.magnitude is None:
            raise ValueError("online metric row lacks a target")
        errors_by_event[row.event].append(float(prediction) - row.magnitude)
    event_mae = [np.mean(np.abs(values)) for values in errors_by_event.values()]
    event_mse = [np.mean(np.square(values)) for values in errors_by_event.values()]
    event_bias = [np.mean(values) for values in errors_by_event.values()]
    return {
        "event_count": len(errors_by_event),
        "snapshot_count": len(rows),
        "event_equal_online_mae": float(np.mean(event_mae)),
        "event_equal_online_rmse": float(np.sqrt(np.mean(event_mse))),
        "event_equal_online_bias": float(np.mean(event_bias)),
    }


def _predict(
    model: CausalRadialEventNet,
    features: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        values = model(
            torch.as_tensor(features, dtype=torch.float32, device=device)
        )
    result = values.detach().cpu().numpy().astype(np.float64)
    if result.shape != (features.shape[0],) or not np.isfinite(result).all():
        raise FloatingPointError("causal model produced invalid predictions")
    return result


def _weighted_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return torch.sum(weights * torch.square(predictions - targets)) / torch.sum(
        weights
    )


def _training_log_row(
    *,
    phase: str,
    epoch: int,
    loss: torch.Tensor,
    fit_loss: torch.Tensor,
    penalty: torch.Tensor,
    validation_mae: float,
    learning_rate: float,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "epoch": epoch,
        "loss": float(loss.detach().cpu()),
        "fit_loss": float(fit_loss.detach().cpu()),
        "penalty": float(penalty.detach().cpu()),
        "validation_mae": validation_mae,
        "learning_rate": learning_rate,
    }


def _train_seed(
    *,
    seed: int,
    split_sets: Mapping[str, SnapshotSet],
    split_manifest: Mapping[str, Any],
    spec: CausalRadialEventSpec,
    training: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    train = split_sets["train"]
    validation = split_sets["validation"]
    test = split_sets["test"]
    train_final_indices = _final_indices(train.rows, spec.total_steps)
    validation_final_indices = _final_indices(validation.rows, spec.total_steps)
    anchor_train = train.features[
        train_final_indices, : spec.anchor_feature_count
    ]
    anchor_mean = anchor_train.mean(axis=0)
    anchor_scale = _positive_scale(anchor_train)
    online_mean = train.features.mean(axis=0)
    online_scale = _positive_scale(train.features)
    final_targets = train.targets[train_final_indices]
    target_mean = float(final_targets.mean())
    target_scale = float(final_targets.std())
    if target_scale < 1.0e-6:
        raise ValueError("training event targets have no variation")

    configure_runtime(seed, device)
    random.seed(seed)
    np.random.seed(seed)
    model = CausalRadialEventNet(
        anchor_feature_mean=anchor_mean,
        anchor_feature_scale=anchor_scale,
        online_feature_mean=online_mean,
        online_feature_scale=online_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        spec=spec,
    ).to(device)
    train_final_x = torch.as_tensor(
        train.features[train_final_indices], dtype=torch.float32, device=device
    )
    train_final_y = torch.as_tensor(
        train.targets[train_final_indices], dtype=torch.float32, device=device
    )
    train_final_y_std = (train_final_y - model.target_mean) / model.target_scale
    validation_final_x = validation.features[validation_final_indices]
    validation_final_y = validation.targets[validation_final_indices]
    gradient_clip = float(training["gradient_clip_norm"])
    log_rows: list[dict[str, Any]] = []

    for parameter in model.prefix_branch.parameters():
        parameter.requires_grad_(False)
    anchor_optimizer = torch.optim.Adam(
        model.anchor_branch.parameters(),
        lr=float(training["anchor_learning_rate"]),
    )
    anchor_epochs = int(training["anchor_epochs"])
    anchor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        anchor_optimizer,
        T_max=anchor_epochs,
        eta_min=float(training["anchor_learning_rate"]) * 0.01,
    )
    best_anchor_mae = float("inf")
    best_anchor_epoch = 0
    best_anchor_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, anchor_epochs + 1):
        model.train()
        anchor_optimizer.zero_grad(set_to_none=True)
        anchor_values, _ = model.standardized_components(train_final_x)
        fit_loss = F.mse_loss(anchor_values, train_final_y_std)
        penalty = (
            float(training["anchor_l2_alpha"])
            / len(train_final_indices)
            * model.anchor_branch.weight.square().sum()
        )
        loss = fit_loss + penalty
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.anchor_branch.parameters(), gradient_clip
        )
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"seed {seed} anchor training is non-finite")
        anchor_optimizer.step()
        anchor_scheduler.step()
        validation_prediction = _predict(model, validation_final_x, device=device)
        validation_mae = float(
            np.mean(np.abs(validation_prediction - validation_final_y))
        )
        log_rows.append(
            _training_log_row(
                phase="anchor",
                epoch=epoch,
                loss=loss,
                fit_loss=fit_loss,
                penalty=penalty,
                validation_mae=validation_mae,
                learning_rate=float(anchor_optimizer.param_groups[0]["lr"]),
            )
        )
        if validation_mae < best_anchor_mae:
            best_anchor_mae = validation_mae
            best_anchor_epoch = epoch
            best_anchor_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_anchor_state is None:
        raise RuntimeError("anchor training produced no checkpoint")
    model.load_state_dict(best_anchor_state, strict=True)

    for parameter in model.anchor_branch.parameters():
        parameter.requires_grad_(False)
    for parameter in model.prefix_branch.parameters():
        parameter.requires_grad_(True)
    prefix_optimizer = torch.optim.Adam(
        model.prefix_branch.parameters(),
        lr=float(training["prefix_learning_rate"]),
    )
    prefix_epochs = int(training["prefix_epochs"])
    prefix_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        prefix_optimizer,
        T_max=prefix_epochs,
        eta_min=float(training["prefix_learning_rate"]) * 0.01,
    )
    train_online_x = torch.as_tensor(
        train.features, dtype=torch.float32, device=device
    )
    train_online_y = torch.as_tensor(
        train.targets, dtype=torch.float32, device=device
    )
    train_online_weights = torch.as_tensor(
        _event_equal_weights(train.rows), dtype=torch.float32, device=device
    )
    validation_baseline = _predict(model, validation.features, device=device)
    best_prefix_mae = float(
        _online_metrics(validation.rows, validation_baseline)[
            "event_equal_online_mae"
        ]
    )
    best_prefix_epoch = 0
    best_prefix_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    for epoch in range(1, prefix_epochs + 1):
        model.train()
        prefix_optimizer.zero_grad(set_to_none=True)
        prediction = model(train_online_x)
        fit_loss = _weighted_mse(
            (prediction - model.target_mean) / model.target_scale,
            (train_online_y - model.target_mean) / model.target_scale,
            train_online_weights,
        )
        raw_penalty = sum(
            parameter.square().sum()
            for parameter in model.prefix_branch.parameters()
            if parameter.ndim >= 2
        )
        penalty = (
            float(training["prefix_l2_alpha"])
            / len(train.rows)
            * raw_penalty
        )
        loss = fit_loss + penalty
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.prefix_branch.parameters(), gradient_clip
        )
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"seed {seed} prefix training is non-finite")
        prefix_optimizer.step()
        prefix_scheduler.step()
        validation_prediction = _predict(model, validation.features, device=device)
        validation_mae = float(
            _online_metrics(validation.rows, validation_prediction)[
                "event_equal_online_mae"
            ]
        )
        log_rows.append(
            _training_log_row(
                phase="prefix",
                epoch=epoch,
                loss=loss,
                fit_loss=fit_loss,
                penalty=penalty,
                validation_mae=validation_mae,
                learning_rate=float(prefix_optimizer.param_groups[0]["lr"]),
            )
        )
        if validation_mae < best_prefix_mae:
            best_prefix_mae = validation_mae
            best_prefix_epoch = epoch
            best_prefix_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    model.load_state_dict(best_prefix_state, strict=True)

    split_predictions = {
        name: _predict(model, split_sets[name].features, device=device)
        for name in SPLIT_NAMES
    }
    split_online_metrics = {
        name: _online_metrics(split_sets[name].rows, split_predictions[name])
        for name in SPLIT_NAMES
    }
    split_final_metrics: dict[str, dict[str, Any]] = {}
    for name in SPLIT_NAMES:
        indices = _final_indices(split_sets[name].rows, spec.total_steps)
        split_final_metrics[name] = _basic_metrics(
            split_predictions[name][indices], split_sets[name].targets[indices]
        )
    final_features = validation.features[validation_final_indices]
    final_tensor = torch.as_tensor(final_features, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, final_residual = model.standardized_components(final_tensor)
    maximum_final_residual = float(final_residual.abs().max().cpu())
    if maximum_final_residual != 0.0:
        raise RuntimeError("prefix residual is nonzero at the final horizon")

    summary = {
        "seed": seed,
        "best_anchor_epoch": best_anchor_epoch,
        "best_anchor_validation_final_mae": best_anchor_mae,
        "best_prefix_epoch": best_prefix_epoch,
        "validation_online_mae": split_online_metrics["validation"][
            "event_equal_online_mae"
        ],
        "validation_final_mae": split_final_metrics["validation"]["event_mae"],
        "split_online_metrics": split_online_metrics,
        "split_final_metrics": split_final_metrics,
        "maximum_final_prefix_residual_standardized": maximum_final_residual,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "split_assignment_sha256": split_manifest["assignment_sha256"],
    }
    checkpoint = {
        "model_type": "causal_radial_event_neural_v1",
        "model_state_dict": model.state_dict(),
        "spec": spec.to_dict(),
        "feature_names": list(causal_event_feature_names(spec)),
        "summary": summary,
    }
    _atomic_torch_save(output_dir / "best_model.pth", checkpoint)
    _atomic_write(output_dir / "training_log.csv", _csv_bytes(log_rows))
    _atomic_write(output_dir / "split.json", _json_bytes(split_manifest))
    _atomic_write(output_dir / "summary.json", _json_bytes(summary))
    for name in ("validation", "test"):
        rows = _prediction_rows(split_sets[name].rows, split_predictions[name])
        _atomic_write(output_dir / f"{name}_online_predictions.csv", _csv_bytes(rows))
        _atomic_write(
            output_dir / f"{name}_horizon_metrics.csv",
            _csv_bytes(_horizon_metric_rows(split_sets[name].rows, split_predictions[name])),
        )
    return summary


def _prediction_rows(
    snapshots: Sequence[CausalEventSnapshot],
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot, prediction in zip(snapshots, predictions):
        if snapshot.magnitude is None:
            raise ValueError("prediction row lacks a target")
        error = float(prediction) - snapshot.magnitude
        rows.append(
            {
                "event": snapshot.event,
                "horizon_sec": snapshot.horizon_sec,
                "mw_pred": float(prediction),
                "mw_reference": snapshot.magnitude,
                "error": error,
                "abs_error": abs(error),
                "active_station_count": snapshot.active_station_count,
                "used_station_count": len(snapshot.used_stations),
                "used_stations": "|".join(snapshot.used_stations),
            }
        )
    return rows


def _horizon_metric_rows(
    snapshots: Sequence[CausalEventSnapshot],
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, snapshot in enumerate(snapshots):
        grouped[snapshot.horizon_step].append(index)
    rows: list[dict[str, Any]] = []
    for horizon_step in sorted(grouped):
        indices = np.asarray(grouped[horizon_step], dtype=np.int64)
        targets = np.asarray(
            [snapshots[index].magnitude for index in indices], dtype=np.float64
        )
        metrics = _basic_metrics(predictions[indices], targets)
        rows.append(
            {
                "horizon_sec": snapshots[int(indices[0])].horizon_sec,
                **metrics,
            }
        )
    return rows


def _load_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> CausalRadialEventNet:
    payload = torch.load(path, map_location=device, weights_only=False)
    spec = CausalRadialEventSpec.from_dict(payload["spec"])
    state = payload["model_state_dict"]
    model = CausalRadialEventNet(
        anchor_feature_mean=state["anchor_feature_mean"],
        anchor_feature_scale=state["anchor_feature_scale"],
        online_feature_mean=state["online_feature_mean"],
        online_feature_scale=state["online_feature_scale"],
        target_mean=float(state["target_mean"]),
        target_scale=float(state["target_scale"]),
        spec=spec,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _external_observations(
    *,
    event_root: Path,
    base_config: Mapping[str, Any],
    spec: CausalRadialEventSpec,
) -> tuple[list[CausalRadialStationObservation], dict[str, int]]:
    waveform_config = waveform_config_from_v2(dict(base_config))
    alpha = float(base_config["physics"]["alpha"])
    observations: list[CausalRadialStationObservation] = []
    rejection_count = 0
    for directory_name in EXTERNAL_EVENT_NAMES:
        bundle = load_event_bundle(event_root / directory_name)
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
                rejection_count += 1
                continue
            sample["p_arrival_sec"] = float(sample["source_distance_m"]) / alpha
            observations.append(_sample_to_observation(sample, spec=spec))
    return observations, {
        "accepted_station_count": len(observations),
        "rejected_station_count": rejection_count,
        "event_count": len({row.event for row in observations}),
    }


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"console.log", "summary.json", "COMPLETE"}
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
    spec = CausalRadialEventSpec.from_dict(experiment["event_model"])
    base_config = _runtime_base_config(base_config_path, seed=REQUIRED_SEEDS[0])
    expected_latency = _expected_causal_latency(base_config)
    if spec.causal_latency_samples != expected_latency:
        raise ValueError(
            f"causal latency {spec.causal_latency_samples} differs from "
            f"required conservative latency {expected_latency}"
        )
    expected_events = int(experiment["audit"]["expected_event_count"])
    expected_stations = int(experiment["audit"]["expected_station_count"])
    training = dict(experiment["training"])
    seeds = REQUIRED_SEEDS[:1] if smoke else REQUIRED_SEEDS
    if smoke:
        training["anchor_epochs"] = min(3, int(training["anchor_epochs"]))
        training["prefix_epochs"] = min(3, int(training["prefix_epochs"]))

    seed_summaries: dict[int, dict[str, Any]] = {}
    station_pool_sha256: str | None = None
    for seed in seeds:
        runtime = _runtime_base_config(base_config_path, seed=seed)
        train_loader, validation_loader, test_loader, split_manifest = (
            get_data_loaders_v2(runtime)
        )
        total_stations = sum(
            int(split_manifest[f"{name}_record_count"]) for name in SPLIT_NAMES
        )
        all_events = set(split_manifest["train_events"])
        if total_stations != expected_stations or len(all_events) != expected_events:
            raise ValueError(
                f"causal dataset is {len(all_events)} events/{total_stations} stations; "
                f"expected {expected_events}/{expected_stations}"
            )
        current_pool_sha256 = _station_pool_sha256(split_manifest)
        if station_pool_sha256 is None:
            station_pool_sha256 = current_pool_sha256
        elif current_pool_sha256 != station_pool_sha256:
            raise RuntimeError("causal station pool changed between seeds")
        loaders = {
            "train": train_loader,
            "validation": validation_loader,
            "test": test_loader,
        }
        split_sets = {
            name: _build_snapshot_set(
                [
                    _sample_to_observation(sample, spec=spec)
                    for sample in _subset_samples(loaders[name])
                ],
                spec=spec,
            )
            for name in SPLIT_NAMES
        }
        seed_dir = output_root / f"seed_{seed}"
        summary = _train_seed(
            seed=seed,
            split_sets=split_sets,
            split_manifest=split_manifest,
            spec=spec,
            training=training,
            output_dir=seed_dir,
            device=device,
        )
        seed_summaries[seed] = summary

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
        external_observations, external_audit = _external_observations(
            event_root=event_root,
            base_config=base_config,
            spec=spec,
        )
        expected_external_events = int(
            experiment["audit"]["expected_external_event_count"]
        )
        expected_external_stations = int(
            experiment["audit"]["expected_external_station_count"]
        )
        if (
            external_audit["event_count"] != expected_external_events
            or external_audit["accepted_station_count"] != expected_external_stations
        ):
            raise ValueError(
                f"external causal data is {external_audit}; expected "
                f"{expected_external_events} events/{expected_external_stations} stations"
            )
        external_set = _build_snapshot_set(external_observations, spec=spec)
        external_predictions = _predict(
            selected_model, external_set.features, device=device
        )
        external_online = _online_metrics(
            external_set.rows, external_predictions
        )
        final_indices = _final_indices(external_set.rows, spec.total_steps)
        external_final = _basic_metrics(
            external_predictions[final_indices], external_set.targets[final_indices]
        )
        prediction_rows = _prediction_rows(
            external_set.rows, external_predictions
        )
        horizon_rows = _horizon_metric_rows(
            external_set.rows, external_predictions
        )
        final_rows = [prediction_rows[int(index)] for index in final_indices]
        _atomic_write(
            output_root / "external_online_predictions.csv",
            _csv_bytes(prediction_rows),
        )
        _atomic_write(
            output_root / "external_horizon_metrics.csv",
            _csv_bytes(horizon_rows),
        )
        _atomic_write(
            output_root / "external_final_event_predictions.csv",
            _csv_bytes(final_rows),
        )
        external_summary = {
            "audit": external_audit,
            "online_metrics": external_online,
            "final_metrics": external_final,
            "target_passed": float(external_final["event_mae"])
            <= float(experiment["target"]["final_event_mae_maximum"]),
        }

    summary = {
        "status": "smoke_complete" if smoke else "complete",
        "method": "causal_radial_event_neural_v1",
        "framework": "pytorch",
        "deep_learning": True,
        "input_components": ["R"],
        "uses_future_waveform": False,
        "uses_final_peak_for_station_selection": False,
        "uses_ensemble": False,
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
        description="Train and select one strictly causal R-only event model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiments" / "causal_radial_event_neural.yaml",
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
    external = summary["external"]
    if external is None:
        print(f"causal smoke complete; selected seed {selected}")
    else:
        print(
            f"causal run complete; selected seed {selected}; "
            f"external final MAE={external['final_metrics']['event_mae']:.6f}"
        )


if __name__ == "__main__":
    main()
