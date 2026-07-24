from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_v2 import CorrectedEarthquakeDataset  # noqa: E402
from src.data.loaders_v2 import _runtime_config  # noqa: E402
from src.models.event_magnitude import (  # noqa: E402
    RadialPINNEventNet,
    RadialPINNEventSpec,
    RadialPINNStationObservation,
    build_radial_pinn_event_features,
    radial_pinn_event_feature_names,
)
from src.models.model import PINNModel  # noqa: E402
from src.training.train import _prepare_v2_batch  # noqa: E402
from src.utils.device import configure_runtime, get_preferred_device  # noqa: E402
from src.utils.provenance import (  # noqa: E402
    current_git_commit,
    git_is_dirty,
    sha256_file,
    utc_now_iso,
)


REQUIRED_SEEDS = (17, 42, 73)
SPLIT_NAMES = ("train", "validation", "test")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return payload


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(dict(payload), temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize an empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(config))
    payload.pop("paths", None)
    training = payload.get("training")
    if isinstance(training, dict):
        training.pop("random_seed", None)
    return payload


def _validated_paths(
    experiment_config: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Path]], dict[int, dict[str, Any]]]:
    raw_models = experiment_config.get("base_models")
    if not isinstance(raw_models, dict):
        raise ValueError("base_models must be a mapping")
    normalized = {int(seed): value for seed, value in raw_models.items()}
    if tuple(sorted(normalized)) != REQUIRED_SEEDS:
        raise ValueError("base_models must contain seeds 17, 42, and 73")

    paths: dict[int, dict[str, Path]] = {}
    configs: dict[int, dict[str, Any]] = {}
    scientific_reference: dict[str, Any] | None = None
    data_hash: str | None = None
    for seed in REQUIRED_SEEDS:
        item = normalized[seed]
        if not isinstance(item, dict):
            raise ValueError(f"base model {seed} must be a mapping")
        model_dir = Path(item["model_dir"]).resolve()
        current = {
            "model_dir": model_dir,
            "model": model_dir / "best_model.pth",
            "config": model_dir / "config.yaml",
            "split": model_dir / "split.json",
            "external": Path(item["external_station_csv"]).resolve(),
        }
        missing = [str(path) for name, path in current.items() if name != "model_dir" and not path.is_file()]
        if missing:
            raise FileNotFoundError(f"base model {seed} is incomplete: {missing}")
        config = _load_yaml(current["config"])
        if int(config["training"]["random_seed"]) != seed:
            raise ValueError(f"base model seed mismatch for {seed}")
        if config.get("workflow") != "station_random_shifted_stf":
            raise ValueError("base models must use the active station workflow")
        if config.get("model", {}).get("input_components", ["R"]) not in (["R"], ["radial"]):
            raise ValueError("base models must be R-only")
        scientific = _scientific_config(config)
        if scientific_reference is None:
            scientific_reference = scientific
        elif scientific != scientific_reference:
            raise ValueError("base model scientific configurations differ")
        configured_data = Path(config["paths"]["data_path"]).resolve()
        configured_hash = sha256_file(configured_data)
        if data_hash is None:
            data_hash = configured_hash
        elif configured_hash != data_hash:
            raise ValueError("base models use different training data")
        paths[seed] = current
        configs[seed] = config
    return paths, configs


def _event_spec(config: Mapping[str, Any]) -> RadialPINNEventSpec:
    payload = dict(config.get("event_model", {}))
    payload["distance_exponents"] = tuple(
        float(value) for value in payload["distance_exponents"]
    )
    payload["pinn_view_names"] = tuple(f"seed_{seed}" for seed in REQUIRED_SEEDS)
    return RadialPINNEventSpec(**payload)


def _inferred_station_predictions(
    dataset: CorrectedEarthquakeDataset,
    paths: Mapping[int, Mapping[str, Path]],
    configs: Mapping[int, Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[tuple[str, str], tuple[float, ...]]:
    expected_keys = [
        (str(sample["event"]), str(sample["station"]))
        for sample in dataset.samples
    ]
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("training data contains duplicate event/station keys")
    per_seed: dict[int, list[float]] = {}
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    for seed in REQUIRED_SEEDS:
        config = dict(configs[seed])
        model = PINNModel(config).to(device)
        checkpoint = torch.load(
            paths[seed]["model"],
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        values: list[float] = []
        with torch.no_grad():
            for batch in loader:
                prepared = _prepare_v2_batch(batch, config, device)
                prediction = model.predict_heads(
                    prepared.model_input,
                    meta=prepared.metadata,
                ).catalog_mw
                if not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError(f"base seed {seed} produced non-finite Mw")
                values.extend(float(value) for value in prediction.cpu())
        if len(values) != len(expected_keys):
            raise RuntimeError(f"base seed {seed} prediction count mismatch")
        per_seed[seed] = values
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        key: tuple(per_seed[seed][index] for seed in REQUIRED_SEEDS)
        for index, key in enumerate(expected_keys)
    }


def _training_observations(
    dataset: CorrectedEarthquakeDataset,
    predictions: Mapping[tuple[str, str], tuple[float, ...]],
) -> list[RadialPINNStationObservation]:
    observations: list[RadialPINNStationObservation] = []
    for sample in dataset.samples:
        key = (str(sample["event"]), str(sample["station"]))
        observations.append(
            RadialPINNStationObservation(
                event=key[0],
                station=key[1],
                radial_peak_cm=float(sample["radial_peak_cm"]),
                source_distance_km=float(sample["source_distance_m"]) / 1000.0,
                pinn_mw=predictions[key],
                magnitude=float(sample["magnitude_catalog"]),
            )
        )
    return observations


def _float_equal(left: str, right: str, *, tolerance: float = 1.0e-6) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _external_observations(
    paths: Mapping[int, Mapping[str, Path]],
) -> list[RadialPINNStationObservation]:
    rows_by_seed: dict[int, dict[tuple[str, str], dict[str, str]]] = {}
    for seed in REQUIRED_SEEDS:
        rows: dict[tuple[str, str], dict[str, str]] = {}
        with paths[seed]["external"].open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["event"], row["station"])
                if key in rows:
                    raise ValueError(f"duplicate external key for seed {seed}: {key}")
                rows[key] = row
        rows_by_seed[seed] = rows
    expected_keys = set(rows_by_seed[REQUIRED_SEEDS[0]])
    if any(set(rows_by_seed[seed]) != expected_keys for seed in REQUIRED_SEEDS[1:]):
        raise ValueError("external station keys differ across base seeds")

    result: list[RadialPINNStationObservation] = []
    invariant_fields = ("max_radial_cm", "source_distance_km", "mw_catalog")
    for key in sorted(expected_keys):
        reference = rows_by_seed[REQUIRED_SEEDS[0]][key]
        for seed in REQUIRED_SEEDS[1:]:
            candidate = rows_by_seed[seed][key]
            if any(
                not _float_equal(reference[field], candidate[field])
                for field in invariant_fields
            ):
                raise ValueError(f"external inputs differ across seeds for {key}")
        result.append(
            RadialPINNStationObservation(
                event=key[0],
                station=key[1],
                radial_peak_cm=float(reference["max_radial_cm"]),
                source_distance_km=float(reference["source_distance_km"]),
                pinn_mw=tuple(
                    float(rows_by_seed[seed][key]["mw_pred"])
                    for seed in REQUIRED_SEEDS
                ),
                magnitude=float(reference["mw_catalog"]),
            )
        )
    return result


def _group_observations(
    observations: Iterable[RadialPINNStationObservation],
) -> dict[str, list[RadialPINNStationObservation]]:
    grouped: dict[str, list[RadialPINNStationObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.event].append(observation)
    if not grouped:
        raise ValueError("no station observations supplied")
    return dict(grouped)


def _event_matrix(
    observations: Sequence[RadialPINNStationObservation],
    spec: RadialPINNEventSpec,
) -> tuple[list[str], np.ndarray, np.ndarray, list[int], list[int]]:
    grouped = _group_observations(observations)
    events = sorted(grouped)
    features: list[np.ndarray] = []
    targets: list[float] = []
    available_counts: list[int] = []
    used_counts: list[int] = []
    for event in events:
        rows = grouped[event]
        magnitudes = {float(row.magnitude) for row in rows if row.magnitude is not None}
        if len(magnitudes) != 1:
            raise ValueError(f"event must have exactly one magnitude: {event}")
        features.append(build_radial_pinn_event_features(rows, spec))
        targets.append(magnitudes.pop())
        available_counts.append(len(rows))
        used_counts.append(min(len(rows), int(spec.top_k)))
    matrix = np.stack(features)
    target_array = np.asarray(targets, dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.isfinite(target_array).all():
        raise ValueError("event matrix contains non-finite values")
    return events, matrix, target_array, available_counts, used_counts


def _observations_for_split(
    observations: Sequence[RadialPINNStationObservation],
    split_path: Path,
) -> dict[str, list[RadialPINNStationObservation]]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("protocol") != "within_event_station":
        raise ValueError("event head requires the within-event station protocol")
    split_keys = {
        name: set(str(value) for value in split["sample_keys"][name])
        for name in SPLIT_NAMES
    }
    if any(split_keys[left] & split_keys[right] for left in SPLIT_NAMES for right in SPLIT_NAMES if left < right):
        raise ValueError("split station keys overlap")
    by_key = {f"{row.event}::{row.station}": row for row in observations}
    known = set().union(*split_keys.values())
    if known != set(by_key):
        raise ValueError("split keys do not exactly cover training observations")
    return {
        name: [by_key[key] for key in sorted(split_keys[name])]
        for name in SPLIT_NAMES
    }


def _metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    errors = np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("metric inputs must be finite and non-empty")
    return {
        "event_count": int(errors.size),
        "event_mae": float(np.mean(np.abs(errors))),
        "event_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "event_bias": float(np.mean(errors)),
    }


def _prediction_rows(
    model: RadialPINNEventNet,
    events: Sequence[str],
    features: np.ndarray,
    targets: np.ndarray,
    available_counts: Sequence[int],
    used_counts: Sequence[int],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    with torch.no_grad():
        predictions = model(tensor)
        linear, nonlinear = model.standardized_components(tensor)
        nonlinear_delta = model.target_scale * nonlinear
    prediction_values = predictions.cpu().numpy().astype(np.float64)
    nonlinear_values = nonlinear_delta.cpu().numpy().astype(np.float64)
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        error = float(prediction_values[index] - targets[index])
        rows.append(
            {
                "event": event,
                "mw_pred": float(prediction_values[index]),
                "mw_reference": float(targets[index]),
                "error": error,
                "abs_error": abs(error),
                "station_count_available": int(available_counts[index]),
                "station_count_used": int(used_counts[index]),
                "nonlinear_delta_mw": float(nonlinear_values[index]),
            }
        )
    return rows


def _train_head(
    *,
    head_seed: int,
    split_observations: Mapping[str, Sequence[RadialPINNStationObservation]],
    spec: RadialPINNEventSpec,
    training_config: Mapping[str, Any],
    device: torch.device,
) -> tuple[RadialPINNEventNet, dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    matrices = {
        name: _event_matrix(list(split_observations[name]), spec)
        for name in SPLIT_NAMES
    }
    train_events, train_features, train_targets, _, _ = matrices["train"]
    if len(train_events) < 2:
        raise ValueError("event head requires at least two training events")
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    feature_scale[feature_scale < 1.0e-6] = 1.0
    target_mean = float(train_targets.mean())
    target_scale = float(train_targets.std())
    if target_scale < 1.0e-6:
        raise ValueError("training targets have no variation")

    configure_runtime(head_seed, device)
    random.seed(head_seed)
    np.random.seed(head_seed)
    model = RadialPINNEventNet(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        spec=spec,
    ).to(device)
    train_x = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(train_targets, dtype=torch.float32, device=device)
    train_y_standardized = (train_y - model.target_mean) / model.target_scale
    validation_x = torch.as_tensor(
        matrices["validation"][1],
        dtype=torch.float32,
        device=device,
    )
    validation_y = torch.as_tensor(
        matrices["validation"][2],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config["learning_rate"]))
    epochs = int(training_config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(training_config["learning_rate"]) * 0.01,
    )
    linear_alpha = float(training_config["linear_ridge_alpha"])
    nonlinear_alpha = float(training_config["nonlinear_l2_alpha"])
    linear_warmup_epochs = int(training_config["linear_warmup_epochs"])
    if not 0 <= linear_warmup_epochs < epochs:
        raise ValueError("linear_warmup_epochs must be in [0, epochs)")
    gradient_clip = float(training_config["gradient_clip_norm"])
    best_epoch = 0
    best_validation_mae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    log_rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        linear, nonlinear = model.standardized_components(train_x)
        nonlinear_active = epoch > linear_warmup_epochs
        active_nonlinear = nonlinear if nonlinear_active else nonlinear.detach() * 0.0
        fit_loss = F.mse_loss(
            linear + active_nonlinear,
            train_y_standardized,
        )
        linear_penalty = (
            linear_alpha
            / len(train_events)
            * model.linear_branch.weight.square().sum()
        )
        raw_nonlinear_penalty = sum(
            parameter.square().sum()
            for parameter in model.nonlinear_branch.parameters()
            if parameter.ndim >= 2
        )
        nonlinear_penalty = (
            nonlinear_alpha / len(train_events) * raw_nonlinear_penalty
            if nonlinear_active
            else raw_nonlinear_penalty.detach() * 0.0
        )
        loss = fit_loss + linear_penalty + nonlinear_penalty
        if not torch.isfinite(loss):
            raise FloatingPointError(f"head seed {head_seed} produced non-finite loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip,
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"head seed {head_seed} produced non-finite gradients")
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            train_prediction = model(train_x)
            validation_prediction = model(validation_x)
            train_mae = float(torch.mean(torch.abs(train_prediction - train_y)).cpu())
            validation_mae = float(
                torch.mean(torch.abs(validation_prediction - validation_y)).cpu()
            )
        log_rows.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "fit_loss": float(fit_loss.detach().cpu()),
                "linear_penalty": float(linear_penalty.detach().cpu()),
                "nonlinear_penalty": float(nonlinear_penalty.detach().cpu()),
                "train_event_mae": train_mae,
                "validation_event_mae": validation_mae,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if epoch >= linear_warmup_epochs and validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None or best_epoch < 1:
        raise RuntimeError("event head did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)

    prediction_rows: dict[str, list[dict[str, Any]]] = {}
    split_metrics: dict[str, Any] = {}
    for name in SPLIT_NAMES:
        events, features, targets, available, used = matrices[name]
        rows = _prediction_rows(
            model,
            events,
            features,
            targets,
            available,
            used,
            device=device,
        )
        prediction_rows[name] = rows
        split_metrics[name] = _metrics(
            np.asarray([row["mw_pred"] for row in rows]),
            targets,
        )
    audit = {
        "head_seed": head_seed,
        "best_epoch": best_epoch,
        "best_validation_event_mae": best_validation_mae,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "nonlinear_parameter_count": sum(
            parameter.numel() for parameter in model.nonlinear_branch.parameters()
        ),
        "feature_count": len(radial_pinn_event_feature_names(spec)),
        "training_event_count": len(train_events),
        "linear_warmup_epochs": linear_warmup_epochs,
        "split_metrics": split_metrics,
    }
    return model, audit, prediction_rows, log_rows


def _checkpoint_payload(
    *,
    model: RadialPINNEventNet,
    spec: RadialPINNEventSpec,
    audit: Mapping[str, Any],
    feature_names: Sequence[str],
    source_paths: Mapping[int, Mapping[str, Path]],
) -> dict[str, Any]:
    return {
        "model_type": "radial_pinn_event_neural_v2",
        "framework": "pytorch",
        "deep_learning": True,
        "uses_ridge_prediction": False,
        "head_seed": int(audit["head_seed"]),
        "best_epoch": int(audit["best_epoch"]),
        "spec": spec.to_dict(),
        "feature_names": list(feature_names),
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "base_model_sha256": {
            str(seed): sha256_file(source_paths[seed]["model"])
            for seed in REQUIRED_SEEDS
        },
        "created_at_utc": utc_now_iso(),
    }


def run(*, config_path: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    experiment_config = _load_yaml(config_path)
    if experiment_config.get("method") != "radial_pinn_event_neural_v2":
        raise ValueError("unsupported experiment method")
    paths, base_configs = _validated_paths(experiment_config)
    spec = _event_spec(experiment_config)
    training_config = experiment_config["training"]
    head_seeds = tuple(int(seed) for seed in training_config["head_seeds"])
    if head_seeds != REQUIRED_SEEDS:
        raise ValueError("head_seeds must be [17, 42, 73]")
    target_config = experiment_config["target"]
    target_mae = float(target_config["event_mae_maximum"])
    minimum_count = int(target_config["event_count_minimum"])
    preferred_count = int(target_config["event_count_preferred"])

    device = get_preferred_device()
    configure_runtime(REQUIRED_SEEDS[0], device)
    reference_config = base_configs[REQUIRED_SEEDS[0]]
    dataset = CorrectedEarthquakeDataset(_runtime_config(reference_config))
    inferred = _inferred_station_predictions(
        dataset,
        paths,
        base_configs,
        device=device,
    )
    internal_observations = _training_observations(dataset, inferred)
    external_observations = _external_observations(paths)
    external_matrix = _event_matrix(external_observations, spec)
    if len(external_matrix[0]) != preferred_count:
        raise ValueError(
            f"formal external evaluation requires {preferred_count} events"
        )

    output_root.mkdir(parents=True)
    _atomic_write(output_root / "config.yaml", config_path.read_bytes())
    head_models: dict[int, RadialPINNEventNet] = {}
    head_external_rows: dict[int, list[dict[str, Any]]] = {}
    head_summaries: dict[str, Any] = {}
    artifact_hashes: dict[str, str] = {}
    feature_names = radial_pinn_event_feature_names(spec)
    for head_seed in head_seeds:
        split_observations = _observations_for_split(
            internal_observations,
            paths[head_seed]["split"],
        )
        model, audit, internal_rows, log_rows = _train_head(
            head_seed=head_seed,
            split_observations=split_observations,
            spec=spec,
            training_config=training_config,
            device=device,
        )
        external_rows = _prediction_rows(
            model,
            *external_matrix,
            device=device,
        )
        external_metrics = _metrics(
            np.asarray([row["mw_pred"] for row in external_rows]),
            external_matrix[2],
        )
        nonlinear_values = np.asarray(
            [float(row["nonlinear_delta_mw"]) for row in external_rows],
            dtype=np.float64,
        )
        audit["external_metrics"] = external_metrics
        audit["external_nonlinear_delta_mean_abs"] = float(
            np.mean(np.abs(nonlinear_values))
        )
        audit["external_nonlinear_delta_max_abs"] = float(
            np.max(np.abs(nonlinear_values))
        )

        seed_root = output_root / f"seed_{head_seed}"
        checkpoint_path = seed_root / "best_model.pth"
        _atomic_torch_save(
            checkpoint_path,
            _checkpoint_payload(
                model=model,
                spec=spec,
                audit=audit,
                feature_names=feature_names,
                source_paths=paths,
            ),
        )
        _atomic_write(seed_root / "training_log.csv", _csv_bytes(log_rows))
        for split_name, rows in internal_rows.items():
            _atomic_write(
                seed_root / f"internal_{split_name}_predictions.csv",
                _csv_bytes(rows),
            )
        _atomic_write(
            seed_root / "external_event_predictions.csv",
            _csv_bytes(external_rows),
        )
        _atomic_write(seed_root / "summary.json", _json_bytes(audit))
        for artifact in seed_root.iterdir():
            if artifact.is_file():
                artifact_hashes[str(artifact.relative_to(output_root))] = sha256_file(artifact)
        head_models[head_seed] = model
        head_external_rows[head_seed] = external_rows
        head_summaries[str(head_seed)] = audit

    expected_events = list(external_matrix[0])
    ensemble_rows: list[dict[str, Any]] = []
    for index, event in enumerate(expected_events):
        predictions = []
        nonlinear_deltas = []
        for head_seed in head_seeds:
            row = head_external_rows[head_seed][index]
            if row["event"] != event:
                raise RuntimeError("external event order differs across heads")
            predictions.append(float(row["mw_pred"]))
            nonlinear_deltas.append(float(row["nonlinear_delta_mw"]))
        prediction = float(np.mean(predictions))
        reference = float(external_matrix[2][index])
        error = prediction - reference
        ensemble_rows.append(
            {
                "event": event,
                "mw_pred": prediction,
                "mw_reference": reference,
                "error": error,
                "abs_error": abs(error),
                "station_count_available": int(external_matrix[3][index]),
                "station_count_used": int(external_matrix[4][index]),
                "head_prediction_std": float(np.std(predictions)),
                "nonlinear_delta_mw_mean": float(np.mean(nonlinear_deltas)),
            }
        )
    ensemble_metrics = _metrics(
        np.asarray([row["mw_pred"] for row in ensemble_rows]),
        external_matrix[2],
    )
    ensemble_path = output_root / "ensemble_external_event_predictions.csv"
    _atomic_write(ensemble_path, _csv_bytes(ensemble_rows))
    artifact_hashes[str(ensemble_path.relative_to(output_root))] = sha256_file(ensemble_path)

    target_passed = (
        int(ensemble_metrics["event_count"]) >= minimum_count
        and float(ensemble_metrics["event_mae"]) <= target_mae
    )
    summary = {
        "status": "complete",
        "method": "radial_pinn_event_neural_v2",
        "framework": "pytorch",
        "deep_learning": True,
        "uses_ridge_prediction": False,
        "created_at_utc": utc_now_iso(),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "git_dirty": git_is_dirty(PROJECT_ROOT),
        "device": str(device),
        "base_pinn_seeds": list(REQUIRED_SEEDS),
        "event_head_seeds": list(head_seeds),
        "training_event_count": len({row.event for row in internal_observations}),
        "training_station_count": len(internal_observations),
        "feature_count": len(feature_names),
        "head_summaries": head_summaries,
        "external_ensemble": ensemble_metrics,
        "target": {
            "event_count_minimum": minimum_count,
            "event_count_preferred": preferred_count,
            "event_mae_maximum": target_mae,
            "passed": target_passed,
        },
        "source_sha256": {
            str(seed): {
                "base_model": sha256_file(paths[seed]["model"]),
                "base_config": sha256_file(paths[seed]["config"]),
                "base_split": sha256_file(paths[seed]["split"]),
                "external_station_csv": sha256_file(paths[seed]["external"]),
            }
            for seed in REQUIRED_SEEDS
        },
        "artifact_sha256": artifact_hashes,
    }
    _atomic_write(output_root / "summary.json", _json_bytes(summary))
    _atomic_write(output_root / "COMPLETE", b"\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the R-only PINN neural event magnitude heads."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        config_path=args.config.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
