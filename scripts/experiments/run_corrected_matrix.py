from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders_v2 import get_data_loaders_v2
from src.evaluation.evaluate import evaluate
from src.evaluation.evaluate_unseen import evaluate_unseen_events
from src.evaluation.metrics import (
    aggregate_event_predictions,
    summarize_predictions,
)
from src.models.model import PINNModel
from src.training.checkpointing import load_full_checkpoint
from src.training.train import (
    _build_stf_rate_criterion,
    _prepare_v2_batch,
    train,
)
from src.utils.config_v2 import validate_config_v2
from src.utils.device import get_preferred_device
from src.utils.provenance import (
    current_git_commit,
    git_is_dirty,
    sha256_file,
    utc_now_iso,
)


SEEDS = (17, 42, 73)
MATRIX_IDS = (
    "V2-BASE",
    "V2-FULL",
    "V2-NOSYNTH",
    "V2-NOSTF",
    "V2-NOMETA",
    "V2-GAIN144",
    "V2-DELTA-META",
    "V2-CATALOG-SCALED-STF",
)
EXTERNAL_EVENT_NAMES = (
    "iquique-aftershock-2014-chile",
    "nepal-aftershock-2015",
    "kodiak-2018-alaska",
    "samos-2020-greece",
    "luding-2022-china",
    "xizang-2025-southern-tibetan-plateau",
    "myanmar-2025-mandalay",
    "sand-point-2025-alaska",
)
DECLARED_DIFFS = {
    "V2-BASE": set(),
    "V2-FULL": {"training.stf_rate_loss.include_intermediate_field"},
    "V2-NOSYNTH": {"training.stf_rate_loss.lambda_synth"},
    "V2-NOSTF": {
        "training.stf_rate_loss.lambda_MSE",
        "training.stf_rate_loss.lambda_shape",
    },
    "V2-NOMETA": {"model.use_meta"},
    "V2-GAIN144": {"physics.amplitude_gain"},
    "V2-DELTA-META": {"geometry.network_distance"},
    "V2-CATALOG-SCALED-STF": {"dataset.stf.magnitude_target"},
}
MAX_VALIDATED_RESUMES = 2


def config_diff_paths(
    base: dict[str, Any],
    modified: dict[str, Any],
    prefix: str = "",
) -> set[str]:
    differences: set[str] = set()
    for key in set(base) | set(modified):
        path = f"{prefix}.{key}" if prefix else key
        if key not in base or key not in modified:
            differences.add(path)
            continue
        left = base[key]
        right = modified[key]
        if isinstance(left, dict) and isinstance(right, dict):
            differences.update(config_diff_paths(left, right, path))
        elif left != right:
            differences.add(path)
    return differences


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


def load_matrix_configs(config_dir: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(config_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"matrix config directory not found: {root}")
    expected_files = {f"{experiment_id}.yaml" for experiment_id in MATRIX_IDS}
    actual_files = {
        path.name
        for path in root.glob("V2-*.yaml")
        if path.name != "V2-SELECTED.yaml"
    }
    if actual_files != expected_files:
        raise ValueError(
            "matrix config files differ: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    configs = {
        experiment_id: _load_yaml(root / f"{experiment_id}.yaml")
        for experiment_id in MATRIX_IDS
    }
    base = configs["V2-BASE"]
    for experiment_id, config in configs.items():
        validate_config_v2(config)
        actual_differences = config_diff_paths(base, config)
        if actual_differences != DECLARED_DIFFS[experiment_id]:
            raise ValueError(
                f"{experiment_id} changes undeclared factors: "
                f"expected={sorted(DECLARED_DIFFS[experiment_id])}, "
                f"actual={sorted(actual_differences)}"
            )
    return configs


def metrics_at_thresholds(
    station_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label, threshold_cm in (("cm0", 0.0), ("cm1", 1.0), ("cm2", 2.0)):
        retained = [
            row
            for row in station_rows
            if math.isfinite(float(row.get("max_radial_cm", float("nan"))))
            and float(row["max_radial_cm"]) > threshold_cm
        ]
        event_rows = aggregate_event_predictions(
            retained,
            reference_key="mw_catalog",
        )
        metrics = summarize_predictions(
            retained,
            event_rows,
            reference_key="mw_catalog",
        )
        result[label] = {
            "threshold_cm": threshold_cm,
            "event_count": metrics["event_count"],
            "station_count": metrics["station_count"],
            "event_mae_catalog": metrics["event_mae"],
            "event_rmse_catalog": metrics["event_rmse"],
            "event_bias_catalog": metrics["event_bias"],
            "station_mae_catalog": metrics["station_mae"],
            "station_rmse_catalog": metrics["station_rmse"],
            "station_bias_catalog": metrics["station_bias"],
        }
    return result


def select_configuration(
    rows: list[dict[str, Any]],
    *,
    required_seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    experiment_ids = sorted({str(row["experiment_id"]) for row in rows})
    if not experiment_ids:
        raise ValueError("selection has no experiment rows")
    required = set(required_seeds)
    candidates: list[dict[str, Any]] = []
    for experiment_id in experiment_ids:
        successful = [
            row
            for row in rows
            if str(row["experiment_id"]) == experiment_id
            and row.get("status") == "ok"
        ]
        seeds = [int(row["seed"]) for row in successful]
        if set(seeds) != required or len(seeds) != len(required):
            raise ValueError(
                f"{experiment_id} does not have the required seed set"
            )
        maes = [float(row["cm2_event_mae_catalog"]) for row in successful]
        biases = [
            abs(float(row["cm2_event_bias_catalog"]))
            for row in successful
        ]
        parameter_counts = {int(row["parameter_count"]) for row in successful}
        if (
            not all(math.isfinite(value) for value in maes + biases)
            or len(parameter_counts) != 1
        ):
            raise ValueError(f"{experiment_id} has invalid selection values")
        candidates.append(
            {
                "experiment_id": experiment_id,
                "cm2_event_mae_catalog_mean": sum(maes) / len(maes),
                "cm2_absolute_event_bias_catalog_mean": (
                    sum(biases) / len(biases)
                ),
                "parameter_count": parameter_counts.pop(),
                "seeds": sorted(seeds),
            }
        )
    candidates.sort(
        key=lambda item: (
            item["cm2_event_mae_catalog_mean"],
            item["cm2_absolute_event_bias_catalog_mean"],
            item["parameter_count"],
            item["experiment_id"],
        )
    )
    return {**candidates[0], "ranking": candidates}


def _atomic_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_json(path.with_suffix(".json"), rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _runtime_config(
    config: dict[str, Any],
    *,
    run_root: Path,
    seed: int,
    epochs: int | None,
    split_protocol: str | None = None,
    dataset_manifest_path: Path | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["training"]["random_seed"] = int(seed)
    if epochs is not None:
        result["training"]["epochs"] = int(epochs)
    if split_protocol is not None:
        result["training"]["split_protocol"] = split_protocol
    if split_protocol == "within_event_station":
        result["training"]["test_event_fraction"] = 0.15
    result["paths"].update(
        {
            "output_dir": str(run_root),
            "models_dir": str(run_root / "models"),
            "logs_dir": str(run_root / "logs"),
            "results_dir": str(run_root / "results"),
        }
    )
    if dataset_manifest_path is not None:
        result["paths"]["dataset_manifest_path"] = str(
            dataset_manifest_path
        )
    validate_config_v2(result)
    return result


def _parameter_count(config: dict[str, Any]) -> int:
    return sum(
        parameter.numel()
        for parameter in PINNModel(config).parameters()
        if parameter.requires_grad
    )


def _assert_backpropagates(config: dict[str, Any], train_loader: Any) -> None:
    device = get_preferred_device()
    model = PINNModel(config).to(device)
    criterion = _build_stf_rate_criterion(config, device)
    batch = next(iter(train_loader))
    prepared = _prepare_v2_batch(batch, config, device)
    active_workflow = config.get("workflow") == "station_random_shifted_stf"
    if active_workflow:
        heads = model.predict_heads(
            prepared.radial,
            meta=prepared.metadata,
        )
        prediction = heads.stf_encoded
        pred_catalog_mw = heads.catalog_mw
        if prediction.shape[1] != 300:
            raise ValueError("active smoke requires a 300-step STF output")
        if not bool(torch.isfinite(pred_catalog_mw).all()):
            raise FloatingPointError("smoke catalog Mw prediction is non-finite")
    else:
        prediction = model(prepared.radial, meta=prepared.metadata)
        pred_catalog_mw = None
    loss, metrics = criterion(
        prediction,
        pred_catalog_mw=pred_catalog_mw,
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
    )
    values = [float(loss.detach().cpu()), *map(float, metrics.values())]
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError(f"non-finite smoke loss: {metrics}")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    if not gradients or any(gradient is None for gradient in gradients):
        raise FloatingPointError("smoke backward missed trainable parameters")
    if not all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    ):
        raise FloatingPointError("smoke backward produced non-finite gradients")
    del criterion, model, prediction, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _matching_model_dirs(config: dict[str, Any]) -> list[Path]:
    models_root = Path(config["paths"]["models_dir"])
    matches: list[Path] = []
    if not models_root.is_dir():
        return matches
    for candidate in models_root.iterdir():
        snapshot = candidate / "config.yaml"
        if candidate.is_dir() and snapshot.is_file():
            if _load_yaml(snapshot) == config:
                matches.append(candidate)
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)


def _result_from_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    payload = load_full_checkpoint(checkpoint_path)
    state = payload["run_state"]
    model_path = state.get("best_model_path")
    if not model_path or not Path(model_path).is_file():
        raise FileNotFoundError("completed run has no best model")
    models_dir = Path(state["models_dir"])
    return {
        "run_id": state["run_id"],
        "models_dir": models_dir,
        "results_dir": Path(state["results_dir"]),
        "best_model_path": Path(model_path),
        "best_model_swa_path": state.get("best_model_swa_path"),
        "config_snapshot_path": models_dir / "config.yaml",
        "split_manifest_path": models_dir / "split.json",
        "run_manifest_path": models_dir / "run_manifest.json",
        "log_file": Path(state["log_file"]),
        "last_checkpoint_path": checkpoint_path,
        "resumed_from_epoch": payload["completed_epoch"],
    }


def _train_or_resume(
    config: dict[str, Any],
    loaders: tuple[Any, ...],
    *,
    resume: bool,
) -> dict[str, Any]:
    if resume:
        for model_dir in _matching_model_dirs(config):
            last_state = model_dir / "last_state.pth"
            manifest_path = model_dir / "run_manifest.json"
            if not last_state.is_file() or not manifest_path.is_file():
                continue
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            if manifest.get("completed_at_utc"):
                return _result_from_checkpoint(last_state)
            history = model_dir / "resume_history.jsonl"
            resume_count = (
                sum(1 for line in history.read_text(encoding="utf-8").splitlines() if line)
                if history.is_file()
                else 0
            )
            if resume_count >= MAX_VALIDATED_RESUMES:
                raise RuntimeError(
                    f"validated resume limit reached for {model_dir}"
                )
            emergency = model_dir / "emergency_state.pth"
            resume_path = emergency if emergency.is_file() else last_state
            return train(
                config=config,
                data_loaders=loaders,
                resume_checkpoint=resume_path,
            )
    return train(config=config, data_loaders=loaders)


def _verify_training_result(
    config: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    model_path = Path(result["best_model_path"])
    log_path = Path(result["log_file"])
    if not model_path.is_file() or not log_path.is_file():
        raise FileNotFoundError("training did not produce a model and log")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model = PINNModel(config)
    model.load_state_dict(state, strict=True)
    if not all(
        bool(torch.isfinite(value).all())
        for value in state.values()
        if torch.is_tensor(value)
    ):
        raise FloatingPointError("checkpoint contains non-finite tensors")
    with log_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("training log contains no epochs")
    for row in rows:
        for key, value in row.items():
            if key != "Epoch" and not math.isfinite(float(value)):
                raise FloatingPointError(
                    f"training log contains non-finite {key}"
                )
    return {
        "checkpoint_path": str(model_path),
        "checkpoint_sha256": sha256_file(model_path),
        "training_log_path": str(log_path),
        "training_log_sha256": sha256_file(log_path),
        "run_manifest_path": str(result["run_manifest_path"]),
        "split_manifest_path": str(result["split_manifest_path"]),
        "parameter_count": _parameter_count(config),
    }


def _event_directories(event_root: str | Path) -> list[Path]:
    root = Path(event_root)
    paths = [root / name for name in EXTERNAL_EVENT_NAMES]
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing external events: {missing}")
    return paths


def _flat_threshold_metrics(
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        f"{label}_{key}": value
        for label, values in metrics.items()
        for key, value in values.items()
    }


def _completed_row_is_valid(row: dict[str, Any]) -> bool:
    if row.get("status") != "ok":
        return False
    try:
        checkpoint = Path(row["checkpoint_path"])
        training_log = Path(row["training_log_path"])
        run_manifest = Path(row["run_manifest_path"])
        if (
            not checkpoint.is_file()
            or not training_log.is_file()
            or not run_manifest.is_file()
            or sha256_file(checkpoint) != row["checkpoint_sha256"]
            or sha256_file(training_log) != row["training_log_sha256"]
        ):
            return False
        with run_manifest.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        return bool(manifest.get("completed_at_utc")) and (
            manifest.get("checkpoint_sha256") == row["checkpoint_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _prepare_output_root(output_root: Path, *, resume: bool) -> None:
    if not output_root.exists():
        output_root.mkdir(parents=True)
        return
    if not output_root.is_dir():
        raise FileExistsError(f"campaign output is not a directory: {output_root}")
    if resume:
        return
    unexpected = [
        path.name
        for path in output_root.iterdir()
        if path.name != "console.log" or not path.is_file()
    ]
    if unexpected:
        raise FileExistsError(
            f"campaign output is not empty: {output_root}; "
            f"unexpected={sorted(unexpected)}"
        )


def _run_one(
    *,
    experiment_id: str,
    config: dict[str, Any],
    seed: int,
    mode: str,
    output_root: Path,
    epochs: int | None,
    max_events: int | None,
    event_dirs: list[Path] | None,
    resume: bool,
    dataset_manifest_path: Path | None,
) -> dict[str, Any]:
    run_root = output_root / experiment_id / f"seed_{seed}"
    split_protocol = (
        "within_event_station" if mode == "within-event-station" else None
    )
    runtime = _runtime_config(
        config,
        run_root=run_root,
        seed=seed,
        epochs=epochs,
        split_protocol=split_protocol,
        dataset_manifest_path=dataset_manifest_path,
    )
    loaders = get_data_loaders_v2(runtime, max_events=max_events)
    if mode == "smoke":
        _assert_backpropagates(runtime, loaders[0])
    train_result = _train_or_resume(runtime, loaders, resume=resume)
    verified = _verify_training_result(runtime, train_result)
    row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "seed": seed,
        "mode": mode,
        "status": "ok",
        **verified,
    }
    if mode == "external-validation":
        if event_dirs is None:
            raise ValueError("external validation requires event directories")
        external = evaluate_unseen_events(
            event_dirs=event_dirs,
            model_dir=train_result["models_dir"],
            output_dir=run_root / "external_cm0",
            radial_peak_min_cm_override=0.0,
            save_plots=False,
        )
        threshold_metrics = metrics_at_thresholds(external["station_rows"])
        for label, values in threshold_metrics.items():
            if int(values["event_count"]) == 0 or not all(
                math.isfinite(float(values[key]))
                for key in (
                    "event_mae_catalog",
                    "event_rmse_catalog",
                    "event_bias_catalog",
                )
            ):
                raise ValueError(f"{label} external metrics are incomplete")
        _atomic_json(run_root / "external_threshold_metrics.json", threshold_metrics)
        row.update(_flat_threshold_metrics(threshold_metrics))
    elif mode == "within-event-station":
        evaluation = evaluate(
            model_path=train_result["best_model_path"],
            results_run_id=train_result["run_id"],
            config=runtime,
            save_plots=False,
            show_plots=False,
            save_metrics=True,
            test_loader=loaders[2],
        )
        metrics = evaluation["metrics"]
        row.update(
            {
                "within_event_station_event_mae_catalog": metrics["event_mae"],
                "within_event_station_event_bias_catalog": metrics["event_bias"],
                "within_event_station_station_mae_catalog": metrics["station_mae"],
            }
        )
    return row


def run_matrix(
    *,
    configs: dict[str, dict[str, Any]],
    mode: str,
    output_root: Path,
    seeds: tuple[int, ...],
    epochs: int | None,
    max_events: int | None,
    event_root: Path | None,
    resume: bool,
    dataset_manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    _prepare_output_root(output_root, resume=resume)
    summary_csv = output_root / f"{mode}_summary.csv"
    summary_json = summary_csv.with_suffix(".json")
    rows: list[dict[str, Any]] = []
    if resume and summary_json.is_file():
        with summary_json.open("r", encoding="utf-8") as stream:
            rows = json.load(stream)
    completed = {
        (str(row["experiment_id"]), int(row["seed"]))
        for row in rows
        if _completed_row_is_valid(row)
    }
    event_dirs = (
        _event_directories(event_root)
        if event_root is not None
        else None
    )
    run_seeds = (seeds[0],) if mode == "smoke" else seeds
    for experiment_id, config in configs.items():
        for seed in run_seeds:
            key = (experiment_id, seed)
            if key in completed:
                continue
            rows = [
                row
                for row in rows
                if (str(row["experiment_id"]), int(row["seed"])) != key
            ]
            try:
                row = _run_one(
                    experiment_id=experiment_id,
                    config=config,
                    seed=seed,
                    mode=mode,
                    output_root=output_root,
                    epochs=epochs,
                    max_events=max_events,
                    event_dirs=event_dirs,
                    resume=resume,
                    dataset_manifest_path=dataset_manifest_path,
                )
            except BaseException as error:
                row = {
                    "experiment_id": experiment_id,
                    "seed": seed,
                    "mode": mode,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
                rows.append(row)
                _write_rows(summary_csv, rows)
                raise
            rows.append(row)
            _write_rows(summary_csv, rows)
    return rows


def _default_event_root() -> Path | None:
    pinn_root = os.environ.get("PINN_ROOT")
    if not pinn_root:
        return None
    return Path(pinn_root) / "incoming" / "legacy-task17" / "GNSS_EQDATA"


def _default_output_root(mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "outputs_experiments_v2" / f"{mode}-{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed corrected v2 experiment matrix"
    )
    parser.add_argument(
        "--config-dir",
        default=str(PROJECT_ROOT / "configs" / "experiments_v2"),
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("smoke", "external-validation", "within-event-station"),
    )
    parser.add_argument("--output-root")
    parser.add_argument("--event-root", default=_default_event_root())
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config_dir)
    if args.mode == "within-event-station" and config_path.is_file():
        selected = _load_yaml(config_path)
        validate_config_v2(selected)
        configs = {config_path.stem: selected}
    else:
        configs = load_matrix_configs(config_path)
    seeds = tuple(args.seeds)
    if args.mode != "smoke" and seeds != SEEDS:
        raise ValueError(f"formal modes require seeds {SEEDS}")
    epochs = args.epochs
    if args.mode == "smoke" and epochs is None:
        epochs = 2
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be positive")
    max_events = args.max_events if args.mode == "smoke" else None
    output_root = (
        Path(args.output_root)
        if args.output_root
        else _default_output_root(args.mode)
    )
    dataset_manifest_path = (
        Path(args.dataset_manifest).resolve()
        if args.dataset_manifest
        else None
    )
    if dataset_manifest_path is not None and not dataset_manifest_path.is_file():
        raise FileNotFoundError(
            f"dataset manifest not found: {dataset_manifest_path}"
        )
    if args.mode == "external-validation" and dataset_manifest_path is None:
        raise ValueError(
            "external-validation requires --dataset-manifest"
        )
    rows = run_matrix(
        configs=configs,
        mode=args.mode,
        output_root=output_root,
        seeds=seeds,
        epochs=epochs,
        max_events=max_events,
        event_root=(Path(args.event_root) if args.event_root else None),
        resume=args.resume,
        dataset_manifest_path=dataset_manifest_path,
    )
    if args.mode == "external-validation":
        selection = select_configuration(rows, required_seeds=SEEDS)
        selected_id = str(selection["experiment_id"])
        selected_config = configs[selected_id]
        _atomic_json(output_root / "selection.json", selection)
        with (output_root / "V2-SELECTED.yaml").open(
            "w", encoding="utf-8"
        ) as stream:
            yaml.safe_dump(
                selected_config,
                stream,
                allow_unicode=True,
                sort_keys=False,
            )
    _atomic_json(
        output_root / "campaign_manifest.json",
        {
            "mode": args.mode,
            "created_at_utc": utc_now_iso(),
            "git_commit": current_git_commit(PROJECT_ROOT),
            "git_dirty": git_is_dirty(PROJECT_ROOT),
            "seeds": list((seeds[:1] if args.mode == "smoke" else seeds)),
            "matrix_ids": list(configs),
            "row_count": len(rows),
            "dataset_manifest_path": (
                str(dataset_manifest_path) if dataset_manifest_path else ""
            ),
            "dataset_manifest_sha256": (
                sha256_file(dataset_manifest_path)
                if dataset_manifest_path
                else ""
            ),
        },
    )
    print(f"matrix output: {output_root}")


if __name__ == "__main__":
    main()
