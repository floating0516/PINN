from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import EventGroupSplit
from src.utils.provenance import (
    current_git_commit,
    git_is_dirty,
    sha256_file,
    sha256_if_file,
    utc_now_iso,
)


SEEDS = (17, 42, 73)
ARMS = ("phase39", "no_synth")
N_FOLDS = 5
ARM_DIFF_PATH = "training.stf_rate_loss.lambda_synth"
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "phase39_confirmatory_grouped_cv.yaml"
)
EXTERNAL_EVENTS_CLOSED = (
    "iquique-aftershock-2014-chile",
    "nepal-aftershock-2015",
    "kodiak-2018-alaska",
    "samos-2020-greece",
    "luding-2022-china",
    "xizang-2025-southern-tibetan-plateau",
    "myanmar-2025-mandalay",
    "sand-point-2025-alaska",
)


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


def _atomic_write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def _atomic_json(path: Path, payload: Any) -> Path:
    return _atomic_write(path, _json_bytes(payload))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def _coerce_csv_row(row: Mapping[str, str]) -> dict[str, Any]:
    integer_fields = {"fold", "seed", "n_stations"}
    text_fields = {"event", "station", "arm"}
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key in text_fields:
            result[key] = str(value)
        elif key in integer_fields:
            result[key] = int(value)
        elif value == "":
            result[key] = ""
        else:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [_coerce_csv_row(row) for row in csv.DictReader(stream)]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def config_diff_paths(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    prefix: str = "",
) -> set[str]:
    differences: set[str] = set()
    for key in set(left) | set(right):
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in left or key not in right:
            differences.add(path)
            continue
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
            differences.update(config_diff_paths(left_value, right_value, path))
        elif left_value != right_value:
            differences.add(path)
    return differences


def build_arm_configs(base_config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    phase39 = copy.deepcopy(dict(base_config))
    phase39.pop("campaign", None)
    phase39.pop("workflow", None)
    training = phase39["training"]
    training["split_protocol"] = "grouped_event"
    training["validation_event_fraction"] = 0.2
    training["test_event_fraction"] = 0.2
    training["strict_within_event_split_audit"] = False
    loss = training["stf_rate_loss"]
    required = {
        "radiation_coefficient_contract": "glehman_scalar",
        "synth_polarity_mode": "global_invariant",
        "include_intermediate_field": False,
    }
    for key, expected in required.items():
        if loss.get(key) != expected:
            raise ValueError(
                f"Phase39 requires training.stf_rate_loss.{key}={expected!r}"
            )
    if float(loss.get("lambda_synth", float("nan"))) != 0.5:
        raise ValueError("Phase39 requires lambda_synth=0.5")

    no_synth = copy.deepcopy(phase39)
    no_synth["training"]["stf_rate_loss"]["lambda_synth"] = 0.0
    differences = config_diff_paths(phase39, no_synth)
    if differences != {ARM_DIFF_PATH}:
        raise ValueError(
            "arm configs do not isolate lambda_synth: "
            f"actual={sorted(differences)}"
        )
    return {"phase39": phase39, "no_synth": no_synth}


def _event_rows(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_event: dict[str, dict[str, Any]] = {}
    sample_keys: set[tuple[str, str]] = set()
    for sample in samples:
        event = str(sample["event"])
        station = str(sample["station"])
        key = (event, station)
        if key in sample_keys:
            raise ValueError(f"duplicate event/station sample key: {event}::{station}")
        sample_keys.add(key)
        magnitude = float(sample["magnitude_catalog"])
        if not math.isfinite(magnitude):
            raise ValueError(f"non-finite catalog magnitude for event {event}")
        row = by_event.setdefault(
            event,
            {"event": event, "magnitudes": [], "n_stations": 0},
        )
        row["magnitudes"].append(magnitude)
        row["n_stations"] += 1

    rows: list[dict[str, Any]] = []
    for event, row in by_event.items():
        magnitudes = np.asarray(row["magnitudes"], dtype=np.float64)
        if float(np.ptp(magnitudes)) > 1.0e-6:
            raise ValueError(f"catalog magnitude is inconsistent within event {event}")
        rows.append(
            {
                "event": event,
                "magnitude_catalog": float(np.median(magnitudes)),
                "n_stations": int(row["n_stations"]),
            }
        )
    return sorted(rows, key=lambda row: (row["magnitude_catalog"], row["event"]))


def build_event_folds(
    samples: Sequence[Mapping[str, Any]],
    *,
    n_folds: int = N_FOLDS,
) -> dict[str, Any]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    events = _event_rows(samples)
    if len(events) < n_folds:
        raise ValueError("event count must be at least n_folds")

    station_totals = [0] * n_folds
    event_totals = [0] * n_folds
    assigned: dict[str, int] = {}
    strata: dict[str, int] = {}
    for stratum, start in enumerate(range(0, len(events), n_folds)):
        chunk = events[start : start + n_folds]
        available = set(range(n_folds))
        for row in sorted(
            chunk,
            key=lambda value: (-value["n_stations"], value["event"]),
        ):
            fold = min(
                available,
                key=lambda candidate: (
                    station_totals[candidate],
                    event_totals[candidate],
                    candidate,
                ),
            )
            event = str(row["event"])
            assigned[event] = fold
            strata[event] = stratum
            station_totals[fold] += int(row["n_stations"])
            event_totals[fold] += 1
            available.remove(fold)

    event_assignments = [
        {**row, "fold": assigned[str(row["event"])], "stratum": strata[str(row["event"])]}
        for row in events
    ]
    canonical = [
        {
            "event": row["event"],
            "fold": row["fold"],
            "magnitude_catalog": row["magnitude_catalog"],
            "n_stations": row["n_stations"],
        }
        for row in event_assignments
    ]
    digest = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    folds: list[dict[str, Any]] = []
    for fold in range(n_folds):
        members = [row for row in event_assignments if row["fold"] == fold]
        magnitudes = [float(row["magnitude_catalog"]) for row in members]
        folds.append(
            {
                "fold": fold,
                "n_events": len(members),
                "n_stations": sum(int(row["n_stations"]) for row in members),
                "events": sorted(str(row["event"]) for row in members),
                "magnitude_minimum": min(magnitudes),
                "magnitude_mean": float(np.mean(magnitudes)),
                "magnitude_maximum": max(magnitudes),
            }
        )
    return {
        "n_folds": n_folds,
        "n_events": len(events),
        "n_records": len(samples),
        "assignment_sha256": digest,
        "events": event_assignments,
        "folds": folds,
    }


def _fold_lookup(fold_manifest: Mapping[str, Any]) -> dict[str, int]:
    lookup = {
        str(row["event"]): int(row["fold"])
        for row in fold_manifest["events"]
    }
    if len(lookup) != int(fold_manifest["n_events"]):
        raise ValueError("fold manifest contains duplicate events")
    return lookup


def make_outer_split(
    samples: Sequence[Mapping[str, Any]],
    fold_manifest: Mapping[str, Any],
    *,
    outer_fold: int,
) -> EventGroupSplit:
    n_folds = int(fold_manifest["n_folds"])
    if not 0 <= outer_fold < n_folds:
        raise ValueError(f"outer_fold must be in [0, {n_folds})")
    lookup = _fold_lookup(fold_manifest)
    sample_events = {str(sample["event"]) for sample in samples}
    if sample_events != set(lookup):
        raise ValueError("fold manifest events do not match the dataset")
    validation_fold = (outer_fold + 1) % n_folds
    roles = [lookup[str(sample["event"])] for sample in samples]
    return EventGroupSplit(
        train_indices=[
            index
            for index, fold in enumerate(roles)
            if fold not in {outer_fold, validation_fold}
        ],
        validation_indices=[
            index for index, fold in enumerate(roles) if fold == validation_fold
        ],
        test_indices=[
            index for index, fold in enumerate(roles) if fold == outer_fold
        ],
    )


def split_assignment_sha256(
    samples: Sequence[Mapping[str, Any]],
    split: EventGroupSplit,
) -> str:
    sample_keys = [
        f"{sample['event']}::{sample['station']}" for sample in samples
    ]
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("duplicate event/station sample key in dataset")
    payload = {
        "train": sorted(sample_keys[index] for index in split.train_indices),
        "validation": sorted(
            sample_keys[index] for index in split.validation_indices
        ),
        "test": sorted(sample_keys[index] for index in split.test_indices),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _expected_test_events(
    fold_manifest: Mapping[str, Any],
    fold: int,
) -> set[str]:
    return {
        str(row["event"])
        for row in fold_manifest["events"]
        if int(row["fold"]) == fold
    }


def _runtime_config(
    arm_config: Mapping[str, Any],
    *,
    run_root: Path,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    runtime = copy.deepcopy(dict(arm_config))
    runtime["training"]["random_seed"] = int(seed)
    runtime["training"]["epochs"] = int(epochs)
    runtime["paths"].update(
        {
            "output_dir": str(run_root),
            "models_dir": str(run_root / "models"),
            "logs_dir": str(run_root / "logs"),
            "results_dir": str(run_root / "results"),
        }
    )
    return runtime


def _validate_artifact_hash(
    summary: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
) -> Path:
    path = Path(str(summary[path_key]))
    if not path.is_file():
        raise ValueError(f"completed run artifact is missing: {path}")
    actual = sha256_file(path)
    expected = str(summary[hash_key])
    if actual != expected:
        raise ValueError(
            f"completed run artifact hash mismatch for {path}: "
            f"expected={expected}, actual={actual}"
        )
    return path


def load_completed_run(
    run_root: Path,
    *,
    fold: int,
    arm: str,
    seed: int,
    split_assignment_sha256: str,
    expected_events: set[str],
) -> list[dict[str, Any]] | None:
    summary_path = run_root / "run_summary.json"
    if not summary_path.is_file():
        return None
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    expected_contract = {
        "status": "complete",
        "fold": fold,
        "arm": arm,
        "seed": seed,
        "split_assignment_sha256": split_assignment_sha256,
    }
    for key, expected in expected_contract.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"completed run contract mismatch for {key}: "
                f"expected={expected!r}, actual={summary.get(key)!r}"
            )
    event_path = _validate_artifact_hash(
        summary,
        path_key="event_oof_path",
        hash_key="event_oof_sha256",
    )
    _validate_artifact_hash(
        summary,
        path_key="station_oof_path",
        hash_key="station_oof_sha256",
    )
    _validate_artifact_hash(
        summary,
        path_key="best_model_path",
        hash_key="checkpoint_sha256",
    )
    rows = _read_csv(event_path)
    if {str(row["event"]) for row in rows} != expected_events:
        raise ValueError("completed run OOF coverage does not match its test fold")
    if len(rows) != len(expected_events):
        raise ValueError("completed run contains duplicate event OOF rows")
    for row in rows:
        if (
            int(row["fold"]) != fold
            or str(row["arm"]) != arm
            or int(row["seed"]) != seed
        ):
            raise ValueError("completed run OOF row contract mismatch")
    return rows


def _finite_float(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} in OOF row")
    return value


def _validate_oof(
    station_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    *,
    samples: Sequence[Mapping[str, Any]],
    split: EventGroupSplit,
    expected_events: set[str],
) -> None:
    expected_station_keys = {
        (str(samples[index]["event"]), str(samples[index]["station"]))
        for index in split.test_indices
    }
    actual_station_keys = {
        (str(row["event"]), str(row["station"])) for row in station_rows
    }
    if actual_station_keys != expected_station_keys:
        raise ValueError("station OOF coverage does not match the test split")
    if len(station_rows) != len(expected_station_keys):
        raise ValueError("station OOF rows contain duplicates")
    actual_events = {str(row["event"]) for row in event_rows}
    if actual_events != expected_events or len(event_rows) != len(expected_events):
        raise ValueError("event OOF coverage does not match the test split")
    for row in event_rows:
        _finite_float(row, "mw_pred_median")
        _finite_float(row, "mw_catalog")
        _finite_float(row, "error_vs_catalog")


def _enrich_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold: int,
    arm: str,
    seed: int,
) -> list[dict[str, Any]]:
    return [
        {**dict(row), "fold": fold, "arm": arm, "seed": seed}
        for row in rows
    ]


def _resume_checkpoint(run_root: Path) -> Path | None:
    checkpoints = sorted((run_root / "models").glob("*/last_state.pth"))
    if len(checkpoints) > 1:
        raise ValueError(f"multiple resumable checkpoints found under {run_root}")
    if checkpoints:
        return checkpoints[0]
    if run_root.exists() and any(run_root.iterdir()):
        raise ValueError(
            f"partial run has no resumable checkpoint: {run_root}"
        )
    return None


def run_one(
    *,
    samples: Sequence[Mapping[str, Any]],
    fold_manifest: Mapping[str, Any],
    arm_config: Mapping[str, Any],
    output_root: Path,
    fold: int,
    arm: str,
    seed: int,
    epochs: int,
) -> list[dict[str, Any]]:
    from src.data.loaders_v2 import get_data_loaders_v2
    from src.evaluation.evaluate import evaluate
    from src.training.train import train
    from src.utils.config_v2 import validate_config_v2

    split = make_outer_split(samples, fold_manifest, outer_fold=fold)
    assignment_sha256 = split_assignment_sha256(samples, split)
    expected_events = _expected_test_events(fold_manifest, fold)
    run_root = output_root / f"fold_{fold}" / arm / f"seed_{seed}"
    completed = load_completed_run(
        run_root,
        fold=fold,
        arm=arm,
        seed=seed,
        split_assignment_sha256=assignment_sha256,
        expected_events=expected_events,
    )
    if completed is not None:
        print(f"skip completed run: fold={fold} arm={arm} seed={seed}")
        return completed

    runtime = _runtime_config(
        arm_config,
        run_root=run_root,
        seed=seed,
        epochs=epochs,
    )
    validate_config_v2(runtime)
    loaders = get_data_loaders_v2(runtime, explicit_split=split)
    train_loader, validation_loader, test_loader, split_manifest = loaders
    if split_manifest["assignment_sha256"] != assignment_sha256:
        raise ValueError("loader split hash does not match the frozen protocol")
    if set(split_manifest["test_events"]) != expected_events:
        raise ValueError("loader test events do not match the frozen outer fold")

    resume = _resume_checkpoint(run_root)
    print(
        f"run fold={fold} arm={arm} seed={seed} epochs={epochs} "
        f"resume={resume is not None}"
    )
    train_result = train(
        config=runtime,
        data_loaders=(
            train_loader,
            validation_loader,
            test_loader,
            split_manifest,
        ),
        resume_checkpoint=resume,
    )
    model_path = Path(str(train_result["best_model_path"]))
    if not model_path.is_file():
        raise ValueError("training completed without a best model checkpoint")
    evaluation = evaluate(
        model_path=model_path,
        config=runtime,
        save_plots=False,
        show_plots=False,
        save_metrics=False,
        test_loader=test_loader,
    )
    station_rows = list(evaluation["station_rows"])
    event_rows = list(evaluation["event_rows"])
    _validate_oof(
        station_rows,
        event_rows,
        samples=samples,
        split=split,
        expected_events=expected_events,
    )
    enriched_station_rows = _enrich_rows(
        station_rows,
        fold=fold,
        arm=arm,
        seed=seed,
    )
    enriched_event_rows = _enrich_rows(
        event_rows,
        fold=fold,
        arm=arm,
        seed=seed,
    )
    station_path = _write_csv(run_root / "station_oof.csv", enriched_station_rows)
    event_path = _write_csv(run_root / "event_oof.csv", enriched_event_rows)
    summary = {
        "status": "complete",
        "completed_at_utc": utc_now_iso(),
        "fold": fold,
        "arm": arm,
        "seed": seed,
        "epochs": epochs,
        "train_event_count": len(split_manifest["train_events"]),
        "validation_event_count": len(split_manifest["validation_events"]),
        "test_event_count": len(split_manifest["test_events"]),
        "split_assignment_sha256": assignment_sha256,
        "loader_split_manifest_path": str(train_result["split_manifest_path"]),
        "loader_split_manifest_sha256": sha256_if_file(
            train_result["split_manifest_path"]
        ),
        "config_snapshot_path": str(train_result["config_snapshot_path"]),
        "config_snapshot_sha256": sha256_file(train_result["config_snapshot_path"]),
        "best_model_path": str(model_path),
        "checkpoint_sha256": sha256_file(model_path),
        "training_log_path": str(train_result["log_file"]),
        "training_log_sha256": sha256_file(train_result["log_file"]),
        "station_oof_path": str(station_path),
        "station_oof_sha256": sha256_file(station_path),
        "event_oof_path": str(event_path),
        "event_oof_sha256": sha256_file(event_path),
        "event_metrics": evaluation["metrics"],
    }
    _atomic_json(run_root / "run_summary.json", summary)
    return enriched_event_rows


def paired_event_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_events: set[str],
    seeds: Sequence[int],
    n_bootstrap: int,
    n_sign_flips: int,
    seed: int,
) -> dict[str, Any]:
    if n_bootstrap <= 0 or n_sign_flips <= 0:
        raise ValueError("bootstrap and sign-flip sample counts must be positive")
    required = {
        (event, int(run_seed), arm)
        for event in expected_events
        for run_seed in seeds
        for arm in ARMS
    }
    lookup: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["event"]), int(row["seed"]), str(row["arm"]))
        if key in lookup:
            raise ValueError(f"duplicate OOF coverage row: {key}")
        lookup[key] = row
    if set(lookup) != required:
        missing = sorted(required - set(lookup))
        extra = sorted(set(lookup) - required)
        raise ValueError(
            f"OOF coverage mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    event_deltas: list[dict[str, Any]] = []
    seed_deltas: dict[int, list[float]] = {int(value): [] for value in seeds}
    for event in sorted(expected_events):
        matched: dict[str, float] = {}
        per_seed: list[dict[str, Any]] = []
        for run_seed in seeds:
            phase_error = abs(
                _finite_float(
                    lookup[(event, int(run_seed), "phase39")],
                    "error_vs_catalog",
                )
            )
            no_synth_error = abs(
                _finite_float(
                    lookup[(event, int(run_seed), "no_synth")],
                    "error_vs_catalog",
                )
            )
            delta = phase_error - no_synth_error
            matched[str(int(run_seed))] = delta
            seed_deltas[int(run_seed)].append(delta)
            per_seed.append(
                {
                    "seed": int(run_seed),
                    "phase39_absolute_error_mw": phase_error,
                    "no_synth_absolute_error_mw": no_synth_error,
                    "delta_mw": delta,
                }
            )
        event_deltas.append(
            {
                "event": event,
                "mean_delta_mw": float(np.mean(list(matched.values()))),
                "seed_delta_mw": matched,
                "matched_rows": per_seed,
            }
        )

    deltas = np.asarray(
        [row["mean_delta_mw"] for row in event_deltas],
        dtype=np.float64,
    )
    observed = float(np.mean(deltas))
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(deltas),
        size=(n_bootstrap, len(deltas)),
    )
    bootstrap_means = np.mean(deltas[indices], axis=1)
    signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(n_sign_flips, len(deltas)),
    )
    sign_flip_means = np.mean(signs * deltas, axis=1)
    seed_means = {
        str(run_seed): float(np.mean(values))
        for run_seed, values in seed_deltas.items()
    }
    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))
    promotion = {
        "mean_delta_below_zero": observed < 0.0,
        "bootstrap_ci_upper_below_zero": ci_upper < 0.0,
        "all_seed_means_below_zero": all(value < 0.0 for value in seed_means.values()),
    }
    promotion["passed"] = all(promotion.values())

    loeo: list[dict[str, Any]] = []
    for index, row in enumerate(event_deltas):
        retained = np.delete(deltas, index)
        leave_one_out_mean = float(np.mean(retained))
        loeo.append(
            {
                "event_removed": row["event"],
                "mean_delta_without_event_mw": leave_one_out_mean,
                "change_from_full_mean_mw": leave_one_out_mean - observed,
            }
        )
    loeo.sort(key=lambda row: abs(row["change_from_full_mean_mw"]), reverse=True)
    return {
        "statistical_unit": "event",
        "delta_definition": "abs_error_phase39_minus_abs_error_no_synth",
        "n_events": len(deltas),
        "seeds": [int(value) for value in seeds],
        "mean_delta_mw": observed,
        "bootstrap_samples": n_bootstrap,
        "bootstrap_ci_lower_mw": ci_lower,
        "bootstrap_ci_upper_mw": ci_upper,
        "seed_mean_delta_mw": seed_means,
        "sign_flip_samples": n_sign_flips,
        "sign_flip_p_one_sided_phase39_better": float(
            (1 + np.sum(sign_flip_means <= observed)) / (n_sign_flips + 1)
        ),
        "sign_flip_p_two_sided": float(
            (1 + np.sum(np.abs(sign_flip_means) >= abs(observed)))
            / (n_sign_flips + 1)
        ),
        "promotion_gate": promotion,
        "event_deltas": event_deltas,
        "leave_one_event_out_influence": loeo,
    }


def _protocol_payload(
    *,
    config_path: Path,
    base_config: Mapping[str, Any],
    fold_manifest: Mapping[str, Any],
    commit: str,
    mode: str,
    folds: Sequence[int],
    seeds: Sequence[int],
    epochs: int,
) -> dict[str, Any]:
    data_path = Path(str(base_config["paths"]["data_path"]))
    stf_path = Path(str(base_config["dataset"]["stf"]["path"]))
    return {
        "protocol_version": 1,
        "mode": mode,
        "git_commit": commit,
        "base_config_path": str(config_path),
        "base_config_sha256": sha256_file(config_path),
        "training_data_path": str(data_path),
        "training_data_sha256": sha256_file(data_path),
        "stf_directory": str(stf_path),
        "fold_assignment_sha256": fold_manifest["assignment_sha256"],
        "folds": list(folds),
        "arms": list(ARMS),
        "seeds": [int(value) for value in seeds],
        "epochs": epochs,
        "outer_test_role": "fold_k",
        "inner_validation_role": "fold_(k+1)_mod_5",
        "arm_diff_paths": [ARM_DIFF_PATH],
        "primary_reference": "catalog",
        "statistical_unit": "event",
        "external_events_policy": "closed_not_loaded_not_scored",
        "external_events_closed": list(EXTERNAL_EVENTS_CLOSED),
    }


def _initialize_output_root(
    output_root: Path,
    *,
    protocol: Mapping[str, Any],
    fold_manifest: Mapping[str, Any],
    base_config: Mapping[str, Any],
) -> None:
    protocol_path = output_root / "protocol.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not protocol_path.is_file():
            raise ValueError(
                f"non-empty output root has no protocol.json: {output_root}"
            )
        with protocol_path.open("r", encoding="utf-8") as stream:
            frozen = json.load(stream)
        if frozen != dict(protocol):
            raise ValueError("existing campaign protocol does not match this invocation")
        fold_path = output_root / "event_folds.json"
        if not fold_path.is_file() or sha256_file(fold_path) != str(
            frozen["event_folds_file_sha256"]
        ):
            raise ValueError("existing event fold artifact hash mismatch")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    fold_path = _atomic_json(output_root / "event_folds.json", fold_manifest)
    config_path = output_root / "base_phase39_config.yaml"
    _atomic_write(
        config_path,
        yaml.safe_dump(
            dict(base_config),
            allow_unicode=True,
            sort_keys=False,
        ).encode("utf-8"),
    )
    frozen_protocol = {
        **dict(protocol),
        "event_folds_file_sha256": sha256_file(fold_path),
        "base_config_copy_sha256": sha256_file(config_path),
    }
    _atomic_json(protocol_path, frozen_protocol)


def _load_dataset_samples(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    from src.data.dataset_v2 import CorrectedEarthquakeDataset

    dataset = CorrectedEarthquakeDataset(copy.deepcopy(dict(config)))
    return [dict(sample) for sample in dataset.samples]


def run_campaign(
    *,
    config_path: Path,
    output_root: Path,
    smoke: bool,
    n_bootstrap: int,
    n_sign_flips: int,
) -> dict[str, Any]:
    base_config = _load_yaml(config_path)
    arms = build_arm_configs(base_config)
    from src.utils.config_v2 import validate_config_v2

    for config in arms.values():
        validate_config_v2(config)
    samples = _load_dataset_samples(arms["phase39"])
    fold_manifest = build_event_folds(samples, n_folds=N_FOLDS)
    if int(fold_manifest["n_events"]) != 31:
        raise ValueError(
            f"formal internal cohort must contain 31 events, got {fold_manifest['n_events']}"
        )

    folds = (0,) if smoke else tuple(range(N_FOLDS))
    seeds = (SEEDS[0],) if smoke else SEEDS
    epochs = 1 if smoke else int(arms["phase39"]["training"]["epochs"])
    commit = current_git_commit(PROJECT_ROOT)
    dirty = git_is_dirty(PROJECT_ROOT)
    if dirty and not smoke:
        raise ValueError("formal campaign requires a clean git worktree")
    protocol = _protocol_payload(
        config_path=config_path,
        base_config=base_config,
        fold_manifest=fold_manifest,
        commit=commit,
        mode="smoke" if smoke else "formal",
        folds=folds,
        seeds=seeds,
        epochs=epochs,
    )
    protocol = {**protocol, "git_dirty": dirty}
    if output_root.exists() and (output_root / "protocol.json").is_file():
        with (output_root / "protocol.json").open("r", encoding="utf-8") as stream:
            existing_protocol = json.load(stream)
        protocol["event_folds_file_sha256"] = existing_protocol.get(
            "event_folds_file_sha256"
        )
        protocol["base_config_copy_sha256"] = existing_protocol.get(
            "base_config_copy_sha256"
        )
    _initialize_output_root(
        output_root,
        protocol=protocol,
        fold_manifest=fold_manifest,
        base_config=base_config,
    )

    all_rows: list[dict[str, Any]] = []
    total_runs = len(folds) * len(seeds) * len(ARMS)
    completed_runs = 0
    for fold in folds:
        for run_seed in seeds:
            for arm in ARMS:
                all_rows.extend(
                    run_one(
                        samples=samples,
                        fold_manifest=fold_manifest,
                        arm_config=arms[arm],
                        output_root=output_root,
                        fold=fold,
                        arm=arm,
                        seed=run_seed,
                        epochs=epochs,
                    )
                )
                completed_runs += 1
                _atomic_json(
                    output_root / "campaign_status.json",
                    {
                        "updated_at_utc": utc_now_iso(),
                        "completed_runs": completed_runs,
                        "total_runs": total_runs,
                        "mode": "smoke" if smoke else "formal",
                    },
                )

    all_rows.sort(
        key=lambda row: (
            str(row["event"]),
            int(row["seed"]),
            str(row["arm"]),
        )
    )
    oof_path = _write_csv(output_root / "oof_event_predictions.csv", all_rows)
    expected_events = set().union(
        *(_expected_test_events(fold_manifest, fold) for fold in folds)
    )
    statistics = paired_event_statistics(
        all_rows,
        expected_events=expected_events,
        seeds=seeds,
        n_bootstrap=n_bootstrap,
        n_sign_flips=n_sign_flips,
        seed=20260812,
    )
    statistics["decision_role"] = "smoke_only" if smoke else "confirmatory"
    statistics["promotion_gate_applicable"] = not smoke
    statistics_path = _atomic_json(output_root / "paired_statistics.json", statistics)
    summary = {
        "status": "complete",
        "completed_at_utc": utc_now_iso(),
        "mode": "smoke" if smoke else "formal",
        "completed_runs": completed_runs,
        "total_runs": total_runs,
        "event_count": len(expected_events),
        "oof_event_predictions_path": str(oof_path),
        "oof_event_predictions_sha256": sha256_file(oof_path),
        "paired_statistics_path": str(statistics_path),
        "paired_statistics_sha256": sha256_file(statistics_path),
        "promotion_gate_applicable": not smoke,
        "promotion_gate": statistics["promotion_gate"],
    }
    _atomic_json(output_root / "campaign_summary.json", summary)
    _atomic_write(output_root / "COMPLETE", b"\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase39 versus matched no-synth confirmatory grouped CV"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run fold 0, seed 17, both arms, one epoch",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--sign-flips", type=int, default=100_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_campaign(
        config_path=args.config.resolve(),
        output_root=args.output_root.resolve(),
        smoke=bool(args.smoke),
        n_bootstrap=int(args.bootstrap_samples),
        n_sign_flips=int(args.sign_flips),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
