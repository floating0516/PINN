from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments import run_phase39_confirmatory_grouped_cv as grouped
from src.data.splits import EventGroupSplit
from src.utils.provenance import (
    current_git_commit,
    git_is_dirty,
    sha256_file,
    utc_now_iso,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "phase39_expanded_grouped_cv.yaml"
)
SEEDS = (17, 42, 73)

TRAIN_EVENTS = (
    "2013p543824",
    "2013p613797",
    "2021p169083",
    "Aegean2014",
    "Chignic2021",
    "ElMayor2010",
    "Eureka2014",
    "Ibaraki2011",
    "Illapel2015",
    "Kaikoura2016",
    "Kilauea2018",
    "Kumamoto2016",
    "Lefkada2015",
    "Melinka2016",
    "Mentawai2010",
    "Miyagi2011A",
    "Miyagi2011B",
    "Nepal2015",
    "Nicoya2012",
    "Simeonof2020",
    "Tehuantepec2017",
    "Tohoku2011",
    "nc73821036",
    "us6000ah9t",
)
VALIDATION_EVENTS = (
    "Anchorage2018",
    "Maule2010",
    "Noto2024",
    "Parkfield2004",
    "RatIslands2014",
    "SandPoint2020",
)
TEST_EVENTS = (
    "2016p661332",
    "Ecuador2016",
    "Iquique2014",
    "Napa2014",
    "Puebla2017",
    "Ridgecrest2019",
    "Tokachi2003",
    "ak014cbigci8",
    "us7000i9bw",
)
LEGACY_TEST_EVENTS = (
    "Ecuador2016",
    "Iquique2014",
    "Napa2014",
    "Puebla2017",
    "Ridgecrest2019",
    "Tokachi2003",
)
NEW_TEST_EVENTS = (
    "2016p661332",
    "ak014cbigci8",
    "us7000i9bw",
)

EXPECTED_RECORD_COUNTS = {
    "train": 1798,
    "validation": 446,
    "test": 450,
}


def _event_sets() -> dict[str, set[str]]:
    roles = {
        "train": set(TRAIN_EVENTS),
        "validation": set(VALIDATION_EVENTS),
        "test": set(TEST_EVENTS),
    }
    if any(not events for events in roles.values()):
        raise ValueError("every fixed split role must contain events")
    role_names = tuple(roles)
    for index, left in enumerate(role_names):
        for right in role_names[index + 1 :]:
            overlap = roles[left] & roles[right]
            if overlap:
                raise ValueError(
                    f"fixed split event overlap between {left} and {right}: "
                    f"{sorted(overlap)}"
                )
    return roles


def build_fixed_split(
    samples: Sequence[Mapping[str, Any]],
) -> tuple[EventGroupSplit, dict[str, Any]]:
    roles = _event_sets()
    dataset_events = {str(sample["event"]) for sample in samples}
    expected_events = set().union(*roles.values())
    if dataset_events != expected_events:
        missing = sorted(expected_events - dataset_events)
        extra = sorted(dataset_events - expected_events)
        raise ValueError(
            f"fixed split event contract changed; missing={missing}, extra={extra}"
        )

    role_by_event = {
        event: role for role, events in roles.items() for event in events
    }
    indices = {role: [] for role in roles}
    for index, sample in enumerate(samples):
        indices[role_by_event[str(sample["event"])]].append(index)

    split = EventGroupSplit(
        train_indices=indices["train"],
        validation_indices=indices["validation"],
        test_indices=indices["test"],
    )
    event_rows = grouped._event_rows(samples)
    events_by_name = {str(row["event"]): row for row in event_rows}
    role_summaries: dict[str, Any] = {}
    for role, events in roles.items():
        rows = [events_by_name[event] for event in sorted(events)]
        role_summaries[role] = {
            "event_count": len(rows),
            "record_count": sum(int(row["n_stations"]) for row in rows),
            "magnitude_minimum": min(
                float(row["magnitude_catalog"]) for row in rows
            ),
            "magnitude_maximum": max(
                float(row["magnitude_catalog"]) for row in rows
            ),
            "events": rows,
        }
        expected_count = EXPECTED_RECORD_COUNTS[role]
        if int(role_summaries[role]["record_count"]) != expected_count:
            raise ValueError(
                f"{role} record count changed: "
                f"expected={expected_count}, "
                f"actual={role_summaries[role]['record_count']}"
            )

    manifest = {
        "schema_version": "phase39-expanded-fixed-split/v1",
        "status": "frozen",
        "assignment_sha256": grouped.split_assignment_sha256(samples, split),
        "total_event_count": len(dataset_events),
        "total_record_count": len(samples),
        "roles": role_summaries,
        "legacy_test_events": list(LEGACY_TEST_EVENTS),
        "new_test_events": list(NEW_TEST_EVENTS),
    }
    if len(samples) != sum(EXPECTED_RECORD_COUNTS.values()):
        raise ValueError("fixed split does not cover all expected records")
    return split, manifest


def _runtime_config(
    base_config: Mapping[str, Any],
    *,
    run_root: Path,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    runtime = copy.deepcopy(dict(base_config))
    runtime["project_name"] = "phase39_expanded_fixed_split"
    runtime["training"]["random_seed"] = int(seed)
    runtime["training"]["epochs"] = int(epochs)
    runtime["training"]["split_protocol"] = "grouped_event"
    runtime["training"]["strict_within_event_split_audit"] = False
    runtime["paths"].update(
        {
            "output_dir": str(run_root),
            "models_dir": str(run_root / "models"),
            "logs_dir": str(run_root / "logs"),
            "results_dir": str(run_root / "results"),
        }
    )
    return runtime


def _best_epoch(log_path: Path) -> tuple[int, float]:
    with log_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    candidates: list[tuple[float, int]] = []
    for row in rows:
        value = float(row["validation_event_mae_catalog"])
        epoch = int(row["Epoch"])
        if math.isfinite(value):
            candidates.append((value, epoch))
    if not candidates:
        raise ValueError(f"training log has no finite validation metric: {log_path}")
    value, epoch = min(candidates)
    return epoch, value


def _load_completed_candidate(
    run_root: Path,
    *,
    seed: int,
    assignment_sha256: str,
) -> dict[str, Any] | None:
    path = run_root / "candidate_summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "seed": int(seed),
        "split_assignment_sha256": assignment_sha256,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                f"completed candidate contract mismatch for {key}: "
                f"expected={value!r}, actual={summary.get(key)!r}"
            )
    for path_key, hash_key in (
        ("best_model_path", "checkpoint_sha256"),
        ("training_log_path", "training_log_sha256"),
        ("split_manifest_path", "split_manifest_sha256"),
        ("config_snapshot_path", "config_snapshot_sha256"),
    ):
        artifact = Path(str(summary[path_key]))
        if not artifact.is_file() or sha256_file(artifact) != summary[hash_key]:
            raise ValueError(f"completed candidate artifact mismatch: {artifact}")
    return summary


def train_candidate(
    *,
    base_config: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    split: EventGroupSplit,
    split_manifest: Mapping[str, Any],
    output_root: Path,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    from src.data.loaders_v2 import get_data_loaders_v2
    from src.training.train import train
    from src.utils.config_v2 import validate_config_v2

    run_root = output_root / "candidates" / f"seed_{seed}"
    completed = _load_completed_candidate(
        run_root,
        seed=seed,
        assignment_sha256=str(split_manifest["assignment_sha256"]),
    )
    if completed is not None:
        print(f"skip completed candidate seed={seed}")
        return completed

    runtime = _runtime_config(
        base_config,
        run_root=run_root,
        seed=seed,
        epochs=epochs,
    )
    validate_config_v2(runtime)
    train_loader, validation_loader, test_loader, loader_manifest = (
        get_data_loaders_v2(runtime, explicit_split=split)
    )
    if loader_manifest["assignment_sha256"] != split_manifest["assignment_sha256"]:
        raise ValueError("loader split hash does not match the frozen role manifest")
    if set(loader_manifest["train_events"]) != set(TRAIN_EVENTS):
        raise ValueError("loader training events do not match the fixed contract")
    if set(loader_manifest["validation_events"]) != set(VALIDATION_EVENTS):
        raise ValueError("loader validation events do not match the fixed contract")
    if set(loader_manifest["test_events"]) != set(TEST_EVENTS):
        raise ValueError("loader test events do not match the fixed contract")

    resume = grouped._resume_checkpoint(run_root)
    print(
        f"train candidate seed={seed} epochs={epochs} "
        f"resume={resume is not None}"
    )
    train_result = train(
        config=runtime,
        data_loaders=(
            train_loader,
            validation_loader,
            test_loader,
            loader_manifest,
        ),
        resume_checkpoint=resume,
    )
    model_path = Path(str(train_result["best_model_path"]))
    log_path = Path(str(train_result["log_file"]))
    best_epoch, best_validation_mae = _best_epoch(log_path)
    returned_mae = float(train_result["best_mw_mae"])
    if not math.isclose(
        best_validation_mae,
        returned_mae,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise ValueError(
            "training log validation minimum does not match train() result: "
            f"log={best_validation_mae}, returned={returned_mae}"
        )

    summary = {
        "status": "complete",
        "completed_at_utc": utc_now_iso(),
        "seed": int(seed),
        "epochs_requested": int(epochs),
        "best_epoch": int(best_epoch),
        "best_validation_event_mae_mw": best_validation_mae,
        "split_assignment_sha256": str(split_manifest["assignment_sha256"]),
        "best_model_path": str(model_path),
        "checkpoint_sha256": sha256_file(model_path),
        "training_log_path": str(log_path),
        "training_log_sha256": sha256_file(log_path),
        "split_manifest_path": str(train_result["split_manifest_path"]),
        "split_manifest_sha256": sha256_file(train_result["split_manifest_path"]),
        "config_snapshot_path": str(train_result["config_snapshot_path"]),
        "config_snapshot_sha256": sha256_file(train_result["config_snapshot_path"]),
        "run_manifest_path": str(train_result["run_manifest_path"]),
        "run_manifest_sha256": sha256_file(train_result["run_manifest_path"]),
        "test_evaluated": False,
    }
    grouped._atomic_json(run_root / "candidate_summary.json", summary)
    return summary


def _event_metrics(
    event_rows: Sequence[Mapping[str, Any]],
    *,
    events: set[str],
) -> dict[str, Any]:
    selected = [row for row in event_rows if str(row["event"]) in events]
    if {str(row["event"]) for row in selected} != events:
        raise ValueError("event metric subgroup coverage is incomplete")
    errors = np.asarray(
        [float(row["error_vs_catalog"]) for row in selected],
        dtype=np.float64,
    )
    return {
        "event_count": len(selected),
        "mae_mw": float(np.mean(np.abs(errors))),
        "rmse_mw": float(np.sqrt(np.mean(errors**2))),
        "bias_mw": float(np.mean(errors)),
        "within_0_2_fraction": float(np.mean(np.abs(errors) <= 0.2)),
        "within_0_3_fraction": float(np.mean(np.abs(errors) <= 0.3)),
    }


def evaluate_selected_candidate(
    *,
    base_config: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    split: EventGroupSplit,
    split_manifest: Mapping[str, Any],
    output_root: Path,
    candidate: Mapping[str, Any],
    epochs: int,
) -> dict[str, Any]:
    from src.data.loaders_v2 import get_data_loaders_v2
    from src.evaluation.evaluate import evaluate

    selected_seed = int(candidate["seed"])
    run_root = output_root / "candidates" / f"seed_{selected_seed}"
    runtime = _runtime_config(
        base_config,
        run_root=run_root,
        seed=selected_seed,
        epochs=epochs,
    )
    _, _, test_loader, loader_manifest = get_data_loaders_v2(
        runtime,
        explicit_split=split,
    )
    if loader_manifest["assignment_sha256"] != split_manifest["assignment_sha256"]:
        raise ValueError("selected candidate loader split hash changed")

    evaluation = evaluate(
        model_path=Path(str(candidate["best_model_path"])),
        config=runtime,
        save_plots=False,
        show_plots=False,
        save_metrics=False,
        test_loader=test_loader,
    )
    station_rows = list(evaluation["station_rows"])
    event_rows = list(evaluation["event_rows"])
    grouped._validate_oof(
        station_rows,
        event_rows,
        samples=samples,
        split=split,
        expected_events=set(TEST_EVENTS),
    )
    station_path = grouped._write_csv(
        output_root / "selected_test_station_predictions.csv",
        station_rows,
    )
    event_path = grouped._write_csv(
        output_root / "selected_test_event_predictions.csv",
        event_rows,
    )
    return {
        "evaluated_at_utc": utc_now_iso(),
        "selected_seed": selected_seed,
        "selected_checkpoint_sha256": str(candidate["checkpoint_sha256"]),
        "test_event_metrics": dict(evaluation["metrics"]),
        "legacy_test_event_metrics": _event_metrics(
            event_rows,
            events=set(LEGACY_TEST_EVENTS),
        ),
        "new_test_event_metrics": _event_metrics(
            event_rows,
            events=set(NEW_TEST_EVENTS),
        ),
        "station_predictions_path": str(station_path),
        "station_predictions_sha256": sha256_file(station_path),
        "event_predictions_path": str(event_path),
        "event_predictions_sha256": sha256_file(event_path),
    }


def _load_completed_campaign(
    output_root: Path,
    *,
    mode: str,
) -> dict[str, Any] | None:
    complete_path = output_root / "COMPLETE"
    summary_path = output_root / "campaign_summary.json"
    if not complete_path.is_file() and not summary_path.is_file():
        return None
    if not complete_path.is_file() or not summary_path.is_file():
        raise ValueError("campaign completion artifacts are incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("mode") != mode:
        raise ValueError("completed campaign summary contract changed")
    if mode == "formal":
        selection_path = Path(str(summary["selection_path"]))
        test = summary["test"]
        artifacts = (
            (selection_path, str(summary["selection_sha256"])),
            (
                Path(str(test["station_predictions_path"])),
                str(test["station_predictions_sha256"]),
            ),
            (
                Path(str(test["event_predictions_path"])),
                str(test["event_predictions_sha256"]),
            ),
        )
        for path, expected_hash in artifacts:
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"completed campaign artifact mismatch: {path}")
    return summary


def run_campaign(
    *,
    config_path: Path,
    output_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    base_config = grouped.build_arm_configs(grouped._load_yaml(config_path))[
        "phase39"
    ]
    samples = grouped._load_dataset_samples(base_config)
    split, split_manifest = build_fixed_split(samples)
    epochs = 1 if smoke else int(base_config["training"]["epochs"])
    seeds = (SEEDS[0],) if smoke else SEEDS

    protocol = {
        "schema_version": "phase39-expanded-fixed-split-campaign/v1",
        "status": "frozen",
        "mode": "smoke" if smoke else "formal",
        "created_at_utc": utc_now_iso(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_path": str(base_config["paths"]["data_path"]),
        "dataset_sha256": sha256_file(Path(base_config["paths"]["data_path"])),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "git_dirty": git_is_dirty(PROJECT_ROOT),
        "seeds": list(seeds),
        "epochs": int(epochs),
        "selection_metric": "fixed_validation_event_mae_catalog",
        "test_policy": "evaluate_selected_seed_once",
        "lambda_synth": 0.5,
        "split_assignment_sha256": str(split_manifest["assignment_sha256"]),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "protocol.json"
    split_path = output_root / "fixed_split_manifest.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        stable_keys = set(protocol) - {"created_at_utc", "git_dirty"}
        if any(existing.get(key) != protocol.get(key) for key in stable_keys):
            raise ValueError("existing fixed-split campaign protocol changed")
        if not split_path.is_file() or sha256_file(split_path) != existing.get(
            "fixed_split_manifest_sha256"
        ):
            raise ValueError("existing fixed split manifest hash changed")
    else:
        grouped._atomic_json(split_path, split_manifest)
        protocol["fixed_split_manifest_sha256"] = sha256_file(split_path)
        grouped._atomic_json(protocol_path, protocol)

    completed = _load_completed_campaign(
        output_root,
        mode="smoke" if smoke else "formal",
    )
    if completed is not None:
        print(f"skip completed {'smoke' if smoke else 'formal'} campaign")
        return completed

    candidates: list[dict[str, Any]] = []
    for seed in seeds:
        candidates.append(
            train_candidate(
                base_config=base_config,
                samples=samples,
                split=split,
                split_manifest=split_manifest,
                output_root=output_root,
                seed=seed,
                epochs=epochs,
            )
        )

    if smoke:
        summary = {
            "status": "complete",
            "mode": "smoke",
            "completed_at_utc": utc_now_iso(),
            "candidate_count": len(candidates),
            "test_evaluated": False,
            "candidate": candidates[0],
        }
        grouped._atomic_json(output_root / "campaign_summary.json", summary)
        grouped._atomic_write(output_root / "COMPLETE", b"\n")
        return summary

    candidates.sort(
        key=lambda row: (
            float(row["best_validation_event_mae_mw"]),
            int(row["seed"]),
        )
    )
    selected = candidates[0]
    selection = {
        "selected_at_utc": utc_now_iso(),
        "selection_metric": "fixed_validation_event_mae_catalog",
        "selected_seed": int(selected["seed"]),
        "selected_checkpoint_path": str(selected["best_model_path"]),
        "selected_checkpoint_sha256": str(selected["checkpoint_sha256"]),
        "selected_validation_event_mae_mw": float(
            selected["best_validation_event_mae_mw"]
        ),
        "candidates": [
            {
                "seed": int(row["seed"]),
                "best_epoch": int(row["best_epoch"]),
                "best_validation_event_mae_mw": float(
                    row["best_validation_event_mae_mw"]
                ),
                "checkpoint_sha256": str(row["checkpoint_sha256"]),
            }
            for row in candidates
        ],
    }
    selection_path = grouped._atomic_json(output_root / "selection.json", selection)
    test_result = evaluate_selected_candidate(
        base_config=base_config,
        samples=samples,
        split=split,
        split_manifest=split_manifest,
        output_root=output_root,
        candidate=selected,
        epochs=epochs,
    )
    summary = {
        "status": "complete",
        "mode": "formal",
        "completed_at_utc": utc_now_iso(),
        "candidate_count": len(candidates),
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "selection": selection,
        "test": test_result,
    }
    grouped._atomic_json(output_root / "campaign_summary.json", summary)
    grouped._atomic_write(output_root / "COMPLETE", b"\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Phase39 on the frozen expanded 24/6/9 event split and "
            "evaluate only the validation-selected seed."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="train seed 17 for one epoch without evaluating the test cohort",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_campaign(
        config_path=args.config.resolve(),
        output_root=args.output_root.resolve(),
        smoke=bool(args.smoke),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
