from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments import run_phase39_confirmatory_grouped_cv as grouped
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
N_FOLDS = 5
EXPECTED_EVENTS = 39
EXPECTED_RECORDS = 2694


def _metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        reference,
        dtype=np.float64,
    )
    return {
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "bias_mw": float(np.mean(error)),
        "within_0_2_fraction": float(np.mean(np.abs(error) <= 0.2)),
        "within_0_3_fraction": float(np.mean(np.abs(error) <= 0.3)),
    }


def _aggregate_across_seeds(
    rows: Sequence[Mapping[str, Any]],
    *,
    identity_fields: Sequence[str],
    prediction_field: str,
    reference_field: str,
) -> list[dict[str, Any]]:
    grouped_rows: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in identity_fields)
        grouped_rows.setdefault(key, []).append(row)
    result = []
    for key, matched in sorted(grouped_rows.items()):
        if {int(row["seed"]) for row in matched} != set(SEEDS):
            raise ValueError(f"incomplete seed coverage for {key}")
        references = np.asarray(
            [float(row[reference_field]) for row in matched],
            dtype=np.float64,
        )
        if float(np.ptp(references)) > 1.0e-6:
            raise ValueError(f"inconsistent reference for {key}")
        predictions = np.asarray(
            [float(row[prediction_field]) for row in matched],
            dtype=np.float64,
        )
        prediction = float(np.median(predictions))
        result.append(
            {
                **dict(zip(identity_fields, key)),
                reference_field: float(np.median(references)),
                prediction_field: prediction,
                "prediction_seed_mean": float(np.mean(predictions)),
                "prediction_seed_std": float(np.std(predictions)),
                "error_vs_catalog": prediction - float(np.median(references)),
            }
        )
    return result


def _validate_coverage(
    event_rows: Sequence[Mapping[str, Any]],
    station_rows: Sequence[Mapping[str, Any]],
) -> None:
    event_keys = {
        (str(row["event"]), int(row["seed"])) for row in event_rows
    }
    if len(event_rows) != EXPECTED_EVENTS * len(SEEDS):
        raise ValueError("event OOF row count changed")
    events = {str(row["event"]) for row in event_rows}
    expected_event_keys = {(event, seed) for event in events for seed in SEEDS}
    if len(events) != EXPECTED_EVENTS or event_keys != expected_event_keys:
        raise ValueError("event OOF coverage is incomplete")

    station_keys = {
        (str(row["event"]), str(row["station"]), int(row["seed"]))
        for row in station_rows
    }
    if len(station_rows) != EXPECTED_RECORDS * len(SEEDS):
        raise ValueError("station OOF row count changed")
    record_keys = {
        (str(row["event"]), str(row["station"])) for row in station_rows
    }
    expected_station_keys = {
        (event, station, seed)
        for event, station in record_keys
        for seed in SEEDS
    }
    if len(record_keys) != EXPECTED_RECORDS or station_keys != expected_station_keys:
        raise ValueError("station OOF coverage is incomplete")


def run_campaign(
    *,
    config_path: Path,
    output_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    base_config = grouped._load_yaml(config_path)
    phase39 = grouped.build_arm_configs(base_config)["phase39"]
    samples = grouped._load_dataset_samples(phase39)
    if len(samples) != EXPECTED_RECORDS:
        raise ValueError(f"expected {EXPECTED_RECORDS} records, got {len(samples)}")
    fold_manifest = grouped.build_event_folds(samples, n_folds=N_FOLDS)
    if int(fold_manifest["n_events"]) != EXPECTED_EVENTS:
        raise ValueError(
            f"expected {EXPECTED_EVENTS} events, got {fold_manifest['n_events']}"
        )

    folds = (0,) if smoke else tuple(range(N_FOLDS))
    seeds = (SEEDS[0],) if smoke else SEEDS
    epochs = 1 if smoke else int(phase39["training"]["epochs"])
    protocol = {
        "protocol_version": 1,
        "mode": "smoke" if smoke else "expanded_grouped_cv",
        "created_at_utc": utc_now_iso(),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "git_dirty": git_is_dirty(PROJECT_ROOT),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "data_path": str(phase39["paths"]["data_path"]),
        "data_sha256": sha256_file(Path(phase39["paths"]["data_path"])),
        "stf_path": str(phase39["dataset"]["stf"]["path"]),
        "fold_assignment_sha256": fold_manifest["assignment_sha256"],
        "folds": list(folds),
        "seeds": list(seeds),
        "epochs": epochs,
        "arm": "phase39",
        "lambda_synth": 0.5,
        "radiation_coefficient_contract": "glehman_scalar",
        "synth_polarity_mode": "global_invariant",
        "split_protocol": "grouped_event_5fold",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        stable_fields = set(protocol) - {"created_at_utc", "git_dirty"}
        if any(previous.get(key) != protocol.get(key) for key in stable_fields):
            raise ValueError("existing expanded campaign protocol does not match")
    else:
        grouped._atomic_json(protocol_path, protocol)
        grouped._atomic_json(output_root / "event_folds.json", fold_manifest)

    event_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []
    total_runs = len(folds) * len(seeds)
    completed = 0
    for fold in folds:
        for seed in seeds:
            event_rows.extend(
                grouped.run_one(
                    samples=samples,
                    fold_manifest=fold_manifest,
                    arm_config=phase39,
                    output_root=output_root,
                    fold=fold,
                    arm="phase39",
                    seed=seed,
                    epochs=epochs,
                )
            )
            station_path = (
                output_root
                / f"fold_{fold}"
                / "phase39"
                / f"seed_{seed}"
                / "station_oof.csv"
            )
            station_rows.extend(grouped._read_csv(station_path))
            completed += 1
            grouped._atomic_json(
                output_root / "campaign_status.json",
                {
                    "updated_at_utc": utc_now_iso(),
                    "completed_runs": completed,
                    "total_runs": total_runs,
                },
            )

    if smoke:
        summary = {
            "status": "complete",
            "mode": "smoke",
            "completed_runs": completed,
            "total_runs": total_runs,
        }
        grouped._atomic_json(output_root / "campaign_summary.json", summary)
        return summary

    _validate_coverage(event_rows, station_rows)
    event_rows.sort(key=lambda row: (str(row["event"]), int(row["seed"])))
    station_rows.sort(
        key=lambda row: (
            str(row["event"]),
            str(row["station"]),
            int(row["seed"]),
        )
    )
    event_oof_path = grouped._write_csv(
        output_root / "oof_event_predictions_all_seeds.csv",
        event_rows,
    )
    station_oof_path = grouped._write_csv(
        output_root / "oof_station_predictions_all_seeds.csv",
        station_rows,
    )
    event_ensemble = _aggregate_across_seeds(
        event_rows,
        identity_fields=("event",),
        prediction_field="mw_pred_median",
        reference_field="mw_catalog",
    )
    station_ensemble = _aggregate_across_seeds(
        station_rows,
        identity_fields=("event", "station"),
        prediction_field="mw_pred",
        reference_field="mw_catalog",
    )
    event_ensemble_path = grouped._write_csv(
        output_root / "oof_event_predictions_seed_ensemble.csv",
        event_ensemble,
    )
    station_ensemble_path = grouped._write_csv(
        output_root / "oof_station_predictions_seed_ensemble.csv",
        station_ensemble,
    )
    event_reference = np.asarray(
        [float(row["mw_catalog"]) for row in event_ensemble],
        dtype=np.float64,
    )
    event_prediction = np.asarray(
        [float(row["mw_pred_median"]) for row in event_ensemble],
        dtype=np.float64,
    )
    station_reference = np.asarray(
        [float(row["mw_catalog"]) for row in station_ensemble],
        dtype=np.float64,
    )
    station_prediction = np.asarray(
        [float(row["mw_pred"]) for row in station_ensemble],
        dtype=np.float64,
    )
    seed_metrics = {}
    for seed in SEEDS:
        matched = [row for row in event_rows if int(row["seed"]) == seed]
        seed_metrics[str(seed)] = _metrics(
            np.asarray([float(row["mw_catalog"]) for row in matched]),
            np.asarray([float(row["mw_pred_median"]) for row in matched]),
        )
    summary = {
        "status": "complete",
        "mode": "expanded_grouped_cv",
        "completed_at_utc": utc_now_iso(),
        "completed_runs": completed,
        "total_runs": total_runs,
        "event_count": EXPECTED_EVENTS,
        "station_count": EXPECTED_RECORDS,
        "event_metrics_seed_ensemble": _metrics(
            event_reference,
            event_prediction,
        ),
        "station_metrics_seed_ensemble": _metrics(
            station_reference,
            station_prediction,
        ),
        "event_metrics_by_seed": seed_metrics,
        "artifacts": {
            "event_oof_all_seeds": str(event_oof_path),
            "station_oof_all_seeds": str(station_oof_path),
            "event_oof_seed_ensemble": str(event_ensemble_path),
            "station_oof_seed_ensemble": str(station_ensemble_path),
        },
    }
    grouped._atomic_json(output_root / "campaign_summary.json", summary)
    grouped._atomic_write(output_root / "COMPLETE", b"\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Phase39 on the immutable 39-event expanded dataset."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
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
