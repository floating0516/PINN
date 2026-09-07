from __future__ import annotations

import argparse
import copy
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

from scripts.experiments import run_phase39_causal_direct as direct  # noqa: E402
from scripts.experiments import run_phase39_expanded_fixed_split as fixed  # noqa: E402
from src.baseline.scaling_laws import predict_mw  # noqa: E402
from src.data.dataset_v2 import CorrectedEarthquakeDataset  # noqa: E402
from src.data.loaders_v2 import get_data_loaders_v2  # noqa: E402
from src.models.model import PINNModel  # noqa: E402
from src.training.train import _build_stf_rate_criterion  # noqa: E402
from src.utils.config_v2 import validate_config_on_startup  # noqa: E402
from src.utils.provenance import sha256_file  # noqa: E402


DEFAULT_CAUSAL_RUN_ROOT = Path(
    "/home/lihe/PINN_Mag/runs/"
    "phase39-causal-expanded-fixed-seed73-20260831-v1"
)
PGD_METHODS = ("crowell", "ruhl", "melgar")
HORIZONS = tuple(range(1, 201))
ANCHOR_HORIZONS = {30, 60, 90, 120, 160, 200}


def _sample_key(sample: Mapping[str, Any]) -> tuple[str, str]:
    return str(sample["event"]), str(sample["station"])


def _prediction_summary(errors: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "bias": float(np.mean(errors)),
    }


def evaluate_causal_pgd(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    station_curves: dict[str, list[dict[str, Any]]] = {
        method: [] for method in PGD_METHODS
    }
    for sample in samples:
        event, station = _sample_key(sample)
        components = np.stack(
            [
                np.asarray(sample["radial"], dtype=np.float64),
                np.asarray(sample["tangential"], dtype=np.float64),
                np.asarray(sample["vertical"], dtype=np.float64),
            ],
            axis=0,
        )
        if components.shape[1] != len(HORIZONS):
            raise ValueError(f"PGD waveform length changed for {event}/{station}")
        pgd_curve = np.maximum.accumulate(
            np.sqrt(np.sum(np.square(components), axis=0))
        )
        if not bool(np.all(np.isfinite(pgd_curve))) or float(pgd_curve[-1]) <= 0.0:
            raise ValueError(f"invalid causal PGD curve for {event}/{station}")
        distance_km = float(sample["source_distance_m"]) / 1000.0
        catalog = float(sample["magnitude_catalog"])
        for method in PGD_METHODS:
            predictions = np.full(pgd_curve.shape, np.nan, dtype=np.float64)
            valid = pgd_curve > 0.0
            predictions[valid] = np.asarray(
                [
                    predict_mw(
                        law_name=method,
                        pgd_m=float(pgd),
                        source_distance_km=distance_km,
                    )
                    for pgd in pgd_curve[valid]
                ],
                dtype=np.float64,
            )
            station_curves[method].append(
                {
                    "event": event,
                    "station": station,
                    "mw_catalog": catalog,
                    "predictions": predictions,
                }
            )

    horizon_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    endpoint_station_rows: list[dict[str, Any]] = []
    for method, records in station_curves.items():
        if len(records) != fixed.EXPECTED_RECORD_COUNTS["test"]:
            raise ValueError(f"PGD test station coverage changed: {method}")
        events = sorted({str(record["event"]) for record in records})
        for horizon_index, horizon in enumerate(HORIZONS):
            station_errors = np.asarray(
                [
                    float(record["predictions"][horizon_index])
                    - float(record["mw_catalog"])
                    for record in records
                    if math.isfinite(float(record["predictions"][horizon_index]))
                ],
                dtype=np.float64,
            )
            current_event_rows: list[dict[str, Any]] = []
            for event in events:
                selected = [
                    record for record in records if str(record["event"]) == event
                ]
                predictions = np.asarray(
                    [
                        record["predictions"][horizon_index]
                        for record in selected
                        if math.isfinite(
                            float(record["predictions"][horizon_index])
                        )
                    ],
                    dtype=np.float64,
                )
                if predictions.size == 0:
                    continue
                catalog = float(selected[0]["mw_catalog"])
                median = float(np.median(predictions))
                current_event_rows.append(
                    {
                        "method": method,
                        "observation_horizon_sec": int(horizon),
                        "event": event,
                        "mw_pred_median": median,
                        "mw_catalog": catalog,
                        "error_vs_catalog": median - catalog,
                        "n_stations": len(selected),
                        "pred_std": float(np.std(predictions)),
                        "pred_iqr": float(
                            np.percentile(predictions, 75)
                            - np.percentile(predictions, 25)
                        ),
                    }
                )
            event_rows.extend(current_event_rows)
            event_errors = np.asarray(
                [row["error_vs_catalog"] for row in current_event_rows],
                dtype=np.float64,
            )
            event_summary = _prediction_summary(event_errors)
            station_summary = _prediction_summary(station_errors)
            horizon_rows.append(
                {
                    "method": method,
                    "observation_horizon_sec": int(horizon),
                    "reference": "catalog",
                    "event_count": len(current_event_rows),
                    "station_count": int(station_errors.size),
                    "event_mae": event_summary["mae"],
                    "event_rmse": event_summary["rmse"],
                    "event_bias": event_summary["bias"],
                    "station_mae": station_summary["mae"],
                    "station_rmse": station_summary["rmse"],
                    "station_bias": station_summary["bias"],
                }
            )
        for record in records:
            prediction = float(record["predictions"][-1])
            catalog = float(record["mw_catalog"])
            endpoint_station_rows.append(
                {
                    "method": method,
                    "observation_horizon_sec": 200,
                    "event": str(record["event"]),
                    "station": str(record["station"]),
                    "mw_pred": prediction,
                    "mw_catalog": catalog,
                    "error_vs_catalog": prediction - catalog,
                }
            )
    return {
        "horizon_rows": horizon_rows,
        "event_rows": event_rows,
        "endpoint_station_rows": endpoint_station_rows,
    }


def run_evaluation(
    *,
    causal_run_root: Path,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    causal_summary = direct._read_json(causal_run_root / "summary.json")
    causal_protocol = direct._read_json(causal_run_root / "protocol.json")
    if causal_summary.get("status") != "complete":
        raise ValueError("causal validation run is incomplete")
    if causal_summary.get("held_out_test_loader_iterated") is not False:
        raise ValueError("causal validation run already iterated the test loader")
    if causal_protocol.get("development_role") != "fixed_validation_only":
        raise ValueError("causal checkpoint was not selected on fixed validation")
    checkpoint_path = Path(str(causal_summary["checkpoint_path"])).resolve()
    if sha256_file(checkpoint_path) != str(causal_summary["checkpoint_sha256"]):
        raise ValueError("causal checkpoint hash changed")

    experiment = causal_protocol["experiment_config"]
    endpoint_summary_path = Path(
        str(experiment["fixed_endpoint_candidate_summary"])
    ).resolve()
    endpoint_summary = direct._read_json(endpoint_summary_path)
    config_path = Path(str(endpoint_summary["config_snapshot_path"])).resolve()
    if sha256_file(config_path) != str(endpoint_summary["config_snapshot_sha256"]):
        raise ValueError("source config hash changed")
    config = direct._read_yaml(config_path)
    validate_config_on_startup(config)
    samples = [
        dict(sample)
        for sample in CorrectedEarthquakeDataset(copy.deepcopy(config)).samples
    ]
    split, split_manifest = fixed.build_fixed_split(samples)
    expected_assignment = str(experiment["expected_split_assignment_sha256"])
    if split_manifest["assignment_sha256"] != expected_assignment:
        raise ValueError("fixed split assignment changed")
    train_loader, validation_loader, test_loader, loader_manifest = (
        get_data_loaders_v2(config, explicit_split=split)
    )
    del train_loader, validation_loader
    if loader_manifest["assignment_sha256"] != expected_assignment:
        raise ValueError("test loader split assignment changed")

    criterion = _build_stf_rate_criterion(config, device)
    model = PINNModel(config).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True),
        strict=True,
    )
    direct_result = direct.evaluate_horizons(
        model,
        criterion,
        config,
        test_loader,
        method="direct",
        horizons=HORIZONS,
        station_horizons=ANCHOR_HORIZONS,
    )
    test_samples = [samples[index] for index in split.test_indices]
    if {_sample_key(sample)[0] for sample in test_samples} != set(
        fixed.TEST_EVENTS
    ):
        raise ValueError("test sample event identities changed")
    pgd_result = evaluate_causal_pgd(test_samples)

    horizon_rows = direct_result["horizon_rows"] + pgd_result["horizon_rows"]
    event_rows = direct_result["event_rows"] + pgd_result["event_rows"]
    station_rows = (
        direct_result["station_rows"] + pgd_result["endpoint_station_rows"]
    )
    horizon_path = output_root / "test_horizon_metrics.csv"
    event_path = output_root / "test_event_predictions.csv"
    station_path = output_root / "test_anchor_station_predictions.csv"
    direct._write_csv(horizon_path, horizon_rows)
    direct._write_csv(event_path, event_rows)
    direct._write_csv(station_path, station_rows)

    by_method: dict[str, Any] = {}
    for method in ("direct", *PGD_METHODS):
        method_horizons = [row for row in horizon_rows if row["method"] == method]
        method_events = [row for row in event_rows if row["method"] == method]
        by_method[method] = direct.summarize_trajectory(
            method_horizons,
            method_events,
            endpoint_band_tolerance_mw=0.05,
        )
    direct_event_rows = [
        row
        for row in direct_result["event_rows"]
        if int(row["observation_horizon_sec"]) == 200
    ]
    if len(direct_event_rows) != len(fixed.TEST_EVENTS):
        raise ValueError("direct endpoint event coverage changed")
    summary = {
        "status": "complete",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_frozen_from_validation": True,
        "test_training_or_selection_performed": False,
        "test_event_count": len(fixed.TEST_EVENTS),
        "test_station_count": fixed.EXPECTED_RECORD_COUNTS["test"],
        "split_assignment_sha256": expected_assignment,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "lambda_synth": float(causal_summary["lambda_synth"]),
        "moment_scale_augmentation_weight": float(
            causal_summary["moment_scale_augmentation_weight"]
        ),
        "trajectory": by_method,
        "direct_endpoint_events": direct_event_rows,
        "artifacts": {
            "horizon_metrics_path": str(horizon_path),
            "horizon_metrics_sha256": sha256_file(horizon_path),
            "event_predictions_path": str(event_path),
            "event_predictions_sha256": sha256_file(event_path),
            "station_predictions_path": str(station_path),
            "station_predictions_sha256": sha256_file(station_path),
        },
        "interpretation_guard": (
            "The method was frozen on fixed validation before this replay. "
            "The nine-event cohort had previously been inspected by separate "
            "endpoint experiments, so it is an internal fixed test rather than "
            "a pristine never-observed external cohort."
        ),
    }
    direct._write_json(output_root / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the validation-frozen causal Phase 39 checkpoint on the "
            "fixed nine-event test cohort at every second."
        )
    )
    parser.add_argument(
        "--causal-run-root",
        type=Path,
        default=DEFAULT_CAUSAL_RUN_ROOT,
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root or (
        Path("/home/lihe/PINN_Mag/runs")
        / f"phase39-causal-expanded-fixed-test-{timestamp}"
    )
    summary = run_evaluation(
        causal_run_root=args.causal_run_root.resolve(),
        output_root=output_root.resolve(),
        device=device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
