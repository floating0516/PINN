from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.evaluation.evaluate_phase39_second_by_second import (
    _write_csv,
    analyze_convergence,
    load_label_contract,
    records_to_batches,
    validate_endpoint_reproduction,
    validate_fixed_cohort,
)


def test_label_contract_uses_selected_magnitude_and_exact_event_dirs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "labels.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "event",
                "event_dir",
                "mw_selected",
                "mw_source",
                "usgs_event_id",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "event": "Event A M7.4",
                "event_dir": "event-a",
                "mw_selected": "7.4",
                "mw_source": "usgs_preferred",
                "usgs_event_id": "us-a",
            }
        )

    labels = load_label_contract(path, expected_event_dirs=("event-a",))

    assert labels == {
        "event-a": {
            "event": "Event A M7.4",
            "event_dir": "event-a",
            "mw_selected": 7.4,
            "mw_source": "usgs_preferred",
            "usgs_event_id": "us-a",
        }
    }


def test_csv_outputs_use_lf_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "result.csv"

    _write_csv(path, [{"event": "A", "mw": 7.4}], fieldnames=("event", "mw"))

    assert path.read_bytes() == b"event,mw\nA,7.4\n"


def _record(event: str, station: str, value: float) -> dict:
    radial = np.full(200, value, dtype=np.float32)
    return {
        "event": event,
        "station": station,
        "radial": radial,
        "waveform_valid_mask": np.ones(200, dtype=bool),
        "waveform_dt_sec": 1.0,
        "raw_dt_sec": 1.0,
        "stf_dt_sec": 1.0,
        "source_distance_m": 100_000.0,
        "epicentral_distance_m": 90_000.0,
        "theta_deg": 30.0,
        "azimuth_deg": 120.0,
        "magnitude_catalog": 7.4,
        "baseline_source": "pre_event",
    }


def test_fixed_cohort_and_batch_construction_preserve_station_identity() -> None:
    records = [_record("A", "S1", 1.0), _record("B", "S2", 2.0)]
    expected = {("A", "S1"), ("B", "S2")}

    validate_fixed_cohort(records, expected_station_keys=expected)
    batches = list(records_to_batches(records, batch_size=1))

    assert len(batches) == 2
    assert batches[0]["event"] == ["A"]
    assert batches[0]["station"] == ["S1"]
    assert tuple(batches[0]["radial"].shape) == (1, 1, 200)
    assert batches[1]["radial"][0, 0, 0].item() == 2.0
    with pytest.raises(ValueError, match="differs from locked endpoint"):
        validate_fixed_cohort(
            records,
            expected_station_keys={("A", "S1"), ("B", "OTHER")},
        )


def test_suffix_stable_convergence_requires_every_later_second() -> None:
    horizons = (1, 2, 3, 4)
    errors = {
        "A": (0.20, 0.10, 0.18, 0.10),
        "B": (0.10, 0.10, 0.10, 0.10),
    }
    event_rows = []
    horizon_metrics = []
    for index, horizon in enumerate(horizons):
        current = []
        for event in ("A", "B"):
            error = errors[event][index]
            current.append(error)
            event_rows.append(
                {
                    "event": event,
                    "observation_horizon_sec": float(horizon),
                    "release_time_sec": float(horizon + 5),
                    "mw_pred_median": 7.0 + error,
                    "mw_catalog": 7.0,
                    "error": error,
                    "station_count": 1,
                }
            )
        horizon_metrics.append(
            {
                "observation_horizon_sec": float(horizon),
                "release_time_sec": float(horizon + 5),
                "event_count": 2,
                "total_event_count": 2,
                "coverage": 2,
                "coverage_fraction": 1.0,
                "available_station_count": 2,
                "unavailable_station_count": 0,
                "unavailable_reason_counts": {},
                "event_equal_mae": float(np.mean(np.abs(current))),
                "event_equal_rmse": float(np.sqrt(np.mean(np.square(current)))),
                "event_equal_bias": float(np.mean(current)),
            }
        )

    analysis = analyze_convergence(
        {
            "event_rows": event_rows,
            "horizon_metrics": horizon_metrics,
        },
        horizons=horizons,
    )
    rows = {row["event"]: row for row in analysis["event_convergence"]}

    assert rows["A"]["first_within_target_observation_sec"] == 2
    assert rows["A"]["stable_within_target_observation_sec"] == 4
    assert rows["A"]["stable_within_target_release_sec"] == 9.0
    assert rows["B"]["stable_within_target_observation_sec"] == 1


def test_endpoint_gate_checks_every_station_event_and_event_mae() -> None:
    result = {
        "station_rows": [
            {
                "event": "A",
                "station": "S1",
                "observation_horizon_sec": 200.0,
                "mw_pred": 7.1,
            },
            {
                "event": "B",
                "station": "S2",
                "observation_horizon_sec": 200.0,
                "mw_pred": 7.9,
            },
        ],
        "event_rows": [
            {
                "event": "A",
                "observation_horizon_sec": 200.0,
                "mw_pred_median": 7.1,
                "mw_catalog": 7.0,
                "error": 0.1,
            },
            {
                "event": "B",
                "observation_horizon_sec": 200.0,
                "mw_pred_median": 7.9,
                "mw_catalog": 8.0,
                "error": -0.1,
            },
        ],
    }
    reference = {
        "station_predictions": {("A", "S1"): 7.1, ("B", "S2"): 7.9},
        "event_predictions": {"A": 7.1, "B": 7.9},
        "event_mae": 0.1,
    }

    gate = validate_endpoint_reproduction(result, reference, tolerance_mw=1e-8)

    assert gate["max_station_prediction_abs_diff_mw"] == 0.0
    assert gate["max_event_median_abs_diff_mw"] == 0.0
    changed = {
        **reference,
        "station_predictions": {("A", "S1"): 7.2, ("B", "S2"): 7.9},
    }
    with pytest.raises(ValueError, match="station prediction mismatch"):
        validate_endpoint_reproduction(result, changed, tolerance_mw=1e-8)


def test_script_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/evaluate_phase39_second_by_second.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--smoke" in result.stdout
    assert "--device" in result.stdout
