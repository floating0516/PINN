import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import yaml

from src.data.dataset_v2 import CorrectedEarthquakeDataset
from src.data.manifest import (
    MANIFEST_FIELDS,
    REJECTION_REASONS,
    audit_passes,
    build_dataset_summary,
    validate_manifest_rows,
    write_dataset_audit,
)


def _config(npz_path: Path, stf_dir: Path) -> dict[str, Any]:
    return {
        "pipeline_version": 2,
        "dataset": {
            "blacklist_events": [],
            "units": "m",
            "sample_rate_hz": 1.0,
            "radial_peak_min_cm": 0.01,
            "allow_missing_stf": False,
            "waveform": {
                "start_sec": 0.0,
                "duration_sec": 200.0,
                "min_valid_fraction": 1.0,
                "max_interpolation_gap_sec": 2.5,
            },
            "baseline": {
                "method": "median",
                "pre_event_start_sec": -20.0,
                "pre_event_end_sec": 0.0,
                "fallback": "pre_p",
                "fallback_max_sec": 30.0,
                "min_samples": 10,
            },
            "filter": {
                "type": "lowpass",
                "cutoff_hz": 0.1,
                "num_taps": 7,
                "window": "hamming",
            },
            "stf": {
                "path": str(stf_dir),
                "start_sec": 0.0,
                "duration_sec": 200.0,
                "min_retained_moment_fraction": 0.995,
                "preserve_integral": True,
                "m_ref": 1.0e18,
            },
        },
        "physics": {
            "rho": 3400.0,
            "alpha": 7900.0,
            "beta": 4533.0,
            "distance_mode": "hypocentral",
            "delay_mode": "absolute",
            "amplitude_gain": 1.0,
        },
        "paths": {"data_path": str(npz_path)},
        "training": {"rate_representation": "log1p"},
        "evaluation": {
            "primary_reference": "catalog",
            "aggregation": "event_median",
        },
    }


def _write_fixture(tmp_path: Path) -> dict[str, Any]:
    time_sec = np.arange(-20.0, 210.0)
    stations = np.empty(1, dtype=object)
    stations[0] = [
        {
            "name": "GOOD",
            "lat": 35.1,
            "lon": 140.0,
            "t": time_sec,
            "E": np.zeros_like(time_sec),
            "N": 0.1 * np.sin(time_sec / 20.0),
            "U": 0.02 * np.cos(time_sec / 25.0),
        },
        {
            "name": "LOW",
            "lat": 35.2,
            "lon": 140.0,
            "t": time_sec,
            "E": np.zeros_like(time_sec),
            "N": 1.0e-6 * np.sin(time_sec / 20.0),
            "U": np.zeros_like(time_sec),
        },
    ]
    npz_path = tmp_path / "events.npz"
    np.savez(
        npz_path,
        events=np.array(["Audit Event"], dtype=object),
        magnitude=np.array([7.1]),
        latitude=np.array([35.0]),
        longitude=np.array([140.0]),
        depth_km=np.array([20.0]),
        strike=np.array([30.0]),
        dip=np.array([45.0]),
        rake=np.array([90.0]),
        stations=stations,
    )

    stf_dir = tmp_path / "stf"
    stf_dir.mkdir()
    source_time = np.arange(100.0)
    source_rate = (
        np.maximum(0.0, 1.0 - np.abs(source_time - 40.0) / 30.0)
        * 1.0e18
    )
    with (stf_dir / "audit_event.stf").open(
        "w", encoding="utf-8"
    ) as stream:
        for time_value, rate_value in zip(source_time, source_rate):
            stream.write(f"{time_value} {rate_value}\n")
    return _config(npz_path, stf_dir)


def test_manifest_contains_every_accepted_and_rejected_candidate(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path)
    dataset = CorrectedEarthquakeDataset(config)
    manifest_path = tmp_path / "audit" / "dataset_manifest.csv"
    summary_path = tmp_path / "audit" / "dataset_summary.json"

    summary = write_dataset_audit(
        dataset,
        manifest_path=manifest_path,
        summary_path=summary_path,
        minimum_stf_retained_fraction=0.995,
    )

    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == MANIFEST_FIELDS
    assert len(rows) == 2
    rows_by_station = {row["station"]: row for row in rows}
    assert rows_by_station["GOOD"]["accepted"] == "True"
    assert rows_by_station["GOOD"]["rejection_reason"] == ""
    assert rows_by_station["LOW"]["accepted"] == "False"
    assert (
        rows_by_station["LOW"]["rejection_reason"]
        == "below_radial_peak_threshold"
    )
    assert rows_by_station["LOW"]["source_distance_km"] != ""
    assert rows_by_station["LOW"]["radial_peak_cm"] != ""

    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted == summary
    assert summary["candidate_event_count"] == 1
    assert summary["accepted_event_count"] == 1
    assert summary["candidate_station_count"] == 2
    assert summary["accepted_station_count"] == 1
    assert summary["rejection_counts"] == {
        "below_radial_peak_threshold": 1
    }
    assert summary["events"]["Audit Event"]["candidate_station_count"] == 2
    assert summary["events"]["Audit Event"]["accepted_station_count"] == 1
    assert summary["invariants"]["all_waveform_dt_equal_1s"] is True
    assert summary["invariants"]["one_stf_per_event"] is True
    assert summary["invariants"]["one_stf_mw_per_event"] is True
    assert summary["invariants"]["min_stf_retained_fraction"] >= 0.995
    assert audit_passes(summary, minimum_stf_retained_fraction=0.995)


def test_manifest_rejection_reasons_are_a_closed_enum(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    dataset = CorrectedEarthquakeDataset(config)
    assert set(REJECTION_REASONS) == {
        "blacklisted_event",
        "missing_station_coordinates",
        "invalid_waveform",
        "insufficient_baseline",
        "insufficient_valid_fraction",
        "below_radial_peak_threshold",
        "missing_stf",
        "stf_window_too_short",
        "invalid_geometry",
    }
    invalid_rows = [dict(dataset.manifest_rows[0])]
    invalid_rows[0]["accepted"] = False
    invalid_rows[0]["rejection_reason"] = "invented_reason"

    with pytest.raises(ValueError, match="invented_reason"):
        validate_manifest_rows(invalid_rows)


def test_empty_accepted_dataset_fails_invariant_gate(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    config["dataset"]["radial_peak_min_cm"] = 1000.0
    dataset = CorrectedEarthquakeDataset(config)

    summary = build_dataset_summary(dataset)

    assert summary["accepted_station_count"] == 0
    assert not audit_passes(
        summary,
        minimum_stf_retained_fraction=0.995,
    )


def test_writer_rejects_invalid_minimum_retained_fraction(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path)
    dataset = CorrectedEarthquakeDataset(config)

    with pytest.raises(ValueError, match="minimum_stf_retained_fraction"):
        write_dataset_audit(
            dataset,
            manifest_path=tmp_path / "manifest.csv",
            summary_path=tmp_path / "summary.json",
            minimum_stf_retained_fraction=0.0,
        )


def test_audit_cli_writes_outputs_and_returns_success(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "cli" / "manifest.csv"
    summary_path = tmp_path / "cli" / "summary.json"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/data/audit_corrected_dataset.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--summary",
            str(summary_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert manifest_path.is_file()
    assert summary_path.is_file()
