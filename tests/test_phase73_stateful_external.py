from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation.evaluate_phase73_stateful_external import (
    build_causal_hint_cube,
    build_external_tables,
    summarize_trajectory_metrics,
)
from scripts.experiments.run_phase43_streaming_adapter import HORIZONS
from scripts.plotting.plot_phase73_stateful_external_zh import render_readme
from src.baseline.causal_pgd import RawPGDRecord


def test_causal_hint_cube_uses_raw_enu_and_p_arrival() -> None:
    time = np.arange(0.0, 80.0, 1.0)
    record = RawPGDRecord(
        event="event_a",
        station="STA",
        time_sec=time,
        east_m=0.001 * time,
        north_m=0.0005 * time,
        up_m=0.00025 * time,
        source_distance_km=80.0,
        p_arrival_sec=12.0,
        magnitude_catalog=7.0,
    )

    hints = build_causal_hint_cube([record], horizons=(20, 30, 40))

    assert hints["pgd_3d_m"].shape == (1, 3)
    assert hints["crowell_mw"].shape == (1, 3)
    assert hints["p_arrived"].tolist() == [[True, True, True]]
    assert np.all(np.diff(hints["pgd_3d_m"][0]) > 0.0)
    assert np.all(np.isfinite(hints["crowell_mw"]))


def test_external_tables_keep_all_phase73_methods_and_event_medians() -> None:
    station_count = 8
    horizon_count = len(HORIZONS)
    records = [
        SimpleNamespace(
            event=f"event_{index}",
            station=f"STA{index:02d}",
            magnitude_catalog=6.5 + index * 0.1,
            source_distance_m=70_000.0 + index * 1_000.0,
        )
        for index in range(station_count)
    ]
    horizons = np.asarray(HORIZONS, dtype=np.float32)
    base = np.asarray(
        [record.magnitude_catalog for record in records], dtype=np.float32
    ).reshape(-1, 1)
    phase73 = base - 0.35 * np.exp(-(horizons.reshape(1, -1) - 20.0) / 45.0)
    phase39 = phase73 + 0.04 * np.sin(horizons.reshape(1, -1) / 7.0)
    pgd = 0.01 + 0.0001 * horizons.reshape(1, -1) + np.arange(
        station_count, dtype=np.float32
    ).reshape(-1, 1) * 0.0002
    diagnostics = np.zeros((station_count, horizon_count), dtype=np.float32)

    tables = build_external_tables(
        records=records,
        phase73_mw=phase73,
        phase39_mw=phase39,
        pgd_3d_m=pgd,
        crowell_mw=phase73 - 0.1,
        p_arrived=np.ones((station_count, horizon_count), dtype=bool),
        plateau_confidence=diagnostics,
        revision_mw=diagnostics,
        proposal_assimilation_mw=diagnostics,
    )

    assert len(tables["event_rows"]) == station_count * horizon_count
    assert len(tables["station_rows"]) == station_count * horizon_count
    assert len(tables["endpoint_station_rows"]) == station_count
    assert len(tables["horizon_rows"]) == 5 * horizon_count
    assert len(tables["trajectory_diagnostics"]) == station_count
    assert "phase73_mw_pred" in tables["station_rows"][0]
    assert "ruhl_mw_pred_median" in tables["event_rows"][0]
    stability = summarize_trajectory_metrics(tables["event_cubes"])
    assert stability["phase73"]["post160_band_width_p95_mw"] < 0.2


def test_report_readme_keeps_external_development_boundary_explicit() -> None:
    metrics = {
        method: {"event_mae_mw": 0.2, "station_mae_mw": 0.3}
        for method in ("phase73", "phase39", "crowell", "melgar", "ruhl")
    }
    summary = {
        "endpoint_metrics": metrics,
        "trajectory_metrics": {
            method: {
                "post120_abs_step_p95_mw": 0.01,
                "peak_to_final_p95_mw": 0.02,
                "post160_band_width_p95_mw": 0.03,
            }
            for method in ("phase73", "phase39")
        },
        "improved_event_count_vs_phase39": 3,
        "endpoint_phase39_reproduction_gate": {
            "max_station_prediction_abs_diff_mw": 1.0e-6
        },
        "endpoint_events": [
            {
                "event": "event_a",
                "mw_catalog": 7.0,
                "phase39_mw_pred_median": 6.9,
                "phase73_mw_pred_median": 7.1,
                "crowell_mw_pred_median": 6.8,
                "phase39_abs_error_mw": 0.1,
                "phase73_abs_error_mw": 0.1,
            }
        ],
    }

    report = render_readme(summary)

    assert "development_validation" in report
    assert "不能" in report
    assert "internal test 和 grouped held-out test 均未打开" in report


@pytest.mark.parametrize(
    "script",
    (
        "scripts/evaluation/evaluate_phase73_stateful_external.py",
        "scripts/plotting/plot_phase73_stateful_external_zh.py",
    ),
)
def test_phase73_external_cli_help_runs(script: str) -> None:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout
