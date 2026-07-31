from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.experiments.run_phase43_streaming_adapter import HORIZONS
from scripts.plotting.plot_phase73_stateful_validation_zh import (
    METHOD_ORDER,
    SELECTED_EPOCH,
    SELECTED_SEED,
    build_event_rows,
    prediction_metrics,
    trajectory_diagnostics,
)


def test_prediction_metrics_reports_accuracy_and_coverage() -> None:
    metrics = prediction_metrics(
        np.asarray([6.0, 7.0, 8.0]),
        np.asarray([6.1, 6.8, 8.4]),
    )

    assert metrics["count"] == 3
    assert metrics["mae_mw"] == pytest.approx((0.1 + 0.2 + 0.4) / 3.0)
    assert metrics["bias_mw"] == pytest.approx(0.1)
    assert metrics["within_0_15_count"] == 1
    assert metrics["within_0_30_count"] == 2


def test_event_rows_and_trajectory_diagnostics_preserve_stateful_curve() -> None:
    horizons = np.asarray(HORIZONS, dtype=np.float64)
    phase73 = 6.0 + 0.8 * (1.0 - np.exp(-(horizons - 20.0) / 45.0))
    phase39 = phase73 + 0.08 * np.sin(horizons / 8.0)
    crowell = 5.9 + 0.85 * (1.0 - np.exp(-(horizons - 20.0) / 55.0))
    event_cubes = {
        "phase73": phase73.reshape(1, -1),
        "phase39": phase39.reshape(1, -1),
        "crowell": crowell.reshape(1, -1),
        "melgar": (crowell - 0.2).reshape(1, -1),
        "ruhl": (crowell - 0.4).reshape(1, -1),
    }

    rows = build_event_rows(
        split="validation",
        event_names=["event_a"],
        event_catalogs=np.asarray([6.8]),
        station_counts=np.asarray([4]),
        event_cubes=event_cubes,
    )
    diagnostics = trajectory_diagnostics(rows)

    assert len(rows) == len(HORIZONS)
    assert len(diagnostics) == 1
    assert diagnostics[0]["event"] == "event_a"
    assert diagnostics[0]["station_count"] == 4
    assert diagnostics[0]["phase73_start_to_end_change_mw"] > 0.7
    assert diagnostics[0]["phase73_post160_band_width_mw"] < 0.05
    assert diagnostics[0]["phase73_post120_sign_changes"] == 0
    assert all(f"{method}_mw_pred_median" in rows[0] for method in METHOD_ORDER)


def test_phase73_source_summary_keeps_hidden_data_closed() -> None:
    import json

    run_root = Path(
        "/home/lihe/PINN_Mag/runs/"
        "phase73-teacher-weight2-stateful-20260730T154118Z-804bf96"
    )
    if not run_root.is_dir():
        pytest.skip("local frozen Phase73 artifact is unavailable")
    campaign = json.loads((run_root / "campaign_summary.json").read_text())
    seed = next(row for row in campaign["seed_summaries"] if row["seed"] == SELECTED_SEED)

    assert seed["closest_epoch"] == SELECTED_EPOCH
    assert seed["provenance"]["internal_test_iterated"] is False
    assert seed["provenance"]["external_data_loaded"] is False
    assert seed["provenance"]["grouped_test_loaded"] is False
    assert seed["closest_gate"]["pgd_accuracy_passed"] is True
    assert seed["closest_gate"]["trajectory_passed"] is True


def test_phase73_report_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plotting/plot_phase73_stateful_validation_zh.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--run-root" in result.stdout
    assert "--cache-root" in result.stdout
    assert "--hint-cache-root" in result.stdout
    assert "--output-dir" in result.stdout
