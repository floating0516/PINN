from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from scripts.plotting.plot_phase49_posthoc_direct_streaming_zh import (
    trajectory_diagnostics,
)


def test_trajectory_diagnostics_separate_step_and_accumulated_drop() -> None:
    rows = []
    predictions = {
        "phase39": (7.0, 7.5, 7.3, 7.1),
        "phase47": (7.0, 7.2, 7.15, 7.1),
        "phase48": (7.0, 7.15, 7.14, 7.1),
    }
    for model, values in predictions.items():
        for horizon, prediction in zip((20, 60, 61, 200), values, strict=True):
            rows.append(
                {
                    "model": model,
                    "event": "event_a",
                    "observation_horizon_sec": horizon,
                    "mw_catalog": 7.0,
                    "mw_pred_median": prediction,
                    "abs_error": abs(prediction - 7.0),
                }
            )

    diagnostics = trajectory_diagnostics(rows)
    phase39 = next(row for row in diagnostics if row["model"] == "phase39")
    phase48 = next(row for row in diagnostics if row["model"] == "phase48")

    assert phase39["max_down_step_after_60_mw"] == pytest.approx(0.2)
    assert phase39["peak_to_final_drop_after_60_mw"] == pytest.approx(0.4)
    assert phase39["peak_after_60_observation_sec"] == 60
    assert phase48["max_down_step_after_60_mw"] == pytest.approx(0.04)
    assert phase48["peak_to_final_drop_after_60_mw"] == pytest.approx(0.05)


def test_phase49_plot_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plotting/plot_phase49_posthoc_direct_streaming_zh.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--run-root" in result.stdout
    assert "--output-dir" in result.stdout
