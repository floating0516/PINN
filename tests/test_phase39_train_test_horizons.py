from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.plotting.plot_phase39_train_test_horizons_zh import (
    EXPECTED_TEST_COUNT,
    EXPECTED_TEST_EVENT_COUNT,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_TRAIN_EVENT_COUNT,
    SELECTED_HORIZONS,
    build_event_rows,
    load_frozen_predictions,
    plot_all_horizons,
    prediction_metrics,
    summarize_predictions,
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


def test_frozen_phase39_train_test_horizons_reproduce_endpoint() -> None:
    station_rows, event_rows, summary = load_frozen_predictions()

    assert len(station_rows) == (
        EXPECTED_TRAIN_COUNT + EXPECTED_TEST_COUNT
    ) * len(SELECTED_HORIZONS)
    assert len(event_rows) == (
        EXPECTED_TRAIN_EVENT_COUNT + EXPECTED_TEST_EVENT_COUNT
    ) * len(SELECTED_HORIZONS)
    assert summary["test_endpoint_reproduction"]["passed"] is True
    assert summary["test_endpoint_reproduction"]["max_abs_diff_mw"] == 0.0
    metrics = {
        (row["split"], row["observation_horizon_sec"]): row
        for row in summary["metrics"]
    }
    assert metrics[("train", 30)]["event_mae_mw"] == pytest.approx(
        0.8655454827646486
    )
    assert metrics[("test", 30)]["event_mae_mw"] == pytest.approx(
        0.8468829073846632
    )
    assert metrics[("train", 200)]["event_mae_mw"] == pytest.approx(
        0.03394015618891919
    )
    assert metrics[("test", 200)]["event_mae_mw"] == pytest.approx(
        0.15228705818768482
    )


def _synthetic_station_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_index, split in enumerate(("train", "test")):
        for horizon in SELECTED_HORIZONS:
            for event_index, catalog in enumerate((6.5, 7.5)):
                for station_index in range(2):
                    prediction = (
                        catalog
                        + 0.3 * (200 - horizon) / 170.0
                        + 0.02 * split_index
                        + 0.01 * station_index
                    )
                    rows.append(
                        {
                            "split": split,
                            "observation_horizon_sec": horizon,
                            "release_time_sec": horizon + 6.0,
                            "event": f"E{event_index}",
                            "station": f"S{station_index}",
                            "mw_catalog": catalog,
                            "mw_pred": prediction,
                            "error_mw": prediction - catalog,
                            "abs_error_mw": abs(prediction - catalog),
                        }
                    )
    return rows


def test_plot_all_horizons_writes_five_pngs_and_pdfs(tmp_path: Path) -> None:
    station_rows = _synthetic_station_rows()
    event_rows = build_event_rows(station_rows)
    summary = {
        "metrics": summarize_predictions(station_rows, event_rows),
    }

    outputs = plot_all_horizons(
        station_rows,
        event_rows,
        summary,
        figures_dir=tmp_path,
    )

    assert len(outputs) == 10
    assert sum(path.suffix == ".png" for path in outputs) == 5
    assert sum(path.suffix == ".pdf" for path in outputs) == 5
    assert all(path.is_file() and path.stat().st_size > 10_000 for path in outputs)


def test_phase39_train_test_horizon_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plotting/plot_phase39_train_test_horizons_zh.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--cache-root" in result.stdout
    assert "--test-raw-rates" in result.stdout
    assert "--report-dir" in result.stdout
