from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.plotting.plot_phase39_internal_test_scatter_zh import (
    EXPECTED_EVENT_COUNT,
    EXPECTED_EVENT_MAE,
    EXPECTED_STATION_COUNT,
    EXPECTED_STATION_MAE,
    _error_metrics,
    load_frozen_endpoint,
    plot_endpoint_scatter,
)


def test_error_metrics_report_accuracy_and_coverage() -> None:
    catalog = np.asarray([6.0, 7.0, 8.0], dtype=np.float64)
    prediction = np.asarray([6.1, 6.8, 8.4], dtype=np.float64)

    metrics = _error_metrics(catalog, prediction)

    assert metrics["count"] == 3
    assert metrics["mae_mw"] == pytest.approx((0.1 + 0.2 + 0.4) / 3.0)
    assert metrics["bias_mw"] == pytest.approx(0.1)
    assert metrics["within_0_15_count"] == 1
    assert metrics["within_0_30_count"] == 2


def test_frozen_phase39_endpoint_reconstructs_published_metrics() -> None:
    station_rows, event_rows, summary = load_frozen_endpoint()

    assert len(station_rows) == EXPECTED_STATION_COUNT
    assert len(event_rows) == EXPECTED_EVENT_COUNT
    assert summary["station_metrics"]["mae_mw"] == pytest.approx(
        EXPECTED_STATION_MAE,
        abs=1.0e-12,
    )
    assert summary["event_metrics"]["mae_mw"] == pytest.approx(
        EXPECTED_EVENT_MAE,
        abs=1.0e-12,
    )
    assert summary["evaluation_role"] == "within_event_station_internal_test"
    assert summary["largest_event_errors"][0]["event"] == "Lefkada2015"


def test_phase39_endpoint_scatter_writes_png_and_pdf(tmp_path: Path) -> None:
    station_rows, event_rows, summary = load_frozen_endpoint()

    outputs = plot_endpoint_scatter(
        station_rows,
        event_rows,
        summary,
        output_stem=tmp_path / "scatter",
    )

    assert [path.suffix for path in outputs] == [".png", ".pdf"]
    assert all(path.is_file() and path.stat().st_size > 10_000 for path in outputs)


def test_phase39_internal_scatter_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plotting/plot_phase39_internal_test_scatter_zh.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--source-stations" in result.stdout
    assert "--report-dir" in result.stdout
