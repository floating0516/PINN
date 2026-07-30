from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.plotting.plot_phase39_pgd_horizon_comparison_zh import (
    METHOD_ORDER,
    RawPGDRecord,
    causal_pgd_3d,
    evaluate_frozen_comparison,
    plot_all_horizons,
)


def _raw_record() -> RawPGDRecord:
    time_sec = np.arange(-10.0, 211.0)
    east = np.zeros_like(time_sec)
    north = np.zeros_like(time_sec)
    up = np.zeros_like(time_sec)
    east[time_sec == 20.0] = 3.0
    north[time_sec == 20.0] = 4.0
    up[time_sec == 20.0] = 12.0
    return RawPGDRecord(
        event="E",
        station="S",
        time_sec=time_sec,
        east_m=east,
        north_m=north,
        up_m=up,
        source_distance_km=100.0,
        p_arrival_sec=15.0,
        magnitude_catalog=7.0,
    )


def test_causal_pgd_3d_uses_only_released_prefix() -> None:
    record = _raw_record()
    pgd, available, observed = causal_pgd_3d(
        record,
        observation_horizon_sec=30,
    )
    changed_up = record.up_m.copy()
    changed_up[record.time_sec > 36.0] = 1.0e6
    changed = replace(record, up_m=changed_up)
    changed_pgd, changed_available, changed_observed = causal_pgd_3d(
        changed,
        observation_horizon_sec=30,
    )

    assert pgd == pytest.approx(13.0)
    assert changed_pgd == pytest.approx(pgd)
    assert changed_available == available
    assert changed_observed == observed


def test_frozen_phase39_pgd_comparison_has_aligned_cohorts() -> None:
    station_rows, event_rows, metrics, summary = evaluate_frozen_comparison()

    assert summary["station_cohort_gate"]["passed"] is True
    assert summary["event_cohort_gate"]["passed"] is True
    assert len(station_rows) == 10_865
    assert len(event_rows) == 1_220
    assert len(metrics) == 40
    lookup = {
        (row["split"], row["observation_horizon_sec"], row["method"]): row
        for row in metrics
    }
    assert lookup[("test", 60, "crowell")]["event_mae_mw"] == pytest.approx(
        0.45435531510291
    )
    assert lookup[("test", 120, "phase39")]["event_mae_mw"] == pytest.approx(
        0.3976712461625599
    )
    assert lookup[("test", 200, "crowell")]["event_mae_mw"] == pytest.approx(
        0.22885290007072898
    )
    assert lookup[("test", 200, "phase39")]["event_mae_mw"] == pytest.approx(
        0.15228705818768482
    )


def _synthetic_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    event_rows: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for split_index, split in enumerate(("train", "test")):
        for horizon in (30, 60, 90, 120, 200):
            for method_index, method in enumerate(METHOD_ORDER):
                predictions = []
                catalogs = []
                for event_index, catalog in enumerate((6.5, 7.5, 8.5)):
                    prediction = (
                        catalog
                        + 0.25 * (200 - horizon) / 170.0
                        - 0.08 * method_index
                        + 0.02 * split_index
                    )
                    predictions.append(prediction)
                    catalogs.append(catalog)
                    event_rows.append(
                        {
                            "split": split,
                            "observation_horizon_sec": horizon,
                            "release_time_sec": horizon + 6.0,
                            "event": f"E{event_index}",
                            "method": method,
                            "method_label": method,
                            "mw_catalog": catalog,
                            "mw_pred_median": prediction,
                            "error_mw": prediction - catalog,
                            "abs_error_mw": abs(prediction - catalog),
                            "station_count": 4,
                        }
                    )
                errors = np.asarray(predictions) - np.asarray(catalogs)
                metrics.append(
                    {
                        "split": split,
                        "observation_horizon_sec": horizon,
                        "method": method,
                        "event_mae_mw": float(np.mean(np.abs(errors))),
                        "event_bias_mw": float(np.mean(errors)),
                        "event_within_0_15_fraction": float(
                            np.mean(np.abs(errors) <= 0.15)
                        ),
                    }
                )
    return event_rows, metrics


def test_plot_all_horizons_writes_five_pngs_and_pdfs(tmp_path: Path) -> None:
    event_rows, metrics = _synthetic_rows()

    outputs = plot_all_horizons(event_rows, metrics, figures_dir=tmp_path)

    assert len(outputs) == 10
    assert sum(path.suffix == ".png" for path in outputs) == 5
    assert sum(path.suffix == ".pdf" for path in outputs) == 5
    assert all(path.is_file() and path.stat().st_size > 10_000 for path in outputs)


def test_phase39_pgd_horizon_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plotting/plot_phase39_pgd_horizon_comparison_zh.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--report-dir" in result.stdout
    assert "--full-regression" in result.stdout
