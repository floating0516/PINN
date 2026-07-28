from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts.evaluation.evaluate_phase45_posthoc_streaming import (
    EXPECTED_SELECTED_EPOCH,
    HORIZONS,
    PROCESSING_DELAY_SEC,
    _contiguous_segments,
    _suffix_stable_horizon,
    apply_adapter_to_rates,
    build_comparison_tables,
    build_convergence_rows,
)
from src.models.streaming_stf_adapter import StreamingSTFAdapter


def _constant_rate_for_mw(mw: float) -> float:
    moment = 10.0 ** (1.5 * float(mw) + 9.1)
    return moment / 200.0


def test_phase45_posthoc_contract_is_frozen() -> None:
    assert PROCESSING_DELAY_SEC == pytest.approx(6.0)
    assert EXPECTED_SELECTED_EPOCH == 27
    assert HORIZONS == tuple(range(20, 201))


def test_contiguous_segments_does_not_bridge_missing_horizons() -> None:
    indices = np.asarray([0, 1, 2, 5, 6, 10], dtype=np.int64)

    assert _contiguous_segments(indices) == [(0, 3), (5, 7), (10, 11)]
    assert _contiguous_segments(np.asarray([], dtype=np.int64)) == []


def test_suffix_stable_horizon_requires_all_later_predictions_to_pass() -> None:
    rows = [
        {"observation_horizon_sec": 20, "abs_error": 0.10},
        {"observation_horizon_sec": 21, "abs_error": 0.20},
        {"observation_horizon_sec": 22, "abs_error": 0.14},
        {"observation_horizon_sec": 23, "abs_error": 0.12},
    ]

    assert _suffix_stable_horizon(rows, error_key="abs_error") == 22


def test_comparison_tables_use_station_median_for_each_event() -> None:
    station_mw_raw = np.asarray([7.0, 7.2, 6.1], dtype=np.float64)
    station_mw_adapted = np.asarray([7.0, 7.0, 6.0], dtype=np.float64)
    raw = np.empty((len(HORIZONS), 3, 200), dtype=np.float32)
    adapted = np.empty_like(raw)
    for station_index, mw in enumerate(station_mw_raw):
        raw[:, station_index] = _constant_rate_for_mw(float(mw))
    for station_index, mw in enumerate(station_mw_adapted):
        adapted[:, station_index] = _constant_rate_for_mw(float(mw))

    tables = build_comparison_tables(
        raw_rates=raw,
        adapted_rates=adapted,
        available_mask=np.ones(raw.shape[:2], dtype=bool),
        events=["event_a", "event_a", "event_b"],
        stations=["A", "B", "C"],
        catalogs=np.asarray([7.0, 7.0, 6.0], dtype=np.float32),
        source_dt_sec=np.ones(3, dtype=np.float32),
        include_station_trajectories=False,
    )

    final = [
        row
        for row in tables["event_rows"]
        if row["observation_horizon_sec"] == 200
    ]
    event_a = next(row for row in final if row["event"] == "event_a")
    assert event_a["raw_mw_pred_median"] == pytest.approx(7.1, abs=1.0e-5)
    assert event_a["adapted_mw_pred_median"] == pytest.approx(7.0, abs=1.0e-5)
    assert len(tables["endpoint_station_rows"]) == 3
    assert tables["station_rows"] == []


def test_adapter_application_restarts_after_an_availability_gap(tmp_path: Path) -> None:
    torch.manual_seed(42)
    adapter = StreamingSTFAdapter(source_steps=200)
    raw = np.full((len(HORIZONS), 1, 200), np.nan, dtype=np.float32)
    mask = np.zeros((len(HORIZONS), 1), dtype=bool)
    mask[0:2, 0] = True
    mask[5:7, 0] = True
    raw[0, 0] = 1.0e18
    raw[1, 0] = 2.0e18
    raw[5, 0] = 5.0e18
    raw[6, 0] = 6.0e18

    adapted = apply_adapter_to_rates(
        adapter=adapter,
        raw_rates=raw,
        available_mask=mask,
        source_distance_m=np.asarray([0.0], dtype=np.float32),
        source_dt_sec=np.asarray([1.0], dtype=np.float32),
        beta_m_per_s=4_533.0,
        output_path=tmp_path / "adapted.npy",
        batch_size=1,
    )

    assert np.array_equal(np.all(np.isfinite(adapted), axis=2), mask)
    assert np.array_equal(adapted[0, 0], raw[0, 0])
    assert np.array_equal(adapted[5, 0], raw[5, 0])


def test_convergence_rows_report_both_methods() -> None:
    event_rows = [
        {
            "event": "event_a",
            "observation_horizon_sec": 199,
            "mw_catalog": 7.0,
            "raw_mw_pred_median": 7.2,
            "adapted_mw_pred_median": 7.1,
            "raw_abs_error": 0.2,
            "adapted_abs_error": 0.1,
            "station_count": 2,
        },
        {
            "event": "event_a",
            "observation_horizon_sec": 200,
            "mw_catalog": 7.0,
            "raw_mw_pred_median": 7.1,
            "adapted_mw_pred_median": 7.1,
            "raw_abs_error": 0.1,
            "adapted_abs_error": 0.1,
            "station_count": 2,
        },
    ]

    row = build_convergence_rows(event_rows)[0]

    assert row["raw_stable_observation_sec"] == 200
    assert row["adapted_stable_observation_sec"] == 199
    assert row["adapted_stable_release_sec"] == pytest.approx(205.0)


def test_phase45_posthoc_script_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/evaluate_phase45_posthoc_streaming.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--stage" in result.stdout
    assert "external" in result.stdout
