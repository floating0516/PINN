from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.evaluation.evaluate_phase49_posthoc_direct_streaming import (
    HORIZONS,
    MODEL_SPECS,
    ModelSpec,
    build_multimodel_tables,
    compare_recovery_metrics,
)


def _constant_rate_for_mw(mw: float) -> float:
    moment = 10.0 ** (1.5 * float(mw) + 9.1)
    return moment / 200.0


def test_phase49_model_order_is_frozen() -> None:
    assert [spec.slug for spec in MODEL_SPECS] == ["phase39", "phase47", "phase48"]
    assert [spec.epoch for spec in MODEL_SPECS] == [200, 19, 188]


def test_recovery_metric_comparison_accepts_frozen_tolerance() -> None:
    spec = ModelSpec(
        slug="phase47",
        label="candidate",
        checkpoint=Path("checkpoint.pth"),
        expected_sha256="abc",
        seed=73,
        epoch=19,
    )
    original = {
        "endpoint_event_mae": 0.1,
        "endpoint_station_mae": 0.1,
        "streaming_event_mae_mean": 0.2,
        "late_event_abs_step_p95_mw": 0.02,
        "late_station_abs_step_p95_mw": 0.03,
        "late_confirmed_cumulative_log10_l1_p95": 0.04,
        "selection_score": 1.0,
    }
    recovered = dict(original)
    recovered["endpoint_event_mae"] += 4.0e-4
    recovered["selection_score"] += 9.0e-3

    result = compare_recovery_metrics(spec, original, recovered)

    assert result["max_metric_abs_difference"] == pytest.approx(4.0e-4)
    assert result["selection_score_abs_difference"] == pytest.approx(9.0e-3)
    assert result["strict_reproduction_passed"] is True


def test_recovery_metric_comparison_rejects_large_difference() -> None:
    spec = ModelSpec(
        slug="phase48",
        label="candidate",
        checkpoint=Path("checkpoint.pth"),
        expected_sha256="abc",
        seed=73,
        epoch=188,
    )
    original = {
        "endpoint_event_mae": 0.1,
        "endpoint_station_mae": 0.1,
        "streaming_event_mae_mean": 0.2,
        "late_event_abs_step_p95_mw": 0.02,
        "late_station_abs_step_p95_mw": 0.03,
        "late_confirmed_cumulative_log10_l1_p95": 0.04,
        "selection_score": 1.0,
    }
    recovered = dict(original)
    recovered["late_event_abs_step_p95_mw"] += 6.0e-4

    with pytest.raises(ValueError, match="recovery metric difference"):
        compare_recovery_metrics(spec, original, recovered)


def test_recovery_metric_comparison_labels_approximate_reconstruction() -> None:
    spec = ModelSpec(
        slug="phase48",
        label="candidate",
        checkpoint=Path("checkpoint.pth"),
        expected_sha256="abc",
        seed=73,
        epoch=188,
        recovery_metric_tolerance=5.0e-3,
        recovery_score_tolerance=3.0e-2,
        recovery_role="user_authorized_approximate_reconstruction",
    )
    original = {
        "endpoint_event_mae": 0.1,
        "endpoint_station_mae": 0.1,
        "streaming_event_mae_mean": 0.2,
        "late_event_abs_step_p95_mw": 0.02,
        "late_station_abs_step_p95_mw": 0.03,
        "late_confirmed_cumulative_log10_l1_p95": 0.04,
        "selection_score": 1.0,
    }
    recovered = dict(original)
    recovered["endpoint_event_mae"] += 3.4e-3
    recovered["selection_score"] += 1.9e-2

    result = compare_recovery_metrics(spec, original, recovered)

    assert result["role"] == "user_authorized_approximate_reconstruction"
    assert result["strict_reproduction_passed"] is False


def test_multimodel_tables_keep_event_medians_and_deltas() -> None:
    rates_by_model: dict[str, np.ndarray] = {}
    station_mw = {
        "phase39": (7.0, 7.2, 6.1),
        "phase47": (7.0, 7.0, 6.0),
        "phase48": (6.9, 7.1, 6.0),
    }
    for slug, values in station_mw.items():
        rates = np.empty((len(HORIZONS), 3, 200), dtype=np.float32)
        for station_index, mw in enumerate(values):
            rates[:, station_index] = _constant_rate_for_mw(mw)
        rates_by_model[slug] = rates

    tables = build_multimodel_tables(
        rates_by_model=rates_by_model,
        available_mask=np.ones((len(HORIZONS), 3), dtype=bool),
        events=["event_a", "event_a", "event_b"],
        stations=["A", "B", "C"],
        catalogs=np.asarray([7.0, 7.0, 6.0], dtype=np.float32),
        source_dt_sec=np.ones(3, dtype=np.float32),
    )

    final = next(
        row
        for row in tables["endpoint_event_rows"]
        if row["event"] == "event_a"
    )
    assert final["phase39_mw"] == pytest.approx(7.1, abs=1.0e-5)
    assert final["phase47_mw"] == pytest.approx(7.0, abs=1.0e-5)
    assert final["phase48_mw"] == pytest.approx(7.0, abs=1.0e-5)
    assert final["phase47_abs_error_change_vs_phase39"] == pytest.approx(-0.1)
    assert len(tables["endpoint_station_rows"]) == 9
    late_row = next(
        row
        for row in tables["event_rows"]
        if row["model"] == "phase39"
        and row["event"] == "event_a"
        and row["observation_horizon_sec"] == 21
    )
    assert late_row["delta_mw"] == pytest.approx(0.0, abs=1.0e-6)


def test_phase49_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/evaluate_phase49_posthoc_direct_streaming.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout
    assert "--batch-size" in result.stdout
