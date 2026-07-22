from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.evaluation.diagnose_legacy_metadata import (
    DIAGNOSTIC_MODES,
    assert_same_station_keys,
    build_diagnostic_metadata,
    summarize_diagnostic,
)


def test_four_metadata_modes_match_frozen_contract() -> None:
    delta_m = 100_000.0
    theta_deg = 30.0
    azimuth_deg = 120.0

    legacy = build_diagnostic_metadata(
        "legacy_exact",
        delta_m=delta_m,
        theta_deg=theta_deg,
        azimuth_deg=azimuth_deg,
    )
    theta_only = build_diagnostic_metadata(
        "theta_only_fixed",
        delta_m=delta_m,
        theta_deg=theta_deg,
        azimuth_deg=azimuth_deg,
    )
    geometry = build_diagnostic_metadata(
        "geometry_fixed",
        delta_m=delta_m,
        theta_deg=theta_deg,
        azimuth_deg=azimuth_deg,
    )
    disabled = build_diagnostic_metadata(
        "metadata_disabled",
        delta_m=delta_m,
        theta_deg=theta_deg,
        azimuth_deg=azimuth_deg,
    )

    np.testing.assert_allclose(
        legacy,
        [math.log(delta_m), math.sin(math.radians(azimuth_deg)), math.cos(math.radians(azimuth_deg)), 0.0, 1.0],
    )
    np.testing.assert_allclose(
        theta_only,
        [math.log(delta_m), math.sin(math.radians(theta_deg)), math.cos(math.radians(theta_deg)), 0.0, 1.0],
    )
    np.testing.assert_allclose(
        geometry,
        [
            math.log(delta_m),
            math.sin(math.radians(theta_deg)),
            math.cos(math.radians(theta_deg)),
            math.sin(math.radians(azimuth_deg)),
            math.cos(math.radians(azimuth_deg)),
        ],
    )
    assert disabled is None
    assert DIAGNOSTIC_MODES == (
        "legacy_exact",
        "theta_only_fixed",
        "geometry_fixed",
        "metadata_disabled",
    )


def test_unknown_metadata_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown metadata mode"):
        build_diagnostic_metadata(
            "wrong",
            delta_m=1.0,
            theta_deg=1.0,
            azimuth_deg=1.0,
        )


def test_station_key_sets_must_match() -> None:
    rows_by_mode = {
        mode: [
            {"event": "A", "station": "S1", "mw_pred": 7.0, "mw_catalog": 7.1}
        ]
        for mode in DIAGNOSTIC_MODES
    }
    assert_same_station_keys(rows_by_mode)

    rows_by_mode["geometry_fixed"] = [
        {"event": "A", "station": "S2", "mw_pred": 7.0, "mw_catalog": 7.1}
    ]
    with pytest.raises(ValueError, match="station key set"):
        assert_same_station_keys(rows_by_mode)


def test_summary_uses_event_medians_and_aligned_prediction_changes() -> None:
    predictions = {
        "legacy_exact": [7.0, 8.0, 8.0],
        "theta_only_fixed": [7.2, 7.8, 8.4],
        "geometry_fixed": [6.8, 8.2, 7.5],
        "metadata_disabled": [7.1, 7.9, 8.2],
    }
    rows_by_mode = {}
    for mode, values in predictions.items():
        rows_by_mode[mode] = [
            {"event": "A", "station": "S1", "mw_pred": values[0], "mw_catalog": 8.0},
            {"event": "A", "station": "S2", "mw_pred": values[1], "mw_catalog": 8.0},
            {"event": "B", "station": "S3", "mw_pred": values[2], "mw_catalog": 8.0},
        ]

    summary = summarize_diagnostic(rows_by_mode)

    assert set(summary) == {
        "legacy_exact_event_mae_catalog",
        "theta_only_fixed_event_mae_catalog",
        "geometry_fixed_event_mae_catalog",
        "metadata_disabled_event_mae_catalog",
        "median_absolute_prediction_change_theta_only",
        "median_absolute_prediction_change_geometry_fixed",
    }
    assert summary["legacy_exact_event_mae_catalog"] == pytest.approx(0.25)
    assert summary["median_absolute_prediction_change_theta_only"] == pytest.approx(0.2)
    assert summary["median_absolute_prediction_change_geometry_fixed"] == pytest.approx(0.2)


def test_script_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/diagnose_legacy_metadata.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--model-dir" in result.stdout
