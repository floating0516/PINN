import math
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.evaluate_pgd_scaling_laws import (
    azimuth_deg,
    compute_horizontal_pgd,
    compute_pgd_3d,
    compute_radial,
    haversine_m,
    parse_law_names,
    summarize_event,
)


def test_script_exposes_importable_helpers():
    assert callable(haversine_m)
    assert callable(compute_pgd_3d)


def test_event_level_median_aggregation():
    median, iqr, n_stations = summarize_event([7.1, 7.3, 7.2])
    assert median == 7.2
    assert iqr == pytest.approx(0.1)
    assert n_stations == 3


def test_haversine_zero_distance_is_zero():
    assert haversine_m(35.0, 140.0, 35.0, 140.0) == 0.0


def test_azimuth_due_north_is_zero_deg():
    assert azimuth_deg(35.0, 140.0, 36.0, 140.0) == 0.0


def test_compute_radial_uses_event_to_station_azimuth():
    radial = compute_radial(np.array([1.0]), np.array([2.0]), az_deg=0.0)
    assert radial.shape == (1,)
    assert radial[0] == 2.0


def test_compute_horizontal_pgd_matches_vector_peak():
    e = np.array([0.0, 3.0, 0.0])
    n = np.array([0.0, 4.0, 0.0])
    assert compute_horizontal_pgd(e, n) == 5.0


def test_compute_pgd_3d_matches_vector_peak():
    e = np.array([0.0, 3.0, 0.0])
    n = np.array([0.0, 4.0, 0.0])
    u = np.array([0.0, 12.0, 0.0])
    assert compute_pgd_3d(e, n, u) == 13.0


def test_parse_law_names_expands_all():
    names = parse_law_names(["all"])
    assert names == ["crowell", "ruhl", "melgar"]


def test_parse_law_names_rejects_unknown():
    try:
        parse_law_names(["unknown"])
    except ValueError as exc:
        assert "未知方法" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_plot_outputs_creates_scatter_and_bar_figures(tmp_path: Path):
    from scripts.evaluation.evaluate_pgd_scaling_laws import write_plot_outputs

    rows = [
        {"event": "E1", "mw_ref": 7.0, "method": "melgar", "mw_pred": 7.1, "error": 0.1, "iqr": 0.2, "n_stations": 3},
        {"event": "E2", "mw_ref": 8.0, "method": "melgar", "mw_pred": 7.8, "error": -0.2, "iqr": 0.1, "n_stations": 4},
        {"event": "E1", "mw_ref": 7.0, "method": "crowell", "mw_pred": 6.9, "error": -0.1, "iqr": 0.3, "n_stations": 3},
        {"event": "E2", "mw_ref": 8.0, "method": "crowell", "mw_pred": 8.2, "error": 0.2, "iqr": 0.2, "n_stations": 4},
    ]

    paths = write_plot_outputs(rows, tmp_path / "pgd_scaling_law_results.csv")

    assert paths["scatter"].exists()
    assert paths["bar"].exists()
    assert paths["guide_offset"] == pytest.approx(0.3)
    assert paths["style"] == "paper"
