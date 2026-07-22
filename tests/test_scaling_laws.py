import math
import inspect

import pytest

from src.baseline.scaling_laws import (
    AVAILABLE_SCALING_LAWS,
    CROWELL_2013,
    MELGAR_2015,
    RUHL_2019,
    predict_mw,
    predict_mw_from_coefficients,
)


def test_scaling_law_registry_contains_three_laws():
    assert set(AVAILABLE_SCALING_LAWS) == {"melgar", "ruhl", "crowell"}
    for spec in AVAILABLE_SCALING_LAWS.values():
        assert spec.distance_kind == "hypocentral"
        assert spec.citation_key.endswith(("_2013", "_2015", "_2019"))


def test_public_prediction_api_names_source_distance_explicitly():
    parameters = inspect.signature(predict_mw).parameters
    assert "source_distance_km" in parameters
    assert "distance_km" not in parameters


@pytest.mark.parametrize(
    ("spec", "pgd_m", "source_distance_km", "expected"),
    [
        (MELGAR_2015, 0.10, 100.0, (math.log10(10.0) - (-4.434)) / (1.047 - 0.138 * math.log10(100.0))),
        (RUHL_2019, 0.10, 100.0, (math.log10(0.10) - (-5.919)) / (1.009 - 0.145 * math.log10(100.0))),
        (CROWELL_2013, 0.10, 100.0, (math.log10(10.0) - (-6.687)) / (1.500 - 0.214 * math.log10(100.0))),
    ],
)
def test_predict_mw_from_coefficients_matches_closed_form(spec, pgd_m, source_distance_km, expected):
    mw = predict_mw_from_coefficients(pgd_m=pgd_m, source_distance_km=source_distance_km, spec=spec)
    assert mw == pytest.approx(expected, abs=1.0e-10)


@pytest.mark.parametrize("law_name", ["melgar", "ruhl", "crowell"])
def test_predict_mw_dispatch_accepts_lowercase_names(law_name):
    mw = predict_mw(law_name=law_name, pgd_m=0.05, source_distance_km=50.0)
    assert math.isfinite(mw)


@pytest.mark.parametrize("law_name", ["Melgar", "RUHL", "Crowell"])
def test_predict_mw_dispatch_is_case_insensitive(law_name):
    mw = predict_mw(law_name=law_name, pgd_m=0.05, source_distance_km=50.0)
    assert math.isfinite(mw)


@pytest.mark.parametrize(
    ("pgd_m", "source_distance_km"),
    [
        (0.0, 50.0),
        (-0.1, 50.0),
        (0.05, 0.0),
        (0.05, -10.0),
    ],
)
def test_invalid_inputs_raise_value_error(pgd_m, source_distance_km):
    with pytest.raises(ValueError):
        predict_mw(law_name="melgar", pgd_m=pgd_m, source_distance_km=source_distance_km)


def test_unknown_scaling_law_raises_value_error():
    with pytest.raises(ValueError):
        predict_mw(law_name="unknown", pgd_m=0.05, source_distance_km=50.0)
