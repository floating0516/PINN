import inspect
import math

import numpy as np
import pytest

from src.data import stf as stf_module
from src.data.stf import (
    ProcessedSTF,
    STFWindowTooShort,
    moment_to_mw,
    resample_source_stf,
)


def _triangle_stf() -> tuple[np.ndarray, np.ndarray]:
    time_sec = np.arange(0.0, 100.5, 0.5)
    rate_nm_per_s = (
        np.maximum(0.0, 1.0 - np.abs(time_sec - 40.0) / 30.0)
        * 1.0e18
    )
    return time_sec, rate_nm_per_s


def test_source_stf_preserves_discrete_moment() -> None:
    time_sec, rate_nm_per_s = _triangle_stf()

    result = resample_source_stf(
        time_sec,
        rate_nm_per_s,
        start_sec=0.0,
        duration_sec=120.0,
        sample_rate_hz=1.0,
        min_retained_moment_fraction=0.995,
        preserve_integral=True,
    )

    discrete_moment_nm = float(
        np.sum(result.rate_nm_per_s) * result.dt_sec
    )
    assert (
        abs(discrete_moment_nm - result.native_moment_nm)
        / result.native_moment_nm
        < 1.0e-10
    )
    assert result.retained_moment_fraction >= 0.995


def test_source_stf_has_fixed_two_hundred_step_output() -> None:
    time_sec, rate_nm_per_s = _triangle_stf()

    result = resample_source_stf(
        time_sec,
        rate_nm_per_s,
        start_sec=0.0,
        duration_sec=200.0,
        sample_rate_hz=1.0,
        min_retained_moment_fraction=0.995,
        preserve_integral=True,
    )

    assert result.time_sec.shape == (200,)
    assert result.rate_nm_per_s.shape == (200,)
    np.testing.assert_array_equal(result.time_sec, np.arange(200.0))


def test_station_window_shifts_fractionally_and_truncates_without_rescaling() -> None:
    source = ProcessedSTF(
        time_sec=np.arange(4.0),
        rate_nm_per_s=np.array([0.0, 1.0, 2.0, 1.0]),
        dt_sec=1.0,
        native_moment_nm=4.0,
        grid_moment_before_rescale_nm=4.0,
        retained_moment_fraction=1.0,
        mw_native=moment_to_mw(4.0),
    )

    shifted = stf_module.shift_source_stf_to_station_window(
        source,
        p_delay_sec=2.5,
        duration_sec=5.0,
        sample_rate_hz=1.0,
    )

    np.testing.assert_allclose(
        shifted.rate_nm_per_s,
        [0.0, 0.0, 0.0, 0.5, 1.5],
    )
    assert shifted.window_moment_nm == pytest.approx(2.0)
    assert shifted.full_event_moment_nm == pytest.approx(4.0)
    assert shifted.retained_moment_fraction == pytest.approx(0.5)


def test_station_window_allows_complete_tail_loss() -> None:
    source = ProcessedSTF(
        time_sec=np.arange(3.0),
        rate_nm_per_s=np.array([1.0, 1.0, 1.0]),
        dt_sec=1.0,
        native_moment_nm=3.0,
        grid_moment_before_rescale_nm=3.0,
        retained_moment_fraction=1.0,
        mw_native=moment_to_mw(3.0),
    )

    shifted = stf_module.shift_source_stf_to_station_window(
        source,
        p_delay_sec=10.0,
        duration_sec=3.0,
        sample_rate_hz=1.0,
    )

    assert np.count_nonzero(shifted.rate_nm_per_s) == 0
    assert shifted.window_moment_nm == 0.0
    assert shifted.retained_moment_fraction == 0.0
    assert math.isnan(shifted.mw_window)


@pytest.mark.parametrize("delay", [-1.0, math.nan, math.inf])
def test_station_window_rejects_invalid_p_delay(delay: float) -> None:
    source = ProcessedSTF(
        time_sec=np.arange(3.0),
        rate_nm_per_s=np.ones(3),
        dt_sec=1.0,
        native_moment_nm=3.0,
        grid_moment_before_rescale_nm=3.0,
        retained_moment_fraction=1.0,
        mw_native=moment_to_mw(3.0),
    )

    with pytest.raises(ValueError, match="p_delay_sec"):
        stf_module.shift_source_stf_to_station_window(
            source,
            p_delay_sec=delay,
            duration_sec=300.0,
            sample_rate_hz=1.0,
        )


def test_source_stf_api_has_no_station_dependent_parameter() -> None:
    parameters = set(inspect.signature(resample_source_stf).parameters)

    assert parameters == {
        "time_sec",
        "rate_nm_per_s",
        "start_sec",
        "duration_sec",
        "sample_rate_hz",
        "min_retained_moment_fraction",
        "preserve_integral",
    }
    assert parameters.isdisjoint(
        {
            "distance",
            "distance_m",
            "p_shift_sec",
            "p_velocity_mps",
            "station",
        }
    )


def test_same_event_stf_is_repeatable_without_station_context() -> None:
    time_sec = np.arange(0.0, 50.0)
    rate_nm_per_s = np.ones_like(time_sec) * 1.0e18
    kwargs = {
        "start_sec": 0.0,
        "duration_sec": 60.0,
        "sample_rate_hz": 1.0,
        "min_retained_moment_fraction": 0.995,
        "preserve_integral": True,
    }

    first = resample_source_stf(time_sec, rate_nm_per_s, **kwargs)
    second = resample_source_stf(time_sec, rate_nm_per_s, **kwargs)

    assert np.array_equal(first.rate_nm_per_s, second.rate_nm_per_s)
    assert first.native_moment_nm == second.native_moment_nm
    assert first.mw_native == second.mw_native


def test_window_that_loses_continuous_moment_fails() -> None:
    time_sec = np.arange(0.0, 300.0)
    rate_nm_per_s = np.ones_like(time_sec) * 1.0e18

    with pytest.raises(STFWindowTooShort, match="retains"):
        resample_source_stf(
            time_sec,
            rate_nm_per_s,
            start_sec=0.0,
            duration_sec=100.0,
            sample_rate_hz=1.0,
            min_retained_moment_fraction=0.995,
            preserve_integral=True,
        )


def test_preserve_integral_false_leaves_interpolated_grid_unscaled() -> None:
    time_sec = np.array([0.0, 1.0, 2.0])
    rate_nm_per_s = np.ones(3)

    result = resample_source_stf(
        time_sec,
        rate_nm_per_s,
        start_sec=0.0,
        duration_sec=3.0,
        sample_rate_hz=1.0,
        min_retained_moment_fraction=1.0,
        preserve_integral=False,
    )

    assert result.native_moment_nm == pytest.approx(2.0)
    assert result.grid_moment_before_rescale_nm == pytest.approx(3.0)
    assert np.sum(result.rate_nm_per_s) * result.dt_sec == pytest.approx(3.0)


def test_nonfinite_samples_are_dropped_and_duplicate_rates_are_averaged() -> None:
    time_sec = np.array([2.0, 1.0, 1.0, 0.0, np.nan, 3.0])
    rate_nm_per_s = np.array([0.0, 2.0, 4.0, 0.0, 99.0, np.inf])

    result = resample_source_stf(
        time_sec,
        rate_nm_per_s,
        start_sec=0.0,
        duration_sec=3.0,
        sample_rate_hz=1.0,
        min_retained_moment_fraction=1.0,
        preserve_integral=False,
    )

    np.testing.assert_allclose(result.rate_nm_per_s, [0.0, 3.0, 0.0])
    assert result.native_moment_nm == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"duration_sec": 0.0}, "duration_sec"),
        ({"sample_rate_hz": 0.0}, "sample_rate_hz"),
        ({"start_sec": math.nan}, "start_sec"),
        (
            {"min_retained_moment_fraction": 0.0},
            "min_retained_moment_fraction",
        ),
        (
            {"min_retained_moment_fraction": 1.1},
            "min_retained_moment_fraction",
        ),
    ],
)
def test_invalid_grid_parameters_are_rejected(
    overrides: dict[str, float],
    message: str,
) -> None:
    kwargs = {
        "start_sec": 0.0,
        "duration_sec": 3.0,
        "sample_rate_hz": 1.0,
        "min_retained_moment_fraction": 1.0,
        "preserve_integral": True,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        resample_source_stf(
            np.arange(3.0),
            np.ones(3),
            **kwargs,
        )


def test_mismatched_input_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        resample_source_stf(
            np.arange(3.0),
            np.ones(2),
            start_sec=0.0,
            duration_sec=3.0,
            sample_rate_hz=1.0,
            min_retained_moment_fraction=1.0,
            preserve_integral=True,
        )


@pytest.mark.parametrize("moment_nm", [0.0, -1.0, math.nan, math.inf])
def test_moment_to_mw_rejects_nonpositive_or_nonfinite_moment(
    moment_nm: float,
) -> None:
    with pytest.raises(ValueError, match="moment_nm"):
        moment_to_mw(moment_nm)


def test_moment_to_mw_uses_si_moment_contract() -> None:
    assert moment_to_mw(1.0e18) == pytest.approx(
        (2.0 / 3.0) * (18.0 - 9.1)
    )
