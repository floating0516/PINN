from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.data.waveform import (
    WaveformConfig,
    _convert_to_metres,
    _estimate_baseline,
    _fir_lowpass,
    _sort_and_average_duplicates,
    preprocess_waveform,
)


def _config(**overrides: object) -> WaveformConfig:
    values = {
        "sample_rate_hz": 1.0,
        "start_sec": 0.0,
        "duration_sec": 20.0,
        "min_valid_fraction": 0.9,
        "max_interpolation_gap_sec": 2.5,
        "baseline_method": "mean",
        "pre_event_start_sec": -5.0,
        "pre_event_end_sec": 0.0,
        "baseline_fallback": "pre_p",
        "baseline_fallback_max_sec": 10.0,
        "baseline_min_samples": 3,
        "filter_type": "none",
        "cutoff_hz": 0.1,
        "num_taps": 7,
        "filter_window": "hamming",
    }
    values.update(overrides)
    return WaveformConfig(**values)


def test_irregular_waveform_is_resampled_to_one_hz() -> None:
    regular_time = np.arange(-5.0, 20.0, 0.5)
    time_sec = regular_time + 0.08 * np.sin(np.arange(regular_time.size))
    offset_m = 2.0
    slope_m_per_sec = 0.01
    values = offset_m + slope_m_per_sec * time_sec
    config = _config()

    result = preprocess_waveform(
        time_sec,
        values,
        units="m",
        p_arrival_sec=8.0,
        config=config,
    )

    pre_event = (
        (time_sec >= config.pre_event_start_sec)
        & (time_sec < config.pre_event_end_sec)
    )
    expected_baseline_m = float(np.mean(values[pre_event]))
    expected_values_m = (
        offset_m
        + slope_m_per_sec * np.arange(20.0)
        - expected_baseline_m
    )

    assert result.values_m.shape == (20,)
    assert result.valid_mask.shape == (20,)
    assert result.time_sec.dtype == np.float32
    assert result.values_m.dtype == np.float32
    assert result.valid_mask.dtype == np.bool_
    assert result.dt_sec == 1.0
    assert result.raw_dt_sec == pytest.approx(np.median(np.diff(time_sec)))
    assert result.baseline_source == "pre_event"
    assert result.baseline_m == pytest.approx(expected_baseline_m)
    assert result.valid_fraction == 1.0
    assert np.all(result.valid_mask)
    assert np.allclose(result.time_sec, np.arange(20.0))
    np.testing.assert_allclose(
        result.values_m,
        expected_values_m,
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_grid_point_near_right_timestamp_is_exact_within_tolerance() -> None:
    time_sec = np.concatenate(
        ([-5.0, -4.0, -3.0, 5.0e-9], np.arange(1.0, 20.0))
    )
    values = np.ones_like(time_sec)

    result = preprocess_waveform(
        time_sec,
        values,
        units="m",
        p_arrival_sec=8.0,
        config=_config(),
    )

    assert result.valid_mask[0]
    assert result.valid_fraction == 1.0


def test_pre_event_baseline_is_subtracted_from_all_samples() -> None:
    time_sec = np.arange(-5.0, 20.0)
    values = np.where(time_sec < 0.0, 2.0, 2.0 + 0.1 * time_sec)

    result = preprocess_waveform(
        time_sec,
        values,
        units="m",
        p_arrival_sec=8.0,
        config=_config(),
    )

    assert result.baseline_source == "pre_event"
    assert result.baseline_m == pytest.approx(2.0)
    assert result.values_m[0] == pytest.approx(0.0, abs=1.0e-6)
    assert result.values_m[5] == pytest.approx(0.5, abs=1.0e-6)
    assert result.values_m[5] > result.values_m[1]


def test_pre_p_fallback_does_not_flatten_pre_p_segment() -> None:
    time_sec = np.arange(0.0, 20.0)
    values = 1.0 + 0.01 * time_sec

    result = preprocess_waveform(
        time_sec,
        values,
        units="m",
        p_arrival_sec=8.0,
        config=_config(),
    )

    assert result.baseline_source == "pre_p"
    assert result.baseline_m == pytest.approx(np.mean(values[:8]))
    assert np.std(result.values_m[:8]) > 0.0
    assert not np.allclose(result.values_m[:8], 0.0)


def test_short_record_below_valid_fraction_is_rejected() -> None:
    time_sec = np.arange(0.0, 10.0)
    values = np.ones_like(time_sec)

    with pytest.raises(ValueError, match="valid fraction"):
        preprocess_waveform(
            time_sec,
            values,
            units="m",
            p_arrival_sec=8.0,
            config=_config(),
        )


@pytest.mark.parametrize(
    ("units", "values", "expected"),
    [
        ("m", [1.0, -2.0], [1.0, -2.0]),
        ("cm", [100.0, -250.0], [1.0, -2.5]),
        ("mm", [1000.0, -2500.0], [1.0, -2.5]),
    ],
)
def test_explicit_units_are_converted_to_metres(
    units: str,
    values: list[float],
    expected: list[float],
) -> None:
    converted = _convert_to_metres(np.asarray(values), units)

    assert converted.dtype == np.float64
    assert np.allclose(converted, expected)


def test_public_preprocessing_preserves_physical_values_across_units() -> None:
    time_sec = np.arange(-5.0, 20.0, 0.5)
    physical_values_m = (
        0.02 + 0.001 * time_sec + 0.005 * np.maximum(time_sec, 0.0)
    )
    values_by_unit = {
        "m": physical_values_m,
        "cm": physical_values_m * 100.0,
        "mm": physical_values_m * 1000.0,
    }

    results = {
        units: preprocess_waveform(
            time_sec,
            values,
            units=units,
            p_arrival_sec=8.0,
            config=_config(),
        )
        for units, values in values_by_unit.items()
    }

    pre_event = (time_sec >= -5.0) & (time_sec < 0.0)
    expected_baseline_m = float(np.mean(physical_values_m[pre_event]))
    reference = results["m"]
    for result in results.values():
        assert result.baseline_source == "pre_event"
        assert result.baseline_m == pytest.approx(expected_baseline_m)
        assert np.array_equal(result.valid_mask, reference.valid_mask)
        np.testing.assert_allclose(
            result.values_m,
            reference.values_m,
            rtol=1.0e-6,
            atol=1.0e-8,
        )


@pytest.mark.parametrize("units", ["auto", "km", ""])
def test_unsupported_or_automatic_units_are_rejected(units: str) -> None:
    with pytest.raises(ValueError, match="units"):
        _convert_to_metres(np.ones(2), units)


def test_nonfinite_pairs_are_dropped_sorted_and_duplicates_averaged() -> None:
    time_sec = np.array([2.0, np.nan, 1.0, 1.0, 0.0, 3.0, np.inf])
    values_m = np.array([4.0, 99.0, 1.0, 3.0, 0.0, np.nan, 7.0])

    clean_time, clean_values = _sort_and_average_duplicates(time_sec, values_m)

    assert np.array_equal(clean_time, [0.0, 1.0, 2.0])
    assert np.array_equal(clean_values, [0.0, 2.0, 4.0])


@pytest.mark.parametrize(
    ("time_sec", "values_m"),
    [
        ([0.0, np.nan], [1.0, 2.0]),
        ([1.0, 1.0, np.inf], [2.0, 4.0, 6.0]),
    ],
)
def test_too_few_finite_or_unique_timestamps_are_rejected(
    time_sec: list[float],
    values_m: list[float],
) -> None:
    with pytest.raises(ValueError):
        _sort_and_average_duplicates(np.asarray(time_sec), np.asarray(values_m))


def test_median_baseline_method_is_supported() -> None:
    config = _config(baseline_method="median", baseline_min_samples=3)
    time_sec = np.array([-3.0, -2.0, -1.0, 0.0])
    values_m = np.array([2.0, 2.0, 100.0, 3.0])

    baseline_m, source = _estimate_baseline(
        time_sec,
        values_m,
        p_arrival_sec=8.0,
        config=config,
    )

    assert baseline_m == 2.0
    assert source == "pre_event"


def test_insufficient_pre_event_and_pre_p_baseline_is_rejected() -> None:
    with pytest.raises(ValueError, match="insufficient baseline"):
        _estimate_baseline(
            np.arange(0.0, 5.0),
            np.ones(5),
            p_arrival_sec=2.0,
            config=_config(baseline_min_samples=3),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"baseline_method": "mode"},
        {"baseline_fallback": "whole_trace"},
    ],
)
def test_unsupported_baseline_settings_are_rejected(
    overrides: dict[str, object],
) -> None:
    config = _config(**overrides)
    if "baseline_fallback" in overrides:
        time_sec = np.arange(0.0, 20.0)
    else:
        time_sec = np.arange(-5.0, 20.0)

    with pytest.raises(ValueError, match="baseline"):
        _estimate_baseline(
            time_sec,
            np.ones_like(time_sec),
            p_arrival_sec=8.0,
            config=config,
        )


def test_outside_support_and_large_gaps_stay_masked_through_filtering() -> None:
    time_sec = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 6.0, 7.0, 8.0])
    values = 10.0 + time_sec
    config = _config(
        duration_sec=10.0,
        min_valid_fraction=0.0,
        pre_event_start_sec=-2.0,
        baseline_min_samples=2,
        filter_type="lowpass",
    )

    result = preprocess_waveform(
        time_sec,
        values,
        units="m",
        p_arrival_sec=8.0,
        config=config,
    )

    expected_mask = np.array(
        [True, True, True, False, False, False, True, True, True, False]
    )
    centered = values - np.mean(values[:2])
    interpolated = np.interp(result.time_sec, time_sec, centered)
    expected_input = np.where(expected_mask, interpolated, 0.0)
    expected_values = _fir_lowpass(expected_input, config)
    expected_values = np.where(expected_mask, expected_values, 0.0)

    assert np.array_equal(result.valid_mask, expected_mask)
    assert result.valid_fraction == pytest.approx(0.6)
    assert np.all(result.values_m[~expected_mask] == 0.0)
    assert np.allclose(result.values_m, expected_values)


def test_zero_gap_path_gathers_exact_nodes_and_never_interpolates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time_sec = np.array([-3.0, -2.0, -1.0, 0.27, 1.27, 3.27])
    values = np.array([10.0, 10.0, 10.0, 11.0, 12.0, 14.0])
    config = _config(
        start_sec=0.27,
        duration_sec=4.0,
        min_valid_fraction=0.0,
        max_interpolation_gap_sec=0.0,
    )
    monkeypatch.setattr(
        np,
        "interp",
        lambda *_args, **_kwargs: pytest.fail(
            "strict zero-gap preprocessing must not call np.interp"
        ),
    )

    result = preprocess_waveform(
        time_sec,
        values,
        units="m",
        p_arrival_sec=8.0,
        config=config,
    )

    assert result.time_sec == pytest.approx([0.27, 1.27, 2.27, 3.27])
    assert np.array_equal(result.valid_mask, [True, True, False, True])
    assert result.values_m == pytest.approx([1.0, 2.0, 0.0, 4.0])


def test_lowpass_uses_configured_windowed_sinc_fir() -> None:
    config = _config(filter_type="lowpass", num_taps=7, cutoff_hz=0.1)
    values = np.zeros(21)
    values[10] = 1.0

    filtered = _fir_lowpass(values, config)

    taps = config.num_taps
    n = np.arange(taps, dtype=np.float64)
    midpoint = 0.5 * (taps - 1)
    normalized_cutoff = config.cutoff_hz / config.sample_rate_hz
    kernel = 2.0 * normalized_cutoff * np.sinc(
        2.0 * normalized_cutoff * (n - midpoint)
    )
    kernel *= np.hamming(taps)
    kernel /= kernel.sum()
    expected = np.convolve(values, kernel, mode="same")

    assert np.allclose(filtered, expected)


@pytest.mark.parametrize(
    "overrides",
    [
        {"filter_type": "highpass"},
        {"filter_type": "lowpass", "num_taps": 2},
        {"filter_type": "lowpass", "num_taps": 4},
        {"filter_type": "lowpass", "cutoff_hz": 0.0},
        {"filter_type": "lowpass", "cutoff_hz": 0.5},
        {"filter_type": "lowpass", "filter_window": "blackman"},
    ],
)
def test_invalid_filter_settings_are_rejected(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _fir_lowpass(np.ones(20), _config(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"sample_rate_hz": 0.0},
        {"sample_rate_hz": -1.0},
        {"duration_sec": 0.0},
        {"duration_sec": -1.0},
    ],
)
def test_nonpositive_sample_rate_or_duration_is_rejected(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        preprocess_waveform(
            np.arange(-5.0, 20.0),
            np.ones(25),
            units="m",
            p_arrival_sec=8.0,
            config=_config(**overrides),
        )


def test_config_and_result_records_are_frozen() -> None:
    config = _config()
    result = preprocess_waveform(
        np.arange(-5.0, 20.0),
        np.ones(25),
        units="m",
        p_arrival_sec=8.0,
        config=config,
    )

    with pytest.raises(FrozenInstanceError):
        config.duration_sec = 10.0
    with pytest.raises(FrozenInstanceError):
        result.dt_sec = 2.0
