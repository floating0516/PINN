from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


EXACT_TIME_TOLERANCE_SEC = 1.0e-8


@dataclass(frozen=True)
class WaveformConfig:
    sample_rate_hz: float
    start_sec: float
    duration_sec: float
    min_valid_fraction: float
    max_interpolation_gap_sec: float
    baseline_method: str
    pre_event_start_sec: float
    pre_event_end_sec: float
    baseline_fallback: str
    baseline_fallback_max_sec: float
    baseline_min_samples: int
    filter_type: str
    cutoff_hz: float
    num_taps: int
    filter_window: str


def waveform_config_from_v2(config: dict[str, Any]) -> WaveformConfig:
    dataset = config["dataset"]
    waveform = dataset["waveform"]
    baseline = dataset["baseline"]
    filter_config = dataset["filter"]
    return WaveformConfig(
        sample_rate_hz=float(dataset["sample_rate_hz"]),
        start_sec=float(waveform["start_sec"]),
        duration_sec=float(waveform["duration_sec"]),
        min_valid_fraction=float(waveform["min_valid_fraction"]),
        max_interpolation_gap_sec=float(
            waveform["max_interpolation_gap_sec"]
        ),
        baseline_method=str(baseline["method"]),
        pre_event_start_sec=float(baseline["pre_event_start_sec"]),
        pre_event_end_sec=float(baseline["pre_event_end_sec"]),
        baseline_fallback=str(baseline["fallback"]),
        baseline_fallback_max_sec=float(baseline["fallback_max_sec"]),
        baseline_min_samples=int(baseline["min_samples"]),
        filter_type=str(filter_config["type"]),
        cutoff_hz=float(filter_config["cutoff_hz"]),
        num_taps=int(filter_config["num_taps"]),
        filter_window=str(filter_config["window"]),
    )


@dataclass(frozen=True)
class ProcessedWaveform:
    time_sec: np.ndarray
    values_m: np.ndarray
    valid_mask: np.ndarray
    dt_sec: float
    raw_dt_sec: float
    baseline_m: float
    baseline_source: str
    valid_fraction: float


def _convert_to_metres(values: np.ndarray, units: str) -> np.ndarray:
    key = units.lower()
    if key == "m":
        factor = 1.0
    elif key == "cm":
        factor = 1.0e-2
    elif key == "mm":
        factor = 1.0e-3
    else:
        raise ValueError(f"unsupported waveform units: {units}")
    return values.astype(np.float64, copy=False) * factor


def _sort_and_average_duplicates(
    time_sec: np.ndarray,
    values_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    time_array = np.asarray(time_sec, dtype=np.float64)
    value_array = np.asarray(values_m, dtype=np.float64)
    finite = np.isfinite(time_array) & np.isfinite(value_array)
    clean_time = time_array[finite]
    clean_values = value_array[finite]
    if clean_time.size < 2:
        raise ValueError("waveform has fewer than two finite samples")

    order = np.argsort(clean_time, kind="mergesort")
    sorted_time = clean_time[order]
    sorted_values = clean_values[order]
    unique_time, inverse = np.unique(sorted_time, return_inverse=True)
    sums = np.bincount(inverse, weights=sorted_values)
    counts = np.bincount(inverse)
    if unique_time.size < 2:
        raise ValueError("waveform has fewer than two unique timestamps")
    return unique_time, sums / counts


def _estimate_baseline(
    time_sec: np.ndarray,
    values_m: np.ndarray,
    *,
    p_arrival_sec: float,
    config: WaveformConfig,
) -> tuple[float, str]:
    pre_event = (
        (time_sec >= config.pre_event_start_sec)
        & (time_sec < config.pre_event_end_sec)
    )
    if int(pre_event.sum()) >= config.baseline_min_samples:
        selected = values_m[pre_event]
        source = "pre_event"
    elif config.baseline_fallback == "pre_p":
        fallback_end = min(
            float(p_arrival_sec), config.baseline_fallback_max_sec
        )
        pre_p = (time_sec >= 0.0) & (time_sec < fallback_end)
        if int(pre_p.sum()) < config.baseline_min_samples:
            raise ValueError(
                "insufficient baseline: pre-event and pre-P samples are insufficient"
            )
        selected = values_m[pre_p]
        source = "pre_p"
    else:
        raise ValueError(
            f"unsupported baseline fallback: {config.baseline_fallback}"
        )

    if config.baseline_method == "median":
        baseline = float(np.median(selected))
    elif config.baseline_method == "mean":
        baseline = float(np.mean(selected))
    else:
        raise ValueError(
            f"unsupported baseline method: {config.baseline_method}"
        )
    if not math.isfinite(baseline):
        raise ValueError("baseline must be finite")
    return baseline, source


def _fir_lowpass(values: np.ndarray, config: WaveformConfig) -> np.ndarray:
    if config.filter_type == "none":
        return values
    if config.filter_type != "lowpass":
        raise ValueError(
            f"unsupported filter type: {config.filter_type}; expected none or lowpass"
        )

    taps = int(config.num_taps)
    if taps < 3 or taps % 2 == 0:
        raise ValueError("num_taps must be an odd integer of at least 3")
    nyquist_hz = 0.5 * config.sample_rate_hz
    if not 0.0 < config.cutoff_hz < nyquist_hz:
        raise ValueError("cutoff_hz must be strictly between zero and Nyquist")

    n = np.arange(taps, dtype=np.float64)
    midpoint = 0.5 * (taps - 1)
    normalized_cutoff = config.cutoff_hz / config.sample_rate_hz
    kernel = 2.0 * normalized_cutoff * np.sinc(
        2.0 * normalized_cutoff * (n - midpoint)
    )
    if config.filter_window == "hamming":
        kernel *= np.hamming(taps)
    elif config.filter_window in {"hann", "hanning"}:
        kernel *= np.hanning(taps)
    else:
        raise ValueError(f"unsupported FIR window: {config.filter_window}")
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def preprocess_waveform(
    time_sec: np.ndarray,
    values: np.ndarray,
    *,
    units: str,
    p_arrival_sec: float,
    config: WaveformConfig,
) -> ProcessedWaveform:
    if config.sample_rate_hz <= 0.0 or config.duration_sec <= 0.0:
        raise ValueError("sample_rate_hz and duration_sec must be positive")

    clean_time, clean_values = _sort_and_average_duplicates(
        np.asarray(time_sec), np.asarray(values)
    )
    values_m = _convert_to_metres(clean_values, units)
    positive_diffs = np.diff(clean_time)
    positive_diffs = positive_diffs[positive_diffs > 0.0]
    raw_dt_sec = float(np.median(positive_diffs))

    baseline_m, baseline_source = _estimate_baseline(
        clean_time,
        values_m,
        p_arrival_sec=p_arrival_sec,
        config=config,
    )
    centered_values = values_m - baseline_m

    dt_sec = 1.0 / config.sample_rate_hz
    sample_count = int(round(config.duration_sec * config.sample_rate_hz))
    grid = (
        config.start_sec
        + np.arange(sample_count, dtype=np.float64) * dt_sec
    )

    supported = (grid >= clean_time[0]) & (grid <= clean_time[-1])
    left = np.searchsorted(clean_time, grid, side="right") - 1
    left_in_bounds = (left >= 0) & (left < clean_time.size)
    left_safe = np.clip(left, 0, clean_time.size - 1)
    exact_left = supported & left_in_bounds & np.isclose(
        grid,
        clean_time[left_safe],
        rtol=0.0,
        atol=EXACT_TIME_TOLERANCE_SEC,
    )
    right = left + 1
    right_in_bounds = (right >= 0) & (right < clean_time.size)
    right_safe = np.clip(right, 0, clean_time.size - 1)
    exact_right = supported & right_in_bounds & np.isclose(
        grid,
        clean_time[right_safe],
        rtol=0.0,
        atol=EXACT_TIME_TOLERANCE_SEC,
    )
    exact = exact_left | exact_right
    interior = supported & left_in_bounds & right_in_bounds
    gap_ok = np.zeros_like(supported, dtype=bool)
    gap_ok[exact] = True
    interior_nonexact = interior & ~exact
    gap_ok[interior_nonexact] = (
        clean_time[right[interior_nonexact]]
        - clean_time[left[interior_nonexact]]
        <= config.max_interpolation_gap_sec
    )
    valid_mask = supported & gap_ok

    if config.max_interpolation_gap_sec == 0.0:
        resampled = np.zeros_like(grid)
        resampled[exact_left] = centered_values[left_safe[exact_left]]
        exact_right_only = exact_right & ~exact_left
        resampled[exact_right_only] = centered_values[
            right_safe[exact_right_only]
        ]
    else:
        resampled = np.interp(grid, clean_time, centered_values)

    network_values = np.where(valid_mask, resampled, 0.0)
    filtered = _fir_lowpass(network_values, config)
    filtered = np.where(valid_mask, filtered, 0.0)
    valid_fraction = float(np.mean(valid_mask))
    if valid_fraction < config.min_valid_fraction:
        raise ValueError(
            f"valid fraction {valid_fraction:.4f} is below "
            f"{config.min_valid_fraction:.4f}"
        )

    return ProcessedWaveform(
        time_sec=grid.astype(np.float32),
        values_m=filtered.astype(np.float32),
        valid_mask=valid_mask.astype(bool),
        dt_sec=float(dt_sec),
        raw_dt_sec=raw_dt_sec,
        baseline_m=baseline_m,
        baseline_source=baseline_source,
        valid_fraction=valid_fraction,
    )
