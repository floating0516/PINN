from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


class STFWindowTooShort(ValueError):
    """Raised when a source-time window retains too little native moment."""


@dataclass(frozen=True)
class ProcessedSTF:
    time_sec: np.ndarray
    rate_nm_per_s: np.ndarray
    dt_sec: float
    native_moment_nm: float
    grid_moment_before_rescale_nm: float
    retained_moment_fraction: float
    mw_native: float


def moment_to_mw(moment_nm: float) -> float:
    if (
        isinstance(moment_nm, bool)
        or not isinstance(moment_nm, Real)
        or not math.isfinite(float(moment_nm))
        or float(moment_nm) <= 0.0
    ):
        raise ValueError("moment_nm must be positive and finite")
    return (2.0 / 3.0) * (math.log10(float(moment_nm)) - 9.1)


def _finite_real(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _sort_and_average_duplicates(
    time_sec: np.ndarray,
    rate_nm_per_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    time_array = np.asarray(time_sec, dtype=np.float64)
    rate_array = np.asarray(rate_nm_per_s, dtype=np.float64)
    if time_array.ndim != 1 or rate_array.ndim != 1:
        raise ValueError("STF inputs must be one-dimensional")
    if time_array.shape != rate_array.shape:
        raise ValueError("STF time and rate arrays must have the same shape")

    causal_finite = (
        np.isfinite(time_array)
        & np.isfinite(rate_array)
        & (time_array >= 0.0)
    )
    clean_time = time_array[causal_finite]
    clean_rate = np.maximum(rate_array[causal_finite], 0.0)
    if clean_time.size < 2:
        raise ValueError("STF has fewer than two finite causal samples")

    order = np.argsort(clean_time, kind="mergesort")
    sorted_time = clean_time[order]
    sorted_rate = clean_rate[order]
    unique_time, inverse = np.unique(sorted_time, return_inverse=True)
    if unique_time.size < 2:
        raise ValueError("STF has fewer than two unique causal timestamps")
    sums = np.bincount(inverse, weights=sorted_rate)
    counts = np.bincount(inverse)
    return unique_time, sums / counts


def _integrate_interval(
    time_sec: np.ndarray,
    rate_nm_per_s: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> float:
    left = max(start_sec, float(time_sec[0]))
    right = min(end_sec, float(time_sec[-1]))
    if right <= left:
        return 0.0
    interior = (time_sec > left) & (time_sec < right)
    evaluation_time = np.concatenate(
        ([left], time_sec[interior], [right])
    )
    evaluation_rate = np.interp(
        evaluation_time,
        time_sec,
        rate_nm_per_s,
    )
    return float(np.trapezoid(evaluation_rate, evaluation_time))


def resample_source_stf(
    time_sec: np.ndarray,
    rate_nm_per_s: np.ndarray,
    *,
    start_sec: float,
    duration_sec: float,
    sample_rate_hz: float,
    min_retained_moment_fraction: float,
    preserve_integral: bool,
) -> ProcessedSTF:
    start = _finite_real(start_sec, "start_sec")
    duration = _finite_real(duration_sec, "duration_sec")
    sample_rate = _finite_real(sample_rate_hz, "sample_rate_hz")
    minimum_fraction = _finite_real(
        min_retained_moment_fraction,
        "min_retained_moment_fraction",
    )
    if duration <= 0.0:
        raise ValueError("duration_sec must be positive")
    if sample_rate <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError(
            "min_retained_moment_fraction must be in (0, 1]"
        )
    if not isinstance(preserve_integral, bool):
        raise ValueError("preserve_integral must be a boolean")

    clean_time, clean_rate = _sort_and_average_duplicates(
        time_sec,
        rate_nm_per_s,
    )
    native_moment_nm = float(np.trapezoid(clean_rate, clean_time))
    if not math.isfinite(native_moment_nm) or native_moment_nm <= 0.0:
        raise ValueError("native STF moment must be positive and finite")

    end = start + duration
    retained_moment_nm = _integrate_interval(
        clean_time,
        clean_rate,
        start,
        end,
    )
    retained_fraction = retained_moment_nm / native_moment_nm
    if retained_fraction + 1.0e-12 < minimum_fraction:
        raise STFWindowTooShort(
            f"STF window [{start}, {end}) retains "
            f"{retained_fraction:.6f}, below {minimum_fraction:.6f}"
        )

    dt_sec = 1.0 / sample_rate
    sample_count = int(round(duration * sample_rate))
    if sample_count < 1:
        raise ValueError("STF target grid must contain at least one sample")
    target_time = start + np.arange(sample_count, dtype=np.float64) * dt_sec
    target_rate = np.interp(
        target_time,
        clean_time,
        clean_rate,
        left=0.0,
        right=0.0,
    )
    grid_moment_before_rescale_nm = float(np.sum(target_rate) * dt_sec)
    if (
        not math.isfinite(grid_moment_before_rescale_nm)
        or grid_moment_before_rescale_nm <= 0.0
    ):
        raise ValueError("STF discrete target-grid moment must be positive")

    if preserve_integral:
        target_rate = target_rate * (
            native_moment_nm / grid_moment_before_rescale_nm
        )

    return ProcessedSTF(
        time_sec=target_time,
        rate_nm_per_s=target_rate,
        dt_sec=dt_sec,
        native_moment_nm=native_moment_nm,
        grid_moment_before_rescale_nm=grid_moment_before_rescale_nm,
        retained_moment_fraction=float(retained_fraction),
        mw_native=moment_to_mw(native_moment_nm),
    )
