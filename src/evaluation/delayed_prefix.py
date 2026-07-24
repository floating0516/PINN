"""Strict delayed-prefix release for manuscript-aligned online evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from src.data.waveform import (
    ProcessedWaveform,
    WaveformConfig,
    preprocess_waveform,
)


MANUSCRIPT_PROCESSING_DELAY_SEC = 5.0


@dataclass(frozen=True)
class DelayedPrefixRelease:
    """A filtered waveform prefix that is safe to publish after a fixed delay."""

    masked_waveform_m: np.ndarray
    released_valid_mask: np.ndarray
    prefix_steps: int
    prefix_sec: float
    issue_time_sec: float
    processing_delay_sec: float
    fir_lookahead_samples: int


def fir_lookahead_samples(config: WaveformConfig) -> int:
    """Return the future support of the manuscript's centered 7-tap FIR."""

    if config.filter_type != "lowpass":
        raise ValueError("delayed-prefix release requires the lowpass FIR")
    if config.num_taps != 7:
        raise ValueError("delayed-prefix release requires exactly 7 FIR taps")
    return (config.num_taps - 1) // 2


def _validate_online_contract(
    config: WaveformConfig,
    *,
    processing_delay_sec: float,
) -> int:
    if not math.isclose(
        config.max_interpolation_gap_sec,
        0.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "delayed-prefix release requires max_interpolation_gap_sec=0"
        )
    if not math.isfinite(config.sample_rate_hz) or config.sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite")
    if not math.isclose(config.start_sec, 0.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("manuscript delayed-prefix release requires start_sec=0")
    if not math.isfinite(processing_delay_sec) or processing_delay_sec < 0.0:
        raise ValueError("processing_delay_sec must be nonnegative and finite")

    lookahead_samples = fir_lookahead_samples(config)
    delay_samples = processing_delay_sec * config.sample_rate_hz
    if delay_samples + 1.0e-12 < lookahead_samples:
        raise ValueError(
            "processing delay is shorter than the centered FIR lookahead"
        )
    if not math.isclose(
        processing_delay_sec,
        MANUSCRIPT_PROCESSING_DELAY_SEC,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("manuscript delayed-prefix release requires a 5 s delay")
    return lookahead_samples


def release_delayed_prefix(
    processed: ProcessedWaveform,
    *,
    prefix_steps: int,
    config: WaveformConfig,
    processing_delay_sec: float = MANUSCRIPT_PROCESSING_DELAY_SEC,
) -> DelayedPrefixRelease:
    """Release samples ``[0, prefix_steps)`` and mask the remaining window."""

    lookahead_samples = _validate_online_contract(
        config,
        processing_delay_sec=processing_delay_sec,
    )
    if isinstance(prefix_steps, bool) or not isinstance(prefix_steps, int):
        raise ValueError("prefix_steps must be an integer")

    waveform = np.asarray(processed.values_m)
    valid_mask = np.asarray(processed.valid_mask, dtype=bool)
    if waveform.ndim != 1 or valid_mask.shape != waveform.shape:
        raise ValueError("processed waveform and valid mask must be one-dimensional")
    if prefix_steps < 1 or prefix_steps > waveform.size:
        raise ValueError("prefix_steps is outside the processed waveform")

    released = np.arange(waveform.size) < prefix_steps
    released_valid_mask = valid_mask & released
    masked_waveform = np.where(released_valid_mask, waveform, 0.0)
    prefix_sec = prefix_steps / config.sample_rate_hz

    return DelayedPrefixRelease(
        masked_waveform_m=masked_waveform.astype(waveform.dtype, copy=False),
        released_valid_mask=released_valid_mask,
        prefix_steps=prefix_steps,
        prefix_sec=float(prefix_sec),
        issue_time_sec=float(prefix_sec + processing_delay_sec),
        processing_delay_sec=float(processing_delay_sec),
        fir_lookahead_samples=lookahead_samples,
    )


def preprocess_and_release_delayed_prefix(
    time_sec: np.ndarray,
    values: np.ndarray,
    *,
    units: str,
    p_arrival_sec: float,
    prefix_steps: int,
    config: WaveformConfig,
    processing_delay_sec: float = MANUSCRIPT_PROCESSING_DELAY_SEC,
) -> DelayedPrefixRelease:
    """Apply the unchanged offline preprocessor, then release a strict prefix."""

    _validate_online_contract(
        config,
        processing_delay_sec=processing_delay_sec,
    )
    if isinstance(prefix_steps, bool) or not isinstance(prefix_steps, int):
        raise ValueError("prefix_steps must be an integer")
    sample_count = int(round(config.duration_sec * config.sample_rate_hz))
    if prefix_steps < 1 or prefix_steps > sample_count:
        raise ValueError("prefix_steps is outside the processed waveform")

    issue_time_sec = prefix_steps / config.sample_rate_hz + processing_delay_sec
    raw_time = np.asarray(time_sec)
    raw_values = np.asarray(values)
    if raw_time.shape != raw_values.shape:
        raise ValueError("raw waveform times and values must have matching shapes")
    available = np.isfinite(raw_time) & (raw_time <= issue_time_sec)
    if int(np.count_nonzero(available)) < 2:
        raise ValueError("fewer than two raw samples are available at issue time")

    online_config = replace(config, min_valid_fraction=0.0)
    processed = preprocess_waveform(
        raw_time[available],
        raw_values[available],
        units=units,
        p_arrival_sec=p_arrival_sec,
        config=online_config,
    )
    return release_delayed_prefix(
        processed,
        prefix_steps=prefix_steps,
        config=config,
        processing_delay_sec=processing_delay_sec,
    )


__all__ = [
    "DelayedPrefixRelease",
    "MANUSCRIPT_PROCESSING_DELAY_SEC",
    "fir_lookahead_samples",
    "preprocess_and_release_delayed_prefix",
    "release_delayed_prefix",
]
