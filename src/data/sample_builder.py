from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import numpy as np

from src.data.geometry import compute_source_station_geometry
from src.data.records_v2 import NormalizedStationRecord
from src.data.waveform import WaveformConfig, preprocess_waveform


class SampleRejected(ValueError):
    def __init__(
        self,
        reason: str,
        detail: str = "",
        *,
        sample: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail
        self.sample = sample


def rotate_horizontal_to_rt(
    east: np.ndarray,
    north: np.ndarray,
    azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    azimuth = math.radians(float(azimuth_deg))
    radial = east * math.sin(azimuth) + north * math.cos(azimuth)
    tangential = east * math.cos(azimuth) - north * math.sin(azimuth)
    return radial, tangential


def _compute_phi_slip_deg(
    azimuth_deg: float,
    strike_deg: float,
    dip_deg: float,
    rake_deg: float,
) -> float:
    if not all(
        math.isfinite(value)
        for value in (strike_deg, dip_deg, rake_deg)
    ):
        return float(azimuth_deg)
    strike = math.radians(strike_deg)
    dip = math.radians(dip_deg)
    rake = math.radians(rake_deg)
    slip_east = (
        math.cos(rake) * math.sin(strike)
        - math.sin(rake) * math.cos(dip) * math.cos(strike)
    )
    slip_north = (
        math.cos(rake) * math.cos(strike)
        + math.sin(rake) * math.cos(dip) * math.sin(strike)
    )
    slip_azimuth_deg = math.degrees(math.atan2(slip_east, slip_north))
    return float(azimuth_deg - slip_azimuth_deg)


def build_station_sample(
    record: NormalizedStationRecord,
    *,
    units: str,
    waveform_config: WaveformConfig,
    alpha_m_per_s: float,
    radial_peak_min_cm: float,
) -> dict[str, Any]:
    if not math.isfinite(record.station_lat) or not math.isfinite(
        record.station_lon
    ):
        raise SampleRejected("missing_station_coordinates")
    if not math.isfinite(record.event_lat) or not math.isfinite(
        record.event_lon
    ):
        raise SampleRejected(
            "invalid_geometry",
            "missing event coordinates",
        )
    if not math.isfinite(record.depth_km) or record.depth_km < 0.0:
        raise SampleRejected(
            "invalid_geometry",
            "missing or negative source depth",
        )
    if not math.isfinite(alpha_m_per_s) or alpha_m_per_s <= 0.0:
        raise ValueError("alpha_m_per_s must be positive and finite")
    if (
        not math.isfinite(radial_peak_min_cm)
        or radial_peak_min_cm < 0.0
    ):
        raise ValueError("radial_peak_min_cm must be nonnegative and finite")

    geometry = compute_source_station_geometry(
        record.event_lat,
        record.event_lon,
        record.depth_km,
        record.station_lat,
        record.station_lon,
    )
    time_rel = np.asarray(record.time_sec, dtype=np.float64)
    if record.origin_sec is not None:
        time_rel = time_rel - record.origin_sec

    effective_waveform_config = waveform_config
    if record.waveform_start_sec is not None:
        phase_start = float(record.waveform_start_sec)
        target_dt = 1.0 / waveform_config.sample_rate_hz
        phase_offset = phase_start - waveform_config.start_sec
        if (
            not math.isfinite(phase_start)
            or phase_offset < -1.0e-8
            or phase_offset >= target_dt
        ):
            raise SampleRejected(
                "invalid_waveform",
                "waveform_start_sec must fall within the first target interval",
            )
        effective_waveform_config = replace(
            waveform_config,
            start_sec=phase_start,
        )

    east = np.asarray(record.east, dtype=np.float64)
    north = np.asarray(record.north, dtype=np.float64)
    vertical_raw = np.asarray(record.vertical, dtype=np.float64)
    arrays = (time_rel, east, north, vertical_raw)
    if any(array.ndim != 1 for array in arrays) or len(
        {array.size for array in arrays}
    ) != 1:
        raise SampleRejected(
            "invalid_waveform",
            "components must be one-dimensional and equal length",
        )

    radial_raw, tangential_raw = rotate_horizontal_to_rt(
        east,
        north,
        geometry.azimuth_deg,
    )
    p_arrival_sec = geometry.source_distance_m / alpha_m_per_s
    try:
        radial = preprocess_waveform(
            time_rel,
            radial_raw,
            units=units,
            p_arrival_sec=p_arrival_sec,
            config=effective_waveform_config,
        )
        vertical = preprocess_waveform(
            time_rel,
            vertical_raw,
            units=units,
            p_arrival_sec=p_arrival_sec,
            config=effective_waveform_config,
        )
    except ValueError as exc:
        detail = str(exc)
        if "baseline" in detail:
            reason = "insufficient_baseline"
        elif "valid fraction" in detail:
            reason = "insufficient_valid_fraction"
        else:
            reason = "invalid_waveform"
        raise SampleRejected(reason, detail) from exc

    radial_peak_cm = float(np.max(np.abs(radial.values_m)) * 100.0)
    sample = {
        "event": record.event,
        "event_index": record.event_index,
        "station": record.station,
        "mechanism": record.mechanism,
        "magnitude_catalog": record.magnitude_catalog,
        "radial": radial.values_m,
        "vertical": vertical.values_m,
        "waveform_valid_mask": radial.valid_mask,
        "waveform_dt_sec": radial.dt_sec,
        "raw_dt_sec": radial.raw_dt_sec,
        "valid_fraction": radial.valid_fraction,
        "baseline_source": radial.baseline_source,
        "waveform_start_sec": effective_waveform_config.start_sec,
        "radial_peak_cm": radial_peak_cm,
        "epicentral_distance_m": geometry.epicentral_distance_m,
        "source_distance_m": geometry.source_distance_m,
        "theta_deg": geometry.takeoff_angle_deg,
        "azimuth_deg": geometry.azimuth_deg,
        "phi_slip_deg": _compute_phi_slip_deg(
            geometry.azimuth_deg,
            record.strike,
            record.dip,
            record.rake,
        ),
    }
    if radial_peak_cm <= radial_peak_min_cm:
        raise SampleRejected(
            "below_radial_peak_threshold",
            f"{radial_peak_cm:.6f} <= {radial_peak_min_cm:.6f} cm",
            sample=sample,
        )
    try:
        tangential = preprocess_waveform(
            time_rel,
            tangential_raw,
            units=units,
            p_arrival_sec=p_arrival_sec,
            config=effective_waveform_config,
        )
    except ValueError as exc:
        raise ValueError(
            "tangential preprocessing failed for "
            f"{record.event}/{record.station}: {exc}"
        ) from exc
    sample["tangential"] = tangential.values_m
    sample["tangential_peak_cm"] = float(
        np.max(np.abs(tangential.values_m)) * 100.0
    )
    return sample
