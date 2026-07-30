from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.records_v2 import (
    NormalizedStationRecord,
    _iter_normalized_station_records,
)


@dataclass(frozen=True)
class RawPGDRecord:
    event: str
    station: str
    time_sec: np.ndarray
    east_m: np.ndarray
    north_m: np.ndarray
    up_m: np.ndarray
    source_distance_km: float
    p_arrival_sec: float
    magnitude_catalog: float


def unit_factor(units: str) -> float:
    key = str(units).strip().lower()
    if key == "m":
        return 1.0
    if key == "cm":
        return 1.0e-2
    if key == "mm":
        return 1.0e-3
    raise ValueError(f"unsupported PGD waveform units: {units}")


def record_arrays(
    record: NormalizedStationRecord,
    *,
    factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_sec = np.asarray(record.time_sec, dtype=np.float64)
    if record.origin_sec is not None:
        time_sec = time_sec - float(record.origin_sec)
    east = np.asarray(record.east, dtype=np.float64) * factor
    north = np.asarray(record.north, dtype=np.float64) * factor
    up = np.asarray(record.vertical, dtype=np.float64) * factor
    arrays = (time_sec, east, north, up)
    if any(array.ndim != 1 for array in arrays) or len(
        {array.size for array in arrays}
    ) != 1:
        raise ValueError(
            f"invalid raw PGD component shape: {record.event}/{record.station}"
        )
    return arrays


def build_raw_pgd_records(
    config: Mapping[str, Any],
    samples_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[tuple[str, str], RawPGDRecord]:
    selected_samples: dict[tuple[str, str], Mapping[str, Any]] = {}
    for samples in samples_by_split.values():
        for sample in samples:
            key = (str(sample["event"]), str(sample["station"]))
            if key in selected_samples:
                raise ValueError(f"duplicate selected PGD station: {key}")
            selected_samples[key] = sample

    factor = unit_factor(str(config["dataset"]["units"]))
    alpha_m_per_s = float(config["physics"]["alpha"])
    records: dict[tuple[str, str], RawPGDRecord] = {}
    with np.load(str(config["paths"]["data_path"]), allow_pickle=True) as data:
        for record in _iter_normalized_station_records(data):
            key = (record.event, record.station)
            sample = selected_samples.get(key)
            if sample is None:
                continue
            if key in records:
                raise ValueError(f"duplicate raw PGD station: {key}")
            time_sec, east, north, up = record_arrays(record, factor=factor)
            source_distance_m = float(sample["source_distance_m"])
            records[key] = RawPGDRecord(
                event=key[0],
                station=key[1],
                time_sec=time_sec,
                east_m=east,
                north_m=north,
                up_m=up,
                source_distance_km=source_distance_m / 1000.0,
                p_arrival_sec=source_distance_m / alpha_m_per_s,
                magnitude_catalog=float(sample["magnitude_catalog"]),
            )
    if set(records) != set(selected_samples):
        missing = sorted(set(selected_samples) - set(records))
        extra = sorted(set(records) - set(selected_samples))
        raise ValueError(
            f"raw PGD cohort changed; missing={missing[:10]}, extra={extra[:10]}"
        )
    return records


def baseline_value(
    values: np.ndarray,
    time_sec: np.ndarray,
    p_arrival_sec: float,
) -> float:
    pre_p = time_sec < float(p_arrival_sec)
    if int(np.count_nonzero(pre_p)) >= 3:
        return float(np.mean(values[pre_p]))
    count = min(len(values), max(5, int(len(values) * 0.05)))
    if count < 1:
        raise ValueError("PGD baseline has no available samples")
    return float(np.mean(values[:count]))


def causal_pgd_3d(
    record: RawPGDRecord,
    *,
    observation_horizon_sec: int,
    processing_delay_sec: float = 6.0,
) -> tuple[float, int, int]:
    if (
        isinstance(observation_horizon_sec, bool)
        or not isinstance(observation_horizon_sec, int)
        or observation_horizon_sec < 1
    ):
        raise ValueError("observation horizon must be a positive integer")
    if not math.isfinite(processing_delay_sec) or processing_delay_sec < 0.0:
        raise ValueError("processing delay must be nonnegative and finite")
    issue_time_sec = float(observation_horizon_sec) + float(
        processing_delay_sec
    )
    finite = (
        np.isfinite(record.time_sec)
        & np.isfinite(record.east_m)
        & np.isfinite(record.north_m)
        & np.isfinite(record.up_m)
    )
    available = finite & (record.time_sec <= issue_time_sec)
    if int(np.count_nonzero(available)) < 3:
        raise ValueError("PGD prefix has fewer than three available samples")
    time_sec = record.time_sec[available]
    east = record.east_m[available]
    north = record.north_m[available]
    up = record.up_m[available]
    east = east - baseline_value(east, time_sec, record.p_arrival_sec)
    north = north - baseline_value(north, time_sec, record.p_arrival_sec)
    up = up - baseline_value(up, time_sec, record.p_arrival_sec)
    observed = (time_sec >= 0.0) & (
        time_sec < float(observation_horizon_sec)
    )
    observed_count = int(np.count_nonzero(observed))
    if observed_count < 1:
        raise ValueError("PGD prefix has no post-origin observed samples")
    pgd_m = float(
        np.max(
            np.sqrt(
                east[observed] ** 2
                + north[observed] ** 2
                + up[observed] ** 2
            )
        )
    )
    if not math.isfinite(pgd_m) or pgd_m <= 0.0:
        raise ValueError("PGD prefix peak must be positive and finite")
    return pgd_m, int(np.count_nonzero(available)), observed_count
