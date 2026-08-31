from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


def _unwrap_object(value: Any) -> Any:
    while (
        isinstance(value, np.ndarray)
        and value.dtype == object
        and value.size == 1
    ):
        value = value.reshape(-1)[0]
    return value


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(_unwrap_object(value))
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _optional_float(value: Any) -> float | None:
    result = _as_float(value)
    return result if np.isfinite(result) else None


def _as_text(value: Any) -> str:
    value = _unwrap_object(value)
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _get_field(payload: Any, keys: tuple[str, ...]) -> Any:
    payload = _unwrap_object(payload)
    if not isinstance(payload, dict):
        return None
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _station_name(station: dict[str, Any], index: int) -> str:
    value = station.get(
        "name",
        station.get("station", station.get("id", f"st_{index}")),
    )
    return _as_text(value)


def _iter_stations_container(
    container: Any,
) -> list[tuple[str, dict[str, Any]]]:
    container = _unwrap_object(container)
    if isinstance(container, dict):
        result = []
        for name, raw_payload in container.items():
            payload = _unwrap_object(raw_payload)
            if isinstance(payload, dict):
                result.append((_as_text(name), payload))
        return result

    if isinstance(container, np.ndarray):
        container = container.tolist()
    if not isinstance(container, (list, tuple)):
        return []

    result = []
    for index, raw_station in enumerate(container):
        station = _unwrap_object(raw_station)
        if isinstance(station, dict):
            result.append((_station_name(station, index), station))
    return result


def _normalize_station_info(info: Any) -> dict[str, dict[str, Any]]:
    info = _unwrap_object(info)
    if isinstance(info, dict):
        result = {}
        for name, raw_payload in info.items():
            payload = _unwrap_object(raw_payload)
            if isinstance(payload, dict):
                result[_as_text(name)] = payload
        return result

    if isinstance(info, np.ndarray):
        info = info.tolist()
    if not isinstance(info, (list, tuple)):
        return {}

    result = {}
    for index, raw_station in enumerate(info):
        station = _unwrap_object(raw_station)
        if isinstance(station, dict):
            result[_station_name(station, index)] = station
    return result


def mechanism_to_code(value: Any) -> int:
    value = _unwrap_object(value)
    if value is None:
        return -1
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        integer = int(value)
        if integer in {0, 1, 2}:
            return integer
        if integer == 3:
            return 2
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", errors="ignore")
    text = (
        str(value)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )
    if "normal" in text or text in {"nf", "正断", "正斷"}:
        return 0
    if "strike" in text or text in {"strikeslip", "ss", "走滑"}:
        return 1
    if (
        "reverse" in text
        or "thrust" in text
        or text in {"rv", "逆冲", "逆衝", "冲断", "衝斷"}
    ):
        return 2
    return -1


def _rake_to_code(value: Any) -> int:
    try:
        rake = float(_unwrap_object(value))
    except (TypeError, ValueError):
        return -1
    if not np.isfinite(rake):
        return -1
    rake = ((rake + 180.0) % 360.0) - 180.0
    if abs(rake) <= 30.0 or abs(rake) >= 150.0:
        return 1
    if 30.0 <= rake <= 150.0:
        return 2
    if -150.0 <= rake <= -30.0:
        return 0
    return -1


def _event_mechanism_code(
    data: np.lib.npyio.NpzFile,
    event_index: int,
) -> int:
    for key in (
        "mechanism",
        "fault_type",
        "fm_type",
        "source_mechanism",
        "focal_mechanism",
        "mech",
    ):
        if key in data:
            return mechanism_to_code(data[key][event_index])
    for key in ("rake", "rake_deg"):
        if key in data:
            return _rake_to_code(data[key][event_index])
    return -1


@dataclass(frozen=True)
class NormalizedStationRecord:
    event_index: int
    event: str
    magnitude_catalog: float
    event_lat: float
    event_lon: float
    depth_km: float
    strike: float
    dip: float
    rake: float
    mechanism: int
    station: str
    station_lat: float
    station_lon: float
    time_sec: np.ndarray
    east: np.ndarray
    north: np.ndarray
    vertical: np.ndarray
    origin_sec: float | None
    waveform_start_sec: float | None = None


def _iter_normalized_station_records(
    data: np.lib.npyio.NpzFile,
) -> Iterator[NormalizedStationRecord]:
    events = data["events"]
    magnitudes = data["magnitude"]
    event_lats = data["latitude"]
    event_lons = data["longitude"]
    event_count = len(events)
    depths = data["depth_km"] if "depth_km" in data else np.full(event_count, np.nan)
    strikes = data["strike"] if "strike" in data else np.full(event_count, np.nan)
    dips = data["dip"] if "dip" in data else np.full(event_count, np.nan)
    rakes = data["rake"] if "rake" in data else np.full(event_count, np.nan)

    if "enu" in data and "station_info" in data:
        event_containers = data["enu"]
        station_metadata = data["station_info"]
    elif "stations" in data:
        event_containers = data["stations"]
        station_metadata = None
    else:
        raise ValueError("NPZ must contain enu/station_info or stations")

    for event_index, event_value in enumerate(events):
        station_items = _iter_stations_container(
            event_containers[event_index]
        )
        metadata_map = (
            _normalize_station_info(station_metadata[event_index])
            if station_metadata is not None
            else {}
        )
        mechanism = _event_mechanism_code(data, event_index)
        for station_name, payload in station_items:
            metadata = metadata_map.get(station_name, {})
            station_lat = payload.get(
                "lat",
                payload.get(
                    "latitude",
                    metadata.get("lat", metadata.get("latitude", np.nan)),
                ),
            )
            station_lon = payload.get(
                "lon",
                payload.get(
                    "longitude",
                    metadata.get("lon", metadata.get("longitude", np.nan)),
                ),
            )
            time_sec = _get_field(payload, ("t", "time"))
            east = _get_field(payload, ("E", "east"))
            north = _get_field(payload, ("N", "north"))
            vertical = _get_field(payload, ("U", "up", "vertical"))
            origin_sec = _get_field(
                payload,
                (
                    "origin",
                    "origin_s",
                    "origin_time",
                    "origin_epoch",
                    "origin_ts",
                    "t0",
                    "origin_sec",
                ),
            )
            waveform_start_sec = _get_field(
                payload,
                ("waveform_start_sec", "sample_phase_start_sec"),
            )
            if any(
                value is None
                for value in (time_sec, east, north, vertical)
            ):
                continue
            yield NormalizedStationRecord(
                event_index=int(event_index),
                event=_as_text(event_value),
                magnitude_catalog=_as_float(magnitudes[event_index]),
                event_lat=_as_float(event_lats[event_index]),
                event_lon=_as_float(event_lons[event_index]),
                depth_km=_as_float(depths[event_index]),
                strike=_as_float(strikes[event_index]),
                dip=_as_float(dips[event_index]),
                rake=_as_float(rakes[event_index]),
                mechanism=mechanism,
                station=station_name,
                station_lat=_as_float(station_lat),
                station_lon=_as_float(station_lon),
                time_sec=np.asarray(time_sec),
                east=np.asarray(east),
                north=np.asarray(north),
                vertical=np.asarray(vertical),
                origin_sec=_optional_float(origin_sec),
                waveform_start_sec=_optional_float(waveform_start_sec),
            )
