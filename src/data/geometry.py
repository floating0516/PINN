from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SourceStationGeometry:
    epicentral_distance_m: float
    source_distance_m: float
    azimuth_deg: float
    back_azimuth_deg: float
    takeoff_angle_deg: float


def compute_source_station_geometry(
    event_lat: float,
    event_lon: float,
    depth_km: float,
    station_lat: float,
    station_lon: float,
) -> SourceStationGeometry:
    if not all(
        math.isfinite(value)
        for value in (event_lat, event_lon, depth_km, station_lat, station_lon)
    ):
        raise ValueError("事件、深度和台站坐标必须为有限值")
    if depth_km < 0.0:
        raise ValueError("depth_km 不得为负")

    earth_radius_m = 6_371_000.0
    phi1 = math.radians(event_lat)
    phi2 = math.radians(station_lat)
    dphi = math.radians(station_lat - event_lat)
    dlambda = math.radians(station_lon - event_lon)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    epicentral_distance_m = earth_radius_m * c

    y = math.sin(dlambda) * math.cos(phi2)
    x = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    )
    azimuth_deg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    reverse_dlambda = math.radians(event_lon - station_lon)
    reverse_y = math.sin(reverse_dlambda) * math.cos(phi1)
    reverse_x = (
        math.cos(phi2) * math.sin(phi1)
        - math.sin(phi2) * math.cos(phi1) * math.cos(reverse_dlambda)
    )
    back_azimuth_deg = (
        math.degrees(math.atan2(reverse_y, reverse_x)) + 360.0
    ) % 360.0

    depth_m = depth_km * 1000.0
    source_distance_m = math.hypot(epicentral_distance_m, depth_m)
    takeoff_angle_deg = math.degrees(
        math.atan2(epicentral_distance_m, max(depth_m, 1.0e-12))
    )

    return SourceStationGeometry(
        epicentral_distance_m=epicentral_distance_m,
        source_distance_m=source_distance_m,
        azimuth_deg=azimuth_deg,
        back_azimuth_deg=back_azimuth_deg,
        takeoff_angle_deg=takeoff_angle_deg,
    )
