from __future__ import annotations

from src.data.records_v2 import NormalizedStationRecord, mechanism_to_code


def record_from_external_bundle(bundle, station) -> NormalizedStationRecord:
    return NormalizedStationRecord(
        event_index=-1,
        event=str(bundle.event_name),
        magnitude_catalog=float(bundle.magnitude),
        event_lat=float(bundle.latitude),
        event_lon=float(bundle.longitude),
        depth_km=float(bundle.depth_km),
        strike=float(bundle.strike),
        dip=float(bundle.dip),
        rake=float(bundle.rake),
        mechanism=mechanism_to_code(bundle.mechanism),
        station=str(station.station),
        station_lat=float(station.latitude),
        station_lon=float(station.longitude),
        time_sec=station.t.copy(),
        east=station.e_m.copy(),
        north=station.n_m.copy(),
        vertical=station.u_m.copy(),
        origin_sec=0.0,
    )
