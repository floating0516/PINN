from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import math

from src.data.gnss_dataset_loader import load_gnss_catalog


CSV_PATH = Path(r"F:\dataset_catalog\Global_Earthquakes_List.gcmt.csv")
DEFAULT_OUTPUT = Path(r"F:\dataset_catalog\gnss_events_matched.gcmt.npz")


@dataclass
class CsvEventRow:
    event: str
    date_iso: str
    longitude: float
    latitude: float
    depth_km: float
    magnitude: float
    mechanism: str
    strike: float
    dip: float
    rake: float
    stations: int
    country: str


def read_event_csv(path: Path | str = CSV_PATH) -> List[CsvEventRow]:
    """
    读取包含事件信息的 CSV 并返回结构化数据。

    参数
    ----
    path:
        CSV 文件路径。

    返回
    ----
    list[CsvEventRow]
        事件记录列表。
    """
    rows: List[CsvEventRow] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            event_val = _get_row_value(r, "event")
            date_val = _get_row_value(r, "date")
            lon_val = _get_row_value(r, "longitude")
            lat_val = _get_row_value(r, "latitude")
            depth_val = _get_row_value(r, "depth_km")
            mag_val = _get_row_value(r, "magnitude")
            mechanism_val = _get_row_value(r, "mechanism")
            strike_val = _get_row_value(r, "strike")
            dip_val = _get_row_value(r, "dip")
            rake_val = _get_row_value(r, "rake")
            stations_val = _get_row_value(r, "stations")

            event_field = (event_val or "").strip()
            event_main, country = _split_event_and_country(event_field)

            rows.append(
                CsvEventRow(
                    event=event_main,
                    date_iso=(date_val or "").strip(),
                    longitude=_safe_float(lon_val),
                    latitude=_safe_float(lat_val),
                    depth_km=_safe_float(depth_val),
                    magnitude=_safe_float(mag_val),
                    mechanism=(mechanism_val or "").strip(),
                    strike=_safe_float(strike_val),
                    dip=_safe_float(dip_val),
                    rake=_safe_float(rake_val),
                    stations=_safe_int(stations_val),
                    country=country,
                )
            )
    return rows


def _safe_int(val: str | None) -> int:
    try:
        return int(str(val)) if val is not None and str(val) != "" else 0
    except Exception:
        return 0


def _safe_float(val: str | None) -> float:
    try:
        return float(str(val)) if val is not None and str(val) != "" else float("nan")
    except Exception:
        return float("nan")


def _split_event_and_country(event_field: str) -> Tuple[str, str]:
    if "," in event_field:
        left, right = event_field.split(",", 1)
        return left.strip(), right.strip()
    return event_field.strip(), ""


def _normalize_base(name: str) -> str:
    return "".join(ch for ch in name if ch.isalpha() or ch == "." or ch == "_").lower()


def _get_row_value(row: Dict[str, str], target_key_lower: str) -> str | None:
    for k, v in row.items():
        if k is None:
            continue
        if k.strip().lower() == target_key_lower:
            return v
    return None


def _match_key_from_catalog(event_name: str) -> str:
    """
    将目录事件名规范化为 base+year 的匹配键。
    """
    s = event_name.strip()
    base = _normalize_base(s.split("_")[0])
    year = ""
    if "_" in s:
        tail = s.split("_", 1)[1]
        if len(tail) >= 4 and tail[:4].isdigit():
            year = tail[:4]
    else:
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) >= 4:
            year = digits[:4]
    return f"{base}{year}"


def _match_key_from_csv(event_main: str, date_iso: str) -> str:
    """
    将 CSV 的事件名 + 日期规范化为 base+year 的匹配键。
    """
    base_part = "".join(ch for ch in event_main if ch.isalpha() or ch == "." or ch == "_")
    base = _normalize_base(base_part)
    year = date_iso[:4] if len(date_iso) >= 4 and date_iso[:4].isdigit() else ""
    if not year:
        digits = "".join(ch for ch in event_main if ch.isdigit())
        if len(digits) >= 4:
            year = digits[:4]
    return f"{base}{year}"


def build_npz(output_path: Path | str = DEFAULT_OUTPUT, max_station_distance_km: float = 800.0) -> Path:
    """
    读取 GNSS 目录与 CSV 元信息，匹配后生成 NPZ 文件。

    参数
    ----
    output_path:
        输出 NPZ 文件路径。
    max_station_distance_km:
        台站最大震中距阈值（km）。超过该距离的台站会被剔除。

    返回
    ----
    Path
        生成的 NPZ 文件路径。
    """
    catalog = load_gnss_catalog()
    csv_rows = read_event_csv()

    catalog_events = list(catalog.displacements_by_event.keys())
    catalog_key_map: Dict[str, str] = {e: _match_key_from_catalog(e) for e in catalog_events}
    csv_key_map: Dict[str, CsvEventRow] = {}
    for row in csv_rows:
        csv_key = _match_key_from_csv(row.event, row.date_iso)
        csv_key_map[csv_key] = row

    matched_events: List[str] = []
    for event in catalog_events:
        key = catalog_key_map[event]
        if key in csv_key_map:
            matched_events.append(event)

    names: List[str] = []
    countries: List[str] = []
    longitudes: List[float] = []
    latitudes: List[float] = []
    depths_km: List[float] = []
    magnitudes: List[float] = []
    mechanisms: List[str] = []
    strikes: List[float] = []
    dips: List[float] = []
    rakes: List[float] = []
    origin_times: List[str] = []
    station_counts: List[int] = []
    enu_payload: List[Dict[str, Dict[str, np.ndarray]]] = []
    station_info_payload: List[Dict[str, Dict[str, float]]] = []

    for event in matched_events:
        key = catalog_key_map[event]
        meta = csv_key_map[key]
        series_map = catalog.displacements_by_event[event]

        names.append(meta.event if meta.event else event)
        countries.append(meta.country)
        longitudes.append(meta.longitude)
        latitudes.append(meta.latitude)
        depths_km.append(meta.depth_km)
        magnitudes.append(meta.magnitude)
        mechanisms.append(meta.mechanism)
        strikes.append(meta.strike)
        dips.append(meta.dip)
        rakes.append(meta.rake)
        origin_times.append(meta.date_iso)
        station_counts.append(len(series_map))

        enu_event: Dict[str, Dict[str, np.ndarray]] = {}
        station_info_event: Dict[str, Dict[str, float]] = {}
        for station, series in series_map.items():
            t = np.array([s.time_relative_sec for s in series.samples], dtype=np.float64)
            e = np.array([s.east_mm for s in series.samples], dtype=np.float64)
            n = np.array([s.north_mm for s in series.samples], dtype=np.float64)
            u = np.array([s.up_mm for s in series.samples], dtype=np.float64)
            enu_event[station] = {"t": t, "E": e, "N": n, "U": u}
            lat_val = getattr(series, "latitude_deg", None)
            lon_val = getattr(series, "longitude_deg", None)
            lat = float(lat_val) if lat_val is not None else float("nan")
            lon = float(lon_val) if lon_val is not None else float("nan")
            station_info_event[station] = {"lat": lat, "lon": lon}
        if not event.lower().startswith("noto"):
            valid_stations = [s for s, info in station_info_event.items() if not math.isnan(info["lat"]) and not math.isnan(info["lon"])]
            enu_event = {s: enu_event[s] for s in valid_stations}
            station_info_event = {s: station_info_event[s] for s in valid_stations}
            station_counts[-1] = len(enu_event)
        if math.isfinite(meta.latitude) and math.isfinite(meta.longitude) and max_station_distance_km > 0.0:
            enu_event, station_info_event = _filter_station_maps_by_distance_km(
                event_lat_deg=float(meta.latitude),
                event_lon_deg=float(meta.longitude),
                enu_event=enu_event,
                station_info_event=station_info_event,
                max_station_distance_km=max_station_distance_km,
            )
            station_counts[-1] = len(enu_event)
        enu_payload.append(enu_event)
        station_info_payload.append(station_info_event)

    np.savez_compressed(
        Path(output_path),
        events=np.array(names, dtype=object),
        country=np.array(countries, dtype=object),
        longitude=np.array(longitudes, dtype=np.float64),
        latitude=np.array(latitudes, dtype=np.float64),
        depth_km=np.array(depths_km, dtype=np.float64),
        magnitude=np.array(magnitudes, dtype=np.float64),
        mechanism=np.array(mechanisms, dtype=object),
        strike=np.array(strikes, dtype=np.float64),
        dip=np.array(dips, dtype=np.float64),
        rake=np.array(rakes, dtype=np.float64),
        origin_time=np.array(origin_times, dtype=object),
        station_count=np.array(station_counts, dtype=np.int32),
        enu=np.array(enu_payload, dtype=object),
        station_info=np.array(station_info_payload, dtype=object),
    )

    return Path(output_path)


def _filter_station_maps_by_distance_km(
    event_lat_deg: float,
    event_lon_deg: float,
    enu_event: Dict[str, Dict[str, np.ndarray]],
    station_info_event: Dict[str, Dict[str, float]],
    max_station_distance_km: float,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, float]]]:
    kept: List[str] = []
    for station, info in station_info_event.items():
        lat = float(info.get("lat", float("nan")))
        lon = float(info.get("lon", float("nan")))
        if math.isnan(lat) or math.isnan(lon):
            continue
        distance_km = _haversine_km(event_lat_deg, event_lon_deg, lat, lon)
        if math.isfinite(distance_km) and distance_km <= max_station_distance_km:
            kept.append(station)
    return {s: enu_event[s] for s in kept if s in enu_event}, {s: station_info_event[s] for s in kept}


def plot_gnss_event_examples(
    npz_path: Path | str = DEFAULT_OUTPUT,
    output_dir: Path | str = Path(r"F:\dataset_catalog\gnss_event_examples"),
    time_min_sec: float = 0.0,
    time_max_sec: float = 600.0,
    km_per_cm: float | None = None,
    scale_cm: float | None = None,
    min_km_per_cm: float = 2.0,
    max_km_per_cm: float = 80.0,
    scale_cm_candidates: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0),
    file_suffix: str = "png",
) -> None:
    """
    为 NPZ 中的每个地震事件绘制 GNSS 位移 record-section 例图（E/N/U 三分图）。

    参数
    ----
    npz_path:
        `build_npz` 生成的 NPZ 路径。
    output_dir:
        输出图片目录。
    time_min_sec:
        横坐标时间下限（相对发震时刻，秒）。
    time_max_sec:
        横坐标时间上限（相对发震时刻，秒）。
    km_per_cm:
        将位移振幅转换为纵向偏移的比例：1 cm 位移对应 km_per_cm km 的纵向偏移。
        传入 None 时对每个事件自动选择，使图更美观。
    scale_cm:
        图中标注的振幅标度（cm），用于绘制真实比例尺。
        传入 None 时根据当前事件的 km_per_cm 自动选择。
    min_km_per_cm:
        自动选择 km_per_cm 的最小值（越小波形越“高”）。
    max_km_per_cm:
        自动选择 km_per_cm 的最大值（越大波形越“矮”）。
    scale_cm_candidates:
        自动选择比例尺时可用的 cm 候选集合。
    file_suffix:
        输出文件后缀，如 "png" 或 "pdf"。
    """
    import matplotlib.pyplot as plt

    npz = np.load(Path(npz_path), allow_pickle=True)
    events = npz["events"]
    event_lon = npz["longitude"]
    event_lat = npz["latitude"]
    enu = npz["enu"]
    station_info = npz["station_info"]
    magnitude = npz["magnitude"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx in range(len(events)):
        event_name = str(events[idx])
        lon0 = float(event_lon[idx])
        lat0 = float(event_lat[idx])
        enu_event = enu[idx]
        station_info_event = station_info[idx]
        mag0 = float(magnitude[idx])

        fig = _plot_one_event_record_section(
            event_name=event_name,
            event_lat_deg=lat0,
            event_lon_deg=lon0,
            event_magnitude=mag0,
            enu_event=enu_event,
            station_info_event=station_info_event,
            time_min_sec=time_min_sec,
            time_max_sec=time_max_sec,
            km_per_cm=km_per_cm,
            scale_cm=scale_cm,
            min_km_per_cm=min_km_per_cm,
            max_km_per_cm=max_km_per_cm,
            scale_cm_candidates=scale_cm_candidates,
        )
        out_file = output_path / f"{event_name}.gnss_record_section.{file_suffix}"
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _plot_one_event_record_section(
    event_name: str,
    event_lat_deg: float,
    event_lon_deg: float,
    event_magnitude: float,
    enu_event: Dict[str, Dict[str, np.ndarray]],
    station_info_event: Dict[str, Dict[str, float]],
    time_min_sec: float,
    time_max_sec: float,
    km_per_cm: float | None,
    scale_cm: float | None,
    min_km_per_cm: float,
    max_km_per_cm: float,
    scale_cm_candidates: Tuple[float, ...],
):
    import matplotlib.pyplot as plt

    station_distances_km: Dict[str, float] = {}
    for station, info in station_info_event.items():
        lat = float(info.get("lat", float("nan")))
        lon = float(info.get("lon", float("nan")))
        if math.isnan(lat) or math.isnan(lon):
            continue
        station_distances_km[station] = _haversine_km(event_lat_deg, event_lon_deg, lat, lon)

    stations_sorted = sorted(station_distances_km.keys(), key=lambda s: station_distances_km[s])
    km_per_cm_resolved = km_per_cm
    if km_per_cm_resolved is None:
        km_per_cm_resolved = _recommend_km_per_cm(
            stations_sorted=stations_sorted,
            station_distances_km=station_distances_km,
            enu_event=enu_event,
            time_min_sec=time_min_sec,
            time_max_sec=time_max_sec,
            min_km_per_cm=min_km_per_cm,
            max_km_per_cm=max_km_per_cm,
        )
    scale_cm_resolved = scale_cm
    if scale_cm_resolved is None:
        target_bar_km = _recommended_scale_bar_km(stations_sorted, station_distances_km)
        scale_cm_resolved = _choose_scale_cm(target_bar_km, km_per_cm_resolved, scale_cm_candidates)

    km_per_mm = km_per_cm_resolved / 10.0

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12.0, 4.0), sharex=True, sharey=True)
    components = [("E", "E-component"), ("N", "N-component"), ("U", "U-component")]

    all_distances = [station_distances_km[s] for s in stations_sorted]
    if all_distances:
        y_min = min(all_distances)
        y_max = max(all_distances)
    else:
        y_min = 0.0
        y_max = 1.0
    y_pad = max(10.0, 0.05 * (y_max - y_min)) if y_max > y_min else 10.0

    for ax, (comp_key, title) in zip(axes, components, strict=False):
        for station in stations_sorted:
            if station not in enu_event:
                continue
            t = enu_event[station]["t"]
            x = enu_event[station][comp_key]
            distance_km = station_distances_km[station]

            mask = (t >= time_min_sec) & (t <= time_max_sec)
            if not np.any(mask):
                continue
            t_plot = t[mask]
            x_plot = x[mask]
            y_plot = distance_km + x_plot * km_per_mm
            ax.plot(t_plot, y_plot, color="black", linewidth=0.6)

        ax.set_title(title)
        ax.set_xlim(time_min_sec, time_max_sec)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel("Time relative to origin (s)")

    axes[0].set_ylabel("Epicentral distance (km)")
    fig.suptitle(f"{event_name} (M {event_magnitude:.1f})" if math.isfinite(event_magnitude) else event_name)

    _add_amplitude_scale_bar(
        ax=axes[0],
        time_min_sec=time_min_sec,
        time_max_sec=time_max_sec,
        distance_min_km=y_min - y_pad,
        distance_max_km=y_max + y_pad,
        scale_cm=scale_cm_resolved,
        km_per_cm=km_per_cm_resolved,
    )

    fig.tight_layout()
    return fig


def _add_amplitude_scale_bar(
    ax,
    time_min_sec: float,
    time_max_sec: float,
    distance_min_km: float,
    distance_max_km: float,
    scale_cm: float,
    km_per_cm: float,
) -> None:
    x0 = time_min_sec + 0.03 * (time_max_sec - time_min_sec)
    y0 = distance_min_km + 0.08 * (distance_max_km - distance_min_km)
    bar_len_km = scale_cm * km_per_cm
    ax.plot([x0, x0], [y0, y0 + bar_len_km], color="red", linewidth=2.0)
    ax.text(x0, y0 - 0.02 * (distance_max_km - distance_min_km), f"{scale_cm:g} cm", color="red", ha="left", va="top")


def _haversine_km(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg)
    lon2 = math.radians(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.asin(math.sqrt(a))
    return 6371.0 * c


def _recommended_scale_bar_km(stations_sorted: List[str], station_distances_km: Dict[str, float]) -> float:
    distances = np.array([station_distances_km[s] for s in stations_sorted], dtype=np.float64)
    if distances.size < 2:
        return 20.0
    diffs = np.diff(np.sort(distances))
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 20.0
    median_spacing = float(np.median(diffs))
    return max(5.0, 0.8 * median_spacing)


def _recommend_km_per_cm(
    stations_sorted: List[str],
    station_distances_km: Dict[str, float],
    enu_event: Dict[str, Dict[str, np.ndarray]],
    time_min_sec: float,
    time_max_sec: float,
    min_km_per_cm: float,
    max_km_per_cm: float,
) -> float:
    distances = np.array([station_distances_km[s] for s in stations_sorted], dtype=np.float64)
    if distances.size >= 2:
        diffs = np.diff(np.sort(distances))
        diffs = diffs[np.isfinite(diffs)]
        median_spacing = float(np.median(diffs)) if diffs.size else 50.0
    else:
        median_spacing = 50.0

    target_wiggle_km = max(5.0, 0.6 * median_spacing)

    amp_samples_mm: List[float] = []
    for station in stations_sorted:
        if station not in enu_event:
            continue
        t = enu_event[station].get("t")
        if t is None:
            continue
        mask = (t >= time_min_sec) & (t <= time_max_sec)
        if not np.any(mask):
            continue
        for comp_key in ("E", "N", "U"):
            x = enu_event[station].get(comp_key)
            if x is None:
                continue
            xw = x[mask]
            if xw.size == 0:
                continue
            amp = float(np.nanpercentile(np.abs(xw), 95))
            if math.isfinite(amp) and amp > 0.0:
                amp_samples_mm.append(amp)

    amp_mm = float(np.nanpercentile(np.array(amp_samples_mm, dtype=np.float64), 90)) if amp_samples_mm else 1.0
    amp_mm = amp_mm if amp_mm > 0.0 and math.isfinite(amp_mm) else 1.0

    km_per_mm = target_wiggle_km / amp_mm
    km_per_cm = km_per_mm * 10.0
    return float(min(max(km_per_cm, min_km_per_cm), max_km_per_cm))


def _choose_scale_cm(target_bar_km: float, km_per_cm: float, candidates: Tuple[float, ...]) -> float:
    if not candidates:
        return 3.0
    desired = target_bar_km / km_per_cm if km_per_cm > 0 else candidates[0]
    best = candidates[0]
    best_err = abs(best - desired)
    for c in candidates[1:]:
        err = abs(c - desired)
        if err < best_err:
            best = c
            best_err = err
    return float(best)


if __name__ == "__main__":
    build_npz()
