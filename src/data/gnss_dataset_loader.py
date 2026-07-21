from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DATASET_OTHER_ROOT = Path(r"F:\dataset_catalog\dataset_other_etal_2025")
DATASET_GLEHMAN_ROOT = Path(r"F:\dataset_catalog\dataset_glehman_etal_2025")
DATASET_RUHL_ROOT = Path(r"F:\dataset_catalog\dataset_ruhl_etal_2018")


@dataclass
class GnssDisplacementSample:
    time_relative_sec: float
    east_mm: float
    north_mm: float
    up_mm: float


@dataclass
class GnssDisplacementSeries:
    dataset: str
    event: str
    station: str
    samples: List[GnssDisplacementSample]
    latitude_deg: float | None = None
    longitude_deg: float | None = None


@dataclass
class GnssPositionSample:
    timestamp: datetime
    x_m: float
    y_m: float
    z_m: float


@dataclass
class GnssPositionSeries:
    dataset: str
    event: str
    track: str
    samples: List[GnssPositionSample]


@dataclass
class GnssCatalog:
    displacements_by_event: Dict[str, Dict[str, GnssDisplacementSeries]]
    positions_by_event: Dict[str, Dict[str, GnssPositionSeries]]


def load_glehman_displacements(root: Path | str = DATASET_GLEHMAN_ROOT) -> Dict[str, Dict[str, GnssDisplacementSeries]]:
    """
    读取 dataset_glehman_etal_2025 中的 GNSS 位移 txt 数据。

    参数
    ----
    root:
        数据集根目录，默认使用 F 盘 dataset_glehman_etal_2025 位置。

    返回
    ----
    dict
        event_name -> station -> GnssDisplacementSeries 映射。
    """
    root_path = Path(root)
    result: Dict[str, Dict[str, GnssDisplacementSeries]] = {}

    for event_dir in root_path.iterdir():
        if not event_dir.is_dir():
            continue
        txt_dir = event_dir / "GNSS" / "txt"
        if not txt_dir.is_dir():
            continue

        station_map: Dict[str, GnssDisplacementSeries] = {}
        meta_map = _load_glehman_station_metadata(event_dir)
        for file_path in sorted(txt_dir.glob("*.gnss.txt")):
            station = file_path.stem.split(".")[0]
            lat_lon = meta_map.get(station)
            lat_val = lat_lon[0] if lat_lon else None
            lon_val = lat_lon[1] if lat_lon else None
            series = _parse_displacement_file(
                file_path=file_path,
                dataset_name="dataset_glehman_etal_2025",
                event_name=event_dir.name,
                station_name=station,
                latitude_deg=lat_val,
                longitude_deg=lon_val,
            )
            station_map[station] = series

        if station_map:
            result[event_dir.name] = station_map

    return result


def load_ruhl_displacements(root: Path | str = DATASET_RUHL_ROOT) -> Dict[str, Dict[str, GnssDisplacementSeries]]:
    """
    读取 dataset_ruhl_etal_2018 中的 GNSS 位移 txt 数据。

    参数
    ----
    root:
        数据集根目录，默认使用 F 盘 dataset_ruhl_etal_2018 位置。

    返回
    ----
    dict
        event_name -> station -> GnssDisplacementSeries 映射。
    """
    root_path = Path(root)
    result: Dict[str, Dict[str, GnssDisplacementSeries]] = {}

    for event_dir in root_path.iterdir():
        if not event_dir.is_dir():
            continue
        txt_dir = event_dir / "disp" / "txt"
        if not txt_dir.is_dir():
            continue

        station_map: Dict[str, GnssDisplacementSeries] = {}
        meta_map = _load_ruhl_station_metadata(event_dir)
        for file_path in sorted(txt_dir.glob("*.gnss.txt")):
            station = file_path.stem.split(".")[0]
            lat_lon = meta_map.get(station)
            lat_val = lat_lon[0] if lat_lon else None
            lon_val = lat_lon[1] if lat_lon else None
            series = _parse_displacement_file(
                file_path=file_path,
                dataset_name="dataset_ruhl_etal_2018",
                event_name=event_dir.name,
                station_name=station,
                latitude_deg=lat_val,
                longitude_deg=lon_val,
            )
            station_map[station] = series

        if station_map:
            result[event_dir.name] = station_map

    return result


def load_other_displacements(root: Path | str = DATASET_OTHER_ROOT) -> Dict[str, Dict[str, GnssDisplacementSeries]]:
    """
    读取 dataset_other_etal_2025 中的 GNSS 位移 txt 数据。

    本函数假定该数据集中的位移 txt 文件存放在
    主目录/地震事件名/disp/txt 目录下，文件名形如 *.gnss.txt。

    参数
    ----
    root:
        数据集根目录，默认使用 F 盘 dataset_other_etal_2025 位置。

    返回
    ----
    dict
        event_name -> station -> GnssDisplacementSeries 映射。
    """
    root_path = Path(root)
    result: Dict[str, Dict[str, GnssDisplacementSeries]] = {}

    for event_dir in root_path.iterdir():
        if not event_dir.is_dir():
            continue
        txt_dir = event_dir / "disp" / "txt"
        if not txt_dir.is_dir():
            continue

        station_map: Dict[str, GnssDisplacementSeries] = {}
        meta_map = _load_other_station_metadata(event_dir)
        for file_path in sorted(txt_dir.glob("*.gnss.txt")):
            station = file_path.stem.split(".")[0]
            lat_lon = meta_map.get(station)
            lat_val = lat_lon[0] if lat_lon else None
            lon_val = lat_lon[1] if lat_lon else None
            series = _parse_displacement_file(
                file_path=file_path,
                dataset_name="dataset_other_etal_2025",
                event_name=event_dir.name,
                station_name=station,
                latitude_deg=lat_val,
                longitude_deg=lon_val,
            )
            station_map[station] = series

        if station_map:
            result[event_dir.name] = station_map

    return result


def load_other_noto_positions(root: Path | str = DATASET_OTHER_ROOT) -> Dict[str, GnssPositionSeries]:
    """
    读取 dataset_other_etal_2025 中 Noto2024 事件的 GNSS 轨迹 txt 数据。

    该目录中 kin_2024001_xxxx 文件为无表头的时间 + 三维坐标序列。

    参数
    ----
    root:
        数据集根目录，默认使用 F 盘 dataset_other_etal_2025 位置。

    返回
    ----
    dict
        track_name -> GnssPositionSeries 映射。
    """
    root_path = Path(root)
    noto_dir = root_path / "Noto2024"
    result: Dict[str, GnssPositionSeries] = {}

    if not noto_dir.is_dir():
        return result

    for file_path in sorted(noto_dir.iterdir()):
        if not file_path.is_file():
            continue

        track_name = file_path.stem
        series = _parse_noto_position_file(
            file_path=file_path,
            dataset_name="dataset_other_etal_2025",
            event_name="Noto2024",
            track_name=track_name,
        )
        result[track_name] = series

    return result


def load_displacements_by_event(
    other_root: Path | str = DATASET_OTHER_ROOT,
    glehman_root: Path | str = DATASET_GLEHMAN_ROOT,
    ruhl_root: Path | str = DATASET_RUHL_ROOT,
) -> Dict[str, Dict[str, GnssDisplacementSeries]]:
    """
    合并三个地震数据集中的位移型 GNSS 数据（other、glehman 与 ruhl）。

    参数
    ----
    other_root:
        dataset_other_etal_2025 根目录。
    glehman_root:
        dataset_glehman_etal_2025 根目录。
    ruhl_root:
        dataset_ruhl_etal_2018 根目录。

    返回
    ----
    dict
        event_name -> station -> GnssDisplacementSeries 映射。
        若不同数据集中存在同名事件，后加载的数据会覆盖之前的同名键。
    """
    displacements: Dict[str, Dict[str, GnssDisplacementSeries]] = {}

    other_data = load_other_displacements(root=other_root)
    for event_name, station_map in other_data.items():
        displacements[event_name] = dict(station_map)

    glehman_data = load_glehman_displacements(root=glehman_root)
    for event_name, station_map in glehman_data.items():
        displacements[event_name] = dict(station_map)

    ruhl_data = load_ruhl_displacements(root=ruhl_root)
    for event_name, station_map in ruhl_data.items():
        displacements[event_name] = dict(station_map)

    return displacements


def load_noto_positions_by_event(root: Path | str = DATASET_OTHER_ROOT) -> Dict[str, Dict[str, GnssPositionSeries]]:
    """
    按事件组织 Noto2024 的三维坐标型 GNSS 数据。

    参数
    ----
    root:
        dataset_other_etal_2025 根目录。

    返回
    ----
    dict
        event_name -> track_name -> GnssPositionSeries 映射。
        当前仅包含 Noto2024 事件。
    """
    tracks = load_other_noto_positions(root=root)
    return {"Noto2024": tracks} if tracks else {}


def load_gnss_catalog(
    other_root: Path | str = DATASET_OTHER_ROOT,
    glehman_root: Path | str = DATASET_GLEHMAN_ROOT,
    ruhl_root: Path | str = DATASET_RUHL_ROOT,
) -> GnssCatalog:
    """
    一次性读取三个地震目录中的所有 GNSS txt 数据并组织为目录对象。

    参数
    ----
    other_root:
        dataset_other_etal_2025 根目录。
    glehman_root:
        dataset_glehman_etal_2025 根目录。
    ruhl_root:
        dataset_ruhl_etal_2018 根目录。

    返回
    ----
    GnssCatalog
        包含按事件组织的位移型 GNSS 数据和三维坐标型 GNSS 数据。
    """
    displacements = load_displacements_by_event(
        other_root=other_root,
        glehman_root=glehman_root,
        ruhl_root=ruhl_root,
    )
    positions = load_noto_positions_by_event(root=other_root)
    return GnssCatalog(displacements_by_event=displacements, positions_by_event=positions)


def _parse_displacement_file(
    file_path: Path,
    dataset_name: str,
    event_name: str,
    station_name: str,
    latitude_deg: float | None = None,
    longitude_deg: float | None = None,
) -> GnssDisplacementSeries:
    samples: List[GnssDisplacementSample] = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue

            east_mm = float(parts[0])
            north_mm = float(parts[1])
            up_mm = float(parts[2])
            time_relative_sec = float(parts[3])

            samples.append(
                GnssDisplacementSample(
                    time_relative_sec=time_relative_sec,
                    east_mm=east_mm,
                    north_mm=north_mm,
                    up_mm=up_mm,
                )
            )

    return GnssDisplacementSeries(
        dataset=dataset_name,
        event=event_name,
        station=station_name,
        samples=samples,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
    )

def _load_ruhl_station_metadata(event_dir: Path) -> Dict[str, Tuple[float, float]]:
    """
    解析 dataset_ruhl_etal_2018 事件目录中的台站经纬度元数据。
    返回值为 station -> (lat, lon) 的映射；若不存在元数据文件则返回空字典。
    """
    candidates = list(event_dir.glob("*_disp.chan"))
    if not candidates:
        return {}
    chan_path = candidates[0]
    result: Dict[str, Tuple[float, float]] = {}
    with chan_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            station_code = parts[1]
            try:
                lat = float(parts[4])
                lon = float(parts[5])
            except ValueError:
                continue
            result[station_code] = (lat, lon)
    return result

def _load_glehman_station_metadata(event_dir: Path) -> Dict[str, Tuple[float, float]]:
    """
    解析 dataset_glehman_etal_2025 事件目录中的 GNSS 元数据 CSV。
    返回值为 station -> (lat, lon) 的映射；若不存在元数据文件则返回空字典。
    """
    meta_dir = event_dir / "metadata"
    meta_file = meta_dir / f"{event_dir.name}.gnss.meta.csv"
    if not meta_file.is_file():
        return {}
    result: Dict[str, Tuple[float, float]] = {}
    with meta_file.open("r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 4:
                continue
            station = parts[1]
            try:
                lat = round(float(parts[2]), 4)
                lon = round(float(parts[3]), 4)
            except ValueError:
                continue
            result[station] = (lat, lon)
    return result

def _load_other_station_metadata(event_dir: Path) -> Dict[str, Tuple[float, float]]:
    if event_dir.name != "Noto2024":
        return {}
    meta_file = event_dir / "metadata" / "Noto2024.gnss.meta.csv"
    if not meta_file.is_file():
        return {}
    result: Dict[str, Tuple[float, float]] = {}
    with meta_file.open("r", encoding="utf-8-sig") as f:
        first = True
        for line in f:
            s = line.strip()
            if not s:
                continue
            if first:
                first = False
                if s.lower().startswith("gnss"):
                    continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 4:
                continue
            station = parts[1]
            try:
                lat = float(parts[2])
                lon = float(parts[3])
            except ValueError:
                continue
            result[station] = (lat, lon)
    return result

def _parse_noto_position_file(
    file_path: Path,
    dataset_name: str,
    event_name: str,
    track_name: str,
) -> GnssPositionSeries:
    samples: List[GnssPositionSample] = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue

            timestamp_str = f"{parts[0]} {parts[1]}"
            timestamp = datetime.strptime(timestamp_str, "%Y/%m/%d %H:%M:%S")
            x_m = float(parts[2])
            y_m = float(parts[3])
            z_m = float(parts[4])

            samples.append(
                GnssPositionSample(
                    timestamp=timestamp,
                    x_m=x_m,
                    y_m=y_m,
                    z_m=z_m,
                )
            )

    return GnssPositionSeries(
        dataset=dataset_name,
        event=event_name,
        track=track_name,
        samples=samples,
    )


def _find_npz_event_index(events: object, event_name: str) -> int:
    try:
        n = len(events)  # type: ignore[arg-type]
    except Exception:
        return -1
    target = str(event_name)
    for i in range(n):
        if str(events[i]) == target:
            return i
    return -1


def _extract_station_lon_lat(station_info_event: object) -> Tuple[List[float], List[float]]:
    import math

    lons: List[float] = []
    lats: List[float] = []
    if not isinstance(station_info_event, dict):
        return lons, lats
    for _, info in station_info_event.items():
        if not isinstance(info, dict):
            continue
        lat = float(info.get("lat", float("nan")))
        lon = float(info.get("lon", float("nan")))
        if not math.isfinite(lat) or not math.isfinite(lon):
            continue
        lats.append(lat)
        lons.append(lon)
    return lons, lats


def _compute_region_wesn(
    event_lon: float,
    event_lat: float,
    station_lons: List[float],
    station_lats: List[float],
    padding_deg: float,
) -> Tuple[float, float, float, float]:
    import math

    def _norm_lon(x: float) -> float:
        return ((x + 180.0) % 360.0) - 180.0
    cen = _norm_lon(event_lon)
    def _unwrap_near(x: float) -> float:
        v = _norm_lon(x)
        d = v - cen
        if d > 180.0:
            v -= 360.0
        elif d < -180.0:
            v += 360.0
        return v

    lons_src = [event_lon, *station_lons]
    lats_src = [event_lat, *station_lats]
    lons = [_unwrap_near(x) for x in lons_src if math.isfinite(x)]
    lats = [y for y in lats_src if math.isfinite(y)]
    if not lons or not lats:
        pad = max(0.1, float(padding_deg))
        return cen - pad, cen + pad, event_lat - pad, event_lat + pad
    west = min(lons)
    east = max(lons)
    south = min(lats)
    north = max(lats)
    pad = max(0.1, float(padding_deg))
    if west == east:
        west -= pad
        east += pad
    if south == north:
        south -= pad
        north += pad
    w = west - pad
    e = east + pad
    if e - w >= 360.0:
        mid = (west + east) / 2.0
        half = 359.0 / 2.0
        w = mid - half
        e = mid + half
    return w, e, south - pad, north + pad


def _plot_gnss_event_map_core(
    *,
    event_name: str,
    output_path: Path | str,
    event_lon: float,
    event_lat: float,
    depth_km: float,
    magnitude: float,
    station_lons: Sequence[float],
    station_lats: Sequence[float],
    station_names: Sequence[str] | None = None,
    strike: float = float("nan"),
    dip: float = float("nan"),
    rake: float = float("nan"),
    mechanism_label: str | None = None,
    region: Tuple[float, float, float, float] | None = None,
    region_padding_deg: float = 0.6,
    projection_width_cm: float = 14.0,
    relief_resolution: str | None = "15s",
    relief_grid_path: Path | str | None = None,
    relief_cmap: str = "geo",
    relief_colorbar: bool = False,
    relief_colorbar_position: str = "jBR+w5c/0.25c+o0.35c/0.35c",
    relief_transparency: float | None = 30.0,
    station_label: str = "GNSS stations",
    station_symbol: str = "t0.30c",
    station_pen: str = "0.8p,gray30",
    station_fill: str = "white",
    station_name_font: str = "7p,Helvetica,black",
    station_name_dx_deg: float | None = None,
    station_name_dy_deg: float | None = None,
    label_stations: bool = True,
    event_symbol: str = "a0.38c",
    event_fill: str = "red",
    event_pen: str = "0.8p,black",
    beachball_scale: str = "0.70c",
    beachball_pen: str = "0.6p,black",
    beachball_compression_fill: str = "black",
    beachball_extension_fill: str = "white",
    map_scale_km: float = 30.0,
    include_north_arrow: bool = True,
    include_legend: bool = True,
    title_text: str | None = None,
    title_font: str = "18p,Helvetica-Bold,black",
    title_box: str | None = "+gwhite@60+p0.8p,gray30+r0.2c",
    title_auto_justify: bool = True,
    title_corner: str | None = None,
    inset_enabled: bool = False,
    inset_position: str = "jBR+w2.5c/1.8c+o0.35c/0.35c",
    inset_box: str = "+gwhite+p0.8p,gray30",
    inset_region: Tuple[float, float, float, float] | None = None,
    inset_projection: str = "M3c",
    inset_expand_deg: float = 10,
    inset_show_main_extent: bool = True,
    inset_frame_pen: str = "1.0p,red",
    savefig_dpi: int = 300,
    savefig_crop: bool = True,
) -> Path:
    import math

    try:
        import pygmt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("缺少依赖 pygmt：请先安装 PyGMT 与 GMT。") from e

    station_lons_clean = [float(x) for x in station_lons if math.isfinite(float(x))]
    station_lats_clean = [float(y) for y in station_lats if math.isfinite(float(y))]
    station_names_clean = list(station_names or [])
    wesn = region or _compute_region_wesn(
        event_lon=event_lon,
        event_lat=event_lat,
        station_lons=station_lons_clean,
        station_lats=station_lats_clean,
        padding_deg=region_padding_deg,
    )
    west, east, south, north = wesn
    width = max(east - west, 0.1)
    height = max(north - south, 0.1)
    dx_default = 0.018 * width
    dy_default = 0.014 * height
    dx = station_name_dx_deg if station_name_dx_deg is not None else dx_default
    dy = station_name_dy_deg if station_name_dy_deg is not None else dy_default
    center_x = (west + east) / 2.0
    center_y = (south + north) / 2.0

    projection = f"M{float(projection_width_cm):.2f}c"
    fig = pygmt.Figure()
    grid = None
    if relief_grid_path is not None:
        p = Path(relief_grid_path)
        if p.is_file():
            grid = str(p)
    elif relief_resolution is not None and str(relief_resolution).strip() != "":
        try:
            grid = pygmt.datasets.load_earth_relief(resolution=str(relief_resolution), region=wesn)
        except Exception:
            grid = None

    if grid is not None:
        try:
            fig.grdimage(
                grid=grid,
                region=wesn,
                projection=projection,
                cmap=relief_cmap,
                shading=True,
                transparency=relief_transparency,
                frame=["WSne", "a"],
            )
        except Exception:
            fig.grdimage(
                grid=grid,
                region=wesn,
                projection=projection,
                cmap=relief_cmap,
                shading=True,
                frame=["WSne", "a"],
            )
        fig.coast(shorelines="0.8p,gray20", borders="1/0.5p,gray30", lakes="white")
        if relief_colorbar:
            try:
                fig.colorbar(
                    cmap=relief_cmap,
                    position=relief_colorbar_position,
                    frame=["af", "x+lElevation", "y+lm"],
                    box="+gwhite+p0.8p,black",
                )
            except Exception:
                fig.colorbar(cmap=relief_cmap, position=relief_colorbar_position, frame="af", box="+gwhite+p0.8p,black")
    else:
        fig.coast(
            region=wesn,
            projection=projection,
            land="gray95",
            water="lightsteelblue1",
            shorelines="0.8p,gray20",
            borders="1/0.5p,gray30",
            lakes="white",
            frame=["WSne", "a"],
        )

    has_focal = math.isfinite(strike) and math.isfinite(dip) and math.isfinite(rake) and math.isfinite(magnitude)
    if has_focal:
        focal_mechanism = {"strike": strike, "dip": dip, "rake": rake, "magnitude": magnitude}
        try:
            fig.meca(
                spec=focal_mechanism,
                scale=beachball_scale,
                longitude=event_lon,
                latitude=event_lat,
                depth=depth_km if math.isfinite(depth_km) else 0.0,
                compression_fill=beachball_compression_fill,
                extension_fill=beachball_extension_fill,
                pen=beachball_pen,
            )
        except Exception:
            fig.meca(
                spec=focal_mechanism,
                scale=beachball_scale,
                longitude=event_lon,
                latitude=event_lat,
                depth=depth_km if math.isfinite(depth_km) else 0.0,
                compressionfill=beachball_compression_fill,
                extensionfill=beachball_extension_fill,
                pen=beachball_pen,
            )
    else:
        fig.plot(x=[event_lon], y=[event_lat], style=event_symbol, pen=event_pen, fill=event_fill, label="Epicenter")

    if station_lons_clean and station_lats_clean:
        fig.plot(
            x=station_lons_clean,
            y=station_lats_clean,
            style=station_symbol,
            pen=station_pen,
            fill=station_fill,
            label=station_label,
        )
        if label_stations and station_names_clean and len(station_names_clean) == len(station_lons_clean):
            for name, lon, lat in zip(station_names_clean, station_lons_clean, station_lats_clean):
                lon_text = float(lon) + (dx if float(lon) <= center_x else -dx)
                lat_text = float(lat) + (dy if float(lat) <= center_y else -dy)
                justify = "LM" if float(lon) <= center_x else "RM"
                try:
                    fig.text(
                        x=lon_text,
                        y=lat_text,
                        text=str(name),
                        font=station_name_font,
                        justify=justify,
                        pen="0.5p,white",
                        fill="white@35",
                    )
                except Exception:
                    fig.text(x=lon_text, y=lat_text, text=str(name), font=station_name_font, justify=justify)

    if include_north_arrow:
        fig.basemap(rose="jTL+w1.1c+o0.25c/0.25c")
    if map_scale_km > 0:
        fig.basemap(map_scale=f"jBC+w{float(map_scale_km):.0f}k+o0c/0.55c+f")

    label_mag = f"Mw {magnitude:.1f}" if math.isfinite(magnitude) else "Mw ?"
    mechanism_suffix = ""
    if mechanism_label:
        mechanism_suffix = f" | {mechanism_label}"
        if not has_focal:
            mechanism_suffix += " | beachball unavailable"
    label_text = title_text if title_text is not None else f"{event_name}  {label_mag}{mechanism_suffix}"
    fig.basemap(frame=[f'+t{label_text}'])

    if include_legend:
        fig.legend(position="jTR+o0.2c", box="+gwhite+p0.8p,black")

    if inset_enabled:
        inset_reg = inset_region or (wesn[0] - inset_expand_deg, wesn[1] + inset_expand_deg, wesn[2] - inset_expand_deg, wesn[3] + inset_expand_deg)
        with fig.inset(position=inset_position, box="+gwhite+p0.8p,black"):
            fig.coast(region=inset_reg, projection=inset_projection, land="gray90", water="lightsteelblue1", shorelines=True, frame="af")
            if inset_show_main_extent:
                xs = [west, east, east, west, west]
                ys = [south, south, north, north, south]
                try:
                    fig.plot(x=xs, y=ys, pen=inset_frame_pen)
                except Exception:
                    fig.plot(x=xs, y=ys, pen="1p,red")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=int(savefig_dpi), crop=bool(savefig_crop))
    return out


def plot_unseen_event_map_gmt(
    *,
    event_name: str,
    output_path: Path | str,
    event_lon: float,
    event_lat: float,
    depth_km: float,
    magnitude: float,
    station_lons: Sequence[float],
    station_lats: Sequence[float],
    station_names: Sequence[str] | None = None,
    strike: float = float("nan"),
    dip: float = float("nan"),
    rake: float = float("nan"),
    mechanism_label: str | None = None,
    **kwargs,
) -> Path:
    return _plot_gnss_event_map_core(
        event_name=event_name,
        output_path=output_path,
        event_lon=event_lon,
        event_lat=event_lat,
        depth_km=depth_km,
        magnitude=magnitude,
        station_lons=station_lons,
        station_lats=station_lats,
        station_names=station_names,
        strike=strike,
        dip=dip,
        rake=rake,
        mechanism_label=mechanism_label,
        **kwargs,
    )


def plot_gnss_event_map_gmt(
    npz_path: Path | str,
    event_name: str,
    output_path: Path | str,
    region: Tuple[float, float, float, float] | None = None,
    region_padding_deg: float = 0.6,
    projection_width_cm: float = 14.0,
    relief_resolution: str | None = "15s",
    relief_grid_path: Path | str | None = None,
    relief_cmap: str = "geo",
    relief_colorbar: bool = False,
    relief_colorbar_position: str = "jBR+w5c/0.25c+o0.35c/0.35c",
    relief_transparency: float | None = 30.0,
    station_label: str = "GNSS stations",
    station_symbol: str = "t0.30c",
    station_pen: str = "0.8p,gray30",
    station_fill: str = "white",
    beachball_scale: str = "0.70c",
    beachball_pen: str = "0.6p,black",
    beachball_compression_fill: str = "black",
    beachball_extension_fill: str = "white",
    map_scale_km: float = 30.0,
    include_north_arrow: bool = True,
    include_legend: bool = True,
    title_text: str | None = None,
    title_font: str = "18p,Helvetica-Bold,black",
    title_box: str | None = "+gwhite@60+p0.8p,gray30+r0.2c",
    title_auto_justify: bool = True,
    title_corner: str | None = None,
    inset_enabled: bool = False,
    inset_position: str = "jBR+w2.5c/1.8c+o0.35c/0.35c",
    inset_box: str = "+gwhite+p0.8p,gray30",
    inset_region: Tuple[float, float, float, float] | None = None,
    inset_projection: str = "M3c",
    inset_expand_deg: float = 10,
    inset_show_main_extent: bool = True,
    inset_frame_pen: str = "1.0p,red",
    savefig_dpi: int = 300,
    savefig_crop: bool = True,
) -> Path:
    """
    使用 PyGMT 绘制单个 GNSS 事件的台站分布图，并在震中绘制震源球。
    """
    import numpy as np

    npz = np.load(Path(npz_path), allow_pickle=True)
    events = npz["events"]
    idx = _find_npz_event_index(events, event_name)
    if idx < 0:
        raise ValueError(f"NPZ 中找不到事件：{event_name}")

    lon0 = float(npz["longitude"][idx])
    lat0 = float(npz["latitude"][idx])
    depth_km = float(npz["depth_km"][idx])
    magnitude = float(npz["magnitude"][idx])
    station_info_event = npz["station_info"][idx]
    strike = float(npz["strike"][idx]) if "strike" in npz else float("nan")
    dip = float(npz["dip"][idx]) if "dip" in npz else float("nan")
    rake = float(npz["rake"][idx]) if "rake" in npz else float("nan")
    station_lons, station_lats = _extract_station_lon_lat(station_info_event)

    return _plot_gnss_event_map_core(
        event_name=event_name,
        output_path=output_path,
        event_lon=lon0,
        event_lat=lat0,
        depth_km=depth_km,
        magnitude=magnitude,
        station_lons=station_lons,
        station_lats=station_lats,
        station_names=None,
        strike=strike,
        dip=dip,
        rake=rake,
        mechanism_label=None,
        region=region,
        region_padding_deg=region_padding_deg,
        projection_width_cm=projection_width_cm,
        relief_resolution=relief_resolution,
        relief_grid_path=relief_grid_path,
        relief_cmap=relief_cmap,
        relief_colorbar=relief_colorbar,
        relief_colorbar_position=relief_colorbar_position,
        relief_transparency=relief_transparency,
        station_label=station_label,
        station_symbol=station_symbol,
        station_pen=station_pen,
        station_fill=station_fill,
        label_stations=False,
        beachball_scale=beachball_scale,
        beachball_pen=beachball_pen,
        beachball_compression_fill=beachball_compression_fill,
        beachball_extension_fill=beachball_extension_fill,
        map_scale_km=map_scale_km,
        include_north_arrow=include_north_arrow,
        include_legend=include_legend,
        title_text=title_text,
        title_font=title_font,
        title_box=title_box,
        title_auto_justify=title_auto_justify,
        title_corner=title_corner,
        inset_enabled=inset_enabled,
        inset_position=inset_position,
        inset_box=inset_box,
        inset_region=inset_region,
        inset_projection=inset_projection,
        inset_expand_deg=inset_expand_deg,
        inset_show_main_extent=inset_show_main_extent,
        inset_frame_pen=inset_frame_pen,
        savefig_dpi=savefig_dpi,
        savefig_crop=savefig_crop,
    )

