"""
PyGMT per-event station map.

Each subplot shows:
  - ETOPO terrain/bathymetry background
  - GNSS stations as triangles coloured by PINN error (mw_pred - mw_true)
  - Focal mechanism beachball (skipped when strike/dip/rake is nan)
  - North arrow, scale bar, inset overview map
  - Shared RdBu diverging colorbar

Usage:
    python scripts/plotting/plot_station_map.py \
        --station-csv outputs/results/unseen_events_8_1cm/station_predictions.csv \
        --event-csv   outputs/results/unseen_events_8_1cm/event_summary.csv \
        --output      tests/figure/station_map.png
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pygmt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATION_CSV = PROJECT_ROOT / "outputs" / "results" / "unseen_events_8_1cm" / "station_predictions.csv"
DEFAULT_EVENT_CSV   = PROJECT_ROOT / "outputs" / "results" / "unseen_events_8_1cm" / "event_summary.csv"
DEFAULT_OUTPUT      = PROJECT_ROOT / "tests" / "figure" / "station_maps"
RELIEF_GRID         = Path.home() / ".gmt/server/earth/earth_relief/earth_relief_05m_g.grd"

# ── layout ───────────────────────────────────────────────────────────────────
FIGURE_WIDTH_CM = 7.0
FIGURE_HEIGHT_CM = 6.0

# ── style ────────────────────────────────────────────────────────────────────
CMAP          = "polar"          # RdBu-style diverging built into GMT
TRIANGLE_SIZE = "0.25c"
BEACHBALL_SIZE = "0.5c"
FONT_LABEL    = "8p,Helvetica-Bold,black"
FONT_ANNOT    = "7p,Helvetica,black"

# ── padding around stations for map region [deg] ─────────────────────────────
PAD = 2.0


def load_stations(csv_path: Path) -> dict[str, list[dict]]:
    """Returns {event_label: [row, ...]}."""
    data: dict[str, list[dict]] = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ev = row["event"]
            data.setdefault(ev, []).append(row)
    return data


def load_events(csv_path: Path) -> dict[str, dict]:
    """Returns {event_label: row}."""
    data: dict[str, dict] = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            data[row["event"]] = row
        return data


def _region(lats: list[float], lons: list[float]) -> list[float]:
    return [
        min(lons) - PAD, max(lons) + PAD,
        min(lats) - PAD, max(lats) + PAD,
    ]


def _is_valid(*vals: str | float) -> bool:
    for v in vals:
        try:
            if math.isnan(float(v)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _inset_region(main_region: list[float]) -> list[float]:
    lon_min, lon_max, lat_min, lat_max = main_region
    lon_center = (lon_min + lon_max) / 2
    lat_center = (lat_min + lat_max) / 2
    lon_span = max(12.0, (lon_max - lon_min) * 3.0)
    lat_span = max(8.0, (lat_max - lat_min) * 3.0)
    lon_min = max(-180.0, lon_center - lon_span / 2)
    lon_max = min(180.0, lon_center + lon_span / 2)
    lat_min = max(-90.0, lat_center - lat_span / 2)
    lat_max = min(90.0, lat_center + lat_span / 2)
    return [lon_min, lon_max, lat_min, lat_max]


def _slugify_event_name(event_name: str) -> str:
    return (
        event_name.lower()
        .replace(".", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def build_event_plot_plan(*, stations_by_event: dict[str, list[dict]], events: dict[str, dict], output_dir: Path) -> list[dict]:
    all_errors = [float(r["error"]) for rows in stations_by_event.values() for r in rows]
    clim = max(abs(np.nanmin(all_errors)), abs(np.nanmax(all_errors)))
    clim = round(math.ceil(clim * 10) / 10, 1)

    plans: list[dict] = []
    for ev_label in sorted(stations_by_event.keys()):
        plans.append(
            {
                "event": ev_label,
                "rows": stations_by_event[ev_label],
                "event_meta": events.get(ev_label, {}),
                "output_path": output_dir / f"{_slugify_event_name(ev_label)}.png",
                "figure_width_cm": FIGURE_WIDTH_CM,
                "figure_height_cm": FIGURE_HEIGHT_CM,
                "clim_min": -clim,
                "clim_max": clim,
                "dpi": 300,
                "crop": True,
                "target_width_px": 900,
                "target_height_px": 780,
                "triangle_size_cm": 0.22,
                "beachball_size_cm": 0.5,
                "north_arrow_cm": 0.78,
                "inset_width_cm": 1.55,
                "title_font": "10p,Helvetica-Bold,black",
                "relief_grid": RELIEF_GRID,
                "relief_cmap": "grayC",
                "land_fill": "255/255/255",
                "water_fill": "194/226/247",
                "relief_shading": "+a45+nt0.08",
                "shoreline_pen": "0.28p,gray65",
                "border_pens": ["1/0.1p,gray85", "2/0.08p,gray92"],
                "inset_land_fill": "247/243/223",
                "inset_water_fill": "178/216/233",
                "inset_box_pen": "0.55p,red",
                "inset_frame_pen": "0.25p,gray55",
                "frame_style": ["WSen", "xa1f1", "ya1f1"],
                "scale_position": "jBC",
                "scale_offset": "0c/0.14c",
                "colorbar_size": "1.7c/0.12c",
                "colorbar_offset": "0.18c/0.24c",
                "colorbar_frame": ["xa0.2", "y+lM@-w@-"],
            }
        )
    return plans


def _pad_to_canvas(image_path: Path, *, target_width_px: int, target_height_px: int) -> None:
    image = Image.open(image_path).convert("RGBA")
    if image.width > target_width_px or image.height > target_height_px:
        image.thumbnail((target_width_px, target_height_px), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (target_width_px, target_height_px), (255, 255, 255, 255))
    x0 = (target_width_px - image.width) // 2
    y0 = (target_height_px - image.height) // 2
    canvas.paste(image, (x0, y0))
    canvas.convert("RGB").save(image_path)


def _event_short_label(event_name: str, mw_true: str | float) -> str:
    parts = str(event_name).split()
    place = parts[0] if parts else str(event_name)
    return f"Mw {mw_true} {place}"


def _plot_single_event(fig: pygmt.Figure, *, rows: list[dict], ev: dict, clim: float, style: dict) -> None:
    lats = [float(r["station_lat"]) for r in rows]
    lons = [float(r["station_lon"]) for r in rows]
    errs = [float(r["error"]) for r in rows]
    ev_lat = float(ev.get("event_lat", np.mean(lats)))
    ev_lon = float(ev.get("event_lon", np.mean(lons)))

    region = _region(lats + [ev_lat], lons + [ev_lon])
    projection = f"M{FIGURE_WIDTH_CM}c"

    fig.grdimage(
        grid=str(style["relief_grid"]),
        region=region,
        projection=projection,
        cmap=style["relief_cmap"],
        shading=style["relief_shading"],
        frame=style["frame_style"],
    )
    fig.coast(
        region=region,
        projection=projection,
        land=style["land_fill"],
        water=style["water_fill"],
        shorelines=style["shoreline_pen"],
        borders=style["border_pens"],
        resolution="h",
    )

    pygmt.makecpt(cmap=CMAP, series=[-clim, clim, 0.1], background=True)
    fig.plot(
        x=lons,
        y=lats,
        style=f"t{style['triangle_size_cm']}c",
        fill=errs,
        cmap=True,
        pen="0.25p,gray40",
    )

    strike = ev.get("strike", "nan")
    dip = ev.get("dip", "nan")
    rake = ev.get("rake", "nan")
    if _is_valid(strike, dip, rake):
        fig.meca(
            spec={
                "strike": float(strike),
                "dip": float(dip),
                "rake": float(rake),
                "magnitude": float(ev.get("mw_true", 7.0)),
            },
            scale=f"{style['beachball_size_cm']}c",
            longitude=ev_lon,
            latitude=ev_lat,
            depth=10,
            convention="aki",
            compression_fill="black",
            extension_fill="white",
            pen="0.5p,black",
        )
    else:
        fig.plot(x=[ev_lon], y=[ev_lat], style="a0.22c", fill="black", pen="0.4p,black")

    fig.text(
        x=ev_lon + (region[1] - region[0]) * 0.035,
        y=ev_lat - (region[3] - region[2]) * 0.065,
        text=_event_short_label(ev.get("event", ""), ev.get("mw_true", "")),
        font=style["title_font"],
        justify="LM",
    )

    map_width_deg = region[1] - region[0]
    scale_km = int(round(map_width_deg * 111 * 0.25 / 50) * 50)
    scale_km = max(50, scale_km)
    fig.basemap(region=region, projection=projection, map_scale=f"{style['scale_position']}+w{scale_km}k+o{style['scale_offset']}+f+l")
    fig.basemap(region=region, projection=projection, rose=f"jTL+w{style['north_arrow_cm']}c+o0.22c/0.18c")
    with fig.inset(position=f"jBL+w{style['inset_width_cm']}c+o0.08c/0.08c", box=f"+gwhite+p{style['inset_frame_pen']}"):
        inset_region = _inset_region(region)
        fig.coast(
            region=inset_region,
            projection=f"M{style['inset_width_cm']}c",
            land=style["inset_land_fill"],
            water=style["inset_water_fill"],
            shorelines="1/0.14p,gray62",
            borders="1/0.14p,gray80",
            frame=False,
        )
        fig.plot(
            x=[region[0], region[1], region[1], region[0], region[0]],
            y=[region[2], region[2], region[3], region[3], region[2]],
            pen=style["inset_box_pen"],
        )
    colorbar_frame = style["colorbar_frame"].copy()
    colorbar_frame[0] = f"xa{clim/2:.1f}"
    fig.colorbar(cmap=True, position=f"jBR+w{style['colorbar_size']}+h+o{style['colorbar_offset']}", frame=colorbar_frame)


def plot_station_map(
    *,
    station_csv: Path,
    event_csv: Path,
    output: Path,
) -> Path:
    stations_by_event = load_stations(station_csv)
    events = load_events(event_csv)
    output_dir = output if output.suffix == "" else output.parent
    plans = build_event_plot_plan(stations_by_event=stations_by_event, events=events, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for plan in plans:
        fig = pygmt.Figure()
        _plot_single_event(fig, rows=plan["rows"], ev=plan["event_meta"], clim=plan["clim_max"], style=plan)
        fig.savefig(str(plan["output_path"]), dpi=plan["dpi"], crop=plan["crop"])
        _pad_to_canvas(
            plan["output_path"],
            target_width_px=plan["target_width_px"],
            target_height_px=plan["target_height_px"],
        )
        print(f"已保存: {plan['output_path']}")

    return output_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PyGMT per-event station map with PINN error coloring")
    p.add_argument("--station-csv", default=str(DEFAULT_STATION_CSV))
    p.add_argument("--event-csv",   default=str(DEFAULT_EVENT_CSV))
    p.add_argument("--output",      default=str(DEFAULT_OUTPUT))
    return p


def main() -> None:
    args = build_parser().parse_args()
    plot_station_map(
        station_csv=Path(args.station_csv),
        event_csv=Path(args.event_csv),
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
