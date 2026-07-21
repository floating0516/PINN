"""Figure 2: Global event distribution with focal mechanisms (B&W SRL style).

Outputs 4 separate PNG files for manual assembly:
  fig2a_global.png  – Robinson projection, 17.8 cm wide
  fig2b_japan.png   – Japan / NW Pacific,  5.93 cm wide
  fig2c_asia.png    – Himalaya / S. Asia,  5.93 cm wide
  fig2d_alaska.png  – Alaska / Aleutians,  5.93 cm wide

The three regional maps are PIL-padded to identical pixel height.
Offset + leader lines applied to mildly overlapping events in the global map.

Usage:
    python scripts/plotting/plot_global_event_map.py
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pygmt
from PIL import Image, ImageOps

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parents[2]
NPZ_PATH         = PROJECT_ROOT.parent / "dataset" / "gnss_events_matched.gcmt.npz"
UNSEEN_EVENT_CSV = PROJECT_ROOT / "outputs" / "results" / "unseen_events_8_all_stations" / "event_summary.csv"
OUT_DIR          = PROJECT_ROOT / "tests" / "figure"

_BLACKLIST = {"N.Honshu2011", "N.Honshu2012", "N.Honshu2013", "E.Fukushima2011", "Iwate2011"}

# ── figure dimensions ─────────────────────────────────────────────────────────
GLOBAL_W  = 17.8          # cm, global map width
INSET_W   = GLOBAL_W / 3  # cm, each regional panel width ≈ 5.93 cm

# ── regional inset definitions ────────────────────────────────────────────────
_INSET = {
    "b": dict(region=[125.0, 152.0, 28.0, 47.0],  label="(b) Japan"),
    "c": dict(region=[ 75.0, 112.0, 16.0, 38.0],  label="(c) S. Asia"),
    "d": dict(region=[175.0, 222.0, 48.0, 68.0],   label="(d) Alaska"),
}

# ── offset positions for overlapping events in GLOBAL map only ────────────────
_OFF_TRAIN: dict[str, tuple[float, float]] = {
    "Pazarcik2023":            ( 43.0,  31.0),   # Turkey doublet (0.8° apart)
    "Elbistan2023":            ( 33.0,  44.0),
    "RidgecrestForeshock2019": ( 40.5, -115.0),  # California doublet (0.1° apart)
    "Ridgecrest2019":          ( 30.5, -119.5),
    "Nepal2015":               ( 22.0,   77.0),  # Himalaya cluster
    "Iquique2014":             (-14.0,  -77.0),  # Iquique doublet (1° apart)
}
_OFF_UNSEEN: dict[str, tuple[float, float]] = {
    "Nepal 2015 M7.3":  ( 22.0,  90.0),
    "Xizang 2025 M7.1": ( 22.0,  98.0),
    "Iquique 2014 M7.7":(-26.0,  -77.0),
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _sz(mw: float, mn: float, mx: float) -> str:
    frac = max(0.0, min(1.0, (mw - 6.0) / 3.1))
    return f"{mn + frac * (mx - mn):.3f}c"


def _valid(*vals) -> bool:
    for v in vals:
        try:
            if math.isnan(float(v)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _lon360(lon: float) -> float:
    return lon + 360.0 if lon < 0 else lon


def _in_region(ev: dict, reg: list[float]) -> bool:
    lon_min, lon_max, lat_min, lat_max = reg
    if not (lat_min <= ev["lat"] <= lat_max):
        return False
    lon = ev["lon"]
    return (lon_min <= _lon360(lon) <= lon_max) if lon_max > 180 else (lon_min <= lon <= lon_max)


def _adj_lon(lon: float, reg: list[float]) -> float:
    return lon + 360.0 if (reg[1] > 180 and lon < 0) else lon


# ── data loading ──────────────────────────────────────────────────────────────

def load_training(npz_path: Path) -> list[dict]:
    d = np.load(npz_path, allow_pickle=True)
    return [
        dict(name=str(n), lat=float(d["latitude"][i]), lon=float(d["longitude"][i]),
             mw=float(d["magnitude"][i]), strike=float(d["strike"][i]),
             dip=float(d["dip"][i]), rake=float(d["rake"][i]))
        for i, n in enumerate(d["events"]) if n not in _BLACKLIST
    ]


def load_unseen(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        return [dict(name=r["event"], lat=float(r["event_lat"]), lon=float(r["event_lon"]),
                     mw=float(r["mw_true"]), strike=r["strike"], dip=r["dip"], rake=r["rake"])
                for r in csv.DictReader(f)]


# ── drawing primitives ────────────────────────────────────────────────────────

def _beach(fig, lon, lat, ev, size, cfill, pen, proj, reg) -> None:
    """Plot one beachball (or circle if mechanism unavailable)."""
    if not _valid(ev["strike"], ev["dip"], ev["rake"]):
        fig.plot(x=[lon], y=[lat], style=f"c{size}",
                 fill=cfill, pen=pen, projection=proj, region=reg)
        return
    spec = pd.DataFrame(dict(longitude=[lon], latitude=[lat], depth=[10.0],
                             strike=[float(ev["strike"])], dip=[float(ev["dip"])],
                             rake=[float(ev["rake"])],   magnitude=[ev["mw"]]))
    fig.meca(spec=spec, scale=size, compression_fill=cfill,
             extension_fill="white", pen=pen, projection=proj, region=reg)


def _leader(fig, tlon, tlat, dlon, dlat, proj, reg) -> None:
    """Draw leader line + epicenter dot."""
    fig.plot(x=[tlon, dlon], y=[tlat, dlat], pen="0.35p,black",
             projection=proj, region=reg)
    fig.plot(x=[tlon], y=[tlat], style="c0.065c", fill="black",
             pen="0.2p,black", projection=proj, region=reg)


def _box(fig, reg, proj, greg) -> None:
    """Draw dashed bounding box for an inset region."""
    lon_min, lon_max = reg[0], reg[1] - 360.0 if reg[1] > 180 else reg[1]
    lat_min, lat_max = reg[2], reg[3]
    fig.plot(x=[lon_min, lon_max, lon_max, lon_min, lon_min],
             y=[lat_min, lat_min, lat_max, lat_max, lat_min],
             pen="0.8p,black,-", projection=proj, region=greg)


def _gmt_config() -> None:
    pygmt.config(FONT_ANNOT_PRIMARY="7p,Helvetica,black",
                 MAP_FRAME_TYPE="plain", MAP_FRAME_PEN="0.6p,black",
                 MAP_GRID_PEN_PRIMARY="0.2p,gray70,-", FORMAT_GEO_MAP="ddd")


def _gmt_config_regional() -> None:
    """SRL regional map style: fancy checkerboard frame, ddd.xF geo format."""
    pygmt.config(FONT_ANNOT_PRIMARY="8p,Helvetica,black",
                 FONT_LABEL="9p,Helvetica,black",
                 MAP_FRAME_TYPE="fancy", MAP_FRAME_PEN="1p,black",
                 MAP_FRAME_WIDTH="0.15c",
                 MAP_TICK_LENGTH_PRIMARY="0.08c",
                 FORMAT_GEO_MAP="ddd.xF")


# ── figure A: global map ──────────────────────────────────────────────────────

def plot_global(train: list, unseen: list, out: Path) -> None:
    _gmt_config()
    PROJ = f"N160/{GLOBAL_W}c"
    REG  = [-180, 180, -90, 90]
    gsz  = lambda mw: _sz(mw, 0.10, 0.26)

    fig = pygmt.Figure()
    fig.coast(region=REG, projection=PROJ, water="gray88", land="white",
              shorelines="0.3p,black", borders=["1/0.4p,black"],
              frame=["lrtb", "xg60", "yg30"])

    # Dashed inset boxes + letters
    for key, info in _INSET.items():
        _box(fig, info["region"], PROJ, REG)
        lon_min = info["region"][0]
        lon_max = info["region"][1] - 360.0 if info["region"][1] > 180 else info["region"][1]
        fig.text(text=f"({key})", x=lon_min, y=info["region"][3],
                 offset="0.05c/0.10c", font="7p,Helvetica-Bold,black",
                 justify="BL", projection=PROJ, region=REG)

    # Leader lines (draw first, under beachballs)
    for ev in train:
        off = _OFF_TRAIN.get(ev["name"])
        if off:
            _leader(fig, ev["lon"], ev["lat"], off[0], off[1], PROJ, REG)
    for ev in unseen:
        off = _OFF_UNSEEN.get(ev["name"])
        if off:
            _leader(fig, ev["lon"], ev["lat"], off[0], off[1], PROJ, REG)

    # Training beachballs
    for ev in train:
        off = _OFF_TRAIN.get(ev["name"])
        lon, lat = (off[0], off[1]) if off else (ev["lon"], ev["lat"])
        _beach(fig, lon, lat, ev, gsz(ev["mw"]), "gray62", "0.5p,black", PROJ, REG)

    # Unseen beachballs (on top, thick border)
    for ev in unseen:
        off = _OFF_UNSEEN.get(ev["name"])
        lon, lat = (off[0], off[1]) if off else (ev["lon"], ev["lat"])
        _beach(fig, lon, lat, ev, gsz(ev["mw"]), "black", "1.5p,black", PROJ, REG)

    fig.text(text="(a)", x=-178, y=82, font="9p,Helvetica-Bold,black",
             justify="TL", projection=PROJ, region=REG)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=300, crop=True)
    img = Image.open(out)
    pad = int(img.width * 0.025)
    ImageOps.expand(img, border=pad, fill=(255, 255, 255)).save(out)
    print(f"Saved: {out}")


# ── figures B/C/D: regional insets ───────────────────────────────────────────

def plot_regional(key: str, train: list, unseen: list, out: Path) -> None:
    _gmt_config_regional()
    info = _INSET[key]
    reg  = info["region"]
    proj = f"M{INSET_W:.3f}c"

    lon_span = reg[1] - reg[0] if reg[1] <= 180 else reg[1] - 360 - reg[0] + 360
    lat_span = reg[3] - reg[2]
    dx = 5 if lon_span < 40 else 10
    dy = 5 if lat_span < 30 else 10

    fig = pygmt.Figure()
    fig.coast(region=reg, projection=proj, water="white", land="white",
              shorelines="0.5p,black",
              borders=["1/0.5p,gray50", "2/0.25p,gray50"],
              frame=["WSen", f"xa{dx}f{dx//2}", f"ya{dy}f{dy//2}"])

    sz = lambda mw: _sz(mw, 0.20, 0.48)

    for ev in train:
        if _in_region(ev, reg):
            lon = _adj_lon(ev["lon"], reg)
            _beach(fig, lon, ev["lat"], ev, sz(ev["mw"]),
                   "gray62", "0.5p,black", proj, reg)

    for ev in unseen:
        if _in_region(ev, reg):
            lon = _adj_lon(ev["lon"], reg)
            _beach(fig, lon, ev["lat"], ev, sz(ev["mw"]),
                   "black", "1.5p,black", proj, reg)

    # Panel label (inside, upper-left)
    lon_lo = reg[0] if reg[1] <= 180 else reg[0]
    fig.text(text=info["label"], x=lon_lo, y=reg[3], offset="0.10c/-0.25c",
             font="8p,Helvetica-Bold,black", justify="TL", projection=proj, region=reg)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=300, crop=True)
    print(f"Saved: {out}")


# ── equalize heights of three regional maps ───────────────────────────────────

def equalize_heights(paths: list[Path]) -> None:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    max_h = max(img.height for img in imgs)
    for img, p in zip(imgs, paths):
        if img.height < max_h:
            diff = max_h - img.height
            top  = diff // 2
            bot  = diff - top
            ImageOps.expand(img, border=(0, top, 0, bot),
                            fill=(255, 255, 255)).save(p)
    print(f"Heights equalized to {max_h} px across {[p.name for p in paths]}")


# ── main ──────────────────────────────────────────────────────────────────────

def plot_global_event_map(
    npz_path: Path = NPZ_PATH,
    unseen_csv: Path = UNSEEN_EVENT_CSV,
    out_dir: Path = OUT_DIR,
) -> None:
    train  = load_training(npz_path)
    unseen = load_unseen(unseen_csv)

    out_dir.mkdir(parents=True, exist_ok=True)

    plot_global(train, unseen, out_dir / "fig2a_global.png")

    regional_paths = []
    for key in ["b", "c", "d"]:
        p = out_dir / f"fig2{key}_{_INSET[key]['label'].split()[-1].lower()}.png"
        plot_regional(key, train, unseen, p)
        regional_paths.append(p)

    equalize_heights(regional_paths)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Figure 2: global event map (4 separate PNGs)")
    p.add_argument("--npz",          default=str(NPZ_PATH))
    p.add_argument("--unseen-event", default=str(UNSEEN_EVENT_CSV))
    p.add_argument("--out-dir",      default=str(OUT_DIR))
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    plot_global_event_map(Path(a.npz), Path(a.unseen_event), Path(a.out_dir))
