"""Regenerate Figure 1a (global event map) with training/unseen legend.

Training events: gray compression quadrants.
Withheld (unseen) events: black compression quadrants, thicker outline.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pygmt

from scripts.plotting.plot_global_event_map import (
    GLOBAL_W, NPZ_PATH, UNSEEN_EVENT_CSV, _OFF_TRAIN, _OFF_UNSEEN,
    _beach, _gmt_config, _leader, _sz, load_training, load_unseen,
)

OUT = PROJECT_ROOT / "tests" / "figure" / "fig1a_global.png"


def main() -> None:
    train = load_training(NPZ_PATH)
    unseen = load_unseen(UNSEEN_EVENT_CSV)

    _gmt_config()
    PROJ = f"N160/{GLOBAL_W}c"
    REG = [-180, 180, -90, 90]
    gsz = lambda mw: _sz(mw, 0.10, 0.26)

    fig = pygmt.Figure()
    fig.coast(region=REG, projection=PROJ, water="gray88", land="white",
              shorelines="0.3p,black", borders=["1/0.4p,black"],
              frame=["lrtb", "xg60", "yg30"])

    for ev in train:
        off = _OFF_TRAIN.get(ev["name"])
        if off:
            _leader(fig, ev["lon"], ev["lat"], off[0], off[1], PROJ, REG)
    for ev in unseen:
        off = _OFF_UNSEEN.get(ev["name"])
        if off:
            _leader(fig, ev["lon"], ev["lat"], off[0], off[1], PROJ, REG)

    for ev in train:
        off = _OFF_TRAIN.get(ev["name"])
        lon, lat = (off[0], off[1]) if off else (ev["lon"], ev["lat"])
        _beach(fig, lon, lat, ev, gsz(ev["mw"]), "gray62", "0.5p,black", PROJ, REG)

    for ev in unseen:
        off = _OFF_UNSEEN.get(ev["name"])
        lon, lat = (off[0], off[1]) if off else (ev["lon"], ev["lat"])
        _beach(fig, lon, lat, ev, gsz(ev["mw"]), "black", "1.2p,black", PROJ, REG)

    # Legend (lower-left, over Southern Ocean)
    lx, ly = -138.0, -68.0
    fig.plot(x=[lx], y=[ly], style="c0.20c", fill="gray62", pen="0.5p,black",
             projection=PROJ, region=REG)
    fig.text(text="Training (35)", x=lx + 7, y=ly, justify="LM",
             font="7p,Helvetica,black", projection=PROJ, region=REG)
    fig.plot(x=[lx], y=[ly - 9.0], style="c0.20c", fill="black", pen="1.2p,black",
             projection=PROJ, region=REG)
    fig.text(text="Withheld / unseen (8)", x=lx + 7, y=ly - 9.0, justify="LM",
             font="7p,Helvetica,black", projection=PROJ, region=REG)

    fig.text(text="(a)", x=-178, y=82, font="9p,Helvetica-Bold,black",
             justify="TL", projection=PROJ, region=REG)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), dpi=300, crop=True)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
