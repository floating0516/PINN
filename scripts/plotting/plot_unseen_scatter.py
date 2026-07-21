"""Figure 5A: Unseen-event Mw scatter plot (predicted vs. reference).

One point per event for the best PINN config (lp100_full, cm0 threshold).
Marker shape encodes fault mechanism; error bars show prediction IQR.

Usage:
    python scripts/plotting/plot_unseen_scatter.py
    python scripts/plotting/plot_unseen_scatter.py --input outputs/results/.../event_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _style import COLORS, apply_pub_style, style_axes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT  = PROJECT_ROOT / "outputs" / "results" / "unseen_events_8_all_stations" / "event_summary.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "tests" / "figure" / "fig5a_unseen_scatter.png"

# Short event labels for annotation
_SHORT_LABELS = {
    "Iquique 2014 M7.7":  "Iquique\n2014",
    "Nepal 2015 M7.3":    "Nepal\n2015",
    "Kodiak 2018 M7.9":   "Kodiak\n2018",
    "Samos 2020 M7.0":    "Samos\n2020",
    "Luding 2022 M6.6":   "Luding\n2022",
    "Xizang 2025 M7.1":   "Xizang\n2025",
    "Mandalay 2025 M7.7": "Mandalay\n2025",
    "Sand 2025 M7.3":     "Sand\n2025",
}

# Mechanism classification by rake angle
def _mechanism(rake_str: str) -> str:
    try:
        r = float(rake_str)
    except (ValueError, TypeError):
        return "unknown"
    if 45 < r < 135:
        return "thrust"
    if -135 < r < -45:
        return "normal"
    return "strike-slip"

_MECH_STYLE = {
    "thrust":      dict(marker="^", color="#D55E00", label="Thrust"),
    "normal":      dict(marker="v", color="#009E73", label="Normal"),
    "strike-slip": dict(marker="o", color=COLORS["pinn"], label="Strike-slip"),
    "unknown":     dict(marker="D", color="#999999", label="Unknown"),
}


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_unseen_scatter(
    input_csv: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    apply_pub_style()
    rows = load_rows(input_csv)

    mw_true = np.array([float(r["mw_true"])       for r in rows])
    mw_pred = np.array([float(r["mw_pred_median"]) for r in rows])
    iqr     = np.array([float(r["pred_iqr"])        for r in rows])
    mechs   = [_mechanism(r["rake"]) for r in rows]
    labels  = [_SHORT_LABELS.get(r["event"], r["event"]) for r in rows]

    mae  = np.mean(np.abs(mw_pred - mw_true))
    rmse = math.sqrt(np.mean((mw_pred - mw_true) ** 2))

    lo = min(mw_true.min(), mw_pred.min()) - 0.2
    hi = max(mw_true.max(), mw_pred.max()) + 0.3

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    style_axes(ax)

    # ±0.3 Mw tolerance band
    ref = np.linspace(lo, hi, 200)
    ax.fill_between(ref, ref - 0.3, ref + 0.3,
                    color="#EEEEEE", zorder=0, label="±0.3 $M_w$")
    # 1:1 line
    ax.plot(ref, ref, color="#333333", lw=0.9, zorder=1)

    # Per-event label offsets (dx, dy) in data units — manually tuned to avoid overlap
    _offsets: dict[str, tuple[float, float]] = {
        "Iquique\n2014":  (-0.05, +0.03),   # left of point
        "Nepal\n2015":    (-0.06, -0.05),   # below-left  (overlaps with Sand)
        "Kodiak\n2018":   (-0.05, -0.05),   # below-left
        "Samos\n2020":    (+0.05, +0.00),   # right
        "Luding\n2022":   (+0.05, +0.00),   # right
        "Xizang\n2025":   (+0.05, +0.04),   # right-up
        "Mandalay\n2025": (-0.05, -0.05),   # below-left
        "Sand\n2025":     (+0.05, -0.04),   # right-down (separates from Nepal)
    }

    plotted_mechs: set[str] = set()
    for mw_t, mw_p, iq, mech, lbl in zip(mw_true, mw_pred, iqr, mechs, labels):
        sty = _MECH_STYLE[mech]
        ax.errorbar(mw_t, mw_p, yerr=iq / 2,
                    fmt=sty["marker"], color=sty["color"],
                    markersize=6, elinewidth=0.8, capsize=2.5, capthick=0.8,
                    zorder=3, label=sty["label"] if mech not in plotted_mechs else "_")
        plotted_mechs.add(mech)
        dx, dy = _offsets.get(lbl, (0.05, 0.0))
        ha = "left" if dx > 0 else "right"
        ax.text(mw_t + dx, mw_p + dy, lbl,
                fontsize=5.5, va="center", ha=ha, color="#444444", linespacing=1.2)

    # Metrics box — lower right, away from legend
    ax.text(0.97, 0.06,
            f"MAE  = {mae:.3f}\nRMSE = {rmse:.3f}",
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=7, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.6))

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Reference $M_w$", fontsize=9)
    ax.set_ylabel("Predicted $M_w$ (PINN, lp100_full)", fontsize=9)

    # Legend: mechanism markers + band
    band_patch = plt.Rectangle((0, 0), 1, 1, fc="#EEEEEE", ec="none")
    mech_handles = [
        mlines.Line2D([], [], **{k: v for k, v in sty.items() if k != "label"},
                      markersize=5, lw=0, label=sty["label"])
        for mech, sty in _MECH_STYLE.items()
        if mech in plotted_mechs
    ]
    ax.legend(handles=mech_handles + [band_patch],
              labels=[h.get_label() for h in mech_handles] + ["±0.3 $M_w$"],
              loc="upper left", fontsize=6.5,
              handlelength=1.0, handletextpad=0.4, borderpad=0.4)

    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Figure 5A: unseen-event Mw scatter")
    p.add_argument("--input",  default=str(DEFAULT_INPUT))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    plot_unseen_scatter(Path(args.input), Path(args.output))
