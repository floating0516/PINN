"""Shared publication style and color palette for SRL figures.

Import pattern (works regardless of cwd):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from _style import apply_pub_style, COLORS
"""
from __future__ import annotations

import matplotlib as mpl

# Okabe-Ito colorblind-safe palette
COLORS: dict[str, str] = {
    "pinn":     "#0072B2",  # blue  — PINN models
    "dnn":      "#56B4E9",  # sky blue — pure DNN
    "baseline": "#999999",  # gray  — empirical baseline
    "cm0":      "#0072B2",  # blue  — all stations (cm0 threshold)
    "cm1":      "#009E73",  # green — ≥1 cm
    "cm2":      "#D55E00",  # vermillion — ≥2 cm
    "ref":      "#000000",  # black — reference lines
    # PGD scaling laws (consistent with existing plot_unseen_method_comparison)
    "crowell":  "#D55E00",
    "ruhl":     "#009E73",
    "melgar":   "#CC79A7",  # pink
}

# Human-readable threshold labels
THRESHOLD_LABELS: dict[str, str] = {
    "cm0": "cm0 (all stations)",
    "cm1": "cm1 (≥ 1 cm)",
    "cm2": "cm2 (≥ 2 cm)",
}


def apply_pub_style() -> None:
    """Apply SRL/AGU publication-quality rcParams."""
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         8,
        "axes.labelsize":    9,
        "axes.titlesize":    10,
        "axes.titleweight":  "bold",
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   7,
        "legend.frameon":    False,
        "axes.linewidth":    0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth":   1.2,
        "patch.linewidth":   0.5,
        "pdf.fonttype":      42,  # editable text in PDF
        "ps.fonttype":       42,
    })


def style_axes(ax) -> None:
    """Remove top/right spines and add subtle y-grid."""
    ax.set_facecolor("white")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#CCCCCC", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)
