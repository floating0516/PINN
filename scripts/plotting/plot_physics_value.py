"""Figure 4: Physics-constraint value comparison.

Three-bar chart (Baseline / Pure DNN / Best PINN) showing test-set MAE,
with annotated percentage reductions to quantify the gain from physics
constraints (E1.2 DNN vs. E1.4 best PINN farOnly_lp100).

Usage:
    python scripts/plotting/plot_physics_value.py
    python scripts/plotting/plot_physics_value.py --output /path/to/fig4.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _style import COLORS, apply_pub_style, style_axes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "tests" / "figure" / "fig4_physics_value.png"

# ── fixed data from e1_results_summary.md §1.2 + §6.2 ─────────────────────
# Baseline: median of training-set magnitudes (no model)
# Pure DNN: E1.2, λ_mag = λ_synth = 0
# Best PINN: E1.4 farOnly_lp100 (far P+S, λ_mag=1.0, λ_synth=0.5)
_MODELS = [
    ("Empirical\nBaseline",      0.2971, COLORS["baseline"]),
    ("Pure DNN\n(λ_mag = 0)",    0.1410, COLORS["dnn"]),
    ("Best PINN\n(farOnly_lp100)", 0.0790, COLORS["pinn"]),
]

# Pairwise reductions to annotate: (left_idx, right_idx, label)
_REDUCTIONS = [
    (0, 1, "−52.5%\nvs. Baseline"),
    (1, 2, "−44.0%\nvs. DNN"),
    (0, 2, "−73.4%\nvs. Baseline"),
]


def _draw_bracket(ax, x_left: float, x_right: float, y_bracket: float,
                  y_left_bar: float, y_right_bar: float, label: str) -> None:
    """Draw a comparison bracket between two bars with a percent-reduction label."""
    kw = dict(color="#555555", lw=0.7, clip_on=False)
    tick_len = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.018
    xm = (x_left + x_right) / 2
    # Horizontal bridge
    ax.plot([x_left, x_right], [y_bracket, y_bracket], **kw)
    # Vertical ticks from bar tops up to bridge
    ax.plot([x_left, x_left],   [y_left_bar,  y_bracket], **kw)
    ax.plot([x_right, x_right], [y_right_bar, y_bracket], **kw)
    ax.text(xm, y_bracket + tick_len * 0.4, label,
            ha="center", va="bottom", fontsize=6.5, color="#333333",
            linespacing=1.3)


def plot_physics_value(output_path: str | Path = DEFAULT_OUTPUT) -> Path:
    apply_pub_style()

    labels, mae_vals, bar_colors = zip(*_MODELS)
    x = np.arange(len(labels), dtype=float)
    width = 0.45

    # Single-column width: ~3.5 inch; height adjusted for bracket annotations
    fig, ax = plt.subplots(figsize=(3.5, 4.0))
    style_axes(ax)

    bars = ax.bar(x, mae_vals, width=width, color=bar_colors,
                  edgecolor="white", linewidth=0.4, zorder=3)

    # Value labels on top of each bar
    for bar, val in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.004,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Bracket annotations — staggered heights to avoid overlap
    y_max = max(mae_vals)
    bracket_heights = [
        y_max * 1.08,   # DNN vs Baseline
        y_max * 1.20,   # Best PINN vs DNN
        y_max * 1.35,   # Best PINN vs Baseline (outermost)
    ]
    # Start vertical ticks above the value labels (fontsize 7.5 ≈ 0.018 data units here)
    text_clearance = y_max * 0.065
    for (li, ri, label), y_br in zip(_REDUCTIONS, bracket_heights):
        _draw_bracket(ax, x[li], x[ri], y_br,
                      mae_vals[li] + text_clearance, mae_vals[ri] + text_clearance, label)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, linespacing=1.4)
    ax.set_ylabel("Test MAE ($M_w$)", fontsize=9)
    ax.set_ylim(0, y_max * 1.55)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))

    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Figure 4: physics-constraint value comparison")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PNG path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_physics_value(args.output)


if __name__ == "__main__":
    main()
