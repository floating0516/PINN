"""Figure 3B: Ablation study results (E1.3).

Horizontal bar chart showing test-set MAE for five physics-term
configurations, revealing that far-field S-wave alone outperforms
all combinations including the full model.

Usage:
    python scripts/plotting/plot_ablation.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _style import COLORS, apply_pub_style, style_axes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "tests" / "figure" / "fig3b_ablation.png"

# ── data from e1_results_summary.md §1.3 ───────────────────────────────────
# All use full radiation pattern, λ_mag=0.4, λ_synth=0.5
_ABLATION = [
    # (label,             near, far_P, far_S, test_mae)
    ("far S only",        False, False, True,  0.0883),
    ("far P only",        False, True,  False, 0.1027),
    ("far P + S",         False, True,  True,  0.1115),
    ("near field only",   True,  False, False, 0.1197),
    ("full (near+far)",   True,  True,  True,  0.1212),
]
_DNN_MAE      = 0.1410
_BASELINE_MAE = 0.2971

# Physics term presence → bar color (darkening = more terms, not necessarily better)
_BAR_COLOR = COLORS["pinn"]
_DIM_COLOR  = "#6AAFD4"   # lighter blue for weaker configs


def plot_ablation(output_path: str | Path = DEFAULT_OUTPUT) -> Path:
    apply_pub_style()

    labels   = [r[0] for r in _ABLATION]
    mae_vals = np.array([r[4] for r in _ABLATION])
    # Color: best config gets full blue, others get lighter shade
    bar_colors = [_BAR_COLOR if v == mae_vals.min() else _DIM_COLOR for v in mae_vals]

    # Sort best (lowest MAE) at top
    order = np.argsort(mae_vals)
    labels_s   = [labels[i]     for i in order]
    mae_s      = mae_vals[order]
    colors_s   = [bar_colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    style_axes(ax)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, color="#CCCCCC", zorder=0)
    ax.grid(False, axis="y")

    y = np.arange(len(labels_s), dtype=float)
    bars = ax.barh(y, mae_s, height=0.55, color=colors_s,
                   edgecolor="white", linewidth=0.4, zorder=3)

    # Value labels inside/outside bars
    for bar, val in zip(bars, mae_s):
        x_label = val + 0.002
        ax.text(x_label, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=7)

    # DNN reference line (all ablation configs are well below Baseline, so
    # we skip the Baseline vline and annotate it as text to keep x-axis compact)
    ax.axvline(_DNN_MAE, color=COLORS["dnn"], linewidth=1.0,
               linestyle="--", zorder=2, label=f"Pure DNN ({_DNN_MAE:.3f})")
    ax.text(0.995, 0.02, f"Baseline = {_BASELINE_MAE:.3f} →",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color=COLORS["baseline"], style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(labels_s, fontsize=8)
    ax.set_xlabel("Test MAE ($M_w$)", fontsize=9)
    ax.set_xlim(0, 0.22)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.invert_yaxis()   # best config at top

    # Legend: color meaning
    best_patch = mpatches.Patch(color=_BAR_COLOR,  label="Best configuration")
    rest_patch = mpatches.Patch(color=_DIM_COLOR,  label="Other ablation configs")
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles=[best_patch, rest_patch] + handles,
              labels=[best_patch.get_label(), rest_patch.get_label()] + labels_leg,
              loc="lower right", fontsize=6.5,
              handlelength=1.2, handletextpad=0.4, borderpad=0.4,
              bbox_to_anchor=(0.99, 0.10))

    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Figure 3B: ablation study results")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p


if __name__ == "__main__":
    plot_ablation(build_parser().parse_args().output)
