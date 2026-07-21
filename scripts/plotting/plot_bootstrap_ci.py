"""Bootstrap 95% CI summary (forest plot) for the SRL manuscript.

Horizontal error-bar plot showing unseen-event MAE mean and 95% bootstrap
confidence interval for key configurations across three station-quality
thresholds. Highlights that PINN vs. Pure DNN differences are clearest
in the strong-signal regime (cm2).

Output: paper/srl/figures/fig_bootstrap_ci.pdf

Usage:
    python scripts/plotting/plot_bootstrap_ci.py
    python scripts/plotting/plot_bootstrap_ci.py \\
        --bootstrap-csv paper/result_ana/e1_4_bootstrap_ci.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "paper" / "srl" / "figure_sources"))
from style import apply_style, despine, C, COL_SINGLE

DEFAULT_BOOTSTRAP_CSV = PROJECT_ROOT / "paper" / "result_ana" / "e1_4_bootstrap_ci.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper" / "srl" / "figures" / "fig_bootstrap_ci.pdf"

# Configs to show, in display order (top → bottom), labelled in paper terms
_SELECTED = ["lp100_full", "lp080_full", "far_S_only", "far_only", "pure_dnn"]
_LABELS = {
    "lp100_full": "Full radiation\n($\\lambda_\\mathrm{mag}=1.0$)",
    "lp080_full": "Full radiation\n($\\lambda_\\mathrm{mag}=0.8$)",
    "far_S_only": "Far-field\nS only",
    "far_only":   "Far-field\nP+S",
    "pure_dnn":   "Pure DNN",
}
_THRESHOLDS  = ("cm0", "cm1", "cm2")
_THR_COLORS  = {"cm0": C["full"], "cm1": C["glehman"], "cm2": C["far"]}
_THR_MARKERS = {"cm0": "o", "cm1": "s", "cm2": "^"}
_THR_LABELS  = {"cm0": "cm0 (all stations)",
                "cm1": "cm1 ($\\geq$1 cm)",
                "cm2": "cm2 ($\\geq$2 cm)"}
_BASELINE_MAE = 0.2971


def load_bootstrap(csv_path: Path) -> dict[str, dict[str, dict]]:
    """Return {experiment: {threshold: {mean, ci_lower, ci_upper, bias}}}."""
    data: dict[str, dict[str, dict]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            exp = row["experiment"]
            thr = row["threshold"]
            data.setdefault(exp, {})[thr] = {
                "mean":     float(row["mae_mean"]),
                "ci_lower": float(row["mae_ci_lower"]),
                "ci_upper": float(row["mae_ci_upper"]),
                "bias":     float(row["bias_mean"]),
            }
    return data


def plot_bootstrap_ci(
    bootstrap_csv: Path = DEFAULT_BOOTSTRAP_CSV,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    apply_style()
    data = load_bootstrap(bootstrap_csv)

    n_thr = len(_THRESHOLDS)
    group_gap = 0.5    # extra vertical space between config groups
    thr_spacing = 0.28 # vertical distance between thresholds within a group

    y_positions: dict[tuple[str, str], float] = {}
    y_tick_pos: list[float] = []
    y_tick_lab: list[str] = []
    current_y = 0.0
    for cfg in reversed(_SELECTED):   # reversed so best config at top
        group_center = current_y + (n_thr - 1) * thr_spacing / 2
        y_tick_pos.append(group_center)
        y_tick_lab.append(_LABELS[cfg])
        for i, thr in enumerate(_THRESHOLDS):
            y_positions[(cfg, thr)] = current_y + i * thr_spacing
        current_y += n_thr * thr_spacing + group_gap

    fig, ax = plt.subplots(figsize=(COL_SINGLE, 3.4))
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, color="#CCCCCC", zorder=0)
    ax.grid(False, axis="y")

    for cfg in _SELECTED:
        if cfg not in data:
            continue
        for thr in _THRESHOLDS:
            if thr not in data[cfg]:
                continue
            d = data[cfg][thr]
            yp = y_positions[(cfg, thr)]
            err_lo = d["mean"] - d["ci_lower"]
            err_hi = d["ci_upper"] - d["mean"]
            ax.errorbar(d["mean"], yp,
                        xerr=[[err_lo], [err_hi]],
                        fmt=_THR_MARKERS[thr], color=_THR_COLORS[thr],
                        markersize=4.5, elinewidth=0.9,
                        capsize=2.5, capthick=0.9, zorder=4)

    # Group separator lines
    sep_y = -group_gap / 2
    for _ in _SELECTED[:-1]:
        sep_y_val = sep_y + n_thr * thr_spacing
        ax.axhline(sep_y_val, color="#DDDDDD", lw=0.6, zorder=1)
        sep_y += n_thr * thr_spacing + group_gap

    ax.axvline(_BASELINE_MAE, color=C["dnn"], lw=0.9, linestyle=":",
               zorder=2, label=f"Median baseline ({_BASELINE_MAE:.3f})")

    ax.set_yticks(y_tick_pos)
    ax.set_yticklabels(y_tick_lab, fontsize=7, linespacing=1.3)
    ax.set_xlabel(r"Unseen-event MAE ($M_\mathrm{w}$), 95% bootstrap CI")
    ax.set_xlim(0.08, 0.38)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    despine(ax)

    thr_handles = [
        mlines.Line2D([], [], color=_THR_COLORS[t], marker=_THR_MARKERS[t],
                      markersize=4.5, lw=0, label=_THR_LABELS[t])
        for t in _THRESHOLDS
    ]
    ref_handles, ref_labels = ax.get_legend_handles_labels()
    ax.legend(handles=thr_handles + ref_handles,
              labels=[h.get_label() for h in thr_handles] + ref_labels,
              loc="lower center", frameon=False, fontsize=6.5,
              handlelength=1.2, handletextpad=0.4, ncol=2,
              columnspacing=0.8, bbox_to_anchor=(0.5, 1.005))

    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bootstrap CI forest plot")
    p.add_argument("--bootstrap-csv", default=str(DEFAULT_BOOTSTRAP_CSV))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    plot_bootstrap_ci(Path(args.bootstrap_csv), Path(args.output))
