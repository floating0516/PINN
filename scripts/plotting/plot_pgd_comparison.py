"""Figure 5B: PINN vs. PGD scaling-law comparison (cm2 threshold).

Grouped bar chart of per-event |error| for PINN (lp100_full) against
three PGD scaling laws (Crowell, Ruhl, Melgar). Uses the cm2 (≥2 cm)
event summary, where PGD methods have the most reliable signal.
Overall MAE and RMSE are annotated above each method group.

Usage:
    python scripts/plotting/plot_pgd_comparison.py
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _style import COLORS, apply_pub_style, style_axes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT  = PROJECT_ROOT / "outputs" / "results" / "unseen_events_8_2cm" / "event_summary.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "tests" / "figure" / "fig5b_pgd_comparison.png"

_METHODS = [
    ("PINN\n(lp100_full)", "mw_pred_median",           COLORS["pinn"]),
    ("PGD\nCrowell",       "pgd_crowell_mw_pred_median", COLORS["crowell"]),
    ("PGD\nRuhl",          "pgd_ruhl_mw_pred_median",    COLORS["ruhl"]),
    ("PGD\nMelgar",        "pgd_melgar_mw_pred_median",  COLORS["melgar"]),
]

# Short event x-axis labels
_SHORT = {
    "Iquique 2014 M7.7":  "Iquique\n2014",
    "Nepal 2015 M7.3":    "Nepal\n2015",
    "Kodiak 2018 M7.9":   "Kodiak\n2018",
    "Samos 2020 M7.0":    "Samos\n2020",
    "Luding 2022 M6.6":   "Luding\n2022",
    "Xizang 2025 M7.1":   "Xizang\n2025",
    "Mandalay 2025 M7.7": "Mandalay\n2025",
    "Sand 2025 M7.3":     "Sand\n2025",
}


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_pgd_comparison(
    input_csv: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    apply_pub_style()
    rows = load_rows(input_csv)

    event_labels = [_SHORT.get(r["event"], r["event"]) for r in rows]
    mw_true = np.array([float(r["mw_true"]) for r in rows])

    n_events  = len(rows)
    n_methods = len(_METHODS)
    width     = 0.18
    x = np.arange(n_events, dtype=float)
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    style_axes(ax)

    method_mae = {}
    for (label, col_key, color), offset in zip(_METHODS, offsets):
        abs_errors = np.array([
            abs(float(r[col_key]) - float(r["mw_true"])) for r in rows
        ])
        method_mae[label] = np.mean(abs_errors)
        ax.bar(x + offset, abs_errors, width=width,
               color=color, alpha=0.85, zorder=3, label=label)

    # ±0.3 reference line
    ax.axhline(0.3, color="#999999", lw=0.8, linestyle="--", zorder=2,
               label="|error| = 0.3")

    ax.set_xticks(x)
    ax.set_xticklabels(event_labels, fontsize=7, linespacing=1.3)
    ax.set_ylabel("|Predicted − Reference| ($M_w$)", fontsize=9)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

    y_top = max(
        abs(float(r[col_key]) - float(r["mw_true"]))
        for _, col_key, _ in _METHODS
        for r in rows
    ) * 1.18
    ax.set_ylim(0, y_top)

    # Summary MAE text box — upper right, one line per method
    summary_lines = [f"{'Method':<18} {'MAE':>5}"]
    summary_lines.append("─" * 24)
    for (label, col_key, color), _ in zip(_METHODS, offsets):
        short = label.replace("\n", " ")
        summary_lines.append(f"{short:<18} {method_mae[label]:.3f}")
    ax.text(0.99, 0.98, "\n".join(summary_lines),
            transform=ax.transAxes, va="top", ha="right",
            fontsize=6, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", lw=0.6))

    ax.legend(loc="upper left", ncol=n_methods + 1, fontsize=6.5,
              handlelength=1.2, handletextpad=0.4, columnspacing=0.8, borderpad=0.4)

    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Figure 5B: PINN vs PGD comparison")
    p.add_argument("--input",  default=str(DEFAULT_INPUT))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    plot_pgd_comparison(Path(args.input), Path(args.output))
