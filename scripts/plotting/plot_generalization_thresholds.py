"""Figure 8: Three-threshold generalization comparison.

Grouped bar chart of unseen-event MAE across cm0/cm1/cm2 thresholds
for a curated set of configurations, showing how physics constraints
improve generalization—especially for weak-signal stations (cm0).

Bootstrap 95% CI is loaded from paper/result_ana/e1_4_bootstrap_ci.csv
for E1.1–E1.3 configs.  E1.4/E1.5 point estimates are hardcoded from
e1_results_summary.md §8 (no bootstrap run for those configs).

Usage:
    python scripts/plotting/plot_generalization_thresholds.py
    python scripts/plotting/plot_generalization_thresholds.py \\
        --bootstrap-csv paper/result_ana/e1_4_bootstrap_ci.csv \\
        --output tests/figure/fig8_generalization_thresholds.png
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _style import COLORS, THRESHOLD_LABELS, apply_pub_style, style_axes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOOTSTRAP_CSV = PROJECT_ROOT / "paper" / "result_ana" / "e1_4_bootstrap_ci.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "tests" / "figure" / "fig8_generalization_thresholds.png"

BASELINE_MAE = 0.2971  # empirical magnitude-median baseline


# ── configuration registry ──────────────────────────────────────────────────

@dataclass
class ConfigEntry:
    """One model configuration for the figure."""
    key: str          # matches 'experiment' column in bootstrap CSV (or "hardcoded")
    label: str        # x-axis tick label (newline ok)
    cm0: float        # unseen-event MAE, all stations
    cm1: float        # unseen-event MAE, ≥1 cm
    cm2: float        # unseen-event MAE, ≥2 cm
    # Bootstrap 95% CI — filled from CSV; None means no bootstrap available
    cm0_ci: Optional[tuple[float, float]] = field(default=None)
    cm1_ci: Optional[tuple[float, float]] = field(default=None)
    cm2_ci: Optional[tuple[float, float]] = field(default=None)

    @property
    def has_ci(self) -> bool:
        return self.cm0_ci is not None


# Point estimates from e1_results_summary.md §4.1 and §8.1–§8.2.
# Ordering: global winner → full-physics → stable → specialist → DNN.
_CONFIGS: list[ConfigEntry] = [
    ConfigEntry("farOnly_lp100", "farOnly\nlp100\n(E1.4)",
                cm0=0.1470, cm1=0.1465, cm2=0.1313),
    ConfigEntry("ls030_full",    "ls030\nfull\n(E1.5)",
                cm0=0.1517, cm1=0.1489, cm2=0.1626),
    ConfigEntry("lp100_full",    "lp100\nfull\n(E1.1)",
                cm0=0.1584, cm1=0.1281, cm2=0.1326),
    ConfigEntry("lp080_full",    "lp080\nfull\n(E1.1)",
                cm0=0.1528, cm1=0.1541, cm2=0.1568),
    ConfigEntry("far_only",      "far_only\n(E1.3)",
                cm0=0.2072, cm1=0.1519, cm2=0.1149),
    ConfigEntry("far_S_only",    "far_S\nonly\n(E1.3)",
                cm0=0.1676, cm1=0.1724, cm2=0.1302),
    ConfigEntry("pure_dnn",      "Pure DNN\n(E1.2)",
                cm0=0.2804, cm1=0.2464, cm2=0.2353),
]

_THRESHOLD_KEYS = ("cm0", "cm1", "cm2")


# ── bootstrap CI loading ────────────────────────────────────────────────────

def load_bootstrap_ci(csv_path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    """Return {experiment: {threshold: (ci_lower, ci_upper)}} from bootstrap CSV."""
    ci: dict[str, dict[str, tuple[float, float]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            exp = row["experiment"]
            thr = row["threshold"]
            ci.setdefault(exp, {})[thr] = (
                float(row["mae_ci_lower"]),
                float(row["mae_ci_upper"]),
            )
    return ci


def attach_ci(configs: list[ConfigEntry], bootstrap_csv: Path) -> None:
    """Populate ConfigEntry.{cm0,cm1,cm2}_ci from the bootstrap CSV in-place."""
    if not bootstrap_csv.exists():
        return
    ci_data = load_bootstrap_ci(bootstrap_csv)
    for cfg in configs:
        if cfg.key not in ci_data:
            continue
        thr_ci = ci_data[cfg.key]
        cfg.cm0_ci = thr_ci.get("cm0")
        cfg.cm1_ci = thr_ci.get("cm1")
        cfg.cm2_ci = thr_ci.get("cm2")


# ── plotting ────────────────────────────────────────────────────────────────

def plot_generalization_thresholds(
    *,
    bootstrap_csv: Path = DEFAULT_BOOTSTRAP_CSV,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    attach_ci(_CONFIGS, bootstrap_csv)
    apply_pub_style()

    n = len(_CONFIGS)
    x = np.arange(n, dtype=float)
    width = 0.22
    offsets = {"cm0": -width, "cm1": 0.0, "cm2": width}
    thr_colors = {k: COLORS[k] for k in _THRESHOLD_KEYS}

    # Double-column width: 183 mm ≈ 7.2 inch
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    style_axes(ax)

    for thr, offset in offsets.items():
        mae_vals  = np.array([getattr(c, thr) for c in _CONFIGS])
        ci_lower  = np.array([
            getattr(c, f"{thr}_ci")[0] if getattr(c, f"{thr}_ci") else np.nan
            for c in _CONFIGS
        ])
        ci_upper  = np.array([
            getattr(c, f"{thr}_ci")[1] if getattr(c, f"{thr}_ci") else np.nan
            for c in _CONFIGS
        ])

        err_low  = mae_vals - ci_lower   # downward error
        err_high = ci_upper - mae_vals   # upward error

        has_ci  = ~np.isnan(ci_lower)
        no_ci   = ~has_ci

        # Bars with CI — solid
        ax.bar(x[has_ci] + offset, mae_vals[has_ci], width=width,
               color=thr_colors[thr], alpha=0.85, zorder=3)
        ax.errorbar(x[has_ci] + offset, mae_vals[has_ci],
                    yerr=[err_low[has_ci], err_high[has_ci]],
                    fmt="none", ecolor="#333333", elinewidth=0.8,
                    capsize=2.5, capthick=0.8, zorder=4)

        # Bars without CI — hatched to signal "point estimate only"
        if no_ci.any():
            ax.bar(x[no_ci] + offset, mae_vals[no_ci], width=width,
                   color=thr_colors[thr], alpha=0.85, zorder=3,
                   hatch="//", edgecolor="white")

    # Baseline reference line
    baseline_line = ax.axhline(BASELINE_MAE, color="#AA0000", linewidth=0.9,
                               linestyle="--", zorder=2)

    # Build legend manually: one entry per threshold + baseline + hatch note
    thr_patches = [
        mpatches.Patch(color=thr_colors[thr], alpha=0.85,
                       label=THRESHOLD_LABELS[thr])
        for thr in _THRESHOLD_KEYS
    ]
    hatch_patch = mpatches.Patch(facecolor="#BBBBBB", hatch="//",
                                 edgecolor="white", label="No CI (point estimate)")
    ax.legend(
        handles=thr_patches + [baseline_line, hatch_patch],
        labels=[p.get_label() for p in thr_patches]
              + [f"Baseline ({BASELINE_MAE:.3f})", hatch_patch.get_label()],
        loc="upper right", ncol=2,
        handlelength=1.4, handletextpad=0.4, columnspacing=0.8, borderpad=0.4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([c.label for c in _CONFIGS], fontsize=7, linespacing=1.3)
    ax.set_ylabel("Unseen-event MAE ($M_w$)", fontsize=9)
    ax.set_xlabel("Configuration", fontsize=9)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.set_ylim(0, 0.36)

    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


# ── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Figure 8: three-threshold generalization comparison"
    )
    parser.add_argument("--bootstrap-csv", default=str(DEFAULT_BOOTSTRAP_CSV),
                        help="Bootstrap CI CSV (paper/result_ana/e1_4_bootstrap_ci.csv)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output PNG path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_generalization_thresholds(
        bootstrap_csv=Path(args.bootstrap_csv),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
