"""λ hyperparameter sensitivity figures for the SRL manuscript.

Produces the two single figures used in the paper:
  paper/srl/figures/fig_lambda_mag.pdf   — test MAE vs λ_mag (full / simplified / far-only)
  paper/srl/figures/fig_lambda_synth.pdf — test MAE vs λ_synth at λ_mag=2.0 (full / simplified)

Usage:
    python scripts/plotting/plot_lambda_sensitivity.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "paper" / "srl" / "figure_sources"))
from style import apply_style, despine, add_panel_label, C, COL_SINGLE

OUT_DIR = PROJECT_ROOT / "paper" / "srl" / "figures"
OUT_LAMBDA_MAG = OUT_DIR / "fig_lambda_mag.pdf"
OUT_LAMBDA_SYNTH = OUT_DIR / "fig_lambda_synth.pdf"

# ── data from e1_results_summary.md §1.1, §6.1, §6.2, §7.1 ────────────────

# λ_mag sweep, λ_synth=0.5 fixed
_FULL_LMAG = {
    0.1: 0.2229, 0.2: 0.1269, 0.4: 0.1212,
    0.6: 0.1084, 0.8: 0.1026, 1.0: 0.0987,
    1.5: 0.0893, 2.0: 0.0869, 3.0: 0.0894,
}
_SIMP_LMAG = {
    0.1: 0.2456, 0.2: 0.2062, 0.4: 0.1985,
    0.6: 0.1664, 0.8: 0.1708, 1.0: 0.1380,
}
_FAR_LMAG = {1.0: 0.0790, 1.5: 0.0805}

# λ_synth sweep at λ_mag=2.0
_FULL_LSYN = {0.1: 0.0854, 0.3: 0.0883, 0.5: 0.0869, 0.7: 0.0850, 1.0: 0.0897}
_SIMP_LSYN = {0.1: 0.0971, 0.3: 0.1093, 0.5: 0.1210, 0.7: 0.1354, 1.0: 0.1293}

_DNN_MAE = 0.1410


def _dict_to_xy(d: dict) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array(sorted(d))
    return xs, np.array([d[x] for x in xs])


def _mark_optimum(ax, x_opt: float, y_opt: float, color: str) -> None:
    ax.scatter([x_opt], [y_opt], s=70, color=color, zorder=5,
               marker="*", edgecolors="white", linewidths=0.5)


def _star_handle() -> mlines.Line2D:
    return mlines.Line2D([], [], color="#555555", marker="*",
                         markersize=7, lw=0, label="Optimum (per mode)")


# Identical axes rectangle for both panels so they render at the same width
_AXES_RECT = dict(left=0.15, right=0.97, top=0.90, bottom=0.17)


def _save(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Fixed canvas (no tight cropping) so both figures share the same geometry
    fig.savefig(output_path, bbox_inches=None)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches=None)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── λ_mag sensitivity ───────────────────────────────────────────────────────

def plot_lambda_mag(output_path: Path = OUT_LAMBDA_MAG) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.6))
    fig.subplots_adjust(**_AXES_RECT)

    xf, yf = _dict_to_xy(_FULL_LMAG)
    ax.plot(xf, yf, color=C["full"], marker="o", ms=3.5,
            label="Full radiation")
    _mark_optimum(ax, 2.0, _FULL_LMAG[2.0], C["full"])

    xs, ys = _dict_to_xy(_SIMP_LMAG)
    ax.plot(xs, ys, color=C["simpl"], marker="s", ms=3.5,
            linestyle="--", label="Simplified radiation")
    _mark_optimum(ax, 1.0, _SIMP_LMAG[1.0], C["simpl"])

    xfo, yfo = _dict_to_xy(_FAR_LMAG)
    ax.plot(xfo, yfo, color=C["far"], marker="^", ms=4,
            linestyle="-.", label="Far-field only")
    _mark_optimum(ax, 1.0, _FAR_LMAG[1.0], C["far"])

    ax.axhline(_DNN_MAE, color=C["dnn"], lw=0.9,
               linestyle=":", label=f"Pure DNN ({_DNN_MAE:.3f})")

    ax.set_xlabel(r"$\lambda_\mathrm{mag}$")
    ax.set_ylabel(r"Test MAE ($M_\mathrm{w}$)")
    ax.set_xlim(0.0, 3.2)
    ax.set_ylim(0.05, 0.27)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    despine(ax)
    add_panel_label(ax, "(a)", x=-0.16, y=1.06)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [_star_handle()],
              labels=labels + ["Optimum (per mode)"],
              loc="upper right", frameon=False, fontsize=6.5,
              handlelength=1.8, handletextpad=0.4)

    _save(fig, output_path)
    return output_path


# ── λ_synth sensitivity ─────────────────────────────────────────────────────

def plot_lambda_synth(output_path: Path = OUT_LAMBDA_SYNTH) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.6))
    fig.subplots_adjust(**_AXES_RECT)

    opt_full_x, opt_full_y = 0.7, _FULL_LSYN[0.7]
    opt_simp_x, opt_simp_y = 0.1, _SIMP_LSYN[0.1]

    xf, yf = _dict_to_xy(_FULL_LSYN)
    ax.plot(xf, yf, color=C["full"], marker="o", ms=3.5,
            label=r"Full radiation ($\lambda_\mathrm{mag}=2.0$)")

    xs, ys = _dict_to_xy(_SIMP_LSYN)
    ax.plot(xs, ys, color=C["simpl"], marker="s", ms=3.5,
            linestyle="--", label=r"Simplified radiation ($\lambda_\mathrm{mag}=2.0$)")

    ax.axhline(_DNN_MAE, color=C["dnn"], lw=0.9,
               linestyle=":", label=f"Pure DNN ({_DNN_MAE:.3f})")

    _mark_optimum(ax, opt_full_x, opt_full_y, C["full"])
    _mark_optimum(ax, opt_simp_x, opt_simp_y, C["simpl"])

    ax.axvline(opt_full_x, color=C["full"], lw=0.7, linestyle="--",
               alpha=0.5, zorder=1)
    ax.axvline(opt_simp_x, color=C["simpl"], lw=0.7, linestyle="--",
               alpha=0.5, zorder=1)

    ax.text(opt_full_x + 0.03, opt_full_y - 0.005,
            r"$\lambda^{*}=0.7$", color=C["full"],
            fontsize=6.5, va="top", ha="left")
    ax.text(opt_simp_x + 0.03, opt_simp_y - 0.004,
            r"$\lambda^{*}=0.1$", color=C["simpl"],
            fontsize=6.5, va="top", ha="left")

    ax.set_xlabel(r"$\lambda_\mathrm{synth}$")
    ax.set_ylabel(r"Test MAE ($M_\mathrm{w}$)")
    ax.set_xlim(0.0, 1.1)
    ax.set_ylim(0.07, 0.175)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
    despine(ax)
    add_panel_label(ax, "(b)", x=-0.16, y=1.06)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [_star_handle()],
              labels=labels + ["Optimum (per mode)"],
              loc="upper left", frameon=False, fontsize=6.5,
              handlelength=1.8, handletextpad=0.4, ncol=2,
              columnspacing=1.0)

    _save(fig, output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="λ sensitivity figures")
    p.add_argument("--out-mag", default=str(OUT_LAMBDA_MAG))
    p.add_argument("--out-synth", default=str(OUT_LAMBDA_SYNTH))
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    plot_lambda_mag(Path(args.out_mag))
    plot_lambda_synth(Path(args.out_synth))
