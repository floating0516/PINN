from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "results" / "unseen_events_8_all_stations" / "event_summary.csv"

# ── scientific-visualization skill: publication style ──────────────────────
# Okabe-Ito colorblind-safe palette (4 methods + reference black)
_COLORS = {
    "pinn":    "#0072B2",  # blue
    "crowell": "#D55E00",  # vermillion
    "ruhl":    "#009E73",  # green
    "melgar":  "#CC79A7",  # pink
    "ref":     "#000000",  # black
}

def _apply_pub_style() -> None:
    """Apply publication-quality rcParams (scientific-visualization skill)."""
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          8,
        "axes.labelsize":     9,
        "axes.titlesize":     10,
        "axes.titleweight":   "bold",
        "xtick.labelsize":    7,
        "ytick.labelsize":    7,
        "legend.fontsize":    7,
        "legend.frameon":     False,
        "axes.linewidth":     0.8,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.minor.width":  0.5,
        "ytick.minor.width":  0.5,
        "lines.linewidth":    1.2,
        "patch.linewidth":    0.5,
        "pdf.fonttype":       42,   # editable text in PDF
        "ps.fonttype":        42,
    })
# ───────────────────────────────────────────────────────────────────────────


def default_output_path(csv_path: str | Path) -> Path:
    csv_file = Path(csv_path)
    return csv_file.parent / "unseen_event_method_comparison.png"


def load_event_summary_rows(csv_path: str | Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mw_true = float(row["mw_true"])
            pinn = float(row["mw_pred_median"])
            crowell = float(row["pgd_crowell_mw_pred_median"])
            ruhl = float(row["pgd_ruhl_mw_pred_median"])
            melgar = float(row["pgd_melgar_mw_pred_median"])
            rows.append(
                {
                    "event": str(row["event"]),
                    "mw_true": mw_true,
                    "pinn": pinn,
                    "crowell": crowell,
                    "ruhl": ruhl,
                    "melgar": melgar,
                    "pinn_error": pinn - mw_true,
                    "crowell_error": crowell - mw_true,
                    "ruhl_error": ruhl - mw_true,
                    "melgar_error": melgar - mw_true,
                }
            )
    return rows


def plot_method_comparison(*, csv_path: str | Path, output_path: str | Path) -> Path:
    _apply_pub_style()

    rows = load_event_summary_rows(csv_path)
    if not rows:
        raise ValueError("event_summary.csv 不能为空")

    labels = [str(row["event"]) for row in rows]
    x = np.arange(len(labels), dtype=float)
    width = 0.18

    # Double-column width: 183 mm ≈ 7.2 inch; height ratio 3:1.4
    fig, (ax_top, ax_err) = plt.subplots(
        2, 1,
        figsize=(7.2, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.4], "hspace": 0.0},
    )

    # ── subtle y-grid only, no top/right spines ─────────────────────────
    for ax in (ax_top, ax_err):
        ax.set_facecolor("white")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#CCCCCC", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_linewidth(0.8)

    true_vals    = np.asarray([float(row["mw_true"])       for row in rows], dtype=float)
    pinn_vals    = np.asarray([float(row["pinn"])           for row in rows], dtype=float)
    crowell_vals = np.asarray([float(row["crowell"])        for row in rows], dtype=float)
    ruhl_vals    = np.asarray([float(row["ruhl"])           for row in rows], dtype=float)
    melgar_vals  = np.asarray([float(row["melgar"])         for row in rows], dtype=float)

    pinn_err    = np.asarray([float(row["pinn_error"])    for row in rows], dtype=float)
    crowell_err = np.asarray([float(row["crowell_error"]) for row in rows], dtype=float)
    ruhl_err    = np.asarray([float(row["ruhl_error"])    for row in rows], dtype=float)
    melgar_err  = np.asarray([float(row["melgar_error"])  for row in rows], dtype=float)

    # ── panel A: Mw bars + reference line ───────────────────────────────
    ax_top.bar(x - 1.5*width, pinn_vals,    width=width, label="PINN",        color=_COLORS["pinn"],    zorder=3)
    ax_top.bar(x - 0.5*width, crowell_vals, width=width, label="PGD-Crowell", color=_COLORS["crowell"], zorder=3)
    ax_top.bar(x + 0.5*width, ruhl_vals,    width=width, label="PGD-Ruhl",    color=_COLORS["ruhl"],    zorder=3)
    ax_top.bar(x + 1.5*width, melgar_vals,  width=width, label="PGD-Melgar",  color=_COLORS["melgar"],  zorder=3)
    ax_top.plot(x, true_vals, color=_COLORS["ref"], linewidth=1.4,
                marker="o", markersize=3.5, label="Reference $M_w$", zorder=4)

    ax_top.set_ylabel("$M_w$")
    ax_top.legend(ncol=3, loc="upper right",
                  handlelength=1.2, handletextpad=0.4, columnspacing=0.8,
                  borderpad=0.3)

    all_vals = np.concatenate([true_vals, pinn_vals, crowell_vals, ruhl_vals, melgar_vals])
    lo = np.floor(np.nanmin(all_vals) * 10.0) / 10.0 - 0.1
    hi = np.ceil(np.nanmax(all_vals)  * 10.0) / 10.0 + 0.1
    ax_top.set_ylim(lo, hi)

    # Panel label A
    ax_top.text(-0.06, 1.04, "A", transform=ax_top.transAxes,
                fontsize=10, fontweight="bold", va="top")

    # ── panel B: error bars ──────────────────────────────────────────────
    ax_err.bar(x - 1.5*width, pinn_err,    width=width, color=_COLORS["pinn"],    zorder=3)
    ax_err.bar(x - 0.5*width, crowell_err, width=width, color=_COLORS["crowell"], zorder=3)
    ax_err.bar(x + 0.5*width, ruhl_err,    width=width, color=_COLORS["ruhl"],    zorder=3)
    ax_err.bar(x + 1.5*width, melgar_err,  width=width, color=_COLORS["melgar"],  zorder=3)
    ax_err.axhline(0.0,  color="black",   linewidth=0.9, zorder=4)
    ax_err.axhline( 0.3, color="#999999", linewidth=0.7, linestyle="--", zorder=2)
    ax_err.axhline(-0.3, color="#999999", linewidth=0.7, linestyle="--", zorder=2)
    ax_err.set_ylabel(r"Error ($\Delta M_w$)")
    ax_err.set_xlabel("Event")
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(labels, rotation=20, ha="right")

    err_vals = np.concatenate([pinn_err, crowell_err, ruhl_err, melgar_err, [0.3, -0.3]])
    err_lim  = np.ceil((np.max(np.abs(err_vals)) + 0.05) * 10.0) / 10.0
    ax_err.set_ylim(-err_lim, err_lim)

    # Panel label B
    ax_err.text(-0.06, 1.08, "B", transform=ax_err.transAxes,
                fontsize=10, fontweight="bold", va="top")

    # ── export: 300 DPI PNG ──────────────────────────────────────────────
    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="绘制 8 个未见事件的 PINN 与三种 PGD 标度律对比图")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT), help="event_summary.csv 路径")
    parser.add_argument("--output", default=None, help="输出图片路径，默认写到输入 CSV 同目录")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = args.output if args.output else default_output_path(args.input_csv)
    saved = plot_method_comparison(csv_path=args.input_csv, output_path=output_path)
    print(f"已保存: {saved}")


if __name__ == "__main__":
    main()
