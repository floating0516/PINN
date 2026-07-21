"""E1 系列实验结果可视化

生成 4 张图，保存至 tests/figure/：
  e1_fig1_lambda_sensitivity.png   — E1.1 λ_mag 敏感性曲线
  e1_fig2_ablation.png             — E1.3 消融实验 × 三阈值（含 bootstrap CI）
  e1_fig3_forest_plot.png          — 核心配置 bootstrap CI Forest Plot
  e1_fig4_generalization_gap.png   — Test MAE vs Unseen MAE 泛化散点图

运行：
    python scripts/experiments/plot_e1_results.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "tests" / "figure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_CSV = PROJECT_ROOT / "paper" / "result_ana" / "e1_4_bootstrap_ci.csv"

# ── Okabe-Ito 色盲友好调色板 ──────────────────────────────────────────────
C = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
    "gray":   "#999999",
}

# ── 全局样式 ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "DejaVu Sans"],
    "font.size":        8,
    "axes.labelsize":   9,
    "axes.titlesize":   9,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.8,
    "lines.linewidth":  1.4,
    "lines.markersize": 5,
})

# ── 数据 ──────────────────────────────────────────────────────────────────

# E1.1 λ_mag 敏感性（test MAE + unseen cm0 MAE）
LAMBDA_VALS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

E11_TEST = {
    "full":       [0.2229, 0.1269, 0.1212, 0.1084, 0.1026, 0.0987],
    "simplified": [0.2456, 0.2062, 0.1985, 0.1664, 0.1708, 0.1380],
}
E11_CM0 = {
    "full":       [0.2173, 0.1661, 0.1806, 0.1678, 0.1528, 0.1584],
    "simplified": [0.1893, 0.1795, 0.1962, 0.2311, 0.1932, 0.2142],
}
E11_CM2 = {
    "full":       [0.1841, 0.1408, 0.1551, 0.1664, 0.1568, 0.1326],
    "simplified": [0.1954, 0.1825, 0.1691, 0.2005, 0.1524, 0.1463],
}

# E1.3 消融 test MAE
ABLATION_NAMES_DISPLAY = [
    "Full\n(all terms)",
    "Far P+S\n(no interm.)",
    "Far P only",
    "Far S only",
    "Interm. only",
]
ABLATION_KEYS = ["ablation_full", "far_only", "far_P_only", "far_S_only", "int_only"]
ABLATION_TEST_MAE = [0.1212, 0.1115, 0.1027, 0.0883, 0.1197]

# Unseen MAE（来自 e1_results_summary.md 第 4.1 节）
UNSEEN_RAW = {
    # config: (cm0, cm1, cm2)
    "lp010_full":        (0.2173, 0.1906, 0.1841),
    "lp010_simplified":  (0.1893, 0.1712, 0.1954),
    "lp020_full":        (0.1661, 0.1380, 0.1408),
    "lp020_simplified":  (0.1795, 0.1783, 0.1825),
    "lp040_full":        (0.1806, 0.1940, 0.1551),
    "lp040_simplified":  (0.1962, 0.1744, 0.1691),
    "lp060_full":        (0.1678, 0.1807, 0.1664),
    "lp060_simplified":  (0.2311, 0.1915, 0.2005),
    "lp080_full":        (0.1528, 0.1541, 0.1568),
    "lp080_simplified":  (0.1932, 0.1635, 0.1524),
    "lp100_full":        (0.1584, 0.1281, 0.1326),
    "lp100_simplified":  (0.2142, 0.2214, 0.1463),
    "pure_dnn":          (0.2804, 0.2464, 0.2353),
    "ablation_full":     (0.1806, 0.1940, 0.1551),
    "far_only":          (0.2072, 0.1519, 0.1149),
    "far_P_only":        (0.1515, 0.1621, 0.1345),
    "far_S_only":        (0.1676, 0.1724, 0.1302),
    "int_only":          (0.1783, 0.1579, 0.1497),
}
TEST_MAE_ALL = {
    "lp010_full": 0.2229, "lp010_simplified": 0.2456,
    "lp020_full": 0.1269, "lp020_simplified": 0.2062,
    "lp040_full": 0.1212, "lp040_simplified": 0.1985,
    "lp060_full": 0.1084, "lp060_simplified": 0.1664,
    "lp080_full": 0.1026, "lp080_simplified": 0.1708,
    "lp100_full": 0.0987, "lp100_simplified": 0.1380,
    "pure_dnn":          0.1410,
    "ablation_full":     0.1212,
    "far_only":          0.1115,
    "far_P_only":        0.1027,
    "far_S_only":        0.0883,
    "int_only":          0.1197,
}

BASELINE_TEST = 0.2971


def load_bootstrap(csv_path: Path) -> dict[tuple[str, str], dict]:
    """返回 {(experiment, threshold): row_dict}"""
    data: dict[tuple[str, str], dict] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["experiment"], row["threshold"])
            data[key] = {k: float(v) if v and k not in ("experiment","threshold","aggregation","source_csv","error") else v
                         for k, v in row.items()}
    return data


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1: λ_mag 敏感性曲线（E1.1）
# ═══════════════════════════════════════════════════════════════════════════

def fig1_lambda_sensitivity():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)

    datasets = [
        (E11_TEST,  "Test split MAE",     axes[0]),
        (E11_CM0,   "Unseen MAE (cm0, all stations)", axes[1]),
        (E11_CM2,   "Unseen MAE (cm2, ≥2 cm)",        axes[2]),
    ]

    for data, title, ax in datasets:
        ax.plot(LAMBDA_VALS, data["full"], color=C["blue"], marker="o",
                label="Full radiation", zorder=3)
        ax.plot(LAMBDA_VALS, data["simplified"], color=C["orange"], marker="s",
                linestyle="--", label="Simplified", zorder=3)
        ax.axhline(BASELINE_TEST, color=C["gray"], linewidth=0.9,
                   linestyle=":", label=f"Baseline ({BASELINE_TEST:.3f})")
        ax.set_xlabel("λ$_{mag}$")
        ax.set_ylabel("MAE (Mw units)")
        ax.set_title(title)
        ax.set_xticks(LAMBDA_VALS)
        ax.set_xlim(0.05, 1.05)
        ax.set_ylim(0.06, 0.32)

    # 标注全局最优点
    ax0 = axes[0]
    ax0.annotate("Best test\n0.0987", xy=(1.0, 0.0987),
                 xytext=(0.75, 0.14), fontsize=6.5,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=C["blue"]),
                 color=C["blue"])

    axes[0].legend(loc="upper right", frameon=False)
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    # 计算相对降幅注释
    for ax, data in zip(axes, [E11_TEST, E11_CM0, E11_CM2]):
        for i, lm in enumerate(LAMBDA_VALS):
            full_v = data["full"][i]
            simp_v = data["simplified"][i]
            reduction = (simp_v - full_v) / simp_v * 100
            if abs(reduction) > 10:
                ax.annotate(f"−{reduction:.0f}%",
                            xy=(lm, (full_v + simp_v) / 2),
                            fontsize=5.5, ha="center", va="center",
                            color=C["green"], fontweight="bold")

    for i, ax in enumerate(axes):
        ax.text(-0.12, 1.06, "ABC"[i], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")

    fig.suptitle("E1.1  λ$_{mag}$ Sensitivity: Full vs Simplified Radiation Pattern",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = OUTPUT_DIR / "e1_fig1_lambda_sensitivity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2: E1.3 消融实验 × 三阈值（水平条形 + bootstrap CI）
# ═══════════════════════════════════════════════════════════════════════════

def fig2_ablation(bs: dict):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    n_configs = len(ABLATION_KEYS)
    n_thr = 3  # cm0 / cm1 / cm2
    thresholds = ["cm0", "cm1", "cm2"]
    thr_colors = [C["blue"], C["sky"], C["green"]]
    thr_labels = ["cm0 (all stations, n=170)",
                  "cm1 (≥1 cm, n=112)",
                  "cm2 (≥2 cm, n=75)"]

    bar_h = 0.2
    offsets = np.array([-bar_h, 0, bar_h])
    y_pos = np.arange(n_configs)

    for ti, (thr, color, label) in enumerate(zip(thresholds, thr_colors, thr_labels)):
        means, lows, highs = [], [], []
        for key in ABLATION_KEYS:
            row = bs.get((key, thr))
            if row:
                means.append(row["mae_mean"])
                lows.append(row["mae_mean"] - row["mae_ci_lower"])
                highs.append(row["mae_ci_upper"] - row["mae_mean"])
            else:
                means.append(float("nan"))
                lows.append(0)
                highs.append(0)
        y = y_pos + offsets[ti]
        bars = ax.barh(y, means, height=bar_h * 0.85, color=color,
                       alpha=0.85, label=label, zorder=3)
        ax.errorbar(means, y, xerr=[lows, highs],
                    fmt="none", color="black", capsize=3, linewidth=1, zorder=4)

    # test MAE 作为菱形标记叠加在每个 config 右侧
    for i, (key, test_v) in enumerate(zip(ABLATION_KEYS, ABLATION_TEST_MAE)):
        ax.plot(test_v, i, marker="D", color=C["red"], markersize=6,
                zorder=5, label="Test split MAE" if i == 0 else "")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ABLATION_NAMES_DISPLAY, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MAE (Mw units)")
    ax.set_title("E1.3  Physics Term Ablation × Station Threshold (Bootstrap 95% CI)",
                 fontsize=9, fontweight="bold")
    ax.axvline(BASELINE_TEST, color=C["gray"], lw=0.8, linestyle=":", label=f"Baseline ({BASELINE_TEST:.2f})")
    ax.set_xlim(0.05, 0.35)
    ax.grid(axis="x", alpha=0.3, linewidth=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=7)

    # 标注最优 test
    best_idx = ABLATION_TEST_MAE.index(min(ABLATION_TEST_MAE))
    ax.annotate(f"Best test\n{min(ABLATION_TEST_MAE):.4f}",
                xy=(min(ABLATION_TEST_MAE), best_idx),
                xytext=(min(ABLATION_TEST_MAE) + 0.04, best_idx - 0.5),
                fontsize=6.5, color=C["red"],
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C["red"]))

    fig.tight_layout()
    out = OUTPUT_DIR / "e1_fig2_ablation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3: 核心配置 Bootstrap CI Forest Plot（三阈值并列）
# ═══════════════════════════════════════════════════════════════════════════

def fig3_forest_plot(bs: dict):
    # 选核心 8 个配置（含对照组）
    CONFIGS = [
        ("lp100_full",   "lp100 full\n(λ=1.0, all)",  C["blue"]),
        ("lp080_full",   "lp080 full\n(λ=0.8, all)",  C["sky"]),
        ("lp060_full",   "lp060 full\n(λ=0.6, all)",  "#7FCFE8"),
        ("far_S_only",   "far S only\n(λ=0.4)",        C["green"]),
        ("far_P_only",   "far P only\n(λ=0.4)",        C["orange"]),
        ("far_only",     "far P+S\n(λ=0.4)",           C["yellow"]),
        ("int_only",     "interm. only\n(λ=0.4)",      C["purple"]),
        ("pure_dnn",     "Pure DNN\n(no physics)",     C["red"]),
    ]

    thresholds = ["cm0", "cm1", "cm2"]
    thr_titles = ["cm0  All stations (n=170)", "cm1  ≥1 cm (n=112)", "cm2  ≥2 cm (n=75)"]
    n_cfg = len(CONFIGS)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)

    for ti, (thr, title) in enumerate(zip(thresholds, thr_titles)):
        ax = axes[ti]
        y_pos = np.arange(n_cfg)
        for yi, (key, label, color) in enumerate(CONFIGS):
            row = bs.get((key, thr))
            if row:
                mean = row["mae_mean"]
                lo = row["mae_ci_lower"]
                hi = row["mae_ci_upper"]
                ax.errorbar(mean, yi, xerr=[[mean - lo], [hi - mean]],
                            fmt="o", color=color, markersize=6,
                            capsize=4, linewidth=1.2, markeredgecolor="black",
                            markeredgewidth=0.4, zorder=4)
        ax.axvline(BASELINE_TEST, color=C["gray"], lw=0.8, linestyle=":", alpha=0.7)
        ax.set_yticks(y_pos)
        if ti == 0:
            ax.set_yticklabels([cfg[1] for cfg in CONFIGS], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("MAE (Mw)")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlim(0.05, 0.38)
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        ax.text(-0.08 if ti == 0 else -0.04, 1.04, "ABC"[ti],
                transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

    # 在 cm0 面板标注 lp080_full 最窄 CI
    ax0 = axes[0]
    row = bs.get(("lp080_full", "cm0"))
    if row:
        ax0.annotate("Narrowest CI\n(most stable)",
                     xy=(row["mae_mean"], 1),
                     xytext=(row["mae_mean"] + 0.06, 1.8),
                     fontsize=6, color=C["sky"],
                     arrowprops=dict(arrowstyle="->", lw=0.7, color=C["sky"]))

    fig.suptitle("E1.4  Bootstrap 95% CI — Key Configurations × Station Threshold",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = OUTPUT_DIR / "e1_fig3_forest_plot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4: Test MAE vs Unseen MAE 泛化差距散点图
# ═══════════════════════════════════════════════════════════════════════════

def fig4_generalization_gap():
    # 分组着色
    SERIES = {
        "E1.1 full":        (C["blue"],   "o"),
        "E1.1 simplified":  (C["orange"], "s"),
        "E1.2 pure DNN":    (C["red"],    "X"),
        "E1.3 ablation":    (C["green"],  "^"),
    }

    def get_series(key: str) -> str:
        if "simplified" in key:
            return "E1.1 simplified"
        if "pure_dnn" in key:
            return "E1.2 pure DNN"
        if any(x in key for x in ["far_", "int_only", "ablation_"]):
            return "E1.3 ablation"
        return "E1.1 full"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    for ax, (thr_idx, thr_label) in zip(axes, [(0, "cm0  All stations"), (2, "cm2  ≥2 cm")]):
        plotted_series: set[str] = set()
        for key, (cm0, cm1, cm2) in UNSEEN_RAW.items():
            test_v = TEST_MAE_ALL.get(key)
            if test_v is None:
                continue
            unseen_v = [cm0, cm1, cm2][thr_idx]
            series = get_series(key)
            color, marker = SERIES[series]
            label = series if series not in plotted_series else None
            plotted_series.add(series)
            sc = ax.scatter(test_v, unseen_v, color=color, marker=marker,
                            s=55, zorder=4, alpha=0.85,
                            edgecolors="black", linewidths=0.4, label=label)
            # 标注关键点
            if key in ("lp100_full", "far_S_only", "pure_dnn", "lp080_full", "far_only"):
                short = key.replace("ablation_", "")
                ax.annotate(short, (test_v, unseen_v),
                            textcoords="offset points", xytext=(5, 3),
                            fontsize=6, color=color)

        # y=x 等值线（完美泛化）
        lim_min, lim_max = 0.05, 0.32
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=0.8, alpha=0.5,
                label="y = x (perfect generalization)")
        ax.fill_between([lim_min, lim_max], [lim_min, lim_min],
                        [lim_min, lim_max], alpha=0.04, color="green", label="Better unseen")
        ax.fill_between([lim_min, lim_max], [lim_min, lim_max],
                        [lim_max, lim_max], alpha=0.04, color="red", label="Worse unseen")

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("Test split MAE (Mw)")
        ax.set_ylabel(f"Unseen event MAE — {thr_label} (Mw)")
        ax.set_title(f"Generalization Gap ({thr_label})", fontsize=9)
        ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=6.5)

    axes[0].text(-0.12, 1.06, "A", transform=axes[0].transAxes,
                 fontsize=10, fontweight="bold", va="top")
    axes[1].text(-0.12, 1.06, "B", transform=axes[1].transAxes,
                 fontsize=10, fontweight="bold", va="top")

    fig.suptitle("E1  Generalization Gap: Test MAE vs Unseen Event MAE",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = OUTPUT_DIR / "e1_fig4_generalization_gap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("加载 bootstrap CSV ...")
    bs = load_bootstrap(BOOTSTRAP_CSV)
    print(f"  已加载 {len(bs)} 组\n")

    print("绘制 Fig1: λ 敏感性曲线 ...")
    fig1_lambda_sensitivity()

    print("绘制 Fig2: 消融实验 ...")
    fig2_ablation(bs)

    print("绘制 Fig3: Forest Plot ...")
    fig3_forest_plot(bs)

    print("绘制 Fig4: 泛化差距散点图 ...")
    fig4_generalization_gap()

    print(f"\n全部完成，图片保存于: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
