"""E1.4 Bootstrap 批量脚本

自动遍历 outputs_experiments 下所有配置 × 三个阈值（cm0/cm1/cm2）的
station_predictions.csv，对每组跑 bootstrap 不确定性量化，
输出汇总 CSV 至 paper/result_ana/e1_4_bootstrap_ci.csv。

用法:
    python scripts/experiments/run_bootstrap_e1_4.py
    python scripts/experiments/run_bootstrap_e1_4.py --n-bootstrap 2000
    python scripts/experiments/run_bootstrap_e1_4.py --exp-root outputs_experiments
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bootstrap import bootstrap_event_metrics, load_station_predictions

# 阈值子目录名 → 标签
THRESHOLD_DIRS = {
    "unseen_8events_cm0": "cm0",
    "unseen_8events_cm1": "cm1",
    "unseen_8events_cm2": "cm2",
}

# 实验目录 → 规范化名称（保持与 e1_results_summary.md 一致）
EXP_DIR_RENAME: dict[str, str] = {
    "pure_dnn_simplified": "pure_dnn",
    "ablation_far_only":   "far_only",
    "ablation_far_P_only": "far_P_only",
    "ablation_far_S_only": "far_S_only",
    "ablation_int_only":   "int_only",
    "ablation_full":       "ablation_full",
}


def discover_csvs(exp_root: Path) -> list[tuple[str, str, Path]]:
    """遍历 exp_root，返回 (experiment_name, threshold_label, csv_path) 三元组列表"""
    entries: list[tuple[str, str, Path]] = []
    for results_dir in sorted(exp_root.glob("*/results")):
        for config_dir in sorted(results_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            raw_name = config_dir.name
            exp_name = EXP_DIR_RENAME.get(raw_name, raw_name)
            for thr_dir, thr_label in THRESHOLD_DIRS.items():
                csv_path = config_dir / thr_dir / "station_predictions.csv"
                if csv_path.exists():
                    entries.append((exp_name, thr_label, csv_path))
    return entries


def run_bootstrap_all(
    exp_root: Path,
    output_csv: Path,
    n_bootstrap: int,
    seed: int,
    agg: str,
) -> None:
    entries = discover_csvs(exp_root)
    if not entries:
        print(f"未找到任何 station_predictions.csv，请检查路径: {exp_root}")
        return

    print(f"发现 {len(entries)} 组 (配置 × 阈值)，开始 bootstrap（n={n_bootstrap}）...\n")

    rows: list[dict] = []
    for i, (exp_name, thr_label, csv_path) in enumerate(entries, 1):
        print(f"[{i:>3d}/{len(entries)}] {exp_name:30s} | {thr_label} ...", end=" ", flush=True)
        try:
            station_rows = load_station_predictions(csv_path)
            result = bootstrap_event_metrics(
                station_rows, n_bootstrap=n_bootstrap, seed=seed, agg=agg
            )
            mae  = result["mae"]
            rmse = result["rmse"]
            bias = result["bias"]
            row = {
                "experiment":   exp_name,
                "threshold":    thr_label,
                "n_events":     result["n_events"],
                "n_stations":   len(station_rows),
                "n_bootstrap":  n_bootstrap,
                "aggregation":  agg,
                # MAE
                "mae_mean":     round(mae["mean"],     4),
                "mae_std":      round(mae["std"],      4),
                "mae_ci_lower": round(mae["ci_lower"], 4),
                "mae_ci_upper": round(mae["ci_upper"], 4),
                # RMSE
                "rmse_mean":    round(rmse["mean"],     4),
                "rmse_std":     round(rmse["std"],      4),
                "rmse_ci_lower":round(rmse["ci_lower"], 4),
                "rmse_ci_upper":round(rmse["ci_upper"], 4),
                # Bias
                "bias_mean":    round(bias["mean"],     4),
                "bias_ci_lower":round(bias["ci_lower"], 4),
                "bias_ci_upper":round(bias["ci_upper"], 4),
                # 来源路径（方便溯源）
                "source_csv":   str(csv_path.relative_to(PROJECT_ROOT)),
            }
            # 按机制追加 MAE
            for mech, mm in result.get("by_mechanism", {}).items():
                row[f"mae_{mech}_mean"]     = round(mm["mae"]["mean"],     4)
                row[f"mae_{mech}_ci_lower"] = round(mm["mae"]["ci_lower"], 4)
                row[f"mae_{mech}_ci_upper"] = round(mm["mae"]["ci_upper"], 4)
                row[f"n_{mech}"]            = mm["n_events"]
            rows.append(row)
            print(f"MAE={mae['mean']:.4f} [{mae['ci_lower']:.4f}, {mae['ci_upper']:.4f}]")
        except Exception as exc:
            print(f"ERROR: {exc}")
            rows.append({
                "experiment": exp_name,
                "threshold":  thr_label,
                "error":      str(exc),
                "source_csv": str(csv_path.relative_to(PROJECT_ROOT)),
            })

    # 写 CSV（列对齐：先固定列，动态 mech 列追加在后）
    fixed_cols = [
        "experiment", "threshold", "n_events", "n_stations", "n_bootstrap", "aggregation",
        "mae_mean", "mae_std", "mae_ci_lower", "mae_ci_upper",
        "rmse_mean", "rmse_std", "rmse_ci_lower", "rmse_ci_upper",
        "bias_mean", "bias_ci_lower", "bias_ci_upper",
        "source_csv",
    ]
    extra_cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in fixed_cols and k not in extra_cols and k != "error":
                extra_cols.append(k)
    all_cols = fixed_cols + sorted(extra_cols) + ["error"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n完成。结果已写入: {output_csv}")
    print(f"共 {len(rows)} 行（{sum(1 for r in rows if 'error' not in r or not r['error'])} 成功，"
          f"{sum(1 for r in rows if r.get('error'))} 失败）")


def main() -> None:
    parser = argparse.ArgumentParser(description="E1.4 Bootstrap 批量脚本")
    parser.add_argument(
        "--exp-root",
        default=str(PROJECT_ROOT / "outputs_experiments"),
        help="实验根目录（默认: outputs_experiments）",
    )
    parser.add_argument(
        "--output-csv",
        default=str(PROJECT_ROOT / "paper" / "result_ana" / "e1_4_bootstrap_ci.csv"),
        help="输出 CSV 路径",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="重采样次数（默认 1000）")
    parser.add_argument("--seed",        type=int, default=42,   help="随机种子")
    parser.add_argument(
        "--agg",
        choices=["median", "mean"],
        default="median",
        help="事件级聚合方式（默认 median）",
    )
    args = parser.parse_args()

    run_bootstrap_all(
        exp_root=Path(args.exp_root),
        output_csv=Path(args.output_csv),
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        agg=args.agg,
    )


if __name__ == "__main__":
    main()
