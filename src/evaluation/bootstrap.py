"""E1.4 Bootstrap 不确定性量化

从 evaluate_unseen 输出的 station_predictions.csv 读取台站级预测，
通过 bootstrap 重采样计算事件级和总体指标的 95% 置信区间。

用法:
    python -m src.evaluation.bootstrap \
        --csv outputs_experiments/e1_1/results/.../unseen/station_predictions.csv

    # 对比两组实验结果（计算 p 值）
    python -m src.evaluation.bootstrap \
        --csv results_a/station_predictions.csv \
        --csv-b results_b/station_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_station_predictions(csv_path: str | Path) -> list[dict[str, Any]]:
    """从 station_predictions.csv 读取台站级预测"""
    rows: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "event": str(row["event"]),
                "station": str(row["station"]),
                "mw_true": float(row["mw_true"]),
                "mw_pred": float(row["mw_pred"]),
                "mechanism": str(row.get("mechanism", "")),
            })
    return rows


def bootstrap_event_metrics(
    rows: list[dict[str, Any]],
    n_bootstrap: int = 1000,
    seed: int = 42,
    agg: str = "median",
) -> dict[str, Any]:
    """Bootstrap 重采样计算事件级 MAE/RMSE/Bias 的 95% CI

    步骤：
    1. 对每个事件的台站预测做 bootstrap 重采样
    2. 每次采样取中位数/均值作为事件级 Mw 预测
    3. 汇总所有事件计算 MAE/RMSE/Bias
    4. 重复 n_bootstrap 次，得到指标分布
    """
    rng = np.random.RandomState(seed)

    # 按事件分组
    events: dict[str, dict[str, Any]] = {}
    for row in rows:
        ev = row["event"]
        if ev not in events:
            events[ev] = {
                "mw_true": row["mw_true"],
                "preds": [],
                "mechanism": row["mechanism"],
            }
        events[ev]["preds"].append(row["mw_pred"])

    event_names = sorted(events.keys())
    n_events = len(event_names)
    if n_events == 0:
        return {"error": "无有效事件"}

    agg_fn = np.median if agg == "median" else np.mean

    # Bootstrap
    mae_samples = np.zeros(n_bootstrap)
    rmse_samples = np.zeros(n_bootstrap)
    bias_samples = np.zeros(n_bootstrap)
    event_mw_samples: dict[str, np.ndarray] = {
        ev: np.zeros(n_bootstrap) for ev in event_names
    }

    for b in range(n_bootstrap):
        errors = []
        for ev in event_names:
            preds = np.array(events[ev]["preds"])
            mw_true = events[ev]["mw_true"]
            # 台站级 bootstrap 重采样
            idx = rng.randint(0, len(preds), size=len(preds))
            mw_pred = float(agg_fn(preds[idx]))
            event_mw_samples[ev][b] = mw_pred
            errors.append(mw_pred - mw_true)
        errors_arr = np.array(errors)
        mae_samples[b] = np.mean(np.abs(errors_arr))
        rmse_samples[b] = np.sqrt(np.mean(errors_arr ** 2))
        bias_samples[b] = np.mean(errors_arr)

    def _ci(arr: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }

    result: dict[str, Any] = {
        "n_events": n_events,
        "n_bootstrap": n_bootstrap,
        "aggregation": agg,
        "mae": _ci(mae_samples),
        "rmse": _ci(rmse_samples),
        "bias": _ci(bias_samples),
    }

    # 每个事件的 Mw 预测 CI
    event_ci: list[dict[str, Any]] = []
    for ev in event_names:
        ci = _ci(event_mw_samples[ev])
        ci["event"] = ev
        ci["mw_true"] = events[ev]["mw_true"]
        ci["mechanism"] = events[ev]["mechanism"]
        event_ci.append(ci)
    result["event_ci"] = event_ci

    # 按机制分类
    by_mech: dict[str, list[str]] = {}
    for ev in event_names:
        mech = events[ev]["mechanism"]
        by_mech.setdefault(mech, []).append(ev)

    mech_metrics: dict[str, dict[str, Any]] = {}
    for mech, mech_events in sorted(by_mech.items()):
        mech_mae = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            errs = []
            for ev in mech_events:
                pred = event_mw_samples[ev][b]
                true = events[ev]["mw_true"]
                errs.append(pred - true)
            mech_mae[b] = np.mean(np.abs(errs))
        mech_metrics[mech] = {
            "n_events": len(mech_events),
            "mae": _ci(mech_mae),
        }
    result["by_mechanism"] = mech_metrics

    return result


def compare_two_experiments(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    n_bootstrap: int = 1000,
    seed: int = 42,
    agg: str = "median",
) -> dict[str, Any]:
    """比较两组实验结果，计算 MAE 差异的 bootstrap p 值"""
    rng = np.random.RandomState(seed)

    def _group(rows: list[dict[str, Any]]) -> dict[str, dict]:
        events: dict[str, dict] = {}
        for row in rows:
            ev = row["event"]
            if ev not in events:
                events[ev] = {"mw_true": row["mw_true"], "preds": []}
            events[ev]["preds"].append(row["mw_pred"])
        return events

    events_a = _group(rows_a)
    events_b = _group(rows_b)
    common = sorted(set(events_a.keys()) & set(events_b.keys()))

    if not common:
        return {"error": "两组实验无共同事件"}

    agg_fn = np.median if agg == "median" else np.mean

    delta_mae = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        mae_a, mae_b = 0.0, 0.0
        for ev in common:
            preds_a = np.array(events_a[ev]["preds"])
            preds_b = np.array(events_b[ev]["preds"])
            true = events_a[ev]["mw_true"]
            idx_a = rng.randint(0, len(preds_a), size=len(preds_a))
            idx_b = rng.randint(0, len(preds_b), size=len(preds_b))
            mae_a += abs(float(agg_fn(preds_a[idx_a])) - true)
            mae_b += abs(float(agg_fn(preds_b[idx_b])) - true)
        delta_mae[b] = (mae_a - mae_b) / len(common)

    # p 值: A 比 B 差的概率（delta > 0 表示 A 的 MAE 更大）
    p_a_worse = float(np.mean(delta_mae > 0))

    return {
        "n_common_events": len(common),
        "delta_mae_mean": float(np.mean(delta_mae)),
        "delta_mae_ci_lower": float(np.percentile(delta_mae, 2.5)),
        "delta_mae_ci_upper": float(np.percentile(delta_mae, 97.5)),
        "p_value_a_worse": p_a_worse,
    }


def format_results(result: dict[str, Any]) -> str:
    """格式化输出结果"""
    lines: list[str] = []
    lines.append(f"Bootstrap 不确定性量化 (n={result['n_bootstrap']}, agg={result['aggregation']})")
    lines.append(f"事件数: {result['n_events']}")
    lines.append("")
    for metric in ("mae", "rmse", "bias"):
        m = result[metric]
        lines.append(
            f"  {metric.upper():>5s}: {m['mean']:.4f}  "
            f"95% CI [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]"
        )
    lines.append("")
    lines.append("事件级 Mw 预测 CI:")
    for ec in result["event_ci"]:
        lines.append(
            f"  {ec['event']:<30s}  "
            f"true={ec['mw_true']:.2f}  "
            f"pred={ec['mean']:.2f}  "
            f"95% CI [{ec['ci_lower']:.2f}, {ec['ci_upper']:.2f}]"
        )
    if result.get("by_mechanism"):
        lines.append("")
        lines.append("按机制分类 MAE:")
        for mech, mm in result["by_mechanism"].items():
            lines.append(
                f"  {mech:<15s} (n={mm['n_events']}): "
                f"{mm['mae']['mean']:.4f}  "
                f"95% CI [{mm['mae']['ci_lower']:.4f}, {mm['mae']['ci_upper']:.4f}]"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="E1.4 Bootstrap 不确定性量化")
    parser.add_argument("--csv", required=True, help="station_predictions.csv 路径")
    parser.add_argument("--csv-b", default=None, help="第二组实验的 CSV（可选，用于对比）")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="重采样次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--agg", choices=["median", "mean"], default="median", help="事件级聚合方式")
    args = parser.parse_args()

    rows = load_station_predictions(args.csv)
    print(f"已加载 {len(rows)} 条台站预测记录\n")

    result = bootstrap_event_metrics(rows, n_bootstrap=args.n_bootstrap, seed=args.seed, agg=args.agg)
    print(format_results(result))

    if args.csv_b:
        rows_b = load_station_predictions(args.csv_b)
        print(f"\n已加载对比组 {len(rows_b)} 条记录")
        comp = compare_two_experiments(rows, rows_b, n_bootstrap=args.n_bootstrap, seed=args.seed, agg=args.agg)
        print(f"\n对比结果 (A vs B):")
        print(f"  共同事件数: {comp['n_common_events']}")
        print(f"  ΔMAE (A-B): {comp['delta_mae_mean']:.4f}  "
              f"95% CI [{comp['delta_mae_ci_lower']:.4f}, {comp['delta_mae_ci_upper']:.4f}]")
        print(f"  P(A 更差): {comp['p_value_a_worse']:.4f}")


if __name__ == "__main__":
    main()
