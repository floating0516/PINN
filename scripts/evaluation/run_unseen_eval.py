#!/usr/bin/env python
r"""批量 unseen 评估脚本 — 对已训练模型执行多阈值 evaluate_unseen

功能：
  1. 扫描指定 models 目录下所有已训练模型（含 best_model.pth + config.yaml）
  2. 对每个模型 × 每个 radial_peak_min_cm 阈值执行 evaluate_unseen
  3. 结果保存至 results/<run_id>/unseen_<threshold_label>/ 目录

用法示例：

  # 对 phase1 所有模型执行 0cm / 1cm / 2cm 三种阈值评估
  python scripts/evaluation/run_unseen_eval.py \
    --models-root ./outputs_experiments/phase1/models \
    --results-root ./outputs_experiments/phase1/results \
    --event-data-root "/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/GNSS_EQDATA" \
    --thresholds 0 1 2 \
    --resume

  # 仅对特定模型目录评估
  python scripts/evaluation/run_unseen_eval.py \
      --model-dir ./outputs_experiments/phase1/models/20260412_135249 \
      --results-root ./outputs_experiments/phase1/results \
      --event-data-root /Users/lihe/Library/Mobile\ Documents/com~apple~CloudDocs/GNSS_EQDATA \
      --thresholds 0 1 2

  # 仅跑全台站（0cm）
  python scripts/evaluation/run_unseen_eval.py \
      --models-root ./outputs_experiments/phase1/models \
      --results-root ./outputs_experiments/phase1/results \
      --event-data-root /Users/lihe/Library/Mobile\ Documents/com~apple~CloudDocs/GNSS_EQDATA \
      --thresholds 0
"""
from __future__ import annotations

import argparse
import csv
import datetime
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def discover_model_dirs(models_root: Path) -> list[Path]:
    """扫描 models_root 下所有含 best_model.pth + config.yaml 的子目录"""
    dirs: list[Path] = []
    if not models_root.is_dir():
        return dirs
    for child in sorted(models_root.iterdir()):
        if child.is_dir() and (child / "best_model.pth").exists() and (child / "config.yaml").exists():
            dirs.append(child)
    return dirs


def discover_event_dirs(event_data_root: Path) -> list[Path]:
    """扫描事件数据根目录，返回含 event.json 的子目录"""
    dirs: list[Path] = []
    if not event_data_root.is_dir():
        return dirs
    for child in sorted(event_data_root.iterdir()):
        if child.is_dir() and (child / "event.json").exists():
            dirs.append(child)
    return dirs


def threshold_label(threshold_cm: float) -> str:
    if threshold_cm <= 0:
        return "all_stations"
    return f"{threshold_cm:.0f}cm"


def run_unseen_eval_single(
    *,
    model_dir: Path,
    event_dirs: list[Path],
    output_dir: Path,
    threshold_cm: float,
) -> dict[str, Any]:
    """对单个模型 × 单个阈值执行 evaluate_unseen"""
    from src.evaluation.evaluate_unseen import evaluate_unseen_events

    override = threshold_cm if threshold_cm > 0 else 0.0
    result = evaluate_unseen_events(
        event_dirs=[str(d) for d in event_dirs],
        model_dir=str(model_dir),
        output_dir=str(output_dir),
        radial_peak_min_cm_override=override,
    )
    return result


def append_unseen_summary_csv(
    row: dict[str, Any],
    csv_path: Path,
) -> None:
    """追加一行结果到汇总 CSV"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
        all_fields = list(dict.fromkeys(existing_fields + list(row.keys())))
        rows_existing = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows_existing = list(csv.DictReader(f))
        rows_existing.append(row)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows_existing)
    else:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)


def load_completed_unseen(csv_path: Path) -> set[str]:
    """读取已完成的 (run_id, threshold) 组合"""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    completed: set[str] = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row.get('run_id', '')}|{row.get('threshold_label', '')}"
            error = row.get("error", "")
            if not error:
                completed.add(key)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量 unseen 评估：对已训练模型执行多阈值 evaluate_unseen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--models-root",
        type=str,
        help="模型根目录，自动扫描所有含 best_model.pth 的子目录",
    )
    group.add_argument(
        "--model-dir",
        type=str,
        action="append",
        help="指定单个或多个模型目录（可多次使用）",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        required=True,
        help="结果根目录（输出保存至 <results-root>/<run_id>/unseen_<threshold>/）",
    )
    parser.add_argument(
        "--event-data-root",
        type=str,
        required=True,
        help="unseen 事件数据根目录（含多个事件子目录，每个有 event.json）",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0],
        help="radial_peak_min_cm 阈值列表（默认: 0 1 2）",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="汇总 CSV 路径（默认: <results-root>/unseen_eval_summary.csv）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过已完成的 (run_id, threshold) 组合",
    )
    args = parser.parse_args()

    # 发现模型目录
    if args.models_root:
        model_dirs = discover_model_dirs(Path(args.models_root))
        if not model_dirs:
            print(f"错误: 未在 {args.models_root} 下找到模型目录")
            sys.exit(1)
    else:
        model_dirs = [Path(d) for d in args.model_dir]
        for d in model_dirs:
            if not (d / "best_model.pth").exists():
                print(f"警告: {d} 下未找到 best_model.pth")

    # 发现事件目录
    event_dirs = discover_event_dirs(Path(args.event_data_root))
    if not event_dirs:
        print(f"错误: 未在 {args.event_data_root} 下找到事件目录")
        sys.exit(1)

    results_root = Path(args.results_root)
    summary_csv = Path(args.summary_csv) if args.summary_csv else results_root / "unseen_eval_summary.csv"

    # 断点续传
    completed = set()
    if args.resume:
        completed = load_completed_unseen(summary_csv)
        if completed:
            print(f"✔ 断点续传: 已跳过 {len(completed)} 个已完成评估")

    print(f"模型数: {len(model_dirs)}")
    print(f"事件数: {len(event_dirs)}")
    print(f"阈值:   {args.thresholds}")
    total = len(model_dirs) * len(args.thresholds)
    print(f"总评估次数: {total}")
    print()

    done = 0
    for i, model_dir in enumerate(model_dirs, 1):
        run_id = model_dir.name
        # 从 experiment_summary.csv 查找实验名（如果存在）
        exp_name = run_id
        exp_summary = results_root / "experiment_summary.csv"
        if exp_summary.exists():
            with open(exp_summary, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if run_id in str(row.get("models_dir", "")):
                        exp_name = row.get("experiment", run_id)
                        break

        for threshold_cm in args.thresholds:
            t_label = threshold_label(threshold_cm)
            key = f"{run_id}|{t_label}"
            done += 1

            if args.resume and key in completed:
                print(f"⏭ [{done}/{total}] 跳过: {exp_name} / {t_label}")
                continue

            output_dir = results_root / run_id / f"unseen_{t_label}"
            print(f"\n{'=' * 60}")
            print(f"[{done}/{total}] {exp_name} | threshold={t_label}")
            print(f"  model_dir : {model_dir}")
            print(f"  output_dir: {output_dir}")
            print(f"{'=' * 60}")

            error_msg = ""
            n_events = 0
            n_stations = 0
            try:
                result = run_unseen_eval_single(
                    model_dir=model_dir,
                    event_dirs=event_dirs,
                    output_dir=output_dir,
                    threshold_cm=threshold_cm,
                )
                n_events = len(result.get("event_rows", []))
                n_stations = len(result.get("station_rows", []))

                # 汇总指标
                event_rows = result.get("event_rows", [])
                if event_rows:
                    import math
                    errors = [float(r["error"]) for r in event_rows]
                    abs_errors = [abs(e) for e in errors]
                    mae = sum(abs_errors) / len(abs_errors)
                    rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
                    bias = sum(errors) / len(errors)
                else:
                    mae = rmse = bias = float("nan")

                print(f"  ✓ 完成: {n_events} 事件, {n_stations} 台站, MAE={mae:.4f}")

            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                mae = rmse = bias = float("nan")
                print(f"  ✗ 出错: {error_msg}")
                traceback.print_exc()

            row = {
                "experiment": exp_name,
                "run_id": run_id,
                "threshold_cm": threshold_cm,
                "threshold_label": t_label,
                "n_events": n_events,
                "n_stations": n_stations,
                "mae": mae,
                "rmse": rmse,
                "bias": bias,
                "output_dir": str(output_dir),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "error": error_msg,
            }
            append_unseen_summary_csv(row, summary_csv)

    print(f"\n{'─' * 60}")
    print(f"全部完成。汇总 CSV: {summary_csv}")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
