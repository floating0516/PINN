#!/usr/bin/env python
"""对 outputs_experiments 下所有模型批量跑 8 个未见事件的评估。

结果输出到：outputs_experiments/<group>/results/<run_id>/unseen_8events/
汇总 CSV：outputs_experiments/<group>/results/unseen_8events_summary.csv

用法：
    python scripts/evaluation/batch_unseen_8events.py
    python scripts/evaluation/batch_unseen_8events.py --resume   # 跳过已完成
    python scripts/evaluation/batch_unseen_8events.py --threshold 2  # 指定阈值(cm)
"""
from __future__ import annotations

import argparse
import csv
import datetime
import math
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

GNSS_DATA_ROOT = Path("/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/GNSS_EQDATA")

UNSEEN_8_EVENTS = [
    GNSS_DATA_ROOT / "iquique-aftershock-2014-chile",
    GNSS_DATA_ROOT / "nepal-aftershock-2015",
    GNSS_DATA_ROOT / "kodiak-2018-alaska",
    GNSS_DATA_ROOT / "samos-2020-greece",
    GNSS_DATA_ROOT / "luding-2022-china",
    GNSS_DATA_ROOT / "xizang-2025-southern-tibetan-plateau",
    GNSS_DATA_ROOT / "myanmar-2025-mandalay",
    GNSS_DATA_ROOT / "sand-point-2025-alaska",
]

EXPERIMENT_GROUPS = [
    {
        "name": "phase1",
        "models_root": PROJECT_ROOT / "outputs_experiments" / "phase1" / "models",
        "results_root": PROJECT_ROOT / "outputs_experiments" / "phase1" / "results",
        "exp_summary": PROJECT_ROOT / "outputs_experiments" / "phase1" / "results" / "experiment_summary.csv",
    },
    {
        "name": "e1_1",
        "models_root": PROJECT_ROOT / "outputs_experiments" / "e1_1" / "models",
        "results_root": PROJECT_ROOT / "outputs_experiments" / "e1_1" / "results",
        "exp_summary": PROJECT_ROOT / "outputs_experiments" / "e1_1" / "results" / "experiment_summary.csv",
    },
    {
        "name": "e1_2",
        "models_root": PROJECT_ROOT / "outputs_experiments" / "e1_2" / "models",
        "results_root": PROJECT_ROOT / "outputs_experiments" / "e1_2" / "results",
        "exp_summary": PROJECT_ROOT / "outputs_experiments" / "e1_2" / "results" / "experiment_summary.csv",
    },
    {
        "name": "e1_3",
        "models_root": PROJECT_ROOT / "outputs_experiments" / "e1_3" / "models",
        "results_root": PROJECT_ROOT / "outputs_experiments" / "e1_3" / "results",
        "exp_summary": PROJECT_ROOT / "outputs_experiments" / "e1_3" / "results" / "experiment_summary.csv",
    },
]


def discover_model_dirs(models_root: Path) -> list[Path]:
    dirs = []
    if not models_root.is_dir():
        return dirs
    for child in sorted(models_root.iterdir()):
        if child.is_dir() and (child / "best_model.pth").exists() and (child / "config.yaml").exists():
            dirs.append(child)
    return dirs


def get_exp_name(run_id: str, exp_summary: Path) -> str:
    if not exp_summary.exists():
        return run_id
    with open(exp_summary, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if run_id in str(row.get("models_dir", "")):
                return row.get("experiment", run_id)
    return run_id


def load_completed(summary_csv: Path) -> set[str]:
    if not summary_csv.exists():
        return set()
    completed = set()
    with open(summary_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("error"):
                completed.add(str(row.get("run_id", "")))
    return completed


def append_summary_row(row: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量评估 8 个未见事件")
    parser.add_argument("--resume", action="store_true", help="跳过已完成的模型")
    parser.add_argument("--threshold", type=float, default=2.0, help="radial_peak_min_cm 阈值（默认 2.0 cm）")
    args = parser.parse_args()

    # 验证事件目录存在
    missing = [str(p) for p in UNSEEN_8_EVENTS if not p.exists()]
    if missing:
        print(f"警告: 以下事件目录不存在: {missing}")
    event_dirs = [p for p in UNSEEN_8_EVENTS if p.exists()]
    print(f"有效事件目录数: {len(event_dirs)}")

    from src.evaluation.evaluate_unseen import evaluate_unseen_events

    total_models = sum(
        len(discover_model_dirs(g["models_root"])) for g in EXPERIMENT_GROUPS
    )
    print(f"总模型数: {total_models}，阈值: {args.threshold} cm\n")

    done = 0
    for group in EXPERIMENT_GROUPS:
        group_name = group["name"]
        models_root = group["models_root"]
        results_root = group["results_root"]
        exp_summary = group["exp_summary"]
        thr_tag = f"cm{int(args.threshold)}" if args.threshold == int(args.threshold) else f"cm{args.threshold}"
        summary_csv = results_root / f"unseen_8events_{thr_tag}_summary.csv"

        model_dirs = discover_model_dirs(models_root)
        if not model_dirs:
            print(f"[{group_name}] 未找到模型，跳过")
            continue

        completed = load_completed(summary_csv) if args.resume else set()

        print(f"{'='*60}")
        print(f"实验组: {group_name}，模型数: {len(model_dirs)}")

        for model_dir in model_dirs:
            done += 1
            run_id = model_dir.name
            exp_name = get_exp_name(run_id, exp_summary)

            if args.resume and run_id in completed:
                print(f"  ⏭ [{done}/{total_models}] 跳过 (已完成): {exp_name}")
                continue

            output_dir = results_root / run_id / f"unseen_8events_{thr_tag}"
            print(f"\n  [{done}/{total_models}] {group_name}/{exp_name}")
            print(f"    model : {model_dir.name}")
            print(f"    output: {output_dir}")

            error_msg = ""
            n_events = n_stations = 0
            mae = rmse = bias = float("nan")

            try:
                result = evaluate_unseen_events(
                    event_dirs=[str(p) for p in event_dirs],
                    model_dir=str(model_dir),
                    output_dir=str(output_dir),
                    radial_peak_min_cm_override=args.threshold,
                )
                event_rows = result.get("event_rows", [])
                station_rows = result.get("station_rows", [])
                n_events = len(event_rows)
                n_stations = len(station_rows)
                if event_rows:
                    errors = [float(r["error"]) for r in event_rows]
                    abs_errors = [abs(e) for e in errors]
                    mae = sum(abs_errors) / len(abs_errors)
                    rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
                    bias = sum(errors) / len(errors)
                print(f"    ✓ {n_events} 事件, {n_stations} 台站 | MAE={mae:.4f}, RMSE={rmse:.4f}, Bias={bias:+.4f}")
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                print(f"    ✗ 出错: {error_msg}")
                traceback.print_exc()

            append_summary_row(
                {
                    "group": group_name,
                    "experiment": exp_name,
                    "run_id": run_id,
                    "threshold_cm": args.threshold,
                    "n_events": n_events,
                    "n_stations": n_stations,
                    "mae": mae,
                    "rmse": rmse,
                    "bias": bias,
                    "output_dir": str(output_dir),
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    "error": error_msg,
                },
                summary_csv,
            )

    print(f"\n{'─'*60}")
    print("全部完成。")
    for g in EXPERIMENT_GROUPS:
        csv_path = g["results_root"] / f"unseen_8events_{thr_tag}_summary.csv"
        if csv_path.exists():
            print(f"  {g['name']}: {csv_path}")


if __name__ == "__main__":
    main()
