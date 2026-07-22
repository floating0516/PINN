#!/usr/bin/env python
"""E1.5 留一事件交叉验证 (Leave-One-Event-Out Cross-Validation)

对数据集中每个事件依次留出作为测试集，其余训练，评估留出事件上的表现。
汇总所有 fold 的 MAE/RMSE。

用法:
    # 列出所有可用事件
    python scripts/experiments/loeo_cv.py --list-events

    # 运行全部 fold
    python scripts/experiments/loeo_cv.py --output-root ./outputs_experiments/e1_5_loeo

    # 只跑指定事件（按 event_index）
    python scripts/experiments/loeo_cv.py --output-root ./outputs_experiments/e1_5_loeo --event-index 0 3 5
"""
from __future__ import annotations

import argparse
import csv
import datetime
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _list_events(config: dict) -> list[dict[str, Any]]:
    if int(config.get("pipeline_version", 1)) == 2:
        from src.data.loaders_v2 import list_events_v2

        return list_events_v2(config)
    from src.data.data_loader import list_event_indices

    return list_event_indices(config)


def _make_loeo_loaders(
    config: dict,
    *,
    event_index: int,
    event_name: str,
):
    if int(config.get("pipeline_version", 1)) == 2:
        from src.data.loaders_v2 import get_data_loaders_v2

        loeo_config = {**config, "training": {**config["training"]}}
        loeo_config["training"]["split_protocol"] = "loeo"
        return get_data_loaders_v2(
            loeo_config,
            leave_out_event=event_name,
        )
    from src.data.data_loader import get_data_loaders_loeo

    return get_data_loaders_loeo(
        config,
        leave_out_event_index=event_index,
    )


def run_loeo_cv(
    config: dict,
    output_root: Path,
    event_indices: list[int] | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    from src.training.train import train
    from src.evaluation.evaluate import evaluate

    # 获取事件列表
    all_events = _list_events(config)
    if event_indices is not None:
        all_events = [e for e in all_events if e["event_index"] in event_indices]

    print(f"LOEO-CV: 共 {len(all_events)} 个 fold")
    for e in all_events:
        print(f"  event_index={e['event_index']:3d}  {e['event']}")

    if dry_run:
        print("\n[dry-run] 仅列出事件，不执行训练")
        return []

    summary_rows: list[dict[str, Any]] = []
    csv_path = output_root / "loeo_summary.csv"

    for fold_i, ev_info in enumerate(all_events, 1):
        ev_idx = ev_info["event_index"]
        ev_name = ev_info["event"]
        print(f"\n{'#' * 60}")
        print(f"# LOEO Fold {fold_i}/{len(all_events)}: 留出 '{ev_name}' (idx={ev_idx})")
        print(f"{'#' * 60}")

        # 每个 fold 使用独立的输出目录
        fold_config = {**config}
        fold_output = output_root / f"fold_{ev_idx:03d}_{ev_name.replace(' ', '_').replace('/', '_')}"
        fold_config["paths"] = {
            **config["paths"],
            "models_dir": str(fold_output / "models"),
            "logs_dir": str(fold_output / "logs"),
            "results_dir": str(fold_output / "results"),
        }

        row: dict[str, Any] = {
            "fold": fold_i,
            "event_index": ev_idx,
            "event": ev_name,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }

        try:
            # LOEO 数据划分
            loaders = _make_loeo_loaders(
                fold_config,
                event_index=ev_idx,
                event_name=ev_name,
            )

            # 训练
            train_result = train(fold_config, data_loaders=loaders)
            row["run_id"] = train_result.get("run_id", "")
            row["best_val_loss"] = train_result.get("best_val_loss", float("nan"))
            row["best_mw_mae"] = train_result.get("best_mw_mae", float("nan"))

            # 评估：必须在 LOEO 留出事件上评估（注入留出事件的 test_loader），
            # 否则 evaluate() 会重建标准随机划分，得到的是同事件内插而非跨事件外推。
            eval_result = evaluate(
                model_path=train_result["best_model_path"],
                results_run_id=train_result["run_id"],
                config=fold_config,
                save_plots=False,
                show_plots=False,
                save_metrics=False,
                test_loader=loaders[2],
            )
            row["test_mae"] = eval_result.get("mae", float("nan"))
            row["test_rmse"] = eval_result.get("rmse", float("nan"))
            row["test_sample_count"] = eval_result.get("sample_count", 0)
            row["error"] = ""

        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"✗ Fold {fold_i} 失败: {row['error']}")
            traceback.print_exc()

        summary_rows.append(row)

        # 逐步写入 CSV
        _append_csv(row, csv_path)
        print(f"  MAE={row.get('test_mae', 'N/A')}, RMSE={row.get('test_rmse', 'N/A')}")

    # 打印汇总
    if summary_rows:
        _print_summary(summary_rows)

    return summary_rows


def _append_csv(row: dict[str, Any], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 60}")
    print("LOEO-CV 汇总")
    print(f"{'=' * 60}")
    maes = [r["test_mae"] for r in rows if isinstance(r.get("test_mae"), (int, float)) and math.isfinite(r["test_mae"])]
    rmses = [r["test_rmse"] for r in rows if isinstance(r.get("test_rmse"), (int, float)) and math.isfinite(r["test_rmse"])]
    n_ok = len(maes)
    n_fail = len(rows) - n_ok
    if maes:
        print(f"  成功: {n_ok} / {len(rows)} fold")
        print(f"  平均 MAE : {sum(maes)/len(maes):.4f}")
        print(f"  平均 RMSE: {sum(rmses)/len(rmses):.4f}")
        print(f"  MAE 范围 : [{min(maes):.4f}, {max(maes):.4f}]")
    if n_fail:
        print(f"  失败: {n_fail} fold")


def main() -> None:
    parser = argparse.ArgumentParser(description="E1.5 LOEO 交叉验证")
    parser.add_argument("--config", type=str, default=None, help="基础配置文件")
    parser.add_argument("--output-root", type=str, default=None, help="输出根目录（--list-events 时可省略）")
    parser.add_argument("--event-index", type=int, nargs="*", default=None,
                        help="只运行指定 event_index（默认全部）")
    parser.add_argument("--list-events", action="store_true", help="仅列出事件后退出")
    parser.add_argument("--dry-run", action="store_true", help="仅打印计划，不执行")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    if args.list_events:
        events = _list_events(config)
        print(f"共 {len(events)} 个事件:")
        for e in events:
            print(f"  event_index={e['event_index']:3d}  {e['event']}")
        return

    if args.output_root is None:
        parser.error("--output-root is required unless --list-events is given")
    output_root = Path(args.output_root)
    run_loeo_cv(
        config=config,
        output_root=output_root,
        event_indices=args.event_index,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
