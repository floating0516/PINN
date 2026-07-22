#!/usr/bin/env python
"""批量实验执行器 — E0.1 实验配置管理脚本

功能：
  1. 接收 YAML override 参数，自动深度合并到基础 config
  2. 触发 train → evaluate → evaluate_unseen 流水线
  3. 将结果汇总至统一 CSV（outputs/results/experiment_summary.csv）

用法示例：

  # 单次实验，覆盖 lambda_mag
  python scripts/experiments/run_experiment.py \\
      --set training.stf_rate_loss.lambda_mag=0.1

  # 批量网格搜索
  python scripts/experiments/run_experiment.py \\
      --set training.stf_rate_loss.lambda_mag=0.1 \\
      --set training.stf_rate_loss.radiation_pattern_mode=simplified \\
      --name "lp01_simplified"

  # 仅预览合并后的配置（不训练）
  python scripts/experiments/run_experiment.py \\
      --set training.stf_rate_loss.lambda_mag=0.1 \\
      --dry-run

  # 跳过 evaluate_unseen 步骤
  python scripts/experiments/run_experiment.py --skip-unseen

  # 从 YAML 文件读取多组实验配置，顺序执行
  python scripts/experiments/run_experiment.py --sweep sweep.yaml
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.provenance import write_json

# ── 配置操作工具 ──────────────────────────────────────────────────────────

def load_base_config(path: Path | None = None) -> dict:
    """加载基础配置文件"""
    if path is None:
        path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict) -> dict:
    """递归深度合并两个字典，override 中的值优先"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_dot_notation(key: str, value: str) -> dict:
    """将 'a.b.c=v' 形式的 key-value 转换为嵌套字典

    自动推断类型：bool / int / float / null / str
    """
    parsed_value = _infer_type(value)
    parts = key.split(".")
    result: dict = {}
    current = result
    for part in parts[:-1]:
        current[part] = {}
        current = current[part]
    current[parts[-1]] = parsed_value
    return result


def _infer_type(value: str) -> Any:
    """推断字符串值的 Python 类型"""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """将 --set 参数列表应用到配置字典上

    每个 override 格式为 'dotted.key=value'
    """
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override 格式错误（缺少 '='）: {item}")
        key, value = item.split("=", 1)
        patch = parse_dot_notation(key.strip(), value.strip())
        result = deep_merge(result, patch)
    return result


def config_diff(base: dict, modified: dict, prefix: str = "") -> list[str]:
    """对比两个配置字典，返回变更说明列表"""
    diffs: list[str] = []
    all_keys = set(list(base.keys()) + list(modified.keys()))
    for key in sorted(all_keys):
        full_key = f"{prefix}.{key}" if prefix else key
        v_old = base.get(key)
        v_new = modified.get(key)
        if isinstance(v_old, dict) and isinstance(v_new, dict):
            diffs.extend(config_diff(v_old, v_new, full_key))
        elif v_old != v_new:
            diffs.append(f"  {full_key}: {v_old!r} → {v_new!r}")
    return diffs


# ── 流水线步骤 ────────────────────────────────────────────────────────────

def step_train(config: dict) -> dict[str, Any]:
    """执行训练步骤"""
    from src.training.train import train
    print("\n" + "=" * 60)
    print("▶ 步骤 1/3：训练")
    print("=" * 60)
    return train(config)


def step_evaluate(config: dict, train_result: dict) -> dict[str, Any]:
    """执行测试集评估步骤"""
    from src.evaluation.evaluate import evaluate
    print("\n" + "=" * 60)
    print("▶ 步骤 2/3：测试集评估")
    print("=" * 60)
    model_path = train_result.get("best_model_path")
    run_id = train_result.get("run_id", "eval")
    return evaluate(
        model_path=model_path,
        results_run_id=run_id,
        config=config,
        save_plots=True,
        show_plots=False,
        save_metrics=True,
    )


def step_evaluate_unseen(
    config: dict,
    train_result: dict,
    unseen_event_dirs: list[str | Path],
) -> dict[str, Any]:
    """执行未见事件评估步骤"""
    from src.evaluation.evaluate_unseen import evaluate_unseen_events
    print("\n" + "=" * 60)
    print("▶ 步骤 3/3：未见事件评估")
    print("=" * 60)
    models_dir = train_result["models_dir"]
    results_dir = Path(train_result["results_dir"])
    output_dir = results_dir / "unseen"
    result = evaluate_unseen_events(
        event_dirs=[str(d) for d in unseen_event_dirs],
        model_dir=str(models_dir),
        output_dir=str(output_dir),
    )
    return result


# ── 指标汇总 ──────────────────────────────────────────────────────────────

def _mechanism_label(code: int | str) -> str:
    mapping = {0: "normal", 1: "strike_slip", 2: "thrust"}
    if isinstance(code, str):
        return code
    return mapping.get(int(code), f"mech_{code}")


def _compute_unseen_metrics(unseen_result: dict) -> dict[str, Any]:
    """从 evaluate_unseen 结果提取汇总指标"""
    event_rows = unseen_result.get("event_rows", [])
    station_rows = unseen_result.get("station_rows", [])
    if not event_rows:
        return {}

    shared_metrics = unseen_result.get("metrics", {}) or {}
    metrics: dict[str, Any] = {
        "unseen_n_events": len(event_rows),
        "unseen_n_stations": len(station_rows),
        "unseen_event_mae_catalog": shared_metrics.get(
            "event_mae", float("nan")
        ),
        "unseen_event_rmse_catalog": shared_metrics.get(
            "event_rmse", float("nan")
        ),
        "unseen_event_bias_catalog": shared_metrics.get(
            "event_bias", float("nan")
        ),
    }

    # 按机制分类统计
    by_mech: dict[str, list[float]] = {}
    for row in event_rows:
        mech = _mechanism_label(row.get("mechanism", "unknown"))
        by_mech.setdefault(mech, []).append(
            float(row["error_vs_catalog"])
        )
    for mech, errs in sorted(by_mech.items()):
        abs_e = [abs(e) for e in errs]
        metrics[f"unseen_{mech}_mae"] = sum(abs_e) / len(abs_e)
        metrics[f"unseen_{mech}_n"] = len(errs)

    return metrics


def _artifact_paths(value: Any) -> list[Path]:
    if isinstance(value, Path):
        return [value]
    if isinstance(value, str):
        path = Path(value)
        return [path] if path.suffix.lower() in {".csv", ".png", ".pdf"} else []
    if isinstance(value, dict):
        paths: list[Path] = []
        for item in value.values():
            paths.extend(_artifact_paths(item))
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for item in value:
            paths.extend(_artifact_paths(item))
        return paths
    return []


def update_result_registry(
    *,
    experiment_name: str,
    overrides: list[str],
    eval_result: dict[str, Any] | None,
    unseen_result: dict[str, Any] | None,
) -> Path | None:
    if not eval_result or not eval_result.get("result_registry_path"):
        return None
    registry_path = Path(eval_result["result_registry_path"])
    with registry_path.open("r", encoding="utf-8") as stream:
        registry = json.load(stream)
    registry["experiment_name"] = experiment_name
    registry["overrides"] = list(overrides)
    if unseen_result:
        external_artifacts = _artifact_paths(
            {
                key: unseen_result.get(key)
                for key in (
                    "station_csv",
                    "event_csv",
                    "station_scatter",
                    "event_summary_figure",
                    "station_panels",
                    "event_mw_figures",
                )
            }
        )
        registry["external_evaluation"] = {
            "metrics": unseen_result.get("metrics", {}),
            "artifacts": [
                str(path) for path in external_artifacts if path.is_file()
            ],
        }
        artifacts = registry.setdefault("artifacts", {})
        csv_paths = artifacts.setdefault("csv", [])
        figure_paths = artifacts.setdefault("figures", [])
        for path in external_artifacts:
            if not path.is_file():
                continue
            target = csv_paths if path.suffix.lower() == ".csv" else figure_paths
            if str(path) not in target:
                target.append(str(path))
    write_json(registry_path, registry)
    return registry_path


def collect_summary_row(
    *,
    experiment_name: str,
    config: dict,
    train_result: dict | None,
    eval_result: dict | None,
    unseen_result: dict | None,
    overrides: list[str],
    error_msg: str | None = None,
) -> dict[str, Any]:
    """汇总一次实验的全部指标为一行字典"""
    row: dict[str, Any] = {
        "experiment": experiment_name,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "overrides": "; ".join(overrides) if overrides else "",
        "error": error_msg or "",
    }

    # 训练指标
    if train_result:
        row["run_id"] = train_result.get("run_id", "")
        row["models_dir"] = str(train_result.get("models_dir", ""))
        row["best_val_loss"] = train_result.get("best_val_loss", float("nan"))
        row["best_mw_mae"] = train_result.get("best_mw_mae", float("nan"))

    # 测试集评估指标
    if eval_result:
        eval_metrics = eval_result.get("metrics", {}) or {}
        row["test_event_mae"] = eval_metrics.get("event_mae", float("nan"))
        row["test_event_rmse"] = eval_metrics.get("event_rmse", float("nan"))
        row["test_event_bias"] = eval_metrics.get("event_bias", float("nan"))
        row["test_station_mae"] = eval_metrics.get("station_mae", float("nan"))
        row["test_station_rmse"] = eval_metrics.get("station_rmse", float("nan"))
        row["test_baseline_mae"] = eval_result.get("baseline_mae", float("nan"))
        row["test_baseline_rmse"] = eval_result.get("baseline_rmse", float("nan"))
        row["test_sample_count"] = eval_result.get("sample_count", 0)
        row["result_registry_path"] = str(
            eval_result.get("result_registry_path", "")
        )

    # Unseen 评估指标
    if unseen_result:
        unseen_metrics = _compute_unseen_metrics(unseen_result)
        row.update(unseen_metrics)

    # 关键超参数快照（方便快速筛选）
    stf_loss_cfg = (config.get("training", {}) or {}).get("stf_rate_loss", {}) or {}
    row["lambda_mag"] = stf_loss_cfg.get("lambda_mag", "")
    row["lambda_synth"] = stf_loss_cfg.get("lambda_synth", "")
    row["lambda_MSE"] = stf_loss_cfg.get("lambda_MSE", "")
    if int(config.get("pipeline_version", 1)) != 2:
        row["lambda_nonneg"] = stf_loss_cfg.get("lambda_nonneg", "")
    row["lambda_shape"] = stf_loss_cfg.get("lambda_shape", "")
    row["radiation_pattern_mode"] = stf_loss_cfg.get("radiation_pattern_mode", "")
    row["include_intermediate_field"] = stf_loss_cfg.get("include_intermediate_field", "")
    row["learning_rate"] = (config.get("training", {}) or {}).get("learning_rate", "")
    row["epochs"] = (config.get("training", {}) or {}).get("epochs", "")
    row["hidden_dim"] = (config.get("model", {}) or {}).get("hidden_dim", "")

    return row


def load_completed_experiments(csv_path: Path) -> set[str]:
    """读取汇总 CSV，返回已成功完成的实验名称集合"""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    completed: set[str] = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("experiment", "")
            error = row.get("error", "")
            if name and not error:
                completed.add(name)
    return completed


def append_summary_csv(row: dict[str, Any], csv_path: Path) -> None:
    """将一行实验结果追加到汇总 CSV"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    # 若文件已有列则读取列名，保证列对齐
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
        # 合并新列（可能有新的 mech 分类指标）
        all_fields = list(dict.fromkeys(existing_fields + list(row.keys())))
        # 需要重写文件
        rows_existing = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows_existing = list(reader)
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

    print(f"已追加实验结果至: {csv_path}")


# ── 单次实验运行 ──────────────────────────────────────────────────────────

def run_single_experiment(
    *,
    base_config: dict,
    overrides: list[str],
    experiment_name: str,
    unseen_event_dirs: list[str | Path],
    skip_unseen: bool,
    summary_csv_path: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """执行一次完整实验流水线"""

    # 1. 应用 override
    config = apply_overrides(base_config, overrides)

    # 2. 打印配置变更
    diffs = config_diff(base_config, config)
    print(f"\n{'=' * 60}")
    print(f"实验: {experiment_name}")
    print(f"{'=' * 60}")
    if diffs:
        print("配置变更:")
        for d in diffs:
            print(d)
    else:
        print("配置变更: 无（使用基础配置）")

    if dry_run:
        print("\n[dry-run] 合并后的配置:")
        print(yaml.dump(config, allow_unicode=True, sort_keys=False, default_flow_style=False))
        return {"status": "dry_run", "config": config}

    # 3. 执行流水线
    train_result = None
    eval_result = None
    unseen_result = None
    error_msg = None

    try:
        # 训练
        train_result = step_train(config)

        # 测试集评估
        eval_result = step_evaluate(config, train_result)

        # 未见事件评估
        if not skip_unseen and unseen_event_dirs:
            unseen_result = step_evaluate_unseen(config, train_result, unseen_event_dirs)
        elif not unseen_event_dirs and not skip_unseen:
            print("\n⚠ 跳过 evaluate_unseen：未指定 unseen_event_dirs")

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        print(f"\n✗ 实验 '{experiment_name}' 出错: {error_msg}")
        traceback.print_exc()

    result_registry_path = update_result_registry(
        experiment_name=experiment_name,
        overrides=overrides,
        eval_result=eval_result,
        unseen_result=unseen_result,
    )

    # 4. 汇总指标
    row = collect_summary_row(
        experiment_name=experiment_name,
        config=config,
        train_result=train_result,
        eval_result=eval_result,
        unseen_result=unseen_result,
        overrides=overrides,
        error_msg=error_msg,
    )
    append_summary_csv(row, summary_csv_path)

    # 5. 打印摘要
    print(f"\n{'─' * 60}")
    print(f"实验 '{experiment_name}' 完成")
    if train_result:
        print(f"  run_id      : {train_result.get('run_id', 'N/A')}")
        print(f"  models_dir  : {train_result.get('models_dir', 'N/A')}")
    if eval_result:
        print(f"  test_MAE    : {eval_result.get('mae', float('nan')):.4f}")
        print(f"  test_RMSE   : {eval_result.get('rmse', float('nan')):.4f}")
    if unseen_result:
        um = _compute_unseen_metrics(unseen_result)
        print(f"  unseen_MAE  : {um.get('unseen_event_mae_catalog', float('nan')):.4f}")
        print(f"  unseen_RMSE : {um.get('unseen_event_rmse_catalog', float('nan')):.4f}")
        print(f"  unseen_bias : {um.get('unseen_event_bias_catalog', float('nan')):.4f}")
    if result_registry_path:
        print(f"  registry    : {result_registry_path}")
    if error_msg:
        print(f"  error       : {error_msg}")
    print(f"{'─' * 60}")

    return {
        "status": "error" if error_msg else "ok",
        "train_result": train_result,
        "eval_result": eval_result,
        "unseen_result": unseen_result,
        "result_registry_path": result_registry_path,
        "summary_row": row,
    }


# ── Sweep 模式 ────────────────────────────────────────────────────────────

def load_sweep_config(sweep_path: Path) -> list[dict[str, Any]]:
    """从 YAML 文件加载多组实验配置

    sweep.yaml 格式示例:

    ```yaml
    experiments:
      - name: lp01_simplified
        set:
          - training.stf_rate_loss.lambda_mag=0.1
          - training.stf_rate_loss.radiation_pattern_mode=simplified

      - name: lp04_full
        set:
          - training.stf_rate_loss.lambda_mag=0.4
          - training.stf_rate_loss.radiation_pattern_mode=full
    ```
    """
    with open(sweep_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    experiments = data.get("experiments", [])
    if not experiments:
        raise ValueError(f"sweep 文件中未找到 'experiments' 列表: {sweep_path}")
    return experiments


# ── CLI 入口 ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PINN 实验批量执行器（E0.1）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="基础配置文件路径（默认: configs/config.yaml）",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="覆盖配置参数（点分路径），可多次使用。例: --set training.epochs=100",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="实验名称（用于 CSV 标识；默认使用时间戳）",
    )
    parser.add_argument(
        "--unseen-event-dir",
        action="append",
        default=[],
        dest="unseen_event_dirs",
        metavar="DIR",
        help="未见事件目录（可多次指定）",
    )
    parser.add_argument(
        "--skip-unseen",
        action="store_true",
        help="跳过 evaluate_unseen 步骤",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="汇总 CSV 路径（默认: outputs/results/experiment_summary.csv）",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Sweep 配置文件路径（YAML），包含多组实验参数",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印合并后的配置，不执行训练",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="实验输出根目录（默认使用 config 中的 paths；\n"
             "指定后自动将 models_dir/logs_dir/results_dir 重定向至此目录下）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续传：跳过汇总 CSV 中已成功完成的实验",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 加载基础配置
    config_path = Path(args.config) if args.config else None
    base_config = load_base_config(config_path)

    # 输出目录隔离：--output-root 会覆盖 config 中的 paths
    if args.output_root:
        output_root = Path(args.output_root)
        base_config["paths"]["models_dir"] = str(output_root / "models")
        base_config["paths"]["logs_dir"] = str(output_root / "logs")
        base_config["paths"]["results_dir"] = str(output_root / "results")
        print(f"输出根目录已重定向至: {output_root}")

    # 汇总 CSV 路径
    if args.summary_csv:
        summary_csv_path = Path(args.summary_csv)
    else:
        results_dir = Path(base_config.get("paths", {}).get("results_dir", "./outputs/results"))
        summary_csv_path = results_dir / "experiment_summary.csv"

    # 合并 CLI 指定的 unseen_event_dirs 与 config 中的配置
    unseen_dirs_from_config = (base_config.get("evaluation", {}) or {}).get("unseen_event_dirs", []) or []
    all_unseen_dirs = args.unseen_event_dirs or unseen_dirs_from_config

    # 断点续传：加载已完成的实验
    completed = set()
    if args.resume:
        completed = load_completed_experiments(summary_csv_path)
        if completed:
            print(f"\u2714 断点续传: 已跳过 {len(completed)} 个已完成实验")

    # Sweep 模式 vs 单次实验
    if args.sweep:
        sweep_experiments = load_sweep_config(Path(args.sweep))
        print(f"Sweep 模式: 共 {len(sweep_experiments)} 组实验")
        for i, exp in enumerate(sweep_experiments, 1):
            exp_name = exp.get("name", f"sweep_{i:03d}")
            if args.resume and exp_name in completed:
                print(f"\n⏭ 跳过已完成: {exp_name} ({i}/{len(sweep_experiments)})")
                continue
            exp_overrides = exp.get("set", [])
            exp_unseen = args.unseen_event_dirs or exp.get("unseen_event_dirs", []) or all_unseen_dirs
            print(f"\n{'#' * 60}")
            print(f"# Sweep {i}/{len(sweep_experiments)}: {exp_name}")
            print(f"{'#' * 60}")
            run_single_experiment(
                base_config=base_config,
                overrides=exp_overrides,
                experiment_name=exp_name,
                unseen_event_dirs=exp_unseen,
                skip_unseen=args.skip_unseen,
                summary_csv_path=summary_csv_path,
                dry_run=args.dry_run,
            )
    else:
        exp_name = args.name or f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_single_experiment(
            base_config=base_config,
            overrides=args.overrides,
            experiment_name=exp_name,
            unseen_event_dirs=all_unseen_dirs,
            skip_unseen=args.skip_unseen,
            summary_csv_path=summary_csv_path,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
