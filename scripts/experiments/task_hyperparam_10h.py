"""
10 小时超参数搜索任务 —— 聚焦 stf_rate_loss 损失函数权重调优

策略概览 (7 阶段，逐步消融 + 联合精调):
  Phase 1  Baseline       — 仅 MSE，确定基准
  Phase 2  lambda_nonneg  — 非负约束消融
  Phase 3  lambda_mag — 震级物理一致性消融
  Phase 4  lambda_shape   — 波形形状约束消融
  Phase 5  lambda_synth — 波动方程消融
  Phase 6  Combined       — 前 5 阶段最优组合 + 随机微扰精调
  Phase 7  Mode switches  — radiation_pattern_mode / skip_travel_delays /
                            include_intermediate_field 开关组合

每阶段保留当前最优损失参数，后续阶段基于已有最优向下搜索。
模型结构 & 优化器超参固定为已知较优值，避免交叉干扰。
"""
import copy
import csv
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs" / "config.yaml"
MODELS_DIR = ROOT / "outputs" / "models"
LOGS_DIR = ROOT / "outputs" / "logs"
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"
TASK_NAME = "hyperparam_task_full"
TASK_DURATION_HOURS = 999
TASK_RANDOM_SEED = 20260317
RUNTIME_STATE_FILE = RESULTS_DIR / f"{TASK_NAME}_runtime_state.json"
EXECUTION_MODE = "task"              # "task" = 执行训练  "monitor" = 仅监控
MONITOR_TARGET_TASK_ID = ""
# ── 断点续跑：填入上次任务目录名，已完成的 run_id 会被自动跳过 ──────────
RESUME_TASK_ID = "hyperparam_task_10h_20260318_104331"  # 留空则全新开始
MONITOR_POLL_SECONDS = 30
MONITOR_STABLE_ROUNDS = 2
MONITOR_WAIT_FOR_TASK = True

# ────────────────────────────────────────────────────────────────────
# 固定的模型结构 & 优化器超参 (不参与本次搜索)
# ────────────────────────────────────────────────────────────────────
FIXED_TRAINING = {
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "dropout": 0.2,
    "hidden_dim": 128,
    "num_layers": 3,
    "num_tcn_blocks": 6,
    "batch_size": 64,
    "scheduler_T0": 15,
    "scheduler_T_mult": 2,
    "warmup_epochs": 5,
    "epochs": 160,
    "patience": 40,
}

# ────────────────────────────────────────────────────────────────────
# 损失参数默认值 (各 phase 在此基础上覆盖)
# ────────────────────────────────────────────────────────────────────
DEFAULT_LOSS = {
    "lambda_MSE": 1.0,
    "lambda_nonneg": 0.0,
    "lambda_mag": 0.0,
    "lambda_shape": 0.0,
    "lambda_synth": 0.0,
    "include_intermediate_field": True,
    "radiation_pattern_mode": "simplified",
    "skip_travel_delays": True,
}


# ═════════════════════════════════════════════════════════════════════
# Phase 定义
# ═════════════════════════════════════════════════════════════════════

def _build_phase_configs() -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    返回 (run_id, phase_name, loss_overrides) 列表。
    loss_overrides 中的键会覆盖 DEFAULT_LOSS 对应值。
    """
    cfgs: List[Tuple[str, str, Dict[str, Any]]] = []

    # ── Phase 1: Baseline (仅 MSE) ──────────────────────────────────
    cfgs.append(("P1_base", "P1_baseline", {}))

    # ── Phase 2: lambda_nonneg 消融 ─────────────────────────────────
    for val in [0.1, 0.3, 0.5, 0.8, 1.0]:
        cfgs.append((
            f"P2_nn{val:.1f}".replace(".", ""),
            "P2_nonneg",
            {"lambda_nonneg": val},
        ))

    # ── Phase 3: lambda_mag 消融 ────────────────────────────────
    for val in [0.05, 0.1, 0.2, 0.4, 0.6]:
        cfgs.append((
            f"P3_ph{val:.2f}".replace(".", ""),
            "P3_physics",
            {"lambda_mag": val},
        ))

    # ── Phase 4: lambda_shape 消融 ──────────────────────────────────
    for val in [0.05, 0.1, 0.2, 0.3, 0.5]:
        cfgs.append((
            f"P4_sh{val:.2f}".replace(".", ""),
            "P4_shape",
            {"lambda_shape": val},
        ))

    # ── Phase 5: lambda_synth 消融 ───────────────────────────────────
    for wv in [0.02, 0.05, 0.1, 0.2, 0.5]:
        cfgs.append((
            f"P5_wv{wv:.2f}".replace(".", ""),
            "P5_wave",
            {"lambda_synth": wv},
        ))

    # Phase 6 & 7 在运行时动态构建 (依赖前面阶段的最优结果)
    return cfgs


def _build_phase6_configs(
    best_per_phase: Dict[str, Dict[str, Any]],
    rng: random.Random,
    n_random: int = 6,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Phase 6: 取各阶段最优 lambda 组合 + 随机微扰精调。
    """
    best_nonneg = best_per_phase.get("P2_nonneg", {}).get("lambda_nonneg", 0.5)
    best_physics = best_per_phase.get("P3_physics", {}).get("lambda_mag", 0.2)
    best_shape = best_per_phase.get("P4_shape", {}).get("lambda_shape", 0.1)
    best_wave = best_per_phase.get("P5_wave", {}).get("lambda_synth", 0.0)

    base_combo = {
        "lambda_nonneg": best_nonneg,
        "lambda_mag": best_physics,
        "lambda_shape": best_shape,
        "lambda_synth": best_wave,
    }

    cfgs: List[Tuple[str, str, Dict[str, Any]]] = []
    # 精确最优组合
    cfgs.append(("P6_best", "P6_combined", dict(base_combo)))

    # 随机微扰
    for idx in range(1, n_random + 1):
        perturbed = {}
        for key, val in base_combo.items():
            scale = rng.uniform(0.7, 1.3)
            perturbed[key] = round(max(0.0, val * scale), 4)
        cfgs.append((f"P6_rnd{idx:02d}", "P6_combined", perturbed))

    return cfgs


def _build_phase7_configs(
    best_loss_params: Dict[str, Any],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Phase 7: 在最优损失权重基础上测试模式开关组合。
    """
    switches = [
        {"include_intermediate_field": True,  "radiation_pattern_mode": "simplified", "skip_travel_delays": True},
        {"include_intermediate_field": True,  "radiation_pattern_mode": "simplified", "skip_travel_delays": False},
        {"include_intermediate_field": False, "radiation_pattern_mode": "simplified", "skip_travel_delays": True},
        {"include_intermediate_field": True,  "radiation_pattern_mode": "full",       "skip_travel_delays": True},
        {"include_intermediate_field": True,  "radiation_pattern_mode": "full",       "skip_travel_delays": False},
        {"include_intermediate_field": False, "radiation_pattern_mode": "full",       "skip_travel_delays": True},
    ]
    cfgs: List[Tuple[str, str, Dict[str, Any]]] = []
    for idx, sw in enumerate(switches, 1):
        combo = dict(best_loss_params)
        combo.update(sw)
        tag_parts = []
        tag_parts.append("int" if sw["include_intermediate_field"] else "noint")
        tag_parts.append(sw["radiation_pattern_mode"][:4])
        tag_parts.append("skipT" if sw["skip_travel_delays"] else "fullT")
        cfgs.append((f"P7_{'_'.join(tag_parts)}", "P7_modes", combo))
    return cfgs


# ═════════════════════════════════════════════════════════════════════
# 配置 I/O
# ═════════════════════════════════════════════════════════════════════

def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_config(config: Dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True, sort_keys=False)


def apply_config(
    template: Dict[str, Any],
    loss_overrides: Dict[str, Any],
    run_seed: int,
) -> Dict[str, Any]:
    """将固定训练超参 + 损失权重覆盖项写入配置。"""
    config = copy.deepcopy(template)
    training_cfg = config["training"]
    model_cfg = config["model"]
    stf_cfg = training_cfg["stf_rate_loss"]

    # 固定训练 / 模型超参
    training_cfg["learning_rate"] = FIXED_TRAINING["lr"]
    training_cfg["weight_decay"] = FIXED_TRAINING["weight_decay"]
    training_cfg["loss_name"] = "stf_rate"
    training_cfg["pretrain_path"] = None
    training_cfg["epochs"] = FIXED_TRAINING["epochs"]
    training_cfg["early_stop_patience"] = FIXED_TRAINING["patience"]
    training_cfg["early_stop_min_delta"] = 1.0e-4
    training_cfg["batch_size"] = FIXED_TRAINING["batch_size"]
    training_cfg["warmup_epochs"] = FIXED_TRAINING["warmup_epochs"]
    training_cfg["grad_clip_norm"] = 1.0
    training_cfg["scheduler_T0"] = FIXED_TRAINING["scheduler_T0"]
    training_cfg["scheduler_T_mult"] = FIXED_TRAINING["scheduler_T_mult"]
    training_cfg["swa_start"] = max(20, int(FIXED_TRAINING["epochs"] * 0.35))
    training_cfg["random_seed"] = int(run_seed)
    model_cfg["hidden_dim"] = FIXED_TRAINING["hidden_dim"]
    model_cfg["num_layers"] = FIXED_TRAINING["num_layers"]
    model_cfg["num_tcn_blocks"] = FIXED_TRAINING["num_tcn_blocks"]
    model_cfg["dropout"] = FIXED_TRAINING["dropout"]

    # 损失权重：先写默认值，再覆盖
    merged_loss = dict(DEFAULT_LOSS)
    merged_loss.update(loss_overrides)
    stf_cfg["lambda_MSE"] = float(merged_loss["lambda_MSE"])
    stf_cfg["lambda_nonneg"] = float(merged_loss["lambda_nonneg"])
    stf_cfg["lambda_mag"] = float(merged_loss["lambda_mag"])
    stf_cfg["lambda_shape"] = float(merged_loss["lambda_shape"])
    stf_cfg["lambda_synth"] = float(merged_loss["lambda_synth"])
    stf_cfg["include_intermediate_field"] = bool(merged_loss["include_intermediate_field"])
    stf_cfg["radiation_pattern_mode"] = str(merged_loss["radiation_pattern_mode"])
    stf_cfg["skip_travel_delays"] = bool(merged_loss["skip_travel_delays"])

    return config


def find_latest_training_log() -> Path | None:
    log_files = sorted(LOGS_DIR.glob("training_log_*.csv"), key=lambda item: item.stat().st_mtime)
    if not log_files:
        return None
    return log_files[-1]


def get_best_val_loss_from_log(log_file: Path | None) -> Tuple[float, int, int]:
    if log_file is None or not log_file.exists():
        return float("inf"), -1, 0
    best_val_loss = float("inf")
    best_epoch = -1
    total_epochs = 0
    with open(log_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            total_epochs += 1
            val_loss = float(row.get("Val_Loss", "inf"))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = int(row.get("Epoch", -1))
    return best_val_loss, best_epoch, total_epochs


def evaluate_model(config: Dict[str, Any]) -> float:
    from src.data.data_loader import get_data_loaders
    from src.models.model import PINNModel
    from src.training.loss_stf_rate import STFRateWaveformLoss
    from src.training.physics import PhysicsLoss

    model_path = MODELS_DIR / "best_model.pth"
    if not model_path.exists():
        return float("inf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINNModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    _, _, test_loader = get_data_loaders(config)
    criterion = PhysicsLoss(config).to(device)
    loss_fn = STFRateWaveformLoss(config).to(device)
    errors: List[float] = []
    with torch.no_grad():
        for batch in test_loader:
            radial = batch["radial"].to(device)
            stf_true = batch.get("stf", None)
            magnitude = batch.get("magnitude", None)
            dt_value = float(batch["dt"].mean().item())
            if stf_true is not None:
                stf_true = stf_true.to(device)
            if magnitude is not None:
                magnitude = magnitude.to(device)
            pred_log = model(radial)
            pred_dot_m0 = loss_fn._decode_rate(pred_log)
            pred_mw = criterion.utils.magnitude_from_rate(pred_dot_m0, dt_value)
            true_mw = criterion.utils.magnitude_from_rate(stf_true, dt_value) if stf_true is not None else magnitude
            if true_mw is None:
                continue
            diff = torch.abs(pred_mw.view(-1) - true_mw.view(-1))
            valid_mask = torch.isfinite(diff)
            if valid_mask.any():
                errors.extend(diff[valid_mask].cpu().tolist())
    if not errors:
        return float("inf")
    return float(np.mean(np.array(errors, dtype=np.float64)))


def write_trial_records(records: List[Dict[str, Any]], output_csv: Path) -> None:
    fields = [
        "run_id",
        "phase",
        "run_start",
        "run_end",
        "elapsed_sec",
        "test_mae",
        "best_val_loss",
        "best_epoch",
        "total_epochs",
        "train_log_file",
        "status",
        "error",
        "lambda_MSE",
        "lambda_nonneg",
        "lambda_mag",
        "lambda_shape",
        "lambda_synth",
        "include_intermediate_field",
        "radiation_pattern_mode",
        "skip_travel_delays",
        "random_seed",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fields})


def save_runtime_state(payload: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload_with_time = dict(payload)
    payload_with_time["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(RUNTIME_STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(payload_with_time, file, ensure_ascii=False, indent=2)


def update_figures(records: List[Dict[str, Any]], task_dir: Path) -> None:
    valid = [item for item in records if item.get("status") == "ok" and np.isfinite(item.get("test_mae", np.inf))]
    if not valid:
        return
    plt.rcParams["font.sans-serif"] = ["SimSun"]
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False

    # ── 图 1: 进度曲线 (按 phase 着色) ──────────────────────────────
    phase_colors = {
        "P1_baseline": "#1f77b4",
        "P2_nonneg": "#ff7f0e",
        "P3_physics": "#2ca02c",
        "P4_shape": "#d62728",
        "P5_smooth_wave": "#9467bd",
        "P6_combined": "#8c564b",
        "P7_modes": "#e377c2",
    }
    trial_indices = np.arange(1, len(valid) + 1, dtype=np.int32)
    mae_values = np.array([float(item["test_mae"]) for item in valid], dtype=np.float64)
    best_curve = np.minimum.accumulate(mae_values)

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    for idx, item in enumerate(valid):
        color = phase_colors.get(item.get("phase", ""), "#333333")
        ax1.scatter(idx + 1, item["test_mae"], color=color, s=36, zorder=3)
    ax1.plot(trial_indices, best_curve, linewidth=2.0, color="black", label="Best MAE")
    # 图例 (仅出现过的 phase)
    seen_phases = sorted({item.get("phase", "") for item in valid})
    for ph in seen_phases:
        c = phase_colors.get(ph, "#333333")
        ax1.scatter([], [], color=c, s=36, label=ph)
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Test MAE")
    ax1.set_title("Loss Hyperparam Search Progress")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=7, ncol=2)
    fig1.tight_layout()
    fig1.savefig(task_dir / "task_progress_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig1)

    # ── 图 2: 各 lambda 散点图 ──────────────────────────────────────
    loss_keys = [
        "lambda_MSE", "lambda_nonneg", "lambda_mag",
        "lambda_shape", "lambda_synth",
    ]
    fig2, axes = plt.subplots(2, 3, figsize=(16, 9))
    for index, key in enumerate(loss_keys):
        axis = axes[index // 3, index % 3]
        for item in valid:
            x_val = float(item.get(key, 0.0))
            y_val = float(item["test_mae"])
            color = phase_colors.get(item.get("phase", ""), "#333333")
            axis.scatter(x_val, y_val, color=color, alpha=0.8, s=30)
        axis.set_xlabel(key)
        axis.set_ylabel("Test MAE")
        axis.set_title(f"{key} vs MAE")
        axis.grid(True, alpha=0.2)
    fig2.suptitle("Loss Weights vs Test MAE", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(task_dir / "task_loss_lambda_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig2)

    # ── 图 3: 各 phase 最优 MAE 柱状图 ─────────────────────────────
    phase_best: Dict[str, float] = {}
    for item in valid:
        ph = item.get("phase", "unknown")
        mae = float(item["test_mae"])
        if ph not in phase_best or mae < phase_best[ph]:
            phase_best[ph] = mae
    if phase_best:
        phases_sorted = sorted(phase_best.keys())
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        bars = ax3.bar(
            range(len(phases_sorted)),
            [phase_best[p] for p in phases_sorted],
            color=[phase_colors.get(p, "#333333") for p in phases_sorted],
        )
        ax3.set_xticks(range(len(phases_sorted)))
        ax3.set_xticklabels(phases_sorted, rotation=30, ha="right", fontsize=8)
        ax3.set_ylabel("Best Test MAE")
        ax3.set_title("Best MAE per Phase")
        ax3.grid(True, axis="y", alpha=0.25)
        for bar, ph in zip(bars, phases_sorted):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{phase_best[ph]:.4f}", ha="center", va="bottom", fontsize=7)
        fig3.tight_layout()
        fig3.savefig(task_dir / "task_phase_best_bar.png", dpi=220, bbox_inches="tight")
        plt.close(fig3)


def read_last_log_metrics(log_file: Path | None) -> Dict[str, Any]:
    if log_file is None or (not log_file.exists()):
        return {"epoch": -1, "val_loss": float("inf"), "lr": float("inf")}
    last_row: Dict[str, str] | None = None
    with open(log_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            last_row = row
    if last_row is None:
        return {"epoch": -1, "val_loss": float("inf"), "lr": float("inf")}
    return {
        "epoch": int(last_row.get("Epoch", -1)),
        "val_loss": float(last_row.get("Val_Loss", "inf")),
        "lr": float(last_row.get("LR", "inf")),
    }


def resolve_monitor_task_dir() -> Path | None:
    if MONITOR_TARGET_TASK_ID:
        explicit = RESULTS_DIR / MONITOR_TARGET_TASK_ID
        if explicit.exists():
            return explicit
        return None
    if RUNTIME_STATE_FILE.exists():
        with open(RUNTIME_STATE_FILE, "r", encoding="utf-8") as file:
            payload = json.load(file)
        task_id = str(payload.get("task_id", "")).strip()
        if task_id:
            runtime_dir = RESULTS_DIR / task_id
            if runtime_dir.exists():
                return runtime_dir
    candidates = sorted(
        [item for item in RESULTS_DIR.glob(f"{TASK_NAME}_*") if item.is_dir()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def read_trial_summary(summary_csv: Path) -> List[Dict[str, str]]:
    if not summary_csv.exists():
        return []
    rows: List[Dict[str, str]] = []
    with open(summary_csv, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    return rows


def build_monitor_snapshot(task_dir: Path) -> Dict[str, Any]:
    summary_csv = task_dir / "trial_summary.csv"
    report_file = task_dir / "task_report.txt"
    rows = read_trial_summary(summary_csv)
    valid_rows = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            metric = float(row.get("test_mae", "inf"))
        except ValueError:
            continue
        if np.isfinite(metric):
            valid_rows.append((metric, row))
    valid_rows.sort(key=lambda item: item[0])
    best_mae = valid_rows[0][0] if valid_rows else float("inf")
    best_run_id = valid_rows[0][1].get("run_id", "") if valid_rows else ""
    failed_count = sum(1 for row in rows if row.get("status") == "failed")
    latest_log = find_latest_training_log()
    live_metrics = read_last_log_metrics(latest_log)
    # 各 phase 最优
    phase_best: Dict[str, Tuple[float, str]] = {}
    for _, row in valid_rows:
        ph = row.get("phase", "?")
        mae = float(row.get("test_mae", "inf"))
        if ph not in phase_best or mae < phase_best[ph][0]:
            phase_best[ph] = (mae, row.get("run_id", ""))
    last_phase = rows[-1].get("phase", "") if rows else ""
    return {
        "task_id": task_dir.name,
        "trial_count": len(rows),
        "failed_count": failed_count,
        "best_mae": best_mae,
        "best_run_id": best_run_id,
        "last_run_id": rows[-1].get("run_id", "") if rows else "",
        "last_phase": last_phase,
        "phase_best": {k: v[0] for k, v in phase_best.items()},
        "is_finished": report_file.exists(),
        "latest_epoch": live_metrics["epoch"],
        "latest_val_loss": live_metrics["val_loss"],
        "latest_lr": live_metrics["lr"],
    }


def _extract_best_per_phase(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """从已完成记录中提取每个 phase 的最优损失参数。"""
    best_per_phase: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        if rec.get("status") != "ok":
            continue
        mae = rec.get("test_mae", float("inf"))
        if not np.isfinite(mae):
            continue
        ph = rec.get("phase", "")
        if ph not in best_per_phase or mae < best_per_phase[ph].get("test_mae", float("inf")):
            best_per_phase[ph] = dict(rec)
    return best_per_phase


def monitor_task() -> None:
    task_dir = resolve_monitor_task_dir()
    while task_dir is None and MONITOR_WAIT_FOR_TASK:
        print("未检测到任务目录，继续等待任务启动...")
        time.sleep(MONITOR_POLL_SECONDS)
        task_dir = resolve_monitor_task_dir()
    if task_dir is None:
        print("未找到可监控任务目录。")
        return
    print(f"开始持续监控: {task_dir}")
    last_signature = ""
    stable_rounds = 0
    while True:
        snapshot = build_monitor_snapshot(task_dir)
        signature = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        if signature != last_signature:
            best_text = f"{snapshot['best_mae']:.4f}" if np.isfinite(snapshot["best_mae"]) else "NA"
            val_text = f"{snapshot['latest_val_loss']:.4f}" if np.isfinite(snapshot["latest_val_loss"]) else "NA"
            lr_text = f"{snapshot['latest_lr']:.2e}" if np.isfinite(snapshot["latest_lr"]) else "NA"
            phase_text = snapshot.get("last_phase", "?")
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"phase={phase_text} trial={snapshot['trial_count']} "
                f"failed={snapshot['failed_count']} "
                f"best={best_text}({snapshot['best_run_id']}) "
                f"last={snapshot['last_run_id']} epoch={snapshot['latest_epoch']} "
                f"val={val_text} lr={lr_text} finished={snapshot['is_finished']}"
            )
            # 显示各 phase 最优
            phase_best = snapshot.get("phase_best", {})
            if phase_best:
                parts = [f"{k}={v:.4f}" for k, v in sorted(phase_best.items())]
                print(f"  phase_best: {' | '.join(parts)}")
            last_signature = signature
            stable_rounds = 0
        else:
            stable_rounds += 1
        if snapshot["is_finished"] and stable_rounds >= MONITOR_STABLE_ROUNDS:
            print("监控结束：检测到任务已完成。")
            break
        time.sleep(MONITOR_POLL_SECONDS)


def _run_single_trial(
    run_id: str,
    phase_name: str,
    loss_overrides: Dict[str, Any],
    template_config: Dict[str, Any],
    trial_seed: int,
    train_fn,
) -> Dict[str, Any]:
    """执行单次训练试验并返回记录字典。"""
    run_start = datetime.now()
    run_config = apply_config(template_config, loss_overrides, run_seed=trial_seed)
    save_config(run_config)

    pre_log = find_latest_training_log()
    best_model_path = MODELS_DIR / "best_model.pth"
    if best_model_path.exists():
        best_model_path.unlink()

    status = "ok"
    error_message = ""
    try:
        train_fn()
    except Exception as error:
        status = "failed"
        error_message = str(error)

    post_log = find_latest_training_log()
    run_end = datetime.now()
    elapsed = (run_end - run_start).total_seconds()
    if pre_log is not None and post_log == pre_log:
        post_log = None

    best_val_loss, best_epoch, total_epochs = get_best_val_loss_from_log(post_log)
    test_mae = float("inf")
    if status == "ok":
        try:
            test_mae = evaluate_model(run_config)
        except Exception as error:
            status = "failed"
            error_message = str(error)

    # 合并损失参数到记录
    merged_loss = dict(DEFAULT_LOSS)
    merged_loss.update(loss_overrides)

    record: Dict[str, Any] = {
        "run_id": run_id,
        "phase": phase_name,
        "run_start": run_start.strftime("%Y-%m-%d %H:%M:%S"),
        "run_end": run_end.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(elapsed, 2),
        "test_mae": float(test_mae),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "total_epochs": int(total_epochs),
        "train_log_file": post_log.name if post_log is not None else "",
        "status": status,
        "error": error_message,
        "lambda_MSE": float(merged_loss["lambda_MSE"]),
        "lambda_nonneg": float(merged_loss["lambda_nonneg"]),
        "lambda_mag": float(merged_loss["lambda_mag"]),
        "lambda_shape": float(merged_loss["lambda_shape"]),
        "lambda_synth": float(merged_loss["lambda_synth"]),
        "include_intermediate_field": bool(merged_loss["include_intermediate_field"]),
        "radiation_pattern_mode": str(merged_loss["radiation_pattern_mode"]),
        "skip_travel_delays": bool(merged_loss["skip_travel_delays"]),
        "random_seed": int(trial_seed),
    }
    return record


def main() -> None:
    from src.training.train import train

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    task_start = datetime.now()
    deadline = task_start + timedelta(hours=TASK_DURATION_HOURS)
    task_id = f"{TASK_NAME}_{task_start.strftime('%Y%m%d_%H%M%S')}"
    task_dir = RESULTS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    progress_jsonl = task_dir / "trial_records.jsonl"
    summary_csv = task_dir / "trial_summary.csv"
    final_report = task_dir / "task_report.txt"

    template_config = load_config()
    rng = random.Random(TASK_RANDOM_SEED)
    all_records: List[Dict[str, Any]] = []
    global_best_mae = float("inf")
    global_best_model = task_dir / "best_model_task.pth"
    run_index = 0

    # ── 断点续跑：从上次任务加载已有记录 ──────────────────────────────
    completed_run_ids: set = set()
    if RESUME_TASK_ID:
        prev_jsonl = RESULTS_DIR / RESUME_TASK_ID / "trial_records.jsonl"
        if prev_jsonl.exists():
            with open(prev_jsonl, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line:
                        continue
                    _rec = json.loads(_line)
                    completed_run_ids.add(_rec.get("run_id", ""))
                    all_records.append(_rec)
                    _mae = _rec.get("test_mae", float("inf"))
                    if _rec.get("status") == "ok" and np.isfinite(_mae) and _mae < global_best_mae:
                        global_best_mae = _mae
            print(f"[续跑] 从 {RESUME_TASK_ID} 加载 {len(completed_run_ids)} 条历史记录，global_best_mae={global_best_mae:.4f}")

    save_runtime_state({"task_id": task_id, "status": "running", "run_id": "", "trial_count": len(all_records)})

    # ── Phase 1–5: 静态配置 ──────────────────────────────────────────
    static_configs = _build_phase_configs()
    phase6_done = False
    phase7_done = False

    def _process_trial(run_id: str, phase_name: str, loss_ov: Dict[str, Any]) -> None:
        nonlocal run_index, global_best_mae
        run_index += 1
        trial_seed = TASK_RANDOM_SEED + run_index

        record = _run_single_trial(
            run_id=run_id,
            phase_name=phase_name,
            loss_overrides=loss_ov,
            template_config=template_config,
            trial_seed=trial_seed,
            train_fn=train,
        )
        all_records.append(record)

        # 保存全局最优模型
        best_model_path = MODELS_DIR / "best_model.pth"
        mae = record["test_mae"]
        if record["status"] == "ok" and np.isfinite(mae) and mae < global_best_mae and best_model_path.exists():
            global_best_mae = mae
            shutil.copy2(best_model_path, global_best_model)

        # 持久化
        save_runtime_state({
            "task_id": task_id,
            "status": "running",
            "run_id": run_id,
            "phase": phase_name,
            "trial_count": len(all_records),
            "last_status": record["status"],
            "best_mae": global_best_mae,
        })
        with open(progress_jsonl, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_trial_records(all_records, summary_csv)
        update_figures(all_records, task_dir)
        print(
            f"[{phase_name}] {run_id} | status={record['status']} | "
            f"test_mae={record['test_mae']:.4f} | best_val={record['best_val_loss']:.4f} | "
            f"elapsed={record['elapsed_sec']:.0f}s"
        )

    try:
        # Phase 1-5
        for run_id, phase_name, loss_ov in static_configs:
            if datetime.now() >= deadline:
                break
            if run_id in completed_run_ids:
                print(f"[跳过] {run_id} ({phase_name}) 已在历史记录中，跳过")
                continue
            _process_trial(run_id, phase_name, loss_ov)

        # ── Phase 6: 联合精调 (动态构建) ─────────────────────────────
        if datetime.now() < deadline:
            best_per_phase = _extract_best_per_phase(all_records)
            phase6_cfgs = _build_phase6_configs(best_per_phase, rng)
            for run_id, phase_name, loss_ov in phase6_cfgs:
                if datetime.now() >= deadline:
                    break
                _process_trial(run_id, phase_name, loss_ov)
            phase6_done = True

        # ── Phase 7: 模式开关组合 (动态构建) ─────────────────────────
        if datetime.now() < deadline:
            best_per_phase = _extract_best_per_phase(all_records)
            # 取全局最优的损失参数作为 Phase 7 基准
            global_best_rec = min(
                [r for r in all_records if r.get("status") == "ok" and np.isfinite(r.get("test_mae", np.inf))],
                key=lambda r: r["test_mae"],
                default=None,
            )
            if global_best_rec is not None:
                best_loss_for_p7 = {
                    k: global_best_rec[k]
                    for k in ["lambda_MSE", "lambda_nonneg", "lambda_mag",
                              "lambda_shape", "lambda_synth"]
                }
            else:
                best_loss_for_p7 = {}
            phase7_cfgs = _build_phase7_configs(best_loss_for_p7)
            for run_id, phase_name, loss_ov in phase7_cfgs:
                if datetime.now() >= deadline:
                    break
                _process_trial(run_id, phase_name, loss_ov)
            phase7_done = True

    finally:
        save_config(template_config)

    # ── 生成最终报告 ─────────────────────────────────────────────────
    valid_records = [
        item for item in all_records
        if item.get("status") == "ok" and np.isfinite(item.get("test_mae", np.inf))
    ]
    valid_records.sort(key=lambda item: item["test_mae"])

    loss_keys = [
        "lambda_MSE", "lambda_nonneg", "lambda_mag",
        "lambda_shape", "lambda_synth",
        "include_intermediate_field", "radiation_pattern_mode", "skip_travel_delays",
    ]

    with open(final_report, "w", encoding="utf-8") as file:
        file.write(f"{'='*80}\n")
        file.write(f"损失函数超参数搜索报告\n")
        file.write(f"{'='*80}\n")
        file.write(f"任务名称: {TASK_NAME}\n")
        file.write(f"任务目录: {task_dir}\n")
        file.write(f"开始时间: {task_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"目标时长(小时): {TASK_DURATION_HOURS}\n")
        file.write(f"实际试验次数: {len(all_records)}\n")
        file.write(f"成功试验次数: {len(valid_records)}\n")
        file.write(f"Phase 6 完成: {phase6_done}  Phase 7 完成: {phase7_done}\n")

        # 各 Phase 最优
        file.write(f"\n{'─'*80}\n各 Phase 最优结果:\n{'─'*80}\n")
        best_per_phase = _extract_best_per_phase(all_records)
        for ph in sorted(best_per_phase.keys()):
            rec = best_per_phase[ph]
            file.write(
                f"  {ph:<18s} run={rec['run_id']:<16s} MAE={rec['test_mae']:.6f} "
                f"val={rec['best_val_loss']:.6f} ep={rec['best_epoch']}\n"
            )
            params_str = " ".join(f"{k}={rec.get(k, '?')}" for k in loss_keys)
            file.write(f"    {params_str}\n")

        # 全局 Top 10
        if valid_records:
            best = valid_records[0]
            file.write(f"\n{'─'*80}\n全局最优试验: {best['run_id']} (Phase: {best.get('phase','')})\n")
            file.write(f"Test MAE: {best['test_mae']:.6f}\n")
            file.write(f"最优损失参数:\n")
            file.write(f"  {json.dumps({k: best.get(k) for k in loss_keys}, ensure_ascii=False, indent=2)}\n")

        file.write(f"\n{'─'*80}\nTop 10 结果:\n{'─'*80}\n")
        for item in valid_records[:10]:
            file.write(
                f"  {item['run_id']:<16s} phase={item.get('phase',''):<16s} "
                f"MAE={item['test_mae']:.6f} val={item['best_val_loss']:.6f} "
                f"ep={item['best_epoch']} "
                f"nn={item.get('lambda_nonneg',0):.2f} "
                f"ph={item.get('lambda_mag',0):.2f} "
                f"sh={item.get('lambda_shape',0):.2f} "
                f"wv={item.get('lambda_synth',0):.2f}\n"
            )

    print(f"任务完成，结果目录: {task_dir}")
    print(f"试验明细: {summary_csv}")
    print(f"任务报告: {final_report}")
    save_runtime_state({
        "task_id": task_id,
        "status": "finished",
        "run_id": valid_records[0]["run_id"] if valid_records else "",
        "trial_count": len(all_records),
        "best_mae": valid_records[0]["test_mae"] if valid_records else float("inf"),
    })


if __name__ == "__main__":
    if EXECUTION_MODE == "monitor":
        monitor_task()
    else:
        main()
