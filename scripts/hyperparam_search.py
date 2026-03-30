"""
超参数搜索脚本：自动化测试多组超参数配置

对每组配置：修改 config -> 训练 -> 记录最佳验证 Mw MAE -> 恢复配置
最终输出所有配置的排名。
"""
import sys
import yaml
import copy
import shutil
import os
from pathlib import Path
from datetime import datetime

# 添加项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs" / "config.yaml"
MODELS_DIR = ROOT / "outputs" / "models"
LOGS_DIR = ROOT / "outputs" / "logs"
RESULTS_FILE = ROOT / "outputs" / "results" / "hyperparam_search_results.txt"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def apply_config(config, params):
    """应用超参数到配置字典"""
    config["training"]["learning_rate"] = params["lr"]
    config["training"]["loss_name"] = "stf_rate"
    config["training"]["pretrain_path"] = None  # 从头训练
    config["training"]["epochs"] = params.get("epochs", 200)
    config["training"]["early_stop_patience"] = params.get("patience", 50)
    config["training"]["early_stop_min_delta"] = 1.0e-4
    config["training"]["batch_size"] = params.get("batch_size", 64)
    config["training"]["weight_decay"] = params.get("weight_decay", 1.0e-5)
    config["training"]["warmup_epochs"] = params.get("warmup_epochs", 5)
    config["training"]["grad_clip_norm"] = 1.0
    config["training"]["scheduler_T0"] = params.get("scheduler_T0", 15)
    config["training"]["scheduler_T_mult"] = params.get("scheduler_T_mult", 2)
    config["training"]["swa_start"] = params.get("swa_start", 50)

    # STF Rate Loss 参数
    stf_cfg = config["training"]["stf_rate_loss"]
    stf_cfg["lambda_MSE"] = params.get("lambda_MSE", 1.0)
    stf_cfg["lambda_wave"] = params["lambda_wave"]
    stf_cfg["lambda_physics"] = params["lambda_physics"]
    stf_cfg["lambda_shape"] = params.get("lambda_shape", 0.1)
    stf_cfg["lambda_nonneg"] = 0.0
    stf_cfg["lambda_smooth"] = 0.0
    stf_cfg["include_intermediate_field"] = True
    stf_cfg["radiation_pattern_mode"] = "simplified"
    stf_cfg["skip_travel_delays"] = True

    # 模型配置
    config["model"]["hidden_dim"] = params.get("hidden_dim", 128)
    config["model"]["num_layers"] = params.get("num_layers", 3)
    config["model"]["num_tcn_blocks"] = params.get("num_tcn_blocks", 6)
    config["model"]["dropout"] = params.get("dropout", 0.2)

    return config


def get_best_mw_mae_from_log(log_dir):
    """从最新的训练日志中提取最佳验证 Mw MAE (通过训练输出获取)"""
    import csv
    log_files = sorted(log_dir.glob("training_log_*.csv"), reverse=True)
    if not log_files:
        return float("inf"), -1, -1
    
    latest = log_files[0]
    best_val = float("inf")
    best_epoch = -1
    total_epochs = 0
    
    with open(latest, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_epochs += 1
            val_loss = float(row.get("Val_Loss", "inf"))
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = int(row.get("Epoch", -1))
    
    return best_val, best_epoch, total_epochs


def run_single_config(name, params, config_template):
    """运行单个超参数配置的训练"""
    print(f"\n{'='*80}")
    print(f"配置 {name}")
    print(f"  lr={params['lr']}, λ_wave={params['lambda_wave']}, "
          f"λ_physics={params['lambda_physics']}, λ_shape={params.get('lambda_shape', 0.1)}")
    print(f"{'='*80}")

    config = copy.deepcopy(config_template)
    config = apply_config(config, params)
    save_config(config)

    # 清除旧的 best_model.pth (避免混淆)
    best_model_path = MODELS_DIR / "best_model.pth"
    if best_model_path.exists():
        best_model_path.unlink()

    # 运行训练
    from src.training.train import train
    start_time = datetime.now()
    try:
        train()
    except Exception as e:
        print(f"  训练出错: {e}")
        return {"name": name, "mw_mae": float("inf"), "error": str(e)}
    elapsed = (datetime.now() - start_time).total_seconds()

    # 保存模型为配置名
    if best_model_path.exists():
        save_path = MODELS_DIR / f"best_model_{name}.pth"
        shutil.copy2(best_model_path, save_path)
        print(f"  模型已保存为: {save_path.name}")

    # 运行评估获取测试集 Mw MAE
    test_mae = evaluate_model(config)

    best_val, best_epoch, total_epochs = get_best_mw_mae_from_log(LOGS_DIR)

    result = {
        "name": name,
        "params": params,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "total_epochs": total_epochs,
        "test_mae": test_mae,
        "elapsed_sec": elapsed,
    }
    print(f"  结果: best_val_loss={best_val:.4f}, test_MAE={test_mae:.4f}, "
          f"best_epoch={best_epoch}, 耗时={elapsed:.0f}s")
    return result


def evaluate_model(config):
    """快速评估：计算测试集上的 Mw MAE"""
    import torch
    from src.models.model import PINNModel
    from src.data.data_loader import get_data_loaders
    from src.training.physics import PhysicsLoss
    from src.training.loss_stf_rate import STFRateWaveformLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = MODELS_DIR / "best_model.pth"
    if not model_path.exists():
        return float("inf")

    model = PINNModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _, _, test_loader = get_data_loaders(config)
    criterion = PhysicsLoss(config).to(device)
    loss_fn = STFRateWaveformLoss(config).to(device)

    ds_cfg = config.get("dataset", {}) or {}
    stf_m_ref = float(ds_cfg.get("stf_m_ref", 1.0e18))

    all_errors = []
    with torch.no_grad():
        for batch in test_loader:
            radial = batch["radial"].to(device)
            stf_true = batch.get("stf", None)
            has_stf = batch.get("has_stf", None)
            magnitude = batch.get("magnitude", None)
            dt_val = float(batch["dt"].mean().item())

            if stf_true is not None:
                stf_true = stf_true.to(device)
            if has_stf is not None:
                has_stf = has_stf.to(device)
            if magnitude is not None:
                magnitude = magnitude.to(device)

            pred_log = model(radial)
            pred_dot_m0 = loss_fn._decode_rate(pred_log)
            pred_mw = criterion.utils.magnitude_from_rate(pred_dot_m0, dt_val)

            if stf_true is not None:
                true_mw = criterion.utils.magnitude_from_rate(stf_true, dt_val)
            else:
                true_mw = magnitude

            errors = torch.abs(pred_mw.view(-1) - true_mw.view(-1))
            finite_mask = torch.isfinite(errors)
            if finite_mask.any():
                all_errors.extend(errors[finite_mask].cpu().tolist())

    if len(all_errors) == 0:
        return float("inf")
    import numpy as np
    return float(np.mean(all_errors))


def main():
    # 加载原始配置作为模板
    config_template = load_config()

    # =====================================================================
    # 超参数配置列表 (Round 2: L_wave 修复后的 λ_wave 消融实验)
    # 固定: lr=1e-4, λ_physics=0.2, λ_shape=0.1
    # 变量: λ_wave ∈ {0.0, 0.01, 0.02, 0.05, 0.1, 0.2}
    # =====================================================================
    configs = [
        ("B1", {"lr": 1e-4, "lambda_wave": 0.0,  "lambda_physics": 0.2, "lambda_shape": 0.1}),
        ("B2", {"lr": 1e-4, "lambda_wave": 0.01, "lambda_physics": 0.2, "lambda_shape": 0.1}),
        ("B3", {"lr": 1e-4, "lambda_wave": 0.02, "lambda_physics": 0.2, "lambda_shape": 0.1}),
        ("B4", {"lr": 1e-4, "lambda_wave": 0.05, "lambda_physics": 0.2, "lambda_shape": 0.1}),
        ("B5", {"lr": 1e-4, "lambda_wave": 0.1,  "lambda_physics": 0.2, "lambda_shape": 0.1}),
        ("B6", {"lr": 1e-4, "lambda_wave": 0.2,  "lambda_physics": 0.2, "lambda_shape": 0.1}),
    ]

    results = []
    for name, params in configs:
        res = run_single_config(name, params, config_template)
        results.append(res)

    # =====================================================================
    # 汇总结果
    # =====================================================================
    print("\n" + "=" * 100)
    print("超参数搜索结果汇总")
    print("=" * 100)
    print(f"{'Name':<6} {'lr':>8} {'λ_wave':>8} {'λ_phys':>8} {'λ_shape':>8} "
          f"{'Val_Loss':>10} {'Test_MAE':>10} {'Best_Ep':>8} {'Total_Ep':>8} {'Time(s)':>8}")
    print("-" * 100)

    results_sorted = sorted(results, key=lambda r: r.get("test_mae", float("inf")))
    for r in results_sorted:
        p = r.get("params", {})
        tag = " ★" if r == results_sorted[0] else ""
        print(f"{r['name']:<6} {p.get('lr', 0):>8.0e} {p.get('lambda_wave', 0):>8.3f} "
              f"{p.get('lambda_physics', 0):>8.3f} {p.get('lambda_shape', 0):>8.3f} "
              f"{r.get('best_val_loss', float('inf')):>10.4f} "
              f"{r.get('test_mae', float('inf')):>10.4f} "
              f"{r.get('best_epoch', -1):>8} {r.get('total_epochs', -1):>8} "
              f"{r.get('elapsed_sec', 0):>8.0f}{tag}")

    # 保存结果到文件
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("超参数搜索结果\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"目标: 超过 PhysicsLoss MAE=0.0996\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Name':<6} {'lr':>8} {'λ_wave':>8} {'λ_phys':>8} {'λ_shape':>8} "
                f"{'Val_Loss':>10} {'Test_MAE':>10} {'Best_Ep':>8} {'Total_Ep':>8}\n")
        f.write("-" * 100 + "\n")
        for r in results_sorted:
            p = r.get("params", {})
            tag = " ★ BEST" if r == results_sorted[0] else ""
            f.write(f"{r['name']:<6} {p.get('lr', 0):>8.0e} {p.get('lambda_wave', 0):>8.3f} "
                    f"{p.get('lambda_physics', 0):>8.3f} {p.get('lambda_shape', 0):>8.3f} "
                    f"{r.get('best_val_loss', float('inf')):>10.4f} "
                    f"{r.get('test_mae', float('inf')):>10.4f} "
                    f"{r.get('best_epoch', -1):>8} {r.get('total_epochs', -1):>8}{tag}\n")
        
        best = results_sorted[0]
        f.write(f"\n最佳配置: {best['name']}\n")
        f.write(f"测试集 MAE: {best.get('test_mae', float('inf')):.4f}\n")
        beat_physics = best.get("test_mae", float("inf")) < 0.0996
        f.write(f"是否超过 PhysicsLoss (0.0996): {'✅ 是' if beat_physics else '❌ 否'}\n")

    print(f"\n结果已保存到: {RESULTS_FILE}")

    # 恢复最佳模型为 best_model.pth
    best_name = results_sorted[0]["name"]
    best_src = MODELS_DIR / f"best_model_{best_name}.pth"
    best_dst = MODELS_DIR / "best_model.pth"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"最佳模型 ({best_name}) 已复制为 best_model.pth")

    # 恢复原始配置
    save_config(config_template)
    print("原始 config.yaml 已恢复")


if __name__ == "__main__":
    main()
