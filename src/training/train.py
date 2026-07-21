import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
import yaml
import os
import csv
import random
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 将项目根目录加入 sys.path 以便导入 src
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.model import PINNModel
from src.data.data_loader import get_data_loaders
from src.training.physics import PhysicsLoss
from src.training.loss_stf_rate import STFRateWaveformLoss
from src.utils.device import configure_runtime, get_preferred_device
from src.utils.run_dirs import create_run_dir, make_run_id

def train(config: dict | None = None, data_loaders: tuple | None = None) -> dict[str, object]:
    """训练 PINN 模型（包含物理约束项）
    参数:
        无（从 configs/config.yaml 读取超参数与路径）
    返回:
        dict（返回本次 run 的目录、权重、日志等元数据）
    设计原因（Why）:
        - 组合数据项与物理项的损失，提升模型对地球物理规律的一致性与泛化能力；
        - 统一， I/O 与日志路径，便于复现实验结果与排查问题。
    """
    # 加载配置
    if config is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    # 设置全局随机种子，保证实验可复现
    seed = int((config.get('training', {}) or {}).get('random_seed', 42))
    device = get_preferred_device()
    configure_runtime(seed, device)
    print(f"随机种子: {seed}")

    # 设置设备
    print(f"使用设备: {device}")
    
    # 创建输出目录（按本次 run 归档，避免覆盖历史结果）
    models_root = Path(config['paths']['models_dir'])
    logs_dir = Path(config['paths']['logs_dir'])
    results_root = Path(config['paths']['results_dir'])
    models_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id()
    _, models_dir = create_run_dir(models_root, run_id=run_id)
    _, results_dir = create_run_dir(results_root, run_id=run_id)
    config_snapshot_path = models_dir / 'config.yaml'
    with open(config_snapshot_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    print(f"本次训练输出目录: {models_dir}")
    print(f"已保存配置快照: {config_snapshot_path}")
    
    # 加载数据集（训练/验证/测试划分）；data_loaders 不为空时使用注入的加载器（如 LOEO-CV）
    if data_loaders is not None:
        train_loader, val_loader, test_loader = data_loaders
    else:
        train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # 数据集检查输出
    try:
        train_count = len(train_loader.dataset)
        val_count = len(val_loader.dataset)
        test_count = len(test_loader.dataset)
    except Exception:
        train_count = val_count = test_count = -1
    print(f"数据集统计 | 训练: {train_count} | 验证: {val_count} | 测试: {test_count}")

    
    
    # 初始化模型
    model = PINNModel(config).to(device)
    
    # 初始化优化器与损失（包含物理约束项）
    weight_decay = float(config['training'].get('weight_decay', 0.0) or 0.0)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=weight_decay)
    
    # 学习率调度器（Cosine + Warmup）
    warmup_epochs = int(config['training'].get('warmup_epochs', 5) or 5)
    grad_clip_norm = float(config['training'].get('grad_clip_norm', 1.0) or 1.0)
    scheduler_T0 = int(config['training'].get('scheduler_T0', 50) or 50)
    scheduler_T_mult = int(config['training'].get('scheduler_T_mult', 2) or 2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=scheduler_T0, T_mult=scheduler_T_mult, eta_min=1e-6
    )
    print(f"调度器: CosineAnnealingWarmRestarts(T_0={scheduler_T0}, T_mult={scheduler_T_mult})")
    
    criterion_1 = PhysicsLoss(config).to(device)
    criterion_2 = STFRateWaveformLoss(config).to(device)
    loss_name = str((config.get('training', {}) or {}).get('loss_name', 'physics')).lower()
    use_stf_rate_loss = loss_name in ['stf_rate', 'stf-rate', 'stf_rate_wave', 'waveform_rate']

    # 预训练模型加载（用于微调）
    pretrain_path = config['training'].get('pretrain_path', None)
    if pretrain_path and os.path.exists(pretrain_path):
        print(f"加载预训练权重: {pretrain_path}")
        state_dict = torch.load(pretrain_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("  预训练权重加载完成！")

    # SWA（随机权重平均）- 收集训练过程中多个快照的平均权重以提升泛化
    swa_start = int(config['training'].get('swa_start', 0) or 0)
    swa_model = None
    if swa_start > 0:
        print(f"SWA: 将从 Epoch {swa_start} 开始收集权重平均")

    # 早停指标：stf_rate 模式使用 Mw MAE，否则使用 val_loss
    es_metric = config['training'].get('early_stop_metric', 'auto')
    if es_metric == 'auto':
        es_metric = 'mw_mae' if use_stf_rate_loss else 'val_loss'
    print(f"早停指标: {es_metric}")
    
    # # 日志记录（CSV）
    log_file = logs_dir / f"training_log_{run_id}.csv"
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Data_Loss', 'Train_Phys_Loss', 'Val_Loss', 'Val_MAE', 'LR'])

    best_model_path: Path | None = None
    best_model_swa_path: Path | None = None
    best_val_loss = float('inf')
    best_mw_mae = float('inf')
    epochs = config['training']['epochs']
    es_patience = int(config['training'].get('early_stop_patience', 0) or 0)
    es_min_delta = float(config['training'].get('early_stop_min_delta', 0.0) or 0.0)
    es_counter = 0
    
    # 课程学习配置（E2.4）
    cl_cfg = (config.get('training', {}) or {}).get('curriculum_learning', {}) or {}
    cl_enabled = bool(cl_cfg.get('enabled', False))
    cl_strategy = str(cl_cfg.get('strategy', 'switch')).lower()
    cl_switch_epoch = int(cl_cfg.get('switch_epoch', 100))
    cl_start_mode = str(cl_cfg.get('start_mode', 'simplified')).lower()
    cl_end_mode = str(cl_cfg.get('end_mode', 'full')).lower()
    cl_linear_start = int(cl_cfg.get('linear_start_epoch', 0))
    cl_linear_end = int(cl_cfg.get('linear_end_epoch', 100))
    if cl_enabled:
        print(f"课程学习: 策略={cl_strategy}, {cl_start_mode} → {cl_end_mode}")
        if cl_strategy == 'switch':
            print(f"  硬切换于 Epoch {cl_switch_epoch}")
        else:
            print(f"  线性过渡 Epoch {cl_linear_start} → {cl_linear_end}")

    print("开始训练...")
    for epoch in range(epochs):
        # 课程学习：动态调整辐射模式
        if cl_enabled and use_stf_rate_loss:
            if cl_strategy == 'switch':
                if epoch < cl_switch_epoch:
                    criterion_2.radiation_mode = cl_start_mode
                else:
                    criterion_2.radiation_mode = cl_end_mode
            elif cl_strategy == 'linear':
                # 线性过渡：在 start_mode 和 end_mode 之间混合
                # 通过随机选择模式来近似线性过渡
                if epoch < cl_linear_start:
                    criterion_2.radiation_mode = cl_start_mode
                elif epoch >= cl_linear_end:
                    criterion_2.radiation_mode = cl_end_mode
                else:
                    progress = (epoch - cl_linear_start) / max(cl_linear_end - cl_linear_start, 1)
                    # 以 progress 的概率使用 end_mode
                    if random.random() < progress:
                        criterion_2.radiation_mode = cl_end_mode
                    else:
                        criterion_2.radiation_mode = cl_start_mode

        model.train()
        train_loss_total = 0.0
        train_data_loss_total = 0.0
        train_phys_loss_total = 0.0
        train_seen = 0
        
        for batch in train_loader:
            # 输入张量
            radial = batch['radial'].to(device)
            vertical = batch['vertical'].to(device)
            distance = batch['distance'].to(device)
            magnitude = batch['magnitude'].to(device)
            theta_deg = batch.get('theta_deg', torch.tensor(0.0)).to(device)
            phi_deg = batch.get('phi_deg', torch.tensor(0.0)).to(device)
            phi_slip_deg = batch.get('phi_slip_deg', phi_deg).to(device)
            stf_log = batch.get('stf_log', None)
            stf_true = batch.get('stf', None)
            has_stf = batch.get('has_stf', None)
            if stf_log is not None:
                stf_log = stf_log.to(device)
            if stf_true is not None:
                stf_true = stf_true.to(device)
            if has_stf is not None:
                has_stf = has_stf.to(device)
            dt_val = batch['dt'].mean().item() # 批次平均 dt（近似）

            # 构建元数据张量 [log(dist), sin(θ), cos(θ), sin(φ), cos(φ)]
            dist_log = torch.log(distance.view(-1).clamp(min=1.0))  # (B,)
            theta_r = torch.deg2rad(theta_deg.view(-1))
            phi_r = torch.deg2rad(phi_deg.view(-1))
            meta = torch.stack([
                dist_log,
                torch.sin(theta_r), torch.cos(theta_r),
                torch.sin(phi_r),   torch.cos(phi_r),
            ], dim=1)  # (B, 5)

            optimizer.zero_grad()
            
            # 前向计算
            pred_log = model(radial, meta=meta)  # (B, T)

            # 统一使用 STF 积分 Mw 作为物理约束目标（与 evaluate.py 一致）
            if stf_true is not None:
                mag_from_stf = criterion_1.utils.magnitude_from_rate(stf_true, dt_val)
            else:
                mag_from_stf = magnitude

            if use_stf_rate_loss:
                loss, loss_dict = criterion_2(
                    pred_log,
                    radial_obs=radial,
                    r_m=distance,
                    theta_deg=theta_deg,
                    phi_deg=phi_slip_deg,
                    dt=dt_val,
                    stf_true=stf_true,
                    has_stf=has_stf,
                    true_mag=mag_from_stf,
                )
                data_loss = torch.tensor(float(loss_dict.get('L_MSE', 0.0)), device=device)
                phys_loss = torch.tensor(float(loss_dict.get('L_mag', 0.0) + loss_dict.get('L_synth', 0.0)), device=device)
            else:
                loss, data_loss, phys_loss, _ = criterion_1(
                    pred_log, stf_log, mag_from_stf, distance,
                    theta_deg,
                    phi_slip_deg,
                    dt=dt_val,
                    stf_true=stf_true,
                    has_stf=has_stf,
                )
            
            # loss = criterion(pred_log, stf_log)
            
            # 反向传播与参数更新
            if not torch.isfinite(loss):
                continue
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            
            optimizer.step()
            
            batch_n = radial.size(0)
            train_seen += batch_n
            train_loss_total += float(loss.detach().cpu()) * batch_n
            train_data_loss_total += float(data_loss.detach().cpu()) * batch_n
            train_phys_loss_total += float(phys_loss.detach().cpu()) * batch_n
            
        # 验证阶段
        model.eval()
        val_loss_total = 0.0
        val_mae_total = 0.0
        val_mw_mae_total = 0.0
        val_seen = 0
        val_mw_seen = 0
        
        with torch.no_grad():
            for batch in val_loader:
                radial = batch['radial'].to(device)
                vertical = batch['vertical'].to(device)
                distance = batch['distance'].to(device)
                magnitude = batch['magnitude'].to(device)
                theta_deg = batch.get('theta_deg', torch.tensor(0.0)).to(device)
                phi_deg = batch.get('phi_deg', torch.tensor(0.0)).to(device)
                phi_slip_deg = batch.get('phi_slip_deg', phi_deg).to(device)
                stf_log = batch.get('stf_log', None)
                stf_true = batch.get('stf', None)
                has_stf = batch.get('has_stf', None)
                if stf_log is not None:
                    stf_log = stf_log.to(device)
                if stf_true is not None:
                    stf_true = stf_true.to(device)
                if has_stf is not None:
                    has_stf = has_stf.to(device)
                dt_val = batch['dt'].mean().item()

                # 构建元数据张量
                dist_log = torch.log(distance.view(-1).clamp(min=1.0))
                theta_r = torch.deg2rad(theta_deg.view(-1))
                phi_r = torch.deg2rad(phi_deg.view(-1))
                meta = torch.stack([
                    dist_log,
                    torch.sin(theta_r), torch.cos(theta_r),
                    torch.sin(phi_r),   torch.cos(phi_r),
                ], dim=1)  # (B, 5)

                pred_log = model(radial, meta=meta)
                
                # 统一使用 STF 积分 Mw（与 evaluate.py 一致）
                if stf_true is not None:
                    mag_from_stf = criterion_1.utils.magnitude_from_rate(stf_true, dt_val)
                else:
                    mag_from_stf = magnitude

                if use_stf_rate_loss:
                    loss, loss_dict = criterion_2(
                        pred_log,
                        radial_obs=radial,
                        r_m=distance,
                        theta_deg=theta_deg,
                        phi_deg=phi_slip_deg,
                        dt=dt_val,
                        stf_true=stf_true,
                        has_stf=has_stf,
                        true_mag=mag_from_stf,
                    )
                    pred_dot_m0 = criterion_2._decode_rate(pred_log)

                    # 计算 Mw MAE（与 evaluate.py 一致的指标）
                    pred_mw = criterion_1.utils.magnitude_from_rate(pred_dot_m0, dt_val)
                    true_mw = mag_from_stf
                    mw_diff = torch.abs(pred_mw.view(-1) - true_mw.view(-1))
                    mw_diff = mw_diff[torch.isfinite(mw_diff)]
                    if mw_diff.numel() > 0:
                        val_mw_mae_total += mw_diff.sum().item()
                        val_mw_seen += mw_diff.numel()

                    if stf_true is not None:
                        if has_stf is None:
                            val_mae_total += torch.mean(torch.abs(torch.nan_to_num(pred_dot_m0, nan=0.0) - torch.nan_to_num(stf_true, nan=0.0)), dim=1).sum().item()
                        else:
                            mask = has_stf.bool().view(-1)
                            if torch.any(mask):
                                val_mae_total += torch.mean(torch.abs(torch.nan_to_num(pred_dot_m0[mask], nan=0.0) - torch.nan_to_num(stf_true[mask], nan=0.0)), dim=1).sum().item()
                    if torch.isfinite(loss):
                        batch_n = radial.size(0)
                        val_seen += batch_n
                        val_loss_total += float(loss.detach().cpu()) * batch_n
                    continue
                if stf_true is not None:
                    mag_from_stf = criterion_1.utils.magnitude_from_rate(stf_true, dt_val)
                else:
                    mag_from_stf = magnitude
                loss, _, _, _ = criterion_1(
                    pred_log, stf_log, mag_from_stf, distance,
                    theta_deg, phi_slip_deg, dt=dt_val, stf_true=stf_true, has_stf=has_stf
                )
                # loss = criterion(pred_log, stf_log)

                if torch.isfinite(loss):
                    batch_n = radial.size(0)
                    val_seen += batch_n
                    val_loss_total += float(loss.detach().cpu()) * batch_n
                pred_log_safe = torch.nan_to_num(pred_log, nan=0.0, posinf=0.0, neginf=0.0)
                stf_log_safe = torch.nan_to_num(stf_log, nan=0.0, posinf=0.0, neginf=0.0)
                val_mae_total += torch.mean(torch.abs(pred_log_safe - stf_log_safe), dim=1).sum().item()
                
        # 计算指标
        train_count = max(int(train_seen), 1)
        val_count = max(int(val_seen), 1)
        
        avg_train_loss = train_loss_total / train_count
        avg_train_data = train_data_loss_total / train_count
        avg_train_phys = train_phys_loss_total / train_count
        avg_val_loss = val_loss_total / val_count
        avg_val_mae = val_mae_total / val_count
        avg_mw_mae = val_mw_mae_total / max(val_mw_seen, 1) if val_mw_seen > 0 else float('nan')
        current_lr = optimizer.param_groups[0]['lr']
        
        mw_str = f" | Mw_MAE: {avg_mw_mae:.4f}" if val_mw_seen > 0 else ""
        print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.2e} | 训练损失: {avg_train_loss:.4f} (数据: {avg_train_data:.4f}, 物理: {avg_train_phys:.4f}) | 验证损失: {avg_val_loss:.4f}{mw_str}")
        
        # 追加日志
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_train_data, avg_train_phys, avg_val_loss, avg_val_mae, current_lr])
        
        # 更新学习率（Warmup 后启用调度器）
        if epoch >= warmup_epochs:
            scheduler.step()
            
        # 保存最佳模型/早停计数
        if es_metric == 'mw_mae' and val_mw_seen > 0:
            # 使用 Mw MAE 作为早停指标
            if avg_mw_mae < best_mw_mae - es_min_delta:
                best_mw_mae = avg_mw_mae
                best_val_loss = avg_val_loss
                best_model_path = models_dir / 'best_model.pth'
                torch.save(model.state_dict(), best_model_path)
                print(f"  已保存新的最佳模型权重！(Mw_MAE={avg_mw_mae:.4f})")
                es_counter = 0
            else:
                es_counter += 1
        else:
            # 使用 val_loss 作为早停指标
            if avg_val_loss < best_val_loss - es_min_delta:
                best_val_loss = avg_val_loss
                best_model_path = models_dir / 'best_model.pth'
                torch.save(model.state_dict(), best_model_path)
                print("  已保存新的最佳模型权重！")
                es_counter = 0
            else:
                es_counter += 1
        
        if es_patience > 0 and es_counter >= es_patience:
            print(f"早停触发（连续 {es_counter} 轮无改进，阈值 {es_min_delta}）")
            # 早停前最后一次 SWA 收集
            if swa_start > 0 and (epoch + 1) >= swa_start:
                if swa_model is None:
                    swa_model = AveragedModel(model, device=device)
                    print(f"  SWA: 开始收集权重（Epoch {epoch+1}）")
                swa_model.update_parameters(model)
            break

        # SWA 权重收集（每个 epoch 更新一次平均权重）
        if swa_start > 0 and (epoch + 1) >= swa_start:
            if swa_model is None:
                swa_model = AveragedModel(model, device=device)
                print(f"  SWA: 开始收集权重（Epoch {epoch+1}）")
            swa_model.update_parameters(model)

    # 保存 SWA 平均模型
    if swa_model is not None:
        swa_path = models_dir / 'best_model_swa.pth'
        torch.save(swa_model.module.state_dict(), swa_path)
        best_model_swa_path = swa_path
        n_avg = swa_model.n_averaged.item() if hasattr(swa_model.n_averaged, 'item') else int(swa_model.n_averaged)
        print(f"已保存 SWA 平均模型 ({n_avg} 个快照): {swa_path}")

    train_result = {
        'run_id': run_id,
        'models_dir': models_dir,
        'results_dir': results_dir,
        'best_model_path': best_model_path,
        'best_model_swa_path': best_model_swa_path,
        'config_snapshot_path': config_snapshot_path,
        'log_file': log_file,
        'best_val_loss': float(best_val_loss),
        'best_mw_mae': float(best_mw_mae),
        'device': str(device),
    }
    print("训练完成。")
    return train_result

if __name__ == '__main__':
    train()
