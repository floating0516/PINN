import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
import yaml
import os
import csv
import json
import math
import random
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# 将项目根目录加入 sys.path 以便导入 src
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.model import PINNModel
from src.data.data_loader import get_data_loaders
from src.data.metadata import build_metadata_tensor
from src.data.metadata import metadata_distance_from_config
from src.evaluation.metrics import (
    aggregate_event_predictions,
    summarize_predictions,
)
from src.training.physics import PhysicsLoss
from src.training.loss_stf_rate import STFRateWaveformLoss
from src.training.loss_stf_rate_v2 import (
    STFRateWaveformLossV2,
    moment_magnitude_from_rate,
)
from src.training.checkpointing import (
    CheckpointValidationError,
    TrainingSignalState,
    atomic_torch_save,
    build_full_checkpoint,
    install_training_signal_handlers,
    load_full_checkpoint,
    restore_training_state,
    validate_checkpoint_provenance,
)
from src.utils.config_v2 import validate_config_on_startup
from src.utils.device import configure_runtime, get_preferred_device
from src.utils.provenance import (
    RUN_MANIFEST_FIELDS,
    configured_dataset_manifest_path,
    current_git_commit,
    git_is_dirty,
    sha256_file,
    sha256_if_file,
    utc_now_iso,
    write_json,
)
from src.utils.run_dirs import create_run_dir, make_run_id


@dataclass(frozen=True)
class _PreparedV2Batch:
    radial: torch.Tensor
    source_distance_m: torch.Tensor
    theta_deg: torch.Tensor
    phi_slip_deg: torch.Tensor
    source_dt_sec: torch.Tensor
    observation_dt_sec: torch.Tensor
    waveform_valid_mask: torch.Tensor
    stf_true: torch.Tensor
    has_stf: torch.Tensor
    true_mag: torch.Tensor
    metadata: torch.Tensor


def _build_stf_rate_criterion(
    config: dict,
    device: torch.device,
) -> torch.nn.Module:
    if int(config.get("pipeline_version", 1)) == 2:
        return STFRateWaveformLossV2(config).to(device)
    return STFRateWaveformLoss(config).to(device)


def _select_early_stop_value(
    metric: str,
    *,
    val_loss: float,
    station_mw_mae: float,
    event_mae_catalog: float,
) -> float:
    values = {
        "val_loss": val_loss,
        "mw_mae": station_mw_mae,
        "event_mae_catalog": event_mae_catalog,
    }
    if metric not in values:
        raise ValueError(f"unsupported training.early_stop_metric: {metric}")
    return float(values[metric])


def _prepare_v2_batch(
    batch: dict,
    config: dict,
    device: torch.device,
) -> _PreparedV2Batch:
    radial = batch["radial"].to(device)
    source_distance_m = batch["source_distance_m"].to(device)
    epicentral_distance_m = batch.get(
        "epicentral_distance_m",
        batch["source_distance_m"],
    ).to(device)
    theta_deg = batch["theta_deg"].to(device)
    azimuth_deg = batch["azimuth_deg"].to(device)
    phi_slip_deg = batch["phi_slip_deg"].to(device)
    source_dt_sec = batch["stf_dt_sec"].to(device)
    observation_dt_sec = batch["waveform_dt_sec"].to(device)
    waveform_valid_mask = batch["waveform_valid_mask"].to(device)
    stf_true = batch["stf"].to(device)
    has_stf = batch["has_stf"].to(device)
    magnitude_target = str(
        config["dataset"]["stf"]["magnitude_target"]
    )
    if magnitude_target == "stf_native":
        true_mag = batch["mw_stf_native"].to(device)
    elif magnitude_target == "catalog":
        true_mag = batch["magnitude_catalog"].to(device)
    else:
        raise ValueError(f"unknown magnitude_target: {magnitude_target}")
    metadata_distance_m = metadata_distance_from_config(
        config,
        source_distance_m=source_distance_m,
        epicentral_distance_m=epicentral_distance_m,
    )
    metadata = build_metadata_tensor(
        metadata_distance_m,
        theta_deg,
        azimuth_deg,
    )
    return _PreparedV2Batch(
        radial=radial,
        source_distance_m=source_distance_m,
        theta_deg=theta_deg,
        phi_slip_deg=phi_slip_deg,
        source_dt_sec=source_dt_sec,
        observation_dt_sec=observation_dt_sec,
        waveform_valid_mask=waveform_valid_mask,
        stf_true=stf_true,
        has_stf=has_stf,
        true_mag=true_mag,
        metadata=metadata,
    )

def _train_impl(
    config: dict | None = None,
    data_loaders: tuple | None = None,
    resume_checkpoint: str | Path | None = None,
    *,
    signal_state: TrainingSignalState,
) -> dict[str, object]:
    """训练 PINN 模型（包含物理约束项）
    参数:
        无（从 configs/config.yaml 读取超参数与路径）
    返回:
        dict（返回本次 run 的目录、权重、日志等元数据）
    设计原因（Why）:
        - 组合数据项与物理项的损失，提升模型对地球物理规律的一致性与泛化能力；
        - 统一， I/O 与日志路径，便于复现实验结果与排查问题。
    """
    resume_path = Path(resume_checkpoint) if resume_checkpoint is not None else None
    if config is None:
        config_path = (
            resume_path.parent / 'config.yaml'
            if resume_path is not None
            else Path(__file__).parent.parent.parent / 'configs' / 'config.yaml'
        )
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    validate_config_on_startup(config)
    pretrain_path = config['training'].get('pretrain_path', None)
    if resume_path is not None and pretrain_path:
        raise ValueError('pretrain_path and resume_checkpoint are mutually exclusive')

    pipeline_version = int(config.get('pipeline_version', 1))
    repository_root = Path(__file__).resolve().parents[2]
    started_at_utc = utc_now_iso()
    git_commit = current_git_commit(repository_root)
    git_dirty = git_is_dirty(repository_root)
    resume_payload = (
        load_full_checkpoint(resume_path) if resume_path is not None else None
    )

    # 设置全局随机种子，保证实验可复现
    seed = int((config.get('training', {}) or {}).get('random_seed', 42))
    device = get_preferred_device()
    configure_runtime(seed, device)
    print(f"随机种子: {seed}")

    # 设置设备
    print(f"使用设备: {device}")
    
    models_root = Path(config['paths']['models_dir'])
    logs_dir = Path(config['paths']['logs_dir'])
    results_root = Path(config['paths']['results_dir'])
    if resume_payload is None:
        models_root.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        results_root.mkdir(parents=True, exist_ok=True)
        run_id = make_run_id()
        _, models_dir = create_run_dir(models_root, run_id=run_id)
        _, results_dir = create_run_dir(results_root, run_id=run_id)
        config_snapshot_path = models_dir / 'config.yaml'
        with open(config_snapshot_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    else:
        stored_run_state = resume_payload['run_state']
        run_id = str(stored_run_state['run_id'])
        models_dir = Path(stored_run_state['models_dir'])
        results_dir = Path(stored_run_state['results_dir'])
        config_snapshot_path = models_dir / 'config.yaml'
        if resume_path.parent.resolve() != models_dir.resolve():
            raise CheckpointValidationError(
                'resume checkpoint is outside its recorded model directory'
            )
        with config_snapshot_path.open('r', encoding='utf-8') as stream:
            frozen_config = yaml.safe_load(stream)
        if frozen_config != config:
            raise CheckpointValidationError(
                'resume config does not match the frozen run config'
            )
        if not results_dir.is_dir():
            raise CheckpointValidationError('recorded results directory is missing')
    print(f"本次训练输出目录: {models_dir}")
    print(f"配置快照: {config_snapshot_path}")
    
    # 加载数据集（训练/验证/测试划分）；data_loaders 不为空时使用注入的加载器（如 LOEO-CV）
    split_manifest = None
    if data_loaders is not None:
        if len(data_loaders) == 4:
            train_loader, val_loader, test_loader, split_manifest = data_loaders
        else:
            train_loader, val_loader, test_loader = data_loaders
    elif pipeline_version == 2:
        from src.data.loaders_v2 import get_data_loaders_v2

        train_loader, val_loader, test_loader, split_manifest = (
            get_data_loaders_v2(config)
        )
    else:
        train_loader, val_loader, test_loader = get_data_loaders(config)
    split_manifest_path = models_dir / 'split.json'
    if resume_payload is None:
        if split_manifest is not None:
            with split_manifest_path.open('w', encoding='utf-8') as stream:
                json.dump(
                    split_manifest,
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                stream.write('\n')
        else:
            split_manifest_path = None
    elif split_manifest is not None:
        if not split_manifest_path.is_file():
            raise CheckpointValidationError('recorded split manifest is missing')
        with split_manifest_path.open('r', encoding='utf-8') as stream:
            frozen_split = json.load(stream)
        if frozen_split != split_manifest:
            raise CheckpointValidationError(
                'resume split does not match the frozen split manifest'
            )
    elif not split_manifest_path.is_file():
        split_manifest_path = None
    dataset_manifest_path = configured_dataset_manifest_path(
        config,
        root=repository_root,
    )
    provenance = {
        'git_commit': git_commit,
        'git_dirty': git_dirty,
        'config_sha256': sha256_file(config_snapshot_path),
        'dataset_manifest_sha256': sha256_if_file(dataset_manifest_path),
        'split_sha256': sha256_if_file(split_manifest_path),
    }
    run_manifest_path = models_dir / 'run_manifest.json'
    if resume_payload is None:
        run_manifest = {
            'pipeline_version': pipeline_version,
            **provenance,
            'checkpoint_sha256': '',
            'python_version': sys.version.split()[0],
            'torch_version': str(torch.__version__),
            'numpy_version': str(np.__version__),
            'random_seed': seed,
            'started_at_utc': started_at_utc,
            'completed_at_utc': '',
        }
        if tuple(run_manifest) != RUN_MANIFEST_FIELDS:
            raise RuntimeError('run manifest field contract mismatch')
        write_json(run_manifest_path, run_manifest)
    else:
        validate_checkpoint_provenance(resume_payload, provenance)
        with run_manifest_path.open('r', encoding='utf-8') as stream:
            run_manifest = json.load(stream)
    
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
    
    criterion_1 = (
        None if pipeline_version == 2 else PhysicsLoss(config).to(device)
    )
    criterion_2 = _build_stf_rate_criterion(config, device)
    loss_name = str((config.get('training', {}) or {}).get('loss_name', 'physics')).lower()
    use_stf_rate_loss = loss_name in ['stf_rate', 'stf-rate', 'stf_rate_wave', 'waveform_rate']
    if pipeline_version == 2 and not use_stf_rate_loss:
        raise ValueError('pipeline_version=2 requires the STF-rate loss')

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
    
    best_model_path: Path | None = None
    best_model_swa_path: Path | None = None
    best_val_loss = float('inf')
    best_mw_mae = float('inf')
    epochs = int(config['training']['epochs'])
    es_patience = int(config['training'].get('early_stop_patience', 0) or 0)
    es_min_delta = float(config['training'].get('early_stop_min_delta', 0.0) or 0.0)
    es_counter = 0

    if resume_payload is None:
        log_file = logs_dir / f"training_log_{run_id}.csv"
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    'Epoch',
                    'Train_Loss',
                    'Train_Data_Loss',
                    'Train_Phys_Loss',
                    'Val_Loss',
                    'Val_MAE',
                    'Val_Event_MAE_Catalog',
                    'LR',
                ]
            )
    else:
        stored_run_state = resume_payload['run_state']
        log_file = Path(stored_run_state['log_file'])
        best_model_value = stored_run_state.get('best_model_path')
        best_model_swa_value = stored_run_state.get('best_model_swa_path')
        best_model_path = Path(best_model_value) if best_model_value else None
        best_model_swa_path = (
            Path(best_model_swa_value) if best_model_swa_value else None
        )
        if not log_file.is_file():
            raise CheckpointValidationError('recorded training log is missing')

    loader_generator = getattr(train_loader, 'generator', None)
    last_checkpoint_path = models_dir / 'last_state.pth'
    emergency_checkpoint_path = models_dir / 'emergency_state.pth'
    completed_epoch = 0

    def _raise_if_interrupted() -> None:
        signal_state.checkpoint_and_raise(
            last_checkpoint_path,
            emergency_checkpoint_path,
        )

    def _early_stop_state() -> dict[str, object]:
        return {
            'metric': str(es_metric),
            'best_val_loss': float(best_val_loss),
            'best_mw_mae': float(best_mw_mae),
            'counter': int(es_counter),
            'patience': int(es_patience),
            'min_delta': float(es_min_delta),
        }

    def _run_state() -> dict[str, object]:
        return {
            'run_id': run_id,
            'models_dir': str(models_dir),
            'results_dir': str(results_dir),
            'log_file': str(log_file),
            'best_model_path': str(best_model_path) if best_model_path else None,
            'best_model_swa_path': (
                str(best_model_swa_path) if best_model_swa_path else None
            ),
        }

    def _save_stable_checkpoint(epoch_count: int, reason: str) -> None:
        payload = build_full_checkpoint(
            completed_epoch=epoch_count,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stop_state=_early_stop_state(),
            swa_model=swa_model,
            loader_generator=loader_generator,
            run_state=_run_state(),
            provenance=provenance,
            reason=reason,
        )
        atomic_torch_save(payload, last_checkpoint_path)

    if resume_payload is None:
        _save_stable_checkpoint(0, 'initial')
    else:
        if resume_payload['swa_state_dict'] is not None:
            swa_model = AveragedModel(model, device=device)
        restore_training_state(
            resume_payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            swa_model=swa_model,
            loader_generator=loader_generator,
        )
        completed_epoch = int(resume_payload['completed_epoch'])
        early_stop = resume_payload['early_stop_state']
        if str(early_stop['metric']) != str(es_metric):
            raise CheckpointValidationError('early-stop metric mismatch')
        best_val_loss = float(early_stop['best_val_loss'])
        best_mw_mae = float(early_stop['best_mw_mae'])
        es_counter = int(early_stop['counter'])
        with log_file.open('r', encoding='utf-8', newline='') as stream:
            log_rows = list(csv.reader(stream))
        logged_epochs = [int(row[0]) for row in log_rows[1:] if row]
        if logged_epochs != list(range(1, completed_epoch + 1)):
            raise CheckpointValidationError(
                'training log epochs do not match the resume checkpoint'
            )
        resume_history_path = models_dir / 'resume_history.jsonl'
        with resume_history_path.open('a', encoding='utf-8') as stream:
            stream.write(
                json.dumps(
                    {
                        'checkpoint_sha256': sha256_file(resume_path),
                        'completed_epoch': completed_epoch,
                        'reason': str(resume_payload['reason']),
                        'resumed_at_utc': utc_now_iso(),
                    },
                    sort_keys=True,
                )
                + '\n'
            )
    
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
    for epoch in range(completed_epoch, epochs):
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
            if pipeline_version == 2:
                prepared_v2 = _prepare_v2_batch(batch, config, device)
                radial = prepared_v2.radial
                distance = prepared_v2.source_distance_m
                magnitude = prepared_v2.true_mag
                theta_deg = prepared_v2.theta_deg
                phi_slip_deg = prepared_v2.phi_slip_deg
                stf_true = prepared_v2.stf_true
                has_stf = prepared_v2.has_stf
                meta = prepared_v2.metadata
            else:
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
                meta = build_metadata_tensor(distance, theta_deg, phi_deg)

            optimizer.zero_grad()
            
            # 前向计算
            pred_log = model(radial, meta=meta)  # (B, T)

            # 统一使用 STF 积分 Mw 作为物理约束目标（与 evaluate.py 一致）
            if pipeline_version == 2:
                mag_from_stf = prepared_v2.true_mag
            elif stf_true is not None:
                mag_from_stf = criterion_1.utils.magnitude_from_rate(stf_true, dt_val)
            else:
                mag_from_stf = magnitude

            if use_stf_rate_loss:
                if pipeline_version == 2:
                    loss, loss_dict = criterion_2(
                        pred_log,
                        radial_obs=radial,
                        source_distance_m=prepared_v2.source_distance_m,
                        theta_deg=prepared_v2.theta_deg,
                        phi_slip_deg=prepared_v2.phi_slip_deg,
                        source_dt_sec=prepared_v2.source_dt_sec,
                        observation_dt_sec=prepared_v2.observation_dt_sec,
                        waveform_valid_mask=prepared_v2.waveform_valid_mask,
                        stf_true=prepared_v2.stf_true,
                        has_stf=prepared_v2.has_stf,
                        true_mag=prepared_v2.true_mag,
                    )
                else:
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
                _raise_if_interrupted()
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
            _raise_if_interrupted()
            
        # 验证阶段
        model.eval()
        val_loss_total = 0.0
        val_mae_total = 0.0
        val_mw_mae_total = 0.0
        val_seen = 0
        val_mw_seen = 0
        val_event_station_rows: list[dict[str, object]] = []
        
        with torch.no_grad():
            for batch in val_loader:
                if pipeline_version == 2:
                    prepared_v2 = _prepare_v2_batch(batch, config, device)
                    radial = prepared_v2.radial
                    distance = prepared_v2.source_distance_m
                    magnitude = prepared_v2.true_mag
                    theta_deg = prepared_v2.theta_deg
                    phi_slip_deg = prepared_v2.phi_slip_deg
                    stf_true = prepared_v2.stf_true
                    has_stf = prepared_v2.has_stf
                    meta = prepared_v2.metadata
                else:
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
                    meta = build_metadata_tensor(distance, theta_deg, phi_deg)

                pred_log = model(radial, meta=meta)
                
                # 统一使用 STF 积分 Mw（与 evaluate.py 一致）
                if pipeline_version == 2:
                    mag_from_stf = prepared_v2.true_mag
                elif stf_true is not None:
                    mag_from_stf = criterion_1.utils.magnitude_from_rate(stf_true, dt_val)
                else:
                    mag_from_stf = magnitude

                if use_stf_rate_loss:
                    if pipeline_version == 2:
                        loss, loss_dict = criterion_2(
                            pred_log,
                            radial_obs=radial,
                            source_distance_m=prepared_v2.source_distance_m,
                            theta_deg=prepared_v2.theta_deg,
                            phi_slip_deg=prepared_v2.phi_slip_deg,
                            source_dt_sec=prepared_v2.source_dt_sec,
                            observation_dt_sec=prepared_v2.observation_dt_sec,
                            waveform_valid_mask=prepared_v2.waveform_valid_mask,
                            stf_true=prepared_v2.stf_true,
                            has_stf=prepared_v2.has_stf,
                            true_mag=prepared_v2.true_mag,
                        )
                    else:
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
                    if pipeline_version == 2:
                        pred_mw = moment_magnitude_from_rate(
                            pred_dot_m0,
                            prepared_v2.source_dt_sec,
                        )
                    else:
                        pred_mw = criterion_1.utils.magnitude_from_rate(pred_dot_m0, dt_val)
                    true_mw = mag_from_stf
                    mw_diff = torch.abs(pred_mw.view(-1) - true_mw.view(-1))
                    mw_diff = mw_diff[torch.isfinite(mw_diff)]
                    if mw_diff.numel() > 0:
                        val_mw_mae_total += mw_diff.sum().item()
                        val_mw_seen += mw_diff.numel()
                    if pipeline_version == 2:
                        event_names = batch.get('event')
                        catalog_magnitude = batch.get('magnitude_catalog')
                        if event_names is not None and torch.is_tensor(
                            catalog_magnitude
                        ):
                            catalog_values = catalog_magnitude.view(-1)
                            for index, predicted_value in enumerate(
                                pred_mw.view(-1)
                            ):
                                event_name = (
                                    event_names[index]
                                    if isinstance(event_names, (list, tuple))
                                    else str(event_names)
                                )
                                val_event_station_rows.append(
                                    {
                                        'event': str(event_name),
                                        'mw_pred': float(
                                            predicted_value.detach().cpu()
                                        ),
                                        'mw_catalog': float(
                                            catalog_values[index].detach().cpu()
                                        ),
                                    }
                                )

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
                    _raise_if_interrupted()
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
                _raise_if_interrupted()
                
        # 计算指标
        train_count = max(int(train_seen), 1)
        val_count = max(int(val_seen), 1)
        
        avg_train_loss = train_loss_total / train_count
        avg_train_data = train_data_loss_total / train_count
        avg_train_phys = train_phys_loss_total / train_count
        avg_val_loss = val_loss_total / val_count
        avg_val_mae = val_mae_total / val_count
        avg_mw_mae = val_mw_mae_total / max(val_mw_seen, 1) if val_mw_seen > 0 else float('nan')
        if val_event_station_rows:
            validation_event_rows = aggregate_event_predictions(
                val_event_station_rows,
                reference_key='mw_catalog',
            )
            validation_event_metrics = summarize_predictions(
                val_event_station_rows,
                validation_event_rows,
                reference_key='mw_catalog',
            )
            avg_event_mae_catalog = float(
                validation_event_metrics['event_mae']
            )
        else:
            avg_event_mae_catalog = float('nan')
        current_lr = optimizer.param_groups[0]['lr']
        
        mw_str = f" | Mw_MAE: {avg_mw_mae:.4f}" if val_mw_seen > 0 else ""
        event_mw_str = (
            f" | Event_MAE_Catalog: {avg_event_mae_catalog:.4f}"
            if math.isfinite(avg_event_mae_catalog)
            else ""
        )
        print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.2e} | 训练损失: {avg_train_loss:.4f} (数据: {avg_train_data:.4f}, 物理: {avg_train_phys:.4f}) | 验证损失: {avg_val_loss:.4f}{mw_str}{event_mw_str}")
        
        # 追加日志
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_train_data, avg_train_phys, avg_val_loss, avg_val_mae, avg_event_mae_catalog, current_lr])
        
        # 更新学习率（Warmup 后启用调度器）
        if epoch >= warmup_epochs:
            scheduler.step()
            
        # 保存最佳模型/早停计数
        early_stop_value = _select_early_stop_value(
            str(es_metric),
            val_loss=avg_val_loss,
            station_mw_mae=avg_mw_mae,
            event_mae_catalog=avg_event_mae_catalog,
        )
        if not math.isfinite(early_stop_value):
            raise FloatingPointError(
                f"non-finite early-stop metric {es_metric}"
            )
        if es_metric in {'mw_mae', 'event_mae_catalog'}:
            if early_stop_value < best_mw_mae - es_min_delta:
                best_mw_mae = early_stop_value
                best_val_loss = avg_val_loss
                best_model_path = models_dir / 'best_model.pth'
                torch.save(model.state_dict(), best_model_path)
                print(
                    "  已保存新的最佳模型权重！"
                    f"({es_metric}={early_stop_value:.4f})"
                )
                es_counter = 0
            else:
                es_counter += 1
        else:
            # 使用 val_loss 作为早停指标
            if early_stop_value < best_val_loss - es_min_delta:
                best_val_loss = avg_val_loss
                best_model_path = models_dir / 'best_model.pth'
                torch.save(model.state_dict(), best_model_path)
                print("  已保存新的最佳模型权重！")
                es_counter = 0
            else:
                es_counter += 1
        
        should_stop = es_patience > 0 and es_counter >= es_patience
        if should_stop:
            print(f"早停触发（连续 {es_counter} 轮无改进，阈值 {es_min_delta}）")

        # SWA 权重收集（每个 epoch 更新一次平均权重）
        if swa_start > 0 and (epoch + 1) >= swa_start:
            if swa_model is None:
                swa_model = AveragedModel(model, device=device)
                print(f"  SWA: 开始收集权重（Epoch {epoch+1}）")
            swa_model.update_parameters(model)

        completed_epoch = epoch + 1
        _save_stable_checkpoint(completed_epoch, 'epoch')
        _raise_if_interrupted()
        if should_stop:
            break

    # 保存 SWA 平均模型
    if swa_model is not None:
        swa_path = models_dir / 'best_model_swa.pth'
        torch.save(swa_model.module.state_dict(), swa_path)
        best_model_swa_path = swa_path
        n_avg = swa_model.n_averaged.item() if hasattr(swa_model.n_averaged, 'item') else int(swa_model.n_averaged)
        print(f"已保存 SWA 平均模型 ({n_avg} 个快照): {swa_path}")
        _save_stable_checkpoint(completed_epoch, 'final')
        _raise_if_interrupted()

    run_manifest['checkpoint_sha256'] = sha256_if_file(best_model_path)
    run_manifest['completed_at_utc'] = utc_now_iso()
    write_json(run_manifest_path, run_manifest)

    train_result = {
        'run_id': run_id,
        'models_dir': models_dir,
        'results_dir': results_dir,
        'best_model_path': best_model_path,
        'best_model_swa_path': best_model_swa_path,
        'config_snapshot_path': config_snapshot_path,
        'split_manifest_path': split_manifest_path,
        'run_manifest_path': run_manifest_path,
        'log_file': log_file,
        'last_checkpoint_path': last_checkpoint_path,
        'resumed_from_epoch': (
            int(resume_payload['completed_epoch']) if resume_payload else None
        ),
        'best_val_loss': float(best_val_loss),
        'best_mw_mae': float(best_mw_mae),
        'device': str(device),
    }
    print("训练完成。")
    return train_result


def train(
    config: dict | None = None,
    data_loaders: tuple | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, object]:
    signal_state = TrainingSignalState()
    with install_training_signal_handlers(signal_state):
        return _train_impl(
            config=config,
            data_loaders=data_loaders,
            resume_checkpoint=resume_checkpoint,
            signal_state=signal_state,
        )

if __name__ == '__main__':
    train()
