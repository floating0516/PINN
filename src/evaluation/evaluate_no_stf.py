import copy
from pathlib import Path
from typing import Any
import sys

# 将项目根目录加入 sys.path 以便导入 src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Subset
from src.baseline import Baseline

from src.models.model import PINNModel
from src.data.data_loader import get_data_loaders
from src.training.physics import PhysicsLoss
from src.visualization.visualize import plot_mwg_time_evolution
from src.utils.device import get_preferred_device
from src.utils.run_dirs import create_run_dir, resolve_model_path


def magnitude_series_from_rate(rate: torch.Tensor, dt: float) -> np.ndarray:
    m0_seq = torch.cumsum(torch.clamp(rate, min=0.0), dim=0) * dt
    m0_seq = torch.clamp(m0_seq, min=1.0e9)
    mw_seq = (2.0 / 3.0) * (torch.log10(m0_seq) - 9.1)
    return mw_seq.detach().cpu().numpy()


def _ensure_time_steps(x: torch.Tensor, time_steps: int) -> torch.Tensor:
    t = int(x.size(-1))
    if t == int(time_steps):
        return x
    if t > int(time_steps):
        return x[..., : int(time_steps)]
    pad_len = int(time_steps) - t
    pad_shape = list(x.shape)
    pad_shape[-1] = pad_len
    pad = x.new_zeros(pad_shape)
    return torch.cat([x, pad], dim=-1)


def evaluate_no_stf() -> None:
    """评估无 STF 样本在目录震级上的预测效果并生成可视化图
    参数:
        无（从 configs/config.yaml 读取路径与参数）
    返回:
        None（在 results/ 目录写入指标与图片）
    """
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = get_preferred_device()
    ds_cfg = config.get('dataset', {}) or {}
    stf_m_ref = float(ds_cfg.get('stf_m_ref', 1.0e18))
    train_cfg = config.get('training', {}) or {}
    time_steps = int(train_cfg.get('time_steps', 250))

    models_dir = Path(config['paths']['models_dir'])
    results_root = Path(config['paths']['results_dir'])
    results_root.mkdir(parents=True, exist_ok=True)
    _, results_dir = create_run_dir(results_root)

    model = PINNModel(config).to(device)
    model_path = resolve_model_path(models_dir, filename='best_model.pth')
    if model_path is None:
        print(f"未找到模型权重: {models_dir}")
        return
    print(f"无 STF 评估模型权重: {model_path}")
    print(f"无 STF 评估结果目录: {results_dir}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    config_all: dict[str, Any] = copy.deepcopy(config)
    config_all.setdefault('dataset', {})
    config_all['dataset']['allow_missing_stf'] = True
    config_all.setdefault('training', {})
    config_all['training']['validation_split'] = 0.0
    config_all['training']['test_split'] = 0.0

    train_loader_all, _, _ = get_data_loaders(config_all)
    if len(train_loader_all) == 0:
        print("未找到数据。")
        return

    train_subset = train_loader_all.dataset
    base_dataset = train_subset.dataset
    no_stf_indices = [i for i in train_subset.indices if not bool(base_dataset.samples[i].get('has_stf', False))]
    if len(no_stf_indices) == 0:
        print("未找到无 STF 样本。")
        return

    batch_size = int(config.get('training', {}).get('batch_size', 32))
    no_stf_loader = DataLoader(Subset(base_dataset, no_stf_indices), batch_size=batch_size, shuffle=False)

    true_mags: list[float] = []
    pred_mags: list[float] = []
    distances_km: list[float] = []
    mechanisms: list[int] = []
    criterion = PhysicsLoss(config).to(device)
    mwg_items: list[dict[str, object]] = []

    print(f"开始评估无 STF 样本（n={len(no_stf_indices)}）...")
    with torch.no_grad():
        for batch in no_stf_loader:
            radial = batch['radial'].to(device)
            radial = _ensure_time_steps(radial, time_steps)
            magnitude = batch.get('magnitude', None)
            distance_m = batch.get('distance', None)
            mechanism = batch.get('mechanism', None)
            event_names = batch.get('event', None)
            station_names = batch.get('station', None)
            theta_deg = batch.get('theta_deg', torch.zeros(radial.size(0))).to(device)
            phi_deg = batch.get('phi_deg', torch.zeros(radial.size(0))).to(device)
            dt_val = batch['dt'].mean().item()

            if distance_m is not None:
                dist_log = torch.log(distance_m.to(device).view(-1).clamp(min=1.0))
            else:
                dist_log = torch.zeros(radial.size(0), device=device)
            theta_r = torch.deg2rad(theta_deg.view(-1))
            phi_r = torch.deg2rad(phi_deg.view(-1))
            meta = torch.stack([
                dist_log,
                torch.sin(theta_r), torch.cos(theta_r),
                torch.sin(phi_r),   torch.cos(phi_r),
            ], dim=1)

            rate_log = model(radial, meta=meta)
            dot_m0 = stf_m_ref * (torch.pow(10.0, rate_log) - 1.0)
            dot_m0 = torch.clamp(dot_m0, min=0.0)
            mw_pred = criterion.utils.magnitude_from_rate(dot_m0, dt_val)

            pred_mags.extend(mw_pred.detach().cpu().numpy().flatten().tolist())
            if magnitude is not None:
                true_mags.extend(magnitude.detach().cpu().numpy().flatten().tolist())
            
            dt_batch = batch.get('dt', None)
            B = int(radial.size(0))
            for i in range(B):
                if torch.is_tensor(dt_batch) and int(dt_batch.numel()) >= B:
                    dt_i = float(dt_batch.view(-1)[i].item())
                else:
                    dt_i = float(dt_val)
                ev = event_names[i] if isinstance(event_names, (list, tuple)) and len(event_names) > i else (str(event_names) if event_names is not None else 'unknown')
                st = station_names[i] if isinstance(station_names, (list, tuple)) and len(station_names) > i else ''
                mech_i = int(mechanism.view(-1)[i].item()) if torch.is_tensor(mechanism) and int(mechanism.numel()) >= B else None
                if distance_m is not None and torch.is_tensor(distance_m) and int(distance_m.numel()) >= B:
                    distance_km_i = float(distance_m.view(-1)[i].item()) / 1000.0
                else:
                    distance_km_i = float('nan')
                mw_series = magnitude_series_from_rate(dot_m0[i].view(-1), dt_i)
                mwg_items.append({'event': str(ev), 'station': str(st), 'dt': dt_i, 'mw_series': mw_series, 'mechanism': mech_i, 'distance_km': distance_km_i})

            if distance_m is not None:
                distances_km.extend((distance_m.detach().cpu().numpy().flatten() / 1000.0).tolist())
            else:
                distances_km.extend([np.nan] * int(radial.size(0)))

            if mechanism is not None:
                mechanisms.extend(np.asarray(mechanism.detach().cpu().numpy().flatten(), dtype=np.int64).tolist())
            else:
                mechanisms.extend([-1] * int(radial.size(0)))

    true_mags_np = np.asarray(true_mags, dtype=np.float64)
    pred_mags_np = np.asarray(pred_mags, dtype=np.float64)
    distances_km_np = np.asarray(distances_km, dtype=np.float64)
    mechanisms_np = np.asarray(mechanisms, dtype=np.int64)

    mask_xy = np.isfinite(true_mags_np) & np.isfinite(pred_mags_np)
    true_mags_np = true_mags_np[mask_xy]
    pred_mags_np = pred_mags_np[mask_xy]
    distances_km_np = distances_km_np[mask_xy]
    mechanisms_np = mechanisms_np[mask_xy]

    mae = mean_absolute_error(true_mags_np, pred_mags_np)
    rmse = float(np.sqrt(mean_squared_error(true_mags_np, pred_mags_np)))

    with open(results_dir / 'metrics_no_stf.txt', 'w', encoding='utf-8') as f:
        f.write(f"无 STF 样本数：{int(true_mags_np.size)}\n")
        f.write(f"MAE：{mae:.4f}\n")
        f.write(f"RMSE：{rmse:.4f}\n")

    out_mwg = results_dir / 'mwg_time_evolution_no_stf.png'
    plot_mwg_time_evolution(mwg_items, save_path=out_mwg, time_steps=time_steps)
    print(f"已保存 Mw_g 时间演化图：{out_mwg}")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(4.8, 3.6))
    ax = plt.gca()

    has_mechanism = np.any(np.isin(mechanisms_np, np.array([0, 1, 2], dtype=np.int64)))
    has_distance = distances_km_np.size > 0 and np.any(np.isfinite(distances_km_np))

    if has_mechanism and has_distance:
        base_mask = np.isfinite(distances_km_np)
        d_vals = distances_km_np[base_mask]
        norm = plt.Normalize(vmin=float(np.nanmin(d_vals)), vmax=float(np.nanmax(d_vals)))
        cmap = plt.get_cmap('viridis')
        marker_map = {0: '^', 1: 's', 2: 'o'}
        label_map = {0: '正断层', 1: '走滑', 2: '逆冲'}

        for code in [0, 1, 2]:
            m = base_mask & (mechanisms_np == code)
            if np.any(m):
                plt.scatter(
                    true_mags_np[m],
                    pred_mags_np[m],
                    c=distances_km_np[m],
                    cmap=cmap,
                    norm=norm,
                    marker=marker_map[code],
                    alpha=0.75,
                    linewidths=0.0,
                )

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('震中距 (km)')

        for code in [0, 1, 2]:
            plt.scatter([], [], marker=marker_map[code], color='black', label=label_map[code])
    else:
        if has_distance:
            mask_d = np.isfinite(distances_km_np)
            sc = plt.scatter(true_mags_np[mask_d], pred_mags_np[mask_d], c=distances_km_np[mask_d], cmap='viridis', alpha=0.75)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('震中距 (km)')
        else:
            plt.scatter(true_mags_np, pred_mags_np, alpha=0.75)

    if true_mags_np.size > 0:
        within = float(np.mean(np.abs(pred_mags_np - true_mags_np) <= 0.3))
        plt.plot([], [], alpha=0.0, label=f'±0.3: {within:.1%}（n={int(true_mags_np.size)}）')

    min_val = float(np.nanmin(np.concatenate([true_mags_np, pred_mags_np])))
    max_val = float(np.nanmax(np.concatenate([true_mags_np, pred_mags_np])))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.plot([min_val, max_val], [min_val - 0.3, max_val - 0.3], color='gray', linewidth=1.0, linestyle='--', alpha=0.7)
    plt.plot([min_val, max_val], [min_val + 0.3, max_val + 0.3], color='gray', linewidth=1.0, linestyle='--', alpha=0.7)
    plt.axis([min_val - 0.3, max_val + 0.3, min_val - 0.3, max_val + 0.3])
    plt.legend()
    plt.xlabel('目录震级（无STF）')
    plt.ylabel('预测震级（STF积分）')
    plt.title(f'无 STF 样本预测（MAE={mae:.3f}）')
    plt.grid(True)
    out_scatter = results_dir / 'prediction_scatter_no_stf.png'
    plt.savefig(out_scatter, dpi=150)
    plt.close()
    print(f"已保存散点图：{out_scatter}")

    sample_grid_path = results_dir / 'sample_grid_no_stf.png'
    print("生成无 STF 样本诊断图...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(no_stf_loader))
        radial = sample_batch['radial'].to(device)
        radial = _ensure_time_steps(radial, time_steps)
        dt = sample_batch.get('dt', None)
        distance_m = sample_batch.get('distance', None)
        magnitude_true = sample_batch.get('magnitude', None)

        dist_m_s = sample_batch.get('distance', None)
        if dist_m_s is not None:
            dist_log_s = torch.log(dist_m_s.to(device).view(-1).clamp(min=1.0))
        else:
            dist_log_s = torch.zeros(radial.size(0), device=device)
        theta_deg_s = sample_batch.get('theta_deg', torch.zeros(radial.size(0))).to(device).view(-1)
        phi_deg_s = sample_batch.get('phi_deg', torch.zeros(radial.size(0))).to(device).view(-1)
        meta_s = torch.stack([
            dist_log_s,
            torch.sin(torch.deg2rad(theta_deg_s)), torch.cos(torch.deg2rad(theta_deg_s)),
            torch.sin(torch.deg2rad(phi_deg_s)),   torch.cos(torch.deg2rad(phi_deg_s)),
        ], dim=1)
        rate_log = model(radial, meta=meta_s)
        dot_m0_pred = stf_m_ref * (torch.pow(10.0, rate_log) - 1.0)
        dot_m0_pred = torch.clamp(dot_m0_pred, min=0.0)

        if dt is None:
            dt = torch.ones(radial.size(0), device=device, dtype=radial.dtype)
        else:
            dt = dt.to(device)

        if magnitude_true is not None:
            magnitude_true = magnitude_true.to(device)

        B = int(radial.size(0))
        eval_cfg = config.get('evaluation', {}) if isinstance(config, dict) else {}
        n_show_cfg = int(eval_cfg.get('sample_grid_rows', 12))
        if n_show_cfg <= 0:
            n_show_cfg = 12
        n_show = min(n_show_cfg, B)
        t_len = int(radial.size(-1))

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, axes = plt.subplots(n_show, 3, figsize=(18, 2.8 * n_show), sharex=False)
        if n_show == 1:
            axes = np.expand_dims(axes, axis=0)

        col_titles = ["径向分量", "预测 STF", "Mw(t)（预测 vs 目录）"]
        for j in range(3):
            axes[0, j].set_title(col_titles[j])

        for i in range(n_show):
            dt_i = float(dt[i].item())
            t_axis = np.arange(t_len, dtype=np.float32) * dt_i
            radial_i = radial[i].squeeze(0).detach().cpu().numpy()

            axes[i, 0].plot(t_axis, radial_i, color='tab:blue', linewidth=1.0)
            axes[i, 0].set_ylabel("径向位移")
            axes[i, 0].grid(True, linestyle='--', alpha=0.3)

            stf_pred_i = dot_m0_pred[i].detach().cpu().numpy()
            axes[i, 1].plot(t_axis, stf_pred_i, color='tab:red', linewidth=1.0, alpha=0.85, label='预测STF' if i == 0 else None)
            axes[i, 1].set_ylabel("矩率")
            axes[i, 1].grid(True, linestyle='--', alpha=0.3)
            if i == 0:
                axes[i, 1].legend()

            mw_pred_series = magnitude_series_from_rate(dot_m0_pred[i], dt_i)
            axes[i, 2].plot(t_axis, mw_pred_series, color='tab:purple', linewidth=1.0, label=f'预测Mw {mw_pred_series[-1]:.2f}' if i == 0 else None)
            if magnitude_true is not None:
                mw_cat = float(magnitude_true[i].item())
                axes[i, 2].axhline(mw_cat, color='tab:gray', linestyle='--', linewidth=1.0, label=f'目录Mw {mw_cat:.2f}' if i == 0 else None)
            axes[i, 2].set_ylabel("Mw(t)")
            axes[i, 2].grid(True, linestyle='--', alpha=0.3)
            if i == 0:
                axes[i, 2].legend()

            row_parts = []
            if distance_m is not None:
                row_parts.append(f"r={float(distance_m[i].item())/1000.0:.0f}km")
            if magnitude_true is not None:
                row_parts.append(f"Mw={float(magnitude_true[i].item()):.2f}")
            if len(row_parts) > 0:
                axes[i, 0].text(0.02, 0.92, " | ".join(row_parts), transform=axes[i, 0].transAxes, fontsize=10, va='top')

            if i == n_show - 1:
                for j in range(3):
                    axes[i, j].set_xlabel("时间 (s)")

        fig.tight_layout()
        fig.savefig(sample_grid_path, dpi=150)
        plt.close(fig)
    print(f"已保存样本诊断图：{sample_grid_path}")


if __name__ == '__main__':
    evaluate_no_stf()
