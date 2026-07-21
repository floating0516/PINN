import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# 将项目根目录加入 sys.path 以便导入 src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.model import PINNModel
from src.data.data_loader import get_data_loaders
from src.training.physics import PhysicsLoss
from src.baseline import Baseline
from src.visualization.visualize import plot_mwg_time_evolution, set_srl_plot_style, maybe_show_plot
from src.utils.config_v2 import validate_config_on_startup
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

def evaluate(
    model_path: str | Path | None = None,
    results_run_id: str | None = None,
    config: dict | None = None,
    save_plots: bool = True,
    show_plots: bool = False,
    save_metrics: bool = True,
    test_loader=None,
) -> dict[str, object]:
    """评估已训练模型在测试集上的表现并生成可视化散点图
    参数:
        model_path: 可选，显式指定待评估权重路径
        results_run_id: 可选，显式指定评估结果目录名
        config: 可选，直接传入配置字典
        save_plots: 是否保存图片
        show_plots: 是否直接显示图片
        save_metrics: 是否保存指标文本
    返回:
        dict（返回指标、结果目录、图片路径等元数据）
    设计原因（Why）:
        - 使用 MAE 与 RMSE 量化预测误差；
        - 通过散点图对比真值与预测，直观检查模型偏差与离群点。
    """
    # 加载配置
    if config is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    validate_config_on_startup(config)
        
    device = get_preferred_device()
    ds_cfg = config.get('dataset', {}) or {}
    stf_m_ref = float(ds_cfg.get('stf_m_ref', 1.0e18))
    train_cfg = config.get('training', {}) or {}
    time_steps = int(train_cfg.get('time_steps', 250))
    
    # 路径设置
    models_dir = Path(config['paths']['models_dir'])
    results_root = Path(config['paths']['results_dir'])
    results_root.mkdir(parents=True, exist_ok=True)
    results_run_id = results_run_id or "show_only"
    results_dir = results_root / results_run_id
    if save_plots or save_metrics:
        _, results_dir = create_run_dir(results_root, run_id=results_run_id)
        results_run_id = results_dir.name
    
    # 加载测试集；test_loader 不为空时使用注入的测试集（如 LOEO-CV 留出事件）
    if test_loader is None:
        _, _, test_loader = get_data_loaders(config)
    
    if len(test_loader) == 0:
        print("未找到测试数据。")
        return {
            'model_path': None,
            'results_dir': results_dir,
            'results_run_id': results_run_id,
            'mae': float('nan'),
            'rmse': float('nan'),
            'baseline_mae': float('nan'),
            'baseline_rmse': float('nan'),
            'sample_count': 0,
        }
    
    resolved_model_path = Path(model_path) if model_path is not None else resolve_model_path(models_dir, filename='best_model.pth')
    if resolved_model_path is None:
        print(f"未找到模型权重: {models_dir}")
        return {
            'model_path': None,
            'results_dir': results_dir,
            'results_run_id': results_run_id,
            'mae': float('nan'),
            'rmse': float('nan'),
            'baseline_mae': float('nan'),
            'baseline_rmse': float('nan'),
            'sample_count': 0,
        }
    print(f"评估模型权重: {resolved_model_path}")
    print(f"评估结果目录: {results_dir}")

    checkpoint = torch.load(resolved_model_path, map_location=device)
    model = PINNModel(config).to(device)
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as error:
        message = str(error)
        if "meta_embed.0.weight" in message and "Missing key(s) in state_dict" in message:
            config_no_meta = dict(config)
            config_no_meta["model"] = dict(config.get("model", {}) or {})
            config_no_meta["model"]["use_meta"] = False
            model = PINNModel(config_no_meta).to(device)
            model.load_state_dict(checkpoint)
            print("检测到权重不含 meta_embed，已自动切换为 use_meta=False 加载。")
        else:
            raise
    model.eval()
    
    # 推理预测
    true_mags = []
    pred_mags = []
    baseline_mags = []
    distances_km = []
    mechanisms = []
    criterion = PhysicsLoss(config).to(device)
    baseline = Baseline.from_config(config)
    mwg_items: list[dict[str, object]] = []
    
    print("开始评估...")
    with torch.no_grad():
        for batch in test_loader:
            radial = batch['radial'].to(device)
            radial = _ensure_time_steps(radial, time_steps)
            stf_true = batch.get('stf', None)
            has_stf = batch.get('has_stf', None)
            magnitude = batch.get('magnitude', None)
            distance_m = batch.get('distance', None)
            mechanism = batch.get('mechanism', None)
            event_names = batch.get('event', None)
            station_names = batch.get('station', None)
            theta_deg = batch.get('theta_deg', torch.tensor(45.0)).to(device)
            phi_deg = batch.get('phi_deg', torch.tensor(0.0)).to(device)
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
            pred_mags.extend(mw_pred.cpu().numpy().flatten())
            dt_batch = batch.get('dt', None)

            mw_true_batch = None
            if stf_true is not None and torch.is_tensor(stf_true):
                stf_true = stf_true.to(device)
                mw_true_batch = criterion.utils.magnitude_from_rate(stf_true, dt_val)

            B = int(radial.size(0))
            has_mask = None
            if has_stf is not None and torch.is_tensor(has_stf) and int(has_stf.numel()) >= B:
                has_mask = has_stf.view(-1)[:B].bool()
            for i in range(B):
                use_stf = bool(has_mask[i].item()) if has_mask is not None else (mw_true_batch is not None)
                if use_stf and mw_true_batch is not None and int(mw_true_batch.numel()) >= B:
                    true_mags.append(float(mw_true_batch.view(-1)[i].item()))
                elif magnitude is not None and torch.is_tensor(magnitude) and int(magnitude.numel()) >= B:
                    true_mags.append(float(magnitude.view(-1)[i].item()))
                else:
                    true_mags.append(float('nan'))

            dt_batch = batch.get('dt', None)
            for i in range(B):
                if torch.is_tensor(dt_batch) and int(dt_batch.numel()) >= B:
                    dt_i = float(dt_batch.view(-1)[i].item())
                else:
                    dt_i = float(dt_val)
                ev = event_names[i] if isinstance(event_names, (list, tuple)) and len(event_names) > i else (str(event_names) if event_names is not None else 'unknown')
                st = station_names[i] if isinstance(station_names, (list, tuple)) and len(station_names) > i else ''
                true_mw = None
                if mw_true_batch is not None and int(mw_true_batch.numel()) >= B:
                    if has_mask is None or bool(has_mask[i].item()):
                        true_mw = float(mw_true_batch.view(-1)[i].item())
                mech_i = int(mechanism.view(-1)[i].item()) if torch.is_tensor(mechanism) and int(mechanism.numel()) >= B else None
                if distance_m is not None and torch.is_tensor(distance_m) and int(distance_m.numel()) >= B:
                    distance_km_i = float(distance_m.view(-1)[i].item()) / 1000.0
                else:
                    distance_km_i = float('nan')
                mw_series = magnitude_series_from_rate(dot_m0[i].view(-1), dt_i)
                stf_np = None
                if stf_true is not None and torch.is_tensor(stf_true) and int(stf_true.size(0)) >= B:
                    if has_mask is None or bool(has_mask[i].item()):
                        stf_np = stf_true[i].detach().cpu().numpy()
                mwg_items.append({'event': str(ev), 'station': str(st), 'dt': dt_i, 'mw_series': mw_series, 'true_mw': true_mw, 'stf': stf_np, 'mechanism': mech_i, 'distance_km': distance_km_i})
            if distance_m is not None:
                u_hr = radial.squeeze(1)
                distance_cpu = distance_m.view(-1).detach().cpu()
                theta_cpu = theta_deg.view(-1).detach().cpu()
                phi_cpu = phi_deg.view(-1).detach().cpu()
                # 基线内部使用 float64，MPS 不支持，因此统一在 CPU 上计算
                for i in range(B):
                    if torch.is_tensor(dt_batch) and int(dt_batch.numel()) >= B:
                        dt_i = float(dt_batch.view(-1)[i].item())
                    else:
                        dt_i = float(dt_val)
                    mw_base_i = baseline.calculate_mwg(
                        u_hr=u_hr[i:i+1].detach().cpu(),
                        r_m=distance_cpu[i:i+1],
                        theta_deg=theta_cpu[i:i+1],
                        phi_deg=phi_cpu[i:i+1],
                        dt=dt_i,
                        apply_radiation_pattern=False,
                    )
                    baseline_mags.append(float(mw_base_i[0].item()))
            else:
                baseline_mags.extend([np.nan] * int(radial.size(0)))
            # true_mags 已在上方依据 STF 或 magnitude 汇总
            if distance_m is not None:
                distances_km.extend((distance_m.cpu().numpy().flatten() / 1000.0).tolist())
            else:
                distances_km.extend([np.nan] * int(radial.size(0)))
            if mechanism is not None:
                mechanisms.extend(mechanism.cpu().numpy().flatten().tolist())
            else:
                mechanisms.extend([-1] * int(radial.size(0)))
            
    true_mags = np.array(true_mags)
    pred_mags = np.array(pred_mags)
    baseline_mags = np.array(baseline_mags)
    distances_km = np.array(distances_km) if len(distances_km) > 0 else None
    mechanisms = np.array(mechanisms, dtype=np.int64) if len(mechanisms) > 0 else None
    
    # 指标计算
    mae = mean_absolute_error(true_mags, pred_mags)
    rmse = np.sqrt(mean_squared_error(true_mags, pred_mags))
    
    print("测试集结果：")
    print(f"MAE：{mae:.4f}")
    print(f"RMSE：{rmse:.4f}")
    
    metrics_path = results_dir / 'metrics.txt' if save_metrics else None
    if save_metrics:
        with open(metrics_path, 'w') as f:
            f.write(f"MAE：{mae:.4f}\n")
            f.write(f"RMSE：{rmse:.4f}\n")

    out_mwg = results_dir / 'mwg_time_evolution.png' if save_plots else None
    mwg_plot_path = plot_mwg_time_evolution(
        mwg_items,
        save_path=out_mwg,
        time_steps=time_steps,
        show=show_plots,
    )
    if mwg_plot_path is not None:
        print(f"已保存 Mw_g 时间演化图：{mwg_plot_path}")

    mask_base = np.isfinite(true_mags) & np.isfinite(baseline_mags)
    if int(np.sum(mask_base)) > 0:
        mae_base = mean_absolute_error(true_mags[mask_base], baseline_mags[mask_base])
        rmse_base = np.sqrt(mean_squared_error(true_mags[mask_base], baseline_mags[mask_base]))
    else:
        mae_base = float('nan')
        rmse_base = float('nan')
    baseline_metrics_path = results_dir / 'metrics_baseline.txt' if save_metrics else None
    if save_metrics:
        with open(baseline_metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"Baseline 样本数：{int(np.sum(mask_base))}\n")
            f.write(f"MAE：{mae_base:.4f}\n")
            f.write(f"RMSE：{rmse_base:.4f}\n")
        
    set_srl_plot_style(base_font_size=9)
    
    # 散点图
    plt.figure(figsize=(3.5, 3.5))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')  # 设置为完美正方形
    
    has_mechanism = (
        mechanisms is not None
        and distances_km is not None
        and distances_km.shape[0] == true_mags.shape[0]
        and np.any(np.isin(mechanisms, np.array([0, 1, 2], dtype=np.int64)))
    )
    if has_mechanism:
        base_mask = np.isfinite(distances_km) & np.isfinite(true_mags) & np.isfinite(pred_mags)
        d_vals = distances_km[base_mask]
        norm = plt.Normalize(vmin=float(np.nanmin(d_vals)), vmax=float(np.nanmax(d_vals)))
        cmap = plt.get_cmap('cividis')
        marker_map = {0: '^', 1: 's', 2: 'o'}

        for code in [0, 1, 2]:
            m = base_mask & (mechanisms == code)
            if np.any(m):
                plt.scatter(
                    true_mags[m],
                    pred_mags[m],
                    c=distances_km[m],
                    cmap=cmap,
                    norm=norm,
                    marker=marker_map[code],
                    alpha=0.8,
                    linewidths=0.0,
                )

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Epicentral distance (km)')
    else:
        if distances_km is not None and distances_km.shape[0] == true_mags.shape[0]:
            mask = np.isfinite(distances_km)
            sc = plt.scatter(true_mags[mask], pred_mags[mask], c=distances_km[mask], cmap='cividis', alpha=0.8, linewidths=0.0)
            cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
            cbar.set_label('Epicentral distance (km)')
        else:
            plt.scatter(true_mags, pred_mags, alpha=0.7, linewidths=0.0)
    
    x_vals = np.array(true_mags, dtype=np.float64)
    y_vals = np.array(pred_mags, dtype=np.float64)
    mask_xy = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[mask_xy]
    y_vals = y_vals[mask_xy]
    
    if x_vals.size > 0:
        within_03 = float(np.mean(np.abs(y_vals - x_vals) <= 0.3))
        ax.text(
            0.02,
            0.98,
            f"MAE={mae:.3f}\nRMSE={rmse:.3f}\nWithin ±0.3: {within_03:.1%}\nN={int(x_vals.size)}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
        )
    
    min_val = float(np.nanmin(np.concatenate([x_vals, y_vals]))) if x_vals.size > 0 else 0.0
    max_val = float(np.nanmax(np.concatenate([x_vals, y_vals]))) if x_vals.size > 0 else 1.0
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1.0)
    
    plt.plot([min_val, max_val], [min_val - 0.3, max_val - 0.3],
             color='0.5', linewidth=0.8, linestyle='--', alpha=0.9)
    plt.plot([min_val, max_val], [min_val + 0.3, max_val + 0.3],
             color='0.5', linewidth=0.8, linestyle='--', alpha=0.9)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', label='Thrust', markerfacecolor='black', markersize=5),
        Line2D([0], [0], marker='s', color='none', label='Strike-slip', markerfacecolor='black', markersize=5),
        Line2D([0], [0], marker='^', color='none', label='Normal', markerfacecolor='black', markersize=5),
    ]
    if has_mechanism:
        ax.legend(handles=legend_elements, loc='lower right', frameon=True, framealpha=0.9)
    
    plt.axis([min_val-0.3, max_val+0.3, min_val-0.3, max_val+0.3])
    
    plt.xlabel('Reference $M_w$')
    plt.ylabel('Predicted $M_w$')
    plt.grid(True, linestyle='--', alpha=0.15)
    ax.minorticks_on()
    
    scatter_path = results_dir / 'prediction_scatter.png' if save_plots else None
    if save_plots:
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        print(f"已保存散点图：{scatter_path}")
    maybe_show_plot(show_plots)
    plt.close()

    plt.figure(figsize=(3.5, 3.5))
    ax = plt.gca()
    if distances_km is not None and distances_km.shape[0] == true_mags.shape[0]:
        mask = np.isfinite(distances_km) & np.isfinite(true_mags) & np.isfinite(baseline_mags)
        if np.any(mask):
            sc = plt.scatter(true_mags[mask], baseline_mags[mask], c=distances_km[mask], cmap='cividis', alpha=0.8, linewidths=0.0)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Epicentral distance (km)')
    else:
        mask = np.isfinite(true_mags) & np.isfinite(baseline_mags)
        if np.any(mask):
            plt.scatter(true_mags[mask], baseline_mags[mask], alpha=0.7, linewidths=0.0)

    if np.any(mask):
        within = float(np.mean(np.abs(baseline_mags[mask] - true_mags[mask]) <= 0.3))
        ax.text(0.02, 0.98, f"MAE={mae_base:.3f}\nWithin ±0.3: {within:.1%}\nN={int(np.sum(mask))}", transform=ax.transAxes, fontsize=8, va="top")
        min_val_b = float(np.nanmin(np.concatenate([true_mags[mask], baseline_mags[mask]])))
        max_val_b = float(np.nanmax(np.concatenate([true_mags[mask], baseline_mags[mask]])))
        plt.plot([min_val_b, max_val_b], [min_val_b, max_val_b], color="black", linewidth=1.0)
        plt.plot([min_val_b, max_val_b], [min_val_b - 0.3, max_val_b - 0.3], color='0.5', linewidth=0.8, linestyle='--', alpha=0.9)
        plt.plot([min_val_b, max_val_b], [min_val_b + 0.3, max_val_b + 0.3], color='0.5', linewidth=0.8, linestyle='--', alpha=0.9)
        plt.axis([min_val_b - 0.3, max_val_b + 0.3, min_val_b - 0.3, max_val_b + 0.3])
    else:
        plt.text(0.5, 0.5, "No valid baseline samples", ha='center', va='center', transform=ax.transAxes)

    plt.xlabel('Reference $M_w$')
    plt.ylabel('Baseline $M_w$')
    plt.grid(True, linestyle='--', alpha=0.15)
    ax.minorticks_on()
    baseline_scatter_path = results_dir / 'prediction_scatter_baseline.png' if save_plots else None
    if save_plots:
        plt.savefig(baseline_scatter_path, dpi=300, bbox_inches='tight')
    maybe_show_plot(show_plots)
    plt.close()

    sample_grid_path = results_dir / 'sample_grid.png' if save_plots else None
    print("生成样本诊断图...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        radial = sample_batch['radial'].to(device)
        radial = _ensure_time_steps(radial, time_steps)
        stf_true = sample_batch.get('stf', None)
        dt = sample_batch.get('dt', None)
        distance_m = sample_batch.get('distance', None)
        magnitude_true = sample_batch.get('magnitude', None)

        if distance_m is not None:
            dist_log_s = torch.log(distance_m.to(device).view(-1).clamp(min=1.0))
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

        if stf_true is not None:
            stf_true = stf_true.to(device)

        if dt is None:
            dt = torch.ones(radial.size(0), device=device, dtype=radial.dtype)
        else:
            dt = dt.to(device)

        if magnitude_true is not None:
            magnitude_true = magnitude_true.to(device)

        B = int(radial.size(0))
        eval_cfg = config.get('evaluation', {}) if isinstance(config, dict) else {}
        n_show_cfg = int(eval_cfg.get('sample_grid_rows', 4))
        if n_show_cfg <= 0:
            n_show_cfg = 4
        n_show = min(n_show_cfg, B)
        t_len = int(radial.size(-1))

        set_srl_plot_style(base_font_size=9)
        
        fig, axes = plt.subplots(n_show, 3, figsize=(7.2, 1.75 * n_show), sharex=False, constrained_layout=True)
        if n_show == 1:
            axes = np.expand_dims(axes, axis=0)

        # 列标题 - 英文版本（适合论文发表）
        col_titles = ["Radial Component", "Source Time Function (STF)", "Moment Magnitude Mw(t)"]
        for j in range(3):
            axes[0, j].set_title(col_titles[j])

        panel_labels = ["(a)", "(b)", "(c)"]
        for j in range(3):
            axes[0, j].text(
                0.02,
                0.98,
                panel_labels[j],
                transform=axes[0, j].transAxes,
                ha="left",
                va="top",
                fontweight="bold",
            )

        for i in range(n_show):
            dt_i = float(dt[i].item())
            t_axis = np.arange(t_len, dtype=np.float32) * dt_i
            radial_i = radial[i].squeeze(0).detach().cpu().numpy()

            # 左列：径向位移 - 使用更粗的线条
            axes[i, 0].plot(t_axis, radial_i, color='#0072B2', linewidth=1.1)
            axes[i, 0].set_ylabel("Radial disp. (mm)")
            axes[i, 0].grid(True, linestyle='--', alpha=0.15, zorder=0)
            axes[i, 0].minorticks_on()

            # 中列：STF - 黑色实线（真实）vs 红色虚线（预测）
            if stf_true is not None:
                stf_true_i = stf_true[i].detach().cpu().numpy()
                stf_pred_i = dot_m0_pred[i].detach().cpu().numpy()
                axes[i, 1].plot(t_axis, stf_true_i, color='black', linewidth=1.1, 
                              linestyle='-', label='True')
                axes[i, 1].plot(t_axis, stf_pred_i, color='#D55E00', linewidth=1.1, 
                              linestyle='--', label='Predicted')
                axes[i, 1].set_ylabel("Moment Rate (N·m/s)")
                axes[i, 1].grid(True, linestyle='--', alpha=0.15, zorder=0)
                axes[i, 1].minorticks_on()
                if i == 0:
                    axes[i, 1].legend(loc='upper right', framealpha=0.9, handlelength=2.2)
            else:
                axes[i, 1].text(0.5, 0.5, "No STF Label", ha='center', va='center', 
                              transform=axes[i, 1].transAxes, fontsize=12)
                axes[i, 1].set_axis_off()

            # 右列：Mw(t) - 黑色实线（真实）vs 红色虚线（预测），标注最终震级
            mw_pred_series = magnitude_series_from_rate(dot_m0_pred[i], dt_i)
            if stf_true is not None:
                mw_true_series = magnitude_series_from_rate(stf_true[i], dt_i)
            else:
                mw_true_series = None

            if mw_true_series is not None:
                final_mw_true = mw_true_series[-1]
                axes[i, 2].plot(t_axis, mw_true_series, color='black', linewidth=1.1, 
                              linestyle='-', label=f'True (Final Mw: {final_mw_true:.2f})')
            
            final_mw_pred = mw_pred_series[-1]
            axes[i, 2].plot(t_axis, mw_pred_series, color='#D55E00', linewidth=1.1, 
                          linestyle='--', label=f'Predicted (Final Mw: {final_mw_pred:.2f})')
            axes[i, 2].set_ylabel("Mw(t)")
            axes[i, 2].grid(True, linestyle='--', alpha=0.15, zorder=0)
            axes[i, 2].minorticks_on()
            if i == 0:
                axes[i, 2].legend(loc='lower right', framealpha=0.9, handlelength=2.2)

            # 在左上角显示事件信息
            row_parts = []
            if distance_m is not None:
                row_parts.append(f"Distance: {float(distance_m[i].item())/1000.0:.0f} km")
            if magnitude_true is not None:
                if mw_true_series is not None:
                    row_parts.append(f"True Mw: {float(mw_true_series[-1]):.2f}")
                else:
                    row_parts.append(f"True Mw: {float(magnitude_true[i].item()):.2f}")
            if len(row_parts) > 0:
                axes[i, 0].text(0.02, 0.96, " | ".join(row_parts), 
                              transform=axes[i, 0].transAxes, fontsize=8, va='top')

            # 只在最底行显示X轴标签
            if i == n_show - 1:
                for j in range(3):
                    axes[i, j].set_xlabel("Time (s)")
            else:
                # 其他行隐藏X轴标签但保留刻度
                for j in range(3):
                    axes[i, j].tick_params(labelbottom=False)

        if save_plots:
            fig.savefig(sample_grid_path, dpi=600, bbox_inches='tight')
        maybe_show_plot(show_plots)
        plt.close(fig)
    if sample_grid_path is not None:
        print(f"已保存样本诊断图：{sample_grid_path}")

    return {
        'model_path': resolved_model_path,
        'results_dir': results_dir,
        'results_run_id': results_run_id,
        'mae': float(mae),
        'rmse': float(rmse),
        'baseline_mae': float(mae_base),
        'baseline_rmse': float(rmse_base),
        'sample_count': int(x_vals.size),
        'metrics_path': metrics_path,
        'baseline_metrics_path': baseline_metrics_path,
        'scatter_path': scatter_path,
        'baseline_scatter_path': baseline_scatter_path,
        'sample_grid_path': sample_grid_path,
        'mwg_plot_path': mwg_plot_path,
    }

if __name__ == '__main__':
    evaluate()
