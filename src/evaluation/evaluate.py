import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path
import sys

# 将项目根目录加入 sys.path 以便导入 src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.model import PINNModel
from src.data.data_loader import get_data_loaders
from src.data.loaders_v2 import get_data_loaders_v2
from src.data.metadata import build_metadata_tensor
from src.evaluation.metrics import (
    aggregate_event_predictions,
    summarize_predictions,
)
from src.training.physics import PhysicsLoss
from src.training.loss_stf_rate_v2 import moment_magnitude_from_rate
from src.baseline import Baseline
from src.visualization.visualize import plot_mwg_time_evolution, set_srl_plot_style, maybe_show_plot
from src.utils.config_v2 import validate_config_on_startup
from src.utils.device import get_preferred_device
from src.utils.provenance import (
    configured_dataset_manifest_path,
    file_reference,
    split_protocol_from_manifest,
    write_json,
)
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


def _evaluation_time_steps(config: dict) -> tuple[int, int]:
    if int(config.get('pipeline_version', 1)) == 2:
        dataset = config['dataset']
        sample_rate_hz = float(dataset['sample_rate_hz'])
        waveform_steps = int(
            round(float(dataset['waveform']['duration_sec']) * sample_rate_hz)
        )
        source_steps = int(
            round(float(dataset['stf']['duration_sec']) * sample_rate_hz)
        )
        return waveform_steps, source_steps
    legacy_steps = int((config.get('training', {}) or {}).get('time_steps', 250))
    return legacy_steps, legacy_steps


def _magnitude_from_rate(
    rate_nm_per_s: torch.Tensor,
    dt_sec,
    *,
    pipeline_version: int,
    legacy_criterion,
) -> torch.Tensor:
    if pipeline_version == 2:
        dt_tensor = torch.as_tensor(
            dt_sec,
            dtype=rate_nm_per_s.dtype,
            device=rate_nm_per_s.device,
        )
        return moment_magnitude_from_rate(rate_nm_per_s, dt_tensor)
    if legacy_criterion is None:
        raise ValueError("legacy magnitude conversion requires PhysicsLoss")
    dt_value = float(torch.as_tensor(dt_sec).float().mean().item())
    return legacy_criterion.utils.magnitude_from_rate(
        rate_nm_per_s,
        dt_value,
    )

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
    stf_m_ref = float(
        (ds_cfg.get('stf', {}) or {}).get(
            'm_ref',
            ds_cfg.get('stf_m_ref', 1.0e18),
        )
    )
    pipeline_version = int(config.get('pipeline_version', 1))
    input_time_steps, source_time_steps = _evaluation_time_steps(config)
    
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
        if pipeline_version == 2:
            _, _, test_loader, _ = get_data_loaders_v2(config)
        else:
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
    criterion = (
        None if pipeline_version == 2 else PhysicsLoss(config).to(device)
    )
    baseline = Baseline.from_config(config)
    mwg_items: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    primary_reference = str(
        (config.get('evaluation', {}) or {}).get(
            'primary_reference',
            'catalog',
        )
    )
    primary_reference_key = (
        'mw_stf_native'
        if primary_reference == 'stf_native'
        else 'mw_catalog'
    )
    
    print("开始评估...")
    with torch.no_grad():
        for batch in test_loader:
            radial = batch['radial'].to(device)
            radial = _ensure_time_steps(radial, input_time_steps)
            stf_true = batch.get('stf', None)
            has_stf = batch.get('has_stf', None)
            magnitude = batch.get(
                'magnitude_catalog' if pipeline_version == 2 else 'magnitude',
                None,
            )
            distance_m = batch.get('source_distance_m' if pipeline_version == 2 else 'distance', None)
            epicentral_distance_m = batch.get('epicentral_distance_m', None)
            mechanism = batch.get('mechanism', None)
            event_names = batch.get('event', None)
            station_names = batch.get('station', None)
            theta_deg = batch.get('theta_deg', torch.full((radial.size(0),), 45.0)).to(device)
            azimuth_deg = batch.get(
                'azimuth_deg' if pipeline_version == 2 else 'phi_deg',
                torch.zeros(radial.size(0)),
            ).to(device)
            dt_batch = batch.get(
                'waveform_dt_sec' if pipeline_version == 2 else 'dt',
                None,
            )
            if dt_batch is None:
                dt_batch = torch.ones(radial.size(0), dtype=radial.dtype)
            dt_val = dt_batch.mean().item()
            magnitude_dt_batch = (
                batch.get('stf_dt_sec', dt_batch)
                if pipeline_version == 2
                else dt_batch
            )
            metadata_distance_m = (
                distance_m.to(device)
                if distance_m is not None
                else torch.ones(radial.size(0), device=device)
            )
            meta = build_metadata_tensor(metadata_distance_m, theta_deg, azimuth_deg)
            rate_log = model(radial, meta=meta)
            dot_m0 = stf_m_ref * (torch.pow(10.0, rate_log) - 1.0)
            dot_m0 = torch.clamp(dot_m0, min=0.0)
            mw_pred = _magnitude_from_rate(
                dot_m0,
                magnitude_dt_batch,
                pipeline_version=pipeline_version,
                legacy_criterion=criterion,
            )
            mw_pred_values = mw_pred.cpu().numpy().flatten()
            pred_mags.extend(mw_pred_values)

            mw_stf_batch = None
            if pipeline_version == 2 and torch.is_tensor(
                batch.get('mw_stf_native', None)
            ):
                mw_stf_batch = batch['mw_stf_native'].to(device)
            elif stf_true is not None and torch.is_tensor(stf_true):
                stf_true = stf_true.to(device)
                mw_stf_batch = _magnitude_from_rate(
                    stf_true,
                    dt_batch,
                    pipeline_version=pipeline_version,
                    legacy_criterion=criterion,
                )

            B = int(radial.size(0))
            has_mask = None
            if has_stf is not None and torch.is_tensor(has_stf) and int(has_stf.numel()) >= B:
                has_mask = has_stf.view(-1)[:B].bool()
            for i in range(B):
                if torch.is_tensor(dt_batch) and int(dt_batch.numel()) >= B:
                    dt_i = float(dt_batch.view(-1)[i].item())
                else:
                    dt_i = float(dt_val)
                ev = event_names[i] if isinstance(event_names, (list, tuple)) and len(event_names) > i else (str(event_names) if event_names is not None else 'unknown')
                st = station_names[i] if isinstance(station_names, (list, tuple)) and len(station_names) > i else ''
                stf_reference_mw = float('nan')
                if mw_stf_batch is not None and int(mw_stf_batch.numel()) >= B:
                    if has_mask is None or bool(has_mask[i].item()):
                        stf_reference_mw = float(mw_stf_batch.view(-1)[i].item())
                catalog_mw = (
                    float(magnitude.view(-1)[i].item())
                    if torch.is_tensor(magnitude) and int(magnitude.numel()) >= B
                    else float('nan')
                )
                primary_mw = (
                    stf_reference_mw
                    if primary_reference_key == 'mw_stf_native'
                    else catalog_mw
                )
                true_mags.append(primary_mw)
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
                epicentral_distance_km_i = (
                    float(epicentral_distance_m.view(-1)[i].item()) / 1000.0
                    if torch.is_tensor(epicentral_distance_m)
                    and int(epicentral_distance_m.numel()) >= B
                    else float('nan')
                )
                azimuth_i = float(azimuth_deg.view(-1)[i].item())
                theta_i = float(theta_deg.view(-1)[i].item())
                station_rows.append(
                    {
                        'event': str(ev),
                        'station': str(st),
                        'mw_pred': float(mw_pred_values[i]),
                        'mw_catalog': catalog_mw,
                        'mw_stf_native': stf_reference_mw,
                        'error_vs_catalog': float(mw_pred_values[i]) - catalog_mw,
                        'error_vs_stf_native': float(mw_pred_values[i]) - stf_reference_mw,
                        'epicentral_distance_km': epicentral_distance_km_i,
                        'source_distance_km': distance_km_i,
                        'theta_deg': theta_i,
                        'azimuth_deg': azimuth_i,
                        'threshold_cm': float(ds_cfg.get('radial_peak_min_cm', 0.0)),
                    }
                )
                mwg_items.append({'event': str(ev), 'station': str(st), 'dt': dt_i, 'mw_series': mw_series, 'true_mw': stf_reference_mw, 'stf': stf_np, 'mechanism': mech_i, 'distance_km': distance_km_i})
            if distance_m is not None:
                u_hr = radial.squeeze(1)
                distance_cpu = distance_m.view(-1).detach().cpu()
                theta_cpu = theta_deg.view(-1).detach().cpu()
                phi_cpu = azimuth_deg.view(-1).detach().cpu()
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
    event_rows = aggregate_event_predictions(
        station_rows,
        reference_key=primary_reference_key,
    )
    evaluation_metrics = summarize_predictions(
        station_rows,
        event_rows,
        reference_key=primary_reference_key,
    )
    mae = float(evaluation_metrics['event_mae'])
    rmse = float(evaluation_metrics['event_rmse'])
    
    print("测试集事件级结果：")
    print(f"Event MAE：{mae:.4f}")
    print(f"Event RMSE：{rmse:.4f}")
    
    metrics_path = results_dir / 'metrics.txt' if save_metrics else None
    station_csv_path = results_dir / 'station_predictions.csv' if save_metrics else None
    event_csv_path = results_dir / 'event_summary.csv' if save_metrics else None
    if save_metrics:
        with open(metrics_path, 'w', encoding='utf-8') as f:
            for key, value in evaluation_metrics.items():
                f.write(f"{key}: {value}\n")
        station_fields = list(station_rows[0]) if station_rows else []
        if station_fields:
            with open(station_csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=station_fields)
                writer.writeheader()
                writer.writerows(station_rows)
        event_fields = list(event_rows[0]) if event_rows else []
        if event_fields:
            with open(event_csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=event_fields)
                writer.writeheader()
                writer.writerows(event_rows)

    out_mwg = results_dir / 'mwg_time_evolution.png' if save_plots else None
    mwg_plot_path = plot_mwg_time_evolution(
        mwg_items,
        save_path=out_mwg,
        time_steps=source_time_steps,
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
        radial = _ensure_time_steps(radial, input_time_steps)
        stf_true = sample_batch.get('stf', None)
        dt = sample_batch.get(
            'waveform_dt_sec' if pipeline_version == 2 else 'dt',
            None,
        )
        source_dt = sample_batch.get(
            'stf_dt_sec' if pipeline_version == 2 else 'dt',
            dt,
        )
        distance_m = sample_batch.get('source_distance_m' if pipeline_version == 2 else 'distance', None)
        magnitude_true = sample_batch.get(
            'magnitude_catalog' if pipeline_version == 2 else 'magnitude',
            None,
        )

        theta_deg_s = sample_batch.get('theta_deg', torch.zeros(radial.size(0))).to(device).view(-1)
        azimuth_deg_s = sample_batch.get(
            'azimuth_deg' if pipeline_version == 2 else 'phi_deg',
            torch.zeros(radial.size(0)),
        ).to(device).view(-1)
        metadata_distance_m = (
            distance_m.to(device)
            if distance_m is not None
            else torch.ones(radial.size(0), device=device)
        )
        meta_s = build_metadata_tensor(metadata_distance_m, theta_deg_s, azimuth_deg_s)
        rate_log = model(radial, meta=meta_s)
        dot_m0_pred = stf_m_ref * (torch.pow(10.0, rate_log) - 1.0)
        dot_m0_pred = torch.clamp(dot_m0_pred, min=0.0)

        if stf_true is not None:
            stf_true = stf_true.to(device)

        if dt is None:
            dt = torch.ones(radial.size(0), device=device, dtype=radial.dtype)
        else:
            dt = dt.to(device)
        if source_dt is None:
            source_dt = torch.ones(
                radial.size(0),
                device=device,
                dtype=radial.dtype,
            )
        else:
            source_dt = source_dt.to(device)

        if magnitude_true is not None:
            magnitude_true = magnitude_true.to(device)

        B = int(radial.size(0))
        eval_cfg = config.get('evaluation', {}) if isinstance(config, dict) else {}
        n_show_cfg = int(eval_cfg.get('sample_grid_rows', 4))
        if n_show_cfg <= 0:
            n_show_cfg = 4
        n_show = min(n_show_cfg, B)
        waveform_length = int(radial.size(-1))

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
            waveform_dt_i = float(dt[i].item())
            source_dt_i = float(source_dt[i].item())
            waveform_axis = (
                np.arange(waveform_length, dtype=np.float32) * waveform_dt_i
            )
            radial_i = radial[i].squeeze(0).detach().cpu().numpy()

            # 左列：径向位移 - 使用更粗的线条
            axes[i, 0].plot(waveform_axis, radial_i, color='#0072B2', linewidth=1.1)
            axes[i, 0].set_ylabel("Radial disp. (mm)")
            axes[i, 0].grid(True, linestyle='--', alpha=0.15, zorder=0)
            axes[i, 0].minorticks_on()

            # 中列：STF - 黑色实线（真实）vs 红色虚线（预测）
            if stf_true is not None:
                stf_true_i = stf_true[i].detach().cpu().numpy()
                stf_pred_i = dot_m0_pred[i].detach().cpu().numpy()
                stf_true_axis = np.arange(stf_true_i.size) * source_dt_i
                stf_pred_axis = np.arange(stf_pred_i.size) * source_dt_i
                axes[i, 1].plot(stf_true_axis, stf_true_i, color='black', linewidth=1.1,
                              linestyle='-', label='True')
                axes[i, 1].plot(stf_pred_axis, stf_pred_i, color='#D55E00', linewidth=1.1,
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
            mw_pred_series = magnitude_series_from_rate(
                dot_m0_pred[i],
                source_dt_i,
            )
            if stf_true is not None:
                mw_stf_series = magnitude_series_from_rate(
                    stf_true[i],
                    source_dt_i,
                )
            else:
                mw_stf_series = None

            if mw_stf_series is not None:
                final_mw_stf = mw_stf_series[-1]
                mw_stf_axis = np.arange(mw_stf_series.size) * source_dt_i
                axes[i, 2].plot(mw_stf_axis, mw_stf_series, color='black', linewidth=1.1,
                              linestyle='-', label=f'STF native (Final Mw: {final_mw_stf:.2f})')
            
            final_mw_pred = mw_pred_series[-1]
            mw_pred_axis = np.arange(mw_pred_series.size) * source_dt_i
            axes[i, 2].plot(mw_pred_axis, mw_pred_series, color='#D55E00', linewidth=1.1,
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
                if mw_stf_series is not None:
                    row_parts.append(f"STF Mw: {float(mw_stf_series[-1]):.2f}")
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

    metrics_json_path = results_dir / 'metrics.json' if save_metrics else None
    result_registry_path = (
        results_dir / 'result_registry.json' if save_metrics else None
    )
    if save_metrics:
        metrics_payload = {
            **evaluation_metrics,
            'baseline_station_mae': float(mae_base),
            'baseline_station_rmse': float(rmse_base),
        }
        write_json(metrics_json_path, metrics_payload)
        repository_root = Path(__file__).resolve().parents[2]
        config_snapshot_path = resolved_model_path.parent / 'config.yaml'
        split_manifest_path = resolved_model_path.parent / 'split.json'
        dataset_manifest_path = configured_dataset_manifest_path(
            config,
            root=repository_root,
        )
        evaluation_config = config.get('evaluation', {}) or {}
        csv_artifacts = [
            str(path)
            for path in (station_csv_path, event_csv_path)
            if path is not None and Path(path).is_file()
        ]
        figure_artifacts = [
            str(path)
            for path in (
                scatter_path,
                baseline_scatter_path,
                sample_grid_path,
                mwg_plot_path,
            )
            if path is not None and Path(path).is_file()
        ]
        result_registry = {
            'checkpoint': file_reference(resolved_model_path),
            'config': file_reference(config_snapshot_path),
            'dataset_manifest': file_reference(dataset_manifest_path),
            'split_manifest': file_reference(split_manifest_path),
            'split_protocol': split_protocol_from_manifest(
                split_manifest_path,
                configured_protocol=str(
                    (config.get('training', {}) or {}).get(
                        'split_protocol',
                        'legacy_random_station',
                    )
                ),
            ),
            'primary_reference': str(
                evaluation_config.get('primary_reference', 'catalog')
            ),
            'secondary_reference': str(
                evaluation_config.get('secondary_reference', 'stf_native')
            ),
            'station_metrics': {
                'count': evaluation_metrics['station_count'],
                'mae': evaluation_metrics['station_mae'],
                'rmse': evaluation_metrics['station_rmse'],
                'bias': evaluation_metrics['station_bias'],
            },
            'event_metrics': {
                'count': evaluation_metrics['event_count'],
                'mae': evaluation_metrics['event_mae'],
                'rmse': evaluation_metrics['event_rmse'],
                'bias': evaluation_metrics['event_bias'],
            },
            'artifacts': {
                'csv': csv_artifacts,
                'figures': figure_artifacts,
                'metrics': [str(metrics_json_path)],
            },
        }
        write_json(result_registry_path, result_registry)

    return {
        'model_path': resolved_model_path,
        'results_dir': results_dir,
        'results_run_id': results_run_id,
        'mae': float(mae),
        'rmse': float(rmse),
        'metrics': evaluation_metrics,
        'station_rows': station_rows,
        'event_rows': event_rows,
        'baseline_mae': float(mae_base),
        'baseline_rmse': float(rmse_base),
        'sample_count': int(x_vals.size),
        'metrics_path': metrics_path,
        'station_csv_path': station_csv_path,
        'event_csv_path': event_csv_path,
        'metrics_json_path': metrics_json_path,
        'result_registry_path': result_registry_path,
        'baseline_metrics_path': baseline_metrics_path,
        'scatter_path': scatter_path,
        'baseline_scatter_path': baseline_scatter_path,
        'sample_grid_path': sample_grid_path,
        'mwg_plot_path': mwg_plot_path,
    }

if __name__ == '__main__':
    evaluate()
