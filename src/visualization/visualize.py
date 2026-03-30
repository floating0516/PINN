import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Any, Dict, Sequence

from src.models.model import PINNModel
from src.data.data_loader import get_data_loaders
from src.utils.device import get_preferred_device
from src.utils.run_dirs import resolve_model_path


NON_INTERACTIVE_BACKENDS = ("agg", "pdf", "ps", "svg", "template")


def maybe_show_plot(show: bool) -> None:
    if not show:
        return
    backend = str(plt.get_backend()).lower()
    if any(name in backend for name in NON_INTERACTIVE_BACKENDS):
        print(f"当前 matplotlib 后端 {backend} 非交互，跳过 show。")
        return
    plt.show()

def set_srl_plot_style(base_font_size: int = 9) -> None:
    from matplotlib import cycler

    okabe_ito = [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ]
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": float(base_font_size),
            "axes.labelsize": float(base_font_size),
            "axes.titlesize": float(base_font_size + 1),
            "legend.fontsize": float(base_font_size - 1),
            "xtick.labelsize": float(base_font_size - 1),
            "ytick.labelsize": float(base_font_size - 1),
            "axes.linewidth": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.0,
            "axes.prop_cycle": cycler(color=okabe_ito),
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

def magnitude_series_from_rate(rate: torch.Tensor, dt: float) -> np.ndarray:
    eps = 1e9
    m0_seq = torch.cumsum(torch.clamp(rate, min=0.0), dim=0) * dt
    m0_seq = torch.clamp(m0_seq, min=eps)
    mw_seq = (2.0 / 3.0) * (torch.log10(m0_seq) - 9.1)
    return mw_seq.detach().cpu().numpy()

def plot_cascade_sample(split: str = "train", sample_index: int = 0, save_path: Optional[Path] = None) -> Path:
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    device = get_preferred_device()
    train_loader, val_loader, test_loader = get_data_loaders(config)
    if split == 'train':
        loader = train_loader
    elif split == 'val':
        loader = val_loader
    else:
        loader = test_loader
    batch = next(iter(loader))
    radial = batch['radial'].to(device)
    stf = batch['stf'].to(device)
    dt = batch['dt'].to(device)
    theta_deg = batch.get('theta_deg', torch.zeros_like(dt)).to(device)
    phi_deg = batch.get('phi_deg', torch.zeros_like(dt)).to(device)
    i = int(max(0, min(sample_index, radial.size(0) - 1)))
    x = radial[i:i+1]
    t_steps = x.size(-1)
    t_axis = np.arange(t_steps) * float(dt[i].item())
    models_dir = Path(config['paths']['models_dir'])
    model = PINNModel(config).to(device)
    model_path = resolve_model_path(models_dir, filename='best_model.pth')
    if model_path is not None and model_path.exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred_rate = model(x).squeeze(0)  # meta 可选，此处不传入（无距离信息）
    pred_rate_np = pred_rate.detach().cpu().numpy()
    stf_np = stf[i].detach().cpu().numpy()
    mw_series_pred = magnitude_series_from_rate(pred_rate, float(dt[i].item()))
    mw_series_true = magnitude_series_from_rate(stf[i], float(dt[i].item()))
    set_srl_plot_style(base_font_size=9)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True, constrained_layout=True)
    axes[0].plot(t_axis, x.squeeze(0).squeeze(0).detach().cpu().numpy(), color='tab:blue', linewidth=1.0)
    axes[0].set_ylabel("Radial disp.")
    axes[0].grid(True, linestyle='--', alpha=0.15)
    # axes[1].plot(t_axis, pred_rate_np, color='tab:red', linewidth=1.0, label='预测STF')
    axes[1].plot(t_axis, stf_np, color='tab:green', linewidth=1.0, alpha=0.7, label='STF (reference)')
    axes[1].set_ylabel("Moment rate")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.15)
    axes[2].plot(t_axis, mw_series_true, color='tab:gray', linewidth=1.0, label=f'Mw(t) ref: {mw_series_true[-1]:.2f}')
    # axes[2].plot(t_axis, mw_series_pred, color='tab:purple', linewidth=1.0, label=f'预测Mw(t){mw_series_pred[-1]:.2f}')
    axes[2].set_ylabel("Mw(t)")
    axes[2].legend()
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, linestyle='--', alpha=0.15)
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_path or (results_dir / f"cascade_plot_{split}_idx{sample_index}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mwg_time_evolution(
    items: Sequence[Dict[str, Any]],
    save_path: Path | None = None,
    max_cols: int = 3,
    point_count: int = 40,
    time_steps: int | None = None,
    show: bool = False,
) -> Path | None:
    """
    绘制按事件分组的 Mw_g 时间演化图。

    参数:
        items: 每个元素至少包含 event, dt, mw_series；可选包含 stf/true_mw, mechanism。
        save_path: 可选，输出图片路径；为 None 时不保存。
        max_cols: 子图最大列数。
        point_count: 红色统计点的最大数量（用于下采样）。
        time_steps: 若提供，则强制横轴为固定步数（0..time_steps），不按秒扩展。
        show: 是否直接显示图片。
    返回:
        第一张保存后的图片路径；若未保存则返回 None。
    """
    if items is None or len(items) == 0:
        raise ValueError("items 为空，无法绘制 Mw_g 时间演化图。")

    def magnitude_from_rate_np(dot_m0: np.ndarray, dt: float) -> float:
        try:
            s = np.asarray(dot_m0, dtype=np.float64).reshape(-1)
        except Exception:
            return float('nan')
        if s.size == 0:
            return float('nan')
        dt_val = float(dt) if (np.isfinite(dt) and dt > 0.0) else 1.0
        m0_seq = np.cumsum(np.maximum(s, 0.0)) * dt_val
        m0_peak = float(np.nanmax(m0_seq)) if m0_seq.size > 0 else float('nan')
        if not np.isfinite(m0_peak):
            return float('nan')
        m0_peak = max(m0_peak, 1.0e-10)
        return float((2.0 / 3.0) * (np.log10(m0_peak) - 9.1))

    event_to_items: Dict[str, list[Dict[str, Any]]] = {}
    for it in items:
        ev = str(it.get('event', '') or 'unknown')
        event_to_items.setdefault(ev, []).append(it)

    # 按样本数量排序，展示全部事件（分页，每页最多9个）
    event_keys = sorted(event_to_items.keys(), key=lambda ev: len(event_to_items[ev]), reverse=True)

    page_size = 9
    pages = [event_keys[i:i + page_size] for i in range(0, len(event_keys), page_size)]
    n_pages = len(pages)

    mech_map = {0: "Normal", 1: "Strike-slip", 2: "Thrust"}

    def _draw_page(page_keys: list, pg_save_path: Path | None) -> None:
        n_events = len(page_keys)
        n_cols = int(max(1, min(int(max_cols), n_events)))
        n_rows = int((n_events + n_cols - 1) // n_cols)

        set_srl_plot_style(base_font_size=9)

        fig_w = 7.2 if n_cols > 1 else 3.5
        fig_h = 2.35 * float(n_rows)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_w, fig_h),
            squeeze=False,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        for k_idx, ev in enumerate(page_keys):
            ax = axes[k_idx // n_cols, k_idx % n_cols]
            ev_items = event_to_items[ev]

            series_list: list[np.ndarray] = []
            dt_list: list[float] = []
            true_list: list[float] = []
            mech_list: list[int] = []

            for it in ev_items:
                mw_series = it.get('mw_series', None)
                if mw_series is None:
                    continue
                s = np.asarray(mw_series, dtype=np.float64).reshape(-1)
                if s.size == 0:
                    continue
                dt = float(it.get('dt', 1.0))
                if not (np.isfinite(dt) and dt > 0.0):
                    dt = 1.0
                series_list.append(s)
                dt_list.append(dt)

                stf_val = it.get('stf', None)
                if stf_val is not None:
                    mw_true = magnitude_from_rate_np(stf_val, dt)
                    if np.isfinite(mw_true):
                        true_list.append(float(mw_true))

                true_val = it.get('true_mw', None)
                if true_val is not None:
                    try:
                        cat_f = float(true_val)
                    except Exception:
                        cat_f = float('nan')
                    if np.isfinite(cat_f):
                        true_list.append(cat_f)

                mech = it.get('mechanism', None)
                if mech is not None:
                    try:
                        mech_i = int(mech)
                    except Exception:
                        mech_i = -1
                    mech_list.append(mech_i)

            if len(series_list) == 0:
                ax.text(0.5, 0.5, "No Valid Station", ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_axis_off()
                continue

            fixed_steps = None
            if time_steps is not None:
                try:
                    fixed_steps = int(time_steps)
                except Exception:
                    fixed_steps = None
            if fixed_steps is not None and fixed_steps > 0:
                n_steps = fixed_steps
                t_axis = np.arange(n_steps, dtype=np.float64)
                mat = np.full((len(series_list), n_steps), np.nan, dtype=np.float64)
                for row_i, s in enumerate(series_list):
                    k = int(min(int(s.size), n_steps))
                    if k <= 0:
                        continue
                    seg = np.asarray(s[:k], dtype=np.float64)
                    mask = np.isfinite(seg)
                    if np.any(mask):
                        mat[row_i, np.arange(k, dtype=np.int64)[mask]] = seg[mask]
                ax.set_xlim(0.0, float(n_steps))
            else:
                dt_vals = np.asarray(dt_list, dtype=np.float64)
                dt_ref = float(np.nanmedian(dt_vals)) if np.any(np.isfinite(dt_vals)) else 1.0
                dt_ref = float(dt_ref) if (np.isfinite(dt_ref) and dt_ref > 0.0) else 1.0

                t_end_max = 0.0
                for s, dt in zip(series_list, dt_list):
                    if s.size >= 2 and np.isfinite(dt) and dt > 0.0:
                        t_end_max = max(t_end_max, float((s.size - 1) * dt))

                n_steps = int(max(2, round(t_end_max / dt_ref) + 1))
                t_axis = np.arange(n_steps, dtype=np.float64) * dt_ref

                mat = np.full((len(series_list), n_steps), np.nan, dtype=np.float64)
                for row_i, (s, dt) in enumerate(zip(series_list, dt_list)):
                    if s.size == 0:
                        continue
                    dt_i = float(dt) if (np.isfinite(dt) and dt > 0.0) else dt_ref
                    t_i = np.arange(s.size, dtype=np.float64) * dt_i
                    mask = np.isfinite(t_i) & np.isfinite(s)
                    if int(np.sum(mask)) < 2:
                        continue
                    t_valid = t_i[mask]
                    s_valid = s[mask]
                    t_max = float(t_valid[-1])
                    if t_max <= 0.0:
                        continue
                    use = t_axis <= t_max
                    if not np.any(use):
                        continue
                    t_dst = t_axis[use]
                    mat[row_i, use] = np.interp(t_dst, t_valid, s_valid)

            # 绘制背景线条
            for row_i in range(mat.shape[0]):
                y = mat[row_i]
                m = np.isfinite(y)
                if int(np.sum(m)) >= 2:
                    ax.plot(t_axis[m], y[m], color="#7f7f7f", linewidth=0.6, alpha=0.15, zorder=1)

            # 绘制统计曲线
            median = np.nanmedian(mat, axis=0)
            q25 = np.nanpercentile(mat, 25.0, axis=0)
            q75 = np.nanpercentile(mat, 75.0, axis=0)
            m_med = np.isfinite(median)
            if np.any(m_med):
                ax.fill_between(
                    t_axis[m_med],
                    q25[m_med],
                    q75[m_med],
                    color="#D55E00",
                    alpha=0.18,
                    linewidth=0.0,
                    zorder=2,
                )
                ax.plot(t_axis[m_med], median[m_med], color="#D55E00", linewidth=1.3, zorder=3)

                idx_last = int(np.where(m_med)[0][-1])
                true_mw_ref = float(np.nanmedian(np.asarray(true_list, dtype=np.float64))) if len(true_list) > 0 else float('nan')

                mech_code = -1
                if len(mech_list) > 0:
                    mech_arr = np.asarray(mech_list, dtype=np.int64)
                    keep = np.isin(mech_arr, np.array([0, 1, 2], dtype=np.int64))
                    if np.any(keep):
                        vals, counts = np.unique(mech_arr[keep], return_counts=True)
                        mech_code = int(vals[int(np.argmax(counts))])

                final_med = float(median[idx_last])
                final_spread = float(0.5 * (q75[idx_last] - q25[idx_last])) if (np.isfinite(q25[idx_last]) and np.isfinite(q75[idx_last])) else float('nan')

                if np.isfinite(true_mw_ref):
                    ax.axhline(true_mw_ref, color="black", linestyle=(0, (4, 3)), linewidth=1.0, zorder=4)

                title = f"{ev}"
                if np.isfinite(true_mw_ref):
                    title += f"  Mw={true_mw_ref:.1f}"
                if mech_code in mech_map:
                    title += f"  {mech_map[mech_code]}"
                ax.set_title(title)

                text = f"N={int(mat.shape[0])}\nPred={final_med:.2f}"
                if np.isfinite(final_spread):
                    text += f"±{final_spread:.2f}"
                ax.text(
                    0.98,
                    0.02,
                    text,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                )
            else:
                pass

            ax.grid(True, linestyle="--", alpha=0.15, zorder=0)
            ax.minorticks_on()

        # 隐藏多余的子图
        for k_idx in range(n_events, n_rows * n_cols):
            axes[k_idx // n_cols, k_idx % n_cols].set_axis_off()

        # 只在最左列和最下行显示坐标轴标签
        for r in range(n_rows):
            for c in range(n_cols):
                ax = axes[r, c]
                if not ax.has_data():
                    continue
                if r == n_rows - 1:
                    ax.set_xlabel('Time step' if (time_steps is not None and int(time_steps) > 0) else 'Time after origin (s)')
                if c == 0:
                    ax.set_ylabel('Mw')

        if pg_save_path is not None:
            pg_save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(pg_save_path, bbox_inches="tight")
        maybe_show_plot(show)
        plt.close(fig)

    save_path = Path(save_path) if save_path is not None else None
    stem = save_path.stem if save_path is not None else None
    suffix = save_path.suffix if save_path is not None else None
    parent = save_path.parent if save_path is not None else None

    saved_paths: list[Path] = []
    for page_idx, page_keys in enumerate(pages):
        if save_path is None:
            pg_path = None
        elif n_pages == 1:
            pg_path = save_path
        else:
            pg_path = parent / f"{stem}_page{page_idx + 1}{suffix}"
        _draw_page(page_keys, pg_path)
        if pg_path is not None:
            saved_paths.append(pg_path)

    return saved_paths[0] if saved_paths else None

if __name__ == '__main__':
    p = plot_cascade_sample(split='train', sample_index=0, save_path=None)
    print(str(p))
