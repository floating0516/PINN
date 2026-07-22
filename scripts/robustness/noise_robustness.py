#!/usr/bin/env python
"""E2.2 噪声鲁棒性测试 — 评估真实 GNSS 噪声条件下的性能退化

功能：
  1. 使用已训练好的模型（无需重新训练）
  2. 在推断时向输入波形叠加有色噪声（模拟 GNSS 实时定位噪声）
  3. 噪声模型：高斯白噪声 + 1/f 滤波，幅度 σ ∈ {1, 3, 5, 10} mm
  4. 每种噪声水平重复多次取平均
  5. 评估不同噪声水平下的 MAE

用法示例：

  python scripts/robustness/noise_robustness.py \
      --model-dir ./outputs_experiments/phase1/models/20260412_141044 \
      --event-data-root "/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/GNSS_EQDATA" \
      --output-dir ./outputs_experiments/phase2/e2_2_noise

  # 自定义噪声水平和重复次数
  python scripts/robustness/noise_robustness.py \
      --model-dir ./outputs_experiments/phase1/models/20260412_141044 \
      --event-data-root "/path/to/events" \
      --output-dir ./outputs_experiments/phase2/e2_2_noise \
      --noise-levels 0 1 3 5 10 20 \
      --n-repeats 10
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.metadata import build_metadata_tensor
from src.data.waveform import waveform_config_from_v2
from src.evaluation.evaluate import _ensure_time_steps, magnitude_series_from_rate
from src.evaluation.evaluate_unseen import (
    EventBundle,
    StationWaveform,
    _apply_pub_style,
    _format_event_display_name,
    _OKABE_ITO,
    _station_sample_from_bundle,
    load_event_bundle,
)
from src.models.model import PINNModel
from src.training.physics import PhysicsLoss
from src.utils.device import get_preferred_device


def discover_event_dirs(event_data_root: Path) -> list[Path]:
    """扫描事件数据根目录，返回含 event.json 的子目录"""
    dirs: list[Path] = []
    if not event_data_root.is_dir():
        return dirs
    for child in sorted(event_data_root.iterdir()):
        if child.is_dir() and (child / "event.json").exists():
            dirs.append(child)
    return dirs


def generate_colored_noise(n_samples: int, sigma_m: float, dt: float, rng: np.random.RandomState) -> np.ndarray:
    """生成有色噪声（白噪声 + 1/f 滤波），模拟 GNSS 实时定位噪声

    噪声模型：
    - 白噪声基底 σ
    - 1/f 滤波（粉红噪声特征，低频分量增强）

    参数：
        n_samples: 样本数
        sigma_m: 噪声标准差（米）
        dt: 采样间隔（秒）
        rng: 随机数生成器

    返回：
        有色噪声时间序列（米）
    """
    if sigma_m <= 0 or n_samples == 0:
        return np.zeros(n_samples, dtype=np.float32)

    # 生成白噪声
    white = rng.randn(n_samples).astype(np.float64)

    # 1/f 滤波（粉红噪声）
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    fft_white = np.fft.rfft(white)

    # 1/f 滤波器：避免 DC 分量为 0
    with np.errstate(divide="ignore", invalid="ignore"):
        pink_filter = np.where(freqs > 0, 1.0 / np.sqrt(freqs), 0.0)
    # 限制最大增益
    if np.any(pink_filter > 0):
        pink_filter = pink_filter / np.max(pink_filter[pink_filter > 0])

    fft_colored = fft_white * pink_filter
    colored = np.fft.irfft(fft_colored, n=n_samples)

    # 归一化到目标 sigma
    std = np.std(colored)
    if std > 0:
        colored = colored / std * sigma_m

    return colored.astype(np.float32)


def run_noise_robustness(
    *,
    model_dir: Path,
    event_dirs: list[Path],
    output_dir: Path,
    noise_levels_mm: list[float],
    n_repeats: int = 10,
    seed: int = 42,
    radial_peak_min_cm_override: float | None = None,
) -> dict[str, Any]:
    """对不同噪声水平执行推断"""
    model_path = model_dir / "best_model.pth"
    config_path = model_dir / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ds_cfg = config.get("dataset", {}) or {}
    train_cfg = config.get("training", {}) or {}
    waveform_config = waveform_config_from_v2(config)

    device = get_preferred_device()
    checkpoint = torch.load(model_path, map_location=device)
    model = PINNModel(config).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    criterion = PhysicsLoss(config).to(device)
    time_steps = int(train_cfg.get("time_steps", 200))
    stf_m_ref = float(ds_cfg.get("stf_m_ref", 1.0e18))

    # 加载事件并预计算干净样本
    print("加载事件数据并预计算干净样本...")
    clean_samples: list[dict[str, Any]] = []
    for ed in event_dirs:
        bundle = load_event_bundle(ed)
        event_label = _format_event_display_name(
            event_name=bundle.event_name,
            event_dir_name=bundle.event_dir_name,
            magnitude=bundle.magnitude,
        )
        for station in bundle.stations:
            sample = _station_sample_from_bundle(
                bundle,
                station,
                config,
                waveform_config=waveform_config,
                radial_peak_min_cm_override=radial_peak_min_cm_override,
            )
            if sample is None:
                continue
            sample["event_label"] = event_label
            sample["mw_true"] = bundle.magnitude
            sample["mechanism"] = bundle.mechanism
            clean_samples.append(sample)
    print(f"  共 {len(clean_samples)} 个有效台站样本")

    rng = np.random.RandomState(seed)
    noise_results: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for noise_mm in sorted(noise_levels_mm):
            sigma_m = noise_mm / 1000.0  # mm -> m
            print(f"\n── 噪声水平: σ = {noise_mm:.1f} mm ──")

            repeat_maes: list[float] = []
            repeat_rmses: list[float] = []

            for rep in range(n_repeats):
                event_preds: dict[str, dict[str, list[float]]] = {}

                for sample in clean_samples:
                    radial_clean = np.array(sample["radial"], dtype=np.float32)
                    dt = float(sample["waveform_dt_sec"])

                    # 添加噪声
                    if sigma_m > 0:
                        noise = generate_colored_noise(len(radial_clean), sigma_m, dt, rng)
                        radial_noisy = radial_clean + noise
                    else:
                        radial_noisy = radial_clean

                    radial_t = torch.tensor(
                        radial_noisy, dtype=torch.float32, device=device,
                    ).unsqueeze(0).unsqueeze(0)
                    radial_t = _ensure_time_steps(radial_t, time_steps)
                    meta_t = build_metadata_tensor(
                        torch.tensor([sample["source_distance_m"]], dtype=torch.float32, device=device),
                        torch.tensor([sample["theta_deg"]], dtype=torch.float32, device=device),
                        torch.tensor([sample["azimuth_deg"]], dtype=torch.float32, device=device),
                    )

                    rate_log = model(radial_t, meta=meta_t)
                    dot_m0 = stf_m_ref * (torch.pow(10.0, rate_log) - 1.0)
                    dot_m0 = torch.clamp(dot_m0, min=0.0)
                    mw_pred = float(criterion.utils.magnitude_from_rate(dot_m0, dt)[0].item())

                    ev = sample["event_label"]
                    if ev not in event_preds:
                        event_preds[ev] = {"mw_true": sample["mw_true"], "preds": []}
                    event_preds[ev]["preds"].append(mw_pred)

                # 事件级汇总
                errors = []
                for ev_name, ev_data in event_preds.items():
                    mw_median = float(np.median(ev_data["preds"]))
                    err = mw_median - ev_data["mw_true"]
                    errors.append(err)

                    if rep == 0:
                        detail_rows.append({
                            "noise_mm": noise_mm,
                            "repeat": rep,
                            "event": ev_name,
                            "mw_true": ev_data["mw_true"],
                            "mw_pred_median": mw_median,
                            "error": err,
                            "n_stations": len(ev_data["preds"]),
                        })

                mae = float(np.mean(np.abs(errors)))
                rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
                repeat_maes.append(mae)
                repeat_rmses.append(rmse)

            mean_mae = float(np.mean(repeat_maes))
            std_mae = float(np.std(repeat_maes))
            mean_rmse = float(np.mean(repeat_rmses))
            std_rmse = float(np.std(repeat_rmses))
            print(f"  MAE: {mean_mae:.4f} ± {std_mae:.4f} | RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

            noise_results.append({
                "noise_mm": noise_mm,
                "mae_mean": mean_mae,
                "mae_std": std_mae,
                "rmse_mean": mean_rmse,
                "rmse_std": std_rmse,
                "n_repeats": n_repeats,
            })

    # ── 保存结果 ──
    output_dir.mkdir(parents=True, exist_ok=True)

    # 汇总 CSV
    summary_csv = output_dir / "noise_summary.csv"
    if noise_results:
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(noise_results[0].keys()))
            writer.writeheader()
            writer.writerows(noise_results)

    # 事件级详细 CSV
    detail_csv = output_dir / "noise_event_detail.csv"
    if detail_rows:
        with detail_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)

    # 绘图
    _plot_noise_degradation(noise_results, output_dir / "noise_degradation.png")

    print(f"\n结果已保存至: {output_dir}")
    return {"noise_summary": noise_results, "detail_rows": detail_rows}


def _plot_noise_degradation(
    noise_results: list[dict[str, Any]],
    save_path: Path,
) -> None:
    """绘制噪声水平 vs MAE/RMSE 退化曲线"""
    if not noise_results:
        return
    _apply_pub_style()

    noise_levels = [r["noise_mm"] for r in noise_results]
    mae_means = [r["mae_mean"] for r in noise_results]
    mae_stds = [r["mae_std"] for r in noise_results]
    rmse_means = [r["rmse_mean"] for r in noise_results]
    rmse_stds = [r["rmse_std"] for r in noise_results]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.errorbar(noise_levels, mae_means, yerr=mae_stds, marker="o", markersize=5,
                color=_OKABE_ITO[0], linewidth=1.5, capsize=3, label="MAE")
    ax.errorbar(noise_levels, rmse_means, yerr=rmse_stds, marker="s", markersize=5,
                color=_OKABE_ITO[1], linewidth=1.5, capsize=3, label="RMSE")

    ax.set_xlabel("Noise Level σ (mm)")
    ax.set_ylabel("Magnitude Error ($M_w$)")
    ax.set_title("Noise Robustness: MAE/RMSE vs. Noise Level")
    ax.legend()
    ax.grid(True, axis="both", linestyle=":", linewidth=0.4, color="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2.2 噪声鲁棒性测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-dir", required=True, help="模型目录")
    parser.add_argument("--event-data-root", required=True, help="unseen 事件数据根目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument(
        "--noise-levels", type=float, nargs="+",
        default=[0, 1, 3, 5, 10],
        help="噪声标准差列表（mm），默认: 0 1 3 5 10",
    )
    parser.add_argument("--n-repeats", type=int, default=10,
                        help="每种噪声水平的重复次数（默认: 10）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--radial-peak-min-cm", type=float, default=None,
                        help="可选，覆盖径向峰值阈值")
    args = parser.parse_args()

    event_dirs = discover_event_dirs(Path(args.event_data_root))
    if not event_dirs:
        print(f"错误: 未在 {args.event_data_root} 下找到事件目录")
        sys.exit(1)

    print(f"模型目录: {args.model_dir}")
    print(f"事件数: {len(event_dirs)}")
    print(f"噪声水平: {args.noise_levels} mm")
    print(f"重复次数: {args.n_repeats}")

    run_noise_robustness(
        model_dir=Path(args.model_dir),
        event_dirs=event_dirs,
        output_dir=Path(args.output_dir),
        noise_levels_mm=args.noise_levels,
        n_repeats=args.n_repeats,
        seed=args.seed,
        radial_peak_min_cm_override=args.radial_peak_min_cm,
    )


if __name__ == "__main__":
    main()
