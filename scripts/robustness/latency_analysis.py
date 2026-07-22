#!/usr/bin/env python
"""E2.1 时间窗口演化分析 — 评估 PINN 在不同时间窗口下的震级收敛速度

功能：
  1. 使用已训练好的最优模型（无需重新训练）
  2. 在 unseen 事件推断时，逐步截断输入波形至不同时间窗口
  3. 追踪每个窗口下的事件级中位数 Mw 和 MAE
  4. 生成时间窗口 vs MAE 曲线图和事件级 Mw(t) 演化面板

用法示例：

  # 基本用法
  python scripts/robustness/latency_analysis.py \
      --model-dir ./outputs_experiments/phase1/models/20260412_141044 \
      --event-data-root "/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/GNSS_EQDATA" \
      --output-dir ./outputs_experiments/phase2/e2_1_latency

  # 自定义时间窗口
  python scripts/robustness/latency_analysis.py \
      --model-dir ./outputs_experiments/phase1/models/20260412_141044 \
      --event-data-root "/path/to/events" \
      --output-dir ./outputs_experiments/phase2/e2_1_latency \
      --windows 30 60 90 120 180 300 600
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.external_records import record_from_external_bundle
from src.data.metadata import build_metadata_tensor
from src.data.sample_builder import SampleRejected, build_station_sample
from src.data.waveform import WaveformConfig, waveform_config_from_v2
from src.evaluation.evaluate import _ensure_time_steps, magnitude_series_from_rate
from src.evaluation.evaluate_unseen import (
    EventBundle,
    StationWaveform,
    _apply_pub_style,
    _OKABE_ITO,
    _format_event_display_name,
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


def _station_sample_with_window(
    bundle: EventBundle,
    station: StationWaveform,
    config: dict[str, Any],
    waveform_config: WaveformConfig,
    window_max_sec: float,
    radial_peak_min_cm_override: float | None = None,
) -> dict[str, Any] | None:
    """从事件 bundle 中为指定台站构建样本，使用自定义时间窗口"""
    threshold_cm = float(config["dataset"]["radial_peak_min_cm"])
    if radial_peak_min_cm_override is not None:
        threshold_cm = float(radial_peak_min_cm_override)
    window_config = replace(
        waveform_config,
        duration_sec=float(window_max_sec),
    )
    try:
        return build_station_sample(
            record_from_external_bundle(bundle, station),
            units="m",
            waveform_config=window_config,
            alpha_m_per_s=float(config["physics"]["alpha"]),
            radial_peak_min_cm=threshold_cm,
        )
    except SampleRejected:
        return None


def run_latency_analysis(
    *,
    model_dir: Path,
    event_dirs: list[Path],
    output_dir: Path,
    windows: list[float],
    radial_peak_min_cm_override: float | None = None,
) -> dict[str, Any]:
    """对不同时间窗口执行推断，追踪 Mw 收敛"""
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

    # 加载所有事件 bundle
    bundles: list[EventBundle] = []
    for ed in event_dirs:
        bundles.append(load_event_bundle(ed))

    results_per_window: dict[float, list[dict[str, Any]]] = {}
    station_detail_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for window_sec in sorted(windows):
            print(f"\n── 时间窗口: {window_sec:.0f}s ──")
            event_preds: dict[str, dict[str, Any]] = {}

            for bundle in bundles:
                event_label = _format_event_display_name(
                    event_name=bundle.event_name,
                    event_dir_name=bundle.event_dir_name,
                    magnitude=bundle.magnitude,
                )
                preds: list[float] = []

                for station in bundle.stations:
                    sample = _station_sample_with_window(
                        bundle,
                        station,
                        config,
                        waveform_config,
                        window_sec,
                        radial_peak_min_cm_override,
                    )
                    if sample is None:
                        continue

                    radial = torch.tensor(
                        sample["radial"], dtype=torch.float32, device=device,
                    ).unsqueeze(0).unsqueeze(0)
                    radial = _ensure_time_steps(radial, time_steps)
                    meta = build_metadata_tensor(
                        torch.tensor([sample["source_distance_m"]], dtype=torch.float32, device=device),
                        torch.tensor([sample["theta_deg"]], dtype=torch.float32, device=device),
                        torch.tensor([sample["azimuth_deg"]], dtype=torch.float32, device=device),
                    )

                    rate_log = model(radial, meta=meta)
                    dot_m0 = stf_m_ref * (torch.pow(10.0, rate_log) - 1.0)
                    dot_m0 = torch.clamp(dot_m0, min=0.0)
                    mw_pred = float(criterion.utils.magnitude_from_rate(
                        dot_m0, float(sample["waveform_dt_sec"]),
                    )[0].item())
                    preds.append(mw_pred)

                    station_detail_rows.append({
                        "window_sec": window_sec,
                        "event": event_label,
                        "station": station.station,
                        "mw_true": bundle.magnitude,
                        "mw_pred": mw_pred,
                        "error": mw_pred - bundle.magnitude,
                        "mechanism": bundle.mechanism,
                    })

                if preds:
                    mw_median = float(np.median(preds))
                    event_preds[event_label] = {
                        "mw_true": bundle.magnitude,
                        "mw_pred_median": mw_median,
                        "error": mw_median - bundle.magnitude,
                        "n_stations": len(preds),
                        "mechanism": bundle.mechanism,
                    }

            # 汇总此窗口的指标
            errors = [v["error"] for v in event_preds.values()]
            abs_errors = [abs(e) for e in errors]
            mae = float(np.mean(abs_errors)) if errors else float("nan")
            rmse = float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else float("nan")
            bias = float(np.mean(errors)) if errors else float("nan")
            print(f"  事件数: {len(event_preds)}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, Bias: {bias:+.4f}")

            results_per_window[window_sec] = []
            for ev_name, ev_data in sorted(event_preds.items()):
                results_per_window[window_sec].append({
                    "window_sec": window_sec,
                    "event": ev_name,
                    **ev_data,
                })

    # ── 保存结果 ──
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 台站级详细 CSV
    station_csv = output_dir / "station_detail.csv"
    if station_detail_rows:
        with station_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(station_detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(station_detail_rows)

    # 2. 事件级汇总 CSV
    event_rows_all: list[dict[str, Any]] = []
    for rows in results_per_window.values():
        event_rows_all.extend(rows)
    event_csv = output_dir / "event_window_summary.csv"
    if event_rows_all:
        with event_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(event_rows_all[0].keys()))
            writer.writeheader()
            writer.writerows(event_rows_all)

    # 3. 窗口级汇总 CSV
    window_summary_rows: list[dict[str, Any]] = []
    for w in sorted(results_per_window.keys()):
        rows = results_per_window[w]
        errors = [r["error"] for r in rows]
        abs_errors = [abs(e) for e in errors]
        window_summary_rows.append({
            "window_sec": w,
            "n_events": len(rows),
            "mae": float(np.mean(abs_errors)) if abs_errors else float("nan"),
            "rmse": float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else float("nan"),
            "bias": float(np.mean(errors)) if errors else float("nan"),
        })
    window_csv = output_dir / "window_summary.csv"
    if window_summary_rows:
        with window_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(window_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(window_summary_rows)

    # ── 绘图 ──
    _plot_window_mae_curve(window_summary_rows, output_dir / "window_mae_curve.png")
    _plot_event_mw_evolution(event_rows_all, output_dir / "event_mw_evolution.png")

    print(f"\n结果已保存至: {output_dir}")
    return {
        "window_summary": window_summary_rows,
        "event_rows": event_rows_all,
        "station_detail_rows": station_detail_rows,
    }


def _plot_window_mae_curve(
    window_rows: list[dict[str, Any]],
    save_path: Path,
) -> None:
    """绘制时间窗口 vs MAE/RMSE 曲线"""
    if not window_rows:
        return
    _apply_pub_style()

    windows = [r["window_sec"] for r in window_rows]
    mae_vals = [r["mae"] for r in window_rows]
    rmse_vals = [r["rmse"] for r in window_rows]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(windows, mae_vals, marker="o", markersize=5, color=_OKABE_ITO[0],
            linewidth=1.5, label="MAE")
    ax.plot(windows, rmse_vals, marker="s", markersize=5, color=_OKABE_ITO[1],
            linewidth=1.5, label="RMSE")
    ax.set_xlabel("Time Window (s)")
    ax.set_ylabel("Magnitude Error ($M_w$)")
    ax.set_title("Magnitude Estimation vs. Time Window")
    ax.legend()
    ax.grid(True, axis="both", linestyle=":", linewidth=0.4, color="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_event_mw_evolution(
    event_rows: list[dict[str, Any]],
    save_path: Path,
) -> None:
    """绘制每个事件在不同时间窗口下的 Mw 演化面板"""
    if not event_rows:
        return
    _apply_pub_style()

    # 按事件分组
    events: dict[str, list[dict[str, Any]]] = {}
    for row in event_rows:
        events.setdefault(row["event"], []).append(row)

    n_events = len(events)
    ncols = min(3, n_events)
    nrows = math.ceil(n_events / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.8 * nrows), squeeze=False)

    for idx, (event_name, rows) in enumerate(sorted(events.items())):
        ax = axes[idx // ncols, idx % ncols]
        rows_sorted = sorted(rows, key=lambda r: r["window_sec"])
        windows = [r["window_sec"] for r in rows_sorted]
        mw_preds = [r["mw_pred_median"] for r in rows_sorted]
        mw_true = rows_sorted[0]["mw_true"]

        ax.plot(windows, mw_preds, marker="o", markersize=4, color=_OKABE_ITO[0], linewidth=1.2)
        ax.axhline(mw_true, color="black", linestyle="--", linewidth=0.9)
        ax.axhline(mw_true + 0.3, color="#9E9E9E", linestyle=":", linewidth=0.7)
        ax.axhline(mw_true - 0.3, color="#9E9E9E", linestyle=":", linewidth=0.7)
        ax.set_title(event_name, fontsize=7)
        ax.set_xlabel("Window (s)", fontsize=7)
        ax.set_ylabel("$M_w$", fontsize=7)
        ax.set_ylim(mw_true - 1.5, mw_true + 1.0)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.4, color="#CCCCCC")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 隐藏多余子图
    for idx in range(n_events, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("Event $M_w$ Convergence vs. Time Window", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2.1 时间窗口演化分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-dir", required=True, help="模型目录（含 best_model.pth + config.yaml）")
    parser.add_argument("--event-data-root", required=True, help="unseen 事件数据根目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument(
        "--windows", type=float, nargs="+",
        default=[30, 60, 90, 120, 180, 300, 600],
        help="时间窗口列表（秒），默认: 30 60 90 120 180 300 600",
    )
    parser.add_argument("--radial-peak-min-cm", type=float, default=None,
                        help="可选，覆盖径向峰值阈值")
    args = parser.parse_args()

    event_dirs = discover_event_dirs(Path(args.event_data_root))
    if not event_dirs:
        print(f"错误: 未在 {args.event_data_root} 下找到事件目录")
        sys.exit(1)

    print(f"模型目录: {args.model_dir}")
    print(f"事件数: {len(event_dirs)}")
    print(f"时间窗口: {args.windows}")

    run_latency_analysis(
        model_dir=Path(args.model_dir),
        event_dirs=event_dirs,
        output_dir=Path(args.output_dir),
        windows=args.windows,
        radial_peak_min_cm_override=args.radial_peak_min_cm,
    )


if __name__ == "__main__":
    main()
