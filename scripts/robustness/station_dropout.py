#!/usr/bin/env python
"""E2.3 网络稀疏性测试 — 评估台站缺失时的性能退化

功能：
  1. 使用已训练好的模型（无需重新训练）
  2. 对每个事件，随机丢弃 20%/40%/60%/80% 的台站
  3. 每种丢弃率重复 100 次取平均
  4. 追踪台站保留比例 vs 事件级 MAE

用法示例：

  python scripts/robustness/station_dropout.py \
      --model-dir ./outputs_experiments/phase1/models/20260412_141044 \
      --event-data-root "/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/GNSS_EQDATA" \
      --output-dir ./outputs_experiments/phase2/e2_3_dropout

  # 自定义丢弃率和重复次数
  python scripts/robustness/station_dropout.py \
      --model-dir ./outputs_experiments/phase1/models/20260412_141044 \
      --event-data-root "/path/to/events" \
      --output-dir ./outputs_experiments/phase2/e2_3_dropout \
      --drop-rates 0.0 0.2 0.4 0.6 0.8 \
      --n-repeats 100
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
from src.evaluation.evaluate import _ensure_time_steps
from src.evaluation.evaluate_unseen import (
    EventBundle,
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


def run_station_dropout(
    *,
    model_dir: Path,
    event_dirs: list[Path],
    output_dir: Path,
    drop_rates: list[float],
    n_repeats: int = 100,
    seed: int = 42,
    radial_peak_min_cm_override: float | None = None,
) -> dict[str, Any]:
    """对不同台站丢弃率执行推断"""
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

    # 加载事件并预计算所有有效台站的预测
    print("加载事件数据并预计算台站级预测...")
    event_station_preds: dict[str, dict[str, Any]] = {}

    with torch.no_grad():
        for ed in event_dirs:
            bundle = load_event_bundle(ed)
            event_label = _format_event_display_name(
                event_name=bundle.event_name,
                event_dir_name=bundle.event_dir_name,
                magnitude=bundle.magnitude,
            )
            station_mw: list[float] = []

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
                station_mw.append(mw_pred)

            if station_mw:
                event_station_preds[event_label] = {
                    "mw_true": bundle.magnitude,
                    "station_preds": np.array(station_mw),
                    "mechanism": bundle.mechanism,
                    "n_total": len(station_mw),
                }
                print(f"  {event_label}: {len(station_mw)} 台站")

    n_events = len(event_station_preds)
    print(f"共 {n_events} 个事件有有效台站")

    rng = np.random.RandomState(seed)
    dropout_results: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for drop_rate in sorted(drop_rates):
        keep_rate = 1.0 - drop_rate
        print(f"\n── 丢弃率: {drop_rate * 100:.0f}% (保留 {keep_rate * 100:.0f}%) ──")

        repeat_maes: list[float] = []
        repeat_rmses: list[float] = []

        for rep in range(n_repeats):
            errors: list[float] = []

            for ev_name, ev_data in event_station_preds.items():
                all_preds = ev_data["station_preds"]
                n_total = len(all_preds)
                n_keep = max(1, int(round(n_total * keep_rate)))

                if drop_rate <= 0:
                    # 保留全部
                    subset = all_preds
                else:
                    idx = rng.choice(n_total, size=n_keep, replace=False)
                    subset = all_preds[idx]

                mw_median = float(np.median(subset))
                err = mw_median - ev_data["mw_true"]
                errors.append(err)

                if rep == 0:
                    detail_rows.append({
                        "drop_rate": drop_rate,
                        "keep_rate": keep_rate,
                        "repeat": rep,
                        "event": ev_name,
                        "mw_true": ev_data["mw_true"],
                        "mw_pred_median": mw_median,
                        "error": err,
                        "n_total": n_total,
                        "n_kept": len(subset),
                        "mechanism": ev_data["mechanism"],
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

        dropout_results.append({
            "drop_rate": drop_rate,
            "keep_rate": keep_rate,
            "mae_mean": mean_mae,
            "mae_std": std_mae,
            "rmse_mean": mean_rmse,
            "rmse_std": std_rmse,
            "n_repeats": n_repeats,
            "n_events": n_events,
        })

    # ── 保存结果 ──
    output_dir.mkdir(parents=True, exist_ok=True)

    # 汇总 CSV
    summary_csv = output_dir / "dropout_summary.csv"
    if dropout_results:
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(dropout_results[0].keys()))
            writer.writeheader()
            writer.writerows(dropout_results)

    # 事件级详细 CSV
    detail_csv = output_dir / "dropout_event_detail.csv"
    if detail_rows:
        with detail_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)

    # 绘图
    _plot_dropout_curve(dropout_results, output_dir / "dropout_curve.png")
    _plot_dropout_event_detail(detail_rows, output_dir / "dropout_event_detail.png")

    print(f"\n结果已保存至: {output_dir}")
    return {"dropout_summary": dropout_results, "detail_rows": detail_rows}


def _plot_dropout_curve(
    results: list[dict[str, Any]],
    save_path: Path,
) -> None:
    """绘制台站保留比例 vs MAE/RMSE 曲线"""
    if not results:
        return
    _apply_pub_style()

    keep_pcts = [r["keep_rate"] * 100 for r in results]
    mae_means = [r["mae_mean"] for r in results]
    mae_stds = [r["mae_std"] for r in results]
    rmse_means = [r["rmse_mean"] for r in results]
    rmse_stds = [r["rmse_std"] for r in results]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.errorbar(keep_pcts, mae_means, yerr=mae_stds, marker="o", markersize=5,
                color=_OKABE_ITO[0], linewidth=1.5, capsize=3, label="MAE")
    ax.errorbar(keep_pcts, rmse_means, yerr=rmse_stds, marker="s", markersize=5,
                color=_OKABE_ITO[1], linewidth=1.5, capsize=3, label="RMSE")

    ax.set_xlabel("Station Retention Rate (%)")
    ax.set_ylabel("Magnitude Error ($M_w$)")
    ax.set_title("Station Sparsity: MAE/RMSE vs. Retention Rate")
    ax.set_xlim(15, 105)
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, axis="both", linestyle=":", linewidth=0.4, color="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_dropout_event_detail(
    detail_rows: list[dict[str, Any]],
    save_path: Path,
) -> None:
    """绘制每个事件在不同丢弃率下的误差"""
    if not detail_rows:
        return
    _apply_pub_style()

    # 按事件分组
    events: dict[str, list[dict[str, Any]]] = {}
    for row in detail_rows:
        events.setdefault(row["event"], []).append(row)

    n_events = len(events)
    if n_events == 0:
        return

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for idx, (ev_name, rows) in enumerate(sorted(events.items())):
        rows_sorted = sorted(rows, key=lambda r: r["keep_rate"])
        keep_pcts = [r["keep_rate"] * 100 for r in rows_sorted]
        abs_errors = [abs(r["error"]) for r in rows_sorted]
        color = _OKABE_ITO[idx % len(_OKABE_ITO)]
        ax.plot(keep_pcts, abs_errors, marker="o", markersize=3, linewidth=1.0,
                color=color, alpha=0.7, label=ev_name)

    ax.set_xlabel("Station Retention Rate (%)")
    ax.set_ylabel("$|M_w$ Error$|$")
    ax.set_title("Per-Event Error vs. Station Retention")
    ax.invert_xaxis()
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, color="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2.3 网络稀疏性测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-dir", required=True, help="模型目录")
    parser.add_argument("--event-data-root", required=True, help="unseen 事件数据根目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument(
        "--drop-rates", type=float, nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8],
        help="台站丢弃率列表，默认: 0.0 0.2 0.4 0.6 0.8",
    )
    parser.add_argument("--n-repeats", type=int, default=100,
                        help="每种丢弃率的重复次数（默认: 100）")
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
    print(f"丢弃率: {args.drop_rates}")
    print(f"重复次数: {args.n_repeats}")

    run_station_dropout(
        model_dir=Path(args.model_dir),
        event_dirs=event_dirs,
        output_dir=Path(args.output_dir),
        drop_rates=args.drop_rates,
        n_repeats=args.n_repeats,
        seed=args.seed,
        radial_peak_min_cm_override=args.radial_peak_min_cm,
    )


if __name__ == "__main__":
    main()
