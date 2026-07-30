#!/usr/bin/env python3
"""Publish the frozen Phase66 internal-test and eight-event report in Chinese."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
PROJECT_HOME = REPO_ROOT.parent.parent
DEFAULT_RUN_ROOT = (
    PROJECT_HOME
    / "runs"
    / "phase66-stateful-test-external-report-20260730T0521Z-9ee21b2"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "docs"
    / "reports"
    / "phase66-stateful-test-external-zh"
)
SOURCE_MODEL_COMMIT = "9ee21b22116a6bfab9f0b93e808c7805e241b465"
TARGET_MAE = 0.15
COLORS = {
    "raw": "#5B6573",
    "phase66": "#D55E00",
    "blue": "#0072B2",
    "green": "#009E73",
    "purple": "#8E5AA9",
    "yellow": "#E69F00",
    "red": "#B42318",
    "grid": "#D8DEE6",
    "ink": "#20262E",
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_plotting() -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.is_file():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        mpl.rcParams["font.family"] = family
    mpl.rcParams.update(
        {
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titleweight": "normal",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
        }
    )


def _style_axis(axis: Any) -> None:
    axis.grid(color=COLORS["grid"], linewidth=0.7)
    axis.set_axisbelow(True)


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _stable_horizon(rows: Sequence[Mapping[str, str]], key: str) -> int | None:
    for index, row in enumerate(rows):
        if all(float(item[key]) <= TARGET_MAE for item in rows[index:]):
            return int(float(row["observation_horizon_sec"]))
    return None


def _format_horizon(value: int | None) -> str:
    return ">200 s" if value is None else f"{value} s"


def _pct_change(before: float, after: float) -> float:
    return 100.0 * (after / before - 1.0)


def _copy_public_tables(run_root: Path, output_dir: Path) -> dict[str, str]:
    mapping = {
        "training/training_loss_by_epoch.csv": "training_loss_by_epoch.csv",
        "internal/horizon_metrics.csv": "internal_horizon_metrics.csv",
        "internal/event_predictions.csv": "internal_event_predictions.csv",
        "internal/endpoint_station_predictions.csv": (
            "internal_endpoint_station_predictions.csv"
        ),
        "internal/event_convergence.csv": "internal_event_convergence.csv",
        "internal/trajectory_diagnostics.csv": (
            "internal_trajectory_diagnostics.csv"
        ),
        "external/horizon_metrics.csv": "external_horizon_metrics.csv",
        "external/event_predictions.csv": "external_event_predictions.csv",
        "external/station_predictions.csv": "external_station_predictions.csv",
        "external/endpoint_station_predictions.csv": (
            "external_endpoint_station_predictions.csv"
        ),
        "external/event_convergence.csv": "external_event_convergence.csv",
        "external/trajectory_diagnostics.csv": (
            "external_trajectory_diagnostics.csv"
        ),
    }
    hashes: dict[str, str] = {}
    for source_name, destination_name in mapping.items():
        source = run_root / source_name
        if not source.is_file():
            raise FileNotFoundError(f"missing report source: {source}")
        destination = output_dir / destination_name
        shutil.copy2(source, destination)
        hashes[destination_name] = _sha256(destination)
    return hashes


def _plot_training_loss(rows: Sequence[Mapping[str, str]], output: Path) -> None:
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    total = np.asarray([float(row["train_total_normalized_loss"]) for row in rows])
    selected_epoch = 26
    selected_index = int(np.where(epochs == selected_epoch)[0][0])
    minimum_index = int(np.argmin(total))
    components = (
        ("train_endpoint_science_weighted_normalized", "终点科学", COLORS["blue"]),
        ("train_released_sequence_weighted_normalized", "递归序列", COLORS["green"]),
        ("train_endpoint_teacher_weighted_normalized", "终点 teacher", COLORS["purple"]),
        (
            "train_post60_target_overshoot_weighted_normalized",
            "60 秒后过冲",
            COLORS["phase66"],
        ),
        ("train_downward_step_weighted_normalized", "向下单步", COLORS["yellow"]),
        (
            "train_multiscale_downward_weighted_normalized",
            "多尺度回落",
            "#56B4E9",
        ),
        (
            "train_confirmed_history_weighted_normalized",
            "确认历史",
            "#CC79A7",
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))
    axes[0].plot(epochs, total, color=COLORS["blue"], linewidth=2.0)
    axes[0].scatter(
        [epochs[minimum_index]],
        [total[minimum_index]],
        color=COLORS["green"],
        s=58,
        zorder=3,
    )
    axes[0].scatter(
        [selected_epoch],
        [total[selected_index]],
        color=COLORS["phase66"],
        marker="D",
        s=58,
        zorder=3,
    )
    axes[0].annotate(
        f"最低训练 loss: e{epochs[minimum_index]} / {total[minimum_index]:.3f}",
        (epochs[minimum_index], total[minimum_index]),
        xytext=(-122, 18),
        textcoords="offset points",
        fontsize=9,
    )
    axes[0].annotate(
        f"冻结 checkpoint: e26 / {total[selected_index]:.3f}",
        (selected_epoch, total[selected_index]),
        xytext=(0, 55),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
    axes[0].set_title("训练集总归一化目标")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("加权归一化 loss")
    _style_axis(axes[0])

    for key, label, color in components:
        axes[1].plot(
            epochs,
            [float(row[key]) for row in rows],
            label=label,
            color=color,
            linewidth=1.55,
        )
    axes[1].axvline(
        selected_epoch,
        color=COLORS["red"],
        linestyle="--",
        linewidth=1.1,
        label="冻结 epoch26",
    )
    axes[1].set_title("七项训练目标贡献")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("加权归一化贡献")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    _style_axis(axes[1])
    figure.suptitle("Phase66：训练 loss 最低点与 validation 最接近门槛点不同", y=1.02)
    figure.tight_layout()
    _save_figure(figure, output / "01_training_loss")


def _plot_internal_summary(
    raw: Mapping[str, Any],
    phase66: Mapping[str, Any],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    endpoint_specs = (
        ("endpoint_event_mae", "Event MAE"),
        ("endpoint_station_mae", "Station MAE"),
    )
    x = np.arange(len(endpoint_specs))
    width = 0.34
    raw_values = [float(raw[key]) for key, _ in endpoint_specs]
    phase66_values = [float(phase66[key]) for key, _ in endpoint_specs]
    raw_bars = axes[0].bar(
        x - width / 2,
        raw_values,
        width,
        color=COLORS["raw"],
        label="Phase39 离线提案",
    )
    phase66_bars = axes[0].bar(
        x + width / 2,
        phase66_values,
        width,
        color=COLORS["phase66"],
        label="Phase66 递归流式",
    )
    axes[0].set_xticks(x, [label for _, label in endpoint_specs])
    axes[0].set_ylabel("MAE（Mw）")
    axes[0].set_title("internal test 200 秒端点")
    for bars in (raw_bars, phase66_bars):
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0].set_ylim(0.0, max(raw_values + phase66_values) * 1.22)
    axes[0].legend(frameon=False)
    _style_axis(axes[0])

    stability_specs = (
        ("late_event_abs_step_p95_mw", "后期事件跳变 p95"),
        ("event_downward_step_max_mw", "最大向下单步"),
        ("event_peak_to_final_p95_mw", "峰后回落 p95"),
    )
    raw_stability = [float(raw[key]) for key, _ in stability_specs]
    phase66_stability = [float(phase66[key]) for key, _ in stability_specs]
    raw_bars = axes[1].bar(
        x - width / 2 if len(x) == 3 else np.arange(3) - width / 2,
        raw_stability,
        width,
        color=COLORS["raw"],
        label="Phase39 离线提案",
    )
    stability_x = np.arange(len(stability_specs))
    phase66_bars = axes[1].bar(
        stability_x + width / 2,
        phase66_stability,
        width,
        color=COLORS["phase66"],
        label="Phase66 递归流式",
    )
    axes[1].set_xticks(
        stability_x,
        [label for _, label in stability_specs],
        rotation=12,
        ha="right",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Mw（对数刻度）")
    axes[1].set_title("internal test 轨迹稳定性")
    for bars in (raw_bars, phase66_bars):
        for bar in bars:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.12,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[1].legend(frameon=False)
    _style_axis(axes[1])
    figure.suptitle("internal test：轨迹稳定很多，但端点误差变大", y=1.02)
    figure.tight_layout()
    _save_figure(figure, output / "02_internal_test")


def _plot_external_overall(rows: Sequence[Mapping[str, str]], output: Path) -> None:
    horizons = np.asarray([float(row["observation_horizon_sec"]) for row in rows])
    raw_event = np.asarray([float(row["raw_event_mae"]) for row in rows])
    phase66_event = np.asarray([float(row["phase66_event_mae"]) for row in rows])
    raw_station = np.asarray([float(row["raw_station_mae"]) for row in rows])
    phase66_station = np.asarray(
        [float(row["phase66_station_mae"]) for row in rows]
    )
    raw_stable = _stable_horizon(rows, "raw_event_mae")
    phase66_stable = _stable_horizon(rows, "phase66_event_mae")
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))
    axes[0].plot(
        horizons,
        raw_event,
        color=COLORS["raw"],
        linewidth=1.8,
        label="Phase39 离线提案",
    )
    axes[0].plot(
        horizons,
        phase66_event,
        color=COLORS["phase66"],
        linewidth=2.0,
        label="Phase66 递归流式",
    )
    axes[0].axhline(
        TARGET_MAE,
        color=COLORS["red"],
        linestyle="--",
        linewidth=1.2,
        label="0.15 Mw",
    )
    axes[0].set_title(
        "八事件等权 Event MAE\n"
        f"持续≤0.15：Phase39 {_format_horizon(raw_stable)} / "
        f"Phase66 {_format_horizon(phase66_stable)}"
    )
    axes[0].set_xlabel("观测时长（s）")
    axes[0].set_ylabel("Event MAE（Mw）")
    axes[0].legend(frameon=False)
    _style_axis(axes[0])

    axes[1].plot(
        horizons,
        raw_station,
        color=COLORS["raw"],
        linewidth=1.8,
        label="Phase39 离线提案",
    )
    axes[1].plot(
        horizons,
        phase66_station,
        color=COLORS["phase66"],
        linewidth=2.0,
        label="Phase66 递归流式",
    )
    axes[1].set_title("八事件 Station MAE")
    axes[1].set_xlabel("观测时长（s）")
    axes[1].set_ylabel("Station MAE（Mw）")
    axes[1].legend(frameon=False)
    _style_axis(axes[1])
    figure.suptitle("八事件：Phase66 输出更平稳，但最终精度明显下降", y=1.02)
    figure.tight_layout()
    _save_figure(figure, output / "03_external_overall")


def _plot_external_events(rows: Sequence[Mapping[str, str]], output: Path) -> None:
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["event"]), []).append(row)
    figure, axes = plt.subplots(4, 2, figsize=(12.6, 12.2), sharex=True)
    for axis, event in zip(axes.flat, sorted(grouped), strict=True):
        sequence = sorted(
            grouped[event],
            key=lambda row: float(row["observation_horizon_sec"]),
        )
        horizons = [float(row["observation_horizon_sec"]) for row in sequence]
        raw = [float(row["raw_mw_pred_median"]) for row in sequence]
        phase66 = [float(row["phase66_mw_pred_median"]) for row in sequence]
        catalog = float(sequence[-1]["mw_catalog"])
        axis.plot(
            horizons,
            raw,
            color=COLORS["raw"],
            linewidth=1.25,
            label="Phase39",
        )
        axis.plot(
            horizons,
            phase66,
            color=COLORS["phase66"],
            linewidth=1.7,
            label="Phase66",
        )
        axis.axhline(catalog, color=COLORS["red"], linestyle="--", linewidth=1.0)
        axis.fill_between(
            horizons,
            catalog - TARGET_MAE,
            catalog + TARGET_MAE,
            color=COLORS["green"],
            alpha=0.08,
        )
        axis.set_title(event, fontsize=10)
        axis.text(
            0.98,
            0.05,
            f"200s: {raw[-1]:.3f} → {phase66[-1]:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
        _style_axis(axis)
    for axis in axes[-1]:
        axis.set_xlabel("观测时长（s）")
    for axis in axes[:, 0]:
        axis.set_ylabel("事件中位数 Mw")
    axes.flat[0].legend(loc="upper right", frameon=False, fontsize=8)
    figure.suptitle(
        "训练未包含的 8 事件逐秒输出（发布时刻 = 观测时长 + 6 s）",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    _save_figure(figure, output / "04_external_event_trajectories")


def _plot_revision_diagnostics(
    rows: Sequence[Mapping[str, str]],
    output: Path,
) -> None:
    ordered = sorted(rows, key=lambda row: str(row["event"]))
    events = [str(row["event"]) for row in ordered]
    y = np.arange(len(events))
    specs = (
        (
            "raw_max_down_step_after_60_mw",
            "phase66_max_down_step_after_60_mw",
            "60 秒后最大单步向下修正",
        ),
        (
            "raw_peak_to_final_drop_after_60_mw",
            "phase66_peak_to_final_drop_after_60_mw",
            "60 秒后峰值到 200 秒回落",
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.1), sharey=True)
    for axis, (raw_key, phase66_key, title) in zip(axes, specs, strict=True):
        raw = np.asarray([float(row[raw_key]) for row in ordered])
        phase66 = np.asarray([float(row[phase66_key]) for row in ordered])
        for row_index in range(len(events)):
            axis.plot(
                [max(raw[row_index], 1.0e-4), max(phase66[row_index], 1.0e-4)],
                [row_index, row_index],
                color=COLORS["grid"],
                linewidth=1.1,
            )
        axis.scatter(
            np.maximum(raw, 1.0e-4),
            y,
            color=COLORS["raw"],
            marker="o",
            label="Phase39 离线提案",
            zorder=3,
        )
        axis.scatter(
            np.maximum(phase66, 1.0e-4),
            y,
            color=COLORS["phase66"],
            marker="D",
            label="Phase66 递归流式",
            zorder=3,
        )
        axis.set_xscale("log")
        axis.set_xlabel("Mw（对数刻度）")
        axis.set_title(title)
        _style_axis(axis)
    axes[0].set_yticks(y, events)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle("Phase66 显著限制逐秒下降和累计峰后回落", y=1.01)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    _save_figure(figure, output / "05_external_revision_diagnostics")


def _typed_convergence_row(row: Mapping[str, str]) -> dict[str, Any]:
    integer_keys = {
        "raw_stable_observation_sec",
        "phase66_stable_observation_sec",
        "station_count_200s",
    }
    output: dict[str, Any] = {}
    for key, value in row.items():
        if key == "event":
            output[key] = value
        elif value == "":
            output[key] = None
        elif key in integer_keys:
            output[key] = int(float(value))
        else:
            output[key] = float(value)
    return output


def _render_report(summary: Mapping[str, Any]) -> str:
    training = summary["training"]
    selected = training["closest_epoch_metrics"]
    internal_raw = summary["internal"]["raw"]
    internal_phase66 = summary["internal"]["phase66"]
    external_raw = summary["external"]["raw"]
    external_phase66 = summary["external"]["phase66"]
    event_lines = []
    for row in summary["external"]["events"]:
        event_lines.append(
            "| {event} | {catalog:.1f} | {raw:.3f} | {phase66:.3f} | "
            "{raw_error:.3f} | {phase66_error:.3f} | {phase66_stable} |".format(
                event=row["event"],
                catalog=float(row["mw_catalog"]),
                raw=float(row["raw_final_mw"]),
                phase66=float(row["phase66_final_mw"]),
                raw_error=float(row["raw_final_abs_error"]),
                phase66_error=float(row["phase66_final_abs_error"]),
                phase66_stable=_format_horizon(
                    row["phase66_stable_observation_sec"]
                ),
            )
        )
    component_rows = []
    labels = {
        "endpoint_science": "Endpoint science",
        "released_sequence": "Recurrent sequence",
        "endpoint_teacher": "Endpoint teacher",
        "post60_target_overshoot": "Post-60 overshoot",
        "downward_step": "Downward step",
        "multiscale_downward": "Multiscale downward",
        "confirmed_history": "Confirmed history",
    }
    for name in training["loss_weights"]:
        component_rows.append(
            "| {label} | {raw:.6f} | {weighted:.6f} |".format(
                label=labels[name],
                raw=float(selected[f"train_{name}_raw"]),
                weighted=float(
                    selected[f"train_{name}_weighted_normalized"]
                ),
            )
        )
    return f"""# Phase66 模型内递归流式评估：internal test 与 8 事件

> 固定对象：Phase66 seed17 epoch26，完整 `PINNModel` 检查点。它以 Phase39 Glehman+GI 的每秒完整 STF 提案为证据，在模型内部用上一秒 STF/矩状态递归更新；没有 adapter、后处理单调夹紧或 ensemble。该 checkpoint 未通过原 validation Event MAE 门槛，差 `0.008906 Mw`，本报告依据用户明确授权进行一次性测试。

## 结论

- **internal test**：Event MAE 从 {internal_raw['endpoint_event_mae']:.6f} 变为 {internal_phase66['endpoint_event_mae']:.6f} Mw，Station MAE 从 {internal_raw['endpoint_station_mae']:.6f} 变为 {internal_phase66['endpoint_station_mae']:.6f} Mw，端点精度均变差。
- **8 事件开发集**：Event MAE 从 {external_raw['endpoint_event_mae']:.6f} 变为 {external_phase66['endpoint_event_mae']:.6f} Mw，Station MAE 从 {external_raw['endpoint_station_mae']:.6f} 变为 {external_phase66['endpoint_station_mae']:.6f} Mw；仅 {summary['external']['improved_event_count']}/8 事件最终绝对误差改善。
- **流式稳定性明显改善**：8 事件最大向下单步从 {external_raw['event_downward_step_max_mw']:.6f} 降至 {external_phase66['event_downward_step_max_mw']:.6f} Mw，峰后回落 p95 从 {external_raw['event_peak_to_final_p95_mw']:.6f} 降至 {external_phase66['event_peak_to_final_p95_mw']:.6f} Mw，后期事件跳变 p95 从 {external_raw['late_event_abs_step_p95_mw']:.6f} 降至 {external_phase66['late_event_abs_step_p95_mw']:.6f} Mw。
- **当前判断**：Phase66 已解决“逐秒大幅先升后降”的主要稳定性问题，但在未用于训练的 8 事件上出现高偏差，尤其是 Sand 和 Xizang。因此它暂时不能替代 Phase39 作为精度最好的模型，也不能只凭轨迹变平稳就认定泛化更好。

## 1. 训练损失与 checkpoint

![训练损失](figures/01_training_loss.png)

[PDF 图件](figures/01_training_loss.pdf)

Phase66 的最低训练总 loss 出现在 epoch{training['minimum_training_total']['epoch']}（{training['minimum_training_total']['value']:.6f}），而冻结 checkpoint 是 epoch26（{selected['train_total_normalized_loss']:.6f}）。因此本轮不存在“第四轮训练 loss 最低”的现象；epoch26 是按 validation 最接近全部门槛选出的，而不是按训练 loss 最低选出。

| epoch26 分量 | 原始值 | 加权归一化贡献 |
|---|---:|---:|
{chr(10).join(component_rows)}
| **总计** | — | **{selected['train_total_normalized_loss']:.6f}** |

## 2. internal test

![internal test](figures/02_internal_test.png)

[PDF 图件](figures/02_internal_test.pdf)

| 指标 | Phase39 离线提案 | Phase66 递归流式 | 变化 |
|---|---:|---:|---:|
| Event MAE | {internal_raw['endpoint_event_mae']:.6f} | {internal_phase66['endpoint_event_mae']:.6f} | {internal_phase66['endpoint_event_mae'] - internal_raw['endpoint_event_mae']:+.6f} |
| Station MAE | {internal_raw['endpoint_station_mae']:.6f} | {internal_phase66['endpoint_station_mae']:.6f} | {internal_phase66['endpoint_station_mae'] - internal_raw['endpoint_station_mae']:+.6f} |
| 后期 Event step p95 | {internal_raw['late_event_abs_step_p95_mw']:.6f} | {internal_phase66['late_event_abs_step_p95_mw']:.6f} | {_pct_change(internal_raw['late_event_abs_step_p95_mw'], internal_phase66['late_event_abs_step_p95_mw']):+.1f}% |
| 最大向下单步 | {internal_raw['event_downward_step_max_mw']:.6f} | {internal_phase66['event_downward_step_max_mw']:.6f} | {_pct_change(internal_raw['event_downward_step_max_mw'], internal_phase66['event_downward_step_max_mw']):+.1f}% |
| 峰后回落 p95 | {internal_raw['event_peak_to_final_p95_mw']:.6f} | {internal_phase66['event_peak_to_final_p95_mw']:.6f} | {_pct_change(internal_raw['event_peak_to_final_p95_mw'], internal_phase66['event_peak_to_final_p95_mw']):+.1f}% |

这里的 internal test 是 `within_event_station`：同一地震的不同台站分散在 train/validation/test，因此它衡量同事件未见台站插值，不等于未见事件泛化。

## 3. 训练未包含的 8 事件

![八事件总体逐秒指标](figures/03_external_overall.png)

[PDF 图件](figures/03_external_overall.pdf)

![八事件逐秒轨迹](figures/04_external_event_trajectories.png)

[PDF 图件](figures/04_external_event_trajectories.pdf)

| 事件 | 参考 Mw | Phase39 200 s | Phase66 200 s | Phase39 绝对误差 | Phase66 绝对误差 | Phase66 持续收敛 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(event_lines)}

Phase66 在 Iquique、Nepal、Samos 上改善最终绝对误差；Kodiak、Luding、Mandalay、Sand、Xizang 变差。Sand 和 Xizang 的 200 秒预测分别达到 7.717 和 7.592，说明递归状态保留了偏高的早期估计，虽然不再大幅回落，但也未能充分向正确端点修正。

## 4. 向下修正与峰后回落

![逐事件修正诊断](figures/05_external_revision_diagnostics.png)

[PDF 图件](figures/05_external_revision_diagnostics.pdf)

该图把“单步跳动”和“长期累计回落”分开。Phase66 对所有 8 个事件都显著压低了两者；这证明模型内部递归约束有效，但也解释了端点高偏差：对早期高估的向下纠正能力可能过弱。

## 5. 数据角色与限制

- Phase66 原 validation gate 没有正式通过；本报告使用预先冻结的 seed17 epoch26，不因 test 或 8 事件结果更换 checkpoint。
- internal test 已按用户授权一次性打开，仍属于同事件未见台站插值。
- 8 事件没有进入模型训练，因此对模型而言是未训练事件；但这些事件已被此前多轮开发反复使用，统计角色必须写作 `development_validation`，不能作为新的无偏盲测证明。
- Phase39 原始端点复现门槛通过：最大台站预测差 {summary['external']['endpoint_gate']['max_station_prediction_abs_diff_mw']:.2e} Mw，说明本报告的精度退化不是台站错位或基线回放变化造成的。
- grouped test 没有打开；本报告结果不得用于继续选择 Phase67。

## 6. 可下载工件

- [训练逐 epoch loss](training_loss_by_epoch.csv)
- [internal test 逐秒总体指标](internal_horizon_metrics.csv)
- [internal test 逐事件逐秒输出](internal_event_predictions.csv)
- [8 事件逐事件逐秒输出](external_event_predictions.csv)
- [8 事件逐台站逐秒输出](external_station_predictions.csv)
- [8 事件逐秒总体指标](external_horizon_metrics.csv)
- [8 事件收敛时间](external_event_convergence.csv)
- [逐事件回落诊断](external_trajectory_diagnostics.csv)
- [机器可读总摘要](summary.json)
- [发布清单与 SHA-256](publication_manifest.json)
- [冻结评估器](../../../scripts/evaluation/evaluate_phase66_stateful_streaming.py)
- [可复现绘图脚本](../../../scripts/plotting/plot_phase66_stateful_report_zh.py)

原始 Phase39/Phase66 STF rate 立方体保留在本机正式 run 目录，不提交 GitHub；其 SHA-256 已写入摘要。全仓实现回归：`{summary['full_regression']}`。
"""


def publish(
    run_root: Path,
    output_dir: Path,
    *,
    full_regression: str,
) -> dict[str, Any]:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    training = _read_json(run_root / "training" / "summary.json")
    internal = _read_json(run_root / "internal" / "summary.json")
    external = _read_json(run_root / "external" / "summary.json")
    if training.get("status") != "complete":
        raise ValueError("training trace is incomplete")
    if internal.get("status") != "complete" or not internal.get(
        "internal_test_iterated"
    ):
        raise ValueError("internal test artifact is incomplete")
    if external.get("status") != "complete" or not external.get(
        "external_data_loaded"
    ):
        raise ValueError("external artifact is incomplete")
    if internal.get("grouped_test_loaded") is not False or external.get(
        "grouped_test_loaded"
    ) is not False:
        raise ValueError("grouped test must remain unopened")

    table_hashes = _copy_public_tables(run_root, output_dir)
    training_rows = _read_csv(output_dir / "training_loss_by_epoch.csv")
    external_horizon_rows = _read_csv(output_dir / "external_horizon_metrics.csv")
    external_event_rows = _read_csv(output_dir / "external_event_predictions.csv")
    external_convergence_rows = _read_csv(
        output_dir / "external_event_convergence.csv"
    )
    external_diagnostics_rows = _read_csv(
        output_dir / "external_trajectory_diagnostics.csv"
    )
    raw_stable = _stable_horizon(external_horizon_rows, "raw_event_mae")
    phase66_stable = _stable_horizon(external_horizon_rows, "phase66_event_mae")
    improved_events = sum(
        float(row["phase66_final_abs_error"])
        < float(row["raw_final_abs_error"])
        for row in external_convergence_rows
    )
    summary = {
        "status": "complete",
        "candidate": "Phase66 seed17 epoch26 frozen closest checkpoint",
        "evaluation_role": "internal_test_and_development_validation_reporting",
        "training": training,
        "internal": {
            "station_count": internal["station_count"],
            "event_count": internal["event_count"],
            "raw": internal["raw_metrics"],
            "phase66": internal["phase66_metrics"],
            "artifact_sha256": internal["artifact_sha256"],
        },
        "external": {
            "station_count": external["station_count"],
            "event_count": external["event_count"],
            "raw": external["raw_metrics"],
            "phase66": external["phase66_metrics"],
            "endpoint_gate": external["endpoint_raw_reproduction_gate"],
            "raw_stable_event_mae_observation_sec": raw_stable,
            "phase66_stable_event_mae_observation_sec": phase66_stable,
            "improved_event_count": improved_events,
            "events": [
                _typed_convergence_row(row) for row in external_convergence_rows
            ],
            "trajectory_diagnostics": external_diagnostics_rows,
            "artifact_sha256": external["artifact_sha256"],
        },
        "processing_delay_sec": 6.0,
        "source_model_commit": SOURCE_MODEL_COMMIT,
        "full_regression": full_regression,
        "formal_validation_gate_passed": False,
        "grouped_test_loaded": False,
        "public_table_sha256": table_hashes,
    }
    _write_json(output_dir / "summary.json", summary)

    _configure_plotting()
    _plot_training_loss(training_rows, figures)
    _plot_internal_summary(
        internal["raw_metrics"],
        internal["phase66_metrics"],
        figures,
    )
    _plot_external_overall(external_horizon_rows, figures)
    _plot_external_events(external_event_rows, figures)
    _plot_revision_diagnostics(external_diagnostics_rows, figures)
    (output_dir / "README.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )

    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "publication_manifest.json"
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_role": "internal_test_and_development_validation_reporting",
        "source_model_commit": SOURCE_MODEL_COMMIT,
        "run_root": str(run_root),
        "generator": {
            "path": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(SCRIPT_PATH),
        },
        "files": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    _write_json(output_dir / "publication_manifest.json", manifest)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Chinese Phase66 stateful streaming report."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--full-regression",
        default="pending final repository regression",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = publish(
        args.run_root.resolve(),
        args.output_dir.resolve(),
        full_regression=str(args.full_regression),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(args.output_dir.resolve()),
                "external_event_mae": summary["external"]["phase66"][
                    "endpoint_event_mae"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
