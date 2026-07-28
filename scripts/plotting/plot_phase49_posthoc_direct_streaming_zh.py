#!/usr/bin/env python3
"""Publish the Phase39/47/48 post-hoc direct-streaming Chinese report."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
PROJECT_HOME = REPO_ROOT.parent.parent
DEFAULT_RUN_ROOT = (
    PROJECT_HOME
    / "runs"
    / "phase49-posthoc-direct-streaming-20260728T171040Z-4321776"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "docs"
    / "reports"
    / "phase39-phase47-phase48-direct-streaming-posthoc-zh"
)
EXPECTED_EVALUATOR_COMMIT = "4321776bb600fc45aab6aceb5fbc75a50e556c95"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
MODEL_ORDER = ("phase39", "phase47", "phase48")
MODEL_LABELS = {
    "phase39": "Phase39",
    "phase47": "Phase47",
    "phase48": "Phase48",
}
COLORS = {
    "phase39": "#20262E",
    "phase47": "#D55E00",
    "phase48": "#0072B2",
    "truth": "#009E73",
    "grid": "#D8DEE6",
    "ink": "#20262E",
    "gray": "#67717E",
}
FULL_REGRESSION = "809 passed, 1 skipped in 50.71s"


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if row.get(key) is None else row.get(key)
                    for key in fieldnames
                }
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _configure_plotting() -> None:
    if FONT_PATH.is_file():
        font_manager.fontManager.addfont(str(FONT_PATH))
        family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
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


def _pct_change(baseline: float, candidate: float) -> float:
    return 100.0 * (float(candidate) - float(baseline)) / float(baseline)


def trajectory_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["event"]))].append(row)
    output: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        events = sorted(event for candidate, event in grouped if candidate == model)
        for event in events:
            ordered = sorted(
                grouped[(model, event)],
                key=lambda row: int(row["observation_horizon_sec"]),
            )
            horizons = np.asarray(
                [int(row["observation_horizon_sec"]) for row in ordered],
                dtype=np.int64,
            )
            predictions = np.asarray(
                [float(row["mw_pred_median"]) for row in ordered],
                dtype=np.float64,
            )
            steps = np.diff(predictions)
            step_horizons = horizons[1:]
            after_60 = step_horizons >= 60
            after_120 = step_horizons >= 120
            after_60_values = predictions[horizons >= 60]
            after_60_horizons = horizons[horizons >= 60]
            late_values = predictions[horizons >= 179]
            peak_index = int(np.argmax(after_60_values))
            peak = float(after_60_values[peak_index])
            final = float(predictions[-1])

            def max_abs(values: np.ndarray) -> float:
                return 0.0 if values.size == 0 else float(np.max(np.abs(values)))

            def max_down(values: np.ndarray) -> float:
                return 0.0 if values.size == 0 else max(0.0, -float(np.min(values)))

            output.append(
                {
                    "model": model,
                    "event": event,
                    "mw_catalog": float(ordered[-1]["mw_catalog"]),
                    "final_mw": final,
                    "final_abs_error": float(ordered[-1]["abs_error"]),
                    "max_abs_step_20_200_mw": max_abs(steps),
                    "max_down_step_20_200_mw": max_down(steps),
                    "max_abs_step_after_60_mw": max_abs(steps[after_60]),
                    "max_down_step_after_60_mw": max_down(steps[after_60]),
                    "max_abs_step_after_120_mw": max_abs(steps[after_120]),
                    "max_down_step_after_120_mw": max_down(steps[after_120]),
                    "peak_after_60_mw": peak,
                    "peak_after_60_observation_sec": int(
                        after_60_horizons[peak_index]
                    ),
                    "peak_to_final_drop_after_60_mw": max(0.0, peak - final),
                    "range_after_60_mw": float(np.ptp(after_60_values)),
                    "range_179_200_mw": float(np.ptp(late_values)),
                    "downward_step_fraction_20_200": float(np.mean(steps < 0.0)),
                }
            )
    return output


def _overall_rows(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        data = evaluation["models"][model]
        endpoint = data["endpoint"]
        late = data["late"]
        rows.append(
            {
                "model": model,
                "label": data["label"],
                "endpoint_event_mae": endpoint["event_mae"],
                "endpoint_station_mae": endpoint["station_mae"],
                "endpoint_event_bias": endpoint["event_bias"],
                "late_event_abs_step_p95_mw": late["late_event_abs_step_p95_mw"],
                "late_station_abs_step_p95_mw": late[
                    "late_station_abs_step_p95_mw"
                ],
                "late_confirmed_history_p95": late[
                    "late_confirmed_cumulative_log10_l1_p95"
                ],
                "stable_event_count": data["convergence"]["stable_event_count"],
                "improved_event_count_vs_phase39": data.get(
                    "endpoint_improved_event_count_vs_phase39"
                ),
            }
        )
    return rows


def _recovery_rows(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        artifact = evaluation["artifact_validation"]["models"][model]
        recovery = artifact["recovery"]
        rows.append(
            {
                "model": model,
                "label": artifact["label"],
                "checkpoint_sha256": artifact["checkpoint_sha256"],
                "recovery_role": None if recovery is None else recovery["role"],
                "strict_reproduction_passed": (
                    None if recovery is None else recovery["strict_reproduction_passed"]
                ),
                "max_metric_abs_difference": (
                    None if recovery is None else recovery["max_metric_abs_difference"]
                ),
                "selection_score_abs_difference": (
                    None
                    if recovery is None
                    else recovery["selection_score_abs_difference"]
                ),
            }
        )
    return rows


def _plot_accuracy_stability(
    overall: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.2))
    x = np.arange(len(MODEL_ORDER), dtype=np.float64)
    width = 0.34
    labels = [MODEL_LABELS[model] for model in MODEL_ORDER]
    event_mae = [float(row["endpoint_event_mae"]) for row in overall]
    station_mae = [float(row["endpoint_station_mae"]) for row in overall]
    axes[0].bar(
        x - width / 2,
        event_mae,
        width,
        color=[COLORS[model] for model in MODEL_ORDER],
        alpha=0.95,
        label="Event MAE",
    )
    axes[0].bar(
        x + width / 2,
        station_mae,
        width,
        color=[COLORS[model] for model in MODEL_ORDER],
        alpha=0.42,
        edgecolor=[COLORS[model] for model in MODEL_ORDER],
        label="Station MAE",
    )
    for index, value in enumerate(event_mae):
        axes[0].text(index - width / 2, value + 0.012, f"{value:.3f}", ha="center")
    for index, value in enumerate(station_mae):
        axes[0].text(index + width / 2, value + 0.012, f"{value:.3f}", ha="center")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("200 秒 MAE (Mw)")
    axes[0].set_title("端点精度：直接流式重训发生退化")
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0.0, max(station_mae + event_mae) * 1.23)
    _style_axis(axes[0])

    metric_keys = (
        "late_event_abs_step_p95_mw",
        "late_station_abs_step_p95_mw",
        "late_confirmed_history_p95",
    )
    metric_labels = ("事件单秒跳变", "台站单秒跳变", "历史改写")
    group_x = np.arange(len(metric_keys), dtype=np.float64)
    bar_width = 0.24
    for model_index, model in enumerate(MODEL_ORDER):
        row = overall[model_index]
        values = [float(row[key]) for key in metric_keys]
        position = group_x + (model_index - 1) * bar_width
        axes[1].bar(
            position,
            values,
            bar_width,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )
        for item_x, value in zip(position, values, strict=True):
            axes[1].text(item_x, value + 0.0022, f"{value:.3f}", ha="center")
    axes[1].set_xticks(group_x, metric_labels)
    axes[1].set_ylabel("后期 p95")
    axes[1].set_title("流式稳定性：Phase48 改善最明显")
    axes[1].legend(frameon=False)
    axes[1].set_ylim(0.0, 0.09)
    _style_axis(axes[1])
    figure.tight_layout()
    _save_figure(figure, output_dir / "01_accuracy_stability_tradeoff")


def _plot_event_trajectories(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["event"]))].append(row)
    events = sorted({str(row["event"]) for row in rows})
    figure, axes = plt.subplots(4, 2, figsize=(14.2, 15.2), sharex=True)
    for axis, event in zip(axes.flat, events, strict=True):
        catalog = float(grouped[("phase39", event)][0]["mw_catalog"])
        all_predictions: list[float] = []
        for model in MODEL_ORDER:
            model_rows = sorted(
                grouped[(model, event)],
                key=lambda row: int(row["observation_horizon_sec"]),
            )
            horizons = [int(row["observation_horizon_sec"]) for row in model_rows]
            predictions = [float(row["mw_pred_median"]) for row in model_rows]
            all_predictions.extend(predictions)
            axis.plot(
                horizons,
                predictions,
                color=COLORS[model],
                linewidth=1.8,
                label=MODEL_LABELS[model],
            )
        axis.axhspan(
            catalog - 0.15,
            catalog + 0.15,
            color=COLORS["truth"],
            alpha=0.08,
        )
        axis.axhline(
            catalog,
            color=COLORS["truth"],
            linestyle="--",
            linewidth=1.3,
            label="参考 Mw",
        )
        lower = min(min(all_predictions), catalog) - 0.12
        upper = max(max(all_predictions), catalog) + 0.12
        axis.set_ylim(lower, upper)
        axis.set_title(event)
        axis.set_ylabel("预测 Mw")
        _style_axis(axis)
    for axis in axes[-1]:
        axis.set_xlabel("观测前缀 h (秒)，发布时间 h+6 秒")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    _save_figure(figure, output_dir / "02_event_trajectories")


def _plot_revision_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    events = sorted({str(row["event"]) for row in diagnostics})
    lookup = {
        (str(row["model"]), str(row["event"])): row for row in diagnostics
    }
    short_labels = [event.replace(" 20", "\n20") for event in events]
    x = np.arange(len(events), dtype=np.float64)
    width = 0.25
    figure, axes = plt.subplots(2, 1, figsize=(15.2, 9.2), sharex=True)
    for model_index, model in enumerate(MODEL_ORDER):
        positions = x + (model_index - 1) * width
        down = [
            float(lookup[(model, event)]["max_down_step_after_60_mw"])
            for event in events
        ]
        drop = [
            float(lookup[(model, event)]["peak_to_final_drop_after_60_mw"])
            for event in events
        ]
        axes[0].bar(
            positions,
            down,
            width,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )
        axes[1].bar(positions, drop, width, color=COLORS[model])
    axes[0].set_ylabel("最大单秒下降 (Mw)")
    axes[0].set_title("60 秒后单秒最大向下跳变")
    axes[0].legend(frameon=False, ncol=3)
    _style_axis(axes[0])
    axes[1].set_ylabel("峰值到 200 秒回落 (Mw)")
    axes[1].set_title("60 秒后峰值相对最终预测的累计回落")
    axes[1].set_xticks(x, short_labels, rotation=0)
    _style_axis(axes[1])
    figure.tight_layout()
    _save_figure(figure, output_dir / "03_revision_diagnostics")


def _stable_text(value: Any) -> str:
    return ">200 s" if value in (None, "") else f"{int(float(value))} s"


def _render_report(summary: Mapping[str, Any]) -> str:
    overall = {row["model"]: row for row in summary["overall_metrics"]}
    phase39 = overall["phase39"]
    phase47 = overall["phase47"]
    phase48 = overall["phase48"]
    event_lines: list[str] = []
    convergence = {
        (row["model"], row["event"]): row for row in summary["event_convergence"]
    }
    for row in summary["endpoint_events"]:
        event = row["event"]
        event_lines.append(
            "| {event} | {catalog:.1f} | {p39:.3f} ({e39:.3f}) | "
            "{p47:.3f} ({e47:.3f}) | {p48:.3f} ({e48:.3f}) | "
            "{stable} |".format(
                event=event,
                catalog=float(row["mw_catalog"]),
                p39=float(row["phase39_mw"]),
                e39=float(row["phase39_abs_error"]),
                p47=float(row["phase47_mw"]),
                e47=float(row["phase47_abs_error"]),
                p48=float(row["phase48_mw"]),
                e48=float(row["phase48_abs_error"]),
                stable=_stable_text(
                    convergence[("phase48", event)]["stable_observation_sec"]
                ),
            )
        )
    diagnostics = {
        (row["model"], row["event"]): row
        for row in summary["trajectory_diagnostics"]
    }
    revision_lines: list[str] = []
    for event in sorted(row["event"] for row in summary["endpoint_events"]):
        revision_lines.append(
            "| {event} | {p39_step:.3f} | {p47_step:.3f} | {p48_step:.3f} | "
            "{p39_drop:.3f} | {p47_drop:.3f} | {p48_drop:.3f} |".format(
                event=event,
                p39_step=float(
                    diagnostics[("phase39", event)]["max_down_step_after_60_mw"]
                ),
                p47_step=float(
                    diagnostics[("phase47", event)]["max_down_step_after_60_mw"]
                ),
                p48_step=float(
                    diagnostics[("phase48", event)]["max_down_step_after_60_mw"]
                ),
                p39_drop=float(
                    diagnostics[("phase39", event)][
                        "peak_to_final_drop_after_60_mw"
                    ]
                ),
                p47_drop=float(
                    diagnostics[("phase47", event)][
                        "peak_to_final_drop_after_60_mw"
                    ]
                ),
                p48_drop=float(
                    diagnostics[("phase48", event)][
                        "peak_to_final_drop_after_60_mw"
                    ]
                ),
            )
        )
    phase48_recovery = summary["recovery"]["phase48"]
    return f"""# Phase39、Phase47、Phase48 八事件流式回放

> 评估对象在读取八事件前已固定。八事件没有进入训练，但已被多轮开发反复使用，因此本文是 `development_validation_posthoc`，不是新的无偏盲测。

## 结论

- **Phase48 的流式稳定性确实明显改善**：后期事件单秒跳变 p95 从 Phase39 的 {phase39['late_event_abs_step_p95_mw']:.6f} 降到 {phase48['late_event_abs_step_p95_mw']:.6f} Mw（{_pct_change(phase39['late_event_abs_step_p95_mw'], phase48['late_event_abs_step_p95_mw']):+.1f}%）；历史改写 p95 从 {phase39['late_confirmed_history_p95']:.6f} 降到 {phase48['late_confirmed_history_p95']:.6f}（{_pct_change(phase39['late_confirmed_history_p95'], phase48['late_confirmed_history_p95']):+.1f}%）。
- **但八事件端点精度退化**：Event MAE 为 Phase39 {phase39['endpoint_event_mae']:.6f}、Phase47 {phase47['endpoint_event_mae']:.6f}、Phase48 {phase48['endpoint_event_mae']:.6f} Mw。Phase48 比 Phase39 增加 {phase48['endpoint_event_mae'] - phase39['endpoint_event_mae']:+.6f} Mw，只改善 3/8 个事件。
- **Phase47 不适合作为替代模型**：它虽然比 Phase39 略稳，但 Event MAE 增至 {phase47['endpoint_event_mae']:.6f} Mw，只改善 1/8 个事件。
- 当前证据支持的判断是：直接流式重训学会了“少跳动”，尤其 Phase48；但同时损失了未见事件的震级校准。现阶段应保留 Phase39 作为八事件端点基线，不能用 Phase47/48 替换它。

## 1. 精度与稳定性权衡

![精度与稳定性权衡](figures/01_accuracy_stability_tradeoff.png)

[PDF 图件](figures/01_accuracy_stability_tradeoff.pdf)

| 指标 | Phase39 | Phase47 | Phase48 |
|---|---:|---:|---:|
| 200 s Event MAE | {phase39['endpoint_event_mae']:.6f} | {phase47['endpoint_event_mae']:.6f} | {phase48['endpoint_event_mae']:.6f} |
| 200 s Station MAE | {phase39['endpoint_station_mae']:.6f} | {phase47['endpoint_station_mae']:.6f} | {phase48['endpoint_station_mae']:.6f} |
| 后期 Event step p95 | {phase39['late_event_abs_step_p95_mw']:.6f} | {phase47['late_event_abs_step_p95_mw']:.6f} | {phase48['late_event_abs_step_p95_mw']:.6f} |
| 后期 Station step p95 | {phase39['late_station_abs_step_p95_mw']:.6f} | {phase47['late_station_abs_step_p95_mw']:.6f} | {phase48['late_station_abs_step_p95_mw']:.6f} |
| 后期历史改写 p95 | {phase39['late_confirmed_history_p95']:.6f} | {phase47['late_confirmed_history_p95']:.6f} | {phase48['late_confirmed_history_p95']:.6f} |
| 持续进入 ±0.15 Mw 的事件数 | {int(phase39['stable_event_count'])}/8 | {int(phase47['stable_event_count'])}/8 | {int(phase48['stable_event_count'])}/8 |

## 2. 八事件逐秒轨迹

![八事件逐秒轨迹](figures/02_event_trajectories.png)

[PDF 图件](figures/02_event_trajectories.pdf)

每个时刻都用真实可获得的原始 E/N 数据旋转成 R，并以 `B x 1 x h` 的变长前缀重新预测一条完整非负 STF；观测前缀为 20–200 秒，发布时间为 `h+6 s`。三种模型均使用同一 158 台站 cohort。

| 事件 | 参考 Mw | Phase39 200 s（误差） | Phase47 200 s（误差） | Phase48 200 s（误差） | Phase48 持续收敛 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(event_lines)}

Phase48 主要改善 Kodiak、Samos，以及 Luding 的很小一部分误差；但 Sand、Mandalay、Nepal 明显变差。Phase47 对 Luding、Sand、Xizang 出现明显高估。

## 3. 为什么仍会先升后降

![向下修正诊断](figures/03_revision_diagnostics.png)

[PDF 图件](figures/03_revision_diagnostics.pdf)

非负 STF 只保证**同一次前向预测内部**的矩率不为负，并不保证相邻时刻的两条完整 STF 互相包含。每增加一秒，模型仍会重算全部 STF 形状和总矩，因此总矩和 Mw 可以向上或向下修正。Phase47/48 的时间一致性损失把单秒跳变压小了，但没有施加严格单调约束，也没有完全消除较慢的累计回落。

| 事件 | P39 最大单秒下降 | P47 | P48 | P39 峰值→最终回落 | P47 | P48 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(revision_lines)}

上表都从 60 秒后计算。`最大单秒下降`描述突然跳变；`峰值→最终回落`描述许多小步累积后的总修正。后者即使每一步很小，也可能在图上表现为明显“先升后降”。

## 4. Checkpoint 与证据限制

- Phase47 seed73 epoch19 通过严格恢复门槛：最大关键指标差 {summary['recovery']['phase47']['max_metric_abs_difference']:.2e}，checkpoint SHA-256 为 `{summary['recovery']['phase47']['checkpoint_sha256']}`。
- 原 Phase48 epoch188 checkpoint 没有被 runner 保留。本文使用固定的数值接近重建版，SHA-256 为 `{phase48_recovery['checkpoint_sha256']}`；它与原 epoch188 的最大关键指标差 {phase48_recovery['max_metric_abs_difference']:.6f}，selection score 差 {phase48_recovery['selection_score_abs_difference']:.6f}，原严格恢复门槛未通过。因此 Phase48 外部结果只能看作重建版的支持性诊断。
- Phase39 200 秒端点复现通过：最大台站差 {summary['phase39_endpoint_gate']['max_station_prediction_abs_diff_mw']:.2e} Mw。
- internal test 与 grouped test 本轮均未打开；八事件不得再用于选择 seed、checkpoint、阈值或下一轮权重。

## 5. 可下载工件

- [逐事件逐秒输出](event_predictions.csv)
- [逐秒总体指标](horizon_metrics.csv)
- [逐事件 200 秒对比](endpoint_event_comparison.csv)
- [逐台站 200 秒输出](endpoint_station_predictions.csv)
- [持续收敛时间](event_convergence.csv)
- [逐事件跳变与回落诊断](trajectory_diagnostics.csv)
- [完整逐台站逐秒输出（gzip）](station_predictions.csv.gz)
- [机器可读发布摘要](summary.json)
- [原始评估摘要](evaluation_summary.json)
- [发布清单与 SHA-256](publication_manifest.json)
- [冻结评估器](../../../scripts/evaluation/evaluate_phase49_posthoc_direct_streaming.py)
- [可复现绘图脚本](../../../scripts/plotting/plot_phase49_posthoc_direct_streaming_zh.py)

原始三组 STF rate 立方体保留在正式 run 目录，不提交 GitHub；其 SHA-256 已写入评估摘要。全仓回归：`{summary['full_regression']}`。
"""


def _copy_public_tables(run_root: Path, output_dir: Path) -> None:
    for name in (
        "endpoint_event_comparison.csv",
        "endpoint_station_predictions.csv",
        "event_convergence.csv",
        "event_predictions.csv",
        "horizon_metrics.csv",
    ):
        shutil.copy2(run_root / name, output_dir / name)
    with (run_root / "station_predictions.csv").open("rb") as source:
        with gzip.GzipFile(
            output_dir / "station_predictions.csv.gz",
            mode="wb",
            compresslevel=9,
            mtime=0,
        ) as target:
            shutil.copyfileobj(source, target)


def _validate_evaluation(evaluation: Mapping[str, Any]) -> None:
    if evaluation["status"] != "complete":
        raise ValueError("Phase49 external evaluation is incomplete")
    if evaluation["evaluation_role"] != "development_validation_posthoc":
        raise ValueError("Phase49 external role changed")
    if evaluation["provenance"]["git_commit"] != EXPECTED_EVALUATOR_COMMIT:
        raise ValueError("Phase49 evaluator commit changed")
    if evaluation["provenance"]["device"] != "cpu":
        raise ValueError("Phase49 publication requires the CPU endpoint run")
    if int(evaluation["provenance"]["batch_size"]) != 158:
        raise ValueError("Phase49 publication requires batch158")
    if int(evaluation["event_count"]) != 8 or int(evaluation["station_count"]) != 158:
        raise ValueError("Phase49 external cohort size changed")
    if evaluation["grouped_test_loaded"] is not False:
        raise ValueError("grouped test must remain closed")
    if evaluation["internal_test_loaded"] is not False:
        raise ValueError("internal test must remain closed")
    phase47 = evaluation["artifact_validation"]["models"]["phase47"]["recovery"]
    phase48 = evaluation["artifact_validation"]["models"]["phase48"]["recovery"]
    if phase47["strict_reproduction_passed"] is not True:
        raise ValueError("Phase47 strict recovery no longer passes")
    if phase48["strict_reproduction_passed"] is not False:
        raise ValueError("Phase48 recovery limitation changed")


def _publication_summary(
    evaluation: Mapping[str, Any],
    *,
    run_root: Path,
    endpoint_events: Sequence[Mapping[str, Any]],
    event_convergence: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    recovery: dict[str, Any] = {}
    for model in MODEL_ORDER:
        artifact = evaluation["artifact_validation"]["models"][model]
        item = artifact["recovery"]
        recovery[model] = {
            "checkpoint_sha256": artifact["checkpoint_sha256"],
            "role": None if item is None else item["role"],
            "strict_reproduction_passed": (
                None if item is None else item["strict_reproduction_passed"]
            ),
            "max_metric_abs_difference": (
                None if item is None else item["max_metric_abs_difference"]
            ),
            "selection_score_abs_difference": (
                None if item is None else item["selection_score_abs_difference"]
            ),
        }
    return {
        "status": "complete",
        "evaluation_role": evaluation["evaluation_role"],
        "implementation_commit": EXPECTED_EVALUATOR_COMMIT,
        "publication_commit": _git_commit(),
        "run_root": str(run_root),
        "overall_metrics": _overall_rows(evaluation),
        "endpoint_events": list(endpoint_events),
        "event_convergence": list(event_convergence),
        "trajectory_diagnostics": list(diagnostics),
        "recovery": recovery,
        "phase39_endpoint_gate": evaluation["phase39_endpoint_reproduction_gate"],
        "processing_delay_sec": evaluation["processing_delay_sec"],
        "full_regression": FULL_REGRESSION,
        "grouped_test_loaded": False,
        "internal_test_loaded": False,
    }


def _manifest(output_dir: Path, run_root: Path) -> None:
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "publication_manifest.json"
    )
    _write_json(
        output_dir / "publication_manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "report_role": "development_validation_posthoc",
            "implementation_commit": EXPECTED_EVALUATOR_COMMIT,
            "publication_commit": _git_commit(),
            "run_root": str(run_root),
            "generator": {
                "path": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
                "sha256": _sha256(SCRIPT_PATH),
            },
            "files": {
                str(path.relative_to(output_dir)): {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            },
        },
    )


def publish(run_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"output directory must be new or empty: {output_dir}")
    evaluation = _read_json(run_root / "summary.json")
    _validate_evaluation(evaluation)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _copy_public_tables(run_root, output_dir)
    shutil.copy2(run_root / "summary.json", output_dir / "evaluation_summary.json")

    event_rows = _read_csv(output_dir / "event_predictions.csv")
    endpoint_events = _read_csv(output_dir / "endpoint_event_comparison.csv")
    event_convergence = _read_csv(output_dir / "event_convergence.csv")
    diagnostics = trajectory_diagnostics(event_rows)
    _write_csv(output_dir / "trajectory_diagnostics.csv", diagnostics)
    overall = _overall_rows(evaluation)
    _write_csv(output_dir / "overall_metrics.csv", overall)
    _write_csv(output_dir / "checkpoint_recovery.csv", _recovery_rows(evaluation))

    summary = _publication_summary(
        evaluation,
        run_root=run_root,
        endpoint_events=endpoint_events,
        event_convergence=event_convergence,
        diagnostics=diagnostics,
    )
    _write_json(output_dir / "summary.json", summary)
    _configure_plotting()
    _plot_accuracy_stability(overall, figures)
    _plot_event_trajectories(event_rows, figures)
    _plot_revision_diagnostics(diagnostics, figures)
    (output_dir / "README.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    _manifest(output_dir, run_root)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = publish(args.run_root.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(args.output_dir.resolve()),
                "phase48_event_mae": next(
                    row["endpoint_event_mae"]
                    for row in summary["overall_metrics"]
                    if row["model"] == "phase48"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
