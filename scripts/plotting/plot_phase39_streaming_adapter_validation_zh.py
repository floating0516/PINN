#!/usr/bin/env python3
"""Publish the Phase39 streaming-adapter validation campaign in Chinese."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")

from matplotlib import font_manager
import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
PROJECT_HOME = REPO_ROOT.parent.parent
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "docs" / "reports" / "phase39-streaming-adapter-validation-zh"
)
DEFAULT_RUNS = {
    "Phase43": (
        PROJECT_HOME
        / "runs"
        / "phase43-streaming-adapter-20260728T121138Z-7bc63eb"
    ),
    "Phase44": (
        PROJECT_HOME
        / "runs"
        / "phase44-streaming-adapter-20260728T122020Z-7bc63eb"
    ),
    "Phase45": (
        PROJECT_HOME
        / "runs"
        / "phase45-streaming-adapter-20260728T122754Z-7bc63eb"
    ),
}
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
COLORS = {
    "Phase39": "#5B6573",
    "Phase43": "#0072B2",
    "Phase44": "#009E73",
    "Phase45": "#D55E00",
    "target": "#B42318",
    "grid": "#D8DEE6",
    "ink": "#20262E",
    "pass_fill": "#E8F5EE",
}
PHASE_ORDER = ("Phase39", "Phase43", "Phase44", "Phase45")
STABILITY_METRICS = (
    (
        "late_event_abs_step_p95_mw",
        "事件逐秒跳变 p95",
        "late_event_abs_step_p95_mw_max",
        0.8,
    ),
    (
        "late_station_abs_step_p95_mw",
        "台站逐秒跳变 p95",
        "late_station_abs_step_p95_mw_max",
        1.0,
    ),
    (
        "late_confirmed_cumulative_log10_l1_p95",
        "已确认历史变化 p95",
        "late_confirmed_cumulative_log10_l1_p95_max",
        0.8,
    ),
)


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


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _same_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)


def _load_campaign(run_dirs: Mapping[str, Path]) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    for phase, run_dir in run_dirs.items():
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing {phase} summary: {summary_path}")
        summary = _read_json(summary_path)
        if summary.get("status") != "validation_gate_failed":
            raise ValueError(f"{phase} status changed: {summary.get('status')}")
        if summary.get("passed") is not False:
            raise ValueError(f"{phase} unexpectedly passed")
        if len(summary.get("seed_summaries", ())) != 3:
            raise ValueError(f"{phase} must contain exactly three formal seeds")
        summaries[phase] = summary

    reference_gates = summaries["Phase45"]["protocol"]["validation_gates"]
    for phase, summary in summaries.items():
        gates = summary["protocol"]["validation_gates"]
        for name, expected in reference_gates.items():
            if not _same_float(float(gates[name]), float(expected)):
                raise ValueError(f"{phase} validation gate changed: {name}")
        for seed_summary in summary["seed_summaries"]:
            if seed_summary.get("phase39_weights_trained") is not False:
                raise ValueError(f"{phase} changed frozen Phase39 weights")
            for flag in (
                "internal_test_iterated",
                "external_data_loaded",
                "grouped_test_loaded",
            ):
                if seed_summary.get(flag) is not False:
                    raise ValueError(f"{phase} accessed hidden data: {flag}")

    gates = {name: float(value) for name, value in reference_gates.items()}
    baseline = {
        "endpoint_event_mae": gates["endpoint_event_mae_max"] - 0.005,
        "endpoint_station_mae": gates["endpoint_station_mae_max"] - 0.005,
        "late_event_abs_step_p95_mw": (
            gates["late_event_abs_step_p95_mw_max"] / 0.8
        ),
        "late_station_abs_step_p95_mw": gates[
            "late_station_abs_step_p95_mw_max"
        ],
        "late_confirmed_cumulative_log10_l1_p95": (
            gates["late_confirmed_cumulative_log10_l1_p95_max"] / 0.8
        ),
    }

    closest: dict[str, dict[str, Any]] = {}
    seed_rows: list[dict[str, Any]] = []
    for phase, summary in summaries.items():
        eligible = [
            item
            for item in summary["seed_summaries"]
            if item.get("selected_gate") is not None
        ]
        closest[phase] = min(
            eligible,
            key=lambda item: float(item["selected_gate"]["selection_score"]),
        )
        for item in summary["seed_summaries"]:
            metrics = item["selected_metrics"]
            gate = item["selected_gate"]
            seed_rows.append(
                {
                    "phase": phase,
                    "stability_weight": float(
                        summary["protocol"]["loss_weights"]["late_step"]
                    ),
                    "seed": int(item["seed"]),
                    "epoch": int(item["selected_epoch"]),
                    "selection_score": float(gate["selection_score"]),
                    "endpoint_event_mae": float(metrics["endpoint_event_mae"]),
                    "endpoint_station_mae": float(metrics["endpoint_station_mae"]),
                    "late_event_abs_step_p95_mw": float(
                        metrics["late_event_abs_step_p95_mw"]
                    ),
                    "late_station_abs_step_p95_mw": float(
                        metrics["late_station_abs_step_p95_mw"]
                    ),
                    "late_confirmed_cumulative_log10_l1_p95": float(
                        metrics["late_confirmed_cumulative_log10_l1_p95"]
                    ),
                    "endpoint_preserved": bool(gate["endpoint_preserved"]),
                    "stability_passed": bool(gate["stability_passed"]),
                    "passed": bool(gate["passed"]),
                }
            )

    phase_rows: list[dict[str, Any]] = [
        {
            "phase": "Phase39",
            "role": "frozen_baseline",
            "stability_weight": 0.0,
            "seed": 42,
            "epoch": "",
            "selection_score": 1.25,
            **baseline,
            "endpoint_event_delta": 0.0,
            "endpoint_station_delta": 0.0,
            "event_step_improvement_pct": 0.0,
            "station_step_improvement_pct": 0.0,
            "confirmed_history_improvement_pct": 0.0,
            "formal_gate_passed": False,
        }
    ]
    for phase in ("Phase43", "Phase44", "Phase45"):
        item = closest[phase]
        metrics = item["selected_metrics"]
        row = {
            "phase": phase,
            "role": "closest_validation_checkpoint_only",
            "stability_weight": float(
                summaries[phase]["protocol"]["loss_weights"]["late_step"]
            ),
            "seed": int(item["seed"]),
            "epoch": int(item["selected_epoch"]),
            "selection_score": float(item["selected_gate"]["selection_score"]),
            "endpoint_event_mae": float(metrics["endpoint_event_mae"]),
            "endpoint_station_mae": float(metrics["endpoint_station_mae"]),
            "late_event_abs_step_p95_mw": float(
                metrics["late_event_abs_step_p95_mw"]
            ),
            "late_station_abs_step_p95_mw": float(
                metrics["late_station_abs_step_p95_mw"]
            ),
            "late_confirmed_cumulative_log10_l1_p95": float(
                metrics["late_confirmed_cumulative_log10_l1_p95"]
            ),
        }
        row.update(
            {
                "endpoint_event_delta": (
                    row["endpoint_event_mae"] - baseline["endpoint_event_mae"]
                ),
                "endpoint_station_delta": (
                    row["endpoint_station_mae"]
                    - baseline["endpoint_station_mae"]
                ),
                "event_step_improvement_pct": 100.0
                * (
                    1.0
                    - row["late_event_abs_step_p95_mw"]
                    / baseline["late_event_abs_step_p95_mw"]
                ),
                "station_step_improvement_pct": 100.0
                * (
                    1.0
                    - row["late_station_abs_step_p95_mw"]
                    / baseline["late_station_abs_step_p95_mw"]
                ),
                "confirmed_history_improvement_pct": 100.0
                * (
                    1.0
                    - row["late_confirmed_cumulative_log10_l1_p95"]
                    / baseline["late_confirmed_cumulative_log10_l1_p95"]
                ),
                "formal_gate_passed": bool(item["selected_gate"]["passed"]),
            }
        )
        phase_rows.append(row)

    return {
        "summaries": summaries,
        "gates": gates,
        "baseline": baseline,
        "closest": closest,
        "phase_rows": phase_rows,
        "seed_rows": seed_rows,
    }


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_stability_progression(campaign: Mapping[str, Any], output: Path) -> None:
    rows = {row["phase"]: row for row in campaign["phase_rows"]}
    baseline = campaign["baseline"]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    labels = ("Phase39", "Phase43\n0.1", "Phase44\n1.0", "Phase45\n2.0")
    for axis, (metric, title, _, target_ratio) in zip(axes, STABILITY_METRICS):
        values = [100.0 * float(rows[phase][metric]) / baseline[metric] for phase in PHASE_ORDER]
        bars = axis.bar(
            range(len(values)),
            values,
            color=[COLORS[phase] for phase in PHASE_ORDER],
            width=0.66,
        )
        axis.axhline(
            100.0 * target_ratio,
            color=COLORS["target"],
            linestyle="--",
            linewidth=1.4,
            label="门槛",
        )
        axis.set_title(title)
        axis.set_xticks(range(len(labels)), labels)
        axis.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 2.0,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axes[0].set_ylabel("相对 Phase39 原始波动（%）")
    axes[0].set_ylim(0.0, 132.0)
    axes[-1].legend(loc="upper right", frameon=False)
    figure.suptitle("稳定性权重增强后，Phase45 已接近但未通过全部门槛", y=1.02)
    figure.tight_layout()
    _save_figure(figure, output / "01_stability_progression")


def _plot_endpoint_tradeoff(campaign: Mapping[str, Any], output: Path) -> None:
    rows = {row["phase"]: row for row in campaign["phase_rows"]}
    gates = campaign["gates"]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    markers = {"Phase39": "o", "Phase43": "s", "Phase44": "^", "Phase45": "D"}
    panels = (
        ("endpoint_event_mae", "endpoint_event_mae_max", "事件终点 MAE"),
        ("endpoint_station_mae", "endpoint_station_mae_max", "台站终点 MAE"),
    )
    y_positions = tuple(reversed(range(len(PHASE_ORDER))))
    for axis, (metric, gate_name, title) in zip(axes, panels):
        gate = float(gates[gate_name])
        axis.axvspan(94.0, 100.0, color=COLORS["pass_fill"], zorder=0)
        axis.axvline(
            100.0,
            color=COLORS["target"],
            linestyle="--",
            linewidth=1.3,
        )
        for phase, y in zip(PHASE_ORDER, y_positions):
            value = float(rows[phase][metric])
            ratio = 100.0 * value / gate
            axis.scatter(
                [ratio],
                [y],
                s=76,
                marker=markers[phase],
                color=COLORS[phase],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            axis.text(
                ratio + 0.12,
                y,
                f"{value:.6f}",
                va="center",
                fontsize=9,
            )
        axis.set_xlim(94.0, 101.5)
        axis.set_xlabel("占终点 MAE 上限（%）")
        axis.set_title(title)
        axis.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
        axis.set_axisbelow(True)
    axes[0].set_yticks(y_positions, PHASE_ORDER)
    axes[1].text(
        99.85,
        y_positions[0] + 0.42,
        "100% 为门槛",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    figure.suptitle("三轮适配器均保留了 Phase39 的 200 秒终点精度", y=1.02)
    figure.tight_layout()
    _save_figure(figure, output / "02_endpoint_tradeoff")


def _plot_training_curves(
    campaign: Mapping[str, Any],
    run_dirs: Mapping[str, Path],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9.4, 5.0))
    for phase in ("Phase43", "Phase44", "Phase45"):
        closest = campaign["closest"][phase]
        seed = int(closest["seed"])
        rows = _read_csv(run_dirs[phase] / f"seed_{seed}" / "epoch_metrics.csv")
        epochs = [int(row["epoch"]) for row in rows]
        scores = [float(row["selection_score"]) for row in rows]
        axis.plot(
            epochs,
            scores,
            color=COLORS[phase],
            linewidth=2.0,
            label=f"{phase}（seed{seed}）",
        )
        best_index = min(range(len(scores)), key=scores.__getitem__)
        axis.scatter(
            [epochs[best_index]],
            [scores[best_index]],
            color=COLORS[phase],
            s=52,
            zorder=3,
        )
        axis.annotate(
            f"{scores[best_index]:.3f}",
            (epochs[best_index], scores[best_index]),
            xytext=(4, 7),
            textcoords="offset points",
            fontsize=9,
        )
    axis.axhline(
        1.0,
        color=COLORS["target"],
        linestyle="--",
        linewidth=1.4,
        label="全部稳定性门槛",
    )
    axis.set_xlim(1, 30)
    axis.set_ylim(0.98, 1.34)
    axis.set_xlabel("训练 epoch")
    axis.set_ylabel("最差归一化稳定性比值（≤1 才通过）")
    axis.set_title("增加稳定性损失权重使验证分数接近门槛，但 Phase45 仍未越过")
    axis.grid(color=COLORS["grid"], linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    _save_figure(figure, output / "03_training_curves")


def _render_readme(campaign: Mapping[str, Any]) -> str:
    rows = {row["phase"]: row for row in campaign["phase_rows"]}
    gates = campaign["gates"]
    phase45 = rows["Phase45"]
    return f"""# Phase39 流式 STF 适配器验证报告

> 状态：Phase43、Phase44、Phase45 均已完成正式 validation-only 三 seed 训练；三轮均未通过完整稳定性 gate，因此没有正式选中 seed，也没有打开 internal test、外部 8 事件或 grouped test。

## 结论

Phase45 已经明显减少逐秒波动，并保留 200 秒终点精度，但按预先固定的门槛仍应判为 **失败**。最接近门槛的是仅供审计的 seed42 epoch27：

- Event 终点 MAE：**{phase45['endpoint_event_mae']:.6f} Mw**，相对冻结 Phase39 增加 {phase45['endpoint_event_delta']:.6f} Mw。
- Station 终点 MAE：**{phase45['endpoint_station_mae']:.6f} Mw**，相对冻结 Phase39 增加 {phase45['endpoint_station_delta']:.6f} Mw。
- 后期事件逐秒跳变 p95：**{phase45['late_event_abs_step_p95_mw']:.6f} Mw**，改善 {phase45['event_step_improvement_pct']:.1f}%。
- 后期台站逐秒跳变 p95：**{phase45['late_station_abs_step_p95_mw']:.6f} Mw**，改善 {phase45['station_step_improvement_pct']:.1f}%。
- 已确认历史变化 p95：**{phase45['late_confirmed_cumulative_log10_l1_p95']:.6f} log10**，改善 {phase45['confirmed_history_improvement_pct']:.1f}%；门槛要求至少 20%，实际还差约 {20.0 - phase45['confirmed_history_improvement_pct']:.1f} 个百分点。

因此，Phase39 仍是保留模型；Phase45 checkpoint 只能作为验证诊断，不应追认为通过模型。

![稳定性进展](figures/01_stability_progression.png)

[PDF 图件](figures/01_stability_progression.pdf)

## 三轮训练比较

| 方法 | 稳定性权重 | 审计用最近 seed/epoch | Event MAE | Station MAE | 事件跳变 p95 | 台站跳变 p95 | 历史变化 p95 | 最差比值 | 正式通过 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Phase39 冻结基线 | 0 | seed42 | {rows['Phase39']['endpoint_event_mae']:.6f} | {rows['Phase39']['endpoint_station_mae']:.6f} | {rows['Phase39']['late_event_abs_step_p95_mw']:.6f} | {rows['Phase39']['late_station_abs_step_p95_mw']:.6f} | {rows['Phase39']['late_confirmed_cumulative_log10_l1_p95']:.6f} | 1.250 | 否 |
| Phase43 | 0.1 | seed{rows['Phase43']['seed']} / e{rows['Phase43']['epoch']} | {rows['Phase43']['endpoint_event_mae']:.6f} | {rows['Phase43']['endpoint_station_mae']:.6f} | {rows['Phase43']['late_event_abs_step_p95_mw']:.6f} | {rows['Phase43']['late_station_abs_step_p95_mw']:.6f} | {rows['Phase43']['late_confirmed_cumulative_log10_l1_p95']:.6f} | {rows['Phase43']['selection_score']:.3f} | 否 |
| Phase44 | 1.0 | seed{rows['Phase44']['seed']} / e{rows['Phase44']['epoch']} | {rows['Phase44']['endpoint_event_mae']:.6f} | {rows['Phase44']['endpoint_station_mae']:.6f} | {rows['Phase44']['late_event_abs_step_p95_mw']:.6f} | {rows['Phase44']['late_station_abs_step_p95_mw']:.6f} | {rows['Phase44']['late_confirmed_cumulative_log10_l1_p95']:.6f} | {rows['Phase44']['selection_score']:.3f} | 否 |
| Phase45 | 2.0 | seed{rows['Phase45']['seed']} / e{rows['Phase45']['epoch']} | {rows['Phase45']['endpoint_event_mae']:.6f} | {rows['Phase45']['endpoint_station_mae']:.6f} | {rows['Phase45']['late_event_abs_step_p95_mw']:.6f} | {rows['Phase45']['late_station_abs_step_p95_mw']:.6f} | {rows['Phase45']['late_confirmed_cumulative_log10_l1_p95']:.6f} | {rows['Phase45']['selection_score']:.3f} | 否 |

固定门槛为：Event 终点 MAE ≤ {gates['endpoint_event_mae_max']:.6f}、Station 终点 MAE ≤ {gates['endpoint_station_mae_max']:.6f}、事件跳变 p95 ≤ {gates['late_event_abs_step_p95_mw_max']:.6f}、台站跳变 p95 ≤ {gates['late_station_abs_step_p95_mw_max']:.6f}、历史变化 p95 ≤ {gates['late_confirmed_cumulative_log10_l1_p95_max']:.6f}。

![终点精度](figures/02_endpoint_tradeoff.png)

[PDF 图件](figures/02_endpoint_tradeoff.pdf)

![训练曲线](figures/03_training_curves.png)

[PDF 图件](figures/03_training_curves.pdf)

## 这次实际训练了什么

- Phase39 Glehman scalar + global invariant、seed42 主干完全冻结，没有重新训练约 101 万参数网络。
- 新增一个 **489 参数**的因果流式 STF retention adapter，只读取截至当前发布时间可得到的原始 E/N 波形，经实时 R 分量重处理后，处理 `20–200 s` 的逐秒前缀；发布时间为观测时长 `h+6 s`。
- 每一秒，冻结 Phase39 先重新预测一条完整非负 STF；adapter 再按 STF 时间格点融合上一状态与当前预测。三轮唯一变化是两项稳定性损失共同权重 `0.1 → 1.0 → 2.0`。
- 固定 seeds 为 17/42/73，每个 seed 30 epochs，只使用 train/validation。没有 ensemble。

## 为什么震级仍可能小幅下降

adapter 对每个 STF 时间格点执行 `state_t = gate * state_(t-1) + (1-gate) * raw_t`。上一状态和当前预测都非负，所以输出 STF 始终非负；但如果当前完整 STF 在某些格点低于上一秒，融合后的累计矩仍可能下降。

这不是“出现负矩率”，而是模型利用新增波形修正上一秒对总矩的估计。强制震级只能上升会系统性保留早期高估，因此本轮目标是**限制后期大幅回撤**，而不是数学上禁止所有下降。Phase45 已把后期事件跳变 p95 压到约 0.020 Mw，但极端台站和已确认历史的一致性仍未达到预定标准。

## 研究边界

- 训练使用的 validation 仍是 `within_event_station`：同一事件的不同台站分散在 train/validation/test，不能证明未见事件泛化。
- internal test、反复使用的外部 8 事件和 grouped test 均未打开。
- 按预先声明的停止规则，Phase45 失败后不再根据同一 validation 继续调高稳定性权重。
- 若继续推进实时模型，应另开一个冻结协议的新阶段，改变状态更新或训练目标；不能把 Phase45 的近门槛结果当作再次调参的依据。

## 工件

- [阶段级指标](validation_metrics.csv)
- [九个 seed 指标](seed_metrics.csv)
- [机器可读摘要](summary.json)
- [发布清单与 SHA-256](publication_manifest.json)
- [训练驱动](../../../scripts/experiments/run_phase43_streaming_adapter.py)
- [流式适配器](../../../src/models/streaming_stf_adapter.py)
- [可复现报告生成器](../../../scripts/plotting/plot_phase39_streaming_adapter_validation_zh.py)

全仓回归：`786 passed, 1 skipped`。报告角色：`within_event_validation_streaming_diagnostic`。
"""


def _publish(
    campaign: Mapping[str, Any],
    run_dirs: Mapping[str, Path],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _configure_plotting()
    _plot_stability_progression(campaign, figures_dir)
    _plot_endpoint_tradeoff(campaign, figures_dir)
    _plot_training_curves(campaign, run_dirs, figures_dir)

    phase_fieldnames = (
        "phase",
        "role",
        "stability_weight",
        "seed",
        "epoch",
        "selection_score",
        "endpoint_event_mae",
        "endpoint_station_mae",
        "late_event_abs_step_p95_mw",
        "late_station_abs_step_p95_mw",
        "late_confirmed_cumulative_log10_l1_p95",
        "endpoint_event_delta",
        "endpoint_station_delta",
        "event_step_improvement_pct",
        "station_step_improvement_pct",
        "confirmed_history_improvement_pct",
        "formal_gate_passed",
    )
    seed_fieldnames = (
        "phase",
        "stability_weight",
        "seed",
        "epoch",
        "selection_score",
        "endpoint_event_mae",
        "endpoint_station_mae",
        "late_event_abs_step_p95_mw",
        "late_station_abs_step_p95_mw",
        "late_confirmed_cumulative_log10_l1_p95",
        "endpoint_preserved",
        "stability_passed",
        "passed",
    )
    _write_csv(
        output_dir / "validation_metrics.csv",
        phase_fieldnames,
        campaign["phase_rows"],
    )
    _write_csv(output_dir / "seed_metrics.csv", seed_fieldnames, campaign["seed_rows"])

    phase45 = next(
        row for row in campaign["phase_rows"] if row["phase"] == "Phase45"
    )
    source_artifacts = {
        phase: {
            "run_root": str(run_dir),
            "summary_sha256": _sha256(run_dir / "summary.json"),
            "protocol_sha256": _sha256(run_dir / "protocol.json"),
            "provenance_sha256": _sha256(run_dir / "provenance.json"),
        }
        for phase, run_dir in run_dirs.items()
    }
    summary = {
        "status": "validation_gate_failed",
        "formal_candidate_selected": False,
        "retained_model": "Phase39 Glehman scalar + global invariant, seed42",
        "closest_validation_checkpoint": {
            "phase": "Phase45",
            "seed": int(phase45["seed"]),
            "epoch": int(phase45["epoch"]),
            "selection_score": float(phase45["selection_score"]),
            "audit_only": True,
        },
        "phase45_metrics": phase45,
        "validation_gates": campaign["gates"],
        "adapter_parameter_count": 489,
        "phase39_weights_trained": False,
        "internal_test_iterated": False,
        "external_data_loaded": False,
        "grouped_test_loaded": False,
        "cohort_role": "within_event_validation_streaming_diagnostic",
        "full_regression": "786 passed, 1 skipped in 45.87s",
        "source_artifacts": source_artifacts,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "README.md").write_text(
        _render_readme(campaign),
        encoding="utf-8",
    )

    publication_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "publication_manifest.json"
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_role": "within_event_validation_streaming_diagnostic",
        "source_phase39_commit": "7bc63eb25b90048d94d6478503e85b77790b9e80",
        "formal_candidate_selected": False,
        "files": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in publication_files
        ],
        "generator": {
            "path": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(SCRIPT_PATH),
        },
        "source_artifacts": source_artifacts,
    }
    _write_json(output_dir / "publication_manifest.json", manifest)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Chinese Phase39 streaming-adapter validation report."
    )
    parser.add_argument("--phase43-run", type=Path, default=DEFAULT_RUNS["Phase43"])
    parser.add_argument("--phase44-run", type=Path, default=DEFAULT_RUNS["Phase44"])
    parser.add_argument("--phase45-run", type=Path, default=DEFAULT_RUNS["Phase45"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_dirs = {
        "Phase43": args.phase43_run.resolve(),
        "Phase44": args.phase44_run.resolve(),
        "Phase45": args.phase45_run.resolve(),
    }
    output_dir = args.output_dir.resolve()
    campaign = _load_campaign(run_dirs)
    _publish(campaign, run_dirs, output_dir)
    print(
        json.dumps(
            {
                "status": "published",
                "output_dir": str(output_dir),
                "formal_candidate_selected": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
