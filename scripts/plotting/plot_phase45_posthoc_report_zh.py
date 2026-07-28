#!/usr/bin/env python3
"""Publish the Phase45 post-hoc train/test/external report in Chinese."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
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
    / "phase46-phase45-posthoc-20260728T131632Z-20d35a9"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "docs"
    / "reports"
    / "phase45-posthoc-train-test-external-zh"
)
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
TARGET_MAE = 0.15
COLORS = {
    "raw": "#5B6573",
    "adapted": "#D55E00",
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


def _stable_horizon(rows: Sequence[Mapping[str, str]], key: str) -> int | None:
    for index, row in enumerate(rows):
        if all(float(item[key]) <= TARGET_MAE for item in rows[index:]):
            return int(float(row["observation_horizon_sec"]))
    return None


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
        "external-cpu/horizon_metrics.csv": "external_horizon_metrics.csv",
        "external-cpu/event_predictions.csv": "external_event_predictions.csv",
        "external-cpu/station_predictions.csv": "external_station_predictions.csv",
        "external-cpu/endpoint_station_predictions.csv": (
            "external_endpoint_station_predictions.csv"
        ),
        "external-cpu/event_convergence.csv": "external_event_convergence.csv",
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
    selected_epoch = 27
    min_index = int(np.argmin(total))
    components = (
        ("train_endpoint_science_weighted_normalized", "终点科学损失", COLORS["blue"]),
        ("train_sequence_target_weighted_normalized", "序列目标", COLORS["green"]),
        ("train_endpoint_teacher_weighted_normalized", "终点 teacher", COLORS["purple"]),
        ("train_late_step_weighted_normalized", "后期逐秒", COLORS["yellow"]),
        ("train_confirmed_history_weighted_normalized", "确认历史", COLORS["adapted"]),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
    axes[0].plot(epochs, total, color=COLORS["blue"], linewidth=2.0)
    axes[0].scatter(
        [epochs[min_index]],
        [total[min_index]],
        color=COLORS["green"],
        s=58,
        zorder=3,
    )
    selected_value = float(total[np.where(epochs == selected_epoch)[0][0]])
    axes[0].scatter(
        [selected_epoch],
        [selected_value],
        color=COLORS["adapted"],
        marker="D",
        s=58,
        zorder=3,
    )
    axes[0].annotate(
        f"最低: e{epochs[min_index]} / {total[min_index]:.3f}",
        (epochs[min_index], total[min_index]),
        xytext=(7, -20),
        textcoords="offset points",
        fontsize=9,
    )
    axes[0].annotate(
        f"选中: e27 / {selected_value:.3f}",
        (selected_epoch, selected_value),
        xytext=(-80, 12),
        textcoords="offset points",
        fontsize=9,
    )
    axes[0].set_title("训练集总归一化目标")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("加权归一化 loss")
    _style_axis(axes[0])

    for key, label, color in components:
        values = [float(row[key]) for row in rows]
        axes[1].plot(epochs, values, label=label, color=color, linewidth=1.8)
    axes[1].axvline(
        selected_epoch,
        color=COLORS["red"],
        linestyle="--",
        linewidth=1.2,
        label="selected epoch27",
    )
    axes[1].set_title("训练目标的五项贡献")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("加权归一化贡献")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    _style_axis(axes[1])
    figure.suptitle("最低训练 loss 与最佳 validation 稳定性不在同一 epoch", y=1.02)
    figure.tight_layout()
    _save_figure(figure, output / "01_training_loss")


def _plot_internal_summary(
    raw: Mapping[str, Any],
    adapted: Mapping[str, Any],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    endpoint_labels = ("Event MAE", "Station MAE")
    raw_endpoint = [raw["endpoint_event_mae"], raw["endpoint_station_mae"]]
    adapted_endpoint = [
        adapted["endpoint_event_mae"],
        adapted["endpoint_station_mae"],
    ]
    x = np.arange(2)
    width = 0.34
    raw_bars = axes[0].bar(
        x - width / 2,
        raw_endpoint,
        width,
        label="Phase39 raw",
        color=COLORS["raw"],
    )
    adapted_bars = axes[0].bar(
        x + width / 2,
        adapted_endpoint,
        width,
        label="Phase45 adapter",
        color=COLORS["adapted"],
    )
    axes[0].set_xticks(x, endpoint_labels)
    axes[0].set_ylabel("MAE（Mw）")
    axes[0].set_title("internal test 200 秒端点")
    axes[0].legend(frameon=False)
    for bars in (raw_bars, adapted_bars):
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0].set_ylim(0.0, max(raw_endpoint + adapted_endpoint) * 1.18)
    _style_axis(axes[0])

    stability_specs = (
        ("late_event_abs_step_p95_mw", "事件跳变"),
        ("late_station_abs_step_p95_mw", "台站跳变"),
        ("late_confirmed_cumulative_log10_l1_p95", "历史改写"),
    )
    raw_values = [float(raw[key]) for key, _ in stability_specs]
    adapted_values = [float(adapted[key]) for key, _ in stability_specs]
    x = np.arange(len(stability_specs))
    raw_bars = axes[1].bar(
        x - width / 2,
        raw_values,
        width,
        color=COLORS["raw"],
        label="Phase39 raw",
    )
    adapted_bars = axes[1].bar(
        x + width / 2,
        adapted_values,
        width,
        color=COLORS["adapted"],
        label="Phase45 adapter",
    )
    axes[1].set_xticks(x, [label for _, label in stability_specs])
    axes[1].set_ylabel("后期 p95")
    axes[1].set_title("internal test 179–200 秒稳定性")
    for index, (before, after) in enumerate(zip(raw_values, adapted_values)):
        axes[1].text(
            index,
            max(before, after) + 0.002,
            f"{_pct_change(before, after):+.1f}%",
            ha="center",
            fontsize=9,
        )
    axes[1].legend(frameon=False)
    axes[1].set_ylim(0.0, max(raw_values + adapted_values) * 1.24)
    _style_axis(axes[1])
    figure.suptitle("internal test：事件层面略有改善，台站端点几乎不变", y=1.02)
    figure.tight_layout()
    _save_figure(figure, output / "02_internal_test")


def _plot_external_overall(rows: Sequence[Mapping[str, str]], output: Path) -> None:
    horizons = np.asarray([float(row["observation_horizon_sec"]) for row in rows])
    raw_event = np.asarray([float(row["raw_event_mae"]) for row in rows])
    adapted_event = np.asarray([float(row["adapted_event_mae"]) for row in rows])
    raw_station = np.asarray([float(row["raw_station_mae"]) for row in rows])
    adapted_station = np.asarray([float(row["adapted_station_mae"]) for row in rows])
    raw_stable = _stable_horizon(rows, "raw_event_mae")
    adapted_stable = _stable_horizon(rows, "adapted_event_mae")
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    axes[0].plot(horizons, raw_event, color=COLORS["raw"], linewidth=1.8, label="Phase39 raw")
    axes[0].plot(
        horizons,
        adapted_event,
        color=COLORS["adapted"],
        linewidth=2.0,
        label="Phase45 adapter",
    )
    axes[0].axhline(
        TARGET_MAE,
        color=COLORS["red"],
        linestyle="--",
        linewidth=1.2,
        label="0.15 Mw",
    )
    if raw_stable is not None:
        axes[0].axvline(raw_stable, color=COLORS["raw"], linestyle=":", linewidth=1.1)
    if adapted_stable is not None:
        axes[0].axvline(
            adapted_stable,
            color=COLORS["adapted"],
            linestyle=":",
            linewidth=1.1,
        )
    axes[0].set_title(
        "八事件等权 Event MAE\n"
        f"持续≤0.15：raw {raw_stable}s / adapter {adapted_stable}s"
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
        label="Phase39 raw",
    )
    axes[1].plot(
        horizons,
        adapted_station,
        color=COLORS["adapted"],
        linewidth=2.0,
        label="Phase45 adapter",
    )
    axes[1].set_title("八事件 Station MAE")
    axes[1].set_xlabel("观测时长（s）")
    axes[1].set_ylabel("Station MAE（Mw）")
    axes[1].legend(frameon=False)
    _style_axis(axes[1])
    figure.suptitle("八事件逐秒表现：事件聚合改善大于台站层面", y=1.02)
    figure.tight_layout()
    _save_figure(figure, output / "03_external_overall")


def _plot_external_events(rows: Sequence[Mapping[str, str]], output: Path) -> None:
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["event"]), []).append(row)
    figure, axes = plt.subplots(4, 2, figsize=(12.4, 12.0), sharex=True)
    for axis, event in zip(axes.flat, sorted(grouped)):
        sequence = sorted(
            grouped[event],
            key=lambda row: float(row["observation_horizon_sec"]),
        )
        horizons = [float(row["observation_horizon_sec"]) for row in sequence]
        raw = [float(row["raw_mw_pred_median"]) for row in sequence]
        adapted = [float(row["adapted_mw_pred_median"]) for row in sequence]
        catalog = float(sequence[-1]["mw_catalog"])
        axis.plot(horizons, raw, color=COLORS["raw"], linewidth=1.3, label="raw")
        axis.plot(
            horizons,
            adapted,
            color=COLORS["adapted"],
            linewidth=1.7,
            label="adapter",
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
            f"200s: {raw[-1]:.3f} → {adapted[-1]:.3f}",
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
    figure.suptitle("八个外部开发事件的逐秒输出（发布时刻 = 观测时长 + 6 s）", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    _save_figure(figure, output / "04_external_event_trajectories")


def _render_report(summary: Mapping[str, Any]) -> str:
    train = summary["training"]
    internal_raw = summary["internal"]["raw"]
    internal_adapted = summary["internal"]["adapted"]
    external_raw = summary["external"]["raw"]
    external_adapted = summary["external"]["adapted"]
    events = summary["external"]["events"]
    event_lines = []
    for row in events:
        raw_stable = row["raw_stable_observation_sec"]
        adapted_stable = row["adapted_stable_observation_sec"]
        event_lines.append(
            "| {event} | {catalog:.1f} | {raw:.3f} | {adapted:.3f} | "
            "{raw_error:.3f} | {adapted_error:.3f} | {raw_stable} | "
            "{adapted_stable} |".format(
                event=row["event"],
                catalog=float(row["mw_catalog"]),
                raw=float(row["raw_final_mw"]),
                adapted=float(row["adapted_final_mw"]),
                raw_error=float(row["raw_final_abs_error"]),
                adapted_error=float(row["adapted_final_abs_error"]),
                raw_stable=(">200 s" if raw_stable is None else f"{raw_stable} s"),
                adapted_stable=(
                    ">200 s" if adapted_stable is None else f"{adapted_stable} s"
                ),
            )
        )
    selected = train["selected_epoch_metrics"]
    return f"""# Phase45 训练、internal test 与八事件逐秒评估

> 固定对象：Phase45 489 参数 streaming STF adapter，seed42 epoch27。Phase39 主干不变；没有根据 test 或八事件结果换 seed、换 checkpoint 或调权重。

## 结论

- **训练集目标**：最低总归一化 loss 在 epoch4（{train['minimum_training_total']['value']:.6f}），而 validation 稳定性最佳 checkpoint 是 epoch27（训练 loss {selected['train_total_normalized_loss']:.6f}）。两者不一致，说明稳定性目标与端点/teacher 约束存在权衡。
- **internal test**：Event MAE 从 {internal_raw['endpoint_event_mae']:.6f} 改善到 {internal_adapted['endpoint_event_mae']:.6f} Mw；Station MAE 从 {internal_raw['endpoint_station_mae']:.6f} 轻微变为 {internal_adapted['endpoint_station_mae']:.6f} Mw。
- **八事件开发集**：Event MAE 从 {external_raw['endpoint_event_mae']:.6f} 改善到 {external_adapted['endpoint_event_mae']:.6f} Mw，5/8 事件改善；Station MAE {external_raw['endpoint_station_mae']:.6f} → {external_adapted['endpoint_station_mae']:.6f}，基本不变。
- **逐秒稳定性**：八事件后期事件跳变 p95 {external_raw['late_event_abs_step_p95_mw']:.6f} → {external_adapted['late_event_abs_step_p95_mw']:.6f} Mw；历史改写 p95 {external_raw['late_confirmed_cumulative_log10_l1_p95']:.6f} → {external_adapted['late_confirmed_cumulative_log10_l1_p95']:.6f}。

这些结果支持“adapter 可以降低 Phase39 的后期逐秒波动，同时基本保留端点精度”。但 Phase45 原 validation gate 没有通过，所以本文仍是事后诊断，不把它升级为正式选中模型。

## 1. 训练集损失函数

![训练损失](figures/01_training_loss.png)

[PDF 图件](figures/01_training_loss.pdf)

Phase45 训练目标为五项固定加权、固定 normalizer 后的和：

| epoch27 分量 | 原始值 | 加权归一化贡献 |
|---|---:|---:|
| Endpoint science | {selected['train_endpoint_science_raw']:.6f} | {selected['train_endpoint_science_weighted_normalized']:.6f} |
| Sequence target | {selected['train_sequence_target_raw']:.6f} | {selected['train_sequence_target_weighted_normalized']:.6f} |
| Endpoint teacher | {selected['train_endpoint_teacher_raw']:.6f} | {selected['train_endpoint_teacher_weighted_normalized']:.6f} |
| Late step | {selected['train_late_step_raw']:.6f} | {selected['train_late_step_weighted_normalized']:.6f} |
| Confirmed history | {selected['train_confirmed_history_raw']:.6f} | {selected['train_confirmed_history_weighted_normalized']:.6f} |
| **总计** | — | **{selected['train_total_normalized_loss']:.6f}** |

这个 loss 是训练目标的无量纲组合，不能直接与 Mw MAE 比数值大小。

## 2. internal test 泛化性能

![internal test](figures/02_internal_test.png)

[PDF 图件](figures/02_internal_test.pdf)

| 指标 | Phase39 raw | Phase45 adapter | 变化 |
|---|---:|---:|---:|
| Event MAE | {internal_raw['endpoint_event_mae']:.6f} | {internal_adapted['endpoint_event_mae']:.6f} | {internal_adapted['endpoint_event_mae'] - internal_raw['endpoint_event_mae']:+.6f} |
| Station MAE | {internal_raw['endpoint_station_mae']:.6f} | {internal_adapted['endpoint_station_mae']:.6f} | {internal_adapted['endpoint_station_mae'] - internal_raw['endpoint_station_mae']:+.6f} |
| Event step p95 | {internal_raw['late_event_abs_step_p95_mw']:.6f} | {internal_adapted['late_event_abs_step_p95_mw']:.6f} | {_pct_change(internal_raw['late_event_abs_step_p95_mw'], internal_adapted['late_event_abs_step_p95_mw']):+.1f}% |
| Station step p95 | {internal_raw['late_station_abs_step_p95_mw']:.6f} | {internal_adapted['late_station_abs_step_p95_mw']:.6f} | {_pct_change(internal_raw['late_station_abs_step_p95_mw'], internal_adapted['late_station_abs_step_p95_mw']):+.1f}% |
| Confirmed-history p95 | {internal_raw['late_confirmed_cumulative_log10_l1_p95']:.6f} | {internal_adapted['late_confirmed_cumulative_log10_l1_p95']:.6f} | {_pct_change(internal_raw['late_confirmed_cumulative_log10_l1_p95'], internal_adapted['late_confirmed_cumulative_log10_l1_p95']):+.1f}% |

这里的 test 是 `within_event_station`：同一地震的不同台站分散在 train/validation/test，因此它测量未见台站插值，不等于未见事件泛化。

## 3. 八事件逐秒结果

![八事件总体逐秒指标](figures/03_external_overall.png)

[PDF 图件](figures/03_external_overall.pdf)

![八事件逐秒轨迹](figures/04_external_event_trajectories.png)

[PDF 图件](figures/04_external_event_trajectories.pdf)

| 事件 | 参考 Mw | raw 200 s | adapter 200 s | raw 绝对误差 | adapter 绝对误差 | raw 持续收敛 | adapter 持续收敛 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(event_lines)}

逐秒输出从 20 s 到 200 s，发布时间为 `h+6 s`。adapter 改善最终误差的同时可能引入少量滞后：部分已收敛事件的 suffix-stable 时间晚 1–2 秒；Xizang 则从 raw 在 200 s 仍略超 0.15 Mw，变为 adapter 在 192 s 后持续达标。

## 4. 数据角色与限制

- Phase45 没有通过原 validation gate；seed42 epoch27 只能称为预先固定的 audit checkpoint。
- internal test 已按用户要求一次性打开，但它仍是同事件未见台站测试。
- 八事件没有进入模型训练，但已被多轮开发反复使用，角色仍是 `development_validation`，不能当作新的无偏盲测。
- 第一次八事件 CUDA/batch64 运行因浮点批处理差异未通过旧 CPU 端点复现门槛，未发布性能结果。正式结果使用锁定端点原始口径 CPU/batch158，并通过最大台站差 {summary['external']['endpoint_gate']['max_station_prediction_abs_diff_mw']:.2e} Mw。
- grouped test 没有打开。

## 5. 可下载工件

- [训练逐 epoch loss](training_loss_by_epoch.csv)
- [internal test 逐秒总体指标](internal_horizon_metrics.csv)
- [internal test 逐事件逐秒输出](internal_event_predictions.csv)
- [八事件逐事件逐秒输出](external_event_predictions.csv)
- [八事件逐台站逐秒输出](external_station_predictions.csv)
- [八事件逐秒总体指标](external_horizon_metrics.csv)
- [八事件收敛时间](external_event_convergence.csv)
- [机器可读总摘要](summary.json)
- [发布清单与 SHA-256](publication_manifest.json)
- [冻结评估器](../../../scripts/evaluation/evaluate_phase45_posthoc_streaming.py)
- [可复现绘图脚本](../../../scripts/plotting/plot_phase45_posthoc_report_zh.py)

原始/adapter STF rate 立方体保留在本机正式 run 目录，不提交 GitHub；其 SHA-256 已写入总摘要。全仓实现回归：`793 passed, 1 skipped`。
"""


def publish(run_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    training = _read_json(run_root / "training" / "summary.json")
    internal = _read_json(run_root / "internal" / "summary.json")
    external = _read_json(run_root / "external-cpu" / "summary.json")
    if training["status"] != "complete":
        raise ValueError("training trace is incomplete")
    if internal["status"] != "complete" or not internal["internal_test_iterated"]:
        raise ValueError("internal test artifact is incomplete")
    if external["status"] != "complete" or not external["external_data_loaded"]:
        raise ValueError("external artifact is incomplete")
    if external["grouped_test_loaded"] is not False:
        raise ValueError("grouped test must remain unopened")

    table_hashes = _copy_public_tables(run_root, output_dir)
    training_rows = _read_csv(output_dir / "training_loss_by_epoch.csv")
    external_horizon_rows = _read_csv(output_dir / "external_horizon_metrics.csv")
    external_event_rows = _read_csv(output_dir / "external_event_predictions.csv")
    external_convergence_rows = _read_csv(output_dir / "external_event_convergence.csv")

    raw_stable = _stable_horizon(external_horizon_rows, "raw_event_mae")
    adapted_stable = _stable_horizon(external_horizon_rows, "adapted_event_mae")
    improved_events = sum(
        float(row["adapted_final_abs_error"]) < float(row["raw_final_abs_error"])
        for row in external_convergence_rows
    )
    summary = {
        "status": "complete",
        "candidate": "Phase45 seed42 epoch27, audit-only",
        "evaluation_role": "posthoc_internal_test_and_development_validation",
        "training": training,
        "internal": {
            "station_count": internal["station_count"],
            "event_count": internal["event_count"],
            "raw": internal["raw_metrics"],
            "adapted": internal["adapted_metrics"],
            "artifact_sha256": internal["artifact_sha256"],
        },
        "external": {
            "station_count": external["station_count"],
            "event_count": external["event_count"],
            "raw": external["raw_metrics"],
            "adapted": external["adapted_metrics"],
            "endpoint_gate": external["endpoint_raw_reproduction_gate"],
            "raw_stable_event_mae_observation_sec": raw_stable,
            "adapted_stable_event_mae_observation_sec": adapted_stable,
            "improved_event_count": improved_events,
            "events": [
                {
                    key: (
                        None
                        if value == ""
                        else int(float(value))
                        if key.endswith("observation_sec") or key == "station_count_200s"
                        else float(value)
                        if key != "event"
                        else value
                    )
                    for key, value in row.items()
                }
                for row in external_convergence_rows
            ],
            "artifact_sha256": external["artifact_sha256"],
        },
        "processing_delay_sec": 6.0,
        "implementation_commit": "20d35a985623e4976543b63f951c3265f62af335",
        "full_regression": "793 passed, 1 skipped in 47.06s",
        "grouped_test_loaded": False,
        "public_table_sha256": table_hashes,
    }
    _write_json(output_dir / "summary.json", summary)

    _configure_plotting()
    _plot_training_loss(training_rows, figures)
    _plot_internal_summary(internal["raw_metrics"], internal["adapted_metrics"], figures)
    _plot_external_overall(external_horizon_rows, figures)
    _plot_external_events(external_event_rows, figures)
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
        "report_role": "posthoc_internal_test_and_development_validation",
        "implementation_commit": "20d35a985623e4976543b63f951c3265f62af335",
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
        description="Generate the Chinese Phase45 post-hoc evaluation report."
    )
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
                "external_event_mae": summary["external"]["adapted"][
                    "endpoint_event_mae"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
