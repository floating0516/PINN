#!/usr/bin/env python3
"""Publish the authorized Phase73 eight-event development-validation report."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
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
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "docs" / "reports" / "phase73-pgd-guided-stateful-external-zh"
)
COLORS = {
    "phase73": "#0072B2",
    "phase39": "#66717E",
    "crowell": "#D55E00",
    "melgar": "#CC79A7",
    "ruhl": "#009E73",
    "target": "#8E5AA9",
    "band": "#009E73",
    "identity": "#20262E",
    "grid": "#D8DEE6",
    "ink": "#20262E",
}
METHODS = ("phase73", "phase39", "crowell", "melgar", "ruhl")
METHOD_LABELS = {
    "phase73": "Phase73 有状态模型",
    "phase39": "Phase39 独立提案",
    "crowell": "Crowell PGD",
    "melgar": "Melgar PGD",
    "ruhl": "Ruhl PGD",
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    context = (
        gzip.open(path, mode="rt", encoding="utf-8", newline="")
        if path.suffix == ".gz"
        else path.open(mode="r", encoding="utf-8", newline="")
    )
    with context as handle:
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


def _validate_new_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output directory must be new or empty: {path}")


def _configure_plotting() -> None:
    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    )
    for path in candidates:
        if path.is_file():
            from matplotlib import font_manager

            font_manager.fontManager.addfont(str(path))
            mpl.rcParams["font.family"] = font_manager.FontProperties(
                fname=str(path)
            ).get_name()
            break
    mpl.rcParams.update(
        {
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
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


def _metric_lookup(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[int, str], Mapping[str, str]]:
    return {
        (int(row["observation_horizon_sec"]), str(row["method"])): row
        for row in rows
    }


def _metric_series(
    rows: Sequence[Mapping[str, str]],
    *,
    method: str,
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        (row for row in rows if str(row["method"]) == method),
        key=lambda row: int(row["observation_horizon_sec"]),
    )
    return (
        np.asarray([float(row["observation_horizon_sec"]) for row in selected]),
        np.asarray([float(row[field]) for row in selected]),
    )


def _identity_background(axis: Any, lower: float, upper: float) -> None:
    values = np.asarray([lower, upper])
    axis.plot(values, values, color=COLORS["identity"], linestyle="--", linewidth=1.0)
    axis.fill_between(
        values,
        values - 0.15,
        values + 0.15,
        color=COLORS["band"],
        alpha=0.08,
        label="+/- 0.15 Mw",
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")


def plot_overall_metrics(
    rows: Sequence[Mapping[str, str]],
    *,
    figures_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharex=True)
    for method in METHODS:
        horizons, event_mae = _metric_series(
            rows, method=method, field="event_mae_mw"
        )
        _, station_mae = _metric_series(rows, method=method, field="station_mae_mw")
        linewidth = 2.1 if method == "phase73" else 1.35
        axes[0].plot(
            horizons,
            event_mae,
            color=COLORS[method],
            linewidth=linewidth,
            label=METHOD_LABELS[method],
        )
        axes[1].plot(
            horizons,
            station_mae,
            color=COLORS[method],
            linewidth=linewidth,
            label=METHOD_LABELS[method],
        )
    axes[0].axhline(
        0.15,
        color=COLORS["target"],
        linestyle="--",
        linewidth=1.1,
        label="0.15 Mw 参考线",
    )
    axes[0].set_title("8 事件等权 Event MAE")
    axes[1].set_title("8 事件所有台站 Station MAE")
    for axis in axes:
        axis.set_xlabel("观测时长（s）")
        axis.set_ylabel("MAE（Mw）")
        _style_axis(axis)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Phase73 外部 development-validation：逐秒精度，不作为盲测泛化结论",
        y=1.02,
    )
    figure.tight_layout()
    _save_figure(figure, figures_dir / "01_external_overall_metrics")


def plot_event_trajectories(
    rows: Sequence[Mapping[str, str]],
    *,
    figures_dir: Path,
) -> None:
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["event"]), []).append(row)
    if len(grouped) != 8:
        raise ValueError("external report requires exactly eight event trajectories")
    figure, axes = plt.subplots(4, 2, figsize=(13.0, 12.4), sharex=True)
    for axis, event in zip(axes.flat, sorted(grouped), strict=True):
        sequence = sorted(
            grouped[event], key=lambda row: int(row["observation_horizon_sec"])
        )
        horizons = np.asarray(
            [float(row["observation_horizon_sec"]) for row in sequence]
        )
        catalog = float(sequence[0]["mw_catalog"])
        for method in ("phase73", "phase39", "crowell"):
            values = np.asarray(
                [float(row[f"{method}_mw_pred_median"]) for row in sequence]
            )
            axis.plot(
                horizons,
                values,
                color=COLORS[method],
                linewidth=2.0 if method == "phase73" else 1.25,
                label=METHOD_LABELS[method],
            )
        axis.axhline(catalog, color=COLORS["identity"], linestyle="--", linewidth=1.0)
        axis.fill_between(
            horizons,
            catalog - 0.15,
            catalog + 0.15,
            color=COLORS["band"],
            alpha=0.08,
        )
        phase39_final = float(sequence[-1]["phase39_mw_pred_median"])
        phase73_final = float(sequence[-1]["phase73_mw_pred_median"])
        axis.text(
            0.98,
            0.05,
            f"200s: {phase39_final:.3f} -> {phase73_final:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
        axis.set_title(event, fontsize=10)
        _style_axis(axis)
    for axis in axes[-1]:
        axis.set_xlabel("观测时长（s）")
    for axis in axes[:, 0]:
        axis.set_ylabel("事件中位数 Mw")
    axes.flat[0].legend(loc="upper right", frameon=False, fontsize=8)
    figure.suptitle(
        "8 个未训练事件的有状态逐秒轨迹（发布时间 = 观测时长 + 6 s）",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    _save_figure(figure, figures_dir / "02_external_event_trajectories")


def plot_endpoint_and_stability(
    *,
    event_rows: Sequence[Mapping[str, str]],
    station_rows: Sequence[Mapping[str, str]],
    trajectory_metrics: Mapping[str, Mapping[str, float]],
    figures_dir: Path,
) -> None:
    events = sorted(event_rows, key=lambda row: str(row["event"]))
    stations = sorted(station_rows, key=lambda row: (str(row["event"]), str(row["station"])))
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 10.4))

    event_catalog = np.asarray([float(row["mw_catalog"]) for row in events])
    event_phase39 = np.asarray([float(row["phase39_mw_pred_median"]) for row in events])
    event_phase73 = np.asarray([float(row["phase73_mw_pred_median"]) for row in events])
    lower = float(min(event_catalog.min(), event_phase39.min(), event_phase73.min()) - 0.2)
    upper = float(max(event_catalog.max(), event_phase39.max(), event_phase73.max()) + 0.2)
    _identity_background(axes[0, 0], lower, upper)
    axes[0, 0].scatter(event_catalog, event_phase39, color=COLORS["phase39"], label="Phase39", s=42)
    axes[0, 0].scatter(event_catalog, event_phase73, color=COLORS["phase73"], label="Phase73", s=48, marker="D")
    axes[0, 0].set_title("200 秒事件中位数")
    axes[0, 0].set_xlabel("目录 Mw")
    axes[0, 0].set_ylabel("预测 Mw")
    axes[0, 0].legend(frameon=False, fontsize=8)
    _style_axis(axes[0, 0])

    station_catalog = np.asarray([float(row["mw_catalog"]) for row in stations])
    station_phase39 = np.asarray([float(row["phase39_mw_pred"]) for row in stations])
    station_phase73 = np.asarray([float(row["phase73_mw_pred"]) for row in stations])
    lower = float(min(station_catalog.min(), station_phase39.min(), station_phase73.min()) - 0.2)
    upper = float(max(station_catalog.max(), station_phase39.max(), station_phase73.max()) + 0.2)
    _identity_background(axes[0, 1], lower, upper)
    axes[0, 1].scatter(station_catalog, station_phase39, color=COLORS["phase39"], alpha=0.55, s=17, label="Phase39")
    axes[0, 1].scatter(station_catalog, station_phase73, color=COLORS["phase73"], alpha=0.55, s=17, marker="D", label="Phase73")
    axes[0, 1].set_title("200 秒台站输出")
    axes[0, 1].set_xlabel("目录 Mw")
    axes[0, 1].set_ylabel("预测 Mw")
    axes[0, 1].legend(frameon=False, fontsize=8)
    _style_axis(axes[0, 1])

    y = np.arange(len(events))
    phase39_abs = np.asarray([float(row["phase39_abs_error_mw"]) for row in events])
    phase73_abs = np.asarray([float(row["phase73_abs_error_mw"]) for row in events])
    width = 0.35
    axes[1, 0].barh(y - width / 2, phase39_abs, width, color=COLORS["phase39"], label="Phase39")
    axes[1, 0].barh(y + width / 2, phase73_abs, width, color=COLORS["phase73"], label="Phase73")
    axes[1, 0].axvline(0.15, color=COLORS["target"], linestyle="--", linewidth=1.0)
    axes[1, 0].set_yticks(y, [str(row["event"]) for row in events], fontsize=8)
    axes[1, 0].set_xlabel("200 秒绝对误差（Mw）")
    axes[1, 0].set_title("逐事件端点误差")
    axes[1, 0].legend(frameon=False, fontsize=8)
    _style_axis(axes[1, 0])

    keys = (
        ("post120_abs_step_p95_mw", "120 秒后逐秒变化 p95"),
        ("peak_to_final_p95_mw", "峰值到最终回落 p95"),
        ("post160_band_width_p95_mw", "160--200 秒平台宽度 p95"),
    )
    x = np.arange(len(keys))
    phase39_values = [float(trajectory_metrics["phase39"][key]) for key, _ in keys]
    phase73_values = [float(trajectory_metrics["phase73"][key]) for key, _ in keys]
    bars_a = axes[1, 1].bar(x - width / 2, phase39_values, width, color=COLORS["phase39"], label="Phase39")
    bars_b = axes[1, 1].bar(x + width / 2, phase73_values, width, color=COLORS["phase73"], label="Phase73")
    axes[1, 1].set_xticks(x, [label for _, label in keys], rotation=10, ha="right")
    axes[1, 1].set_ylabel("Mw")
    axes[1, 1].set_title("事件轨迹稳定性")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for bars in (bars_a, bars_b):
        for bar in bars:
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    _style_axis(axes[1, 1])
    figure.suptitle("200 秒精度与有状态平台诊断", y=1.01)
    figure.tight_layout()
    _save_figure(figure, figures_dir / "03_external_endpoint_and_stability")


def _event_endpoint_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    output = [row for row in rows if int(row["observation_horizon_sec"]) == 200]
    if len(output) != 8:
        raise ValueError("external report requires eight endpoint event rows")
    return output


def _format_delta(value: float) -> str:
    return f"{value:+.6f}"


def render_readme(summary: Mapping[str, Any]) -> str:
    endpoints = summary["endpoint_metrics"]
    phase73 = endpoints["phase73"]
    phase39 = endpoints["phase39"]
    crowell = endpoints["crowell"]
    trajectory = summary["trajectory_metrics"]
    phase73_stability = trajectory["phase73"]
    phase39_stability = trajectory["phase39"]
    event_lines = []
    for row in summary["endpoint_events"]:
        event_lines.append(
            "| {event} | {catalog:.1f} | {phase39:.3f} | {phase73:.3f} | "
            "{crowell:.3f} | {phase39_error:.3f} | {phase73_error:.3f} |".format(
                event=row["event"],
                catalog=float(row["mw_catalog"]),
                phase39=float(row["phase39_mw_pred_median"]),
                phase73=float(row["phase73_mw_pred_median"]),
                crowell=float(row["crowell_mw_pred_median"]),
                phase39_error=float(row["phase39_abs_error_mw"]),
                phase73_error=float(row["phase73_abs_error_mw"]),
            )
        )
    return f"""# Phase73 PGD 引导有状态流式模型：8 事件外部评估

> 固定对象：Phase73 seed17 epoch27，来自原始 validation campaign 的
> `closest_model.pth`。该模型每秒继承上一秒的 STF/Mw/GRU 状态和平台置信度，
> 输入当前 Phase39 R-only STF 提案及基于原始 E/N/U 的因果 Crowell PGD 提示；
> 没有 adapter、外部平滑、单调 clamp 或 ensemble。本次按用户明确授权仅运行一次。

## 结论

- **200 秒 Event MAE**：Phase73 为 **{phase73['event_mae_mw']:.6f} Mw**，Phase39 为 **{phase39['event_mae_mw']:.6f} Mw**，Crowell PGD 为 **{crowell['event_mae_mw']:.6f} Mw**；Phase73 相对 Phase39 的变化是 **{_format_delta(phase73['event_mae_mw'] - phase39['event_mae_mw'])} Mw**。
- **200 秒 Station MAE**：Phase73 为 **{phase73['station_mae_mw']:.6f} Mw**，Phase39 为 **{phase39['station_mae_mw']:.6f} Mw**；变化是 **{_format_delta(phase73['station_mae_mw'] - phase39['station_mae_mw'])} Mw**。
- **逐事件端点**：Phase73 相对 Phase39 改善 **{summary['improved_event_count_vs_phase39']}/8** 个事件的最终绝对误差。
- **轨迹**：Phase73 的 120 秒后逐秒变化 p95、峰值到最终回落 p95、160--200 秒平台宽度 p95 分别为 **{phase73_stability['post120_abs_step_p95_mw']:.6f} / {phase73_stability['peak_to_final_p95_mw']:.6f} / {phase73_stability['post160_band_width_p95_mw']:.6f} Mw**；Phase39 对应 **{phase39_stability['post120_abs_step_p95_mw']:.6f} / {phase39_stability['peak_to_final_p95_mw']:.6f} / {phase39_stability['post160_band_width_p95_mw']:.6f} Mw**。

## 逐秒总体表现

![总体误差](figures/01_external_overall_metrics.png)

[PDF 图件](figures/01_external_overall_metrics.pdf)

## 8 个事件轨迹

![逐事件轨迹](figures/02_external_event_trajectories.png)

[PDF 图件](figures/02_external_event_trajectories.pdf)

| 事件 | 参考 Mw | Phase39 200 s | Phase73 200 s | Crowell 200 s | Phase39 绝对误差 | Phase73 绝对误差 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(event_lines)}

## 端点与平台诊断

![端点和平台](figures/03_external_endpoint_and_stability.png)

[PDF 图件](figures/03_external_endpoint_and_stability.pdf)

## 数据角色与方法边界

- 这 8 个事件没有用于 Phase73 训练，因而对模型是未训练事件；但它们在历史开发中已经被反复使用，本报告只能标为 `development_validation`，**不能**称为新的无偏盲测或未见事件泛化证明。
- 原始 Phase73 campaign 没有通过完整 validation endpoint gate：200 秒 validation Event/Station MAE 是 0.158228 / 0.179794 Mw；本次外部结果不会用于更换 seed、checkpoint、超参数或推理规则。
- internal test 和 grouped held-out test 均未打开。原始 training campaign 的隐藏数据标志仍是 `internal_test_iterated=false`、`external_data_loaded=false`、`grouped_test_loaded=false`；本报告只记录这一次独立的外部 override。
- Phase39 原始提案端点复现通过，最大台站差为 {summary['endpoint_phase39_reproduction_gate']['max_station_prediction_abs_diff_mw']:.2e} Mw；因此比较建立在锁定的台站身份、波形输入和完整 STF 提案上。
- 每个 horizon 使用 `0 <= t < h` 的原始 E/N/U 计算 PGD，发布时间为 `h + 6 s`。神经波形主干始终只使用 R 分量；输出表示当前对完整 STF 和最终 Mw 的预测，而不是累计释放矩。

## 可审计工件

- [评估摘要](evaluation_summary.json)
- [发布摘要](summary.json)
- [逐秒 Event 指标](external_horizon_metrics.csv)
- [逐事件逐秒输出](external_event_predictions.csv)
- [逐台站逐秒输出（gzip）](external_station_predictions.csv.gz)
- [200 秒逐台站输出](external_endpoint_station_predictions.csv)
- [事件轨迹诊断](external_trajectory_diagnostics.csv)
- [发布清单](publication_manifest.json)
- [冻结评估器](../../../scripts/evaluation/evaluate_phase73_stateful_external.py)
- [绘图脚本](../../../scripts/plotting/plot_phase73_stateful_external_zh.py)
"""


def publish(run_root: Path, output_dir: Path) -> dict[str, Any]:
    _validate_new_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir()

    evaluation_path = run_root / "summary.json"
    evaluation = _read_json(evaluation_path)
    if evaluation.get("status") != "complete":
        raise ValueError("Phase73 external evaluation is incomplete")
    if evaluation.get("evaluation_role") != (
        "user_authorized_one_time_external_development_validation"
    ):
        raise ValueError("unexpected Phase73 external evaluation role")
    if evaluation.get("internal_test_iterated") is not False or evaluation.get(
        "grouped_test_loaded"
    ) is not False:
        raise ValueError("internal and grouped test must remain closed")
    if evaluation.get("external_data_loaded") is not True:
        raise ValueError("external evaluation flag is missing")
    if evaluation["historical_training_campaign"]["external_data_loaded"] is not False:
        raise ValueError("historical Phase73 campaign boundary changed")

    table_names = (
        "external_horizon_metrics.csv",
        "external_event_predictions.csv",
        "external_station_predictions.csv.gz",
        "external_endpoint_station_predictions.csv",
        "external_trajectory_diagnostics.csv",
    )
    runtime_hashes = evaluation["artifact_sha256"]["runtime_outputs"]
    public_hashes: dict[str, str] = {}
    for name in table_names:
        source = run_root / name
        if not source.is_file():
            raise FileNotFoundError(f"missing evaluation table: {source}")
        if runtime_hashes.get(name) != _sha256(source):
            raise ValueError(f"evaluation table hash changed: {name}")
        destination = output_dir / name
        shutil.copy2(source, destination)
        public_hashes[name] = _sha256(destination)
    shutil.copy2(evaluation_path, output_dir / "evaluation_summary.json")
    public_hashes["evaluation_summary.json"] = _sha256(
        output_dir / "evaluation_summary.json"
    )

    horizon_rows = _read_csv(output_dir / "external_horizon_metrics.csv")
    event_rows = _read_csv(output_dir / "external_event_predictions.csv")
    station_rows = _read_csv(output_dir / "external_endpoint_station_predictions.csv")
    endpoint_events = _event_endpoint_rows(event_rows)
    metrics = _metric_lookup(horizon_rows)
    endpoint_metrics = {
        method: {
            "event_mae_mw": float(metrics[(200, method)]["event_mae_mw"]),
            "station_mae_mw": float(metrics[(200, method)]["station_mae_mw"]),
        }
        for method in METHODS
    }
    for method, metrics_by_method in endpoint_metrics.items():
        expected = evaluation["endpoint_metrics"][method]
        for key, actual in metrics_by_method.items():
            if not math.isclose(
                actual,
                float(expected[key]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    f"published {method} {key} no longer reproduces evaluation"
                )
    if len(endpoint_events) != int(evaluation["event_count"]):
        raise ValueError("external endpoint event count changed")
    if len(station_rows) != int(evaluation["station_count"]):
        raise ValueError("external endpoint station count changed")
    trajectory_metrics = evaluation["trajectory_metrics"]
    summary = {
        "status": "complete",
        "candidate": evaluation["candidate"],
        "evaluation_role": evaluation["evaluation_role"],
        "cohort_role": evaluation["cohort_role"],
        "event_count": evaluation["event_count"],
        "station_count": evaluation["station_count"],
        "processing_delay_sec": evaluation["processing_delay_sec"],
        "endpoint_metrics": endpoint_metrics,
        "improved_event_count_vs_phase39": evaluation[
            "improved_event_count_vs_phase39"
        ],
        "endpoint_phase39_reproduction_gate": evaluation[
            "endpoint_phase39_reproduction_gate"
        ],
        "trajectory_metrics": trajectory_metrics,
        "formal_validation_gate_passed": evaluation[
            "formal_validation_gate_passed"
        ],
        "historical_training_campaign": evaluation["historical_training_campaign"],
        "internal_test_iterated": False,
        "external_data_loaded": True,
        "grouped_test_loaded": False,
        "source_evaluation": {
            "run_root": str(run_root),
            "summary_sha256": _sha256(evaluation_path),
            "artifact_sha256": evaluation["artifact_sha256"],
        },
        "endpoint_events": endpoint_events,
        "public_table_sha256": public_hashes,
    }
    _configure_plotting()
    plot_overall_metrics(horizon_rows, figures_dir=figures_dir)
    plot_event_trajectories(event_rows, figures_dir=figures_dir)
    plot_endpoint_and_stability(
        event_rows=endpoint_events,
        station_rows=station_rows,
        trajectory_metrics=trajectory_metrics,
        figures_dir=figures_dir,
    )
    (output_dir / "README.md").write_text(render_readme(summary), encoding="utf-8")
    _write_json(output_dir / "summary.json", summary)

    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "publication_manifest.json"
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_role": "user_authorized_one_time_external_development_validation",
        "run_root": str(run_root),
        "generator": {
            "path": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(SCRIPT_PATH),
        },
        "evaluator": {
            "path": "scripts/evaluation/evaluate_phase73_stateful_external.py",
            "sha256": _sha256(
                REPO_ROOT / "scripts/evaluation/evaluate_phase73_stateful_external.py"
            ),
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
        description="Publish the Phase73 eight-event external report in Chinese."
    )
    parser.add_argument("--run-root", required=True, type=Path)
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
                "phase73_event_mae_200s": summary["endpoint_metrics"]["phase73"][
                    "event_mae_mw"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
