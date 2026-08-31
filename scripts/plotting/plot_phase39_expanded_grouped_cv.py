from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path("/home/lihe/PINN_Mag")
DEFAULT_RUN_ROOT = (
    WORKSPACE_ROOT / "runs/phase39-expanded-grouped-cv-20260831T0407Z"
)
DEFAULT_BASELINE_RUN_ROOT = (
    WORKSPACE_ROOT / "runs/phase39-confirmatory-grouped-cv-20260812T0332Z-121197d"
)
EXPECTED_EVENTS = 39
EXPECTED_STATIONS = 2694
EXPECTED_RUNS = 15
SEEDS = (17, 42, 73)
MAP_EVENTS = ("Noto2024", "us7000i9bw", "Ridgecrest2019")
WAVEFORM_EVENT = "Noto2024"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments import run_phase39_confirmatory_grouped_cv as grouped
from scripts.plotting.plot_phase39_moment_scaling_explainer import (
    iter_selected_records,
)
from src.utils.provenance import sha256_file, utc_now_iso


COLORS = {
    "existing": "#4C78A8",
    "new": "#169C78",
    "noto": "#D1495B",
    "truth": "#202124",
    "target": "#E9A23B",
    "grid": "#D7DCE2",
    "station": "#6C757D",
}


def translated(language: str, english: str, chinese: str) -> str:
    return chinese if language == "zh" else english


def configure_matplotlib(language: str) -> None:
    family = "DejaVu Sans"
    if language == "zh":
        font_path = Path(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        )
        if font_path.exists():
            from matplotlib.font_manager import FontProperties

            family = FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_figure(figure: plt.Figure, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, dpi in ((".png", 220), (".pdf", 300)):
        path = output_stem.with_suffix(suffix)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def identity_limits(*series: pd.Series, padding: float = 0.16) -> tuple[float, float]:
    values = np.concatenate([item.to_numpy(dtype=float) for item in series])
    return float(np.nanmin(values) - padding), float(np.nanmax(values) + padding)


def event_class(event: str, role: str) -> str:
    if event == "Noto2024":
        return "noto"
    return "new" if role == "new_event" else "existing"


def class_label(category: str, language: str) -> str:
    labels = {
        "existing": ("Existing Phase 39 events", "原 Phase 39 事件"),
        "new": ("Eight newly added events", "8 个新增事件"),
        "noto": ("Noto 2024", "2024 年能登地震"),
    }
    english, chinese = labels[category]
    return translated(language, english, chinese)


def load_campaign_inputs(
    run_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
    list[pd.DataFrame],
    Path,
]:
    complete_path = run_root / "COMPLETE"
    if not complete_path.exists():
        raise RuntimeError(f"campaign is not complete: {complete_path}")
    summary_path = run_root / "campaign_summary.json"
    summary = load_json(summary_path)
    protocol = load_json(run_root / "protocol.json")
    if summary.get("status") != "complete":
        raise ValueError("campaign summary is not complete")
    if int(summary.get("event_count", -1)) != EXPECTED_EVENTS:
        raise ValueError("event count changed")
    if int(summary.get("station_count", -1)) != EXPECTED_STATIONS:
        raise ValueError("station count changed")
    if int(summary.get("completed_runs", -1)) != EXPECTED_RUNS:
        raise ValueError("completed run count changed")

    event_ensemble = pd.read_csv(
        summary["artifacts"]["event_oof_seed_ensemble"]
    )
    station_ensemble = pd.read_csv(
        summary["artifacts"]["station_oof_seed_ensemble"]
    )
    event_all = pd.read_csv(summary["artifacts"]["event_oof_all_seeds"])
    station_all = pd.read_csv(summary["artifacts"]["station_oof_all_seeds"])
    if len(event_ensemble) != EXPECTED_EVENTS or len(event_all) != EXPECTED_EVENTS * 3:
        raise ValueError("event OOF coverage changed")
    if (
        len(station_ensemble) != EXPECTED_STATIONS
        or len(station_all) != EXPECTED_STATIONS * 3
    ):
        raise ValueError("station OOF coverage changed")
    station_metadata = (
        station_all[
            [
                "event",
                "station",
                "epicentral_distance_km",
                "source_distance_km",
                "azimuth_deg",
            ]
        ]
        .groupby(["event", "station"], as_index=False)
        .agg(
            epicentral_distance_km=("epicentral_distance_km", "median"),
            source_distance_km=("source_distance_km", "median"),
            azimuth_deg=("azimuth_deg", "median"),
        )
    )
    station_ensemble = station_ensemble.merge(
        station_metadata,
        on=["event", "station"],
        validate="one_to_one",
    )

    folds_payload = load_json(run_root / "event_folds.json")
    folds = pd.DataFrame(folds_payload["events"])[
        ["event", "fold", "n_stations", "magnitude_catalog"]
    ]
    dataset_root = Path(protocol["data_path"]).resolve().parent
    sources_path = dataset_root / "provenance/event_sources.csv"
    sources = pd.read_csv(sources_path)[["event", "role", "source_event_id"]]
    event_ensemble = event_ensemble.merge(folds, on="event", validate="one_to_one")
    event_ensemble = event_ensemble.merge(sources, on="event", validate="one_to_one")
    event_ensemble["event_class"] = [
        event_class(event, role)
        for event, role in zip(event_ensemble["event"], event_ensemble["role"])
    ]
    station_ensemble = station_ensemble.merge(
        folds[["event", "fold"]], on="event", validate="many_to_one"
    )
    station_ensemble = station_ensemble.merge(
        sources[["event", "role"]], on="event", validate="many_to_one"
    )
    station_ensemble["event_class"] = [
        event_class(event, role)
        for event, role in zip(station_ensemble["event"], station_ensemble["role"])
    ]

    run_summaries = []
    training_logs = []
    for fold in range(5):
        for seed in SEEDS:
            run_dir = run_root / f"fold_{fold}/phase39/seed_{seed}"
            run_summary = load_json(run_dir / "run_summary.json")
            if run_summary.get("status") != "complete":
                raise ValueError(f"incomplete run summary: fold={fold}, seed={seed}")
            run_summaries.append(run_summary)
            log_paths = sorted((run_dir / "logs").glob("training_log_*.csv"))
            if len(log_paths) != 1:
                raise ValueError(f"unexpected training log count in {run_dir}")
            log = pd.read_csv(log_paths[0])
            log["fold"] = fold
            log["seed"] = seed
            training_logs.append(log)
    return (
        summary,
        protocol,
        event_ensemble,
        station_ensemble,
        event_all,
        station_all,
        run_summaries,
        training_logs,
        sources_path,
    )


def plot_event_scatter(
    events: pd.DataFrame,
    output_stem: Path,
    *,
    language: str,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8.0, 7.0), constrained_layout=True)
    lower, upper = identity_limits(events["mw_catalog"], events["mw_pred_median"])
    x = np.linspace(lower, upper, 200)
    axis.fill_between(
        x,
        x - 0.2,
        x + 0.2,
        color=COLORS["target"],
        alpha=0.14,
        label=translated(language, "+/-0.20 Mw target", "+/-0.20 Mw 目标范围"),
    )
    axis.plot(x, x, color=COLORS["truth"], linewidth=1.3)
    styles = {
        "existing": ("o", 58),
        "new": ("^", 78),
        "noto": ("*", 180),
    }
    for category in ("existing", "new", "noto"):
        rows = events[events["event_class"] == category]
        marker, size = styles[category]
        axis.errorbar(
            rows["mw_catalog"],
            rows["mw_pred_median"],
            yerr=rows["prediction_seed_std"],
            fmt="none",
            ecolor=COLORS[category],
            elinewidth=0.9,
            alpha=0.55,
            capsize=2,
            zorder=2,
        )
        axis.scatter(
            rows["mw_catalog"],
            rows["mw_pred_median"],
            marker=marker,
            s=size,
            color=COLORS[category],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.92,
            label=class_label(category, language),
            zorder=3,
        )
    label_rows = pd.concat(
        [
            events.nlargest(4, "absolute_error"),
            events[events["event_class"] == "new"].nlargest(2, "absolute_error"),
        ]
    ).drop_duplicates("event")
    for row in label_rows.itertuples(index=False):
        axis.annotate(
            str(row.event),
            (float(row.mw_catalog), float(row.mw_pred_median)),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    metrics = events["absolute_error"]
    axis.text(
        0.03,
        0.97,
        translated(
            language,
            f"OOF event MAE = {metrics.mean():.3f} Mw\nWithin +/-0.20 = {(metrics <= 0.2).mean():.1%}",
            f"OOF 事件 MAE = {metrics.mean():.3f} Mw\n落入 +/-0.20 = {(metrics <= 0.2).mean():.1%}",
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": COLORS["grid"], "alpha": 0.9},
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(translated(language, "Catalog magnitude (Mw)", "目录震级（Mw）"))
    axis.set_ylabel(translated(language, "OOF estimated magnitude (Mw)", "OOF 估计震级（Mw）"))
    axis.set_title(
        translated(
            language,
            "Phase 39 expanded dataset: event-level out-of-fold estimates",
            "Phase 39 扩展数据集：事件级折外估计",
        ),
        loc="left",
        fontweight="bold",
    )
    axis.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.75)
    axis.legend(frameon=False, loc="lower right", fontsize=9)
    return save_figure(figure, output_stem)


def plot_station_hexbin(
    stations: pd.DataFrame,
    events: pd.DataFrame,
    output_stem: Path,
    *,
    language: str,
) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.7), constrained_layout=True)
    lower, upper = identity_limits(stations["mw_catalog"], stations["mw_pred"])
    for axis in axes:
        axis.plot([lower, upper], [lower, upper], color=COLORS["truth"], linewidth=1.1)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.grid(True, color=COLORS["grid"], linewidth=0.55, alpha=0.65)
        axis.set_xlabel(translated(language, "Catalog magnitude (Mw)", "目录震级（Mw）"))
        axis.set_ylabel(translated(language, "OOF estimated magnitude (Mw)", "OOF 估计震级（Mw）"))

    density = axes[0].hexbin(
        stations["mw_catalog"],
        stations["mw_pred"],
        gridsize=42,
        mincnt=1,
        cmap="viridis",
        linewidths=0.0,
    )
    colorbar = figure.colorbar(density, ax=axes[0], pad=0.02)
    colorbar.set_label(translated(language, "Station count", "台站数量"))
    axes[0].set_title(
        translated(language, "A. All 2,694 station records", "A. 全部 2,694 条台站记录"),
        loc="left",
        fontweight="bold",
    )

    axes[1].scatter(
        stations["mw_catalog"],
        stations["mw_pred"],
        s=8,
        color=COLORS["station"],
        alpha=0.13,
        linewidth=0,
        rasterized=True,
        label=translated(language, "Stations", "台站"),
    )
    for category, marker, size in (
        ("existing", "o", 48),
        ("new", "^", 72),
        ("noto", "*", 150),
    ):
        rows = events[events["event_class"] == category]
        axes[1].scatter(
            rows["mw_catalog"],
            rows["mw_pred_median"],
            marker=marker,
            s=size,
            color=COLORS[category],
            edgecolor="white",
            linewidth=0.6,
            label=class_label(category, language),
            zorder=3,
        )
    axes[1].set_title(
        translated(
            language,
            "B. Station estimates with event medians",
            "B. 台站估计与事件中位数",
        ),
        loc="left",
        fontweight="bold",
    )
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle(
        translated(
            language,
            "Station-level out-of-fold magnitude estimates",
            "台站级折外震级估计",
        ),
        fontsize=14,
        fontweight="bold",
    )
    return save_figure(figure, output_stem)


def plot_event_errors(
    events: pd.DataFrame,
    output_stem: Path,
    *,
    language: str,
) -> list[Path]:
    ordered = events.sort_values("absolute_error", ascending=True).reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(11.0, 10.5), constrained_layout=True)
    positions = np.arange(len(ordered))
    colors = [COLORS[value] for value in ordered["event_class"]]
    bars = axis.barh(positions, ordered["absolute_error"], color=colors, alpha=0.92)
    axis.axvline(
        0.2,
        color=COLORS["target"],
        linewidth=1.5,
        linestyle="--",
        label=translated(language, "0.20 Mw target", "0.20 Mw 目标"),
    )
    axis.set_yticks(positions, ordered["event"])
    axis.set_xlabel(translated(language, "Absolute OOF error (Mw)", "OOF 绝对误差（Mw）"))
    axis.set_title(
        translated(
            language,
            "Absolute out-of-fold error for all 39 earthquakes",
            "39 个地震事件的折外绝对误差",
        ),
        loc="left",
        fontweight="bold",
    )
    axis.grid(True, axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.75)
    label_limit = max(0.45, float(ordered["absolute_error"].max()) * 1.16)
    axis.set_xlim(0.0, label_limit)
    for bar, value in zip(bars, ordered["absolute_error"]):
        axis.text(
            float(value) + label_limit * 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.3f}",
            va="center",
            fontsize=7.5,
        )
    handles = [
        Line2D([], [], color=COLORS[category], linewidth=7, label=class_label(category, language))
        for category in ("existing", "new", "noto")
    ]
    handles.append(
        Line2D([], [], color=COLORS["target"], linestyle="--", label=translated(language, "0.20 Mw target", "0.20 Mw 目标"))
    )
    axis.legend(handles=handles, frameon=False, loc="lower right", fontsize=8)
    return save_figure(figure, output_stem)


def run_metrics_frame(run_summaries: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for summary in run_summaries:
        metrics = summary["event_metrics"]
        rows.append(
            {
                "fold": int(summary["fold"]),
                "seed": int(summary["seed"]),
                "event_mae": float(metrics["event_mae"]),
                "event_rmse": float(metrics["event_rmse"]),
                "station_mae": float(metrics["station_mae"]),
                "station_rmse": float(metrics["station_rmse"]),
                "test_event_count": int(summary["test_event_count"]),
                "station_count": int(metrics["station_count"]),
                "checkpoint_sha256": str(summary["checkpoint_sha256"]),
            }
        )
    result = pd.DataFrame(rows).sort_values(["fold", "seed"]).reset_index(drop=True)
    if len(result) != EXPECTED_RUNS:
        raise ValueError("run metric count changed")
    return result


def annotated_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    figure: plt.Figure,
) -> None:
    image = axis.imshow(values, cmap="YlGnBu", aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            axis.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=9)
    axis.set_xticks(np.arange(len(SEEDS)), [str(seed) for seed in SEEDS])
    axis.set_yticks(np.arange(5), [f"Fold {fold}" for fold in range(5)])
    axis.set_xlabel("Seed")
    axis.set_title(title, loc="left", fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(colorbar_label)


def plot_fold_seed_metrics(
    run_metrics: pd.DataFrame,
    output_stem: Path,
    *,
    language: str,
) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.1), constrained_layout=True)
    event_values = run_metrics.pivot(index="fold", columns="seed", values="event_mae").loc[
        range(5), list(SEEDS)
    ].to_numpy()
    station_values = run_metrics.pivot(index="fold", columns="seed", values="station_mae").loc[
        range(5), list(SEEDS)
    ].to_numpy()
    annotated_heatmap(
        axes[0],
        event_values,
        title=translated(language, "A. Event-level OOF MAE", "A. 事件级 OOF MAE"),
        colorbar_label="MAE (Mw)",
        figure=figure,
    )
    annotated_heatmap(
        axes[1],
        station_values,
        title=translated(language, "B. Station-level OOF MAE", "B. 台站级 OOF MAE"),
        colorbar_label="MAE (Mw)",
        figure=figure,
    )
    figure.suptitle(
        translated(
            language,
            "Five-fold grouped cross-validation stability across three seeds",
            "五折事件分组交叉验证在三个随机种子下的稳定性",
        ),
        fontsize=14,
        fontweight="bold",
    )
    return save_figure(figure, output_stem)


def aggregate_training_logs(logs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(logs, ignore_index=True)
    columns = [
        "Train_Loss",
        "Train_Data_Loss",
        "Train_Phys_Loss",
        "validation_station_mae_catalog",
        "validation_event_mae_catalog",
        "LR",
    ]
    rows = []
    for epoch, frame in combined.groupby("Epoch"):
        row: dict[str, float] = {"Epoch": int(epoch), "n_runs": len(frame)}
        for column in columns:
            values = frame[column].to_numpy(dtype=float)
            row[f"{column}_median"] = float(np.nanmedian(values))
            row[f"{column}_q25"] = float(np.nanquantile(values, 0.25))
            row[f"{column}_q75"] = float(np.nanquantile(values, 0.75))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Epoch")


def plot_training_curves(
    curves: pd.DataFrame,
    output_stem: Path,
    *,
    language: str,
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)
    x = curves["Epoch"].to_numpy()
    series = (
        ("Train_Loss", translated(language, "Total loss", "总损失"), "#3B6EA8"),
        ("Train_Data_Loss", translated(language, "Data loss", "数据损失"), "#D9853B"),
        ("Train_Phys_Loss", translated(language, "Physics loss", "物理损失"), "#2C9C69"),
    )
    for column, label, color in series:
        median = curves[f"{column}_median"].to_numpy()
        q25 = curves[f"{column}_q25"].to_numpy()
        q75 = curves[f"{column}_q75"].to_numpy()
        axes[0, 0].plot(x, median, color=color, linewidth=1.8, label=label)
        axes[0, 0].fill_between(x, q25, q75, color=color, alpha=0.12)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title(translated(language, "A. Training objectives", "A. 训练目标"), loc="left", fontweight="bold")
    axes[0, 0].set_ylabel(translated(language, "Loss (log scale)", "损失（对数尺度）"))
    axes[0, 0].legend(frameon=False, fontsize=8)

    for column, label, color in (
        (
            "validation_event_mae_catalog",
            translated(language, "Validation event MAE", "验证事件 MAE"),
            COLORS["existing"],
        ),
        (
            "validation_station_mae_catalog",
            translated(language, "Validation station MAE", "验证台站 MAE"),
            COLORS["new"],
        ),
    ):
        median = curves[f"{column}_median"].to_numpy()
        q25 = curves[f"{column}_q25"].to_numpy()
        q75 = curves[f"{column}_q75"].to_numpy()
        axes[0, 1].plot(x, median, color=color, linewidth=1.8, label=label)
        axes[0, 1].fill_between(x, q25, q75, color=color, alpha=0.14)
    axes[0, 1].axhline(0.2, color=COLORS["target"], linestyle="--", linewidth=1.2)
    axes[0, 1].set_title(translated(language, "B. Validation magnitude error", "B. 验证震级误差"), loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("MAE (Mw)")
    axes[0, 1].legend(frameon=False, fontsize=8)

    lr = curves["LR_median"].to_numpy()
    axes[1, 0].plot(x, lr, color="#7A5195", linewidth=1.7)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title(translated(language, "C. Learning-rate schedule", "C. 学习率调度"), loc="left", fontweight="bold")
    axes[1, 0].set_ylabel(translated(language, "Learning rate", "学习率"))

    axes[1, 1].plot(x, curves["n_runs"], color=COLORS["station"], linewidth=1.8)
    axes[1, 1].set_ylim(0, EXPECTED_RUNS + 1)
    axes[1, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1, 1].set_title(
        translated(language, "D. Runs contributing at each epoch", "D. 各轮次参与汇总的运行数"),
        loc="left",
        fontweight="bold",
    )
    axes[1, 1].set_ylabel(translated(language, "Run count", "运行数量"))

    for axis in axes.flat:
        axis.set_xlabel(translated(language, "Epoch", "训练轮次"))
        axis.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    figure.suptitle(
        translated(
            language,
            "Phase 39 training histories across 15 grouped-CV models (median and IQR)",
            "Phase 39 的 15 个分组交叉验证模型训练历史（中位数与四分位距）",
        ),
        fontsize=14,
        fontweight="bold",
    )
    return save_figure(figure, output_stem)


def coordinate_frame(
    data_path: Path,
    station_keys: set[tuple[str, str]],
    selected_events: set[str],
) -> pd.DataFrame:
    rows = []
    with np.load(data_path, allow_pickle=True) as data:
        for record in iter_selected_records(data, selected_events):
            key = (record.event, record.station)
            if key not in station_keys:
                continue
            rows.append(
                {
                    "event": record.event,
                    "station": record.station,
                    "event_lon": record.event_lon,
                    "event_lat": record.event_lat,
                    "station_lon": record.station_lon,
                    "station_lat": record.station_lat,
                }
            )
    result = pd.DataFrame(rows).drop_duplicates(["event", "station"])
    expected = sum(event in selected_events for event, _station in station_keys)
    if len(result) != expected:
        raise ValueError(f"coordinate join changed: expected={expected}, actual={len(result)}")
    return result


def map_region(frame: pd.DataFrame) -> tuple[float, float, float, float]:
    longitudes = np.concatenate(
        [frame["station_lon"].to_numpy(), frame["event_lon"].to_numpy()]
    )
    latitudes = np.concatenate(
        [frame["station_lat"].to_numpy(), frame["event_lat"].to_numpy()]
    )
    lon_span = max(float(np.ptp(longitudes)), 0.1)
    lat_span = max(float(np.ptp(latitudes)), 0.1)
    return (
        float(np.min(longitudes) - max(0.08, 0.10 * lon_span)),
        float(np.max(longitudes) + max(0.08, 0.10 * lon_span)),
        float(np.min(latitudes) - max(0.08, 0.10 * lat_span)),
        float(np.max(latitudes) + max(0.08, 0.10 * lat_span)),
    )


def coastline_segments(region: tuple[float, float, float, float]) -> list[np.ndarray]:
    west, east, south, north = region
    with tempfile.TemporaryDirectory(prefix="phase39-expanded-map-") as temp_name:
        result = subprocess.run(
            [
                "gmt",
                "coast",
                f"-R{west:.6f}/{east:.6f}/{south:.6f}/{north:.6f}",
                "-M",
                "-W0.5p",
                "-Dh",
            ],
            cwd=temp_name,
            check=True,
            text=True,
            capture_output=True,
        )
    segments: list[np.ndarray] = []
    current: list[tuple[float, float]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            if len(current) >= 2:
                segments.append(np.asarray(current, dtype=float))
            current = []
            continue
        tokens = stripped.split()
        if len(tokens) >= 2:
            current.append((float(tokens[0]), float(tokens[1])))
    if len(current) >= 2:
        segments.append(np.asarray(current, dtype=float))
    return segments


def plot_selected_maps(
    station_predictions: pd.DataFrame,
    data_path: Path,
    output_stem: Path,
    *,
    language: str,
) -> list[Path]:
    if subprocess.run(["gmt", "--version"], capture_output=True).returncode != 0:
        raise RuntimeError("GMT is required for station maps")
    selected = station_predictions[station_predictions["event"].isin(MAP_EVENTS)].copy()
    selected["residual_mw"] = selected["mw_pred"] - selected["mw_catalog"]
    keys = set(zip(selected["event"], selected["station"]))
    coordinates = coordinate_frame(data_path, keys, set(MAP_EVENTS))
    joined = selected.merge(coordinates, on=["event", "station"], validate="one_to_one")
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.7), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=0.8)
    station_marks = None
    for axis, event in zip(axes, MAP_EVENTS):
        frame = joined[joined["event"] == event]
        region = map_region(frame)
        west, east, south, north = region
        axis.set_facecolor("#EAF4FB")
        for segment in coastline_segments(region):
            axis.plot(segment[:, 0], segment[:, 1], color="#59636E", linewidth=0.75)
        size = 11 if len(frame) > 200 else 23 if len(frame) > 50 else 48
        station_marks = axis.scatter(
            frame["station_lon"],
            frame["station_lat"],
            c=frame["residual_mw"].clip(-0.8, 0.8),
            cmap="coolwarm",
            norm=norm,
            s=size,
            edgecolor="#263238",
            linewidth=0.25,
            alpha=0.88,
            rasterized=True,
            zorder=3,
        )
        axis.scatter(
            [float(frame["event_lon"].iloc[0])],
            [float(frame["event_lat"].iloc[0])],
            marker="*",
            s=220,
            color="#F2C94C",
            edgecolor="#202124",
            linewidth=0.8,
            zorder=4,
        )
        axis.set_xlim(west, east)
        axis.set_ylim(south, north)
        mean_latitude = 0.5 * (south + north)
        axis.set_aspect(1.0 / max(math.cos(math.radians(mean_latitude)), 0.2))
        axis.set_xlabel(translated(language, "Longitude (deg)", "经度（度）"))
        axis.set_ylabel(translated(language, "Latitude (deg)", "纬度（度）"))
        event_error = abs(
            float(frame["mw_pred"].median()) - float(frame["mw_catalog"].iloc[0])
        )
        axis.set_title(
            translated(
                language,
                f"{event}\n{len(frame)} stations | event error {event_error:.3f} Mw",
                f"{event}\n{len(frame)} 个台站 | 事件误差 {event_error:.3f} Mw",
            ),
            fontweight="bold",
        )
        axis.grid(True, color="#C9D3DC", linewidth=0.5, alpha=0.75)
    if station_marks is None:
        raise ValueError("selected map cohort is empty")
    colorbar = figure.colorbar(station_marks, ax=axes, orientation="horizontal", pad=0.08)
    colorbar.set_label(
        translated(
            language,
            "Station residual: estimated Mw - catalog Mw (clipped at +/-0.8)",
            "台站残差：估计 Mw - 目录 Mw（截断至 +/-0.8）",
        )
    )
    figure.suptitle(
        translated(
            language,
            "Epicenters, GNSS stations, and OOF residuals for selected events",
            "典型事件的震中、GNSS 台站与 OOF 残差",
        ),
        fontsize=14,
        fontweight="bold",
    )
    return save_figure(figure, output_stem)


def load_waveform_sample(
    config_path: Path,
    station_predictions: pd.DataFrame,
) -> tuple[dict[str, Any], str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    arm = grouped.build_arm_configs(config)["phase39"]
    event_rows = station_predictions[station_predictions["event"] == WAVEFORM_EVENT].copy()
    event_rows["absolute_error"] = event_rows["error_vs_catalog"].abs()
    target_error = float(event_rows["absolute_error"].median())
    event_rows["selection_distance"] = (
        event_rows["absolute_error"] - target_error
    ).abs()
    selected_station = str(
        event_rows.sort_values(["selection_distance", "station"]).iloc[0]["station"]
    )
    for sample in grouped._load_dataset_samples(arm):
        if sample["event"] == WAVEFORM_EVENT and sample["station"] == selected_station:
            return sample, selected_station
    raise ValueError(f"waveform sample not found: {WAVEFORM_EVENT}/{selected_station}")


def plot_waveform_example(
    sample: dict[str, Any],
    selected_station: str,
    stations: pd.DataFrame,
    station_all: pd.DataFrame,
    events: pd.DataFrame,
    output_stem: Path,
    *,
    language: str,
) -> list[Path]:
    event_stations = stations[stations["event"] == WAVEFORM_EVENT].sort_values(
        "epicentral_distance_km"
    )
    selected_all = station_all[
        (station_all["event"] == WAVEFORM_EVENT)
        & (station_all["station"] == selected_station)
    ].sort_values("seed")
    event_row = events[events["event"] == WAVEFORM_EVENT].iloc[0]
    radial_cm = np.asarray(sample["radial"], dtype=float) * 100.0
    tangential_cm = np.asarray(sample["tangential"], dtype=float) * 100.0
    vertical_cm = np.asarray(sample["vertical"], dtype=float) * 100.0
    time_sec = float(sample["waveform_start_sec"]) + np.arange(radial_cm.size) * float(
        sample["waveform_dt_sec"]
    )
    norm_cm = np.sqrt(radial_cm**2 + tangential_cm**2 + vertical_cm**2)
    pgd_cm = np.maximum.accumulate(norm_cm)
    catalog = float(event_row["mw_catalog"])

    figure, axes = plt.subplots(3, 1, figsize=(12.5, 9.3), constrained_layout=True)
    axes[0].plot(time_sec, radial_cm, color=COLORS["existing"], linewidth=1.2)
    axes[0].axhline(0.0, color=COLORS["grid"], linewidth=0.8)
    axes[0].set_ylabel(translated(language, "Radial displacement (cm)", "径向位移（cm）"))
    axes[0].set_title(
        translated(language, "A. Radial waveform used by the R-only model", "A. R-only 模型使用的径向波形"),
        loc="left",
        fontweight="bold",
    )

    axes[1].plot(time_sec, norm_cm, color="#8D99A6", linewidth=1.0, label=translated(language, "3-component norm", "三分量位移模长"))
    axes[1].plot(time_sec, pgd_cm, color=COLORS["new"], linewidth=2.0, label=translated(language, "Cumulative PGD", "累积 PGD"))
    axes[1].set_ylabel(translated(language, "Displacement / PGD (cm)", "位移 / PGD（cm）"))
    axes[1].set_title(translated(language, "B. Peak ground displacement", "B. 峰值地表位移"), loc="left", fontweight="bold")
    axes[1].legend(frameon=False)

    axes[2].scatter(
        event_stations["epicentral_distance_km"],
        event_stations["mw_pred"],
        s=13,
        color=COLORS["station"],
        alpha=0.38,
        linewidth=0,
        rasterized=True,
        label=translated(language, "Station seed-ensemble estimates", "台站种子集成估计"),
    )
    selected_ensemble = event_stations[event_stations["station"] == selected_station].iloc[0]
    axes[2].scatter(
        [float(selected_ensemble["epicentral_distance_km"])],
        [float(selected_ensemble["mw_pred"])],
        marker="D",
        s=75,
        color=COLORS["new"],
        edgecolor="white",
        linewidth=0.7,
        label=translated(language, "Waveform station", "波形示例台站"),
        zorder=4,
    )
    axes[2].axhline(catalog, color=COLORS["truth"], linewidth=1.4, label=translated(language, f"Catalog Mw {catalog:.2f}", f"目录 Mw {catalog:.2f}"))
    axes[2].axhline(float(event_row["mw_pred_median"]), color=COLORS["noto"], linewidth=1.5, linestyle="--", label=translated(language, f"OOF event estimate {float(event_row['mw_pred_median']):.2f}", f"OOF 事件估计 {float(event_row['mw_pred_median']):.2f}"))
    axes[2].set_xlabel(translated(language, "Epicentral distance (km)", "震中距（km）"))
    axes[2].set_ylabel(translated(language, "Estimated magnitude (Mw)", "估计震级（Mw）"))
    axes[2].set_title(
        translated(
            language,
            f"C. Endpoint OOF estimates; waveform station seeds = {', '.join(f'{value:.2f}' for value in selected_all['mw_pred'])}",
            f"C. 终点 OOF 估计；示例台站三个种子 = {', '.join(f'{value:.2f}' for value in selected_all['mw_pred'])}",
        ),
        loc="left",
        fontweight="bold",
    )
    axes[2].legend(frameon=False, fontsize=8, ncol=2)

    for axis in axes:
        axis.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    axes[1].set_xlabel(translated(language, "Time since origin (s)", "发震后时间（秒）"))
    figure.suptitle(
        translated(
            language,
            f"Noto 2024 representative station {selected_station}: waveform, PGD, and endpoint estimates",
            f"2024 年能登地震代表台站 {selected_station}：波形、PGD 与终点估计",
        ),
        fontsize=14,
        fontweight="bold",
    )
    return save_figure(figure, output_stem)


def load_original_phase39_ensemble(baseline_run_root: Path) -> pd.DataFrame:
    source_path = baseline_run_root / "oof_event_predictions.csv"
    source = pd.read_csv(source_path)
    source = source[source["arm"] == "phase39"].copy()
    seed_coverage = source.groupby("event")["seed"].agg(lambda values: set(values))
    if len(seed_coverage) != 31 or any(value != set(SEEDS) for value in seed_coverage):
        raise ValueError("original Phase 39 OOF seed coverage changed")
    ensemble = (
        source.groupby("event", as_index=False)
        .agg(
            old_mw_catalog=("mw_catalog", "median"),
            old_mw_pred=("mw_pred_median", "median"),
        )
    )
    ensemble["old_absolute_error"] = (
        ensemble["old_mw_pred"] - ensemble["old_mw_catalog"]
    ).abs()
    return ensemble


def comparison_frame(
    events: pd.DataFrame,
    original: pd.DataFrame,
) -> pd.DataFrame:
    common = events.merge(original, on="event", validate="one_to_one")
    if len(common) != 31:
        raise ValueError(f"expected 31 common events, got {len(common)}")
    if float((common["mw_catalog"] - common["old_mw_catalog"]).abs().max()) > 1.0e-5:
        raise ValueError("catalog magnitudes changed for common events")
    common["absolute_error_improvement"] = (
        common["old_absolute_error"] - common["absolute_error"]
    )
    return common


def plot_snapshot_comparison(
    comparison: pd.DataFrame,
    output_stem: Path,
    *,
    language: str,
) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 6.4), constrained_layout=True)
    maximum = float(
        max(comparison["old_absolute_error"].max(), comparison["absolute_error"].max())
        + 0.05
    )
    axes[0].plot([0.0, maximum], [0.0, maximum], color=COLORS["truth"], linewidth=1.2)
    for category, marker, size in (
        ("existing", "o", 55),
        ("noto", "*", 170),
    ):
        rows = comparison[comparison["event_class"] == category]
        axes[0].scatter(
            rows["old_absolute_error"],
            rows["absolute_error"],
            marker=marker,
            s=size,
            color=COLORS[category],
            edgecolor="white",
            linewidth=0.7,
            label=class_label(category, language),
            zorder=3,
        )
    annotations = pd.concat(
        [
            comparison.nlargest(3, "absolute_error_improvement"),
            comparison.nsmallest(3, "absolute_error_improvement"),
        ]
    ).drop_duplicates("event")
    for row in annotations.itertuples(index=False):
        axes[0].annotate(
            str(row.event),
            (float(row.old_absolute_error), float(row.absolute_error)),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set_xlim(0.0, maximum)
    axes[0].set_ylim(0.0, maximum)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel(
        translated(language, "Original 31-event OOF absolute error (Mw)", "原 31 事件 OOF 绝对误差（Mw）")
    )
    axes[0].set_ylabel(
        translated(language, "Expanded-snapshot OOF absolute error (Mw)", "扩展快照 OOF 绝对误差（Mw）")
    )
    axes[0].set_title(
        translated(language, "A. Common-event error comparison", "A. 共同事件误差对比"),
        loc="left",
        fontweight="bold",
    )
    axes[0].legend(frameon=False, fontsize=8)

    ordered = comparison.sort_values("absolute_error_improvement").reset_index(drop=True)
    positions = np.arange(len(ordered))
    colors = [
        COLORS["new"] if value >= 0.0 else COLORS["noto"]
        for value in ordered["absolute_error_improvement"]
    ]
    axes[1].barh(positions, ordered["absolute_error_improvement"], color=colors, alpha=0.9)
    axes[1].axvline(0.0, color=COLORS["truth"], linewidth=1.0)
    axes[1].set_yticks(positions, ordered["event"], fontsize=7.5)
    axes[1].set_xlabel(
        translated(
            language,
            "Absolute-error improvement: original - expanded (Mw)",
            "绝对误差改善：原快照 - 扩展快照（Mw）",
        )
    )
    axes[1].set_title(
        translated(language, "B. Per-event change; positive is better", "B. 各事件变化；正值表示改善"),
        loc="left",
        fontweight="bold",
    )
    axes[1].grid(True, axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.7)

    old_mae = float(comparison["old_absolute_error"].mean())
    new_mae = float(comparison["absolute_error"].mean())
    figure.suptitle(
        translated(
            language,
            f"Original versus expanded Phase 39 snapshots on 31 common events: MAE {old_mae:.3f} -> {new_mae:.3f} Mw",
            f"Phase 39 原快照与扩展快照在 31 个共同事件上的比较：MAE {old_mae:.3f} -> {new_mae:.3f} Mw",
        ),
        fontsize=14,
        fontweight="bold",
    )
    return save_figure(figure, output_stem)


def build_analysis_summary(
    campaign_summary: dict[str, Any],
    events: pd.DataFrame,
    stations: pd.DataFrame,
    run_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    waveform_station: str,
) -> dict[str, Any]:
    new_events = events[events["event_class"] == "new"]
    existing_events = events[events["event_class"] == "existing"]
    noto = events[events["event_class"] == "noto"].iloc[0]
    station_event_summary = (
        stations.assign(absolute_error=lambda frame: frame["error_vs_catalog"].abs())
        .groupby("event", as_index=False)
        .agg(
            station_count=("station", "count"),
            station_mae=("absolute_error", "mean"),
            station_bias=("error_vs_catalog", "mean"),
        )
    )
    return {
        "status": "complete",
        "created_at_utc": utc_now_iso(),
        "campaign_metrics": campaign_summary,
        "event_subgroups": {
            "existing_phase39_excluding_noto": {
                "event_count": int(len(existing_events)),
                "mae_mw": float(existing_events["absolute_error"].mean()),
            },
            "new_events": {
                "event_count": int(len(new_events)),
                "mae_mw": float(new_events["absolute_error"].mean()),
                "within_0_2_fraction": float((new_events["absolute_error"] <= 0.2).mean()),
            },
            "noto2024": {
                "prediction_mw": float(noto["mw_pred_median"]),
                "catalog_mw": float(noto["mw_catalog"]),
                "absolute_error_mw": float(noto["absolute_error"]),
                "seed_std_mw": float(noto["prediction_seed_std"]),
            },
        },
        "worst_events": events.nlargest(10, "absolute_error")[
            ["event", "mw_catalog", "mw_pred_median", "absolute_error", "event_class", "fold"]
        ].to_dict(orient="records"),
        "fold_seed_event_mae": {
            "mean_mw": float(run_metrics["event_mae"].mean()),
            "std_mw": float(run_metrics["event_mae"].std(ddof=0)),
            "min_mw": float(run_metrics["event_mae"].min()),
            "max_mw": float(run_metrics["event_mae"].max()),
        },
        "original_vs_expanded_common_events": {
            "event_count": int(len(comparison)),
            "original_mae_mw": float(comparison["old_absolute_error"].mean()),
            "expanded_mae_mw": float(comparison["absolute_error"].mean()),
            "mean_absolute_error_improvement_mw": float(
                comparison["absolute_error_improvement"].mean()
            ),
            "improved_event_count": int(
                (comparison["absolute_error_improvement"] > 0.0).sum()
            ),
        },
        "waveform_example": f"{WAVEFORM_EVENT}::{waveform_station}",
        "station_event_summary": station_event_summary.to_dict(orient="records"),
    }


def generate_figures(
    *,
    run_root: Path,
    baseline_run_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    (
        campaign_summary,
        protocol,
        events,
        stations,
        _event_all,
        station_all,
        run_summaries,
        training_logs,
        sources_path,
    ) = load_campaign_inputs(run_root)
    events["absolute_error"] = events["error_vs_catalog"].abs()
    stations["absolute_error"] = stations["error_vs_catalog"].abs()
    run_metrics = run_metrics_frame(run_summaries)
    curves = aggregate_training_logs(training_logs)
    original = load_original_phase39_ensemble(baseline_run_root)
    comparison = comparison_frame(events, original)
    config_path = Path(protocol["config_path"]).resolve()
    waveform_sample, waveform_station = load_waveform_sample(config_path, stations)

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(analysis_dir / "event_error_analysis.csv", index=False)
    run_metrics.to_csv(analysis_dir / "fold_seed_metrics.csv", index=False)
    curves.to_csv(analysis_dir / "training_curve_summary.csv", index=False)
    comparison.to_csv(analysis_dir / "original_vs_expanded_common_events.csv", index=False)
    analysis_summary = build_analysis_summary(
        campaign_summary,
        events,
        stations,
        run_metrics,
        comparison,
        waveform_station,
    )
    write_json(analysis_dir / "analysis_summary.json", analysis_summary)

    generated: list[Path] = []
    for language in ("en", "zh"):
        configure_matplotlib(language)
        language_dir = output_dir / language
        generated.extend(plot_event_scatter(events, language_dir / "01_oof_event_scatter", language=language))
        generated.extend(plot_station_hexbin(stations, events, language_dir / "02_oof_station_scatter", language=language))
        generated.extend(plot_event_errors(events, language_dir / "03_event_absolute_errors", language=language))
        generated.extend(plot_fold_seed_metrics(run_metrics, language_dir / "04_fold_seed_metrics", language=language))
        generated.extend(plot_training_curves(curves, language_dir / "05_training_curves", language=language))
        generated.extend(
            plot_selected_maps(
                stations,
                Path(protocol["data_path"]).resolve(),
                language_dir / "06_selected_event_maps",
                language=language,
            )
        )
        generated.extend(
            plot_waveform_example(
                waveform_sample,
                waveform_station,
                stations,
                station_all,
                events,
                language_dir / "07_noto_waveform_and_predictions",
                language=language,
            )
        )
        generated.extend(
            plot_snapshot_comparison(
                comparison,
                language_dir / "08_original_vs_expanded_snapshot",
                language=language,
            )
        )

    manifest = {
        "status": "complete",
        "created_at_utc": utc_now_iso(),
        "run_root": str(run_root),
        "baseline_run_root": str(baseline_run_root),
        "campaign_summary_sha256": sha256_file(run_root / "campaign_summary.json"),
        "event_sources_sha256": sha256_file(sources_path),
        "languages": ["en", "zh"],
        "map_events": list(MAP_EVENTS),
        "waveform_example": f"{WAVEFORM_EVENT}::{waveform_station}",
        "figures": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in generated
        ],
        "analysis_artifacts": [
            str(path)
            for path in sorted(analysis_dir.iterdir())
            if path.is_file()
        ],
    }
    write_json(output_dir / "figure_manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot bilingual final OOF results for Phase 39 expanded grouped CV."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--baseline-run-root",
        type=Path,
        default=DEFAULT_BASELINE_RUN_ROOT,
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.resolve()
    output_dir = (
        args.output_dir.resolve() if args.output_dir else run_root / "figures"
    )
    manifest = generate_figures(
        run_root=run_root,
        baseline_run_root=args.baseline_run_root.resolve(),
        output_dir=output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
