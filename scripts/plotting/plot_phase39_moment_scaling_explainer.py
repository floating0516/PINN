from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path("/home/lihe/PINN_Mag")
RUN_ROOT = WORKSPACE_ROOT / "runs/phase39-causal-direct-moment-scale-20260830-fold0-seed73"
SOURCE_CONFIG = (
    WORKSPACE_ROOT
    / "runs/phase39-confirmatory-grouped-cv-20260812T0332Z-121197d"
    / "fold_0/phase39/seed_73/models/20260812_113942/config.yaml"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "docs/results/phase39-causal-moment-scaling/figures"
)
RESULT_SUMMARY = (
    REPO_ROOT / "docs/results/phase39-causal-moment-scaling/summary.json"
)

VALIDATION_EVENTS = (
    "Anchorage2018",
    "Maule2010",
    "Noto2024",
    "Parkfield2004",
    "RatIslands2014",
    "SandPoint2020",
)
SELECTED_EVENTS = ("Parkfield2004", "Noto2024", "Maule2010")
ANCHOR_HORIZONS = (30, 60, 90, 120, 160, 200)
WAVEFORM_EVENT = "Parkfield2004"
WAVEFORM_STATION = "HOGS"

BASELINE_COLOR = "#4B5563"
CANDIDATE_COLOR = "#087F5B"
TRUTH_COLOR = "#111827"
GRID_COLOR = "#D1D5DB"
EVENT_COLORS = {
    "Anchorage2018": "#2563EB",
    "Maule2010": "#D97706",
    "Noto2024": "#DC2626",
    "Parkfield2004": "#059669",
    "RatIslands2014": "#7C3AED",
    "SandPoint2020": "#0891B2",
}
EVENT_LABELS_EN = {
    "Anchorage2018": "Anchorage",
    "Maule2010": "Maule",
    "Noto2024": "Noto",
    "Parkfield2004": "Parkfield",
    "RatIslands2014": "Rat Islands",
    "SandPoint2020": "Sand Point",
}
EVENT_LABELS_ZH = {
    "Anchorage2018": "安克雷奇",
    "Maule2010": "毛莱",
    "Noto2024": "能登",
    "Parkfield2004": "帕克菲尔德",
    "RatIslands2014": "拉特群岛",
    "SandPoint2020": "桑德角",
}


def translated(language: str, english: str, chinese: str) -> str:
    return chinese if language == "zh" else english


def event_label(event: str, language: str) -> str:
    labels = EVENT_LABELS_ZH if language == "zh" else EVENT_LABELS_EN
    return labels[event]


def configure_matplotlib(language: str) -> None:
    plt.rcParams.update(
        {
            "font.family": (
                "Noto Sans CJK SC" if language == "zh" else "DejaVu Sans"
            ),
            "font.size": 10,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    with SOURCE_CONFIG.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    split = load_json(RUN_ROOT / "split.json")
    event_predictions = pd.read_csv(RUN_ROOT / "validation_event_predictions.csv")
    station_predictions = pd.read_csv(
        RUN_ROOT / "validation_anchor_station_predictions.csv"
    )
    return config, split, event_predictions, station_predictions


def iter_selected_records(
    data: np.lib.npyio.NpzFile,
    event_names: Iterable[str],
) -> Iterable[Any]:
    from src.data import records_v2 as records

    selected = set(event_names)
    events = data["events"]
    magnitudes = data["magnitude"]
    event_lats = data["latitude"]
    event_lons = data["longitude"]
    event_count = len(events)
    depths = data["depth_km"] if "depth_km" in data else np.full(event_count, np.nan)
    strikes = data["strike"] if "strike" in data else np.full(event_count, np.nan)
    dips = data["dip"] if "dip" in data else np.full(event_count, np.nan)
    rakes = data["rake"] if "rake" in data else np.full(event_count, np.nan)
    event_containers = data["enu"]
    station_metadata = data["station_info"]

    for event_index, event_value in enumerate(events):
        event = records._as_text(event_value)
        if event not in selected:
            continue
        station_items = records._iter_stations_container(event_containers[event_index])
        metadata_map = records._normalize_station_info(station_metadata[event_index])
        mechanism = records._event_mechanism_code(data, event_index)
        for station_name, payload in station_items:
            metadata = metadata_map.get(station_name, {})
            station_lat = payload.get(
                "lat",
                payload.get(
                    "latitude",
                    metadata.get("lat", metadata.get("latitude", np.nan)),
                ),
            )
            station_lon = payload.get(
                "lon",
                payload.get(
                    "longitude",
                    metadata.get("lon", metadata.get("longitude", np.nan)),
                ),
            )
            time_sec = records._get_field(payload, ("t", "time"))
            east = records._get_field(payload, ("E", "east"))
            north = records._get_field(payload, ("N", "north"))
            vertical = records._get_field(payload, ("U", "up", "vertical"))
            origin_sec = records._get_field(
                payload,
                (
                    "origin",
                    "origin_s",
                    "origin_time",
                    "origin_epoch",
                    "origin_ts",
                    "t0",
                    "origin_sec",
                ),
            )
            if any(value is None for value in (time_sec, east, north, vertical)):
                continue
            yield records.NormalizedStationRecord(
                event_index=int(event_index),
                event=event,
                magnitude_catalog=records._as_float(magnitudes[event_index]),
                event_lat=records._as_float(event_lats[event_index]),
                event_lon=records._as_float(event_lons[event_index]),
                depth_km=records._as_float(depths[event_index]),
                strike=records._as_float(strikes[event_index]),
                dip=records._as_float(dips[event_index]),
                rake=records._as_float(rakes[event_index]),
                mechanism=mechanism,
                station=station_name,
                station_lat=records._as_float(station_lat),
                station_lon=records._as_float(station_lon),
                time_sec=np.asarray(time_sec),
                east=np.asarray(east),
                north=np.asarray(north),
                vertical=np.asarray(vertical),
                origin_sec=records._optional_float(origin_sec),
            )


def load_validation_records(
    config: dict[str, Any],
    split: dict[str, Any],
) -> dict[str, Any]:
    validation_keys = set(split["sample_keys"]["validation"])
    validation_events = set(split["validation_events"])
    if validation_events != set(VALIDATION_EVENTS):
        raise ValueError("unexpected validation event assignment")
    records_by_key: dict[str, Any] = {}
    with np.load(config["paths"]["data_path"], allow_pickle=True) as data:
        for record in iter_selected_records(data, validation_events):
            key = f"{record.event}::{record.station}"
            if key in validation_keys:
                records_by_key[key] = record
    if set(records_by_key) != validation_keys:
        missing = sorted(validation_keys - set(records_by_key))
        raise ValueError(f"validation coordinate join failed for {missing[:5]}")
    return records_by_key


def coordinate_frame(records_by_key: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, record in sorted(records_by_key.items()):
        row = asdict(record)
        rows.append(
            {
                "sample_key": key,
                "event": row["event"],
                "station": row["station"],
                "event_lon": row["event_lon"],
                "event_lat": row["event_lat"],
                "depth_km": row["depth_km"],
                "magnitude_catalog": row["magnitude_catalog"],
                "station_lon": row["station_lon"],
                "station_lat": row["station_lat"],
            }
        )
    return pd.DataFrame(rows)


def validate_prediction_tables(
    split: dict[str, Any],
    event_predictions: pd.DataFrame,
    station_predictions: pd.DataFrame,
    coordinates: pd.DataFrame,
) -> None:
    validation_events = set(split["validation_events"])
    if set(event_predictions["event"].unique()) != validation_events:
        raise ValueError("event predictions contain a non-validation event")
    if set(station_predictions["event"].unique()) != validation_events:
        raise ValueError("station predictions contain a non-validation event")
    if len(coordinates) != int(split["validation_record_count"]):
        raise ValueError("coordinate count does not match validation split")
    endpoint_station = station_predictions[
        station_predictions["observation_horizon_sec"] == 200
    ]
    endpoint_event = event_predictions[
        event_predictions["observation_horizon_sec"] == 200
    ]
    for method in ("phase39", "direct"):
        if len(endpoint_station[endpoint_station["method"] == method]) != 424:
            raise ValueError(f"unexpected endpoint station count for {method}")
        if len(endpoint_event[endpoint_event["method"] == method]) != 6:
            raise ValueError(f"unexpected endpoint event count for {method}")

    direct = endpoint_event[endpoint_event["method"] == "direct"]
    baseline = endpoint_event[endpoint_event["method"] == "phase39"]
    direct_mae = float(direct["error_vs_catalog"].abs().mean())
    baseline_mae = float(baseline["error_vs_catalog"].abs().mean())
    if not math.isclose(direct_mae, 0.13783550262451172, abs_tol=1.0e-9):
        raise ValueError("direct endpoint MAE does not match the published run")
    if not math.isclose(baseline_mae, 0.23807891209920248, abs_tol=1.0e-9):
        raise ValueError("Phase 39 endpoint MAE does not match the published run")


def identity_limits(*series: pd.Series, padding: float = 0.15) -> tuple[float, float]:
    values = np.concatenate([item.to_numpy(dtype=float) for item in series])
    lower = float(np.nanmin(values)) - padding
    upper = float(np.nanmax(values)) + padding
    return lower, upper


def station_jitter(frame: pd.DataFrame, width: float = 0.055) -> pd.Series:
    jitter = pd.Series(index=frame.index, dtype=float)
    for event, rows in frame.groupby("event"):
        ordered = rows.sort_values("station")
        offsets = np.linspace(-width, width, len(ordered))
        jitter.loc[ordered.index] = offsets
    return jitter


def plot_result_overview(
    summary: dict[str, Any],
    event_predictions: pd.DataFrame,
    output_path: Path,
    *,
    language: str,
) -> None:
    metrics = pd.read_csv(RUN_ROOT / "validation_horizon_metrics.csv")
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)

    for axis, minimum_horizon, title_en, title_zh in (
        (
            axes[0, 0],
            1,
            "A. Second-by-second causal-prefix error",
            "A. 逐秒因果前缀误差",
        ),
        (
            axes[0, 1],
            120,
            "B. Late-stage convergence (120-200 s)",
            "B. 后期收敛（120-200 秒）",
        ),
    ):
        filtered = metrics[metrics["observation_horizon_sec"] >= minimum_horizon]
        for method, label_en, label_zh, color, linestyle in (
            ("phase39", "Phase 39", "Phase 39", BASELINE_COLOR, "--"),
            (
                "direct",
                "Full method",
                "完整方法",
                CANDIDATE_COLOR,
                "-",
            ),
        ):
            rows = filtered[filtered["method"] == method].sort_values(
                "observation_horizon_sec"
            )
            axis.plot(
                rows["observation_horizon_sec"],
                rows["event_mae"],
                label=translated(language, label_en, label_zh),
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
            )
        axis.axhline(0.20, color="#C2410C", linewidth=1.2, linestyle=":")
        axis.axhline(0.17, color="#C2410C", linewidth=1.2, linestyle="-.")
        axis.set_title(translated(language, title_en, title_zh))
        axis.set_xlabel(
            translated(language, "Observed causal prefix (s)", "已观测因果前缀（秒）")
        )
        axis.set_ylabel(
            translated(language, "Event-level MAE (Mw)", "事件级 MAE（Mw）")
        )
        axis.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.7)
        axis.legend(frameon=False, fontsize=8)

    runs = summary["runs"]
    labels = [f"F{run['fold']} / S{run['seed']}" for run in runs]
    baseline_values = [run["phase39_endpoint_event_mae_mw"] for run in runs]
    candidate_values = [run["candidate_endpoint_event_mae_mw"] for run in runs]
    positions = np.arange(len(runs))
    width = 0.34
    baseline_bars = axes[1, 0].bar(
        positions - width / 2,
        baseline_values,
        width,
        color=BASELINE_COLOR,
        label="Phase 39",
    )
    candidate_bars = axes[1, 0].bar(
        positions + width / 2,
        candidate_values,
        width,
        color=CANDIDATE_COLOR,
        label=translated(language, "Full method", "完整方法"),
    )
    axes[1, 0].axhline(0.20, color="#C2410C", linewidth=1.2, linestyle=":")
    axes[1, 0].axhline(0.17, color="#C2410C", linewidth=1.2, linestyle="-.")
    axes[1, 0].bar_label(baseline_bars, fmt="%.3f", padding=3, fontsize=8)
    axes[1, 0].bar_label(candidate_bars, fmt="%.3f", padding=3, fontsize=8)
    axes[1, 0].set_xticks(positions, labels)
    axes[1, 0].set_ylim(0.0, max(baseline_values) + 0.075)
    axes[1, 0].set_ylabel(
        translated(language, "Endpoint event MAE (Mw)", "终点事件 MAE（Mw）")
    )
    axes[1, 0].set_title(
        translated(
            language,
            "C. Endpoint results across validation runs",
            "C. 多次验证运行的终点结果",
        )
    )
    axes[1, 0].grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    axes[1, 0].legend(frameon=False, fontsize=8)

    endpoint = event_predictions[
        event_predictions["observation_horizon_sec"] == 200
    ].copy()
    endpoint["absolute_error"] = endpoint["error_vs_catalog"].abs()
    baseline = endpoint[endpoint["method"] == "phase39"].set_index("event")
    candidate = endpoint[endpoint["method"] == "direct"].set_index("event")
    events = baseline.sort_values("absolute_error", ascending=True).index.tolist()
    event_positions = np.arange(len(events))
    height = 0.34
    axes[1, 1].barh(
        event_positions + height / 2,
        baseline.loc[events, "absolute_error"],
        height,
        color=BASELINE_COLOR,
        label="Phase 39",
    )
    axes[1, 1].barh(
        event_positions - height / 2,
        candidate.loc[events, "absolute_error"],
        height,
        color=CANDIDATE_COLOR,
        label=translated(language, "Full method", "完整方法"),
    )
    axes[1, 1].axvline(0.20, color="#C2410C", linewidth=1.2, linestyle=":")
    axes[1, 1].set_yticks(
        event_positions,
        [event_label(event, language) for event in events],
    )
    axes[1, 1].set_xlabel(
        translated(language, "Absolute endpoint error (Mw)", "终点绝对误差（Mw）")
    )
    axes[1, 1].set_title(
        translated(
            language,
            "D. Per-event endpoint errors (Fold 0 / Seed 73)",
            "D. 各事件终点误差（Fold 0 / Seed 73）",
        )
    )
    axes[1, 1].grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    axes[1, 1].legend(frameon=False, fontsize=8)

    figure.suptitle(
        translated(
            language,
            "Phase 39 causal moment-scaling screen: internal validation only",
            "Phase 39 因果震矩缩放筛选：仅内部验证",
        ),
        fontsize=15,
        fontweight="bold",
    )
    save_figure(figure, output_path)


def plot_method_workflow(output_path: Path, *, language: str) -> None:
    figure, axis = plt.subplots(figsize=(16.0, 7.0), constrained_layout=True)
    axis.set_xlim(0.0, 16.0)
    axis.set_ylim(0.0, 8.0)
    axis.axis("off")

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        text_value: str,
        *,
        facecolor: str,
        edgecolor: str = "#475569",
        fontsize: float = 10.5,
    ) -> tuple[float, float]:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.2,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2.0,
            y + height / 2.0,
            text_value,
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        return x + width / 2.0, y + height / 2.0

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.3,
                color="#475569",
                shrinkA=3,
                shrinkB=3,
            )
        )

    axis.text(
        8.0,
        7.65,
        translated(
            language,
            "Phase 39 causal training with physics-guided counterfactual moment scaling",
            "Phase 39：物理约束反事实震矩缩放与因果前缀训练",
        ),
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    original = box(
        0.25,
        5.25,
        3.0,
        1.45,
        translated(
            language,
            "Observed sample\nradial waveform, STF, Mw, geometry",
            "观测样本\n径向波形、STF、Mw、几何",
        ),
        facecolor="#E2E8F0",
    )
    scaling = box(
        3.65,
        5.25,
        4.0,
        1.45,
        translated(
            language,
            "Moment-scaling transform\n$\\Delta Mw\\in[-0.75,0.5]$\n$a=10^{1.5\\Delta Mw}$",
            "震矩缩放变换\n$\\Delta Mw\\in[-0.75,0.5]$\n$a=10^{1.5\\Delta Mw}$",
        ),
        facecolor="#FFEDD5",
        edgecolor="#C2410C",
    )
    paired = box(
        8.05,
        5.25,
        3.3,
        1.45,
        translated(
            language,
            "Paired training samples\noriginal + scaled counterfactual",
            "配对训练样本\n原始样本 + 缩放反事实样本",
        ),
        facecolor="#DCFCE7",
        edgecolor="#087F5B",
    )
    horizons = box(
        11.75,
        5.25,
        4.0,
        1.45,
        translated(
            language,
            "Two views per batch\nrandom prefix $h\\in[5,199]$ s\nfull endpoint 200 s",
            "每批次两种视图\n随机前缀 $h\\in[5,199]$ 秒\n完整终点 200 秒",
        ),
        facecolor="#E0F2FE",
        edgecolor="#0369A1",
    )
    arrow((3.25, 5.975), (3.65, 5.975))
    arrow((7.65, 5.975), (8.05, 5.975))
    arrow((11.35, 5.975), (11.75, 5.975))

    model = box(
        0.75,
        2.65,
        4.3,
        1.55,
        translated(
            language,
            "Unchanged Phase 39 backbone\nR-only TCN + SE + Transformer\n1,010,850 parameters",
            "不变的 Phase 39 主干\nR-only TCN + SE + Transformer\n1,010,850 个参数",
        ),
        facecolor="#EDE9FE",
        edgecolor="#6D28D9",
    )
    stf = box(
        5.7,
        2.65,
        4.0,
        1.55,
        translated(
            language,
            "Fixed 200 s nonnegative STF\n$\\dot M_0(t)=M_0p(t)$",
            "固定 200 秒非负 STF\n$\\dot M_0(t)=M_0p(t)$",
        ),
        facecolor="#F1F5F9",
    )
    magnitude = box(
        10.35,
        2.65,
        4.9,
        1.55,
        translated(
            language,
            "Single magnitude path\n$M_0=\\int\\dot M_0(t)dt$\n$Mw=\\frac{2}{3}(\\log_{10}M_0-9.1)$",
            "唯一震级路径\n$M_0=\\int\\dot M_0(t)dt$\n$Mw=\\frac{2}{3}(\\log_{10}M_0-9.1)$",
        ),
        facecolor="#F1F5F9",
    )
    arrow((13.75, 5.25), (2.9, 4.2))
    arrow((5.05, 3.425), (5.7, 3.425))
    arrow((9.7, 3.425), (10.35, 3.425))

    science = box(
        0.55,
        0.25,
        5.25,
        1.45,
        translated(
            language,
            "Science loss at prefix and endpoint\n$L_{phys}=L_{STF}+L_{mag}+0.5L_{synth}$",
            "前缀与终点科学损失\n$L_{phys}=L_{STF}+L_{mag}+0.5L_{synth}$",
        ),
        facecolor="#DCFCE7",
        edgecolor="#087F5B",
    )
    descent = box(
        6.25,
        0.25,
        4.2,
        1.45,
        translated(
            language,
            "Soft error-descent constraint\nendpoint may exceed prefix error\nby at most 0.03 Mw without penalty",
            "软误差下降约束\n终点误差比前缀误差最多高 0.03 Mw\n时不施加惩罚",
        ),
        facecolor="#FEF3C7",
        edgecolor="#A16207",
        fontsize=9.7,
    )
    total = box(
        10.9,
        0.25,
        4.65,
        1.45,
        translated(
            language,
            "Total objective\n$L=0.5L_{phys}^{h}+L_{phys}^{200}+0.5L_{descent}$",
            "总目标函数\n$L=0.5L_{phys}^{h}+L_{phys}^{200}+0.5L_{descent}$",
        ),
        facecolor="#E0F2FE",
        edgecolor="#0369A1",
    )
    arrow((7.7, 2.65), (3.175, 1.7))
    arrow((12.8, 2.65), (8.35, 1.7))
    axis.text(6.02, 0.975, "+", ha="center", va="center", fontsize=22)
    arrow((10.45, 0.975), (10.9, 0.975))

    axis.text(
        8.0,
        0.02,
        translated(
            language,
            "The scaled pair is not a new earthquake: geometry, timing, and masks are unchanged.",
            "缩放样本不是新的独立地震：几何、到时和有效掩码均保持不变。",
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#475569",
    )
    save_figure(figure, output_path)


def plot_prediction_scatter(
    event_predictions: pd.DataFrame,
    station_predictions: pd.DataFrame,
    output_path: Path,
    *,
    language: str,
) -> None:
    endpoint_events = event_predictions[
        event_predictions["observation_horizon_sec"] == 200
    ].copy()
    endpoint_stations = station_predictions[
        station_predictions["observation_horizon_sec"] == 200
    ].copy()
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), constrained_layout=True)

    event_axis = axes[0]
    lower, upper = identity_limits(
        endpoint_events["mw_catalog"], endpoint_events["mw_pred_median"]
    )
    event_axis.plot([lower, upper], [lower, upper], color=TRUTH_COLOR, linewidth=1.2)
    baseline = endpoint_events[endpoint_events["method"] == "phase39"].set_index(
        "event"
    )
    candidate = endpoint_events[endpoint_events["method"] == "direct"].set_index(
        "event"
    )
    for event in VALIDATION_EVENTS:
        true_mw = float(candidate.loc[event, "mw_catalog"])
        base_mw = float(baseline.loc[event, "mw_pred_median"])
        candidate_mw = float(candidate.loc[event, "mw_pred_median"])
        event_axis.plot(
            [true_mw, true_mw],
            [base_mw, candidate_mw],
            color=EVENT_COLORS[event],
            linewidth=1.0,
            alpha=0.55,
        )
        event_axis.scatter(
            true_mw,
            base_mw,
            s=42,
            color=BASELINE_COLOR,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        event_axis.scatter(
            true_mw,
            candidate_mw,
            s=58,
            marker="D",
            color=EVENT_COLORS[event],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        offset = (5, 5)
        if event == "Noto2024":
            offset = (5, -13)
        elif event == "Anchorage2018":
            offset = (-48, 5)
        event_axis.annotate(
            event_label(event, language),
            (true_mw, candidate_mw),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    event_axis.set_xlim(lower, upper)
    event_axis.set_ylim(lower, upper)
    event_axis.set_aspect("equal", adjustable="box")
    event_axis.set_title(
        translated(language, "A. Event-median endpoint estimates", "A. 事件中位数终点估计")
    )
    event_axis.set_xlabel(translated(language, "Catalog magnitude (Mw)", "目录震级（Mw）"))
    event_axis.set_ylabel(translated(language, "Estimated magnitude (Mw)", "估计震级（Mw）"))
    event_axis.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.75)
    event_axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=BASELINE_COLOR, label="Phase 39"),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="",
                color=CANDIDATE_COLOR,
                label=translated(language, "Full method", "完整方法"),
            ),
            Line2D(
                [],
                [],
                color=TRUTH_COLOR,
                label=translated(language, "Ideal", "理想值"),
            ),
        ],
        frameon=False,
        loc="upper left",
    )

    station_lower, station_upper = identity_limits(
        endpoint_stations["mw_catalog"], endpoint_stations["mw_pred"]
    )
    for axis, method, title in (
        (
            axes[1],
            "phase39",
            translated(language, "B. Phase 39 station estimates", "B. Phase 39 台站估计"),
        ),
        (
            axes[2],
            "direct",
            translated(language, "C. Full-method station estimates", "C. 完整方法台站估计"),
        ),
    ):
        method_rows = endpoint_stations[endpoint_stations["method"] == method].copy()
        method_rows["jitter"] = station_jitter(method_rows)
        axis.plot(
            [station_lower, station_upper],
            [station_lower, station_upper],
            color=TRUTH_COLOR,
            linewidth=1.2,
        )
        for event in VALIDATION_EVENTS:
            rows = method_rows[method_rows["event"] == event]
            axis.scatter(
                rows["mw_catalog"] + rows["jitter"],
                rows["mw_pred"],
                s=13,
                alpha=0.42,
                color=EVENT_COLORS[event],
                edgecolor="none",
                rasterized=True,
            )
        station_mae = float(
            (method_rows["mw_pred"] - method_rows["mw_catalog"]).abs().mean()
        )
        axis.set_xlim(station_lower, station_upper)
        axis.set_ylim(station_lower, station_upper)
        axis.set_aspect("equal", adjustable="box")
        mae_label = translated(language, "Station MAE", "台站 MAE")
        axis.set_title(f"{title}\n{mae_label} = {station_mae:.3f} Mw")
        axis.set_xlabel(translated(language, "Catalog magnitude (Mw)", "目录震级（Mw）"))
        axis.set_ylabel(translated(language, "Station estimate (Mw)", "台站估计（Mw）"))
        axis.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.75)

    event_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=EVENT_COLORS[event],
            label=event_label(event, language),
        )
        for event in VALIDATION_EVENTS
    ]
    figure.legend(
        handles=event_handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    figure.suptitle(
        translated(
            language,
            "True versus estimated magnitude at 200 s (internal validation)\n"
            "Primary evaluation uses equal-weight event medians; station panels are station-weighted",
            "200 秒真实震级与估计震级（内部验证）\n"
            "主指标为事件等权中位数；台站面板按台站加权",
        ),
        fontsize=15,
        fontweight="bold",
    )
    save_figure(figure, output_path)


def plot_station_convergence(
    station_predictions: pd.DataFrame,
    output_path: Path,
    *,
    language: str,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)
    for axis, event in zip(axes, SELECTED_EVENTS):
        rows = station_predictions[station_predictions["event"] == event]
        candidate = rows[rows["method"] == "direct"]
        baseline = rows[rows["method"] == "phase39"]
        station_count = int(candidate[candidate["observation_horizon_sec"] == 200].shape[0])
        truth = float(candidate["mw_catalog"].iloc[0])
        alpha = 0.32 if station_count < 30 else 0.08
        point_size = 15 if station_count < 30 else 7
        for horizon in ANCHOR_HORIZONS:
            horizon_rows = candidate[
                candidate["observation_horizon_sec"] == horizon
            ].sort_values("station")
            offsets = np.linspace(-2.3, 2.3, len(horizon_rows))
            axis.scatter(
                horizon + offsets,
                horizon_rows["mw_pred"],
                s=point_size,
                alpha=alpha,
                color=EVENT_COLORS[event],
                edgecolor="none",
                rasterized=True,
            )
        candidate_median = candidate.groupby("observation_horizon_sec")["mw_pred"].median()
        baseline_median = baseline.groupby("observation_horizon_sec")["mw_pred"].median()
        axis.plot(
            candidate_median.index,
            candidate_median.values,
            color=CANDIDATE_COLOR,
            marker="o",
            linewidth=2.2,
            label=translated(language, "Full-method median", "完整方法中位数"),
        )
        axis.plot(
            baseline_median.index,
            baseline_median.values,
            color=BASELINE_COLOR,
            linestyle="--",
            marker="s",
            markersize=4,
            linewidth=1.7,
            label=translated(language, "Phase 39 median", "Phase 39 中位数"),
        )
        axis.axhline(
            truth,
            color=TRUTH_COLOR,
            linewidth=1.4,
            label=translated(language, "Catalog Mw", "目录 Mw"),
        )
        axis.set_xticks(ANCHOR_HORIZONS)
        axis.set_xlabel(
            translated(language, "Observed causal prefix (s)", "已观测因果前缀（秒）")
        )
        axis.set_ylabel(
            translated(language, "Station magnitude estimate (Mw)", "台站震级估计（Mw）")
        )
        station_word = translated(language, "stations", "个台站")
        axis.set_title(
            f"{event_label(event, language)} | Mw {truth:.2f} | "
            f"{station_count} {station_word}"
        )
        axis.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.75)
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    figure.suptitle(
        translated(
            language,
            "Station-level estimates across causal horizons",
            "不同因果时长下的台站级估计",
        ),
        fontsize=15,
        fontweight="bold",
    )
    save_figure(figure, output_path)


def build_waveform_sample(
    config: dict[str, Any],
    records_by_key: dict[str, Any],
) -> dict[str, Any]:
    from src.data.sample_builder import build_station_sample
    from src.data.waveform import waveform_config_from_v2

    key = f"{WAVEFORM_EVENT}::{WAVEFORM_STATION}"
    record = records_by_key[key]
    return build_station_sample(
        record,
        units=str(config["dataset"]["units"]),
        waveform_config=waveform_config_from_v2(config),
        alpha_m_per_s=float(config["physics"]["alpha"]),
        radial_peak_min_cm=float(config["dataset"]["radial_peak_min_cm"]),
    )


def plot_pgd_and_magnitude(
    waveform_sample: dict[str, Any],
    event_predictions: pd.DataFrame,
    output_path: Path,
    *,
    language: str,
) -> None:
    dt_sec = float(waveform_sample["waveform_dt_sec"])
    radial_cm = np.asarray(waveform_sample["radial"], dtype=float) * 100.0
    tangential_cm = np.asarray(waveform_sample["tangential"], dtype=float) * 100.0
    vertical_cm = np.asarray(waveform_sample["vertical"], dtype=float) * 100.0
    time_sec = np.arange(radial_cm.size, dtype=float) * dt_sec
    displacement_norm_cm = np.sqrt(
        radial_cm**2 + tangential_cm**2 + vertical_cm**2
    )
    cumulative_pgd_cm = np.maximum.accumulate(displacement_norm_cm)

    event_rows = event_predictions[event_predictions["event"] == WAVEFORM_EVENT]
    truth = float(event_rows["mw_catalog"].iloc[0])
    figure, axes = plt.subplots(3, 1, figsize=(12.5, 9.2), sharex=True, constrained_layout=True)

    axes[0].plot(time_sec, radial_cm, color="#2563EB", linewidth=1.25)
    axes[0].axhline(0.0, color=GRID_COLOR, linewidth=0.8)
    axes[0].set_ylabel(translated(language, "Radial displacement (cm)", "径向位移（cm）"))
    axes[0].set_title(
        translated(
            language,
            "A. Observed radial displacement used by the R-only model",
            "A. R-only 模型使用的观测径向位移",
        )
    )
    axes[0].grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.7)

    axes[1].plot(
        time_sec,
        displacement_norm_cm,
        color="#94A3B8",
        linewidth=1.0,
        label=translated(language, "3-component displacement norm", "三分量位移模长"),
    )
    axes[1].plot(
        time_sec,
        cumulative_pgd_cm,
        color=CANDIDATE_COLOR,
        linewidth=2.2,
        label=translated(language, "Cumulative PGD", "累积 PGD"),
    )
    axes[1].set_ylabel(translated(language, "Displacement / PGD (cm)", "位移 / PGD（cm）"))
    axes[1].set_title(
        translated(language, "B. Three-component peak ground displacement", "B. 三分量峰值地表位移")
    )
    axes[1].grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    axes[1].legend(frameon=False, loc="lower right")

    for method, label, color, linestyle in (
        ("phase39", "Phase 39", BASELINE_COLOR, "--"),
        (
            "direct",
            translated(language, "Full method", "完整方法"),
            CANDIDATE_COLOR,
            "-",
        ),
    ):
        rows = event_rows[event_rows["method"] == method].sort_values(
            "observation_horizon_sec"
        )
        axes[2].plot(
            rows["observation_horizon_sec"],
            rows["mw_pred_median"],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
    axes[2].axhspan(
        truth - 0.20,
        truth + 0.20,
        color="#FDE68A",
        alpha=0.35,
        label=translated(language, "Catalog +/-0.20 Mw", "目录值 +/-0.20 Mw"),
    )
    axes[2].axhline(
        truth,
        color=TRUTH_COLOR,
        linewidth=1.5,
        label=translated(language, f"Catalog Mw {truth:.2f}", f"目录 Mw {truth:.2f}"),
    )
    axes[2].set_xlabel(
        translated(
            language,
            "Time since origin / observed causal prefix (s)",
            "发震后时间 / 已观测因果前缀（秒）",
        )
    )
    axes[2].set_ylabel(
        translated(language, "Event-median estimate (Mw)", "事件中位数估计（Mw）")
    )
    axes[2].set_title(
        translated(language, "C. Second-by-second causal magnitude estimate", "C. 逐秒因果震级估计")
    )
    axes[2].grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    axes[2].legend(frameon=False, loc="upper right", ncol=2)
    axes[2].set_xlim(0, 200)

    figure.suptitle(
        translated(
            language,
            "Parkfield 2004: waveform, PGD, and causal magnitude evolution\n"
            f"Station {WAVEFORM_STATION} | epicentral distance "
            f"{float(waveform_sample['epicentral_distance_m']) / 1000.0:.1f} km | "
            f"final PGD {float(cumulative_pgd_cm[-1]):.2f} cm",
            "2004 年帕克菲尔德地震：波形、PGD 与因果震级演化\n"
            f"台站 {WAVEFORM_STATION} | 震中距 "
            f"{float(waveform_sample['epicentral_distance_m']) / 1000.0:.1f} km | "
            f"最终 PGD {float(cumulative_pgd_cm[-1]):.2f} cm",
        ),
        fontsize=15,
        fontweight="bold",
    )
    save_figure(figure, output_path)


def map_region(frame: pd.DataFrame) -> tuple[float, float, float, float]:
    event = str(frame["event"].iloc[0])
    if event == "Parkfield2004":
        return (-121.0, -120.1, 35.45, 36.20)
    lons = np.concatenate(
        [frame["station_lon"].to_numpy(), frame["event_lon"].to_numpy()]
    )
    lats = np.concatenate(
        [frame["station_lat"].to_numpy(), frame["event_lat"].to_numpy()]
    )
    lon_span = max(float(np.max(lons) - np.min(lons)), 0.1)
    lat_span = max(float(np.max(lats) - np.min(lats)), 0.1)
    lon_pad = max(0.08, lon_span * 0.10)
    lat_pad = max(0.08, lat_span * 0.10)
    return (
        float(np.min(lons) - lon_pad),
        float(np.max(lons) + lon_pad),
        float(np.min(lats) - lat_pad),
        float(np.max(lats) + lat_pad),
    )


def gmt_coastline_segments(
    region: tuple[float, float, float, float],
) -> list[np.ndarray]:
    west, east, south, north = region
    with tempfile.TemporaryDirectory(prefix="phase39-gmt-coast-") as temp_name:
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
        if len(tokens) < 2:
            continue
        current.append((float(tokens[0]), float(tokens[1])))
    if len(current) >= 2:
        segments.append(np.asarray(current, dtype=float))
    return segments


def plot_event_map(
    event: str,
    map_frame: pd.DataFrame,
    output_dir: Path,
    *,
    language: str,
) -> Path:
    if subprocess.run(
        ["gmt", "--version"], capture_output=True, text=True, check=False
    ).returncode != 0:
        raise RuntimeError("GMT is required for the selected-event maps")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"04_{event.lower()}_station_map"
    output_path = output_dir / f"{output_stem}.png"
    region = map_region(map_frame)
    west, east, south, north = region
    event_lon = float(map_frame["event_lon"].iloc[0])
    event_lat = float(map_frame["event_lat"].iloc[0])
    catalog_mw = float(map_frame["mw_catalog"].iloc[0])
    station_count = len(map_frame)
    coastlines = gmt_coastline_segments(region)

    figure, axis = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    axis.set_facecolor("#EAF4FB")
    for segment in coastlines:
        axis.plot(segment[:, 0], segment[:, 1], color="#475569", linewidth=0.8)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    station_size = 48 if station_count < 20 else 24 if station_count < 50 else 13
    station_marks = axis.scatter(
        map_frame["station_lon"],
        map_frame["station_lat"],
        c=map_frame["residual_mw"].clip(-1.0, 1.0),
        cmap="coolwarm",
        norm=norm,
        s=station_size,
        edgecolor="#334155",
        linewidth=0.35,
        alpha=0.88,
        zorder=3,
        label=translated(language, "GNSS stations", "GNSS 台站"),
        rasterized=True,
    )
    axis.scatter(
        [event_lon],
        [event_lat],
        marker="*",
        s=230,
        color="#FACC15",
        edgecolor="#111827",
        linewidth=0.9,
        zorder=4,
        label=translated(language, "Epicenter", "震中"),
    )
    if event == "Parkfield2004":
        for row in map_frame.itertuples(index=False):
            axis.annotate(
                str(row.station),
                (float(row.station_lon), float(row.station_lat)),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6.5,
                color="#1F2937",
            )
    colorbar = figure.colorbar(station_marks, ax=axis, orientation="horizontal", pad=0.09)
    colorbar.set_label(
        translated(
            language,
            "Station Mw residual (estimate - catalog); clipped at +/-1 Mw",
            "台站 Mw 残差（估计值 - 目录值）；截断至 +/-1 Mw",
        )
    )
    axis.set_xlim(west, east)
    axis.set_ylim(south, north)
    mean_latitude = 0.5 * (south + north)
    axis.set_aspect(1.0 / max(math.cos(math.radians(mean_latitude)), 0.2))
    axis.set_xlabel(translated(language, "Longitude (deg)", "经度（度）"))
    axis.set_ylabel(translated(language, "Latitude (deg)", "纬度（度）"))
    station_word = translated(language, "validation stations", "个验证台站")
    axis.set_title(
        f"{event_label(event, language)} | Mw {catalog_mw:.2f} | "
        f"{station_count} {station_word}"
    )
    axis.grid(True, color="#CBD5E1", linewidth=0.55, alpha=0.8)
    axis.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="best")
    save_figure(figure, output_path)
    return output_path


def plot_selected_event_maps(
    coordinates: pd.DataFrame,
    station_predictions: pd.DataFrame,
    output_dir: Path,
    *,
    language: str,
) -> list[Path]:
    endpoint = station_predictions[
        (station_predictions["method"] == "direct")
        & (station_predictions["observation_horizon_sec"] == 200)
    ].copy()
    endpoint["sample_key"] = endpoint["event"] + "::" + endpoint["station"].astype(str)
    endpoint["residual_mw"] = endpoint["mw_pred"] - endpoint["mw_catalog"]
    joined = endpoint.merge(coordinates, on=["sample_key", "event", "station"], how="inner")
    if len(joined) != 424:
        raise ValueError("endpoint prediction and coordinate join must contain 424 stations")
    map_paths = []
    for event in SELECTED_EVENTS:
        map_paths.append(
            plot_event_map(
                event,
                joined[joined["event"] == event],
                output_dir,
                language=language,
            )
        )
    return map_paths


def combine_map_images(
    map_paths: list[Path],
    output_path: Path,
    *,
    language: str,
) -> None:
    figure, axes = plt.subplots(1, len(map_paths), figsize=(18, 5.4), constrained_layout=True)
    for axis, path in zip(axes, map_paths):
        axis.imshow(plt.imread(path))
        axis.axis("off")
    figure.suptitle(
        translated(
            language,
            "Epicenters and validation-station distributions",
            "震中与验证台站分布",
        ),
        fontsize=15,
        fontweight="bold",
    )
    save_figure(figure, output_path)


def write_manifest(
    output_dir: Path,
    generated_paths: list[Path],
    split: dict[str, Any],
    *,
    language: str,
) -> None:
    payload = {
        "status": "complete",
        "figure_language": "Chinese" if language == "zh" else "English",
        "role": "internal_validation_explainer",
        "fold": 0,
        "seed": 73,
        "validation_events": list(split["validation_events"]),
        "validation_station_count": int(split["validation_record_count"]),
        "selected_events": list(SELECTED_EVENTS),
        "waveform_sample": f"{WAVEFORM_EVENT}::{WAVEFORM_STATION}",
        "test_split_scored": False,
        "external_events_loaded": False,
        "figures": [path.name for path in generated_paths],
    }
    with (output_dir / "figure_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot bilingual explanatory figures for the Phase 39 moment-scaling screen."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG outputs.",
    )
    parser.add_argument(
        "--language",
        choices=("en", "zh", "both"),
        default="both",
        help="Generate English figures, Chinese figures, or both.",
    )
    return parser.parse_args()


def generate_language_figures(
    *,
    language: str,
    output_dir: Path,
    summary: dict[str, Any],
    config: dict[str, Any],
    split: dict[str, Any],
    event_predictions: pd.DataFrame,
    station_predictions: pd.DataFrame,
    records_by_key: dict[str, Any],
    coordinates: pd.DataFrame,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(language)

    workflow_path = output_dir / "00_method_workflow.png"
    overview_path = output_dir / "result_overview.png"
    scatter_path = output_dir / "01_prediction_scatter.png"
    convergence_path = output_dir / "02_station_convergence_scatter.png"
    pgd_path = output_dir / "03_parkfield_pgd_and_mw.png"
    map_composite_path = output_dir / "04_selected_event_maps.png"
    plot_method_workflow(workflow_path, language=language)
    plot_result_overview(
        summary,
        event_predictions,
        overview_path,
        language=language,
    )
    plot_prediction_scatter(
        event_predictions,
        station_predictions,
        scatter_path,
        language=language,
    )
    plot_station_convergence(
        station_predictions,
        convergence_path,
        language=language,
    )
    waveform_sample = build_waveform_sample(config, records_by_key)
    plot_pgd_and_magnitude(
        waveform_sample,
        event_predictions,
        pgd_path,
        language=language,
    )
    map_paths = plot_selected_event_maps(
        coordinates,
        station_predictions,
        output_dir,
        language=language,
    )
    combine_map_images(map_paths, map_composite_path, language=language)

    generated_paths = [
        workflow_path,
        overview_path,
        scatter_path,
        convergence_path,
        pgd_path,
        map_composite_path,
        *map_paths,
    ]
    write_manifest(output_dir, generated_paths, split, language=language)
    return generated_paths


def main() -> None:
    args = parse_args()
    base_output_dir = args.output_dir.resolve()
    sys.path.insert(0, str(REPO_ROOT))

    summary = load_json(RESULT_SUMMARY)
    config, split, event_predictions, station_predictions = load_inputs()
    records_by_key = load_validation_records(config, split)
    coordinates = coordinate_frame(records_by_key)
    validate_prediction_tables(split, event_predictions, station_predictions, coordinates)

    languages = ("en", "zh") if args.language == "both" else (args.language,)
    generated_paths: list[Path] = []
    for language in languages:
        language_output_dir = (
            base_output_dir / "zh" if language == "zh" else base_output_dir
        )
        generated_paths.extend(
            generate_language_figures(
                language=language,
                output_dir=language_output_dir,
                summary=summary,
                config=config,
                split=split,
                event_predictions=event_predictions,
                station_predictions=station_predictions,
                records_by_key=records_by_key,
                coordinates=coordinates,
            )
        )

    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
