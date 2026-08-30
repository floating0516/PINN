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


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path("/home/lihe/PINN_Mag")
RUN_ROOT = WORKSPACE_ROOT / "runs/phase39-causal-direct-moment-scale-20260830-fold0-seed73"
SCREEN_ROOT = WORKSPACE_ROOT / "runs/phase39-causal-direct-moment-scale-screen-20260830"
SOURCE_CONFIG = (
    WORKSPACE_ROOT
    / "runs/phase39-confirmatory-grouped-cv-20260812T0332Z-121197d"
    / "fold_0/phase39/seed_73/models/20260812_113942/config.yaml"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "docs/results/phase39-causal-moment-scaling/figures"
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
EVENT_LABELS = {
    "Anchorage2018": "Anchorage",
    "Maule2010": "Maule",
    "Noto2024": "Noto",
    "Parkfield2004": "Parkfield",
    "RatIslands2014": "Rat Islands",
    "SandPoint2020": "Sand Point",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
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


def plot_prediction_scatter(
    event_predictions: pd.DataFrame,
    station_predictions: pd.DataFrame,
    output_path: Path,
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
            EVENT_LABELS[event],
            (true_mw, candidate_mw),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    event_axis.set_xlim(lower, upper)
    event_axis.set_ylim(lower, upper)
    event_axis.set_aspect("equal", adjustable="box")
    event_axis.set_title("A. Event-median endpoint estimates")
    event_axis.set_xlabel("Catalog magnitude (Mw)")
    event_axis.set_ylabel("Estimated magnitude (Mw)")
    event_axis.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.75)
    event_axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=BASELINE_COLOR, label="Phase 39"),
            Line2D([], [], marker="D", linestyle="", color=CANDIDATE_COLOR, label="Full method"),
            Line2D([], [], color=TRUTH_COLOR, label="Ideal"),
        ],
        frameon=False,
        loc="upper left",
    )

    station_lower, station_upper = identity_limits(
        endpoint_stations["mw_catalog"], endpoint_stations["mw_pred"]
    )
    for axis, method, title in (
        (axes[1], "phase39", "B. Phase 39 station estimates"),
        (axes[2], "direct", "C. Full-method station estimates"),
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
        axis.set_title(f"{title}\nStation MAE = {station_mae:.3f} Mw")
        axis.set_xlabel("Catalog magnitude (Mw)")
        axis.set_ylabel("Station estimate (Mw)")
        axis.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.75)

    event_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=EVENT_COLORS[event],
            label=EVENT_LABELS[event],
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
        "True versus estimated magnitude at 200 s (internal validation)\n"
        "Primary evaluation uses equal-weight event medians; station panels are station-weighted",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(figure, output_path)


def plot_station_convergence(
    station_predictions: pd.DataFrame,
    output_path: Path,
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
            label="Full-method median",
        )
        axis.plot(
            baseline_median.index,
            baseline_median.values,
            color=BASELINE_COLOR,
            linestyle="--",
            marker="s",
            markersize=4,
            linewidth=1.7,
            label="Phase 39 median",
        )
        axis.axhline(truth, color=TRUTH_COLOR, linewidth=1.4, label="Catalog Mw")
        axis.set_xticks(ANCHOR_HORIZONS)
        axis.set_xlabel("Observed causal prefix (s)")
        axis.set_ylabel("Station magnitude estimate (Mw)")
        axis.set_title(f"{EVENT_LABELS[event]} | Mw {truth:.2f} | {station_count} stations")
        axis.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.75)
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    figure.suptitle(
        "Station-level estimates across causal horizons",
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
    axes[0].set_ylabel("Radial displacement (cm)")
    axes[0].set_title("A. Observed radial displacement used by the R-only model")
    axes[0].grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.7)

    axes[1].plot(
        time_sec,
        displacement_norm_cm,
        color="#94A3B8",
        linewidth=1.0,
        label="3-component displacement norm",
    )
    axes[1].plot(
        time_sec,
        cumulative_pgd_cm,
        color=CANDIDATE_COLOR,
        linewidth=2.2,
        label="Cumulative PGD",
    )
    axes[1].set_ylabel("Displacement / PGD (cm)")
    axes[1].set_title("B. Three-component peak ground displacement")
    axes[1].grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    axes[1].legend(frameon=False, loc="lower right")

    for method, label, color, linestyle in (
        ("phase39", "Phase 39", BASELINE_COLOR, "--"),
        ("direct", "Full method", CANDIDATE_COLOR, "-"),
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
        label="Catalog +/-0.20 Mw",
    )
    axes[2].axhline(truth, color=TRUTH_COLOR, linewidth=1.5, label=f"Catalog Mw {truth:.2f}")
    axes[2].set_xlabel("Time since origin / observed causal prefix (s)")
    axes[2].set_ylabel("Event-median estimate (Mw)")
    axes[2].set_title("C. Second-by-second causal magnitude estimate")
    axes[2].grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    axes[2].legend(frameon=False, loc="upper right", ncol=2)
    axes[2].set_xlim(0, 200)

    figure.suptitle(
        "Parkfield 2004: waveform, PGD, and causal magnitude evolution\n"
        f"Station {WAVEFORM_STATION} | epicentral distance "
        f"{float(waveform_sample['epicentral_distance_m']) / 1000.0:.1f} km | "
        f"final PGD {float(cumulative_pgd_cm[-1]):.2f} cm",
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
        label="GNSS stations",
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
        label="Epicenter",
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
    colorbar.set_label("Station Mw residual (estimate - catalog); clipped at +/-1 Mw")
    axis.set_xlim(west, east)
    axis.set_ylim(south, north)
    mean_latitude = 0.5 * (south + north)
    axis.set_aspect(1.0 / max(math.cos(math.radians(mean_latitude)), 0.2))
    axis.set_xlabel("Longitude (deg)")
    axis.set_ylabel("Latitude (deg)")
    axis.set_title(
        f"{EVENT_LABELS[event]} | Mw {catalog_mw:.2f} | "
        f"{station_count} validation stations"
    )
    axis.grid(True, color="#CBD5E1", linewidth=0.55, alpha=0.8)
    axis.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="best")
    save_figure(figure, output_path)
    return output_path


def plot_selected_event_maps(
    coordinates: pd.DataFrame,
    station_predictions: pd.DataFrame,
    output_dir: Path,
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
        map_paths.append(plot_event_map(event, joined[joined["event"] == event], output_dir))
    return map_paths


def combine_map_images(map_paths: list[Path], output_path: Path) -> None:
    figure, axes = plt.subplots(1, len(map_paths), figsize=(18, 5.4), constrained_layout=True)
    for axis, path in zip(axes, map_paths):
        axis.imshow(plt.imread(path))
        axis.axis("off")
    figure.suptitle(
        "Epicenters and validation-station distributions",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(figure, output_path)


def write_manifest(
    output_dir: Path,
    generated_paths: list[Path],
    split: dict[str, Any],
) -> None:
    payload = {
        "status": "complete",
        "figure_language": "English",
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
        description="Plot English explanatory figures for the Phase 39 moment-scaling screen."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO_ROOT))
    configure_matplotlib()

    config, split, event_predictions, station_predictions = load_inputs()
    records_by_key = load_validation_records(config, split)
    coordinates = coordinate_frame(records_by_key)
    validate_prediction_tables(split, event_predictions, station_predictions, coordinates)

    scatter_path = output_dir / "01_prediction_scatter.png"
    convergence_path = output_dir / "02_station_convergence_scatter.png"
    pgd_path = output_dir / "03_parkfield_pgd_and_mw.png"
    map_composite_path = output_dir / "04_selected_event_maps.png"
    plot_prediction_scatter(event_predictions, station_predictions, scatter_path)
    plot_station_convergence(station_predictions, convergence_path)
    waveform_sample = build_waveform_sample(config, records_by_key)
    plot_pgd_and_magnitude(waveform_sample, event_predictions, pgd_path)
    map_paths = plot_selected_event_maps(coordinates, station_predictions, output_dir)
    combine_map_images(map_paths, map_composite_path)

    generated_paths = [
        scatter_path,
        convergence_path,
        pgd_path,
        map_composite_path,
        *map_paths,
    ]
    write_manifest(output_dir, generated_paths, split)
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
