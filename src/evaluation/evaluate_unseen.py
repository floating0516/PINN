from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# ── scientific-visualization skill: publication style ──────────────────────
_OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
               "#E69F00", "#56B4E9", "#F0E442", "#000000"]

def _apply_pub_style() -> None:
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         8,
        "axes.labelsize":    9,
        "axes.titlesize":    10,
        "axes.titleweight":  "bold",
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   7,
        "legend.frameon":    False,
        "axes.linewidth":    0.8,
        "lines.linewidth":   1.2,
        "patch.linewidth":   0.5,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })
# ───────────────────────────────────────────────────────────────────────────

from src.baseline.scaling_laws import predict_mw
from src.data.external_records import record_from_external_bundle
from src.data.metadata import build_metadata_tensor
from src.data.metadata import metadata_distance_from_config
from src.data.sample_builder import SampleRejected, build_station_sample
from src.data.waveform import WaveformConfig, waveform_config_from_v2
from src.evaluation.evaluate import (
    _ensure_time_steps,
    _magnitude_from_rate,
    magnitude_series_from_rate,
)
from src.evaluation.metrics import (
    aggregate_event_predictions,
    summarize_predictions,
)
from src.models.model import PINNModel
from src.training.physics import PhysicsLoss
from src.utils.config_v2 import validate_config_on_startup
from src.utils.device import get_preferred_device
from src.visualization.visualize import set_srl_plot_style


@dataclass
class StationWaveform:
    station: str
    latitude: float
    longitude: float
    t: np.ndarray
    e_m: np.ndarray
    n_m: np.ndarray
    u_m: np.ndarray
    dt: float


@dataclass
class EventBundle:
    event_name: str
    magnitude: float
    latitude: float
    longitude: float
    depth_km: float
    mechanism: str
    stations: list[StationWaveform]
    event_dir_name: str = ""
    strike: float = float("nan")
    dip: float = float("nan")
    rake: float = float("nan")


def _waveform_rescale_factor(*, event_dir_name: str, event_name: str) -> float:
    text = f"{event_dir_name} {event_name}".lower()
    if "iquique" in text and "aftershock" in text and "2014" in text:
        return 1.0 / 1000.0
    if "nepal" in text and "aftershock" in text and "2015" in text:
        return 1.0e-4
    return 1.0


def _format_event_display_name(*, event_name: str, event_dir_name: str, magnitude: float) -> str:
    base = event_dir_name.replace("_", "-").strip().lower()
    parts = [part for part in base.split("-") if part]
    year = next((part for part in parts if len(part) == 4 and part.isdigit()), "")
    stopwords = {
        "aftershock", "mainshock", "foreshock", "doublet", "eq1", "eq2", "earthquake",
        "china", "japan", "greece", "alaska", "chile", "mexico", "burma", "myanmar",
        "new", "zealand", "costa", "rica", "plateau", "southern", "tibetan", "of", "km",
        "sw", "se", "nw", "ne", "n", "s", "e", "w",
    }
    candidates = [part for part in parts if not part.isdigit() and part not in stopwords]
    place = candidates[0].capitalize() if candidates else event_name.strip()
    if year:
        return f"{place} {year} M{float(magnitude):.1f}"
    return f"{place} M{float(magnitude):.1f}"


def load_event_bundle(event_dir: str | Path) -> EventBundle:
    event_path = Path(event_dir)
    meta = json.loads((event_path / "event.json").read_text(encoding="utf-8"))

    station_meta: dict[str, dict[str, float]] = {}
    with (event_path / "stations.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                _lat = float(row["Latitude"])
                _lon = float(row["Longitude"])
            except (TypeError, ValueError):
                continue
            station_meta[str(row["Station"])] = {
                "lat": _lat,
                "lon": _lon,
            }

    per_station: dict[str, dict[str, list[float]]] = {}
    waveform_rescale_factor = _waveform_rescale_factor(
        event_dir_name=event_path.name,
        event_name=str(meta.get("event", "")),
    )
    with gzip.open(event_path / "waveforms.csv.gz", "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            station = str(row["Station"])
            component = str(row["Component"]).upper()
            if component not in {"E", "N", "U"}:
                continue
            offset = float(row["Time_Offset_s"])
            value_m = float(row["Value_m"])
            value_m *= waveform_rescale_factor
            item = per_station.setdefault(
                station,
                {"t": [], "E": [], "N": [], "U": []},
            )
            if component == "E":
                item["t"].append(offset)
            item[component].append(value_m)

    stations: list[StationWaveform] = []
    for station, payload in per_station.items():
        t = np.asarray(payload["t"], dtype=np.float32)
        e_m = np.asarray(payload["E"], dtype=np.float32)
        n_m = np.asarray(payload["N"], dtype=np.float32)
        if len(t) == 0 or not (len(t) == len(e_m) == len(n_m)):
            continue
        if len(payload["U"]) == len(t):
            u_m = np.asarray(payload["U"], dtype=np.float32)
        elif len(payload["U"]) == 0:
            u_m = np.zeros_like(t, dtype=np.float32)
        else:
            continue
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
        meta_row = station_meta.get(station)
        if meta_row is None:
            continue
        stations.append(
            StationWaveform(
                station=station,
                latitude=float(meta_row["lat"]),
                longitude=float(meta_row["lon"]),
                t=t,
                e_m=e_m,
                n_m=n_m,
                u_m=u_m,
                dt=dt if math.isfinite(dt) and dt > 0.0 else 1.0,
            )
        )

    depth_km_raw = meta.get("depth_km", 0.0)
    depth_km = float(depth_km_raw) if depth_km_raw is not None else 0.0
    magnitude = float(meta["magnitude"])

    def _optional_float(value: Any) -> float:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    return EventBundle(
        event_name=str(meta["event"]),
        magnitude=magnitude,
        latitude=float(meta["latitude"]),
        longitude=float(meta["longitude"]),
        depth_km=depth_km,
        mechanism=str(meta.get("mechanism", "")),
        stations=stations,
        event_dir_name=event_path.name,
        strike=_optional_float(meta.get("strike")),
        dip=_optional_float(meta.get("dip")),
        rake=_optional_float(meta.get("rake")),
    )


def summarize_event_predictions(*, event_name: str, mw_catalog: float, predictions: list[float]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("predictions 不能为空")
    station_rows = [
        {
            "event": event_name,
            "mw_pred": prediction,
            "mw_catalog": mw_catalog,
            "mw_stf_native": float("nan"),
        }
        for prediction in predictions
    ]
    return aggregate_event_predictions(
        station_rows,
        reference_key="mw_catalog",
    )[0]


def plot_event_station_waveforms(
    *,
    bundle: EventBundle,
    save_path: str | Path,
) -> Path:
    if not bundle.stations:
        raise ValueError("bundle.stations 不能为空")

    stations = sorted(
        bundle.stations,
        key=lambda st: float(np.nanmax(np.sqrt(st.e_m ** 2 + st.n_m ** 2))) if len(st.e_m) else 0.0,
        reverse=True,
    )
    n_rows = len(stations)
    _apply_pub_style()
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.2, max(1.8 * n_rows, 4.0)), sharex=True, squeeze=False)

    for idx, station in enumerate(stations):
        ax = axes[idx, 0]
        t = np.asarray(station.t, dtype=float)
        e = np.asarray(station.e_m, dtype=float)
        n = np.asarray(station.n_m, dtype=float)
        u = np.asarray(station.u_m, dtype=float)
        radial_scale = float(np.nanmax(np.abs(np.concatenate([e, n, u])))) if len(t) else 0.0
        offset = idx * max(radial_scale * 3.0, 1.0)
        ax.plot(t, e + offset, color=_OKABE_ITO[1], linewidth=0.9, label="E" if idx == 0 else None)
        ax.plot(t, n + offset, color=_OKABE_ITO[2], linewidth=0.9, label="N" if idx == 0 else None)
        ax.plot(t, u + offset, color=_OKABE_ITO[4], linewidth=0.9, label="U" if idx == 0 else None)
        ax.text(0.01, 0.82, station.station, transform=ax.transAxes, ha="left", va="top", fontsize=7)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.4, color="#CCCCCC")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks([])
        if idx == 0:
            ax.legend(loc="upper right", ncol=3)
            ax.set_title(f"{bundle.event_name}  $M_w$={bundle.magnitude:.1f}  {bundle.mechanism}")

    axes[-1, 0].set_xlabel("Time (s)")
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_unseen_event_mw_figure(
    *,
    panel_rows: list[dict[str, Any]],
    event_name: str,
    save_path: str | Path,
) -> Path:
    if not panel_rows:
        raise ValueError("panel_rows 不能为空")
    event_rows = [row for row in panel_rows if str(row.get("event", "")) == event_name]
    if not event_rows:
        raise ValueError("指定事件没有可绘制的 panel_rows")

    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.set_facecolor("white")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, color="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    event_rows = sorted(event_rows, key=lambda row: float(row.get("max_radial_cm", float("-inf"))), reverse=True)
    mw_catalog = float(event_rows[0]["mw_catalog"])
    mechanism = str(event_rows[0].get("mechanism", "")).strip()
    pred_vals = np.asarray([float(row["mw_pred"]) for row in event_rows], dtype=float)
    final_pred = float(np.median(pred_vals))
    spread = float(np.std(pred_vals)) if pred_vals.size > 1 else 0.0

    common_t = np.asarray(event_rows[0]["time_axis"], dtype=float)
    series_stack = []
    for row in event_rows:
        t = np.asarray(row["time_axis"], dtype=float)
        mw_series = np.asarray(row["mw_series"], dtype=float)
        if t.shape != common_t.shape or not np.allclose(t, common_t):
            interp = np.interp(common_t, t, mw_series, left=mw_series[0], right=mw_series[-1])
            mw_series = interp
        series_stack.append(mw_series)
        ax.plot(common_t, mw_series, color="#BFC7D5", linewidth=1.0, alpha=0.35, zorder=1)

    series_arr = np.vstack(series_stack)
    median_series = np.median(series_arr, axis=0)
    q25 = np.percentile(series_arr, 25, axis=0)
    q75 = np.percentile(series_arr, 75, axis=0)

    ax.fill_between(common_t, q25, q75, color="#F4A261", alpha=0.22, zorder=2)
    ax.plot(common_t, median_series, color="#E66100", linewidth=2.0, zorder=3)
    ax.axhline(mw_catalog, color="black", linestyle=(0, (4, 3)), linewidth=1.3, zorder=4)
    ax.axhline(mw_catalog + 0.3, color="#9E9E9E", linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)
    ax.axhline(mw_catalog - 0.3, color="#9E9E9E", linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)

    mech_text = f"  {mechanism}" if mechanism else ""
    ax.set_title(f"{event_name}  $M_w$={mw_catalog:.1f}{mech_text}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("$M_w$")
    ax.set_xlim(float(common_t[0]), float(common_t[-1]))
    ax.set_ylim(mw_catalog - 2.0, mw_catalog + 0.5)
    ax.text(0.98, 0.12, f"N={len(event_rows)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    ax.text(0.98, 0.05, f"Pred={final_pred:.2f}±{spread:.02f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_unseen_station_panels(
    *,
    panel_rows: list[dict[str, Any]],
    save_path: str | Path,
    top_n: int = 4,
    sort_by: str = "max_radial_cm",
) -> Path:
    if not panel_rows:
        raise ValueError("panel_rows 不能为空")
    selected_rows = sorted(
        panel_rows,
        key=lambda row: float(row.get(sort_by, float("-inf"))),
        reverse=True,
    )[: max(1, int(top_n))]

    _apply_pub_style()
    fig, axes = plt.subplots(len(selected_rows), 3, figsize=(7.2, 2.2 * len(selected_rows)), squeeze=False, sharex=False)

    for row_idx, row in enumerate(selected_rows):
        t = np.asarray(row["time_axis"], dtype=float)
        radial = np.asarray(row["radial"], dtype=float)
        pred_rate = np.asarray(row["pred_rate"], dtype=float)
        mw_series = np.asarray(row["mw_series"], dtype=float)
        mw_catalog = float(row["mw_catalog"])
        mw_pred = float(row["mw_pred"])
        dist_km = float(row["source_distance_km"])

        ax0, ax1, ax2 = axes[row_idx]
        ax0.plot(t, radial, color=_OKABE_ITO[0], linewidth=1.0)
        ax0.text(0.02, 0.92, f"Dist: {dist_km:.0f} km | $M_w$: {mw_catalog:.2f}", transform=ax0.transAxes, ha="left", va="top", fontsize=6)
        ax0.set_ylabel("Radial disp.")
        ax0.grid(True, axis="y", linestyle=":", linewidth=0.4, color="#CCCCCC")
        ax0.spines["top"].set_visible(False); ax0.spines["right"].set_visible(False)
        if row_idx == 0:
            ax0.set_title("Radial Component")

        ax1.plot(t, pred_rate, color=_OKABE_ITO[1], linestyle="--", label="Predicted")
        ax1.set_ylabel("Moment Rate")
        ax1.grid(True, axis="y", linestyle=":", linewidth=0.4, color="#CCCCCC")
        ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
        if row_idx == 0:
            ax1.set_title("Predicted Source Time Function")
            ax1.legend(frameon=False)

        ax2.plot(t, mw_series, color=_OKABE_ITO[1], linestyle="--", label=f"Predicted: {mw_pred:.2f}")
        ax2.axhline(mw_catalog, color="black", linewidth=0.9, linestyle="-", label=f"Reference: {mw_catalog:.2f}")
        ax2.set_ylabel("$M_w(t)$")
        ax2.grid(True, axis="y", linestyle=":", linewidth=0.4, color="#CCCCCC")
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
        if row_idx == 0:
            ax2.set_title("Moment Magnitude $M_w(t)$")
            ax2.legend(frameon=False)

        if row_idx == len(selected_rows) - 1:
            ax0.set_xlabel("Time (s)")
            ax1.set_xlabel("Time (s)")
            ax2.set_xlabel("Time (s)")

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def _baseline_correct(trace_m: np.ndarray, t_win: np.ndarray, p_arrival_sec: float) -> np.ndarray:
    pre_p_mask = t_win < p_arrival_sec
    if np.sum(pre_p_mask) >= 3:
        baseline = float(np.mean(trace_m[pre_p_mask]))
    else:
        n_pre = min(len(trace_m), max(5, int(len(trace_m) * 0.05)))
        baseline = float(np.mean(trace_m[:n_pre]))
    return trace_m - baseline


def _compute_pgd_3d(e_m: np.ndarray, n_m: np.ndarray, u_m: np.ndarray) -> float:
    return float(np.max(np.sqrt(e_m ** 2 + n_m ** 2 + u_m ** 2)))


def _summarize_scalar_predictions(prefix: str, mw_catalog: float, predictions: list[float]) -> dict[str, Any]:
    if not predictions:
        return {
            f"{prefix}_mw_pred_median": float("nan"),
            f"{prefix}_error": float("nan"),
            f"{prefix}_n_stations": 0,
            f"{prefix}_pred_iqr": float("nan"),
        }
    pred = np.asarray(predictions, dtype=float)
    mw_pred = float(np.median(pred))
    return {
        f"{prefix}_mw_pred_median": mw_pred,
        f"{prefix}_error": mw_pred - float(mw_catalog),
        f"{prefix}_n_stations": int(pred.size),
        f"{prefix}_pred_iqr": float(np.percentile(pred, 75) - np.percentile(pred, 25)) if pred.size > 1 else 0.0,
    }


def write_unseen_event_outputs(
    *,
    output_dir: str | Path,
    station_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    panel_rows: list[dict[str, Any]] | None = None,
    panel_top_n: int = 4,
    save_plots: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    station_csv = output_path / "station_predictions.csv"
    event_csv = output_path / "event_summary.csv"
    station_scatter = output_path / "station_scatter.png"
    event_summary_figure = output_path / "event_summary.png"
    station_panels_dir = output_path / "station_panels"

    if station_rows:
        with station_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_csv_fieldnames(station_rows))
            writer.writeheader()
            writer.writerows(station_rows)
    if event_rows:
        with event_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_csv_fieldnames(event_rows))
            writer.writeheader()
            writer.writerows(event_rows)

    if save_plots and station_rows:
        _apply_pub_style()
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        palette = _OKABE_ITO
        events = sorted({str(row["event"]) for row in station_rows})
        colors: dict[str, str] = {}
        for idx, event_name in enumerate(events):
            colors[event_name] = palette[idx % len(palette)]
            rows = [row for row in station_rows if str(row["event"]) == event_name]
            x = [float(row["mw_catalog"]) for row in rows]
            y = [float(row["mw_pred"]) for row in rows]
            ax.scatter(x, y, label=event_name, s=20, alpha=0.8, color=colors[event_name], edgecolors="none")
        all_vals = [float(row["mw_catalog"]) for row in station_rows] + [float(row["mw_pred"]) for row in station_rows]
        lo = math.floor(min(all_vals) * 10.0) / 10.0
        hi = math.ceil(max(all_vals) * 10.0) / 10.0
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=0.9)
        ax.plot([lo, hi], [lo + 0.3, hi + 0.3], linestyle="--", color="#999999", linewidth=0.7)
        ax.plot([lo, hi], [lo - 0.3, hi - 0.3], linestyle="--", color="#999999", linewidth=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Reference $M_w$")
        ax.set_ylabel("Station-level predicted $M_w$")
        ax.set_title("Unseen event station predictions")
        ax.legend(loc="upper left")
        ax.grid(True, axis="both", linestyle=":", linewidth=0.4, color="#CCCCCC")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(station_scatter, dpi=300, bbox_inches="tight")
        plt.close(fig)

    if save_plots and event_rows:
        _apply_pub_style()
        labels = [str(row["event"]) for row in event_rows]
        x = np.arange(len(labels), dtype=float)
        true_vals = [float(row["mw_catalog"]) for row in event_rows]
        pred_vals = [float(row["mw_pred_median"]) for row in event_rows]
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot(x, true_vals, marker="o", markersize=4, linewidth=1.2, label="Reference $M_w$", color=_OKABE_ITO[0])
        ax.plot(x, pred_vals, marker="s", markersize=4, linewidth=1.2, label="Predicted $M_w$", color=_OKABE_ITO[1])
        for xi, row in zip(x, event_rows):
            ax.text(xi, float(row["mw_pred_median"]) + 0.03, f"err={float(row['error_vs_catalog']):+.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("$M_w$")
        ax.set_title("Unseen event summary")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.4, color="#CCCCCC")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()
        fig.tight_layout()
        fig.savefig(event_summary_figure, dpi=300, bbox_inches="tight")
        plt.close(fig)

    station_panels: dict[str, list[Path]] = {}
    event_mw_figures: dict[str, Path] = {}
    if save_plots and panel_rows:
        station_panels_dir.mkdir(parents=True, exist_ok=True)
        event_mw_dir = output_path / "event_mw_figures"
        event_mw_dir.mkdir(parents=True, exist_ok=True)
        event_names = sorted({str(row["event"]) for row in panel_rows})
        per_page = max(1, int(panel_top_n))
        max_pages = 3
        for event_name in event_names:
            event_panel_rows = [row for row in panel_rows if str(row["event"]) == event_name]
            sorted_rows = sorted(
                event_panel_rows,
                key=lambda row: float(row.get("max_radial_cm", float("-inf"))),
                reverse=True,
            )
            safe_name = event_name.lower().replace(" ", "_").replace("/", "_")
            event_mw_figures[event_name] = plot_unseen_event_mw_figure(
                panel_rows=event_panel_rows,
                event_name=event_name,
                save_path=event_mw_dir / f"{safe_name}_mw_summary.png",
            )
            station_panels[event_name] = []
            for page_idx in range(max_pages):
                start = page_idx * per_page
                end = start + per_page
                page_rows = sorted_rows[start:end]
                if not page_rows:
                    break
                panel_path = station_panels_dir / f"{safe_name}_panels_{page_idx + 1}.png"
                station_panels[event_name].append(
                    plot_unseen_station_panels(
                        panel_rows=page_rows,
                        save_path=panel_path,
                        top_n=len(page_rows),
                        sort_by="max_radial_cm",
                    )
                )

    return {
        "station_csv": station_csv,
        "event_csv": event_csv,
        "station_scatter": station_scatter,
        "event_summary_figure": event_summary_figure,
        "station_panels": station_panels,
        "event_mw_figures": event_mw_figures,
    }


def _station_sample_from_bundle(
    bundle: EventBundle,
    station: StationWaveform,
    config: dict[str, Any],
    *,
    waveform_config: WaveformConfig | None = None,
    radial_peak_min_cm_override: float | None = None,
) -> dict[str, Any] | None:
    radial_peak_min_cm = float(config["dataset"]["radial_peak_min_cm"])
    if radial_peak_min_cm_override is not None:
        radial_peak_min_cm = float(radial_peak_min_cm_override)
    if waveform_config is None:
        waveform_config = waveform_config_from_v2(config)
    try:
        return build_station_sample(
            record_from_external_bundle(bundle, station),
            units="m",
            waveform_config=waveform_config,
            alpha_m_per_s=float(config["physics"]["alpha"]),
            radial_peak_min_cm=radial_peak_min_cm,
        )
    except SampleRejected:
        return None


def evaluate_unseen_events(
    *,
    event_dirs: list[str | Path],
    model_dir: str | Path,
    output_dir: str | Path,
    radial_peak_min_cm_override: float | None = None,
    save_plots: bool = True,
) -> dict[str, Any]:
    model_path = Path(model_dir) / "best_model.pth"
    config_path = Path(model_dir) / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    validate_config_on_startup(config)

    ds_cfg = config.get("dataset", {}) or {}
    train_cfg = config.get("training", {}) or {}
    waveform_config = waveform_config_from_v2(config)
    threshold_cm = float(ds_cfg["radial_peak_min_cm"])
    if radial_peak_min_cm_override is not None:
        threshold_cm = float(radial_peak_min_cm_override)

    device = get_preferred_device()
    checkpoint = torch.load(model_path, map_location=device)
    model = PINNModel(config).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    pipeline_version = int(config.get("pipeline_version", 1))
    criterion = (
        None if pipeline_version == 2 else PhysicsLoss(config).to(device)
    )
    time_steps = int(train_cfg.get("time_steps", 200))
    stf_m_ref = float(
        (ds_cfg.get("stf", {}) or {}).get(
            "m_ref",
            ds_cfg.get("stf_m_ref", 1.0e18),
        )
    )

    station_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for event_dir in event_dirs:
            bundle = load_event_bundle(event_dir)
            event_label = _format_event_display_name(
                event_name=bundle.event_name,
                event_dir_name=bundle.event_dir_name,
                magnitude=bundle.magnitude,
            )
            predictions: list[float] = []
            pgd_predictions: dict[str, list[float]] = {"crowell": [], "ruhl": [], "melgar": []}
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
                radial = torch.tensor(sample["radial"], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                radial = _ensure_time_steps(radial, time_steps)
                source_distance_tensor = torch.tensor(
                    [sample["source_distance_m"]],
                    dtype=torch.float32,
                    device=device,
                )
                epicentral_distance_tensor = torch.tensor(
                    [sample["epicentral_distance_m"]],
                    dtype=torch.float32,
                    device=device,
                )
                metadata_distance = metadata_distance_from_config(
                    config,
                    source_distance_m=source_distance_tensor,
                    epicentral_distance_m=epicentral_distance_tensor,
                )
                meta = build_metadata_tensor(
                    metadata_distance,
                    torch.tensor([sample["theta_deg"]], dtype=torch.float32, device=device),
                    torch.tensor([sample["azimuth_deg"]], dtype=torch.float32, device=device),
                )
                rate_log = model(radial, meta=meta)
                dot_m0 = stf_m_ref * (torch.pow(10.0, rate_log) - 1.0)
                dot_m0 = torch.clamp(dot_m0, min=0.0)
                sample_dt = float(sample["waveform_dt_sec"])
                mw_pred = float(
                    _magnitude_from_rate(
                        dot_m0,
                        torch.tensor([sample_dt], device=device),
                        pipeline_version=pipeline_version,
                        legacy_criterion=criterion,
                    )[0].item()
                )
                predictions.append(mw_pred)

                source_distance_m = float(sample["source_distance_m"])
                source_distance_km = float(sample["source_distance_m"]) / 1000.0
                epicentral_distance_km = (
                    float(sample["epicentral_distance_m"]) / 1000.0
                )
                p_arrival_sec = source_distance_m / float(
                    config["physics"]["alpha"]
                )
                t_win = np.asarray(station.t, dtype=float)
                e_corr = _baseline_correct(np.asarray(station.e_m, dtype=float), t_win, p_arrival_sec)
                n_corr = _baseline_correct(np.asarray(station.n_m, dtype=float), t_win, p_arrival_sec)
                u_corr = _baseline_correct(np.asarray(station.u_m, dtype=float), t_win, p_arrival_sec)
                pgd_3d_m = _compute_pgd_3d(e_corr, n_corr, u_corr)
                pgd_station_values: dict[str, float] = {}
                for law_name in ("crowell", "ruhl", "melgar"):
                    pgd_mw = predict_mw(
                        law_name=law_name,
                        pgd_m=pgd_3d_m,
                        source_distance_km=source_distance_km,
                    )
                    pgd_station_values[law_name] = pgd_mw
                    if math.isfinite(pgd_mw):
                        pgd_predictions[law_name].append(pgd_mw)

                row = {
                    "event": event_label,
                    "station": station.station,
                    "mw_pred": mw_pred,
                    "mw_catalog": bundle.magnitude,
                    "mw_stf_native": float("nan"),
                    "error_vs_catalog": mw_pred - bundle.magnitude,
                    "error_vs_stf_native": float("nan"),
                    "epicentral_distance_km": float(sample["epicentral_distance_m"]) / 1000.0,
                    "source_distance_km": float(sample["source_distance_m"]) / 1000.0,
                    "theta_deg": float(sample["theta_deg"]),
                    "azimuth_deg": float(sample["azimuth_deg"]),
                    "threshold_cm": threshold_cm,
                    "mechanism": bundle.mechanism,
                    "dt": sample_dt,
                    "max_radial_cm": float(sample["radial_peak_cm"]),
                    "station_lat": float(station.latitude),
                    "station_lon": float(station.longitude),
                    "used_in_event_summary": True,
                    "pgd_source_distance_km": source_distance_km,
                    "pgd_epicentral_distance_km": epicentral_distance_km,
                    "pgd_3d_m": pgd_3d_m,
                    "pgd_mw_crowell": pgd_station_values["crowell"],
                    "pgd_mw_ruhl": pgd_station_values["ruhl"],
                    "pgd_mw_melgar": pgd_station_values["melgar"],
                    "pgd_error_crowell": pgd_station_values["crowell"] - bundle.magnitude,
                    "pgd_error_ruhl": pgd_station_values["ruhl"] - bundle.magnitude,
                    "pgd_error_melgar": pgd_station_values["melgar"] - bundle.magnitude,
                }
                mw_series = magnitude_series_from_rate(dot_m0[0].view(-1), sample_dt)
                station_rows.append(row)
                panel_rows.append(
                    {
                        **row,
                        "radial": np.asarray(sample["radial"], dtype=float),
                        "pred_rate": dot_m0[0].detach().cpu().numpy(),
                        "mw_series": mw_series,
                        "time_axis": np.arange(dot_m0.shape[-1], dtype=float) * sample_dt,
                    }
                )
            if predictions:
                event_row = summarize_event_predictions(
                    event_name=event_label,
                    mw_catalog=bundle.magnitude,
                    predictions=predictions,
                )
                event_row.update(
                    {
                        "event_lat": float(bundle.latitude),
                        "event_lon": float(bundle.longitude),
                        "strike": float(bundle.strike),
                        "dip": float(bundle.dip),
                        "rake": float(bundle.rake),
                    }
                )
                event_row.update(_summarize_scalar_predictions("pgd_crowell", bundle.magnitude, pgd_predictions["crowell"]))
                event_row.update(_summarize_scalar_predictions("pgd_ruhl", bundle.magnitude, pgd_predictions["ruhl"]))
                event_row.update(_summarize_scalar_predictions("pgd_melgar", bundle.magnitude, pgd_predictions["melgar"]))
                event_rows.append(event_row)

    output_paths = write_unseen_event_outputs(
        output_dir=output_dir,
        station_rows=station_rows,
        event_rows=event_rows,
        panel_rows=panel_rows,
        save_plots=save_plots,
    )
    metrics = summarize_predictions(
        station_rows,
        event_rows,
        reference_key="mw_catalog",
    )
    return {
        "station_rows": station_rows,
        "event_rows": event_rows,
        "metrics": metrics,
        **output_paths,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估未见过地震上的模型表现")
    parser.add_argument("--model-dir", required=True, help="模型目录，包含 best_model.pth 与 config.yaml")
    parser.add_argument("--event-dir", action="append", required=True, help="外部事件目录，可重复传入")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument("--radial-peak-min-cm", type=float, default=None, help="可选，覆盖径向峰值阈值；设为 0 可保留全部台站")
    parser.add_argument("--no-plots", action="store_true", help="仅写 CSV 和指标，不生成图件")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = evaluate_unseen_events(
        event_dirs=args.event_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        radial_peak_min_cm_override=args.radial_peak_min_cm,
        save_plots=not args.no_plots,
    )
    print(f"台站级结果: {result['station_csv']}")
    print(f"事件级结果: {result['event_csv']}")
    print(f"台站散点图: {result['station_scatter']}")
    print(f"事件汇总图: {result['event_summary_figure']}")


if __name__ == "__main__":
    main()
