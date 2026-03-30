from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline.scaling_laws import AVAILABLE_SCALING_LAWS, predict_mw


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return radius_m * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def azimuth_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def compute_radial(e_m: np.ndarray, n_m: np.ndarray, az_deg: float) -> np.ndarray:
    az_rad = math.radians(az_deg)
    return e_m * math.sin(az_rad) + n_m * math.cos(az_rad)


def compute_horizontal_pgd(e_m: np.ndarray, n_m: np.ndarray) -> float:
    return float(np.max(np.sqrt(e_m ** 2 + n_m ** 2)))


def compute_pgd_3d(e_m: np.ndarray, n_m: np.ndarray, u_m: np.ndarray) -> float:
    return float(np.max(np.sqrt(e_m ** 2 + n_m ** 2 + u_m ** 2)))


def summarize_event(values: Iterable[float]) -> tuple[float, float, int]:
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        raise ValueError("values 不能为空")
    median = float(np.median(vals))
    iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25)) if vals.size > 1 else 0.0
    return median, iqr, int(vals.size)


def parse_law_names(names: list[str]) -> list[str]:
    resolved: list[str] = []
    for name in names:
        lowered = name.lower()
        if lowered == "all":
            for item in AVAILABLE_SCALING_LAWS.keys():
                if item not in resolved:
                    resolved.append(item)
            continue
        if lowered not in AVAILABLE_SCALING_LAWS:
            raise ValueError(f"未知方法: {name}")
        if lowered not in resolved:
            resolved.append(lowered)
    return resolved


def extract_station_traces_m(waveforms: dict, max_window_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    t = waveforms.get("t")
    e_mm = waveforms.get("E")
    n_mm = waveforms.get("N")
    u_mm = waveforms.get("U")
    if t is None or e_mm is None or n_mm is None or u_mm is None:
        return None

    t = np.asarray(t, dtype=float)
    e_mm = np.asarray(e_mm, dtype=float)
    n_mm = np.asarray(n_mm, dtype=float)
    u_mm = np.asarray(u_mm, dtype=float)
    if len(t) < 10 or len(e_mm) < 10 or len(n_mm) < 10 or len(u_mm) < 10:
        return None

    mask_ok = np.isfinite(t) & np.isfinite(e_mm) & np.isfinite(n_mm) & np.isfinite(u_mm)
    t = t[mask_ok]
    e_mm = e_mm[mask_ok]
    n_mm = n_mm[mask_ok]
    u_mm = u_mm[mask_ok]
    if len(t) < 10:
        return None

    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    if not math.isfinite(dt) or dt <= 0.0:
        dt = 1.0

    if float(np.min(t)) < 0.0:
        origin_mask = t >= 0.0
        t = t[origin_mask]
        e_mm = e_mm[origin_mask]
        n_mm = n_mm[origin_mask]
        u_mm = u_mm[origin_mask]

    win_mask = (t >= 0.0) & (t <= max_window_sec)
    t_win = t[win_mask]
    e_m = e_mm[win_mask] * 0.001
    n_m = n_mm[win_mask] * 0.001
    u_m = u_mm[win_mask] * 0.001
    if len(t_win) < 10:
        return None
    return t_win, e_m, n_m, u_m, dt


def baseline_correct(trace_m: np.ndarray, t_win: np.ndarray, p_arrival_sec: float) -> np.ndarray:
    pre_p_mask = t_win < p_arrival_sec
    if np.sum(pre_p_mask) >= 3:
        baseline = float(np.mean(trace_m[pre_p_mask]))
    else:
        n_pre = min(len(trace_m), max(5, int(len(trace_m) * 0.05)))
        baseline = float(np.mean(trace_m[:n_pre]))
    return trace_m - baseline


def write_plot_outputs(method_event_rows: list[dict[str, object]], csv_path: str | Path) -> dict[str, Path | float]:
    csv_path = Path(csv_path)
    scatter_path = csv_path.with_name(f"{csv_path.stem}_scatter.png")
    bar_path = csv_path.with_name(f"{csv_path.stem}_mae_bar.png")
    guide_offset = 0.3

    method_names = sorted({str(row["method"]) for row in method_event_rows})

    paper_colors = {
        "crowell": "#4C78A8",
        "ruhl": "#72B7B2",
        "melgar": "#B279A2",
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("white")
    ax.grid(True, linestyle="-", linewidth=0.6, color="#E6E6E6", alpha=0.9)
    for method_name in method_names:
        rows = [row for row in method_event_rows if row["method"] == method_name]
        x = [float(row["mw_ref"]) for row in rows]
        y = [float(row["mw_pred"]) for row in rows]
        ax.scatter(
            x,
            y,
            label=method_name,
            alpha=0.8,
            s=34,
            color=paper_colors.get(method_name, "#4C78A8"),
            edgecolors="none",
        )
    if method_event_rows:
        all_vals = [float(row["mw_ref"]) for row in method_event_rows] + [float(row["mw_pred"]) for row in method_event_rows]
        lo = math.floor(min(all_vals) * 10.0) / 10.0
        hi = math.ceil(max(all_vals) * 10.0) / 10.0
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
        ax.plot([lo, hi], [lo + guide_offset, hi + guide_offset], linestyle="--", color="#B8B8B8", linewidth=0.9)
        ax.plot([lo, hi], [lo - guide_offset, hi - guide_offset], linestyle="--", color="#B8B8B8", linewidth=0.9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("Reference Mw")
    ax.set_ylabel("Predicted Mw")
    ax.set_title("PGD scaling law comparison")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(scatter_path, bbox_inches="tight")
    plt.close(fig)

    mae_values = []
    for method_name in method_names:
        rows = [row for row in method_event_rows if row["method"] == method_name]
        errors = np.asarray([abs(float(row["error"])) for row in rows], dtype=float)
        mae_values.append(float(np.mean(errors)))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(method_names, mae_values)
    ax.set_ylabel("MAE")
    ax.set_title("PGD baseline MAE by method")
    fig.tight_layout()
    fig.savefig(bar_path, bbox_inches="tight")
    plt.close(fig)

    return {"scatter": scatter_path, "bar": bar_path, "guide_offset": guide_offset, "style": "paper"}


def evaluate_pgd_scaling_laws(
    npz_path: str,
    laws: list[str],
    output_csv: str | None = None,
    rho: float = 3400.0,
    alpha: float = 7900.0,
    beta: float = 4533.0,
    max_window_sec: float = 200.0,
    max_dist_km: float = 800.0,
    min_pgd_cm: float = 5.0,
    blacklist: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    if blacklist is None:
        blacklist = [
            "N.Honshu2011", "N.Honshu2012", "N.Honshu2013", "E.Fukushima2011", "Iwate2011",
        ]
    blacklist_set = set(blacklist)
    selected_laws = parse_law_names(laws)

    data = np.load(npz_path, allow_pickle=True)
    events = data["events"]
    magnitudes = data["magnitude"]
    event_lats = data["latitude"]
    event_lons = data["longitude"]
    depths_km = data["depth_km"]
    mechanisms = data.get("mechanism", None)
    enu_all = data["enu"]
    station_info_all = data["station_info"]

    method_event_rows: list[dict[str, object]] = []

    for idx in range(len(events)):
        event_name = str(events[idx])
        if event_name in blacklist_set:
            continue

        event_mw = float(magnitudes[idx])
        event_lat = float(event_lats[idx])
        event_lon = float(event_lons[idx])
        event_depth_km = float(depths_km[idx])
        event_mech = str(mechanisms[idx]) if mechanisms is not None else ""
        enu_ev = enu_all[idx]
        station_info = station_info_all[idx]

        per_method_predictions: dict[str, list[float]] = {name: [] for name in selected_laws}

        for station_name, waveforms in enu_ev.items():
            station_meta = station_info.get(station_name, {})
            station_lat = float(station_meta.get("lat", float("nan")))
            station_lon = float(station_meta.get("lon", float("nan")))
            if not math.isfinite(station_lat) or not math.isfinite(station_lon):
                continue

            epicentral_dist_m = haversine_m(event_lat, event_lon, station_lat, station_lon)
            epicentral_dist_km = epicentral_dist_m / 1000.0
            if epicentral_dist_km > max_dist_km:
                continue

            traces = extract_station_traces_m(waveforms, max_window_sec=max_window_sec)
            if traces is None:
                continue
            t_win, e_m, n_m, u_m, dt = traces

            p_arrival_sec = epicentral_dist_m / alpha
            e_m = baseline_correct(e_m, t_win, p_arrival_sec)
            n_m = baseline_correct(n_m, t_win, p_arrival_sec)
            u_m = baseline_correct(u_m, t_win, p_arrival_sec)

            pgd_3d_m = compute_pgd_3d(e_m, n_m, u_m)
            if pgd_3d_m * 100.0 < min_pgd_cm:
                continue

            horizontal_pgd_m = compute_horizontal_pgd(e_m, n_m)

            for law_name in selected_laws:
                mw_pred = predict_mw(
                    law_name=law_name,
                    pgd_m=pgd_3d_m,
                    distance_km=epicentral_dist_km,
                )
                if math.isfinite(mw_pred):
                    per_method_predictions[law_name].append(mw_pred)

        for method_name, predictions in per_method_predictions.items():
            if not predictions:
                continue
            median_pred, iqr, n_stations = summarize_event(predictions)
            method_event_rows.append(
                {
                    "event": event_name,
                    "mechanism": event_mech,
                    "mw_ref": event_mw,
                    "method": method_name,
                    "mw_pred": median_pred,
                    "error": median_pred - event_mw,
                    "iqr": iqr,
                    "n_stations": n_stations,
                }
            )

    if output_csv is None:
        output_csv = str(Path(npz_path).resolve().parent / "pgd_scaling_law_results.csv")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["event", "mechanism", "mw_ref", "method", "mw_pred", "error", "iqr", "n_stations"],
        )
        writer.writeheader()
        writer.writerows(method_event_rows)

    plot_paths = write_plot_outputs(method_event_rows, output_path)

    summary: dict[str, dict[str, float]] = {}
    print("=" * 100)
    print(f"PGD scaling-law evaluation results -> {output_path}")
    print(f"Scatter plot -> {plot_paths['scatter']}")
    print(f"MAE bar plot -> {plot_paths['bar']}")
    print("=" * 100)
    print(f"{'Method':<12} {'Events':>6} {'MAE':>8} {'RMSE':>8} {'Bias':>8}")
    print("-" * 100)
    for method_name in selected_laws:
        rows = [row for row in method_event_rows if row["method"] == method_name]
        if not rows:
            continue
        errors = np.asarray([float(row["error"]) for row in rows], dtype=float)
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        bias = float(np.mean(errors))
        summary[method_name] = {"events": float(len(rows)), "mae": mae, "rmse": rmse, "bias": bias}
        print(f"{method_name:<12} {len(rows):6d} {mae:8.3f} {rmse:8.3f} {bias:8.3f}")
    print("=" * 100)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PGD-Mw scaling-law baselines on raw GNSS NPZ data.")
    parser.add_argument("--npz-path", required=True, help="Absolute path to raw GNSS NPZ file")
    parser.add_argument("--laws", nargs="+", default=["all"], help="Methods to run: all, eew0012, melgar, ruhl, crowell")
    parser.add_argument("--output-csv", default=None, help="Optional CSV output path")
    parser.add_argument("--max-window-sec", type=float, default=200.0)
    parser.add_argument("--max-dist-km", type=float, default=800.0)
    parser.add_argument("--min-pgd-cm", type=float, default=5.0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate_pgd_scaling_laws(
        npz_path=args.npz_path,
        laws=args.laws,
        output_csv=args.output_csv,
        max_window_sec=args.max_window_sec,
        max_dist_km=args.max_dist_km,
        min_pgd_cm=args.min_pgd_cm,
    )


if __name__ == "__main__":
    main()
