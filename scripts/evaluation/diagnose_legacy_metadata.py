from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import EarthquakeDataset
from src.data.geometry import compute_source_station_geometry
from src.evaluation.evaluate import _ensure_time_steps
from src.evaluation.evaluate_unseen import (
    _format_event_display_name,
    load_event_bundle,
)
from src.models.model import PINNModel
from src.training.physics import PhysicsLoss
from src.utils.device import get_preferred_device
from src.utils.provenance import (
    current_git_commit,
    git_is_dirty,
    sha256_file,
    utc_now_iso,
    write_json,
)


DIAGNOSTIC_MODES = (
    "legacy_exact",
    "theta_only_fixed",
    "geometry_fixed",
    "metadata_disabled",
)

EXTERNAL_EVENT_NAMES = (
    "iquique-aftershock-2014-chile",
    "nepal-aftershock-2015",
    "kodiak-2018-alaska",
    "samos-2020-greece",
    "luding-2022-china",
    "xizang-2025-southern-tibetan-plateau",
    "myanmar-2025-mandalay",
    "sand-point-2025-alaska",
)

_CSV_FIELDS = (
    "event",
    "station",
    "mw_pred",
    "mw_catalog",
    "error_vs_catalog",
    "delta_m",
    "source_distance_m",
    "theta_deg",
    "azimuth_deg",
    "radial_peak_cm",
    "dt_sec",
)


def build_diagnostic_metadata(
    mode: str,
    *,
    delta_m: float,
    theta_deg: float,
    azimuth_deg: float,
) -> np.ndarray | None:
    if mode not in DIAGNOSTIC_MODES:
        raise ValueError(f"unknown metadata mode: {mode}")
    if mode == "metadata_disabled":
        return None
    values = (delta_m, theta_deg, azimuth_deg)
    if not all(math.isfinite(float(value)) for value in values) or delta_m <= 0.0:
        raise ValueError("metadata inputs must be finite and delta_m must be positive")

    theta_rad = math.radians(theta_deg)
    azimuth_rad = math.radians(azimuth_deg)
    if mode == "legacy_exact":
        second_angle = azimuth_rad
        fourth = 0.0
        fifth = 1.0
    elif mode == "theta_only_fixed":
        second_angle = theta_rad
        fourth = 0.0
        fifth = 1.0
    else:
        second_angle = theta_rad
        fourth = math.sin(azimuth_rad)
        fifth = math.cos(azimuth_rad)
    return np.asarray(
        [
            math.log(max(delta_m, 1.0)),
            math.sin(second_angle),
            math.cos(second_angle),
            fourth,
            fifth,
        ],
        dtype=np.float32,
    )


def _station_keys(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    keys = sorted((str(row["event"]), str(row["station"])) for row in rows)
    if len(keys) != len(set(keys)):
        raise ValueError("diagnostic rows contain duplicate station keys")
    return keys


def assert_same_station_keys(
    rows_by_mode: dict[str, list[dict[str, Any]]],
) -> None:
    missing_modes = set(DIAGNOSTIC_MODES).difference(rows_by_mode)
    if missing_modes:
        raise ValueError(f"diagnostic rows missing modes: {sorted(missing_modes)}")
    reference = _station_keys(rows_by_mode[DIAGNOSTIC_MODES[0]])
    for mode in DIAGNOSTIC_MODES[1:]:
        if _station_keys(rows_by_mode[mode]) != reference:
            raise ValueError(f"station key set differs for mode {mode}")


def _event_mae_catalog(rows: list[dict[str, Any]]) -> float:
    predictions: dict[str, list[float]] = defaultdict(list)
    catalog_values: dict[str, float] = {}
    for row in rows:
        event = str(row["event"])
        prediction = float(row["mw_pred"])
        catalog = float(row["mw_catalog"])
        if not math.isfinite(prediction) or not math.isfinite(catalog):
            raise ValueError("diagnostic predictions and catalog values must be finite")
        if event in catalog_values and catalog_values[event] != catalog:
            raise ValueError(f"catalog magnitude differs within event {event}")
        catalog_values[event] = catalog
        predictions[event].append(prediction)
    if not predictions:
        raise ValueError("diagnostic produced no event predictions")
    errors = [
        abs(float(np.median(values)) - catalog_values[event])
        for event, values in sorted(predictions.items())
    ]
    return float(np.mean(errors))


def _prediction_vector(rows: list[dict[str, Any]]) -> np.ndarray:
    ordered = sorted(rows, key=lambda row: (str(row["event"]), str(row["station"])))
    return np.asarray([float(row["mw_pred"]) for row in ordered], dtype=float)


def summarize_diagnostic(
    rows_by_mode: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    assert_same_station_keys(rows_by_mode)
    legacy_predictions = _prediction_vector(rows_by_mode["legacy_exact"])
    theta_predictions = _prediction_vector(rows_by_mode["theta_only_fixed"])
    geometry_predictions = _prediction_vector(rows_by_mode["geometry_fixed"])
    return {
        "legacy_exact_event_mae_catalog": _event_mae_catalog(
            rows_by_mode["legacy_exact"]
        ),
        "theta_only_fixed_event_mae_catalog": _event_mae_catalog(
            rows_by_mode["theta_only_fixed"]
        ),
        "geometry_fixed_event_mae_catalog": _event_mae_catalog(
            rows_by_mode["geometry_fixed"]
        ),
        "metadata_disabled_event_mae_catalog": _event_mae_catalog(
            rows_by_mode["metadata_disabled"]
        ),
        "median_absolute_prediction_change_theta_only": float(
            np.median(np.abs(theta_predictions - legacy_predictions))
        ),
        "median_absolute_prediction_change_geometry_fixed": float(
            np.median(np.abs(geometry_predictions - legacy_predictions))
        ),
    }


def _resolve_event_root(value: str | Path | None) -> Path:
    if value is not None:
        root = Path(value)
    else:
        pinn_root = os.environ.get("PINN_ROOT")
        if not pinn_root:
            raise ValueError("provide --event-root or set PINN_ROOT")
        root = Path(pinn_root) / "incoming" / "legacy-task17" / "GNSS_EQDATA"
    if not root.is_dir():
        raise FileNotFoundError(f"external event root not found: {root}")
    return root


def _resolve_data_path(config: dict[str, Any], value: str | Path | None) -> Path:
    if value is not None:
        path = Path(value)
    else:
        configured = Path(config["paths"]["data_path"])
        if configured.is_file():
            path = configured
        else:
            data_root = os.environ.get("PINN_DATA_ROOT")
            if not data_root:
                raise ValueError("provide --data-path or set PINN_DATA_ROOT")
            path = Path(data_root) / configured.name
    if not path.is_file():
        raise FileNotFoundError(f"legacy training NPZ not found: {path}")
    return path


def _legacy_preprocessor(
    config: dict[str, Any],
    *,
    data_path: Path,
    radial_peak_min_cm: float,
) -> EarthquakeDataset:
    dataset = config.get("dataset", {}) or {}
    training = config.get("training", {}) or {}
    return EarthquakeDataset(
        npz_path=data_path,
        time_steps=int(training.get("time_steps", 200)),
        blacklist=dataset.get("blacklist_events", []),
        units="m",
        center_mode=dataset.get("center_mode", "median"),
        window_min_sec=float(dataset.get("window_min_sec", 0.0)),
        window_max_sec=float(dataset.get("window_max_sec", 600.0)),
        stf_path=None,
        stf_m_ref=float(dataset.get("stf_m_ref", 1.0e18)),
        default_theta_deg=float(dataset.get("default_theta_deg", 45.0)),
        default_phi_deg=float(dataset.get("default_phi_deg", 0.0)),
        filter_cfg=dataset.get("filter", {}),
        p_preprocess_enabled=bool(dataset.get("p_preprocess_enabled", False)),
        p_velocity_mps=float(dataset.get("p_velocity_mps", 4500.0)),
        p_arrival_offset_sec=float(dataset.get("p_arrival_offset_sec", 0.0)),
        p_baseline_mode=str(dataset.get("p_baseline_mode", "mean")),
        allow_missing_stf=True,
        radial_peak_min_cm=radial_peak_min_cm,
    )


def _prepare_external_samples(
    config: dict[str, Any],
    *,
    event_root: Path,
    data_path: Path,
    radial_peak_min_cm: float,
) -> list[dict[str, Any]]:
    processor = _legacy_preprocessor(
        config,
        data_path=data_path,
        radial_peak_min_cm=radial_peak_min_cm,
    )
    samples: list[dict[str, Any]] = []
    for directory_name in EXTERNAL_EVENT_NAMES:
        bundle = load_event_bundle(event_root / directory_name)
        event_label = _format_event_display_name(
            event_name=bundle.event_name,
            event_dir_name=bundle.event_dir_name,
            magnitude=bundle.magnitude,
        )
        for station in bundle.stations:
            geometry = compute_source_station_geometry(
                bundle.latitude,
                bundle.longitude,
                bundle.depth_km,
                station.latitude,
                station.longitude,
            )
            azimuth_rad = math.radians(geometry.azimuth_deg)
            radial = (
                np.asarray(station.e_m, dtype=float) * math.sin(azimuth_rad)
                + np.asarray(station.n_m, dtype=float) * math.cos(azimuth_rad)
            )
            times = np.asarray(station.t, dtype=float)
            mask = (
                (times >= 0.0)
                & (times >= processor.window_min_sec)
                & (times <= processor.window_max_sec)
            )
            if not np.any(mask):
                continue
            radial_processed, dt_sec = processor._preprocess_waveform(
                times[mask],
                radial[mask],
            )
            vertical_processed, _ = processor._preprocess_waveform(
                times[mask],
                np.asarray(station.u_m, dtype=float)[mask],
            )
            if processor.p_preprocess_enabled:
                radial_processed, _ = processor._apply_p_baseline(
                    radial_processed,
                    vertical_processed,
                    geometry.epicentral_distance_m,
                    dt_sec,
                )
            peak_cm = float(np.max(np.abs(radial_processed)) * 100.0)
            if not math.isfinite(peak_cm) or peak_cm <= radial_peak_min_cm:
                continue
            samples.append(
                {
                    "event": event_label,
                    "station": station.station,
                    "mw_catalog": float(bundle.magnitude),
                    "radial": np.asarray(radial_processed, dtype=np.float32),
                    "delta_m": float(geometry.epicentral_distance_m),
                    "source_distance_m": float(geometry.source_distance_m),
                    "theta_deg": float(geometry.takeoff_angle_deg),
                    "azimuth_deg": float(geometry.azimuth_deg),
                    "radial_peak_cm": peak_cm,
                    "dt_sec": float(dt_sec),
                }
            )
    samples.sort(key=lambda sample: (str(sample["event"]), str(sample["station"])))
    keys = [(str(sample["event"]), str(sample["station"])) for sample in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("prepared external samples contain duplicate station keys")
    if not samples:
        raise ValueError("legacy diagnostic retained no external samples")
    return samples


def _predict_modes(
    config: dict[str, Any],
    *,
    model_path: Path,
    samples: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    device = get_preferred_device()
    model = PINNModel(config).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    criterion = PhysicsLoss(config).to(device)
    training = config.get("training", {}) or {}
    dataset = config.get("dataset", {}) or {}
    time_steps = int(training.get("time_steps", 200))
    m_ref = float(dataset.get("stf_m_ref", 1.0e18))
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    with torch.no_grad():
        for mode in DIAGNOSTIC_MODES:
            rows: list[dict[str, Any]] = []
            for sample in samples:
                radial = torch.as_tensor(
                    sample["radial"],
                    dtype=torch.float32,
                    device=device,
                ).view(1, 1, -1)
                radial = _ensure_time_steps(radial, time_steps)
                metadata = build_diagnostic_metadata(
                    mode,
                    delta_m=float(sample["delta_m"]),
                    theta_deg=float(sample["theta_deg"]),
                    azimuth_deg=float(sample["azimuth_deg"]),
                )
                meta_tensor = (
                    None
                    if metadata is None
                    else torch.as_tensor(
                        metadata,
                        dtype=torch.float32,
                        device=device,
                    ).view(1, 5)
                )
                rate_log = model(radial, meta=meta_tensor)
                moment_rate = torch.clamp(
                    m_ref * (torch.pow(10.0, rate_log) - 1.0),
                    min=0.0,
                )
                mw_pred = float(
                    criterion.utils.magnitude_from_rate(
                        moment_rate,
                        float(sample["dt_sec"]),
                    )[0].item()
                )
                if not math.isfinite(mw_pred):
                    raise ValueError(
                        f"non-finite prediction for {mode} "
                        f"{sample['event']}/{sample['station']}"
                    )
                rows.append(
                    {
                        "event": sample["event"],
                        "station": sample["station"],
                        "mw_pred": mw_pred,
                        "mw_catalog": sample["mw_catalog"],
                        "error_vs_catalog": mw_pred - float(sample["mw_catalog"]),
                        "delta_m": sample["delta_m"],
                        "source_distance_m": sample["source_distance_m"],
                        "theta_deg": sample["theta_deg"],
                        "azimuth_deg": sample["azimuth_deg"],
                        "radial_peak_cm": sample["radial_peak_cm"],
                        "dt_sec": sample["dt_sec"],
                    }
                )
            rows_by_mode[mode] = rows
    assert_same_station_keys(rows_by_mode)
    return rows_by_mode, str(device)


def _write_mode_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostic(
    *,
    model_dir: str | Path,
    event_root: str | Path | None,
    output_dir: str | Path,
    data_path: str | Path | None = None,
    radial_peak_min_cm: float | None = None,
) -> dict[str, Path]:
    model_directory = Path(model_dir)
    model_path = model_directory / "best_model.pth"
    config_path = model_directory / "config.yaml"
    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("model directory must contain best_model.pth and config.yaml")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    resolved_event_root = _resolve_event_root(event_root)
    resolved_data_path = _resolve_data_path(config, data_path)
    configured_threshold = float(
        (config.get("dataset", {}) or {}).get("radial_peak_min_cm", 0.0)
    )
    threshold = configured_threshold if radial_peak_min_cm is None else float(radial_peak_min_cm)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("radial_peak_min_cm must be finite and non-negative")

    samples = _prepare_external_samples(
        config,
        event_root=resolved_event_root,
        data_path=resolved_data_path,
        radial_peak_min_cm=threshold,
    )
    rows_by_mode, device = _predict_modes(
        config,
        model_path=model_path,
        samples=samples,
    )
    summary = summarize_diagnostic(rows_by_mode)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    mode_paths: dict[str, Path] = {}
    for mode in DIAGNOSTIC_MODES:
        mode_path = output / f"{mode}.csv"
        _write_mode_csv(mode_path, rows_by_mode[mode])
        mode_paths[mode] = mode_path
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    delivery_root = resolved_event_root.parent
    delivery_checksums = delivery_root / "_package" / "SHA256SUMS"
    headline_predictions = (
        delivery_root / "outputs" / "results" / "test_set_predictions_far_only.csv"
    )
    if not delivery_checksums.is_file() or not headline_predictions.is_file():
        raise FileNotFoundError(
            "Task 17 delivery is missing SHA256SUMS or headline predictions"
        )
    manifest_path = output / "diagnostic_manifest.json"
    write_json(
        manifest_path,
        {
            "purpose": "legacy_metadata_diagnostic_only",
            "created_at_utc": utc_now_iso(),
            "device": device,
            "git_commit": current_git_commit(PROJECT_ROOT),
            "git_dirty": git_is_dirty(PROJECT_ROOT),
            "model_sha256": sha256_file(model_path),
            "config_sha256": sha256_file(config_path),
            "data_sha256": sha256_file(resolved_data_path),
            "delivery_checksums_sha256": sha256_file(delivery_checksums),
            "headline_predictions_sha256": sha256_file(headline_predictions),
            "event_root": str(resolved_event_root),
            "radial_peak_min_cm": threshold,
            "station_count": len(samples),
            "event_count": len({str(sample["event"]) for sample in samples}),
            "modes": list(DIAGNOSTIC_MODES),
            "summary_sha256": sha256_file(summary_path),
        },
    )
    return {
        "summary": summary_path,
        "manifest": manifest_path,
        **mode_paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantify the legacy external-event metadata mismatch"
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--event-root")
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--radial-peak-min-cm", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = run_diagnostic(
        model_dir=args.model_dir,
        event_root=args.event_root,
        output_dir=args.output_dir,
        data_path=args.data_path,
        radial_peak_min_cm=args.radial_peak_min_cm,
    )
    print(f"summary: {paths['summary']}")
    print(f"manifest: {paths['manifest']}")


if __name__ == "__main__":
    main()
