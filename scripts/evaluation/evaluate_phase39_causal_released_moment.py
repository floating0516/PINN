"""Frozen replay of a causal Phase 39 checkpoint with released-moment reporting.

For every observation horizon ``h = 1..200`` the same predicted STF is
integrated two ways (see ``src/training/released_moment.py``):

* ``final``        (A) full 200 s integral -- the quantity figure 11 reported;
* ``released``     (B) integral over source time ``< h - tau_P(station)``;
* ``released_ref``     B applied to the SCARDEC reference STF (the label).

Cumulative-PGD Crowell / Ruhl / Melgar are evaluated on the same stations.
Inference only: nothing is trained or selected here. The script works for
both the fixed validation cohort (development) and the fixed test cohort
(one-time replay); the cohort is an explicit argument and is recorded.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments import run_phase39_causal_direct as direct  # noqa: E402
from scripts.experiments import run_phase39_expanded_fixed_split as fixed  # noqa: E402
from scripts.experiments.run_phase39_causal_released_moment import (  # noqa: E402
    CONSTRAINED_SUFFIX,
    FINAL_METHOD,
    RELEASED_METHOD,
    RELEASED_REFERENCE_METHOD,
    evaluate_released_horizons,
)
from src.baseline.scaling_laws import predict_mw  # noqa: E402
from src.data.dataset_v2 import CorrectedEarthquakeDataset  # noqa: E402
from src.data.loaders_v2 import get_data_loaders_v2  # noqa: E402
from src.models.model import PINNModel  # noqa: E402
from src.training.train import _build_stf_rate_criterion  # noqa: E402
from src.utils.config_v2 import validate_config_on_startup  # noqa: E402
from src.utils.provenance import sha256_file  # noqa: E402


HORIZONS = tuple(range(1, 201))
ANCHOR_HORIZONS = {30, 60, 90, 120, 160, 200}
PGD_METHODS = ("crowell", "ruhl", "melgar")
RELEASED_CONSTRAINED_METHOD = f"{RELEASED_METHOD}{CONSTRAINED_SUFFIX}"
RELEASED_REFERENCE_CONSTRAINED_METHOD = f"{RELEASED_REFERENCE_METHOD}{CONSTRAINED_SUFFIX}"
MODEL_METHODS = (
    RELEASED_METHOD,
    RELEASED_CONSTRAINED_METHOD,
    FINAL_METHOD,
    RELEASED_REFERENCE_METHOD,
    RELEASED_REFERENCE_CONSTRAINED_METHOD,
)
BAND_MW = 0.3
RANK_HORIZONS = (10, 20, 30, 60, 90)
METHOD_LABELS = {
    RELEASED_METHOD: "B released (model, all stations)",
    RELEASED_CONSTRAINED_METHOD: "B* released (model, P-arrived stations)",
    FINAL_METHOD: "A final (model)",
    RELEASED_REFERENCE_METHOD: "B released (SCARDEC label)",
    RELEASED_REFERENCE_CONSTRAINED_METHOD: "B* released (SCARDEC label)",
    "crowell": "Crowell 2013 PGD",
    "ruhl": "Ruhl 2019 PGD",
    "melgar": "Melgar 2015 PGD",
}
METHOD_STYLES = {
    RELEASED_METHOD: dict(color="#d62728", lw=1.2, alpha=0.6),
    RELEASED_CONSTRAINED_METHOD: dict(color="#d62728", lw=2.0),
    FINAL_METHOD: dict(color="#ff9896", lw=1.2, ls="--"),
    RELEASED_REFERENCE_METHOD: dict(color="#2ca02c", lw=1.0, ls=":", alpha=0.6),
    RELEASED_REFERENCE_CONSTRAINED_METHOD: dict(color="#2ca02c", lw=1.4, ls=":"),
    "crowell": dict(color="#1f77b4", lw=1.6),
    "ruhl": dict(color="#9467bd", lw=1.0, alpha=0.7),
    "melgar": dict(color="#8c564b", lw=1.0, alpha=0.7),
}


# --------------------------------------------------------------------------- PGD


def evaluate_causal_pgd(
    samples: Sequence[Mapping[str, Any]],
    *,
    expected_record_count: int,
) -> dict[str, list[dict[str, Any]]]:
    station_curves: dict[str, list[dict[str, Any]]] = {m: [] for m in PGD_METHODS}
    for sample in samples:
        event, station = str(sample["event"]), str(sample["station"])
        components = np.stack(
            [
                np.asarray(sample["radial"], dtype=np.float64),
                np.asarray(sample["tangential"], dtype=np.float64),
                np.asarray(sample["vertical"], dtype=np.float64),
            ],
            axis=0,
        )
        if components.shape[1] != len(HORIZONS):
            raise ValueError(f"PGD waveform length changed for {event}/{station}")
        pgd_curve = np.maximum.accumulate(np.sqrt(np.sum(np.square(components), axis=0)))
        if not bool(np.all(np.isfinite(pgd_curve))) or float(pgd_curve[-1]) <= 0.0:
            raise ValueError(f"invalid causal PGD curve for {event}/{station}")
        distance_km = float(sample["source_distance_m"]) / 1000.0
        catalog = float(sample["magnitude_catalog"])
        for method in PGD_METHODS:
            predictions = np.full(pgd_curve.shape, np.nan, dtype=np.float64)
            valid = pgd_curve > 0.0
            predictions[valid] = [
                predict_mw(law_name=method, pgd_m=float(p), source_distance_km=distance_km)
                for p in pgd_curve[valid]
            ]
            station_curves[method].append(
                {"event": event, "station": station, "mw_catalog": catalog, "predictions": predictions}
            )

    horizon_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    endpoint_station_rows: list[dict[str, Any]] = []
    for method, records in station_curves.items():
        if len(records) != expected_record_count:
            raise ValueError(f"PGD station coverage changed: {method}")
        events = sorted({r["event"] for r in records})
        for index, horizon in enumerate(HORIZONS):
            station_errors = np.asarray(
                [
                    float(r["predictions"][index]) - float(r["mw_catalog"])
                    for r in records
                    if math.isfinite(float(r["predictions"][index]))
                ]
            )
            current: list[dict[str, Any]] = []
            for event in events:
                selected = [r for r in records if r["event"] == event]
                preds = np.asarray(
                    [
                        r["predictions"][index]
                        for r in selected
                        if math.isfinite(float(r["predictions"][index]))
                    ],
                    dtype=np.float64,
                )
                if preds.size == 0:
                    continue
                catalog = float(selected[0]["mw_catalog"])
                median = float(np.median(preds))
                current.append(
                    {
                        "method": method,
                        "observation_horizon_sec": int(horizon),
                        "event": event,
                        "mw_pred_median": median,
                        "mw_catalog": catalog,
                        "error_vs_catalog": median - catalog,
                        "n_stations": len(selected),
                        "pred_std": float(np.std(preds)),
                        "pred_iqr": float(np.percentile(preds, 75) - np.percentile(preds, 25)),
                    }
                )
            event_rows.extend(current)
            event_errors = np.asarray([r["error_vs_catalog"] for r in current], dtype=np.float64)
            horizon_rows.append(
                {
                    "method": method,
                    "observation_horizon_sec": int(horizon),
                    "reference": "catalog",
                    "event_count": len(current),
                    "station_count": int(station_errors.size),
                    "event_mae": float(np.mean(np.abs(event_errors))) if current else float("nan"),
                    "event_rmse": float(np.sqrt(np.mean(np.square(event_errors)))) if current else float("nan"),
                    "event_bias": float(np.mean(event_errors)) if current else float("nan"),
                    "station_mae": float(np.mean(np.abs(station_errors))) if station_errors.size else float("nan"),
                    "station_rmse": float(np.sqrt(np.mean(np.square(station_errors)))) if station_errors.size else float("nan"),
                    "station_bias": float(np.mean(station_errors)) if station_errors.size else float("nan"),
                }
            )
        for r in records:
            prediction = float(r["predictions"][-1])
            endpoint_station_rows.append(
                {
                    "method": method,
                    "observation_horizon_sec": 200,
                    "event": r["event"],
                    "station": r["station"],
                    "mw_pred": prediction,
                    "mw_catalog": float(r["mw_catalog"]),
                    "mw_stf_native": float("nan"),
                    "constrained_window_sec": float("nan"),
                }
            )
    return {
        "horizon_rows": horizon_rows,
        "event_rows": event_rows,
        "endpoint_station_rows": endpoint_station_rows,
    }


# --------------------------------------------------------------------- objective table


def _event_curves(event_rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, np.ndarray]:
    curves: dict[str, dict[int, float]] = {}
    catalogs: dict[str, float] = {}
    for row in event_rows:
        if row["method"] != method:
            continue
        curves.setdefault(str(row["event"]), {})[int(row["observation_horizon_sec"])] = float(
            row["mw_pred_median"]
        )
        catalogs[str(row["event"])] = float(row["mw_catalog"])
    out: dict[str, np.ndarray] = {}
    for event, values in curves.items():
        arr = np.full(len(HORIZONS), np.nan)
        for h, v in values.items():
            arr[h - 1] = v
        out[event] = arr
    out["__catalog__"] = np.asarray([catalogs[e] for e in sorted(curves)])
    out["__events__"] = np.asarray(sorted(curves))
    return out


def _nan_to_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nan_to_none(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sign_changes(curve: np.ndarray) -> int:
    finite = curve[np.isfinite(curve)]
    dx = np.diff(finite)
    dx = dx[dx != 0.0]
    if dx.size < 2:
        return 0
    return int(np.sum(np.sign(dx[1:]) * np.sign(dx[:-1]) < 0))


def _first_entry(curve: np.ndarray, catalog: float, band: float) -> int | None:
    inside = np.abs(curve - catalog) <= band
    hits = np.flatnonzero(inside)
    return int(HORIZONS[hits[0]]) if hits.size else None


def _stable_entry(curve: np.ndarray, catalog: float, band: float) -> int | None:
    inside = np.abs(curve - catalog) <= band
    inside = np.where(np.isfinite(curve), inside, False)
    for index in range(len(curve)):
        if bool(np.all(inside[index:])):
            return int(HORIZONS[index])
    return None


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float(np.sum(rx**2) * np.sum(ry**2)))
    return float(np.sum(rx * ry) / denom) if denom > 0 else float("nan")


def objective_table(
    event_rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    per_event: list[dict[str, Any]] = []
    per_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        curves = _event_curves(event_rows, method)
        events = list(curves["__events__"])
        catalogs = curves["__catalog__"]
        rows_m: list[dict[str, Any]] = []
        for event, catalog in zip(events, catalogs):
            c = curves[event]
            late = c[119:]
            late = late[np.isfinite(late)]
            rows_m.append(
                {
                    "method": method,
                    "event": event,
                    "mw_catalog": float(catalog),
                    "mw_1s": float(c[0]),
                    "mw_30s": float(c[29]),
                    "mw_60s": float(c[59]),
                    "mw_120s": float(c[119]),
                    "mw_200s": float(c[199]),
                    "error_200s": float(c[199] - catalog),
                    "first_entry_pm0p3_sec": _first_entry(c, float(catalog), BAND_MW),
                    "stable_entry_pm0p3_sec": _stable_entry(c, float(catalog), BAND_MW),
                    "sign_changes": _sign_changes(c),
                    "mean_abs_step_mw_per_sec": float(np.nanmean(np.abs(np.diff(c)))),
                    "post120_range_mw": float(late.max() - late.min()) if late.size else float("nan"),
                    "post120_max_abs_step_mw": float(np.max(np.abs(np.diff(late)))) if late.size > 1 else 0.0,
                    "early_overshoot_max_mw": (
                        float(np.nanmax(c[:30] - catalog)) if np.any(np.isfinite(c[:30])) else float("nan")
                    ),
                    "first_defined_sec": (
                        int(HORIZONS[np.flatnonzero(np.isfinite(c))[0]]) if np.any(np.isfinite(c)) else None
                    ),
                }
            )
        per_event.extend(rows_m)
        errors_200 = np.asarray([r["error_200s"] for r in rows_m])
        entries = [r["stable_entry_pm0p3_sec"] for r in rows_m]
        per_method[method] = {
            "event_count": len(rows_m),
            "event_mae_200s": float(np.mean(np.abs(errors_200))),
            "event_rmse_200s": float(np.sqrt(np.mean(errors_200**2))),
            "event_bias_200s": float(np.mean(errors_200)),
            "mean_mw_1s": float(np.nanmean([r["mw_1s"] for r in rows_m])),
            "std_mw_1s_across_events": float(np.nanstd([r["mw_1s"] for r in rows_m])),
            "mean_sign_changes": float(np.mean([r["sign_changes"] for r in rows_m])),
            "mean_abs_step_mw_per_sec": float(np.nanmean([r["mean_abs_step_mw_per_sec"] for r in rows_m])),
            "mean_post120_range_mw": float(np.nanmean([r["post120_range_mw"] for r in rows_m])),
            "max_post120_max_abs_step_mw": float(np.nanmax([r["post120_max_abs_step_mw"] for r in rows_m])),
            "events_with_stable_entry": int(sum(e is not None for e in entries)),
            "median_stable_entry_sec": (
                float(np.median([e for e in entries if e is not None]))
                if any(e is not None for e in entries)
                else None
            ),
            "events_overshooting_catalog_at_1s": int(sum(r["mw_1s"] > r["mw_catalog"] for r in rows_m)),
            **{
                f"spearman_vs_catalog_{h:03d}s": _spearman(
                    np.asarray([curves[e][h - 1] for e in events]), catalogs
                )
                for h in RANK_HORIZONS
            },
        }
    return per_event, per_method


def shape_cdf_diagnostic(
    stf_sink: Mapping[str, Any],
    station_rows: Sequence[Mapping[str, Any]],
    *,
    horizons: Sequence[int] = (10, 20, 30, 60),
) -> list[dict[str, Any]]:
    """Fraction of predicted moment inside the constrained window, per event.

    If this fraction barely varies across events at a given horizon, the shape
    head is acting as a fixed template and B's rise carries no waveform
    information.
    """
    stf = stf_sink["stf_over_m_ref"].astype(np.float32)  # (stations, horizons, steps); scale-free here
    keys = stf_sink["keys"]
    horizon_index = {int(h): i for i, h in enumerate(stf_sink["horizons"])}
    window_lookup: dict[tuple[str, str, int], float] = {}
    for row in station_rows:
        if row["method"] == RELEASED_METHOD:
            window_lookup[(str(row["event"]), str(row["station"]), int(row["observation_horizon_sec"]))] = float(
                row["constrained_window_sec"]
            )
    rows: list[dict[str, Any]] = []
    for h in horizons:
        if h not in horizon_index:
            continue
        per_event: dict[str, list[float]] = {}
        for s_index, (event, station) in enumerate(keys):
            window = window_lookup.get((event, station, h))
            if window is None or window < 1.0:
                continue
            rate = np.clip(stf[s_index, horizon_index[h]], 0.0, None)
            total = float(rate.sum())
            if total <= 0.0:
                continue
            k = int(math.floor(window))
            frac = float(rate[:k].sum() + (window - k) * (rate[k] if k < rate.size else 0.0)) / total
            per_event.setdefault(event, []).append(frac)
        for event, values in sorted(per_event.items()):
            rows.append(
                {
                    "observation_horizon_sec": int(h),
                    "event": event,
                    "n_stations_constrained": len(values),
                    "median_moment_fraction_inside_window": float(np.median(values)),
                }
            )
    return rows


# ------------------------------------------------------------------------------ plot


def plot_trajectories(
    event_rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    title: str,
    methods: Sequence[str],
) -> None:
    curves = {m: _event_curves(event_rows, m) for m in methods}
    events = list(curves[RELEASED_METHOD]["__events__"])
    catalogs = dict(zip(events, curves[RELEASED_METHOD]["__catalog__"]))
    n = len(events)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.3 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, event in zip(axes, events):
        for method in methods:
            c = curves[method].get(event)
            if c is None:
                continue
            ax.plot(HORIZONS, c, label=METHOD_LABELS.get(method, method), **METHOD_STYLES.get(method, {}))
        ax.axhline(catalogs[event], color="k", ls=":", lw=1.0, label="catalog Mw")
        ax.axhspan(catalogs[event] - BAND_MW, catalogs[event] + BAND_MW, color="k", alpha=0.05)
        ax.set_title(f"{event} (Mw {catalogs[event]:.2f})", fontsize=10)
        ax.set_ylim(4.5, 9.5)
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[max(0, n - cols):n]:
        ax.set_xlabel("observation horizon (s since origin)")
    for ax in axes[::cols]:
        ax.set_ylabel("event-median Mw")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(output_path, dpi=160)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


# -------------------------------------------------------------------------- pipeline


def _load_cohort(config: dict[str, Any], expected_assignment: str, cohort: str):
    samples = [dict(s) for s in CorrectedEarthquakeDataset(copy.deepcopy(config)).samples]
    split, split_manifest = fixed.build_fixed_split(samples)
    if split_manifest["assignment_sha256"] != expected_assignment:
        raise ValueError("fixed split assignment changed")
    train_loader, validation_loader, test_loader, loader_manifest = get_data_loaders_v2(
        config, explicit_split=split
    )
    if loader_manifest["assignment_sha256"] != expected_assignment:
        raise ValueError("loader split assignment changed")
    del train_loader
    if cohort == "validation":
        del test_loader
        return validation_loader, [samples[i] for i in split.validation_indices], set(fixed.VALIDATION_EVENTS)
    del validation_loader
    return test_loader, [samples[i] for i in split.test_indices], set(fixed.TEST_EVENTS)


def run_replay(
    *,
    checkpoint_path: Path,
    config_path: Path,
    expected_assignment: str,
    cohort: str,
    prefix_presentation: str,
    floor_nm: float,
    output_root: Path,
    device: torch.device,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if cohort not in {"validation", "test"}:
        raise ValueError("cohort must be validation or test")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    config = direct._read_yaml(config_path)
    validate_config_on_startup(config)
    loader, cohort_samples, expected_events = _load_cohort(config, expected_assignment, cohort)
    if {str(s["event"]) for s in cohort_samples} != expected_events:
        raise ValueError(f"{cohort} cohort event identities changed")
    expected_count = fixed.EXPECTED_RECORD_COUNTS[cohort]
    if len(cohort_samples) != expected_count:
        raise ValueError(f"{cohort} record count changed: {len(cohort_samples)}")

    criterion = _build_stf_rate_criterion(config, device)
    model = PINNModel(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True), strict=True)
    stf_sink: dict[str, Any] = {}
    model_result = evaluate_released_horizons(
        model,
        criterion,
        config,
        loader,
        horizons=HORIZONS,
        presentation=prefix_presentation,
        floor_nm=floor_nm,
        station_horizons=ANCHOR_HORIZONS,
        stf_sink=stf_sink,
    )
    pgd_result = evaluate_causal_pgd(cohort_samples, expected_record_count=expected_count)

    horizon_rows = model_result["horizon_rows"] + pgd_result["horizon_rows"]
    event_rows = model_result["event_rows"] + pgd_result["event_rows"]
    station_rows = model_result["station_rows"] + pgd_result["endpoint_station_rows"]
    prefix = cohort
    horizon_path = output_root / f"{prefix}_horizon_metrics.csv"
    event_path = output_root / f"{prefix}_event_predictions.csv"
    station_path = output_root / f"{prefix}_anchor_station_predictions.csv"
    stf_path = output_root / f"{prefix}_prefix_stf.npz"
    direct._write_csv(horizon_path, horizon_rows)
    direct._write_csv(event_path, event_rows)
    direct._write_csv(station_path, station_rows)
    np.savez_compressed(
        stf_path,
        stf_over_m_ref=stf_sink["stf_over_m_ref"],
        m_ref_nm=np.asarray(stf_sink["m_ref_nm"]),
        events=np.asarray([k[0] for k in stf_sink["keys"]]),
        stations=np.asarray([k[1] for k in stf_sink["keys"]]),
        horizons_sec=np.asarray(stf_sink["horizons"], dtype=np.int16),
    )

    all_methods = (*MODEL_METHODS, *PGD_METHODS)
    per_event, per_method = objective_table(event_rows, all_methods)
    direct._write_csv(output_root / f"{prefix}_objective_per_event.csv", per_event)
    cdf_rows = shape_cdf_diagnostic(stf_sink, model_result["station_rows"])
    if cdf_rows:
        direct._write_csv(output_root / f"{prefix}_shape_cdf_diagnostic.csv", cdf_rows)
    trajectories = {
        m: direct.summarize_trajectory(
            [r for r in horizon_rows if r["method"] == m],
            [r for r in event_rows if r["method"] == m],
            endpoint_band_tolerance_mw=0.05,
        )
        for m in all_methods
    }
    figure_path = output_root / f"{prefix}_released_vs_final_vs_pgd.png"
    plot_trajectories(
        event_rows,
        output_path=figure_path,
        title=(
            f"{cohort} cohort: released (B) vs final (A) vs cumulative PGD -- "
            f"{provenance.get('label', 'checkpoint')}"
        ),
        methods=(
            RELEASED_METHOD,
            RELEASED_CONSTRAINED_METHOD,
            FINAL_METHOD,
            RELEASED_REFERENCE_CONSTRAINED_METHOD,
            "crowell",
        ),
    )

    summary = {
        "status": "complete",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": cohort,
        "inference_only": True,
        "training_or_selection_performed": False,
        "method_frozen_from_validation": cohort == "test",
        "event_count": len(expected_events),
        "station_count": expected_count,
        "split_assignment_sha256": expected_assignment,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "prefix_presentation": prefix_presentation,
        "released_moment_floor_nm": floor_nm,
        "released_moment_definition": (
            "B(h) = Mw(sum_k clamp(STF_k,0) * m_k * dt), m_k = clip((h - tau_P)/dt - k, 0, 1), "
            "tau_P = hypocentral_distance / alpha; A(h) = full 200 s integral"
        ),
        "provenance": dict(provenance),
        "trajectory": trajectories,
        "objective_by_method": per_method,
        "artifacts": {
            "horizon_metrics": str(horizon_path),
            "horizon_metrics_sha256": sha256_file(horizon_path),
            "event_predictions": str(event_path),
            "event_predictions_sha256": sha256_file(event_path),
            "anchor_station_predictions": str(station_path),
            "prefix_stf": str(stf_path),
            "prefix_stf_sha256": sha256_file(stf_path),
            "objective_per_event": str(output_root / f"{prefix}_objective_per_event.csv"),
            "shape_cdf_diagnostic": str(output_root / f"{prefix}_shape_cdf_diagnostic.csv"),
            "figure": str(figure_path),
        },
    }
    summary = _nan_to_none(summary)
    direct._write_json(output_root / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="training run directory containing summary.json and protocol.json",
    )
    parser.add_argument("--cohort", choices=("validation", "test"), required=True)
    parser.add_argument(
        "--prefix-presentation",
        choices=("zero_pad", "truncate"),
        default=None,
        help="defaults to the run's protocol (truncate for the older causal runs)",
    )
    parser.add_argument("--floor-nm", type=float, default=1.0e15)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    run_root = args.run_root.resolve()
    summary = direct._read_json(run_root / "summary.json")
    protocol = direct._read_json(run_root / "protocol.json")
    if summary.get("status") != "complete":
        raise ValueError("training run is incomplete")
    if summary.get("held_out_test_loader_iterated") is not False:
        raise ValueError("training run iterated the test loader; refusing to replay")
    if protocol.get("development_role") != "fixed_validation_only":
        raise ValueError("checkpoint was not selected on fixed validation only")
    checkpoint_path = Path(str(summary["checkpoint_path"])).resolve()
    if sha256_file(checkpoint_path) != str(summary["checkpoint_sha256"]):
        raise ValueError("checkpoint hash changed")
    experiment = protocol["experiment_config"]
    endpoint_summary = direct._read_json(Path(str(experiment["fixed_endpoint_candidate_summary"])).resolve())
    config_path = Path(str(endpoint_summary["config_snapshot_path"])).resolve()
    if sha256_file(config_path) != str(endpoint_summary["config_snapshot_sha256"]):
        raise ValueError("source config hash changed")
    presentation = args.prefix_presentation or str(protocol.get("prefix_presentation", "truncate"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root or (
        Path("/home/lihe/PINN_Mag/runs") / f"{run_root.name}-released-replay-{args.cohort}-{timestamp}"
    )
    result = run_replay(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        expected_assignment=str(experiment["expected_split_assignment_sha256"]),
        cohort=args.cohort,
        prefix_presentation=presentation,
        floor_nm=float(args.floor_nm),
        output_root=output_root.resolve(),
        device=device,
        provenance={
            "run_root": str(run_root),
            "label": args.label or run_root.name,
            "training_method": protocol.get("method"),
            "best_epoch": summary.get("best_epoch"),
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
