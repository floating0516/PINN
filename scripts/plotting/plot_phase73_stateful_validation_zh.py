#!/usr/bin/env python3
"""Publish the Phase73 train/validation stateful-streaming report in Chinese."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")

from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import (  # noqa: E402
    run_phase67_pgd_guided_stateful as campaign,
)
from scripts.experiments.build_phase67_crowell_hint_cache import (  # noqa: E402
    load_hint_cache,
)
from scripts.experiments.run_phase43_streaming_adapter import (  # noqa: E402
    HORIZONS,
    load_cache,
)
from scripts.experiments.run_phase73_endpoint_teacher_weight2 import (  # noqa: E402
    configure_phase73,
)
from src.baseline.scaling_laws import (  # noqa: E402
    AVAILABLE_SCALING_LAWS,
    ScalingLawSpec,
)
from src.training.loss_stf_rate_v2 import (  # noqa: E402
    moment_magnitude_from_rate,
)
from src.utils.provenance import (  # noqa: E402
    current_git_commit,
    git_is_dirty,
    sha256_file,
)


PROJECT_HOME = REPO_ROOT.parent.parent
DEFAULT_RUN_ROOT = (
    PROJECT_HOME
    / "runs"
    / "phase73-teacher-weight2-stateful-20260730T154118Z-804bf96"
)
DEFAULT_CACHE_ROOT = (
    PROJECT_HOME
    / "runs"
    / "phase43-streaming-prefix-cache-20260728T115851Z-7bc63eb"
)
DEFAULT_HINT_CACHE_ROOT = (
    PROJECT_HOME
    / "runs"
    / "phase67-crowell-hint-cache-20260730T110514Z-7e4f271"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "docs"
    / "reports"
    / "phase73-pgd-guided-stateful-validation-zh"
)
TEST_PATH = REPO_ROOT / "tests" / "test_plot_phase73_stateful_validation_zh.py"

SELECTED_SEED = 17
SELECTED_EPOCH = 27
SELECTED_HORIZONS = (30, 60, 90, 120, 200)
PROCESSING_DELAY_SEC = 6.0
EXPECTED_TRAIN_COUNT = 1_788
EXPECTED_VALIDATION_COUNT = 385
EXPECTED_TRAIN_EVENT_COUNT = 31
EXPECTED_VALIDATION_EVENT_COUNT = 30
SOURCE_COMMIT = "804bf9657317d3584c3722ede4515592c11667d6"
METHOD_ORDER = ("phase73", "phase39", "crowell", "melgar", "ruhl")
METHOD_LABELS = {
    "phase73": "Phase73 有状态模型",
    "phase39": "Phase39 原始提案",
    "crowell": "Crowell PGD",
    "melgar": "Melgar PGD",
    "ruhl": "Ruhl PGD",
}
SPLIT_LABELS = {"train": "训练集", "validation": "Validation"}
TARGET_BAND_MW = 0.15
REFERENCE_BAND_MW = 0.30
COLORS = {
    "phase73": "#0072B2",
    "phase39": "#66717E",
    "crowell": "#D55E00",
    "melgar": "#CC79A7",
    "ruhl": "#009E73",
    "station": "#66717E",
    "outlier": "#D55E00",
    "identity": "#20262E",
    "target": "#8E5AA9",
    "band": "#009E73",
    "grid": "#D8DEE6",
    "ink": "#20262E",
    "yellow": "#E69F00",
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


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    handle_context = (
        gzip.open(path, mode="wt", encoding="utf-8", newline="")
        if path.suffix == ".gz"
        else path.open(mode="w", encoding="utf-8", newline="")
    )
    with handle_context as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for source in rows:
            writer.writerow(
                {
                    key: "" if source.get(key) is None else source.get(key)
                    for key in fieldnames
                }
            )


def _configure_plotting() -> None:
    font_candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/arphic/ukai.ttc"),
    )
    family = "DejaVu Sans"
    for path in font_candidates:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            break
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.titleweight": "normal",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axis(axis: Any) -> None:
    axis.grid(True, color=COLORS["grid"], linewidth=0.7)
    axis.set_axisbelow(True)


def _save_figure(figure: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [png, pdf]


def prediction_metrics(
    catalog: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    reference = np.asarray(catalog, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if reference.ndim != 1 or reference.shape != estimate.shape:
        raise ValueError("catalog and prediction must be aligned vectors")
    if reference.size == 0 or not np.all(np.isfinite(reference)):
        raise ValueError("catalog must be finite and nonempty")
    if not np.all(np.isfinite(estimate)):
        raise ValueError("prediction must be finite")
    error = estimate - reference
    pearson: float | None = None
    if reference.size > 1 and np.std(reference) > 0.0 and np.std(estimate) > 0.0:
        pearson = float(np.corrcoef(reference, estimate)[0, 1])
    return {
        "count": int(reference.size),
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(np.square(error)))),
        "bias_mw": float(np.mean(error)),
        "pearson_r": pearson,
        "within_0_15_count": int(np.count_nonzero(np.abs(error) <= 0.15)),
        "within_0_15_fraction": float(np.mean(np.abs(error) <= 0.15)),
        "within_0_30_count": int(np.count_nonzero(np.abs(error) <= 0.30)),
        "within_0_30_fraction": float(np.mean(np.abs(error) <= 0.30)),
    }


def _scaling_law_cube(
    pgd_3d_m: np.ndarray,
    source_distance_m: np.ndarray,
    spec: ScalingLawSpec,
) -> np.ndarray:
    pgd = np.asarray(pgd_3d_m, dtype=np.float64)
    distance_km = np.asarray(source_distance_m, dtype=np.float64).reshape(-1, 1) / 1_000.0
    if pgd.ndim != 2 or distance_km.shape[0] != pgd.shape[0]:
        raise ValueError("PGD and distance arrays are not aligned")
    if np.any(pgd <= 0.0) or np.any(distance_km <= 0.0):
        raise ValueError("PGD and distance must be positive")
    pgd_value = pgd if spec.pgd_unit == "m" else pgd * 100.0
    denominator = spec.b + spec.c * np.log10(distance_km)
    return ((np.log10(pgd_value) - spec.a) / denominator).astype(np.float32)


def _event_cubes(
    method_cubes: Mapping[str, np.ndarray],
    *,
    events: Sequence[str],
    catalogs: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    event_array = np.asarray([str(event) for event in events])
    event_names = sorted(set(event_array))
    event_catalogs: list[float] = []
    station_counts: list[int] = []
    output = {method: [] for method in method_cubes}
    for event in event_names:
        mask = event_array == event
        event_catalogs.append(float(np.median(catalogs[mask])))
        station_counts.append(int(np.count_nonzero(mask)))
        for method, cube in method_cubes.items():
            output[method].append(np.median(cube[mask], axis=0))
    return (
        event_names,
        np.asarray(event_catalogs, dtype=np.float64),
        np.asarray(station_counts, dtype=np.int64),
        {method: np.stack(values) for method, values in output.items()},
    )


def build_event_rows(
    *,
    split: str,
    event_names: Sequence[str],
    event_catalogs: np.ndarray,
    station_counts: np.ndarray,
    event_cubes: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(event_names):
        catalog = float(event_catalogs[event_index])
        for horizon_index, horizon in enumerate(HORIZONS):
            row: dict[str, Any] = {
                "split": split,
                "observation_horizon_sec": int(horizon),
                "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                "event": str(event),
                "mw_catalog": catalog,
                "station_count": int(station_counts[event_index]),
            }
            for method in METHOD_ORDER:
                prediction = float(event_cubes[method][event_index, horizon_index])
                row[f"{method}_mw_pred_median"] = prediction
                row[f"{method}_error_mw"] = prediction - catalog
                row[f"{method}_abs_error_mw"] = abs(prediction - catalog)
            rows.append(row)
    return rows


def _horizon_metric_rows(
    *,
    split: str,
    catalogs: np.ndarray,
    method_cubes: Mapping[str, np.ndarray],
    event_catalogs: np.ndarray,
    event_cubes: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon_index, horizon in enumerate(HORIZONS):
        for method in METHOD_ORDER:
            station = prediction_metrics(
                catalogs,
                method_cubes[method][:, horizon_index],
            )
            event = prediction_metrics(
                event_catalogs,
                event_cubes[method][:, horizon_index],
            )
            rows.append(
                {
                    "split": split,
                    "observation_horizon_sec": int(horizon),
                    "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    **{f"station_{key}": value for key, value in station.items()},
                    **{f"event_{key}": value for key, value in event.items()},
                }
            )
    return rows


def _stable_horizon(curve: np.ndarray, reference: float, tolerance: float) -> int | None:
    within = np.abs(np.asarray(curve, dtype=np.float64) - float(reference)) <= tolerance
    for index in range(within.size):
        if bool(np.all(within[index:])):
            return int(HORIZONS[index])
    return None


def trajectory_diagnostics(
    event_rows: Sequence[Mapping[str, Any]],
    *,
    split: str = "validation",
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        if str(row["split"]) == split:
            grouped[str(row["event"])].append(row)
    output: list[dict[str, Any]] = []
    horizons = np.asarray(HORIZONS)
    post120_mask = horizons >= 120
    post160_mask = horizons >= 160
    for event, source_rows in sorted(grouped.items()):
        source_rows = sorted(
            source_rows,
            key=lambda row: int(row["observation_horizon_sec"]),
        )
        phase73 = np.asarray(
            [float(row["phase73_mw_pred_median"]) for row in source_rows]
        )
        crowell = np.asarray(
            [float(row["crowell_mw_pred_median"]) for row in source_rows]
        )
        catalog = float(source_rows[0]["mw_catalog"])
        post120 = phase73[post120_mask]
        post120_steps = np.diff(post120)
        total_variation = float(np.sum(np.abs(post120_steps)))
        net_change = float(abs(post120[-1] - post120[0]))
        active = np.sign(post120_steps[np.abs(post120_steps) >= 0.005])
        sign_changes = int(np.count_nonzero(active[1:] != active[:-1])) if active.size > 1 else 0
        output.append(
            {
                "event": event,
                "mw_catalog": catalog,
                "station_count": int(source_rows[0]["station_count"]),
                "phase73_20s_mw": float(phase73[0]),
                "phase73_200s_mw": float(phase73[-1]),
                "phase73_endpoint_error_mw": float(phase73[-1] - catalog),
                "phase73_endpoint_abs_error_mw": float(abs(phase73[-1] - catalog)),
                "phase73_start_to_end_change_mw": float(phase73[-1] - phase73[0]),
                "phase73_post120_abs_step_p95_mw": float(
                    np.percentile(np.abs(post120_steps), 95)
                ),
                "phase73_post120_abs_step_max_mw": float(
                    np.max(np.abs(post120_steps))
                ),
                "phase73_post120_excess_variation_mw": total_variation - net_change,
                "phase73_post120_sign_changes": sign_changes,
                "phase73_peak_to_final_mw": float(np.max(post120) - post120[-1]),
                "phase73_post160_band_width_mw": float(
                    np.max(phase73[post160_mask]) - np.min(phase73[post160_mask])
                ),
                "phase73_suffix_within_0_15_horizon_sec": _stable_horizon(
                    phase73, catalog, TARGET_BAND_MW
                ),
                "crowell_suffix_within_0_15_horizon_sec": _stable_horizon(
                    crowell, catalog, TARGET_BAND_MW
                ),
                "phase73_plateau_within_0_15_of_final_horizon_sec": _stable_horizon(
                    phase73, float(phase73[-1]), TARGET_BAND_MW
                ),
            }
        )
    return output


def _load_selected_source(
    *,
    run_root: Path,
    cache_root: Path,
    hint_cache_root: Path,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any], torch.nn.Module, dict[str, Any]]:
    campaign_summary_path = run_root / "campaign_summary.json"
    campaign_summary = _read_json(campaign_summary_path)
    if campaign_summary.get("phase") != "Phase73":
        raise ValueError("source campaign is not Phase73")
    if campaign_summary.get("internal_test_iterated") is not False:
        raise ValueError("Phase73 campaign unexpectedly opened internal test")
    if campaign_summary.get("external_data_loaded") is not False:
        raise ValueError("Phase73 campaign unexpectedly opened external data")
    if campaign_summary.get("grouped_test_loaded") is not False:
        raise ValueError("Phase73 campaign unexpectedly opened grouped test")
    seed_summary = next(
        item
        for item in campaign_summary["seed_summaries"]
        if int(item["seed"]) == SELECTED_SEED
    )
    if int(seed_summary["closest_epoch"]) != SELECTED_EPOCH:
        raise ValueError("Phase73 selected epoch changed")
    if seed_summary["provenance"]["git_commit"] != SOURCE_COMMIT:
        raise ValueError("Phase73 source commit changed")
    for key in ("internal_test_iterated", "external_data_loaded", "grouped_test_loaded"):
        if seed_summary["provenance"].get(key) is not False:
            raise ValueError(f"Phase73 hidden-data flag changed: {key}")

    configure_phase73()
    cache = load_cache(cache_root)
    hints = load_hint_cache(hint_cache_root, phase43_cache=cache)
    config = campaign.phase67_config()
    protocol = seed_summary["protocol"]
    if config["model"]["stateful_streaming"]["proposal_assimilation_scale"] != 1.0:
        raise ValueError("Phase73 assimilation scale changed")
    if protocol["loss_weights"]["endpoint_teacher"] != 2.0:
        raise ValueError("Phase73 endpoint-teacher weight changed")
    if protocol["loss_weights"]["plateau_band"] != 0.25:
        raise ValueError("Phase73 plateau-band weight changed")
    if protocol["stateful_total_moment"]["late_proposal_assimilation"]["start_sec"] != 140:
        raise ValueError("Phase73 assimilation start changed")

    model, frozen_source = campaign.load_phase67_model(config, device=device)
    checkpoint_path = Path(seed_summary["closest_model"]["path"])
    if checkpoint_path != run_root / f"seed_{SELECTED_SEED}" / "closest_model.pth":
        raise ValueError("Phase73 checkpoint path changed")
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != seed_summary["closest_model"]["sha256"]:
        raise ValueError("Phase73 checkpoint hash changed")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    campaign._assert_backbone_unchanged(model, frozen_source)
    model.eval()
    source = {
        "campaign_summary_path": str(campaign_summary_path),
        "campaign_summary_sha256": sha256_file(campaign_summary_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "cache_root": str(cache_root),
        "cache_raw_rates_sha256": cache.manifest["raw_rates_sha256"],
        "cache_arrays_sha256": cache.manifest["arrays_sha256"],
        "hint_cache_root": str(hint_cache_root),
        "hint_crowell_mw_sha256": hints.manifest["outputs"]["crowell_mw.npy"],
        "hint_pgd_3d_sha256": hints.manifest["outputs"]["pgd_3d_m.npy"],
        "split_assignment_sha256": cache.manifest["split_assignment_sha256"],
    }
    return cache, hints, config, model, source


def replay_phase73(
    *,
    run_root: Path,
    cache_root: Path,
    hint_cache_root: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    cache, hints, config, model, source = _load_selected_source(
        run_root=run_root,
        cache_root=cache_root,
        hint_cache_root=hint_cache_root,
        device=device,
    )
    split_specs = (("train", 0, EXPECTED_TRAIN_COUNT), ("validation", 1, EXPECTED_VALIDATION_COUNT))
    event_rows: list[dict[str, Any]] = []
    selected_station_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    split_payloads: dict[str, Any] = {}
    selected_indices_by_horizon = {horizon: HORIZONS.index(horizon) for horizon in SELECTED_HORIZONS}

    for split, split_code, expected_count in split_specs:
        indices = np.flatnonzero(cache.arrays["split_code"] == split_code)
        if len(indices) != expected_count:
            raise ValueError(f"Phase73 {split} count changed")
        phase73_batches: list[np.ndarray] = []
        phase39_batches: list[np.ndarray] = []
        confidence_batches: list[np.ndarray] = []
        revision_batches: list[np.ndarray] = []
        assimilation_batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                batch = campaign._tensor_batch_with_hints(
                    cache,
                    hints,
                    batch_indices,
                    device=device,
                )
                _, _, state_mw, _, diagnostics = model.stream_sequence_from_rates(
                    batch["raw_rate"],
                    horizons_sec=HORIZONS,
                    source_distance_m=batch["source_distance_m"],
                    source_dt_sec=batch["source_dt_sec"],
                    beta_m_per_s=float(config["physics"]["beta"]),
                    pgd_mw_hint=batch["pgd_mw"],
                    pgd_valid_hint=batch["pgd_valid"],
                    return_diagnostics=True,
                )
                raw_rate = batch["raw_rate"]
                raw_mw = moment_magnitude_from_rate(
                    raw_rate.reshape(-1, raw_rate.shape[-1]),
                    batch["source_dt_sec"]
                    .reshape(-1, 1)
                    .expand(-1, raw_rate.shape[1])
                    .reshape(-1),
                ).reshape(raw_rate.shape[0], raw_rate.shape[1])
                phase73_batches.append(state_mw.cpu().numpy().astype(np.float32))
                phase39_batches.append(raw_mw.cpu().numpy().astype(np.float32))
                confidence_batches.append(
                    diagnostics["plateau_confidence"].cpu().numpy().astype(np.float32)
                )
                revision_batches.append(
                    diagnostics["revision_mw"].cpu().numpy().astype(np.float32)
                )
                assimilation_batches.append(
                    diagnostics["proposal_assimilation_mw"].cpu().numpy().astype(np.float32)
                )

        phase73 = np.concatenate(phase73_batches, axis=0)
        phase39 = np.concatenate(phase39_batches, axis=0)
        confidence = np.concatenate(confidence_batches, axis=0)
        revision = np.concatenate(revision_batches, axis=0)
        assimilation = np.concatenate(assimilation_batches, axis=0)
        crowell = np.asarray(hints.crowell_mw[indices], dtype=np.float32)
        pgd_3d_m = np.asarray(hints.pgd_3d_m[indices], dtype=np.float32)
        source_distance_m = np.asarray(cache.arrays["source_distance_m"][indices], dtype=np.float64)
        melgar = _scaling_law_cube(
            pgd_3d_m,
            source_distance_m,
            AVAILABLE_SCALING_LAWS["melgar"],
        )
        ruhl = _scaling_law_cube(
            pgd_3d_m,
            source_distance_m,
            AVAILABLE_SCALING_LAWS["ruhl"],
        )
        crowell_recomputed = _scaling_law_cube(
            pgd_3d_m,
            source_distance_m,
            AVAILABLE_SCALING_LAWS["crowell"],
        )
        max_crowell_difference = float(np.max(np.abs(crowell - crowell_recomputed)))
        if max_crowell_difference > 5.0e-6:
            raise ValueError(
                f"cached/recomputed Crowell mismatch: {max_crowell_difference}"
            )
        method_cubes = {
            "phase73": phase73,
            "phase39": phase39,
            "crowell": crowell,
            "melgar": melgar,
            "ruhl": ruhl,
        }
        records = [cache.records[int(index)] for index in indices]
        events = [str(row["event"]) for row in records]
        stations = [str(row["station"]) for row in records]
        catalogs = np.asarray(cache.arrays["magnitude_catalog"][indices], dtype=np.float64)
        event_names, event_catalogs, station_counts, event_cubes = _event_cubes(
            method_cubes,
            events=events,
            catalogs=catalogs,
        )
        expected_events = EXPECTED_TRAIN_EVENT_COUNT if split == "train" else EXPECTED_VALIDATION_EVENT_COUNT
        if len(event_names) != expected_events:
            raise ValueError(f"Phase73 {split} event count changed")
        event_rows.extend(
            build_event_rows(
                split=split,
                event_names=event_names,
                event_catalogs=event_catalogs,
                station_counts=station_counts,
                event_cubes=event_cubes,
            )
        )
        metric_rows.extend(
            _horizon_metric_rows(
                split=split,
                catalogs=catalogs,
                method_cubes=method_cubes,
                event_catalogs=event_catalogs,
                event_cubes=event_cubes,
            )
        )
        for station_index, (event, station) in enumerate(zip(events, stations, strict=True)):
            for horizon, horizon_index in selected_indices_by_horizon.items():
                row = {
                    "split": split,
                    "observation_horizon_sec": horizon,
                    "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                    "event": event,
                    "station": station,
                    "mw_catalog": float(catalogs[station_index]),
                    "plateau_confidence": float(confidence[station_index, horizon_index]),
                    "revision_mw": float(revision[station_index, horizon_index]),
                    "proposal_assimilation_mw": float(assimilation[station_index, horizon_index]),
                }
                for method in METHOD_ORDER:
                    prediction = float(method_cubes[method][station_index, horizon_index])
                    row[f"{method}_mw_pred"] = prediction
                    row[f"{method}_error_mw"] = prediction - float(catalogs[station_index])
                selected_station_rows.append(row)
        split_payloads[split] = {
            "indices": indices,
            "events": events,
            "stations": stations,
            "catalogs": catalogs,
            "method_cubes": method_cubes,
            "event_names": event_names,
            "event_catalogs": event_catalogs,
            "station_counts": station_counts,
            "event_cubes": event_cubes,
            "max_crowell_recompute_abs_diff_mw": max_crowell_difference,
        }

    diagnostics = trajectory_diagnostics(event_rows)
    return {
        "source": source,
        "event_rows": event_rows,
        "selected_station_rows": selected_station_rows,
        "metric_rows": metric_rows,
        "trajectory_diagnostics": diagnostics,
        "splits": split_payloads,
    }


def _metric_lookup(
    metric_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    return {
        (
            str(row["split"]),
            int(row["observation_horizon_sec"]),
            str(row["method"]),
        ): row
        for row in metric_rows
    }


def _load_training_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in (17, 42, 73):
        for source in _read_csv(run_root / f"seed_{seed}" / "epoch_metrics.csv"):
            rows.append({"seed": seed, **source})
    return rows


def _float(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def plot_training_dynamics(
    rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
) -> list[Path]:
    _configure_plotting()
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.7), sharex=True)
    seed_colors = {17: COLORS["phase73"], 42: COLORS["crowell"], 73: COLORS["ruhl"]}
    specs = (
        ("train_total_normalized_loss", "A 训练集总归一化目标", "loss"),
        ("endpoint_event_mae", "B Validation Event MAE", "Mw"),
        ("event_post160_band_width_p95_mw", "C 160--200 秒平台宽度 p95", "Mw"),
        ("selection_score", "D 最差归一化 gate 比例（>2 截断）", "比例"),
    )
    for axis, (key, title, ylabel) in zip(axes.flat, specs, strict=True):
        for seed in (17, 42, 73):
            sequence = sorted(
                (row for row in rows if int(row["seed"]) == seed),
                key=lambda row: int(row["epoch"]),
            )
            epochs = [int(row["epoch"]) for row in sequence]
            values = [_float(row, key) for row in sequence]
            if key == "selection_score":
                values = [min(value, 2.0) for value in values]
            axis.plot(
                epochs,
                values,
                color=seed_colors[seed],
                linewidth=1.55,
                label=f"seed{seed}",
            )
        if key == "endpoint_event_mae":
            axis.axhline(0.1193335183, color=COLORS["target"], linestyle="--", linewidth=1.0, label="Phase39 gate")
        elif key == "event_post160_band_width_p95_mw":
            axis.axhline(0.30, color=COLORS["target"], linestyle="--", linewidth=1.0, label="0.30 gate")
        elif key == "selection_score":
            axis.axhline(1.0, color=COLORS["target"], linestyle="--", linewidth=1.0, label="全部 gate")
        selected = next(
            row
            for row in rows
            if int(row["seed"]) == SELECTED_SEED and int(row["epoch"]) == SELECTED_EPOCH
        )
        axis.scatter(
            [SELECTED_EPOCH],
            [min(_float(selected, key), 2.0) if key == "selection_score" else _float(selected, key)],
            marker="D",
            color=COLORS["yellow"],
            edgecolor="white",
            linewidth=0.8,
            s=62,
            zorder=5,
            label="报告 checkpoint" if key == "train_total_normalized_loss" else None,
        )
        axis.set_title(title)
        axis.set_ylabel("比例（>2 截断）" if key == "selection_score" else ylabel)
        axis.set_xlabel("epoch")
        _style_axis(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.965))
    figure.suptitle("Phase73 三个 seed：训练 loss 与 validation 选择不是同一件事", y=0.995, fontsize=14)
    figure.text(
        0.5,
        0.012,
        "报告采用 seed17 epoch27：PGD 与轨迹 gate 全通过；endpoint gate 未通过。",
        ha="center",
        color="#444444",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.90, hspace=0.26, wspace=0.18)
    return _save_figure(figure, figures_dir / "01_training_dynamics")


def plot_overall_metrics(
    metric_rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
) -> list[Path]:
    _configure_plotting()
    lookup = _metric_lookup(metric_rows)
    horizons = np.asarray(HORIZONS)
    figure, axes = plt.subplots(3, 1, figsize=(10.7, 11.8), sharex=True)
    displayed = ("phase73", "phase39", "crowell")
    for method in displayed:
        values = [
            float(lookup[("validation", horizon, method)]["event_mae_mw"])
            for horizon in HORIZONS
        ]
        axes[0].plot(horizons, values, color=COLORS[method], linewidth=1.8, label=METHOD_LABELS[method])
    axes[0].axhline(TARGET_BAND_MW, color=COLORS["target"], linestyle="--", linewidth=1.0, label="0.15 Mw")
    axes[0].set_ylabel("Event MAE (Mw)")
    axes[0].set_title("A  Validation 事件等权误差")
    axes[0].legend(frameon=False, ncol=2)
    _style_axis(axes[0])

    for method in displayed:
        values = [
            float(lookup[("validation", horizon, method)]["event_bias_mw"])
            for horizon in HORIZONS
        ]
        axes[1].plot(horizons, values, color=COLORS[method], linewidth=1.8, label=METHOD_LABELS[method])
    axes[1].axhline(0.0, color=COLORS["identity"], linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Event bias (Mw)")
    axes[1].set_title("B  Validation 事件有符号偏差")
    _style_axis(axes[1])

    for method in displayed:
        values = [
            int(lookup[("validation", horizon, method)]["event_within_0_15_count"])
            for horizon in HORIZONS
        ]
        axes[2].step(horizons, values, where="mid", color=COLORS[method], linewidth=1.8, label=METHOD_LABELS[method])
    axes[2].set_ylabel("|误差| ≤ 0.15 的事件数")
    axes[2].set_xlabel("震源时刻后的观测时长 (s)")
    axes[2].set_title("C  Validation 达标事件数（共 30 个事件）")
    axes[2].set_ylim(0, EXPECTED_VALIDATION_EVENT_COUNT + 1)
    _style_axis(axes[2])
    figure.suptitle("Phase73 有状态流式输出：20--200 秒 Validation 总体表现", y=0.995, fontsize=14)
    figure.text(
        0.5,
        0.012,
        "Phase73 每秒继承上一秒状态；Phase39 为当前时刻的完整 STF 提案；PGD 为因果 Crowell 提示。",
        ha="center",
        color="#444444",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.07, top=0.95, hspace=0.25)
    return _save_figure(figure, figures_dir / "02_overall_metrics")


def plot_event_trajectories(
    event_rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
) -> list[Path]:
    _configure_plotting()
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        if row["split"] == "validation":
            grouped[str(row["event"])].append(row)
    ordered = sorted(
        grouped,
        key=lambda event: (
            float(grouped[event][0]["mw_catalog"]),
            float(grouped[event][-1]["phase73_abs_error_mw"]),
            event,
        ),
    )
    figure, axes = plt.subplots(6, 5, figsize=(17.5, 19.5), sharex=True)
    for axis, event in zip(axes.flat, ordered, strict=True):
        rows = sorted(grouped[event], key=lambda row: int(row["observation_horizon_sec"]))
        horizons = [int(row["observation_horizon_sec"]) for row in rows]
        catalog = float(rows[0]["mw_catalog"])
        axis.axhspan(catalog - TARGET_BAND_MW, catalog + TARGET_BAND_MW, color=COLORS["band"], alpha=0.10, linewidth=0)
        axis.axhline(catalog, color=COLORS["identity"], linestyle="--", linewidth=0.9)
        for method, linewidth in (("phase39", 1.0), ("crowell", 1.0), ("phase73", 1.8)):
            axis.plot(
                horizons,
                [float(row[f"{method}_mw_pred_median"]) for row in rows],
                color=COLORS[method],
                linewidth=linewidth,
                alpha=0.78 if method != "phase73" else 1.0,
            )
        endpoint = float(rows[-1]["phase73_mw_pred_median"])
        axis.set_title(
            f"{event}  M{catalog:.1f}  200s={endpoint:.2f}",
            fontsize=8.5,
        )
        axis.set_xlim(HORIZONS[0], HORIZONS[-1])
        _style_axis(axis)
    legend = [
        Line2D([0], [0], color=COLORS["phase73"], linewidth=2.0, label="Phase73"),
        Line2D([0], [0], color=COLORS["phase39"], linewidth=1.2, label="Phase39 提案"),
        Line2D([0], [0], color=COLORS["crowell"], linewidth=1.2, label="Crowell PGD"),
        Line2D([0], [0], color=COLORS["identity"], linestyle="--", linewidth=1.0, label="目录 Mw"),
        Patch(facecolor=COLORS["band"], alpha=0.10, edgecolor="none", label="±0.15 Mw"),
    ]
    figure.legend(handles=legend, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.982))
    figure.suptitle("Phase73：30 个 Validation 事件的逐秒有状态轨迹", y=0.997, fontsize=14)
    figure.supxlabel("震源时刻后的观测时长 (s)", y=0.025)
    figure.supylabel("事件台站中位数 Mw", x=0.015)
    figure.text(
        0.5,
        0.008,
        "同一事件 validation 台站预测取中位数；这是 within_event_station validation，不是未见事件测试。",
        ha="center",
        color="#444444",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.055, right=0.99, bottom=0.055, top=0.95, hspace=0.34, wspace=0.18)
    return _save_figure(figure, figures_dir / "03_validation_event_trajectories")


def plot_trajectory_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
) -> list[Path]:
    _configure_plotting()
    endpoint_error = np.asarray([float(row["phase73_endpoint_abs_error_mw"]) for row in diagnostics])
    band = np.asarray([float(row["phase73_post160_band_width_mw"]) for row in diagnostics])
    counts = np.asarray([int(row["station_count"]) for row in diagnostics])
    peak_drop = np.asarray([float(row["phase73_peak_to_final_mw"]) for row in diagnostics])
    excess = np.asarray([float(row["phase73_post120_excess_variation_mw"]) for row in diagnostics])
    names = [str(row["event"]) for row in diagnostics]
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.1))

    sizes = 28.0 + 20.0 * np.sqrt(counts)
    axes[0, 0].scatter(endpoint_error, band, s=sizes, color=COLORS["phase73"], alpha=0.78, edgecolor="white", linewidth=0.7)
    axes[0, 0].axvline(0.15, color=COLORS["target"], linestyle="--", linewidth=1.0)
    axes[0, 0].axhline(0.30, color=COLORS["target"], linestyle="--", linewidth=1.0)
    for index in np.argsort(endpoint_error + band)[-5:]:
        axes[0, 0].annotate(names[index], (endpoint_error[index], band[index]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    axes[0, 0].set_xlabel("200 秒绝对误差 (Mw)")
    axes[0, 0].set_ylabel("160--200 秒平台宽度 (Mw)")
    axes[0, 0].set_title("A  单事件最终精度与平台稳定性")
    _style_axis(axes[0, 0])

    order = np.argsort(peak_drop + excess)
    positions = np.arange(len(order))
    axes[0, 1].barh(positions, excess[order], color=COLORS["crowell"], alpha=0.78, label="120 秒后多余变差")
    axes[0, 1].barh(positions, peak_drop[order], left=excess[order], color=COLORS["phase73"], alpha=0.78, label="峰值到最终回落")
    axes[0, 1].set_yticks(positions)
    axes[0, 1].set_yticklabels([names[index] for index in order], fontsize=7)
    axes[0, 1].set_xlabel("Mw")
    axes[0, 1].set_title("B  后期变差来源（逐事件）")
    axes[0, 1].legend(frameon=False)
    _style_axis(axes[0, 1])

    phase73_stable = [row["phase73_suffix_within_0_15_horizon_sec"] for row in diagnostics]
    crowell_stable = [row["crowell_suffix_within_0_15_horizon_sec"] for row in diagnostics]
    bins = np.arange(20, 211, 10)
    axes[1, 0].hist([value for value in phase73_stable if value is not None], bins=bins, color=COLORS["phase73"], alpha=0.68, label="Phase73")
    axes[1, 0].hist([value for value in crowell_stable if value is not None], bins=bins, histtype="step", linewidth=1.8, color=COLORS["crowell"], label="Crowell PGD")
    axes[1, 0].set_xlabel("进入后持续保持 ±0.15 的观测时长 (s)")
    axes[1, 0].set_ylabel("事件数")
    axes[1, 0].set_title("C  严格 suffix-stable 收敛时间")
    axes[1, 0].legend(frameon=False)
    _style_axis(axes[1, 0])

    validation_rows = [row for row in event_rows if row["split"] == "validation"]
    by_horizon: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        by_horizon[int(row["observation_horizon_sec"])].append(row)
    step_horizons = np.asarray(HORIZONS[1:])
    phase73_event_cube = np.asarray(
        [
            [float(row["phase73_mw_pred_median"]) for row in sorted(by_horizon[h], key=lambda item: str(item["event"]))]
            for h in HORIZONS
        ]
    ).T
    abs_steps = np.abs(np.diff(phase73_event_cube, axis=1))
    axes[1, 1].plot(step_horizons, np.median(abs_steps, axis=0), color=COLORS["phase73"], linewidth=1.8, label="中位数")
    axes[1, 1].plot(step_horizons, np.percentile(abs_steps, 95, axis=0), color=COLORS["crowell"], linewidth=1.5, label="p95")
    axes[1, 1].axhline(0.02, color=COLORS["target"], linestyle="--", linewidth=1.0, label="后期 p95 gate")
    axes[1, 1].axvline(120, color=COLORS["phase39"], linestyle=":", linewidth=1.0)
    axes[1, 1].set_xlabel("观测时长 (s)")
    axes[1, 1].set_ylabel("相邻秒 |ΔMw|")
    axes[1, 1].set_title("D  Validation 事件曲线逐秒变化")
    axes[1, 1].legend(frameon=False)
    _style_axis(axes[1, 1])

    figure.suptitle("Phase73 Validation：精度、平台与逐秒修正诊断", y=0.995, fontsize=14)
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.94, hspace=0.28, wspace=0.24)
    return _save_figure(figure, figures_dir / "04_validation_trajectory_diagnostics")


def _scatter_limits(rows: Sequence[Mapping[str, Any]], *, station: bool = True) -> tuple[float, float]:
    keys = ["mw_catalog"] + [f"{method}_mw_pred" for method in METHOD_ORDER] if station else ["mw_catalog"] + [f"{method}_mw_pred_median" for method in METHOD_ORDER]
    values = np.asarray([float(row[key]) for row in rows for key in keys], dtype=np.float64)
    lower = math.floor((float(np.min(values)) - 0.25) * 2.0) / 2.0
    upper = math.ceil((float(np.max(values)) + 0.25) * 2.0) / 2.0
    return lower, upper


def _identity_background(axis: Any, lower: float, upper: float) -> None:
    line = np.linspace(lower, upper, 300)
    axis.fill_between(line, line - REFERENCE_BAND_MW, line + REFERENCE_BAND_MW, color=COLORS["band"], alpha=0.09, linewidth=0)
    axis.plot(line, line, color=COLORS["identity"], linewidth=1.2)
    axis.plot(line, line + TARGET_BAND_MW, color=COLORS["target"], linestyle="--", linewidth=0.9)
    axis.plot(line, line - TARGET_BAND_MW, color=COLORS["target"], linestyle="--", linewidth=0.9)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    _style_axis(axis)


def _metrics_text(metrics: Mapping[str, Any], prefix: str) -> str:
    pearson = metrics.get(f"{prefix}_pearson_r")
    pearson_text = "NA" if pearson is None else f"{float(pearson):.3f}"
    label = "台站" if prefix == "station" else "事件中位数"
    return (
        f"{label} n={int(metrics[f'{prefix}_count'])}  MAE={float(metrics[f'{prefix}_mae_mw']):.3f}\n"
        f"RMSE={float(metrics[f'{prefix}_rmse_mw']):.3f}  bias={float(metrics[f'{prefix}_bias_mw']):+.3f}\n"
        f"r={pearson_text}  ±0.15={100.0 * float(metrics[f'{prefix}_within_0_15_fraction']):.1f}%"
    )


def plot_endpoint_scatter(
    station_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
) -> list[Path]:
    _configure_plotting()
    station = [row for row in station_rows if row["split"] == "validation" and int(row["observation_horizon_sec"]) == 200]
    events = [row for row in event_rows if row["split"] == "validation" and int(row["observation_horizon_sec"]) == 200]
    lookup = _metric_lookup(metric_rows)
    metrics = lookup[("validation", 200, "phase73")]
    lower, upper = _scatter_limits(station)
    figure, axes = plt.subplots(1, 2, figsize=(12.7, 6.0), sharex=True, sharey=True)
    axes[0].scatter(
        [float(row["mw_catalog"]) for row in station],
        [float(row["phase73_mw_pred"]) for row in station],
        s=20,
        color=COLORS["station"],
        alpha=0.34,
        edgecolors="none",
    )
    _identity_background(axes[0], lower, upper)
    axes[0].set_title("A  Validation 单台站")
    axes[0].set_xlabel("目录震级 Mw")
    axes[0].set_ylabel("Phase73 预测 Mw")
    axes[0].text(0.03, 0.97, _metrics_text(metrics, "station"), transform=axes[0].transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none"}, fontsize=9)

    event_catalog = np.asarray([float(row["mw_catalog"]) for row in events])
    event_prediction = np.asarray([float(row["phase73_mw_pred_median"]) for row in events])
    event_error = np.abs(event_prediction - event_catalog)
    event_sizes = np.asarray([45.0 + 18.0 * math.sqrt(int(row["station_count"])) for row in events])
    for mask, color, label in (
        (event_error <= 0.30, COLORS["phase73"], "|误差|≤0.30"),
        (event_error > 0.30, COLORS["outlier"], "|误差|>0.30"),
    ):
        axes[1].scatter(event_catalog[mask], event_prediction[mask], s=event_sizes[mask], marker="D", color=color, alpha=0.88, edgecolor="white", linewidth=0.8, label=label)
    _identity_background(axes[1], lower, upper)
    axes[1].set_title("B  Validation 事件台站中位数")
    axes[1].set_xlabel("目录震级 Mw")
    axes[1].set_ylabel("Phase73 预测 Mw")
    axes[1].text(0.03, 0.97, _metrics_text(metrics, "event"), transform=axes[1].transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none"}, fontsize=9)
    axes[1].legend(frameon=False, loc="lower right")
    figure.suptitle("Phase73：200 秒 Validation 最终预测", y=0.995, fontsize=14)
    figure.text(0.5, 0.012, "validation 与 train 含相同事件的不同台站；该图衡量同事件台站插值，不等于未见事件泛化。", ha="center", color="#444444", fontsize=9)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.92, wspace=0.16)
    return _save_figure(figure, figures_dir / "05_validation_endpoint_scatter")


def plot_horizon_scatter(
    station_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    figure_index: int,
    figures_dir: Path,
) -> list[Path]:
    _configure_plotting()
    selected_stations = [row for row in station_rows if int(row["observation_horizon_sec"]) == horizon]
    selected_events = [row for row in event_rows if int(row["observation_horizon_sec"]) == horizon]
    lower, upper = _scatter_limits(selected_stations)
    lookup = _metric_lookup(metric_rows)
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 6.4), sharex=True, sharey=True)
    for axis, split, panel in zip(axes, ("train", "validation"), ("A", "B"), strict=True):
        stations = [row for row in selected_stations if row["split"] == split]
        events = [row for row in selected_events if row["split"] == split]
        station_catalog = np.asarray([float(row["mw_catalog"]) for row in stations])
        station_prediction = np.asarray([float(row["phase73_mw_pred"]) for row in stations])
        event_catalog = np.asarray([float(row["mw_catalog"]) for row in events])
        event_prediction = np.asarray([float(row["phase73_mw_pred_median"]) for row in events])
        in_band = np.abs(event_prediction - event_catalog) <= 0.30
        axis.scatter(station_catalog, station_prediction, s=14 if split == "train" else 19, color=COLORS["station"], alpha=0.20 if split == "train" else 0.30, edgecolors="none")
        sizes = np.asarray([42.0 + 18.0 * math.sqrt(int(row["station_count"])) for row in events])
        axis.scatter(event_catalog[in_band], event_prediction[in_band], s=sizes[in_band], marker="D", color=COLORS["phase73"], alpha=0.86, edgecolor="white", linewidth=0.7)
        axis.scatter(event_catalog[~in_band], event_prediction[~in_band], s=sizes[~in_band], marker="D", color=COLORS["outlier"], alpha=0.90, edgecolor="white", linewidth=0.7)
        _identity_background(axis, lower, upper)
        metrics = lookup[(split, horizon, "phase73")]
        axis.set_title(f"{panel}  {SPLIT_LABELS[split]}（{int(metrics['station_count'])} 条记录，{int(metrics['event_count'])} 个事件）")
        axis.set_xlabel("目录震级 Mw")
        axis.set_ylabel("Phase73 预测 Mw")
        axis.text(0.03, 0.97, _metrics_text(metrics, "station") + "\n" + _metrics_text(metrics, "event"), transform=axis.transAxes, va="top", fontsize=8.3, bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none"})
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["station"], alpha=0.45, label="单台站预测"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["phase73"], markeredgecolor="white", label="事件中位数（|误差|≤0.30）"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["outlier"], markeredgecolor="white", label="事件中位数（|误差|>0.30）"),
        Line2D([0], [0], color=COLORS["identity"], linewidth=1.2, label="预测 = 目录 Mw"),
        Line2D([0], [0], color=COLORS["target"], linestyle="--", linewidth=0.9, label="±0.15 Mw"),
        Patch(facecolor=COLORS["band"], alpha=0.09, edgecolor="none", label="±0.30 Mw 区间"),
    ]
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False)
    figure.suptitle(f"Phase73 有状态流式预测：{horizon} 秒观测（{int(horizon + PROCESSING_DELAY_SEC)} 秒发布）", y=0.995, fontsize=14)
    figure.text(0.5, 0.018, "每个时刻继承上一秒 STF/Mw/GRU 状态；Validation 为 within_event_station，同一事件的未见台站插值。", ha="center", color="#444444", fontsize=9)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.82, wspace=0.16)
    return _save_figure(figure, figures_dir / f"{figure_index:02d}_train_validation_{horizon:03d}s")


def plot_method_comparison(
    event_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    figure_index: int,
    figures_dir: Path,
) -> list[Path]:
    _configure_plotting()
    selected = [row for row in event_rows if int(row["observation_horizon_sec"]) == horizon]
    lower, upper = _scatter_limits(selected, station=False)
    lookup = _metric_lookup(metric_rows)
    figure, axes = plt.subplots(2, len(METHOD_ORDER), figsize=(19.0, 8.3), sharex=True, sharey=True)
    for row_index, split in enumerate(("train", "validation")):
        split_rows = [row for row in selected if row["split"] == split]
        catalog = np.asarray([float(row["mw_catalog"]) for row in split_rows])
        for column_index, method in enumerate(METHOD_ORDER):
            axis = axes[row_index, column_index]
            prediction = np.asarray([float(row[f"{method}_mw_pred_median"]) for row in split_rows])
            in_band = np.abs(prediction - catalog) <= 0.30
            axis.scatter(catalog[in_band], prediction[in_band], s=30, color=COLORS[method], alpha=0.78, edgecolor="white", linewidth=0.5)
            axis.scatter(catalog[~in_band], prediction[~in_band], s=34, color=COLORS[method], marker="x", linewidth=1.4)
            _identity_background(axis, lower, upper)
            metrics = lookup[(split, horizon, method)]
            axis.set_title(
                f"{METHOD_LABELS[method]}\nMAE={float(metrics['event_mae_mw']):.3f}  bias={float(metrics['event_bias_mw']):+.3f}  ±0.15={100.0 * float(metrics['event_within_0_15_fraction']):.0f}%",
                fontsize=9,
            )
            if column_index == 0:
                axis.set_ylabel(f"{SPLIT_LABELS[split]}\n事件预测 Mw")
            if row_index == 1:
                axis.set_xlabel("目录震级 Mw")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["phase73"], label="事件 |误差|≤0.30 Mw"),
        Line2D([0], [0], marker="x", color=COLORS["phase73"], linestyle="none", label="事件 |误差|>0.30 Mw"),
        Line2D([0], [0], color=COLORS["identity"], linewidth=1.2, label="预测 = 目录 Mw"),
        Line2D([0], [0], color=COLORS["target"], linestyle="--", linewidth=0.9, label="±0.15 Mw"),
        Patch(facecolor=COLORS["band"], alpha=0.09, edgecolor="none", label="±0.30 Mw 区间"),
    ]
    figure.legend(handles=legend, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.94))
    figure.suptitle(f"Phase73、Phase39 与三种 PGD：{horizon} 秒观测（{int(horizon + PROCESSING_DELAY_SEC)} 秒发布）", y=0.995, fontsize=14)
    figure.text(0.5, 0.012, "事件台站中位数；相同 Train/Validation cohort。Phase73/Phase39 仅用 R 波形主干，PGD 使用 E/N/U。", ha="center", color="#444444", fontsize=9)
    figure.subplots_adjust(left=0.055, right=0.99, bottom=0.085, top=0.86, hspace=0.28, wspace=0.15)
    return _save_figure(figure, figures_dir / f"{figure_index:02d}_method_comparison_{horizon:03d}s")


def _summary_payload(
    replay: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lookup = _metric_lookup(replay["metric_rows"])
    selected_training = next(
        row
        for row in training_rows
        if int(row["seed"]) == SELECTED_SEED and int(row["epoch"]) == SELECTED_EPOCH
    )
    return {
        "status": "complete",
        "phase": "Phase73",
        "evaluation_role": "train_validation_only_stateful_streaming_diagnostic",
        "selected_seed": SELECTED_SEED,
        "selected_epoch": SELECTED_EPOCH,
        "validation_gate_passed": False,
        "hidden_data": {
            "internal_test_loaded": False,
            "external_eight_events_loaded": False,
            "grouped_test_loaded": False,
        },
        "stateful_contract": {
            "horizons_sec": list(HORIZONS),
            "processing_delay_sec": PROCESSING_DELAY_SEC,
            "state_carried_between_horizons": True,
            "output": "complete STF and final-Mw forecast updated each second",
            "pgd_hint": "causal Crowell Mw plus P-arrived validity",
            "waveform_backbone": "frozen Phase39 R-only Glehman+GI",
        },
        "selected_training_row": {
            key: value
            for key, value in selected_training.items()
            if key in {
                "seed",
                "epoch",
                "train_total_normalized_loss",
                "endpoint_event_mae",
                "endpoint_station_mae",
                "event_post160_band_width_p95_mw",
                "selection_score",
            }
        },
        "selected_horizon_metrics": [
            {
                "horizon_sec": horizon,
                "release_time_sec": horizon + PROCESSING_DELAY_SEC,
                "train": {
                    method: {
                        "station_mae_mw": lookup[("train", horizon, method)]["station_mae_mw"],
                        "event_mae_mw": lookup[("train", horizon, method)]["event_mae_mw"],
                    }
                    for method in METHOD_ORDER
                },
                "validation": {
                    method: {
                        "station_mae_mw": lookup[("validation", horizon, method)]["station_mae_mw"],
                        "event_mae_mw": lookup[("validation", horizon, method)]["event_mae_mw"],
                    }
                    for method in METHOD_ORDER
                },
            }
            for horizon in SELECTED_HORIZONS
        ],
        "trajectory_metrics": {
            "validation_event_count": len(replay["trajectory_diagnostics"]),
            "start_to_end_increase_fraction": float(
                np.mean(
                    [
                        float(row["phase73_start_to_end_change_mw"]) >= 0.0
                        for row in replay["trajectory_diagnostics"]
                    ]
                )
            ),
            "post160_band_width_p95_mw": float(
                np.percentile(
                    [float(row["phase73_post160_band_width_mw"]) for row in replay["trajectory_diagnostics"]],
                    95,
                )
            ),
            "peak_to_final_p95_mw": float(
                np.percentile(
                    [float(row["phase73_peak_to_final_mw"]) for row in replay["trajectory_diagnostics"]],
                    95,
                )
            ),
        },
        "source": replay["source"],
    }


def _readme(summary: Mapping[str, Any]) -> str:
    selected = {int(row["horizon_sec"]): row for row in summary["selected_horizon_metrics"]}
    lines = [
        "# Phase73 PGD 引导有状态流式预测：Train / Validation 图件报告",
        "",
        "> 固定 Phase73 seed17 epoch27；不重新训练、不换 seed、不做外部平滑或单调 clamp。",
        "> 本报告只使用训练集与 within_event_station validation；internal test、反复使用的 8 个事件和 grouped test 均未打开。",
        "",
        "## 结论",
        "",
        "Phase73 已经实现真正的逐秒有状态更新：每秒继承上一秒的 STF、Mw、GRU hidden state 和平台置信度，",
        "再结合当前 Phase39 STF 提案与因果 Crowell PGD 提示做小幅修正。它不是 Phase39 那种每秒独立重算。",
        "",
        f"在 validation 上，200 秒 Event/Station MAE 为 **{float(selected[200]['validation']['phase73']['event_mae_mw']):.6f} / {float(selected[200]['validation']['phase73']['station_mae_mw']):.6f} Mw**。",
        f"30 个事件中 **{100.0 * float(summary['trajectory_metrics']['start_to_end_increase_fraction']):.1f}%** 最终高于 20 秒估计；",
        f"160--200 秒事件平台宽度 p95 为 **{float(summary['trajectory_metrics']['post160_band_width_p95_mw']):.6f} Mw**。",
        "",
        "它在五个关键时刻都优于 Crowell PGD 的 validation Event MAE，但 200 秒仍未恢复 Phase39 的最终精度门槛，",
        "因此它是当前最好的流式候选，不是已经通过全部 validation gate 的最终模型。",
        "",
        "![训练过程](figures/01_training_dynamics.png)",
        "",
        "[PDF 图件](figures/01_training_dynamics.pdf)",
        "",
        "## 逐秒总体表现",
        "",
        "![逐秒总体指标](figures/02_overall_metrics.png)",
        "",
        "[PDF 图件](figures/02_overall_metrics.pdf)",
        "",
        "## 30 个 Validation 事件轨迹",
        "",
        "![Validation 事件轨迹](figures/03_validation_event_trajectories.png)",
        "",
        "[PDF 图件](figures/03_validation_event_trajectories.pdf)",
        "",
        "![Validation 轨迹诊断](figures/04_validation_trajectory_diagnostics.png)",
        "",
        "[PDF 图件](figures/04_validation_trajectory_diagnostics.pdf)",
        "",
        "## 200 秒最终散点",
        "",
        "![Validation 最终散点](figures/05_validation_endpoint_scatter.png)",
        "",
        "[PDF 图件](figures/05_validation_endpoint_scatter.pdf)",
        "",
        "## 五个关键时间点",
        "",
        "| 观测/发布 | Train Station MAE | Train Event MAE | Validation Station MAE | Validation Event MAE | Validation Crowell Event MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for horizon in SELECTED_HORIZONS:
        row = selected[horizon]
        lines.append(
            f"| {horizon}/{int(row['release_time_sec'])} s | "
            f"{float(row['train']['phase73']['station_mae_mw']):.6f} | "
            f"{float(row['train']['phase73']['event_mae_mw']):.6f} | "
            f"{float(row['validation']['phase73']['station_mae_mw']):.6f} | "
            f"{float(row['validation']['phase73']['event_mae_mw']):.6f} | "
            f"{float(row['validation']['crowell']['event_mae_mw']):.6f} |"
        )
    lines.append("")
    for figure_index, horizon in enumerate(SELECTED_HORIZONS, start=6):
        stem = f"figures/{figure_index:02d}_train_validation_{horizon:03d}s"
        lines.extend(
            [
                f"### {horizon} 秒 Train / Validation",
                "",
                f"![Phase73 {horizon} 秒散点]({stem}.png)",
                "",
                f"[PDF 图件]({stem}.pdf)",
                "",
            ]
        )
    lines.extend(["## Phase73、Phase39 与三种 PGD", ""])
    for figure_index, horizon in enumerate(SELECTED_HORIZONS, start=11):
        stem = f"figures/{figure_index:02d}_method_comparison_{horizon:03d}s"
        lines.extend(
            [
                f"### {horizon} 秒方法比较",
                "",
                f"![{horizon} 秒方法比较]({stem}.png)",
                "",
                f"[PDF 图件]({stem}.pdf)",
                "",
            ]
        )
    lines.extend(
        [
            "## 方法边界",
            "",
            "- 神经波形主干仍只输入 R 分量；Crowell PGD 提示由原始 E/N/U 计算。",
            "- 每个 horizon 使用 `0 <= t < h` 的数据，并报告 `h+6 s` 发布时间。",
            "- Phase73 预测的是当前最优的完整 STF 与最终 Mw，不是截至当前已释放矩的严格累计量。",
            "- 输出允许小幅向下修正，但模型内部限制后期逐秒修正，并训练平台宽度。",
            "- Train/Validation split 是 `within_event_station`；同一事件台站分散在两个 split，不能据此宣称未见事件泛化。",
            "- Phase73 未通过完整 endpoint gate，因此本报告没有打开 internal test、8 个开发事件或 grouped test。",
            "",
            "## 可审计工件",
            "",
            "- [汇总](summary.json)",
            "- [逐秒 Event 指标](horizon_metrics.csv)",
            "- [逐事件逐秒预测](phase73_event_predictions.csv)",
            "- [五时刻逐站预测（gzip）](phase73_selected_horizon_station_predictions.csv.gz)",
            "- [Validation 轨迹诊断](validation_trajectory_diagnostics.csv)",
            "- [三个 seed 训练记录](training_epoch_metrics.csv)",
            "- [运行来源](provenance.json)",
            "- [发布清单](publication_manifest.json)",
            "- [生成脚本](../../../scripts/plotting/plot_phase73_stateful_validation_zh.py)",
            "- [聚焦测试](../../../tests/test_plot_phase73_stateful_validation_zh.py)",
            "",
        ]
    )
    return "\n".join(lines)


def _prepare_output(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise ValueError(f"output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)


def generate_report(
    *,
    run_root: Path,
    cache_root: Path,
    hint_cache_root: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    _prepare_output(output_dir, overwrite=overwrite)
    replay = replay_phase73(
        run_root=run_root,
        cache_root=cache_root,
        hint_cache_root=hint_cache_root,
        device=device,
        batch_size=batch_size,
    )
    training_rows = _load_training_rows(run_root)
    summary = _summary_payload(replay, training_rows)

    event_fields = tuple(replay["event_rows"][0])
    station_fields = tuple(replay["selected_station_rows"][0])
    metric_fields = tuple(replay["metric_rows"][0])
    diagnostic_fields = tuple(replay["trajectory_diagnostics"][0])
    training_fields = tuple(training_rows[0])
    _write_csv(output_dir / "phase73_event_predictions.csv", replay["event_rows"], fieldnames=event_fields)
    _write_csv(
        output_dir / "phase73_selected_horizon_station_predictions.csv.gz",
        replay["selected_station_rows"],
        fieldnames=station_fields,
    )
    _write_csv(output_dir / "horizon_metrics.csv", replay["metric_rows"], fieldnames=metric_fields)
    _write_csv(output_dir / "validation_trajectory_diagnostics.csv", replay["trajectory_diagnostics"], fieldnames=diagnostic_fields)
    _write_csv(output_dir / "training_epoch_metrics.csv", training_rows, fieldnames=training_fields)
    _write_json(output_dir / "summary.json", summary)

    figures_dir = output_dir / "figures"
    outputs: list[Path] = []
    outputs.extend(plot_training_dynamics(training_rows, figures_dir=figures_dir))
    outputs.extend(plot_overall_metrics(replay["metric_rows"], figures_dir=figures_dir))
    outputs.extend(plot_event_trajectories(replay["event_rows"], figures_dir=figures_dir))
    outputs.extend(plot_trajectory_diagnostics(replay["trajectory_diagnostics"], replay["event_rows"], figures_dir=figures_dir))
    outputs.extend(plot_endpoint_scatter(replay["selected_station_rows"], replay["event_rows"], replay["metric_rows"], figures_dir=figures_dir))
    for figure_index, horizon in enumerate(SELECTED_HORIZONS, start=6):
        outputs.extend(
            plot_horizon_scatter(
                replay["selected_station_rows"],
                replay["event_rows"],
                replay["metric_rows"],
                horizon=horizon,
                figure_index=figure_index,
                figures_dir=figures_dir,
            )
        )
    for figure_index, horizon in enumerate(SELECTED_HORIZONS, start=11):
        outputs.extend(
            plot_method_comparison(
                replay["event_rows"],
                replay["metric_rows"],
                horizon=horizon,
                figure_index=figure_index,
                figures_dir=figures_dir,
            )
        )
    if len(outputs) != 30 or any(not path.is_file() for path in outputs):
        raise RuntimeError("Phase73 report did not generate 15 PNG/PDF figure pairs")

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "generator_sha256": sha256_file(SCRIPT_PATH),
        "git_commit": current_git_commit(REPO_ROOT),
        "git_dirty": git_is_dirty(REPO_ROOT),
        "device": str(device),
        "batch_size": batch_size,
        "selected_seed": SELECTED_SEED,
        "selected_epoch": SELECTED_EPOCH,
        "hidden_data": summary["hidden_data"],
        "source": replay["source"],
        "crowell_reproduction": {
            split: replay["splits"][split]["max_crowell_recompute_abs_diff_mw"]
            for split in ("train", "validation")
        },
    }
    _write_json(output_dir / "provenance.json", provenance)
    (output_dir / "README.md").write_text(_readme(summary), encoding="utf-8")

    published_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "publication_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "report_status": "phase73_train_validation_only_stateful_diagnostic",
        "analysis_contract": {
            "model": "Phase73 PGD-guided stateful forecast",
            "selected_seed": SELECTED_SEED,
            "selected_epoch": SELECTED_EPOCH,
            "horizons_sec": list(HORIZONS),
            "selected_horizons_sec": list(SELECTED_HORIZONS),
            "train_station_count": EXPECTED_TRAIN_COUNT,
            "validation_station_count": EXPECTED_VALIDATION_COUNT,
            "internal_test_opened": False,
            "external_eight_events_opened": False,
            "grouped_test_opened": False,
        },
        "generator": {
            "path": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "tests": {
            "path": str(TEST_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(TEST_PATH) if TEST_PATH.is_file() else None,
        },
        "git_base_commit": current_git_commit(REPO_ROOT),
        "outputs": {
            str(path.relative_to(output_dir)): sha256_file(path)
            for path in published_files
        },
    }
    _write_json(output_dir / "publication_manifest.json", manifest)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Phase73 train/validation Chinese GitHub report."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--hint-cache-root", type=Path, default=DEFAULT_HINT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    summary = generate_report(
        run_root=args.run_root.resolve(),
        cache_root=args.cache_root.resolve(),
        hint_cache_root=args.hint_cache_root.resolve(),
        output_dir=args.output_dir.resolve(),
        device=torch.device(args.device),
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    endpoint = summary["selected_horizon_metrics"][-1]["validation"]["phase73"]
    print(
        "Phase73 report complete: "
        f"validation Event/Station MAE={endpoint['event_mae_mw']:.6f}/"
        f"{endpoint['station_mae_mw']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
