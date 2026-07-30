#!/usr/bin/env python3
"""Publish causal five-horizon Phase39 versus PGD comparison figures."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.run_phase40_streaming_consistency import (  # noqa: E402
    load_frozen_config,
)
from scripts.plotting.plot_phase39_train_test_horizons_zh import (  # noqa: E402
    DEFAULT_REPORT_DIR,
    EXPECTED_TEST_COUNT,
    EXPECTED_TEST_EVENT_COUNT,
    EXPECTED_SPLIT_ASSIGNMENT_SHA256,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_TRAIN_EVENT_COUNT,
    PROCESSING_DELAY_SEC,
    REFERENCE_BAND_MW,
    SELECTED_HORIZONS,
    TARGET_BAND_MW,
    _configure_plotting,
    _read_csv,
    _read_json,
    _write_csv,
    _write_json,
    prediction_metrics,
)
from src.baseline.causal_pgd import (  # noqa: E402
    RawPGDRecord,
    build_raw_pgd_records,
    causal_pgd_3d,
)
from src.baseline.scaling_laws import predict_mw  # noqa: E402
from src.data.loaders_v2 import get_data_loaders_v2  # noqa: E402
from src.utils.provenance import sha256_file  # noqa: E402


METHOD_ORDER = ("phase39", "crowell", "melgar", "ruhl")
PGD_METHODS = METHOD_ORDER[1:]
METHOD_LABELS = {
    "phase39": "Phase39",
    "crowell": "Crowell PGD",
    "melgar": "Melgar PGD",
    "ruhl": "Ruhl PGD",
}
METHOD_COLORS = {
    "phase39": "#0072B2",
    "crowell": "#D55E00",
    "melgar": "#CC79A7",
    "ruhl": "#009E73",
}
STRUCTURE_COLORS = {
    "identity": "#20262E",
    "target": "#8E5AA9",
    "band": "#56B4E9",
    "grid": "#D8DEE6",
    "ink": "#20262E",
}

PHASE39_STATION_FILENAME = "internal_train_test_prefix_station_predictions.csv"
PHASE39_EVENT_FILENAME = "internal_train_test_prefix_event_predictions.csv"
PGD_STATION_FILENAME = "internal_train_test_pgd_station_predictions.csv"
METHOD_EVENT_FILENAME = "internal_train_test_method_event_predictions.csv"
METHOD_METRICS_FILENAME = "internal_train_test_method_metrics.csv"
PGD_SUMMARY_FILENAME = "internal_train_test_pgd_summary.json"
TEST_PATH = REPO_ROOT / "tests" / "test_phase39_pgd_horizon_comparison.py"


def _load_train_test_samples(
    config: dict[str, Any],
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, Any]]:
    train_loader, validation_loader, test_loader, split_manifest = get_data_loaders_v2(
        config
    )
    if split_manifest["assignment_sha256"] != EXPECTED_SPLIT_ASSIGNMENT_SHA256:
        raise ValueError("Phase39 split assignment changed")
    full_dataset = train_loader.dataset.dataset
    train_indices = [int(value) for value in train_loader.dataset.indices]
    test_indices = [int(value) for value in test_loader.dataset.indices]
    if len(train_indices) != EXPECTED_TRAIN_COUNT:
        raise ValueError("Phase39 training count changed")
    if len(test_indices) != EXPECTED_TEST_COUNT:
        raise ValueError("Phase39 internal-test count changed")
    samples = {
        "train": [full_dataset.samples[index] for index in train_indices],
        "test": [full_dataset.samples[index] for index in test_indices],
    }
    del train_loader, validation_loader, test_loader
    return samples, split_manifest


def evaluate_pgd_station_rows(
    samples_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    records: Mapping[tuple[str, str], RawPGDRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "test"):
        for sample in samples_by_split[split]:
            key = (str(sample["event"]), str(sample["station"]))
            record = records[key]
            for horizon in SELECTED_HORIZONS:
                pgd_m, available_count, observed_count = causal_pgd_3d(
                    record,
                    observation_horizon_sec=horizon,
                )
                row: dict[str, Any] = {
                    "split": split,
                    "observation_horizon_sec": horizon,
                    "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                    "event": record.event,
                    "station": record.station,
                    "mw_catalog": record.magnitude_catalog,
                    "source_distance_km": record.source_distance_km,
                    "p_arrival_sec": record.p_arrival_sec,
                    "pgd_3d_m": pgd_m,
                    "raw_available_sample_count": available_count,
                    "pgd_observed_sample_count": observed_count,
                }
                for method in PGD_METHODS:
                    prediction = predict_mw(
                        law_name=method,
                        pgd_m=pgd_m,
                        source_distance_km=record.source_distance_km,
                    )
                    row[f"{method}_mw_pred"] = prediction
                    row[f"{method}_error_mw"] = prediction - record.magnitude_catalog
                    row[f"{method}_abs_error_mw"] = abs(
                        prediction - record.magnitude_catalog
                    )
                rows.append(row)
    return rows


def _source_hash(
    manifest: Mapping[str, Any],
    *,
    relative_path: str,
    absolute_path: Path,
) -> str:
    expected = str(manifest["outputs"].get(relative_path, ""))
    actual = sha256_file(absolute_path)
    if not expected or actual != expected:
        raise ValueError(f"published Phase39 source hash changed: {relative_path}")
    return actual


def load_phase39_sources(
    report_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    manifest = _read_json(report_dir / "publication_manifest.json")
    station_path = report_dir / PHASE39_STATION_FILENAME
    event_path = report_dir / PHASE39_EVENT_FILENAME
    station_hash = _source_hash(
        manifest,
        relative_path=PHASE39_STATION_FILENAME,
        absolute_path=station_path,
    )
    event_hash = _source_hash(
        manifest,
        relative_path=PHASE39_EVENT_FILENAME,
        absolute_path=event_path,
    )
    station_rows = [
        {
            **row,
            "observation_horizon_sec": int(row["observation_horizon_sec"]),
            "release_time_sec": float(row["release_time_sec"]),
            "mw_catalog": float(row["mw_catalog"]),
            "mw_pred": float(row["mw_pred"]),
        }
        for row in _read_csv(station_path)
    ]
    event_rows = [
        {
            **row,
            "observation_horizon_sec": int(row["observation_horizon_sec"]),
            "release_time_sec": float(row["release_time_sec"]),
            "mw_catalog": float(row["mw_catalog"]),
            "mw_pred_median": float(row["mw_pred_median"]),
            "station_count": int(row["station_count"]),
        }
        for row in _read_csv(event_path)
    ]
    return station_rows, event_rows, {
        "phase39_station_predictions_sha256": station_hash,
        "phase39_event_predictions_sha256": event_hash,
    }


def validate_station_cohort(
    phase39_station_rows: Sequence[Mapping[str, Any]],
    pgd_station_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def keys(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, int, str, str]]:
        return {
            (
                str(row["split"]),
                int(row["observation_horizon_sec"]),
                str(row["event"]),
                str(row["station"]),
            )
            for row in rows
        }

    phase39_keys = keys(phase39_station_rows)
    pgd_keys = keys(pgd_station_rows)
    if phase39_keys != pgd_keys:
        missing = sorted(phase39_keys - pgd_keys)
        extra = sorted(pgd_keys - phase39_keys)
        raise ValueError(
            f"PGD/Phase39 station cohort differs; missing={missing[:10]}, extra={extra[:10]}"
        )
    expected = (EXPECTED_TRAIN_COUNT + EXPECTED_TEST_COUNT) * len(
        SELECTED_HORIZONS
    )
    if len(phase39_keys) != expected:
        raise ValueError("PGD/Phase39 station cohort size changed")
    return {
        "passed": True,
        "row_count": len(phase39_keys),
        "train_station_count": EXPECTED_TRAIN_COUNT,
        "test_station_count": EXPECTED_TEST_COUNT,
        "horizon_count": len(SELECTED_HORIZONS),
    }


def build_method_event_rows(
    phase39_event_rows: Sequence[Mapping[str, Any]],
    pgd_station_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in phase39_event_rows:
        prediction = float(source["mw_pred_median"])
        catalog = float(source["mw_catalog"])
        rows.append(
            {
                "split": str(source["split"]),
                "observation_horizon_sec": int(source["observation_horizon_sec"]),
                "release_time_sec": float(source["release_time_sec"]),
                "event": str(source["event"]),
                "method": "phase39",
                "method_label": METHOD_LABELS["phase39"],
                "mw_catalog": catalog,
                "mw_pred_median": prediction,
                "error_mw": prediction - catalog,
                "abs_error_mw": abs(prediction - catalog),
                "station_count": int(source["station_count"]),
            }
        )

    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for source in pgd_station_rows:
        grouped[
            (
                str(source["split"]),
                int(source["observation_horizon_sec"]),
                str(source["event"]),
            )
        ].append(source)
    for (split, horizon, event), stations in sorted(grouped.items()):
        catalogs = np.asarray([float(row["mw_catalog"]) for row in stations])
        if not np.allclose(catalogs, catalogs[0], rtol=0.0, atol=1.0e-7):
            raise ValueError(f"PGD catalog magnitude differs within event {event}")
        catalog = float(catalogs[0])
        for method in PGD_METHODS:
            predictions = np.asarray(
                [float(row[f"{method}_mw_pred"]) for row in stations]
            )
            prediction = float(np.median(predictions))
            rows.append(
                {
                    "split": split,
                    "observation_horizon_sec": horizon,
                    "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                    "event": event,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "mw_catalog": catalog,
                    "mw_pred_median": prediction,
                    "error_mw": prediction - catalog,
                    "abs_error_mw": abs(prediction - catalog),
                    "station_count": len(stations),
                }
            )

    rows.sort(
        key=lambda row: (
            int(row["observation_horizon_sec"]),
            str(row["split"]),
            METHOD_ORDER.index(str(row["method"])),
            str(row["event"]),
        )
    )
    return rows


def validate_event_cohort(method_event_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reference: dict[tuple[str, int], set[str]] = {}
    for split in ("train", "test"):
        expected_count = (
            EXPECTED_TRAIN_EVENT_COUNT if split == "train" else EXPECTED_TEST_EVENT_COUNT
        )
        for horizon in SELECTED_HORIZONS:
            phase39_events = {
                str(row["event"])
                for row in method_event_rows
                if row["split"] == split
                and int(row["observation_horizon_sec"]) == horizon
                and row["method"] == "phase39"
            }
            if len(phase39_events) != expected_count:
                raise ValueError(f"Phase39 {split} event count changed at {horizon}")
            reference[(split, horizon)] = phase39_events
            for method in PGD_METHODS:
                method_events = {
                    str(row["event"])
                    for row in method_event_rows
                    if row["split"] == split
                    and int(row["observation_horizon_sec"]) == horizon
                    and row["method"] == method
                }
                if method_events != phase39_events:
                    raise ValueError(
                        f"{method}/{split} event cohort changed at {horizon}"
                    )
    return {
        "passed": True,
        "train_event_count": EXPECTED_TRAIN_EVENT_COUNT,
        "test_event_count": EXPECTED_TEST_EVENT_COUNT,
        "methods": list(METHOD_ORDER),
    }


def build_method_metrics(
    phase39_station_rows: Sequence[Mapping[str, Any]],
    pgd_station_rows: Sequence[Mapping[str, Any]],
    method_event_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for split in ("train", "test"):
        for horizon in SELECTED_HORIZONS:
            for method in METHOD_ORDER:
                if method == "phase39":
                    station_subset = [
                        row
                        for row in phase39_station_rows
                        if row["split"] == split
                        and int(row["observation_horizon_sec"]) == horizon
                    ]
                    station_predictions = [
                        float(row["mw_pred"]) for row in station_subset
                    ]
                else:
                    station_subset = [
                        row
                        for row in pgd_station_rows
                        if row["split"] == split
                        and int(row["observation_horizon_sec"]) == horizon
                    ]
                    station_predictions = [
                        float(row[f"{method}_mw_pred"]) for row in station_subset
                    ]
                station_metrics = prediction_metrics(
                    [float(row["mw_catalog"]) for row in station_subset],
                    station_predictions,
                )
                event_subset = [
                    row
                    for row in method_event_rows
                    if row["split"] == split
                    and int(row["observation_horizon_sec"]) == horizon
                    and row["method"] == method
                ]
                event_metrics = prediction_metrics(
                    [float(row["mw_catalog"]) for row in event_subset],
                    [float(row["mw_pred_median"]) for row in event_subset],
                )
                metrics.append(
                    {
                        "split": split,
                        "observation_horizon_sec": horizon,
                        "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        **{
                            f"station_{key}": value
                            for key, value in station_metrics.items()
                        },
                        **{
                            f"event_{key}": value for key, value in event_metrics.items()
                        },
                    }
                )
    return metrics


def evaluate_frozen_comparison(
    *,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    config = load_frozen_config()
    samples_by_split, split_manifest = _load_train_test_samples(config)
    phase39_station_rows, phase39_event_rows, phase39_hashes = load_phase39_sources(
        report_dir
    )
    records = build_raw_pgd_records(config, samples_by_split)
    pgd_station_rows = evaluate_pgd_station_rows(samples_by_split, records)
    station_gate = validate_station_cohort(phase39_station_rows, pgd_station_rows)
    method_event_rows = build_method_event_rows(
        phase39_event_rows,
        pgd_station_rows,
    )
    event_gate = validate_event_cohort(method_event_rows)
    metrics = build_method_metrics(
        phase39_station_rows,
        pgd_station_rows,
        method_event_rows,
    )
    pgd_values = np.asarray([float(row["pgd_3d_m"]) for row in pgd_station_rows])
    summary = {
        "status": "complete",
        "evaluation_role": "within_event_station_phase39_vs_causal_pgd_diagnostic",
        "horizons_sec": list(SELECTED_HORIZONS),
        "processing_delay_sec": PROCESSING_DELAY_SEC,
        "methods": [
            {"method": method, "label": METHOD_LABELS[method]}
            for method in METHOD_ORDER
        ],
        "pgd_contract": {
            "components": ["east", "north", "up"],
            "peak": "max sqrt(E^2 + N^2 + U^2) over 0 <= t < h",
            "baseline": (
                "component-wise mean of pre-P raw samples available by release h+6; "
                "existing first-sample fallback if fewer than three pre-P samples"
            ),
            "filter": "none",
            "interpolation": "none",
            "station_threshold": "none; exact Phase39 train/test station cohort",
            "distance": "hypocentral/source distance",
            "aggregation": "event median over identical station cohort",
        },
        "phase39_contract": {
            "components": ["radial"],
            "input": "strict variable-length Bx1xh prefix",
            "state_carried_between_horizons": False,
        },
        "station_cohort_gate": station_gate,
        "event_cohort_gate": event_gate,
        "metrics": metrics,
        "pgd_range_m": {
            "minimum": float(np.min(pgd_values)),
            "maximum": float(np.max(pgd_values)),
        },
        "source": {
            **phase39_hashes,
            "split_assignment_sha256": split_manifest["assignment_sha256"],
            "dataset_path": str(config["paths"]["data_path"]),
        },
        "interpretation_boundary": (
            "PGD uses ENU while Phase39 uses R only; this is a method benchmark, "
            "not an input-component ablation"
        ),
    }
    return pgd_station_rows, method_event_rows, metrics, summary


def _metric_lookup(
    metrics: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    return {
        (
            str(row["split"]),
            int(row["observation_horizon_sec"]),
            str(row["method"]),
        ): row
        for row in metrics
    }


def _style_axis(
    axis: Any,
    *,
    lower: float,
    upper: float,
    metrics: Mapping[str, Any],
    method: str,
    split: str,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    line = np.linspace(lower, upper, 400)
    axis.fill_between(
        line,
        line - REFERENCE_BAND_MW,
        line + REFERENCE_BAND_MW,
        color=STRUCTURE_COLORS["band"],
        alpha=0.08,
        linewidth=0,
    )
    axis.plot(line, line, color=STRUCTURE_COLORS["identity"], linewidth=1.0)
    axis.plot(
        line,
        line + TARGET_BAND_MW,
        color=STRUCTURE_COLORS["target"],
        linestyle="--",
        linewidth=0.75,
    )
    axis.plot(
        line,
        line - TARGET_BAND_MW,
        color=STRUCTURE_COLORS["target"],
        linestyle="--",
        linewidth=0.75,
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, color=STRUCTURE_COLORS["grid"], linewidth=0.6)
    axis.set_axisbelow(True)
    if show_xlabel:
        axis.set_xlabel("目录震级 Mw")
    if show_ylabel:
        axis.set_ylabel("训练集预测 Mw" if split == "train" else "test 预测 Mw")
    axis.set_title(
        f"{METHOD_LABELS[method]}\n"
        f"MAE={float(metrics['event_mae_mw']):.3f}  "
        f"bias={float(metrics['event_bias_mw']):+.3f}  "
        f"±0.15={100.0 * float(metrics['event_within_0_15_fraction']):.0f}%",
        fontsize=9.5,
    )


def plot_horizon_comparison(
    method_event_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    output_stem: Path,
    lower: float,
    upper: float,
) -> list[Path]:
    _configure_plotting()
    metric_by_key = _metric_lookup(metrics)
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(17.8, 9.2),
        sharex=True,
        sharey=True,
    )
    for row_index, split in enumerate(("train", "test")):
        for column_index, method in enumerate(METHOD_ORDER):
            axis = axes[row_index, column_index]
            subset = [
                row
                for row in method_event_rows
                if row["split"] == split
                and int(row["observation_horizon_sec"]) == horizon
                and row["method"] == method
            ]
            catalog = np.asarray([float(row["mw_catalog"]) for row in subset])
            prediction = np.asarray(
                [float(row["mw_pred_median"]) for row in subset]
            )
            in_band = np.abs(prediction - catalog) <= REFERENCE_BAND_MW
            sizes = np.asarray(
                [28.0 + 8.0 * math.sqrt(float(row["station_count"])) for row in subset]
            )
            axis.scatter(
                catalog[in_band],
                prediction[in_band],
                s=sizes[in_band],
                color=METHOD_COLORS[method],
                marker="o",
                alpha=0.82,
                edgecolors="white",
                linewidths=0.55,
                zorder=4,
            )
            axis.scatter(
                catalog[~in_band],
                prediction[~in_band],
                s=sizes[~in_band] * 0.90,
                color=METHOD_COLORS[method],
                marker="x",
                alpha=0.92,
                linewidths=1.5,
                zorder=5,
            )
            _style_axis(
                axis,
                lower=lower,
                upper=upper,
                metrics=metric_by_key[(split, horizon, method)],
                method=method,
                split=split,
                show_ylabel=column_index == 0,
                show_xlabel=row_index == 1,
            )
            if column_index == 0:
                axis.text(
                    -0.24,
                    0.5,
                    "A 训练集" if split == "train" else "B internal test",
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                    color=STRUCTURE_COLORS["ink"],
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#66717E",
            markeredgecolor="white",
            label="事件 |误差|≤0.30 Mw",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#66717E",
            linestyle="none",
            label="事件 |误差|>0.30 Mw",
        ),
        Line2D(
            [0],
            [0],
            color=STRUCTURE_COLORS["identity"],
            linewidth=1.0,
            label="预测 = 目录 Mw",
        ),
        Line2D(
            [0],
            [0],
            color=STRUCTURE_COLORS["target"],
            linestyle="--",
            linewidth=0.8,
            label="±0.15 Mw",
        ),
        Patch(
            facecolor=STRUCTURE_COLORS["band"],
            alpha=0.08,
            edgecolor="none",
            label="±0.30 Mw 区间",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=5,
        frameon=False,
    )
    figure.suptitle(
        (
            f"Phase39 与三种 PGD 标度律：{horizon} 秒观测 "
            f"（{int(horizon + PROCESSING_DELAY_SEC)} 秒发布）"
        ),
        y=0.995,
        fontsize=14,
    )
    figure.text(
        0.5,
        0.012,
        (
            "事件中位数、相同台站 cohort、无 PGD 振幅筛选；"
            "PGD 使用 E/N/U，Phase39 仅使用 R。"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.08,
        top=0.84,
        wspace=0.13,
        hspace=0.23,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [png, pdf]


def plot_all_horizons(
    method_event_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
) -> list[Path]:
    all_values = np.asarray(
        [
            value
            for row in method_event_rows
            for value in (float(row["mw_catalog"]), float(row["mw_pred_median"]))
        ]
    )
    lower = math.floor((float(np.min(all_values)) - 0.25) * 2.0) / 2.0
    upper = math.ceil((float(np.max(all_values)) + 0.25) * 2.0) / 2.0
    outputs: list[Path] = []
    for figure_index, horizon in enumerate(SELECTED_HORIZONS, start=10):
        outputs.extend(
            plot_horizon_comparison(
                method_event_rows,
                metrics,
                horizon=horizon,
                output_stem=figures_dir
                / f"{figure_index:02d}_phase39_pgd_comparison_{horizon:03d}s",
                lower=lower,
                upper=upper,
            )
        )
    return outputs


README_START = "<!-- phase39-pgd-horizons:start -->"
README_END = "<!-- phase39-pgd-horizons:end -->"


def _mae_table(
    metrics: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> list[str]:
    lookup = _metric_lookup(metrics)
    lines = [
        f"| {'Train' if split == 'train' else 'Test'} 观测/发布 | Phase39 | Crowell | Melgar | Ruhl |",
        "|---|---:|---:|---:|---:|",
    ]
    for horizon in SELECTED_HORIZONS:
        values = [
            float(lookup[(split, horizon, method)]["event_mae_mw"])
            for method in METHOD_ORDER
        ]
        lines.append(
            f"| {horizon}/{int(horizon + PROCESSING_DELAY_SEC)} s | "
            + " | ".join(f"{value:.6f}" for value in values)
            + " |"
        )
    return lines


def _readme_section(metrics: Sequence[Mapping[str, Any]]) -> str:
    lookup = _metric_lookup(metrics)
    lines = [
        README_START,
        "## Phase39 与三种 PGD：五个时间节点",
        "",
        "PGD 基线使用同一 train/test 台站 cohort，在每个发布时间仅用当时可获得的原始 E/N/U 数据",
        "重新计算 3D PGD，再分别应用 Crowell、Melgar、Ruhl 标度律。没有额外 PGD 振幅筛选。",
        "Phase39 仍为 R-only 严格变长前缀，因此这是方法基线比较，不是相同输入分量的消融。",
        "",
    ]
    lines.extend(_mae_table(metrics, split="train"))
    lines.append("")
    lines.extend(_mae_table(metrics, split="test"))
    lines.extend(
        [
            "",
            "internal test 上，Crowell 在 30--120 秒的事件 MAE 均低于未做前缀训练的 Phase39；",
            f"200 秒时 Phase39 为 **{float(lookup[('test', 200, 'phase39')]['event_mae_mw']):.6f} Mw**，"
            f"优于 Crowell 的 **{float(lookup[('test', 200, 'crowell')]['event_mae_mw']):.6f} Mw**。",
            "Melgar 和 Ruhl 在早期整体存在明显负偏差。",
            "",
        ]
    )
    for figure_index, horizon in enumerate(SELECTED_HORIZONS, start=10):
        stem = f"figures/{figure_index:02d}_phase39_pgd_comparison_{horizon:03d}s"
        lines.extend(
            [
                f"### {horizon} 秒方法比较",
                "",
                f"![Phase39 与 PGD {horizon} 秒比较]({stem}.png)",
                "",
                f"[PDF 图件]({stem}.pdf)",
                "",
            ]
        )
    lines.append(README_END)
    return "\n".join(lines)


def _update_readme(report_dir: Path, metrics: Sequence[Mapping[str, Any]]) -> None:
    path = report_dir / "README.md"
    text = path.read_text(encoding="utf-8")
    section = _readme_section(metrics)
    if README_START in text:
        before, remainder = text.split(README_START, 1)
        _, after = remainder.split(README_END, 1)
        updated = before.rstrip() + "\n\n" + section + after
    else:
        marker = "## 八个事件轨迹"
        if marker not in text:
            raise ValueError("Phase39 report PGD insertion marker changed")
        updated = text.replace(marker, section + "\n\n" + marker, 1)
    path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def _update_report_summary(report_dir: Path, summary: Mapping[str, Any]) -> None:
    path = report_dir / "summary.json"
    payload = _read_json(path)
    payload["internal_train_test_phase39_pgd_horizons"] = dict(summary)
    _write_json(path, payload)


def _update_publication_manifest(
    report_dir: Path,
    summary: Mapping[str, Any],
    *,
    full_regression: str,
) -> None:
    path = report_dir / "publication_manifest.json"
    manifest = _read_json(path)
    manifest["schema_version"] = 4
    manifest["phase39_pgd_horizons_generated_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    manifest["supplemental_phase39_pgd_horizons"] = {
        "evaluation_role": summary["evaluation_role"],
        "horizons_sec": summary["horizons_sec"],
        "methods": summary["methods"],
        "pgd_contract": summary["pgd_contract"],
        "station_cohort_gate": summary["station_cohort_gate"],
        "event_cohort_gate": summary["event_cohort_gate"],
        "source": summary["source"],
        "interpretation_boundary": summary["interpretation_boundary"],
    }
    manifest["supplemental_phase39_pgd_generator"] = {
        "path": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "sha256": sha256_file(SCRIPT_PATH),
    }
    manifest["supplemental_phase39_pgd_tests"] = {
        "path": str(TEST_PATH.relative_to(REPO_ROOT)),
        "sha256": sha256_file(TEST_PATH),
    }
    verified = manifest.setdefault("verified_metrics", {})
    for row in summary["metrics"]:
        split = str(row["split"])
        horizon = int(row["observation_horizon_sec"])
        method = str(row["method"])
        verified[f"{split}_{horizon}s_{method}_event_mae_mw"] = row[
            "event_mae_mw"
        ]
    verified["full_repository_tests_after_phase39_pgd_horizons"] = full_regression
    outputs: dict[str, str] = {}
    for output in sorted(report_dir.rglob("*")):
        if output.is_file() and output.name != "publication_manifest.json":
            outputs[str(output.relative_to(report_dir))] = sha256_file(output)
    manifest["outputs"] = outputs
    _write_json(path, manifest)


def publish(
    *,
    report_dir: Path,
    full_regression: str,
) -> dict[str, Any]:
    if not report_dir.is_dir():
        raise FileNotFoundError(f"missing Phase39 report directory: {report_dir}")
    pgd_station_rows, method_event_rows, metrics, summary = evaluate_frozen_comparison(
        report_dir=report_dir
    )
    _write_csv(
        report_dir / PGD_STATION_FILENAME,
        pgd_station_rows,
        fieldnames=tuple(pgd_station_rows[0]),
    )
    _write_csv(
        report_dir / METHOD_EVENT_FILENAME,
        method_event_rows,
        fieldnames=tuple(method_event_rows[0]),
    )
    _write_csv(
        report_dir / METHOD_METRICS_FILENAME,
        metrics,
        fieldnames=tuple(metrics[0]),
    )
    _write_json(report_dir / PGD_SUMMARY_FILENAME, summary)
    plot_all_horizons(
        method_event_rows,
        metrics,
        figures_dir=report_dir / "figures",
    )
    _update_readme(report_dir, metrics)
    _update_report_summary(report_dir, summary)
    _update_publication_manifest(
        report_dir,
        summary,
        full_regression=full_regression,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish Phase39 versus causal PGD five-horizon figures."
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--full-regression",
        default="pending final repository regression",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = publish(
        report_dir=args.report_dir.resolve(),
        full_regression=str(args.full_regression),
    )
    metrics = _metric_lookup(summary["metrics"])
    print(
        json.dumps(
            {
                "status": summary["status"],
                "report_dir": str(args.report_dir.resolve()),
                "test_200s_event_mae": {
                    method: metrics[("test", 200, method)]["event_mae_mw"]
                    for method in METHOD_ORDER
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
