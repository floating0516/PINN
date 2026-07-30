#!/usr/bin/env python3
"""Publish Phase39 strict-prefix train/test scatter figures at five horizons."""

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

from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_phase45_posthoc_streaming import (  # noqa: E402
    EXPECTED_SPLIT_ASSIGNMENT_SHA256,
    EXPECTED_TEST_COUNT,
    _load_internal_samples,
    _mw_cube,
)
from scripts.evaluation.evaluate_phase66_stateful_streaming import (  # noqa: E402
    DEFAULT_INTERNAL_RAW_RATES,
    EXPECTED_INTERNAL_RAW_SHA256,
)
from scripts.experiments.run_phase40_streaming_consistency import (  # noqa: E402
    load_frozen_config,
)
from scripts.experiments.run_phase43_streaming_adapter import (  # noqa: E402
    EXPECTED_TRAIN_COUNT,
    HORIZONS,
    PROCESSING_DELAY_SEC,
    load_cache,
)
from scripts.experiments.run_phase50_stateful_incremental_model import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
)
from src.utils.provenance import sha256_file  # noqa: E402


SELECTED_HORIZONS = (30, 60, 90, 120, 200)
EXPECTED_TRAIN_EVENT_COUNT = 31
EXPECTED_TEST_EVENT_COUNT = 30
EXPECTED_TEST_ENDPOINT_SHA256 = (
    "60ce7f363b979cbbd6161e3bb47f01b3d11e8815f67ad80bb1e94318dcda06ae"
)
EXPECTED_TEST_STATION_MAE = 0.11933648616277263
EXPECTED_TEST_EVENT_MAE = 0.1522870352995013
ENDPOINT_TOLERANCE_MW = 1.0e-12
TARGET_BAND_MW = 0.15
REFERENCE_BAND_MW = 0.30

PROJECT_HOME = REPO_ROOT.parent.parent
DEFAULT_TEST_ENDPOINT = (
    PROJECT_HOME
    / "runs"
    / "phase46-phase45-posthoc-20260728T131632Z-20d35a9"
    / "internal"
    / "endpoint_station_predictions.csv"
)
DEFAULT_REPORT_DIR = (
    REPO_ROOT
    / "docs"
    / "reports"
    / "phase39-second-by-second-convergence-zh"
)
TEST_PATH = REPO_ROOT / "tests" / "test_phase39_train_test_horizons.py"

COLORS = {
    "station": "#66717E",
    "event": "#0072B2",
    "outlier": "#D55E00",
    "identity": "#20262E",
    "target": "#8E5AA9",
    "band": "#009E73",
    "grid": "#D8DEE6",
    "ink": "#20262E",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def prediction_metrics(
    catalog: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    reference = np.asarray(catalog, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if reference.shape != estimate.shape or reference.ndim != 1:
        raise ValueError("catalog and prediction must be aligned one-dimensional arrays")
    if reference.size == 0 or not np.all(np.isfinite(reference)):
        raise ValueError("catalog values must be finite and nonempty")
    if not np.all(np.isfinite(estimate)):
        raise ValueError("prediction values must be finite")
    error = estimate - reference
    pearson = None
    if (
        reference.size >= 2
        and float(np.std(reference)) > 0.0
        and float(np.std(estimate)) > 0.0
    ):
        pearson = float(np.corrcoef(reference, estimate)[0, 1])
    return {
        "count": int(error.size),
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "bias_mw": float(np.mean(error)),
        "pearson_r": pearson,
        "within_0_15_count": int(np.count_nonzero(np.abs(error) <= TARGET_BAND_MW)),
        "within_0_15_fraction": float(np.mean(np.abs(error) <= TARGET_BAND_MW)),
        "within_0_30_count": int(
            np.count_nonzero(np.abs(error) <= REFERENCE_BAND_MW)
        ),
        "within_0_30_fraction": float(
            np.mean(np.abs(error) <= REFERENCE_BAND_MW)
        ),
        "max_abs_error_mw": float(np.max(np.abs(error))),
    }


def _station_rows(
    *,
    split: str,
    rates: np.ndarray,
    events: Sequence[str],
    stations: Sequence[str],
    catalogs: np.ndarray,
    source_dt_sec: np.ndarray,
) -> list[dict[str, Any]]:
    cube = np.asarray(rates, dtype=np.float32)
    if cube.shape != (len(SELECTED_HORIZONS), len(events), 200):
        raise ValueError(f"unexpected {split} rate cube shape: {cube.shape}")
    if len(stations) != len(events) or catalogs.shape != (len(events),):
        raise ValueError(f"{split} station metadata is not aligned")
    mw = _mw_cube(cube, source_dt_sec)
    rows: list[dict[str, Any]] = []
    for horizon_index, horizon in enumerate(SELECTED_HORIZONS):
        for station_index, (event, station) in enumerate(
            zip(events, stations, strict=True)
        ):
            prediction = float(mw[horizon_index, station_index])
            catalog = float(catalogs[station_index])
            rows.append(
                {
                    "split": split,
                    "observation_horizon_sec": horizon,
                    "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                    "event": str(event),
                    "station": str(station),
                    "mw_catalog": catalog,
                    "mw_pred": prediction,
                    "error_mw": prediction - catalog,
                    "abs_error_mw": abs(prediction - catalog),
                }
            )
    return rows


def build_event_rows(
    station_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in station_rows:
        grouped[
            (
                str(row["split"]),
                int(row["observation_horizon_sec"]),
                str(row["event"]),
            )
        ].append(row)
    event_rows: list[dict[str, Any]] = []
    for (split, horizon, event), rows in sorted(grouped.items()):
        catalogs = np.asarray([float(row["mw_catalog"]) for row in rows])
        if not np.allclose(catalogs, catalogs[0], rtol=0.0, atol=1.0e-7):
            raise ValueError(f"catalog magnitude differs within event {event}")
        prediction = float(np.median([float(row["mw_pred"]) for row in rows]))
        catalog = float(catalogs[0])
        event_rows.append(
            {
                "split": split,
                "observation_horizon_sec": horizon,
                "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                "event": event,
                "mw_catalog": catalog,
                "mw_pred_median": prediction,
                "error_mw": prediction - catalog,
                "abs_error_mw": abs(prediction - catalog),
                "station_count": len(rows),
            }
        )
    return event_rows


def summarize_predictions(
    station_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for split in ("train", "test"):
        for horizon in SELECTED_HORIZONS:
            stations = [
                row
                for row in station_rows
                if row["split"] == split
                and int(row["observation_horizon_sec"]) == horizon
            ]
            events = [
                row
                for row in event_rows
                if row["split"] == split
                and int(row["observation_horizon_sec"]) == horizon
            ]
            station_metrics = prediction_metrics(
                [float(row["mw_catalog"]) for row in stations],
                [float(row["mw_pred"]) for row in stations],
            )
            event_metrics = prediction_metrics(
                [float(row["mw_catalog"]) for row in events],
                [float(row["mw_pred_median"]) for row in events],
            )
            summary_rows.append(
                {
                    "split": split,
                    "observation_horizon_sec": horizon,
                    "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                    **{
                        f"station_{key}": value
                        for key, value in station_metrics.items()
                    },
                    **{f"event_{key}": value for key, value in event_metrics.items()},
                }
            )
    return summary_rows


def _endpoint_reproduction(
    station_rows: Sequence[Mapping[str, Any]],
    endpoint_path: Path,
) -> dict[str, Any]:
    if sha256_file(endpoint_path) != EXPECTED_TEST_ENDPOINT_SHA256:
        raise ValueError("frozen Phase39 internal-test endpoint CSV changed")
    published_rows = _read_csv(endpoint_path)
    published = {
        (str(row["event"]), str(row["station"])): float(row["raw_mw_pred"])
        for row in published_rows
    }
    generated = {
        (str(row["event"]), str(row["station"])): float(row["mw_pred"])
        for row in station_rows
        if row["split"] == "test"
        and int(row["observation_horizon_sec"]) == 200
    }
    if set(generated) != set(published):
        raise ValueError("generated test endpoint station cohort changed")
    differences = {
        key: abs(generated[key] - published[key]) for key in sorted(generated)
    }
    max_key = max(differences, key=differences.get)
    maximum = float(differences[max_key])
    if maximum > ENDPOINT_TOLERANCE_MW:
        raise ValueError(f"Phase39 test endpoint reproduction changed: {maximum}")
    return {
        "passed": True,
        "station_count": len(generated),
        "max_abs_diff_mw": maximum,
        "max_abs_diff_station": {"event": max_key[0], "station": max_key[1]},
        "tolerance_mw": ENDPOINT_TOLERANCE_MW,
        "source_path": str(endpoint_path),
        "source_sha256": EXPECTED_TEST_ENDPOINT_SHA256,
    }


def load_frozen_predictions(
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    test_raw_rates_path: Path = DEFAULT_INTERNAL_RAW_RATES,
    test_endpoint_path: Path = DEFAULT_TEST_ENDPOINT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    horizon_indices = [HORIZONS.index(horizon) for horizon in SELECTED_HORIZONS]
    cache = load_cache(cache_root)
    train_indices = np.flatnonzero(cache.arrays["split_code"] == 0)
    if train_indices.size != EXPECTED_TRAIN_COUNT:
        raise ValueError("Phase39 training count changed")
    train_rates = np.stack(
        [
            np.asarray(
                cache.raw_rates[train_indices, horizon_index, :],
                dtype=np.float32,
            )
            for horizon_index in horizon_indices
        ]
    )
    train_records = [cache.records[int(index)] for index in train_indices]
    train_rows = _station_rows(
        split="train",
        rates=train_rates,
        events=[str(row["event"]) for row in train_records],
        stations=[str(row["station"]) for row in train_records],
        catalogs=np.asarray(
            cache.arrays["magnitude_catalog"][train_indices], dtype=np.float64
        ),
        source_dt_sec=np.asarray(
            cache.arrays["source_dt_sec"][train_indices], dtype=np.float64
        ),
    )

    if sha256_file(test_raw_rates_path) != EXPECTED_INTERNAL_RAW_SHA256:
        raise ValueError("frozen Phase39 internal-test raw-rate cube changed")
    test_raw_rates = np.load(test_raw_rates_path, mmap_mode="r")
    if test_raw_rates.shape != (len(HORIZONS), EXPECTED_TEST_COUNT, 200):
        raise ValueError("Phase39 internal-test raw-rate cube shape changed")
    config = load_frozen_config()
    test_samples, _, split_manifest = _load_internal_samples(
        config,
        split_name="test",
    )
    if split_manifest["assignment_sha256"] != EXPECTED_SPLIT_ASSIGNMENT_SHA256:
        raise ValueError("Phase39 split assignment changed")
    test_rates = np.asarray(test_raw_rates[horizon_indices], dtype=np.float32)
    test_rows = _station_rows(
        split="test",
        rates=test_rates,
        events=[str(sample["event"]) for sample in test_samples],
        stations=[str(sample["station"]) for sample in test_samples],
        catalogs=np.asarray(
            [float(sample["magnitude_catalog"]) for sample in test_samples],
            dtype=np.float64,
        ),
        source_dt_sec=np.asarray(
            [float(sample["stf_dt_sec"]) for sample in test_samples],
            dtype=np.float64,
        ),
    )

    station_rows = train_rows + test_rows
    event_rows = build_event_rows(station_rows)
    summary_rows = summarize_predictions(station_rows, event_rows)
    if len({row["event"] for row in event_rows if row["split"] == "train"}) != (
        EXPECTED_TRAIN_EVENT_COUNT
    ):
        raise ValueError("Phase39 training event count changed")
    if len({row["event"] for row in event_rows if row["split"] == "test"}) != (
        EXPECTED_TEST_EVENT_COUNT
    ):
        raise ValueError("Phase39 internal-test event count changed")
    summary_by_key = {
        (str(row["split"]), int(row["observation_horizon_sec"])): row
        for row in summary_rows
    }
    test_endpoint = summary_by_key[("test", 200)]
    if not math.isclose(
        float(test_endpoint["station_mae_mw"]),
        EXPECTED_TEST_STATION_MAE,
        rel_tol=0.0,
        abs_tol=5.0e-8,
    ):
        raise ValueError("Phase39 published test Station MAE changed")
    if not math.isclose(
        float(test_endpoint["event_mae_mw"]),
        EXPECTED_TEST_EVENT_MAE,
        rel_tol=0.0,
        abs_tol=5.0e-8,
    ):
        raise ValueError("Phase39 published test Event MAE changed")
    endpoint_gate = _endpoint_reproduction(station_rows, test_endpoint_path)
    summary = {
        "status": "complete",
        "model": "Phase39 Glehman scalar + global invariant, seed42",
        "evaluation_role": "within_event_station_train_vs_internal_test_diagnostic",
        "prefix_contract": {
            "input": "strict variable-length R prefix with shape Bx1xh",
            "independent_predictions": True,
            "state_carried_between_horizons": False,
            "processing_delay_sec": PROCESSING_DELAY_SEC,
            "horizons_sec": list(SELECTED_HORIZONS),
        },
        "split_semantics": (
            "within_event_station; train and test contain stations from the same events"
        ),
        "metrics": summary_rows,
        "test_endpoint_reproduction": endpoint_gate,
        "source": {
            "phase39_checkpoint_sha256": cache.manifest["source_artifact_sha256"][
                "checkpoint"
            ],
            "split_assignment_sha256": EXPECTED_SPLIT_ASSIGNMENT_SHA256,
            "train_cache_root": str(cache_root),
            "train_cache_raw_rates_sha256": cache.manifest["raw_rates_sha256"],
            "test_raw_rates_path": str(test_raw_rates_path),
            "test_raw_rates_sha256": EXPECTED_INTERNAL_RAW_SHA256,
        },
    }
    return station_rows, event_rows, summary


def _metrics_for(
    summary: Mapping[str, Any],
    *,
    split: str,
    horizon: int,
) -> Mapping[str, Any]:
    for row in summary["metrics"]:
        if row["split"] == split and int(row["observation_horizon_sec"]) == horizon:
            return row
    raise KeyError((split, horizon))


def _metrics_text(metrics: Mapping[str, Any]) -> str:
    station_r = metrics["station_pearson_r"]
    event_r = metrics["event_pearson_r"]
    return (
        f"台站 n={int(metrics['station_count'])}  "
        f"MAE={float(metrics['station_mae_mw']):.3f}  "
        f"RMSE={float(metrics['station_rmse_mw']):.3f}  "
        f"r={float(station_r):.3f}\n"
        f"事件中位数 n={int(metrics['event_count'])}  "
        f"MAE={float(metrics['event_mae_mw']):.3f}  "
        f"RMSE={float(metrics['event_rmse_mw']):.3f}  "
        f"r={float(event_r):.3f}\n"
        f"事件 ±0.15={100.0 * float(metrics['event_within_0_15_fraction']):.1f}%  "
        f"bias={float(metrics['event_bias_mw']):+.3f}"
    )


def _style_axis(
    axis: Any,
    *,
    lower: float,
    upper: float,
    title: str,
    metrics: Mapping[str, Any],
) -> None:
    line = np.linspace(lower, upper, 400)
    axis.fill_between(
        line,
        line - REFERENCE_BAND_MW,
        line + REFERENCE_BAND_MW,
        color=COLORS["band"],
        alpha=0.09,
        linewidth=0,
    )
    axis.plot(line, line, color=COLORS["identity"], linewidth=1.2)
    axis.plot(
        line,
        line + TARGET_BAND_MW,
        color=COLORS["target"],
        linestyle="--",
        linewidth=0.9,
    )
    axis.plot(
        line,
        line - TARGET_BAND_MW,
        color=COLORS["target"],
        linestyle="--",
        linewidth=0.9,
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("目录震级 Mw")
    axis.set_ylabel("Phase39 预测 Mw")
    axis.set_title(title)
    axis.grid(True, color=COLORS["grid"], linewidth=0.7)
    axis.set_axisbelow(True)
    axis.text(
        0.03,
        0.97,
        _metrics_text(metrics),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8.7,
        color=COLORS["ink"],
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 3},
        zorder=8,
    )


def plot_horizon(
    station_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    horizon: int,
    output_stem: Path,
    lower: float,
    upper: float,
) -> list[Path]:
    _configure_plotting()
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 6.4), sharex=True, sharey=True)
    for axis, split, panel, split_title in zip(
        axes,
        ("train", "test"),
        ("A", "B"),
        ("训练集", "internal test"),
        strict=True,
    ):
        stations = [
            row
            for row in station_rows
            if row["split"] == split
            and int(row["observation_horizon_sec"]) == horizon
        ]
        events = [
            row
            for row in event_rows
            if row["split"] == split
            and int(row["observation_horizon_sec"]) == horizon
        ]
        station_catalog = np.asarray([float(row["mw_catalog"]) for row in stations])
        station_prediction = np.asarray([float(row["mw_pred"]) for row in stations])
        event_catalog = np.asarray([float(row["mw_catalog"]) for row in events])
        event_prediction = np.asarray(
            [float(row["mw_pred_median"]) for row in events]
        )
        event_abs_error = np.abs(event_prediction - event_catalog)
        event_in_band = event_abs_error <= REFERENCE_BAND_MW
        axis.scatter(
            station_catalog,
            station_prediction,
            s=14 if split == "train" else 19,
            color=COLORS["station"],
            alpha=0.20 if split == "train" else 0.30,
            edgecolors="none",
            zorder=2,
        )
        sizes = np.asarray(
            [42.0 + 18.0 * math.sqrt(float(row["station_count"])) for row in events]
        )
        axis.scatter(
            event_catalog[event_in_band],
            event_prediction[event_in_band],
            s=sizes[event_in_band],
            color=COLORS["event"],
            marker="D",
            alpha=0.86,
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
        )
        axis.scatter(
            event_catalog[~event_in_band],
            event_prediction[~event_in_band],
            s=sizes[~event_in_band],
            color=COLORS["outlier"],
            marker="D",
            alpha=0.90,
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        metrics = _metrics_for(summary, split=split, horizon=horizon)
        _style_axis(
            axis,
            lower=lower,
            upper=upper,
            title=(
                f"{panel}  {split_title}（{int(metrics['station_count'])} 条记录，"
                f"{int(metrics['event_count'])} 个事件）"
            ),
            metrics=metrics,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["station"],
            markeredgecolor="none",
            alpha=0.45,
            label="单台站预测",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=COLORS["event"],
            markeredgecolor="white",
            label="事件中位数（|误差|≤0.30）",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=COLORS["outlier"],
            markeredgecolor="white",
            label="事件中位数（|误差|>0.30）",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["identity"],
            linewidth=1.2,
            label="预测 = 目录 Mw",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["target"],
            linestyle="--",
            linewidth=0.9,
            label="±0.15 Mw",
        ),
        Patch(
            facecolor=COLORS["band"],
            alpha=0.09,
            edgecolor="none",
            label="±0.30 Mw 区间",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        (
            f"Phase39 严格变长前缀：{horizon} 秒观测 "
            f"（{int(horizon + PROCESSING_DELAY_SEC)} 秒发布）"
        ),
        y=0.995,
        fontsize=14,
    )
    figure.text(
        0.5,
        0.018,
        (
            "每个时刻独立输入 B×1×h 的 R 波形；不继承上一秒状态。"
            "test 为 within_event_station，同一事件的未见台站插值。"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.82, wspace=0.16)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [png, pdf]


def plot_all_horizons(
    station_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    figures_dir: Path,
) -> list[Path]:
    all_values = np.asarray(
        [
            float(row[key])
            for rows, key in (
                (station_rows, "mw_catalog"),
                (station_rows, "mw_pred"),
                (event_rows, "mw_catalog"),
                (event_rows, "mw_pred_median"),
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    lower = math.floor((float(np.min(all_values)) - 0.25) * 2.0) / 2.0
    upper = math.ceil((float(np.max(all_values)) + 0.25) * 2.0) / 2.0
    outputs: list[Path] = []
    for figure_index, horizon in enumerate(SELECTED_HORIZONS, start=5):
        outputs.extend(
            plot_horizon(
                station_rows,
                event_rows,
                summary,
                horizon=horizon,
                output_stem=figures_dir
                / f"{figure_index:02d}_train_test_prefix_{horizon:03d}s",
                lower=lower,
                upper=upper,
            )
        )
    return outputs


README_START = "<!-- phase39-train-test-horizons:start -->"
README_END = "<!-- phase39-train-test-horizons:end -->"


def _readme_section(summary: Mapping[str, Any]) -> str:
    metrics = {
        (str(row["split"]), int(row["observation_horizon_sec"])): row
        for row in summary["metrics"]
    }
    lines = [
        README_START,
        "## 训练集与 internal test：五个严格变长前缀",
        "",
        "这组图固定 Phase39 seed42，不重新训练。每个时刻输入真正的 `B×1×h` R 波形前缀，",
        "每次独立预测完整 STF，不继承上一秒状态；发布时刻为 `h+6` 秒。",
        "",
        "| 观测/发布 | Train Station MAE | Train Event MAE | Test Station MAE | Test Event MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for horizon in SELECTED_HORIZONS:
        train = metrics[("train", horizon)]
        test = metrics[("test", horizon)]
        lines.append(
            f"| {horizon}/{int(horizon + PROCESSING_DELAY_SEC)} s | "
            f"{float(train['station_mae_mw']):.6f} | "
            f"{float(train['event_mae_mw']):.6f} | "
            f"{float(test['station_mae_mw']):.6f} | "
            f"{float(test['event_mae_mw']):.6f} |"
        )
    lines.extend(
        [
            "",
            "短前缀结果不是逐步改善：Phase39 只用完整 200 秒训练，30--120 秒属于分布外输入。",
            "到 200 秒时，Train Event MAE 为 "
            f"**{float(metrics[('train', 200)]['event_mae_mw']):.6f} Mw**，"
            "Test Event MAE 为 "
            f"**{float(metrics[('test', 200)]['event_mae_mw']):.6f} Mw**。",
            "",
        ]
    )
    for figure_index, horizon in enumerate(SELECTED_HORIZONS, start=5):
        stem = f"figures/{figure_index:02d}_train_test_prefix_{horizon:03d}s"
        lines.extend(
            [
                f"### {horizon} 秒",
                "",
                f"![Phase39 train/test {horizon} 秒散点]({stem}.png)",
                "",
                f"[PDF 图件]({stem}.pdf)",
                "",
            ]
        )
    lines.append(README_END)
    return "\n".join(lines)


def _update_readme(report_dir: Path, summary: Mapping[str, Any]) -> None:
    path = report_dir / "README.md"
    text = path.read_text(encoding="utf-8")
    section = _readme_section(summary)
    if README_START in text:
        before, remainder = text.split(README_START, 1)
        _, after = remainder.split(README_END, 1)
        updated = before.rstrip() + "\n\n" + section + after
    else:
        marker = "## 八个事件轨迹"
        if marker not in text:
            raise ValueError("Phase39 report insertion marker changed")
        updated = text.replace(marker, section + "\n\n" + marker, 1)
    path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def _update_report_summary(report_dir: Path, summary: Mapping[str, Any]) -> None:
    path = report_dir / "summary.json"
    payload = _read_json(path)
    payload["internal_train_test_strict_prefix_horizons"] = dict(summary)
    _write_json(path, payload)


def _update_publication_manifest(
    report_dir: Path,
    summary: Mapping[str, Any],
    *,
    full_regression: str,
) -> None:
    path = report_dir / "publication_manifest.json"
    manifest = _read_json(path)
    manifest["schema_version"] = 3
    manifest["train_test_horizons_generated_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    manifest["supplemental_train_test_horizons"] = {
        "evaluation_role": summary["evaluation_role"],
        "horizons_sec": list(SELECTED_HORIZONS),
        "train_station_count": EXPECTED_TRAIN_COUNT,
        "test_station_count": EXPECTED_TEST_COUNT,
        "train_event_count": EXPECTED_TRAIN_EVENT_COUNT,
        "test_event_count": EXPECTED_TEST_EVENT_COUNT,
        "test_endpoint_reproduction": summary["test_endpoint_reproduction"],
        "source": summary["source"],
    }
    manifest["supplemental_train_test_generator"] = {
        "path": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "sha256": sha256_file(SCRIPT_PATH),
    }
    manifest["supplemental_train_test_tests"] = {
        "path": str(TEST_PATH.relative_to(REPO_ROOT)),
        "sha256": sha256_file(TEST_PATH),
    }
    verified = manifest.setdefault("verified_metrics", {})
    for row in summary["metrics"]:
        split = str(row["split"])
        horizon = int(row["observation_horizon_sec"])
        verified[f"{split}_{horizon}s_station_mae_mw"] = row["station_mae_mw"]
        verified[f"{split}_{horizon}s_event_mae_mw"] = row["event_mae_mw"]
    verified["train_test_horizon_endpoint_max_abs_diff_mw"] = summary[
        "test_endpoint_reproduction"
    ]["max_abs_diff_mw"]
    verified["full_repository_tests_after_train_test_horizons"] = full_regression
    outputs: dict[str, str] = {}
    for output in sorted(report_dir.rglob("*")):
        if output.is_file() and output.name != "publication_manifest.json":
            outputs[str(output.relative_to(report_dir))] = sha256_file(output)
    manifest["outputs"] = outputs
    _write_json(path, manifest)


def publish(
    *,
    cache_root: Path,
    test_raw_rates_path: Path,
    test_endpoint_path: Path,
    report_dir: Path,
    full_regression: str,
) -> dict[str, Any]:
    if not report_dir.is_dir():
        raise FileNotFoundError(f"missing Phase39 report directory: {report_dir}")
    station_rows, event_rows, summary = load_frozen_predictions(
        cache_root=cache_root,
        test_raw_rates_path=test_raw_rates_path,
        test_endpoint_path=test_endpoint_path,
    )
    _write_csv(
        report_dir / "internal_train_test_prefix_station_predictions.csv",
        station_rows,
        fieldnames=(
            "split",
            "observation_horizon_sec",
            "release_time_sec",
            "event",
            "station",
            "mw_catalog",
            "mw_pred",
            "error_mw",
            "abs_error_mw",
        ),
    )
    _write_csv(
        report_dir / "internal_train_test_prefix_event_predictions.csv",
        event_rows,
        fieldnames=(
            "split",
            "observation_horizon_sec",
            "release_time_sec",
            "event",
            "mw_catalog",
            "mw_pred_median",
            "error_mw",
            "abs_error_mw",
            "station_count",
        ),
    )
    _write_csv(
        report_dir / "internal_train_test_prefix_metrics.csv",
        summary["metrics"],
        fieldnames=tuple(summary["metrics"][0]),
    )
    _write_json(report_dir / "internal_train_test_prefix_summary.json", summary)
    plot_all_horizons(
        station_rows,
        event_rows,
        summary,
        figures_dir=report_dir / "figures",
    )
    _update_readme(report_dir, summary)
    _update_report_summary(report_dir, summary)
    _update_publication_manifest(
        report_dir,
        summary,
        full_regression=full_regression,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish five Phase39 strict-prefix train/test scatter figures in Chinese."
        )
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--test-raw-rates",
        type=Path,
        default=DEFAULT_INTERNAL_RAW_RATES,
    )
    parser.add_argument(
        "--test-endpoint",
        type=Path,
        default=DEFAULT_TEST_ENDPOINT,
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
        cache_root=args.cache_root.resolve(),
        test_raw_rates_path=args.test_raw_rates.resolve(),
        test_endpoint_path=args.test_endpoint.resolve(),
        report_dir=args.report_dir.resolve(),
        full_regression=str(args.full_regression),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "report_dir": str(args.report_dir.resolve()),
                "horizons_sec": list(SELECTED_HORIZONS),
                "endpoint_max_abs_diff_mw": summary[
                    "test_endpoint_reproduction"
                ]["max_abs_diff_mw"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
