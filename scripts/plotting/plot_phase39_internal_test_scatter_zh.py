#!/usr/bin/env python3
"""Add the frozen Phase39 internal-test endpoint scatter to its prior report."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
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

DEFAULT_SOURCE_STATIONS = (
    REPO_ROOT
    / "docs"
    / "reports"
    / "phase66-stateful-test-external-zh"
    / "internal_endpoint_station_predictions.csv"
)
DEFAULT_SOURCE_SUMMARY = (
    REPO_ROOT
    / "docs"
    / "reports"
    / "phase66-stateful-test-external-zh"
    / "summary.json"
)
DEFAULT_REPORT_DIR = (
    REPO_ROOT
    / "docs"
    / "reports"
    / "phase39-second-by-second-convergence-zh"
)
TEST_PATH = REPO_ROOT / "tests" / "test_phase39_internal_test_scatter.py"

EXPECTED_SOURCE_STATIONS_SHA256 = (
    "8ece743ea16ae6ef5d6f063dbfaad29ee3c9200761cf278e8f06542d88b06979"
)
EXPECTED_SOURCE_SUMMARY_SHA256 = (
    "4182fab1ca1d7ae0bb9386b25399bf03a51de8959b4fac04ade7be1442a22d73"
)
EXPECTED_SPLIT_ASSIGNMENT_SHA256 = (
    "5ac2e07ed186dce737a3592694632775b7bbf603bf922a4a74fa6b86a3d5c240"
)
EXPECTED_STATION_COUNT = 385
EXPECTED_EVENT_COUNT = 30
EXPECTED_STATION_MAE = 0.11933648616277263
EXPECTED_EVENT_MAE = 0.1522870352995013
SOURCE_REPORT_COMMIT = "ebcabb4e75665b41b5ab69575724332ccce5175a"
OBSERVATION_HORIZON_SEC = 200
REFERENCE_BAND_MW = 0.30
TARGET_BAND_MW = 0.15

COLORS = {
    "in_band": "#0072B2",
    "outlier": "#D55E00",
    "identity": "#222222",
    "target": "#8E5AA9",
    "band": "#009E73",
    "grid": "#D8DEE6",
    "ink": "#20262E",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        for row in rows:
            writer.writerow(
                {
                    key: "" if row.get(key) is None else row.get(key)
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


def _error_metrics(catalog: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(catalog, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if reference.shape != estimate.shape or reference.ndim != 1:
        raise ValueError("catalog and prediction must be aligned one-dimensional arrays")
    error = estimate - reference
    return {
        "count": int(error.size),
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "bias_mw": float(np.mean(error)),
        "pearson_r": float(np.corrcoef(reference, estimate)[0, 1]),
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


def _display_event(event: str) -> str:
    return re.sub(r"(?<=[A-Za-z])(?=\d{4}$)", " ", str(event))


def load_frozen_endpoint(
    station_path: Path = DEFAULT_SOURCE_STATIONS,
    summary_path: Path = DEFAULT_SOURCE_SUMMARY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if _sha256(station_path) != EXPECTED_SOURCE_STATIONS_SHA256:
        raise ValueError("frozen Phase39 internal endpoint station CSV changed")
    if _sha256(summary_path) != EXPECTED_SOURCE_SUMMARY_SHA256:
        raise ValueError("frozen Phase66 publication summary changed")
    source_summary = _read_json(summary_path)
    internal = source_summary.get("internal", {})
    if internal.get("station_count") != EXPECTED_STATION_COUNT:
        raise ValueError("Phase39 internal station count changed")
    if internal.get("event_count") != EXPECTED_EVENT_COUNT:
        raise ValueError("Phase39 internal event count changed")
    raw_metrics = internal.get("raw", {})
    if not math.isclose(
        float(raw_metrics.get("endpoint_station_mae", float("nan"))),
        EXPECTED_STATION_MAE,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("published Phase39 Station MAE changed")
    if not math.isclose(
        float(raw_metrics.get("endpoint_event_mae", float("nan"))),
        EXPECTED_EVENT_MAE,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("published Phase39 Event MAE changed")

    source_rows = _read_csv(station_path)
    if len(source_rows) != EXPECTED_STATION_COUNT:
        raise ValueError("Phase39 endpoint CSV row count changed")
    required = {
        "event",
        "station",
        "observation_horizon_sec",
        "mw_catalog",
        "raw_mw_pred",
    }
    if not required.issubset(source_rows[0]):
        raise ValueError("Phase39 endpoint CSV columns changed")

    station_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in source_rows:
        if int(float(source["observation_horizon_sec"])) != OBSERVATION_HORIZON_SEC:
            raise ValueError("Phase39 internal endpoint horizon changed")
        key = (str(source["event"]), str(source["station"]))
        if key in seen:
            raise ValueError(f"duplicate Phase39 endpoint station: {key}")
        seen.add(key)
        catalog = float(source["mw_catalog"])
        prediction = float(source["raw_mw_pred"])
        row = {
            "event": key[0],
            "station": key[1],
            "observation_horizon_sec": OBSERVATION_HORIZON_SEC,
            "mw_catalog": catalog,
            "mw_pred": prediction,
            "error_mw": prediction - catalog,
            "abs_error_mw": abs(prediction - catalog),
        }
        station_rows.append(row)
        grouped[key[0]].append(row)

    event_rows: list[dict[str, Any]] = []
    for event in sorted(grouped):
        rows = grouped[event]
        catalogs = np.asarray([float(row["mw_catalog"]) for row in rows])
        if not np.allclose(catalogs, catalogs[0], rtol=0.0, atol=1.0e-7):
            raise ValueError(f"catalog magnitude differs within event {event}")
        prediction = float(np.median([float(row["mw_pred"]) for row in rows]))
        catalog = float(catalogs[0])
        event_rows.append(
            {
                "event": event,
                "display_event": _display_event(event),
                "observation_horizon_sec": OBSERVATION_HORIZON_SEC,
                "mw_catalog": catalog,
                "mw_pred_median": prediction,
                "error_mw": prediction - catalog,
                "abs_error_mw": abs(prediction - catalog),
                "station_count": len(rows),
            }
        )

    station_catalog = np.asarray([row["mw_catalog"] for row in station_rows])
    station_prediction = np.asarray([row["mw_pred"] for row in station_rows])
    event_catalog = np.asarray([row["mw_catalog"] for row in event_rows])
    event_prediction = np.asarray([row["mw_pred_median"] for row in event_rows])
    station_metrics = _error_metrics(station_catalog, station_prediction)
    event_metrics = _error_metrics(event_catalog, event_prediction)
    if not math.isclose(
        station_metrics["mae_mw"],
        EXPECTED_STATION_MAE,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("reconstructed Phase39 Station MAE changed")
    if not math.isclose(
        event_metrics["mae_mw"],
        EXPECTED_EVENT_MAE,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("reconstructed Phase39 Event MAE changed")
    payload = {
        "status": "complete",
        "model": "Phase39 Glehman scalar + global invariant, seed42",
        "evaluation_role": "within_event_station_internal_test",
        "observation_horizon_sec": OBSERVATION_HORIZON_SEC,
        "station_metrics": station_metrics,
        "event_metrics": event_metrics,
        "largest_event_errors": sorted(
            event_rows,
            key=lambda row: (-float(row["abs_error_mw"]), str(row["event"])),
        )[:10],
        "source": {
            "station_predictions_path": str(station_path.relative_to(REPO_ROOT)),
            "station_predictions_sha256": EXPECTED_SOURCE_STATIONS_SHA256,
            "summary_path": str(summary_path.relative_to(REPO_ROOT)),
            "summary_sha256": EXPECTED_SOURCE_SUMMARY_SHA256,
            "source_report_commit": SOURCE_REPORT_COMMIT,
            "split_assignment_sha256": EXPECTED_SPLIT_ASSIGNMENT_SHA256,
        },
        "interpretation": (
            "same-event held-out station interpolation; not unseen-event generalization"
        ),
    }
    return station_rows, event_rows, payload


def _metrics_text(metrics: Mapping[str, Any]) -> str:
    return (
        f"n={int(metrics['count'])}  |  MAE={float(metrics['mae_mw']):.3f}  |  "
        f"RMSE={float(metrics['rmse_mw']):.3f}  |  "
        f"bias={float(metrics['bias_mw']):+.3f}  |  r={float(metrics['pearson_r']):.3f}\n"
        f"±0.15: {100.0 * float(metrics['within_0_15_fraction']):.1f}%  |  "
        f"±0.30: {100.0 * float(metrics['within_0_30_fraction']):.1f}%"
    )


def _style_scatter_axis(
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
        alpha=0.10,
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
        fontsize=9,
        color=COLORS["ink"],
    )


def plot_endpoint_scatter(
    station_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    output_stem: Path,
) -> list[Path]:
    _configure_plotting()
    station_catalog = np.asarray([float(row["mw_catalog"]) for row in station_rows])
    station_prediction = np.asarray([float(row["mw_pred"]) for row in station_rows])
    station_abs_error = np.abs(station_prediction - station_catalog)
    event_catalog = np.asarray([float(row["mw_catalog"]) for row in event_rows])
    event_prediction = np.asarray(
        [float(row["mw_pred_median"]) for row in event_rows]
    )
    event_abs_error = np.abs(event_prediction - event_catalog)
    all_values = np.concatenate(
        [station_catalog, station_prediction, event_catalog, event_prediction]
    )
    lower = math.floor((float(np.min(all_values)) - 0.25) * 2.0) / 2.0
    upper = math.ceil((float(np.max(all_values)) + 0.25) * 2.0) / 2.0

    figure, axes = plt.subplots(1, 2, figsize=(13.4, 6.4), sharex=True, sharey=True)
    station_in_band = station_abs_error <= REFERENCE_BAND_MW
    axes[0].scatter(
        station_catalog[station_in_band],
        station_prediction[station_in_band],
        s=22,
        color=COLORS["in_band"],
        alpha=0.48,
        edgecolors="none",
        zorder=3,
    )
    axes[0].scatter(
        station_catalog[~station_in_band],
        station_prediction[~station_in_band],
        s=28,
        color=COLORS["outlier"],
        marker="D",
        alpha=0.82,
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
    )
    _style_scatter_axis(
        axes[0],
        lower=lower,
        upper=upper,
        title="A  台站级最终预测（385 条 test 记录）",
        metrics=summary["station_metrics"],
    )

    event_in_band = event_abs_error <= REFERENCE_BAND_MW
    sizes = np.asarray(
        [46.0 + 23.0 * math.sqrt(float(row["station_count"])) for row in event_rows]
    )
    axes[1].scatter(
        event_catalog[event_in_band],
        event_prediction[event_in_band],
        s=sizes[event_in_band],
        color=COLORS["in_band"],
        alpha=0.72,
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )
    axes[1].scatter(
        event_catalog[~event_in_band],
        event_prediction[~event_in_band],
        s=sizes[~event_in_band],
        color=COLORS["outlier"],
        marker="D",
        alpha=0.90,
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
    )
    _style_scatter_axis(
        axes[1],
        lower=lower,
        upper=upper,
        title="B  事件级最终预测（30 个事件台站中位数）",
        metrics=summary["event_metrics"],
    )
    label_offsets = {
        "Lefkada2015": (8, -18),
        "Anchorage2018": (8, 14),
        "Napa2014": (8, 9),
    }
    for row in sorted(
        event_rows,
        key=lambda item: (-float(item["abs_error_mw"]), str(item["event"])),
    )[:3]:
        offset = label_offsets.get(str(row["event"]), (7, 8))
        axes[1].annotate(
            f"{row['display_event']} (n={int(row['station_count'])})",
            (float(row["mw_catalog"]), float(row["mw_pred_median"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            color=COLORS["ink"],
            va="bottom" if offset[1] >= 0 else "top",
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["in_band"],
            markeredgecolor="none",
            label="|误差| ≤ 0.30 Mw",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=COLORS["outlier"],
            markeredgecolor="none",
            label="|误差| > 0.30 Mw",
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
            alpha=0.10,
            edgecolor="none",
            label="±0.30 Mw 区间",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=False,
    )
    figure.suptitle(
        "Phase39 完整 200 秒输入的最终震级结果",
        y=0.995,
        fontsize=14,
    )
    figure.text(
        0.5,
        0.018,
        "internal test 为 within_event_station：检验同一事件的未见台站插值，不代表未见事件泛化。",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.84, wspace=0.16)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [png, pdf]


def _update_report_summary(report_dir: Path, endpoint: Mapping[str, Any]) -> None:
    path = report_dir / "summary.json"
    summary = _read_json(path)
    summary["internal_test_endpoint"] = dict(endpoint)
    _write_json(path, summary)


def _update_publication_manifest(
    report_dir: Path,
    endpoint: Mapping[str, Any],
    *,
    full_regression: str,
) -> None:
    path = report_dir / "publication_manifest.json"
    manifest = _read_json(path)
    manifest["schema_version"] = 2
    manifest["supplemental_generated_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    manifest["supplemental_internal_test_endpoint"] = {
        "evaluation_role": endpoint["evaluation_role"],
        "station_count": endpoint["station_metrics"]["count"],
        "event_count": endpoint["event_metrics"]["count"],
        "station_mae_mw": endpoint["station_metrics"]["mae_mw"],
        "event_mae_mw": endpoint["event_metrics"]["mae_mw"],
        "source": endpoint["source"],
    }
    manifest["supplemental_generator"] = {
        "path": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "sha256": _sha256(SCRIPT_PATH),
    }
    manifest["supplemental_tests"] = {
        "path": str(TEST_PATH.relative_to(REPO_ROOT)),
        "sha256": _sha256(TEST_PATH),
    }
    manifest["verified_metrics"].update(
        {
            "internal_test_endpoint_station_mae_mw": endpoint["station_metrics"][
                "mae_mw"
            ],
            "internal_test_endpoint_event_mae_mw": endpoint["event_metrics"][
                "mae_mw"
            ],
            "internal_test_endpoint_station_pearson_r": endpoint[
                "station_metrics"
            ]["pearson_r"],
            "internal_test_endpoint_event_pearson_r": endpoint["event_metrics"][
                "pearson_r"
            ],
            "full_repository_tests_after_scatter": full_regression,
        }
    )
    outputs: dict[str, str] = {}
    for output in sorted(report_dir.rglob("*")):
        if output.is_file() and output.name != "publication_manifest.json":
            outputs[str(output.relative_to(report_dir))] = _sha256(output)
    manifest["outputs"] = outputs
    _write_json(path, manifest)


def publish(
    *,
    source_stations: Path,
    source_summary: Path,
    report_dir: Path,
    full_regression: str,
) -> dict[str, Any]:
    if not report_dir.is_dir():
        raise FileNotFoundError(f"missing Phase39 report directory: {report_dir}")
    station_rows, event_rows, endpoint = load_frozen_endpoint(
        source_stations,
        source_summary,
    )
    _write_csv(
        report_dir / "internal_test_endpoint_station_predictions.csv",
        station_rows,
        fieldnames=(
            "event",
            "station",
            "observation_horizon_sec",
            "mw_catalog",
            "mw_pred",
            "error_mw",
            "abs_error_mw",
        ),
    )
    _write_csv(
        report_dir / "internal_test_endpoint_event_predictions.csv",
        event_rows,
        fieldnames=(
            "event",
            "display_event",
            "observation_horizon_sec",
            "mw_catalog",
            "mw_pred_median",
            "error_mw",
            "abs_error_mw",
            "station_count",
        ),
    )
    _write_json(report_dir / "internal_test_endpoint_summary.json", endpoint)
    plot_endpoint_scatter(
        station_rows,
        event_rows,
        endpoint,
        output_stem=report_dir / "figures" / "04_internal_test_endpoint_scatter",
    )
    _update_report_summary(report_dir, endpoint)
    _update_publication_manifest(
        report_dir,
        endpoint,
        full_regression=full_regression,
    )
    return endpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add the Phase39 200-second internal-test scatter to its report."
    )
    parser.add_argument(
        "--source-stations",
        type=Path,
        default=DEFAULT_SOURCE_STATIONS,
    )
    parser.add_argument(
        "--source-summary",
        type=Path,
        default=DEFAULT_SOURCE_SUMMARY,
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--full-regression",
        default="pending final repository regression",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    endpoint = publish(
        source_stations=args.source_stations.resolve(),
        source_summary=args.source_summary.resolve(),
        report_dir=args.report_dir.resolve(),
        full_regression=str(args.full_regression),
    )
    print(
        json.dumps(
            {
                "status": endpoint["status"],
                "report_dir": str(args.report_dir.resolve()),
                "station_mae_mw": endpoint["station_metrics"]["mae_mw"],
                "event_mae_mw": endpoint["event_metrics"]["mae_mw"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
