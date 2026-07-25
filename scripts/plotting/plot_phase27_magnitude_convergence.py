from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _style import apply_pub_style, style_axes  # noqa: E402
from plot_phase17_causal_online_results import (  # noqa: E402
    _read_csv,
    _read_json,
    _save_figure,
    _sha256,
    _write_csv,
)


EXPECTED_GIT_COMMIT = "e02aecac9b1211851b926d69e57c78da34970d1a"
EXPECTED_HORIZONS = tuple(float(value) for value in range(20, 201, 20))
EXPECTED_EVENT_COUNT = 30
EXPECTED_SELECTED_SEED = 17
PROCESSING_DELAY_SEC = 5.0
TARGET_ERROR_MW = 0.15
CENSOR_PLOT_SEC = 220.0

GROUP_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "mw_lt_7",
        "label": "Mw < 7.0",
        "color": "#009E73",
        "marker": "o",
    },
    {
        "key": "mw_7_to_lt_8",
        "label": "7.0 <= Mw < 8.0",
        "color": "#0072B2",
        "marker": "s",
    },
    {
        "key": "mw_ge_8",
        "label": "Mw >= 8.0",
        "color": "#D55E00",
        "marker": "^",
    },
)
GROUP_BY_KEY = {str(spec["key"]): spec for spec in GROUP_SPECS}


def _magnitude_group(mw: float) -> str:
    if mw < 7.0:
        return "mw_lt_7"
    if mw < 8.0:
        return "mw_7_to_lt_8"
    return "mw_ge_8"


def _panel_label(axis: Any, label: str, *, x: float = -0.14) -> None:
    axis.text(
        x,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def _finite_float(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite")
    return result


def _optional_horizon(
    sequence: Sequence[Mapping[str, Any]], predicate: Any
) -> float | None:
    for index, row in enumerate(sequence):
        if all(bool(predicate(later)) for later in sequence[index:]):
            return float(row["observation_horizon_sec"])
    return None


def _validate_publication_contract(
    *,
    internal_summary: Mapping[str, Any],
    train_summary: Mapping[str, Any],
    selection: Mapping[str, Any],
    metrics: Mapping[str, Any],
    cohort: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> None:
    if internal_summary.get("status") != "complete":
        raise ValueError("internal stage is not complete")
    if not bool(internal_summary.get("validation_gate", {}).get("passed")):
        raise ValueError("validation gate did not pass")
    if not bool(internal_summary.get("candidate_gate", {}).get("passed")):
        raise ValueError("locked internal gate did not pass")
    candidate = internal_summary.get("variants", {}).get("candidate", {})
    if int(candidate.get("selected_seed", -1)) != EXPECTED_SELECTED_SEED:
        raise ValueError("internal summary does not select seed 17")
    if train_summary.get("status") != "complete" or train_summary.get(
        "git_commit"
    ) != EXPECTED_GIT_COMMIT:
        raise ValueError("train summary is not the frozen Phase27 run")
    if int(selection.get("selected_seed", -1)) != EXPECTED_SELECTED_SEED:
        raise ValueError("training selection does not select seed 17")
    if bool(selection.get("ensemble_used")):
        raise ValueError("training selection unexpectedly uses an ensemble")
    if selection.get("selection_metric") != "validation_event_mae_catalog":
        raise ValueError("training selection metric changed")
    if int(metrics.get("event_count", -1)) != EXPECTED_EVENT_COUNT:
        raise ValueError("locked test does not contain 30 events")
    if int(metrics.get("station_count", -1)) != 385:
        raise ValueError("locked test does not contain 385 stations")
    if metrics.get("reference") != "catalog":
        raise ValueError("locked test does not use catalog Mw")
    if not math.isclose(
        float(metrics["event_mae"]),
        float(internal_summary["candidate_gate"]["event_mae"]),
        abs_tol=1.0e-12,
    ):
        raise ValueError("locked test metric differs from the gate")
    expected_cohort = {
        "cohort": "processed radial peak over the full 200 s record >= 2 cm",
        "end_to_end_causal": False,
        "radial_peak_min_cm": 2.0,
        "station_selection_causal": False,
        "waveform_prefix_causal": True,
    }
    if dict(cohort) != expected_cohort:
        raise ValueError("delayed-prefix cohort contract changed")
    if registry.get("primary_reference") != "catalog":
        raise ValueError("result registry primary reference changed")
    if registry.get("split_protocol") != "within_event_station":
        raise ValueError("result registry split protocol changed")
    checkpoint = registry.get("checkpoint", {})
    checkpoint_sha = str(checkpoint.get("sha256", ""))
    if len(checkpoint_sha) != 64:
        raise ValueError("result registry lacks a checkpoint SHA-256")


def _normalize_prediction_rows(
    source_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for source in source_rows:
        event = str(source["event"])
        horizon = _finite_float(
            source["observation_horizon_sec"], field="observation_horizon_sec"
        )
        release = _finite_float(source["release_time_sec"], field="release_time_sec")
        prediction = _finite_float(source["mw_pred_median"], field="mw_pred_median")
        catalog = _finite_float(source["mw_catalog"], field="mw_catalog")
        error = _finite_float(source["error"], field="error")
        station_count = int(source["station_count"])
        key = (event, horizon)
        if key in seen:
            raise ValueError(f"duplicate event-horizon row: {event} at {horizon:g} s")
        seen.add(key)
        if horizon not in EXPECTED_HORIZONS:
            raise ValueError(f"unexpected observation horizon: {horizon:g} s")
        if not math.isclose(
            release, horizon + PROCESSING_DELAY_SEC, abs_tol=1.0e-12
        ):
            raise ValueError("release time does not equal observation plus 5 s")
        if not math.isclose(prediction - catalog, error, abs_tol=2.0e-7):
            raise ValueError("prediction error is inconsistent")
        if station_count < 1:
            raise ValueError("event prediction has no available station")
        normalized.append(
            {
                "event": event,
                "observation_horizon_sec": horizon,
                "release_time_sec": release,
                "mw_pred_median": prediction,
                "mw_catalog": catalog,
                "error": error,
                "abs_error": abs(error),
                "station_count": station_count,
                "magnitude_group": _magnitude_group(catalog),
            }
        )
    expected_rows = EXPECTED_EVENT_COUNT * len(EXPECTED_HORIZONS)
    if len(normalized) != expected_rows:
        raise ValueError(f"expected {expected_rows} event-horizon rows")

    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        by_event[str(row["event"])].append(row)
    if len(by_event) != EXPECTED_EVENT_COUNT:
        raise ValueError("event-horizon table does not cover 30 events")
    for event, rows in by_event.items():
        rows.sort(key=lambda row: float(row["observation_horizon_sec"]))
        if tuple(float(row["observation_horizon_sec"]) for row in rows) != (
            EXPECTED_HORIZONS
        ):
            raise ValueError(f"event horizon grid is incomplete for {event}")
        if len({float(row["mw_catalog"]) for row in rows}) != 1:
            raise ValueError(f"catalog Mw changes over time for {event}")
    normalized.sort(
        key=lambda row: (float(row["observation_horizon_sec"]), str(row["event"]))
    )
    return normalized


def build_magnitude_group_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[
            (float(row["observation_horizon_sec"]), str(row["magnitude_group"]))
        ].append(row)
    rows: list[dict[str, Any]] = []
    for horizon in EXPECTED_HORIZONS:
        for spec in GROUP_SPECS:
            key = str(spec["key"])
            values = grouped[(horizon, key)]
            errors = np.asarray([float(row["error"]) for row in values])
            if errors.size == 0:
                raise ValueError(f"magnitude group {key} is empty")
            rows.append(
                {
                    "observation_horizon_sec": horizon,
                    "release_time_sec": horizon + PROCESSING_DELAY_SEC,
                    "magnitude_group": key,
                    "magnitude_group_label": str(spec["label"]),
                    "event_count": int(errors.size),
                    "event_mae": float(np.mean(np.abs(errors))),
                    "event_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "event_bias": float(np.mean(errors)),
                    "within_0p15_count": int(np.count_nonzero(np.abs(errors) <= TARGET_ERROR_MW)),
                    "within_0p15_fraction": float(
                        np.mean(np.abs(errors) <= TARGET_ERROR_MW)
                    ),
                }
            )
    return rows


def build_event_convergence_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_event: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_event[str(row["event"])].append(row)
    output: list[dict[str, Any]] = []
    for event, sequence in by_event.items():
        sequence.sort(key=lambda row: float(row["observation_horizon_sec"]))
        final = sequence[-1]
        first_within = next(
            (
                float(row["observation_horizon_sec"])
                for row in sequence
                if float(row["abs_error"]) <= TARGET_ERROR_MW
            ),
            None,
        )
        stable_accuracy = _optional_horizon(
            sequence, lambda row: float(row["abs_error"]) <= TARGET_ERROR_MW
        )
        final_prediction = float(final["mw_pred_median"])
        stable_to_final = _optional_horizon(
            sequence,
            lambda row: abs(float(row["mw_pred_median"]) - final_prediction)
            <= TARGET_ERROR_MW,
        )
        output.append(
            {
                "event": event,
                "mw_catalog": float(final["mw_catalog"]),
                "magnitude_group": str(final["magnitude_group"]),
                "magnitude_group_label": str(
                    GROUP_BY_KEY[str(final["magnitude_group"])]["label"]
                ),
                "final_station_count": int(final["station_count"]),
                "final_mw_pred_median": final_prediction,
                "final_error": float(final["error"]),
                "final_abs_error": float(final["abs_error"]),
                "first_within_0p15_observation_sec": first_within,
                "first_within_0p15_release_sec": (
                    None if first_within is None else first_within + PROCESSING_DELAY_SEC
                ),
                "stable_within_0p15_observation_sec": stable_accuracy,
                "stable_within_0p15_release_sec": (
                    None
                    if stable_accuracy is None
                    else stable_accuracy + PROCESSING_DELAY_SEC
                ),
                "stable_accuracy_right_censored": stable_accuracy is None,
                "stable_accuracy_censor_observation_sec": (
                    EXPECTED_HORIZONS[-1] if stable_accuracy is None else None
                ),
                "stable_accuracy_censor_release_sec": (
                    EXPECTED_HORIZONS[-1] + PROCESSING_DELAY_SEC
                    if stable_accuracy is None
                    else None
                ),
                "stable_within_0p15_of_final_observation_sec": stable_to_final,
                "stable_within_0p15_of_final_release_sec": (
                    None if stable_to_final is None else stable_to_final + PROCESSING_DELAY_SEC
                ),
            }
        )
    return sorted(output, key=lambda row: (float(row["mw_catalog"]), str(row["event"])))


def _validate_derived_metrics(
    *,
    prediction_rows: Sequence[Mapping[str, Any]],
    horizon_metrics: Sequence[Mapping[str, Any]],
    final_event_rows: Sequence[Mapping[str, str]],
    metrics: Mapping[str, Any],
) -> None:
    by_horizon: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_horizon[float(row["observation_horizon_sec"])].append(row)
    if len(horizon_metrics) != len(EXPECTED_HORIZONS):
        raise ValueError("persisted horizon metrics have the wrong length")
    persisted = {
        float(row["observation_horizon_sec"]): row for row in horizon_metrics
    }
    if tuple(sorted(persisted)) != EXPECTED_HORIZONS:
        raise ValueError("persisted horizon metric grid changed")
    for horizon, rows in by_horizon.items():
        event_mae = float(np.mean([float(row["abs_error"]) for row in rows]))
        event_bias = float(np.mean([float(row["error"]) for row in rows]))
        source = persisted[horizon]
        if int(source["event_count"]) != EXPECTED_EVENT_COUNT:
            raise ValueError("persisted horizon event count changed")
        if not math.isclose(
            event_mae, float(source["event_equal_mae"]), abs_tol=2.0e-7
        ) or not math.isclose(
            event_bias, float(source["event_equal_bias"]), abs_tol=2.0e-7
        ):
            raise ValueError("event rows do not reproduce persisted horizon metrics")

    final_predictions = {
        str(row["event"]): row
        for row in prediction_rows
        if float(row["observation_horizon_sec"]) == EXPECTED_HORIZONS[-1]
    }
    if len(final_event_rows) != EXPECTED_EVENT_COUNT:
        raise ValueError("final event summary does not contain 30 events")
    for row in final_event_rows:
        event = str(row["event"])
        if event not in final_predictions:
            raise ValueError(f"final event is missing from prefix rows: {event}")
        prefix = final_predictions[event]
        if not math.isclose(
            float(row["mw_pred_median"]),
            float(prefix["mw_pred_median"]),
            abs_tol=2.0e-7,
        ):
            raise ValueError(f"final prediction differs for {event}")
    final_mae = float(
        np.mean([float(row["abs_error"]) for row in final_predictions.values()])
    )
    final_station_count = sum(
        int(row["station_count"]) for row in final_predictions.values()
    )
    if final_station_count != int(metrics["station_count"]):
        raise ValueError("final prefix rows do not reproduce locked station count")
    if not math.isclose(final_mae, float(metrics["event_mae"]), abs_tol=2.0e-7):
        raise ValueError("final prefix rows do not reproduce locked Event MAE")


def _station_availability(
    prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[float, int], float, list[tuple[str, int, int]]]:
    totals: dict[float, int] = {}
    for horizon in EXPECTED_HORIZONS:
        totals[horizon] = sum(
            int(row["station_count"])
            for row in prediction_rows
            if float(row["observation_horizon_sec"]) == horizon
        )
    final_total = totals[EXPECTED_HORIZONS[-1]]
    first_full = next(
        horizon
        for index, horizon in enumerate(EXPECTED_HORIZONS)
        if all(totals[later] == final_total for later in EXPECTED_HORIZONS[index:])
    )
    first_rows = {
        str(row["event"]): int(row["station_count"])
        for row in prediction_rows
        if float(row["observation_horizon_sec"]) == EXPECTED_HORIZONS[0]
    }
    final_rows = {
        str(row["event"]): int(row["station_count"])
        for row in prediction_rows
        if float(row["observation_horizon_sec"]) == EXPECTED_HORIZONS[-1]
    }
    changes = sorted(
        (event, first_rows[event], final_rows[event])
        for event in final_rows
        if first_rows[event] != final_rows[event]
    )
    return totals, first_full, changes


def load_phase27_inputs(run_dir: Path) -> dict[str, Any]:
    internal_dir = run_dir / "internal"
    selected_dir = internal_dir / "candidate" / "results" / "selected_seed_17"
    prefix_dir = internal_dir / "candidate" / "results" / "delayed_prefix"
    paths = {
        "generator_script": Path(__file__).resolve(),
        "train_summary": run_dir / "train" / "summary.json",
        "candidate_selection": run_dir / "train" / "candidate" / "selection.json",
        "internal_summary": internal_dir / "summary.json",
        "selected_metrics": selected_dir / "metrics.json",
        "selected_event_summary": selected_dir / "event_summary.csv",
        "selected_result_registry": selected_dir / "result_registry.json",
        "delayed_prefix_cohort": prefix_dir / "cohort_contract.json",
        "delayed_prefix_horizon_metrics": prefix_dir / "horizon_metrics.json",
        "delayed_prefix_event_predictions": prefix_dir / "event_predictions.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Phase27 publication inputs: " + ", ".join(missing))

    internal_summary = _read_json(paths["internal_summary"])
    train_summary = _read_json(paths["train_summary"])
    selection = _read_json(paths["candidate_selection"])
    metrics = _read_json(paths["selected_metrics"])
    cohort = _read_json(paths["delayed_prefix_cohort"])
    registry = _read_json(paths["selected_result_registry"])
    with paths["delayed_prefix_horizon_metrics"].open(encoding="utf-8") as handle:
        horizon_metrics = json.load(handle)
    if not isinstance(horizon_metrics, list):
        raise ValueError("delayed-prefix horizon metrics are not a list")
    _validate_publication_contract(
        internal_summary=internal_summary,
        train_summary=train_summary,
        selection=selection,
        metrics=metrics,
        cohort=cohort,
        registry=registry,
    )
    predictions = _normalize_prediction_rows(
        _read_csv(paths["delayed_prefix_event_predictions"])
    )
    final_event_rows = _read_csv(paths["selected_event_summary"])
    _validate_derived_metrics(
        prediction_rows=predictions,
        horizon_metrics=horizon_metrics,
        final_event_rows=final_event_rows,
        metrics=metrics,
    )
    return {
        "paths": paths,
        "internal_summary": internal_summary,
        "train_summary": train_summary,
        "selection": selection,
        "metrics": metrics,
        "cohort": cohort,
        "registry": registry,
        "horizon_metrics": horizon_metrics,
        "prediction_rows": predictions,
        "group_rows": build_magnitude_group_rows(predictions),
        "convergence_rows": build_event_convergence_rows(predictions),
    }


def plot_magnitude_group_convergence(
    *,
    rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> tuple[Path, Path]:
    apply_pub_style()
    fig, (ax_mae, ax_bias) = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True)
    for axis in (ax_mae, ax_bias):
        style_axes(axis)
        axis.set_xticks((20, 60, 100, 140, 180, 200))
        axis.set_xlim(16, 204)
        axis.set_xlabel("Observation horizon after origin (s)")
    for spec in GROUP_SPECS:
        group = [row for row in rows if row["magnitude_group"] == spec["key"]]
        horizon = np.asarray([float(row["observation_horizon_sec"]) for row in group])
        mae = np.asarray([float(row["event_mae"]) for row in group])
        bias = np.asarray([float(row["event_bias"]) for row in group])
        count = int(group[0]["event_count"])
        label = f"{spec['label']} (n={count})"
        for axis, values in ((ax_mae, mae), (ax_bias, bias)):
            axis.plot(
                horizon,
                values,
                color=spec["color"],
                marker=spec["marker"],
                markersize=4.2,
                linewidth=1.5,
                label=label,
            )
    ax_mae.axhline(
        TARGET_ERROR_MW,
        color="#666666",
        linestyle="--",
        linewidth=0.9,
        label="0.15 Mw target",
    )
    ax_mae.set_ylim(0.0, 1.62)
    ax_mae.set_ylabel("Event-equal MAE (Mw)")
    ax_mae.set_title("Magnitude-group absolute error")
    ax_mae.legend(loc="upper right")
    ax_bias.axhline(0.0, color="#666666", linestyle="--", linewidth=0.9)
    ax_bias.set_ylim(-1.62, 0.62)
    ax_bias.set_ylabel("Event-equal bias (Mw)")
    ax_bias.set_title("Signed convergence toward catalog Mw")
    release_axis = ax_mae.secondary_xaxis(
        "top",
        functions=(
            lambda value: value + PROCESSING_DELAY_SEC,
            lambda value: value - PROCESSING_DELAY_SEC,
        ),
    )
    release_axis.set_xlabel("Release time (s)")
    release_axis.set_xticks((25, 65, 105, 145, 185, 205))
    _panel_label(ax_mae, "A", x=-0.24)
    _panel_label(ax_bias, "B", x=-0.24)
    fig.suptitle(
        "Phase27 locked internal test: magnitude-dependent convergence",
        y=1.01,
    )
    totals, first_full, _ = _station_availability(prediction_rows)
    final_total = totals[EXPECTED_HORIZONS[-1]]
    if totals[EXPECTED_HORIZONS[0]] == final_total:
        availability = f"{final_total}/{final_total} stations available at every horizon"
    else:
        availability = (
            f"Stations available: {totals[EXPECTED_HORIZONS[0]]}/{final_total} at "
            f"{EXPECTED_HORIZONS[0]:.0f} s; {final_total}/{final_total} from "
            f"{first_full:.0f} s"
        )
    fig.text(
        0.5,
        0.02,
        availability
        + "; 30/30 events; retrospective full-record >=2 cm cohort",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color="#444444",
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.21, top=0.76, wspace=0.38)
    return _save_figure(fig, output_stem)


def plot_high_magnitude_event_trajectories(
    *,
    prediction_rows: Sequence[Mapping[str, Any]],
    convergence_rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> tuple[Path, Path]:
    apply_pub_style()
    convergence = {str(row["event"]): row for row in convergence_rows}
    by_event: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        if float(row["mw_catalog"]) >= 8.0:
            by_event[str(row["event"])].append(row)
    ordered = sorted(
        by_event,
        key=lambda event: (-float(by_event[event][0]["mw_catalog"]), event),
    )
    if len(ordered) != 7:
        raise ValueError("expected seven Mw >= 8 events")

    fig, axes = plt.subplots(4, 2, figsize=(7.2, 8.25), sharex=True)
    flat = list(axes.flat)
    for index, event in enumerate(ordered):
        axis = flat[index]
        style_axes(axis)
        sequence = sorted(
            by_event[event], key=lambda row: float(row["observation_horizon_sec"])
        )
        horizon = np.asarray([float(row["observation_horizon_sec"]) for row in sequence])
        prediction = np.asarray([float(row["mw_pred_median"]) for row in sequence])
        catalog = float(sequence[0]["mw_catalog"])
        axis.axhspan(
            catalog - TARGET_ERROR_MW,
            catalog + TARGET_ERROR_MW,
            color="#009E73",
            alpha=0.10,
            linewidth=0,
            zorder=0,
        )
        axis.axhline(catalog, color="#333333", linestyle="--", linewidth=0.9)
        axis.plot(
            horizon,
            prediction,
            color="#0072B2",
            marker="o",
            markersize=3.7,
            linewidth=1.4,
            zorder=3,
        )
        summary = convergence[event]
        stable = summary["stable_within_0p15_observation_sec"]
        if stable is not None:
            stable_value = next(
                float(row["mw_pred_median"])
                for row in sequence
                if float(row["observation_horizon_sec"]) == float(stable)
            )
            axis.scatter(
                [float(stable)],
                [stable_value],
                color="#009E73",
                marker="D",
                s=24,
                zorder=4,
            )
        margin = max(0.20, 0.08 * (max(prediction.max(), catalog) - min(prediction.min(), catalog)))
        axis.set_ylim(
            min(float(prediction.min()), catalog - TARGET_ERROR_MW) - margin,
            max(float(prediction.max()), catalog + TARGET_ERROR_MW) + margin,
        )
        station_counts = [int(row["station_count"]) for row in sequence]
        if station_counts[0] != station_counts[-1]:
            station_title = (
                f"n20/n200={station_counts[0]}/{station_counts[-1]}"
            )
        else:
            station_title = f"n200={station_counts[-1]}"
        axis.set_title(f"{event} | Mw {catalog:.2f} | {station_title}")
        axis.set_xticks((20, 80, 140, 200))
        _panel_label(axis, chr(ord("A") + index), x=-0.11)
    legend_axis = flat[-1]
    legend_axis.axis("off")
    legend_axis.legend(
        handles=[
            Line2D([0], [0], color="#0072B2", marker="o", label="Event-median prediction"),
            Line2D([0], [0], color="#333333", linestyle="--", label="Catalog Mw"),
            Line2D(
                [0],
                [0],
                color="#009E73",
                marker="D",
                linestyle="none",
                label="Inside band from this point through 200 s",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.78),
    )
    legend_axis.text(
        0.5,
        0.18,
        "Shading: catalog Mw +/- 0.15\n"
        "Release occurs 5 s after observation\n"
        "n200: stations available at 200 s\n"
        "Retrospective >=2 cm cohort; prefix-causal waveforms",
        ha="center",
        va="center",
        fontsize=7,
        transform=legend_axis.transAxes,
    )
    fig.supxlabel("Observation horizon after origin (s)", y=0.027)
    fig.supylabel("Predicted event-median Mw", x=0.025)
    fig.suptitle("Phase27 Mw >= 8 event trajectories", y=0.985)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.075, top=0.94, hspace=0.46, wspace=0.22)
    return _save_figure(fig, output_stem)


def plot_convergence_time_by_magnitude(
    *, rows: Sequence[Mapping[str, Any]], output_stem: Path
) -> tuple[Path, Path]:
    apply_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharex=True, sharey=True)
    panels = (
        (
            axes[0],
            "first_within_0p15_observation_sec",
            "First entry into +/-0.15 Mw band",
        ),
        (
            axes[1],
            "stable_within_0p15_observation_sec",
            "Stable entry (all later points inside)",
        ),
    )
    for panel_index, (axis, field, title) in enumerate(panels):
        style_axes(axis)
        x_positions = _jittered_catalog_positions(rows, field=field)
        for spec in GROUP_SPECS:
            group = [row for row in rows if row["magnitude_group"] == spec["key"]]
            converged = [row for row in group if row[field] is not None]
            censored = [row for row in group if row[field] is None]
            if converged:
                axis.scatter(
                    [x_positions[str(row["event"])] for row in converged],
                    [float(row[field]) for row in converged],
                    color=spec["color"],
                    marker=spec["marker"],
                    s=30,
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=3,
                )
            if censored:
                axis.scatter(
                    [x_positions[str(row["event"])] for row in censored],
                    [CENSOR_PLOT_SEC for _ in censored],
                    facecolors="none",
                    edgecolors=spec["color"],
                    marker=spec["marker"],
                    s=42,
                    linewidth=1.2,
                    zorder=4,
                )
        axis.axhline(
            EXPECTED_HORIZONS[-1],
            color="#999999",
            linestyle=":",
            linewidth=0.8,
            zorder=1,
        )
        axis.set_xlim(5.75, 9.32)
        axis.set_ylim(12, 230)
        axis.set_xticks((6.0, 7.0, 8.0, 9.0))
        primary_ticks = (*EXPECTED_HORIZONS, CENSOR_PLOT_SEC)
        primary_labels = (*[f"{value:.0f}" for value in EXPECTED_HORIZONS], ">200")
        axis.set_yticks(primary_ticks, primary_labels)
        axis.set_xlabel("Catalog Mw")
        axis.set_title(title, fontsize=9.0)
        _panel_label(axis, chr(ord("A") + panel_index), x=-0.16)
    axes[0].set_ylabel("Observation horizon (s)")
    release_axis = axes[1].secondary_yaxis(
        "right",
        functions=(
            lambda value: value + PROCESSING_DELAY_SEC,
            lambda value: value - PROCESSING_DELAY_SEC,
        ),
    )
    release_axis.set_ylabel("Release time (s)")
    release_ticks = tuple(value + PROCESSING_DELAY_SEC for value in EXPECTED_HORIZONS)
    release_axis.set_yticks(
        (*release_ticks, CENSOR_PLOT_SEC + PROCESSING_DELAY_SEC),
        (*[f"{value:.0f}" for value in release_ticks], ">205"),
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=spec["color"],
            marker=spec["marker"],
            linestyle="none",
            label=spec["label"],
        )
        for spec in GROUP_SPECS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#666666",
            marker="o",
            markerfacecolor="none",
            linestyle="none",
            label="Right-censored beyond 200 s",
        )
    )
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.52, 0.90))
    fig.suptitle("Magnitude and time to an accurate event estimate", y=0.99)
    fig.text(
        0.5,
        0.025,
        "Small horizontal offsets separate identical coordinates. "
        "Suffix criterion is evaluated through 200 s only; retrospective >=2 cm cohort.",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color="#444444",
    )
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.22, top=0.70, wspace=0.22)
    return _save_figure(fig, output_stem)


def _csv_ready(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: "" if value is None else value for key, value in row.items()}
        for row in rows
    ]


def _group_lookup(
    rows: Sequence[Mapping[str, Any]], *, group: str, horizon: float
) -> Mapping[str, Any]:
    return next(
        row
        for row in rows
        if row["magnitude_group"] == group
        and float(row["observation_horizon_sec"]) == horizon
    )


def _median_optional(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(np.median(np.asarray(finite)))


def _jittered_catalog_positions(
    rows: Sequence[Mapping[str, Any]], *, field: str, spacing: float = 0.05
) -> dict[str, float]:
    coordinates: dict[tuple[float, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row[field]
        y = CENSOR_PLOT_SEC if value is None else float(value)
        coordinates[(round(float(row["mw_catalog"]), 6), y)].append(row)
    positions: dict[str, float] = {}
    for (mw, _), group in coordinates.items():
        ordered = sorted(group, key=lambda row: str(row["event"]))
        center = (len(ordered) - 1) / 2.0
        for index, row in enumerate(ordered):
            positions[str(row["event"])] = mw + (index - center) * spacing
    return positions


def _kaplan_meier_median_observation(
    rows: Sequence[Mapping[str, Any]],
) -> float | None:
    observations = [
        (
            float(row["stable_within_0p15_observation_sec"])
            if row["stable_within_0p15_observation_sec"] is not None
            else EXPECTED_HORIZONS[-1],
            row["stable_within_0p15_observation_sec"] is not None,
        )
        for row in rows
    ]
    survival = 1.0
    for horizon in sorted({time for time, _ in observations}):
        at_risk = sum(time >= horizon for time, _ in observations)
        events = sum(time == horizon and observed for time, observed in observations)
        if events:
            survival *= 1.0 - events / at_risk
            if survival <= 0.5:
                return horizon
    return None


def _readme_text(data: Mapping[str, Any]) -> str:
    metrics = data["metrics"]
    group_rows = data["group_rows"]
    convergence_rows = data["convergence_rows"]
    registry = data["registry"]
    high = sorted(
        [row for row in convergence_rows if float(row["mw_catalog"]) >= 8.0],
        key=lambda row: (-float(row["mw_catalog"]), str(row["event"])),
    )
    stable_count = sum(
        row["stable_within_0p15_observation_sec"] is not None
        for row in convergence_rows
    )
    lines = [
        "# Phase27 Manuscript STF Magnitude-Convergence Result",
        "",
        "> Locked internal test diagnostics for validation-selected seed 17. This is a same-event, unseen-station split, not an unseen-event test.",
        "",
        "## Headline result",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Validation Event MAE | {float(data['internal_summary']['validation_gate']['candidate']):.6f} |",
        f"| Locked test Event MAE at 200 s | {float(metrics['event_mae']):.6f} |",
        f"| Locked test Station MAE at 200 s | {float(metrics['station_mae']):.6f} |",
        f"| Test events / stations | {int(metrics['event_count'])} / {int(metrics['station_count'])} |",
        f"| Inside +/-0.15 Mw from some sampled horizon through 200 s | {stable_count}/{len(convergence_rows)} events |",
        "| Final observation / release | 200 s / 205 s |",
        "",
        "The model uses R only, predicts one nonnegative STF, and derives Mw only from the STF integral. Seed 17 was selected by internal validation before the locked test was evaluated; there is no seed averaging.",
        "",
        "The waveform prefix at every plotted horizon is causal and is released five seconds later. The displayed cohort is not end-to-end causal, because membership uses the processed radial peak over the complete 200 s record (>=2 cm). The figures must therefore be read as delayed-prefix diagnostics on a retrospective cohort.",
        "",
        "## 1. Magnitude-group convergence",
        "",
        "![Magnitude-group convergence](figures/01_magnitude_group_convergence.png)",
        "",
        "[Download PDF](figures/01_magnitude_group_convergence.pdf)",
        "",
        "High-magnitude events are strongly underestimated early: the Mw>=8 group begins at 1.5068 Mw MAE and -1.5068 Mw bias at a 20 s observation horizon. It reaches 0.1212 Mw MAE at 180 s and 0.1198 Mw at 200 s. This is a group-level delayed convergence pattern, not evidence that every large event converges later than every smaller event.",
        "",
        "| Observation / release | Mw < 7 MAE | 7 <= Mw < 8 MAE | Mw >= 8 MAE | Mw >= 8 bias |",
        "|---:|---:|---:|---:|---:|",
    ]
    for horizon in (20.0, 100.0, 140.0, 160.0, 180.0, 200.0):
        low = _group_lookup(group_rows, group="mw_lt_7", horizon=horizon)
        middle = _group_lookup(group_rows, group="mw_7_to_lt_8", horizon=horizon)
        high_group = _group_lookup(group_rows, group="mw_ge_8", horizon=horizon)
        lines.append(
            f"| {horizon:.0f} / {horizon + 5:.0f} s | {float(low['event_mae']):.4f} | "
            f"{float(middle['event_mae']):.4f} | {float(high_group['event_mae']):.4f} | "
            f"{float(high_group['event_bias']):+.4f} |"
        )
    totals, first_full, availability_changes = _station_availability(
        data["prediction_rows"]
    )
    final_total = totals[EXPECTED_HORIZONS[-1]]
    if totals[EXPECTED_HORIZONS[0]] != final_total:
        changes = ", ".join(
            f"{event} {first}->{final}"
            for event, first, final in availability_changes
        )
        lines.extend(
            [
                "",
                f"Station availability is {totals[EXPECTED_HORIZONS[0]]}/{final_total} at the 20 s observation / 25 s release and {final_total}/{final_total} from the {first_full:.0f} s observation / {first_full + 5:.0f} s release onward. Event coverage remains 30/30. The changing event is {changes} stations.",
            ]
        )
    lines.extend(
        [
            "",
            "## 2. Mw >= 8 event trajectories",
            "",
            "![High-magnitude event trajectories](figures/02_high_magnitude_event_trajectories.png)",
            "",
            "[Download PDF](figures/02_high_magnitude_event_trajectories.pdf)",
            "",
            "| Event | Catalog Mw | First within +/-0.15 | Inside from this horizon through 200 s | Final absolute error | Stations at 200 s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in high:
        first = row["first_within_0p15_observation_sec"]
        stable = row["stable_within_0p15_observation_sec"]
        lines.append(
            f"| {row['event']} | {float(row['mw_catalog']):.2f} | "
            f"{'not reached' if first is None else f'{float(first):.0f} s'} | "
            f"{'right-censored' if stable is None else f'{float(stable):.0f} s'} | "
            f"{float(row['final_abs_error']):.4f} | {int(row['final_station_count'])} |"
        )
    lines.extend(
        [
            "",
            "## 3. Magnitude and convergence time",
            "",
            "![Magnitude and convergence time](figures/03_convergence_time_by_magnitude.png)",
            "",
            "[Download PDF](figures/03_convergence_time_by_magnitude.pdf)",
            "",
            "First entry and suffix-stable entry are different. An event may briefly enter the +/-0.15 Mw band and leave it again. Suffix-stable entry requires every later sampled horizon through 200 s to remain inside; it does not claim stability after 200 s. Events that fail this condition are plotted as right-censored rather than assigned a false 200 s convergence time. Small deterministic horizontal offsets separate events with identical plotted coordinates.",
            "",
            "| Magnitude group | Events | Median first entry among reached | Censor-aware stable median | Stable by 200 s |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for spec in GROUP_SPECS:
        group = [row for row in convergence_rows if row["magnitude_group"] == spec["key"]]
        first = _median_optional(
            [row["first_within_0p15_observation_sec"] for row in group]
        )
        stable = _kaplan_meier_median_observation(group)
        converged = sum(
            row["stable_within_0p15_observation_sec"] is not None for row in group
        )
        lines.append(
            f"| {spec['label']} | {len(group)} | "
            f"{'not reached' if first is None else f'{first:.0f} s'} | "
            f"{'not reached' if stable is None else f'{stable:.0f} s observation / {stable + 5:.0f} s release'} | "
            f"{converged}/{len(group)} |"
        )
    checkpoint_sha = str(registry["checkpoint"]["sha256"])
    lines.extend(
        [
            "",
            "## Data and provenance",
            "",
            "- [All event predictions by horizon](event_predictions_by_horizon.csv)",
            "- [Magnitude-group horizon metrics](magnitude_group_horizon_metrics.csv)",
            "- [Event convergence summary](event_convergence_summary.csv)",
            "- [Publication manifest](publication_manifest.json)",
            "- [Reproducible generator](../../../scripts/plotting/plot_phase27_magnitude_convergence.py)",
            f"- Model/evaluation commit: `{EXPECTED_GIT_COMMIT}`",
            f"- Selected checkpoint SHA-256: `{checkpoint_sha}`",
            "- Formal run: `phase27-manuscript-stf-event-loss-weighted-20260724T221820Z-e02aeca`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_manifest(
    *, path: Path, inputs: Mapping[str, Path], outputs: Mapping[str, Path]
) -> None:
    payload = {
        "schema_version": 1,
        "inputs": {
            name: {"path": str(source), "sha256": _sha256(source)}
            for name, source in sorted(inputs.items())
        },
        "outputs": {
            name: {"path": str(output), "sha256": _sha256(output)}
            for name, output in sorted(outputs.items())
        },
        "analysis_contract": {
            "target_absolute_error_mw": TARGET_ERROR_MW,
            "observation_horizons_sec": list(EXPECTED_HORIZONS),
            "processing_delay_sec": PROCESSING_DELAY_SEC,
            "selected_seed": EXPECTED_SELECTED_SEED,
            "ensemble_used": False,
            "waveform_prefix_causal": True,
            "station_selection_causal": False,
            "right_censor_horizon_sec": EXPECTED_HORIZONS[-1],
        },
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def generate_bundle(*, run_dir: Path, output_dir: Path) -> dict[str, Path]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    data = load_phase27_inputs(run_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True)
    outputs: dict[str, Path] = {}

    figure_specs = (
        (
            "01_magnitude_group_convergence",
            lambda stem: plot_magnitude_group_convergence(
                rows=data["group_rows"],
                prediction_rows=data["prediction_rows"],
                output_stem=stem,
            ),
        ),
        (
            "02_high_magnitude_event_trajectories",
            lambda stem: plot_high_magnitude_event_trajectories(
                prediction_rows=data["prediction_rows"],
                convergence_rows=data["convergence_rows"],
                output_stem=stem,
            ),
        ),
        (
            "03_convergence_time_by_magnitude",
            lambda stem: plot_convergence_time_by_magnitude(
                rows=data["convergence_rows"], output_stem=stem
            ),
        ),
    )
    for name, plotter in figure_specs:
        png, pdf = plotter(figures_dir / name)
        outputs[f"{name}.png"] = png
        outputs[f"{name}.pdf"] = pdf

    csv_outputs = {
        "event_predictions_by_horizon.csv": data["prediction_rows"],
        "magnitude_group_horizon_metrics.csv": data["group_rows"],
        "event_convergence_summary.csv": data["convergence_rows"],
    }
    for name, rows in csv_outputs.items():
        path = output_dir / name
        _write_csv(path, _csv_ready(rows))
        outputs[name] = path

    readme = output_dir / "README.md"
    readme.write_text(_readme_text(data), encoding="utf-8")
    outputs["README.md"] = readme

    manifest = output_dir / "publication_manifest.json"
    _write_manifest(path=manifest, inputs=data["paths"], outputs=outputs)
    outputs["publication_manifest.json"] = manifest
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Phase27 magnitude-convergence result gallery"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = generate_bundle(
        run_dir=args.run_dir.resolve(), output_dir=args.output_dir.resolve()
    )
    print(f"generated {len(artifacts)} Phase27 publication artifacts")


if __name__ == "__main__":
    main()
