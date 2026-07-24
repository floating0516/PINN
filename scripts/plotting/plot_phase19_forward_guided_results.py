from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _style import apply_pub_style, style_axes  # noqa: E402
from plot_phase17_causal_online_results import (  # noqa: E402
    EVENT_ORDER,
    SEEDS,
    SEED_COLORS,
    TARGET_MAE,
    _event_label,
    _full_coverage_horizon,
    _panel_label,
    _read_csv,
    _read_json,
    _save_figure,
    _sha256,
    _stable_target_horizon,
    _write_csv,
    build_dynamic_station_rows,
    build_seed_selection_rows,
    load_final_event_rows,
    load_horizon_rows,
    plot_dynamic_station_selection,
    plot_seed_selection,
)


METHOD = "causal_forward_guided_event_neural_v2"
PHASE17_METHOD = "causal_radial_event_neural_v1"


def load_training_logs(
    run_root: Path,
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    logs: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    expected_lengths: dict[str, int] = {}
    for seed in SEEDS:
        rows = _read_csv(run_root / f"seed_{seed}" / "training_log.csv")
        if {row["phase"] for row in rows} != {"anchor", "deep"}:
            raise ValueError(f"training phases differ for seed {seed}")
        seed_log: dict[str, dict[str, np.ndarray]] = {}
        for phase in ("anchor", "deep"):
            phase_rows = [row for row in rows if row["phase"] == phase]
            epochs = np.asarray([int(row["epoch"]) for row in phase_rows])
            if not np.array_equal(epochs, np.arange(1, len(phase_rows) + 1)):
                raise ValueError(f"{phase} epochs are not consecutive for seed {seed}")
            if phase in expected_lengths and len(phase_rows) != expected_lengths[phase]:
                raise ValueError(f"{phase} logs have different lengths")
            expected_lengths.setdefault(phase, len(phase_rows))
            columns = (
                ("train_total_loss", "validation_online_mae")
                if phase == "anchor"
                else (
                    "train_total_loss",
                    "train_L_MSE",
                    "train_L_synth",
                    "train_L_mag",
                    "train_L_shape",
                    "validation_total_loss",
                    "validation_L_MSE",
                    "validation_L_synth",
                    "validation_L_mag",
                    "validation_L_shape",
                    "validation_online_mae",
                )
            )
            values = {
                column: np.asarray(
                    [float(row[column]) for row in phase_rows], dtype=np.float64
                )
                for column in columns
            }
            if any(not np.isfinite(value).all() for value in values.values()):
                raise ValueError(f"training log is non-finite for seed {seed}")
            seed_log[phase] = {"epoch": epochs, **values}
        logs[seed] = seed_log
    return logs


def _parse_extended_prediction(source: Mapping[str, str]) -> dict[str, Any]:
    row = {
        "event": source["event"],
        "horizon_sec": float(source["horizon_sec"]),
        "mw_pred": float(source["mw_pred"]),
        "mw_reference": float(source["mw_reference"]),
        "error": float(source["error"]),
        "abs_error": float(source["abs_error"]),
        "anchor_mw": float(source["anchor_mw"]),
        "neural_residual_mw": float(source["neural_residual_mw"]),
        "active_station_count": int(source["active_station_count"]),
        "used_station_count": int(source["used_station_count"]),
        "used_stations": source["used_stations"],
    }
    numeric = [
        value for key, value in row.items() if key not in {"event", "used_stations"}
    ]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("prediction row contains a non-finite value")
    if not math.isclose(
        row["anchor_mw"] + row["neural_residual_mw"],
        row["mw_pred"],
        abs_tol=5.0e-7,
    ):
        raise ValueError("prediction does not equal anchor plus residual")
    if not math.isclose(
        row["mw_pred"] - row["mw_reference"], row["error"], abs_tol=2.0e-7
    ) or not math.isclose(abs(row["error"]), row["abs_error"], abs_tol=2.0e-7):
        raise ValueError("prediction error fields are inconsistent")
    if row["active_station_count"] < row["used_station_count"]:
        raise ValueError("used station count exceeds active station count")
    stations = str(row["used_stations"]).split("|")
    if len(stations) != row["used_station_count"] or len(stations) != len(set(stations)):
        raise ValueError("used station list is inconsistent")
    return row


def load_extended_online_rows(
    path: Path,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {label: [] for label in EVENT_ORDER}
    for source in _read_csv(path):
        row = _parse_extended_prediction(source)
        grouped[_event_label(str(row["event"]))].append(row)
    for label, rows in grouped.items():
        if not rows:
            raise ValueError(f"online predictions do not cover {label}")
        rows.sort(key=lambda row: float(row["horizon_sec"]))
        horizons = np.asarray([float(row["horizon_sec"]) for row in rows])
        if not np.allclose(np.diff(horizons), 1.0, atol=0.0):
            raise ValueError(f"online horizons are not consecutive for {label}")
        if len({float(row["mw_reference"]) for row in rows}) != 1:
            raise ValueError(f"reference magnitude changes for {label}")
        predictions = np.asarray([float(row["mw_pred"]) for row in rows])
        station_sets = {str(row["used_stations"]) for row in rows}
        if np.ptp(predictions) <= 0.0 or len(station_sets) < 2:
            raise ValueError(f"online state never changes for {label}")
    return grouped


def _selected_log_row(
    logs: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    summary: Mapping[str, Any],
    seed: int,
) -> dict[str, float]:
    epoch = int(summary["seed_summaries"][str(seed)]["best_deep_epoch"])
    deep = logs[seed]["deep"]
    index = epoch - 1
    if int(deep["epoch"][index]) != epoch:
        raise ValueError("selected deep epoch is missing from the log")
    return {
        name: float(values[index])
        for name, values in deep.items()
        if name != "epoch"
    }


def build_ablation_rows(
    *,
    full_summary: Mapping[str, Any],
    no_synth_summary: Mapping[str, Any],
    full_logs: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    no_synth_logs: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        full_log = _selected_log_row(full_logs, full_summary, seed)
        no_synth_log = _selected_log_row(no_synth_logs, no_synth_summary, seed)
        full_online = float(
            full_summary["seed_summaries"][str(seed)]["validation_online_mae"]
        )
        no_synth_online = float(
            no_synth_summary["seed_summaries"][str(seed)]["validation_online_mae"]
        )
        rows.append(
            {
                "seed": seed,
                "full_validation_online_mae": full_online,
                "no_synth_validation_online_mae": no_synth_online,
                "online_delta_no_synth_minus_full": no_synth_online - full_online,
                "full_validation_L_synth": full_log["validation_L_synth"],
                "no_synth_validation_L_synth": no_synth_log["validation_L_synth"],
                "L_synth_delta_no_synth_minus_full": (
                    no_synth_log["validation_L_synth"]
                    - full_log["validation_L_synth"]
                ),
            }
        )
    return rows


def validate_inputs(
    *,
    full_summary: Mapping[str, Any],
    no_synth_summary: Mapping[str, Any],
    phase17_summary: Mapping[str, Any],
    final_rows: Sequence[Mapping[str, Any]],
    online: Mapping[str, Sequence[Mapping[str, Any]]],
    full_horizons: Sequence[Mapping[str, Any]],
    no_synth_horizons: Sequence[Mapping[str, Any]],
    phase17_horizons: Sequence[Mapping[str, Any]],
) -> None:
    if full_summary.get("method") != METHOD or no_synth_summary.get("method") != METHOD:
        raise ValueError("forward-guided summaries have the wrong method")
    if phase17_summary.get("method") != PHASE17_METHOD:
        raise ValueError("Phase17 summary has the wrong method")
    if no_synth_summary.get("ablation") != "no_forward_loss":
        raise ValueError("ablation summary is not marked no_forward_loss")
    if full_summary["loss"] != {
        "lambda_MSE": 1.0,
        "lambda_synth": 0.5,
        "lambda_mag": 1.0,
        "lambda_shape": 0.1,
    }:
        raise ValueError("full loss weights differ from the frozen contract")
    if float(no_synth_summary["loss"]["lambda_synth"]) != 0.0:
        raise ValueError("no-forward run retains a nonzero synth weight")
    for summary in (full_summary, no_synth_summary, phase17_summary):
        if bool(summary.get("uses_ensemble")):
            raise ValueError("publication input unexpectedly uses a seed ensemble")
        if bool(summary.get("uses_future_waveform")) or bool(
            summary.get("uses_final_peak_for_station_selection")
        ):
            raise ValueError("publication input violates the causal contract")
    if not bool(full_summary.get("uses_original_four_term_loss")):
        raise ValueError("main model does not declare the four-term loss")
    if bool(no_synth_summary.get("uses_original_four_term_loss")):
        raise ValueError("ablation incorrectly declares the four-term loss")

    errors = np.asarray([float(row["error"]) for row in final_rows])
    frozen = full_summary["external"]["final_metrics"]
    reproduced = (
        float(np.mean(np.abs(errors))),
        float(np.sqrt(np.mean(np.square(errors)))),
        float(np.mean(errors)),
    )
    expected = (
        float(frozen["event_mae"]),
        float(frozen["event_rmse"]),
        float(frozen["event_bias"]),
    )
    if any(not math.isclose(a, b, abs_tol=2.0e-7) for a, b in zip(reproduced, expected)):
        raise ValueError("final event metrics do not reproduce")
    flattened = [row for label in EVENT_ORDER for row in online[label]]
    if len(flattened) != int(full_summary["external"]["online_metrics"]["snapshot_count"]):
        raise ValueError("online snapshot count differs from the summary")
    if any(
        abs(float(row["neural_residual_mw"])) > 1.0e-12
        for row in flattened
        if float(row["horizon_sec"]) == 200.0
    ):
        raise ValueError("final neural residual is not exactly zero")
    _align_horizon_rows(full_horizons, no_synth_horizons, phase17_horizons)


def plot_training_dynamics(
    *,
    logs: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    summary: Mapping[str, Any],
    output_stem: Path,
) -> tuple[Path, Path]:
    apply_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    for axis in axes.flat:
        style_axes(axis)
    selected = int(summary["selection"]["selected_seed"])
    selected_deep = logs[selected]["deep"]

    axes[0, 0].plot(
        selected_deep["epoch"],
        selected_deep["train_total_loss"],
        color="#0072B2",
        label="Train total",
    )
    axes[0, 0].plot(
        selected_deep["epoch"],
        selected_deep["validation_total_loss"],
        color="#D55E00",
        label="Validation total",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title(f"Seed {selected}: weighted total loss")
    axes[0, 0].set_ylabel("Loss (log scale)")
    axes[0, 0].legend(loc="upper right")

    components = (
        ("validation_L_MSE", 1.0, "STF MSE", "#0072B2"),
        ("validation_L_synth", 0.5, "Forward waveform", "#D55E00"),
        ("validation_L_mag", 1.0, "Catalog Mw", "#009E73"),
        ("validation_L_shape", 0.1, "STF shape", "#CC79A7"),
    )
    for key, weight, label, color in components:
        axes[0, 1].plot(
            selected_deep["epoch"],
            weight * selected_deep[key],
            color=color,
            label=f"{weight:g} x {label}",
        )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Weighted validation loss terms")
    axes[0, 1].set_ylabel("Contribution (log scale)")
    axes[0, 1].legend(loc="center right", fontsize=6.6)

    for seed in SEEDS:
        deep = logs[seed]["deep"]
        width = 1.5 if seed == selected else 0.9
        axes[1, 0].plot(
            deep["epoch"],
            deep["validation_online_mae"],
            color=SEED_COLORS[seed],
            linewidth=width,
            label=f"Seed {seed}" + (" selected" if seed == selected else ""),
        )
        best_epoch = int(summary["seed_summaries"][str(seed)]["best_deep_epoch"])
        axes[1, 0].scatter(
            [best_epoch],
            [deep["validation_online_mae"][best_epoch - 1]],
            color=SEED_COLORS[seed],
            edgecolors="white",
            linewidths=0.5,
            s=22,
            zorder=5,
        )
        anchor = logs[seed]["anchor"]
        axes[1, 1].plot(
            anchor["epoch"],
            anchor["validation_online_mae"],
            color=SEED_COLORS[seed],
            linewidth=width,
        )
        anchor_epoch = int(
            summary["seed_summaries"][str(seed)]["best_anchor_epoch"]
        )
        axes[1, 1].scatter(
            [anchor_epoch],
            [anchor["validation_online_mae"][anchor_epoch - 1]],
            color=SEED_COLORS[seed],
            edgecolors="white",
            linewidths=0.5,
            s=22,
            zorder=5,
        )
    axes[1, 0].set_title("Deep-stage validation online MAE")
    axes[1, 0].set_ylabel("MAE (Mw)")
    axes[1, 0].legend(loc="upper right")
    axes[1, 1].set_title("Anchor validation final MAE")
    axes[1, 1].set_ylabel("MAE (Mw)")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
    for axis, label in zip(axes.flat, "ABCD"):
        _panel_label(axis, label)
    fig.suptitle(
        "Causal TCN/Transformer training with the four-term loss",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.10,
        top=0.90,
        wspace=0.27,
        hspace=0.34,
    )
    return _save_figure(fig, output_stem)


def _aligned_horizon_arrays(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([float(row["horizon_sec"]) for row in rows]),
        np.asarray([float(row["event_mae"]) for row in rows]),
    )


def _align_horizon_rows(
    *tables: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], ...]:
    if not tables or any(not table for table in tables):
        raise ValueError("comparison horizon table is empty")
    indexed = [
        {float(row["horizon_sec"]): row for row in table}
        for table in tables
    ]
    common = sorted(set.intersection(*(set(table) for table in indexed)))
    if not common:
        raise ValueError("comparison horizon tables do not overlap")
    if any(float(table[-1]["horizon_sec"]) != common[-1] for table in tables):
        raise ValueError("comparison horizon tables have different endpoints")
    if len(common) > 1 and not np.allclose(np.diff(common), 1.0, atol=0.0):
        raise ValueError("common comparison horizons are not consecutive")
    return tuple([[table[horizon] for horizon in common] for table in indexed])


def plot_convergence_comparison(
    *,
    full_rows: Sequence[Mapping[str, Any]],
    no_synth_rows: Sequence[Mapping[str, Any]],
    phase17_rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> tuple[Path, Path]:
    apply_pub_style()
    full_rows, no_synth_rows, phase17_rows = _align_horizon_rows(
        full_rows,
        no_synth_rows,
        phase17_rows,
    )
    horizon, full = _aligned_horizon_arrays(full_rows)
    _, no_synth = _aligned_horizon_arrays(no_synth_rows)
    _, phase17 = _aligned_horizon_arrays(phase17_rows)
    coverage = np.asarray([int(row["event_count"]) for row in full_rows])
    full_coverage = _full_coverage_horizon(full_rows)
    stable = {
        "Full four-term": _stable_target_horizon(full_rows),
        "No forward loss": _stable_target_horizon(no_synth_rows),
        "Phase17": _stable_target_horizon(phase17_rows),
    }

    fig, (ax_error, ax_gain) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.55, 0.8], "hspace": 0.10},
    )
    for axis in (ax_error, ax_gain):
        style_axes(axis)
        axis.axvspan(horizon[0], full_coverage, color="#999999", alpha=0.10)
        axis.axvline(full_coverage, color="#777777", linestyle=":", linewidth=0.9)
    mask = horizon >= 30.0
    ax_error.plot(horizon[mask], full[mask], color="#0072B2", label="Full four-term")
    ax_error.plot(
        horizon[mask],
        no_synth[mask],
        color="#D55E00",
        linestyle="--",
        label="No forward loss",
    )
    ax_error.plot(
        horizon[mask],
        phase17[mask],
        color="#777777",
        linestyle=":",
        linewidth=1.3,
        label="Phase17 data baseline",
    )
    ax_error.axhline(TARGET_MAE, color="#333333", linestyle="--", linewidth=0.8)
    ax_error.set_yscale("log")
    ticks = [0.1, 0.15, 0.3, 0.6]
    ax_error.yaxis.set_major_locator(FixedLocator(ticks))
    ax_error.yaxis.set_major_formatter(FixedFormatter([str(value) for value in ticks]))
    ax_error.minorticks_off()
    ax_error.set_ylim(0.085, 0.75)
    ax_error.set_ylabel("Event MAE (Mw; log scale)")
    ax_error.legend(loc="upper right")
    ax_error.text(32, 0.158, "0.15 target", fontsize=7, va="bottom")
    annotation_y = {"Full four-term": 0.63, "No forward loss": 0.50, "Phase17": 0.40}
    color_by_name = {
        "Full four-term": "#0072B2",
        "No forward loss": "#D55E00",
        "Phase17": "#666666",
    }
    for name, value in stable.items():
        if value is not None:
            ax_error.text(
                float(value) + 2,
                annotation_y[name],
                f"{name}: {value:.0f} s",
                color=color_by_name[name],
                fontsize=6.8,
                va="top",
            )

    common = horizon >= full_coverage
    ax_gain.plot(
        horizon[common],
        phase17[common] - full[common],
        color="#009E73",
        label="Gain vs Phase17",
    )
    ax_gain.plot(
        horizon[common],
        no_synth[common] - full[common],
        color="#CC79A7",
        label="Gain from forward loss",
    )
    ax_gain.axhline(0.0, color="#555555", linewidth=0.8)
    ax_gain.fill_between(
        horizon[common],
        0.0,
        phase17[common] - full[common],
        color="#009E73",
        alpha=0.12,
    )
    ax_gain.set_ylabel("MAE reduction (Mw)")
    ax_gain.set_xlabel("Global horizon after origin time (s)")
    ax_gain.legend(loc="upper right")
    ax_gain.text(
        full_coverage + 2,
        ax_gain.get_ylim()[0] * 0.75,
        f"8/8 coverage from {full_coverage:.0f} s",
        fontsize=6.8,
        va="bottom",
    )
    ax_gain.set_xlim(30.0, 200.0)
    _panel_label(ax_error, "A")
    _panel_label(ax_gain, "B")
    fig.suptitle(
        "Forward-guided model reaches stable target 77 seconds earlier",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.11, top=0.91)
    return _save_figure(fig, output_stem)


def plot_forward_ablation(
    *,
    rows: Sequence[Mapping[str, Any]],
    full_horizons: Sequence[Mapping[str, Any]],
    no_synth_horizons: Sequence[Mapping[str, Any]],
    phase17_horizons: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> tuple[Path, Path]:
    apply_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    for axis in axes.flat:
        style_axes(axis)
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.34
    full_online = np.asarray([float(row["full_validation_online_mae"]) for row in rows])
    no_online = np.asarray([float(row["no_synth_validation_online_mae"]) for row in rows])
    axes[0, 0].bar(x - width / 2, full_online, width, color="#0072B2", label="Full")
    axes[0, 0].bar(
        x + width / 2, no_online, width, color="#D55E00", label="No forward"
    )
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f"Seed {int(row['seed'])}" for row in rows])
    axes[0, 0].set_ylabel("Validation online MAE (Mw)")
    axes[0, 0].set_ylim(0.30, 0.43)
    axes[0, 0].legend(loc="upper left")
    axes[0, 0].set_title(
        "Validation online MAE\n(seed-dependent effect)", fontsize=8.6
    )

    full_synth = np.asarray([float(row["full_validation_L_synth"]) for row in rows])
    no_synth = np.asarray([float(row["no_synth_validation_L_synth"]) for row in rows])
    axes[0, 1].bar(x - width / 2, full_synth, width, color="#0072B2")
    axes[0, 1].bar(x + width / 2, no_synth, width, color="#D55E00")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f"Seed {int(row['seed'])}" for row in rows])
    axes[0, 1].set_ylabel("Validation forward-waveform loss")
    lower = min(float(full_synth.min()), float(no_synth.min())) - 0.002
    upper = max(float(full_synth.max()), float(no_synth.max())) + 0.002
    axes[0, 1].set_ylim(lower, upper)
    axes[0, 1].set_title(
        "Forward-waveform mismatch\n(lower with full loss)", fontsize=8.6
    )

    stable_values = [
        float(_stable_target_horizon(full_horizons) or np.nan),
        float(_stable_target_horizon(no_synth_horizons) or np.nan),
        float(_stable_target_horizon(phase17_horizons) or np.nan),
    ]
    stable_bars = axes[1, 0].bar(
        np.arange(3),
        stable_values,
        color=["#0072B2", "#D55E00", "#888888"],
    )
    axes[1, 0].set_xticks(np.arange(3))
    axes[1, 0].set_xticklabels(["Full", "No forward", "Phase17"])
    axes[1, 0].set_ylabel("First stable <=0.15 horizon (s)")
    axes[1, 0].set_ylim(0, 190)
    axes[1, 0].set_title(
        "Stable <=0.15 horizon\n(2 s earlier)", fontsize=8.6
    )
    for bar, value in zip(stable_bars, stable_values):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 4,
            f"{value:.0f} s",
            ha="center",
            fontsize=7,
        )

    full_by_t = {int(float(row["horizon_sec"])): row for row in full_horizons}
    no_by_t = {int(float(row["horizon_sec"])): row for row in no_synth_horizons}
    key_horizons = (60, 90, 120, 150, 180)
    milli_gain = 1000.0 * np.asarray(
        [
            float(no_by_t[t]["event_mae"]) - float(full_by_t[t]["event_mae"])
            for t in key_horizons
        ]
    )
    axes[1, 1].plot(
        key_horizons,
        milli_gain,
        color="#009E73",
        marker="o",
        markersize=4,
    )
    axes[1, 1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1, 1].set_xticks(key_horizons)
    axes[1, 1].set_xlabel("Global horizon (s)")
    axes[1, 1].set_ylabel("Forward-loss gain (0.001 Mw)")
    axes[1, 1].set_title(
        "External online gain\n(millimagnitude scale)", fontsize=8.6
    )
    for axis, label in zip(axes.flat, "ABCD"):
        _panel_label(axis, label, x=-0.18, y=1.16)
    fig.suptitle(
        "Matched no-forward-loss ablation",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.10,
        top=0.87,
        wspace=0.27,
        hspace=0.50,
    )
    return _save_figure(fig, output_stem)


def plot_event_trajectories(
    *,
    online: Mapping[str, Sequence[Mapping[str, Any]]],
    final_rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> tuple[Path, Path]:
    apply_pub_style()
    final_by_label = {_event_label(str(row["event"])): row for row in final_rows}
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 7.8), sharex=True)
    for index, (axis, label) in enumerate(zip(axes.flat, EVENT_ORDER)):
        style_axes(axis)
        rows = online[label]
        horizon = np.asarray([float(row["horizon_sec"]) for row in rows])
        prediction = np.asarray([float(row["mw_pred"]) for row in rows])
        anchor = np.asarray([float(row["anchor_mw"]) for row in rows])
        reference = float(rows[0]["mw_reference"])
        final_error = float(final_by_label[label]["abs_error"])
        axis.axhspan(
            reference - TARGET_MAE,
            reference + TARGET_MAE,
            color="#009E73",
            alpha=0.12,
        )
        axis.axhline(reference, color="#222222", linestyle="--", linewidth=0.9)
        axis.plot(horizon, anchor, color="#888888", linestyle=":", linewidth=0.9)
        axis.plot(horizon, prediction, color="#0072B2", linewidth=1.1)
        axis.scatter(
            [horizon[-1]],
            [prediction[-1]],
            color="#009E73" if final_error <= TARGET_MAE else "#D55E00",
            edgecolors="white",
            linewidths=0.5,
            s=23,
            zorder=5,
        )
        low = min(float(prediction.min()), float(anchor.min()), reference - TARGET_MAE)
        high = max(float(prediction.max()), float(anchor.max()), reference + TARGET_MAE)
        margin = max(0.08, 0.06 * (high - low))
        axis.set_ylim(low - margin, high + margin)
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.set_title(f"{label}   final |error|={final_error:.3f}", fontsize=8.4)
        axis.text(
            0.02,
            0.05,
            f"starts {horizon[0]:.0f} s",
            transform=axis.transAxes,
            fontsize=6.4,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1},
        )
        axis.set_xlim(0, 200)
        if index % 2 == 0:
            axis.set_ylabel("Magnitude (Mw)")
        if index >= 6:
            axis.set_xlabel("Global horizon (s)")
        _panel_label(axis, chr(ord("A") + index))
    fig.legend(
        handles=[
            Line2D([0], [0], color="#0072B2", label="Forward-guided prediction"),
            Line2D([0], [0], color="#888888", linestyle=":", label="Anchor"),
            Line2D([0], [0], color="#222222", linestyle="--", label="USGS reference"),
            Line2D(
                [0],
                [0],
                color="#009E73",
                linewidth=6,
                alpha=0.35,
                label="Reference +/- 0.15 Mw",
            ),
        ],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.suptitle(
        "Eight-event causal trajectories: deep residual converges to anchor",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.07,
        top=0.90,
        wspace=0.24,
        hspace=0.38,
    )
    return _save_figure(fig, output_stem)


def _readme_text(
    *,
    summary: Mapping[str, Any],
    seed_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    full_horizons: Sequence[Mapping[str, Any]],
    no_synth_horizons: Sequence[Mapping[str, Any]],
    phase17_horizons: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
    station_rows: Sequence[Mapping[str, Any]],
) -> str:
    metrics = summary["external"]["final_metrics"]
    online_mae = float(summary["external"]["online_metrics"]["event_equal_online_mae"])
    stable = _stable_target_horizon(full_horizons)
    full_coverage = _full_coverage_horizon(full_horizons)
    passes = sum(float(row["abs_error"]) <= TARGET_MAE for row in final_rows)
    full_by_t = {int(float(row["horizon_sec"])): row for row in full_horizons}
    no_by_t = {int(float(row["horizon_sec"])): row for row in no_synth_horizons}
    phase17_by_t = {int(float(row["horizon_sec"])): row for row in phase17_horizons}
    lines = [
        "# Phase19 Causal Forward-Guided R-only Event Neural Model",
        "",
        "> Fixed eight-event development validation. The events influenced feature and gate decisions and are not an unbiased final paper test set.",
        "",
        "## Headline result",
        "",
        "| Selected seed | Ensemble | Coverage | All-second MAE | 200 s MAE | 200 s RMSE | Bias | Events <=0.15 | Stable <=0.15 from |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {int(summary['selection']['selected_seed'])} | no | 8/8 | {online_mae:.6f} | "
        f"{float(metrics['event_mae']):.6f} | {float(metrics['event_rmse']):.6f} | "
        f"{float(metrics['event_bias']):+.6f} | {passes}/8 | {stable:.0f} s |",
        "",
        "This is a causal forward-guided multi-task neural network, not a PINN. It retains causal TCN and masked Transformer layers plus the original STF MSE, forward-waveform, catalog-Mw, and STF-shape losses. At each second, dynamic top-five selection uses only the released R prefix after a conservative six-second processing delay.",
        "",
        "The deep Mw residual is active online and is gated to exactly zero at 200 seconds. The final 0.131990 Mw result therefore comes from the stable amplitude-distance anchor; the deep branch's main scalar benefit is earlier convergence, while its shared STF remains trained by all four losses.",
        "",
        "## 1. Four-term training dynamics",
        "",
        "![Four-term training dynamics](figures/01_training_dynamics.png)",
        "",
        "[Download PDF](figures/01_training_dynamics.pdf)",
        "",
        "The full loss is `1.0 L_MSE + 0.5 L_synth + 1.0 L_mag + 0.1 L_shape`. The forward model uses absolute P/S delays and the signed full-radiation coefficients associated with Glehman et al. (2026), DOI `10.1029/2025JB033222`.",
        "",
        "## 2. Internal single-seed selection",
        "",
        "![Internal single-seed selection](figures/02_seed_selection.png)",
        "",
        "[Download PDF](figures/02_seed_selection.pdf)",
        "",
        "| Seed | Validation online MAE | Validation final MAE | Selected |",
        "|---:|---:|---:|:---:|",
    ]
    for row in seed_rows:
        lines.append(
            f"| {int(row['seed'])} | {float(row['validation_online_mae']):.6f} | "
            f"{float(row['validation_final_mae']):.6f} | "
            f"{'yes' if bool(row['selected']) else 'no'} |"
        )
    lines.extend(
        [
            "",
            "All three seeds were trained before any external waveform was loaded. Seed 73 minimizes the predeclared internal all-second validation MAE. External evaluation uses only seed 73; there is no averaging or per-event seed choice.",
            "",
            "## 3. External online convergence",
            "",
            "![External online convergence](figures/03_online_convergence.png)",
            "",
            "[Download PDF](figures/03_online_convergence.pdf)",
            "",
            f"All eight events are covered continuously from {full_coverage:.0f} seconds. The full model remains <=0.15 Mw from {stable:.0f} seconds, compared with 91 seconds without the forward loss and 166 seconds for Phase17.",
            "",
            "| Horizon | Full model | No forward loss | Phase17 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for horizon in (60, 90, 120, 150, 180, 200):
        lines.append(
            f"| {horizon} s | {float(full_by_t[horizon]['event_mae']):.6f} | "
            f"{float(no_by_t[horizon]['event_mae']):.6f} | "
            f"{float(phase17_by_t[horizon]['event_mae']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 4. Matched forward-loss ablation",
            "",
            "![Matched forward-loss ablation](figures/04_forward_loss_ablation.png)",
            "",
            "[Download PDF](figures/04_forward_loss_ablation.pdf)",
            "",
            "The ablation changes only `lambda_synth: 0.5 -> 0.0`. The forward loss lowers validation waveform mismatch for all three seeds, improves three-seed mean validation online MAE by 0.000920 Mw, improves selected-seed external all-second MAE by 0.000842 Mw, and advances stable <=0.15 performance by two seconds. This is a small physical-consistency benefit, not the source of final accuracy.",
            "",
            "| Seed | Full validation online MAE | No-forward MAE | No-forward minus full | Full L_synth | No-forward L_synth |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ablation_rows:
        lines.append(
            f"| {int(row['seed'])} | {float(row['full_validation_online_mae']):.6f} | "
            f"{float(row['no_synth_validation_online_mae']):.6f} | "
            f"{float(row['online_delta_no_synth_minus_full']):+.6f} | "
            f"{float(row['full_validation_L_synth']):.6f} | "
            f"{float(row['no_synth_validation_L_synth']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 5. Eight event trajectories",
            "",
            "![Eight event trajectories](figures/05_event_trajectories.png)",
            "",
            "[Download PDF](figures/05_event_trajectories.pdf)",
            "",
            "| Event | Reference Mw | Predicted Mw at 200 s | Absolute error | Active/used stations |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in final_rows:
        lines.append(
            f"| {_event_label(str(row['event']))} | {float(row['mw_reference']):.1f} | "
            f"{float(row['mw_pred']):.3f} | {float(row['abs_error']):.3f} | "
            f"{int(row['active_station_count'])}/{int(row['used_station_count'])} |"
        )
    lines.extend(
        [
            "",
            "## 6. Dynamic station selection",
            "",
            "![Dynamic station selection](figures/06_dynamic_station_selection.png)",
            "",
            "[Download PDF](figures/06_dynamic_station_selection.pdf)",
            "",
            "| Event | First prediction | Distinct station sets | Set changes | Final active/used |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in station_rows:
        lines.append(
            f"| {row['event']} | {float(row['first_horizon_sec']):.0f} s | "
            f"{int(row['distinct_used_station_sets'])} | "
            f"{int(row['station_set_change_count'])} | "
            f"{int(row['final_active_station_count'])}/{int(row['final_used_station_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Data and provenance",
            "",
            "- [Seed selection](seed_selection.csv)",
            "- [Forward-loss ablation](forward_loss_ablation.csv)",
            "- [Final event predictions](external_final_event_predictions.csv)",
            "- [Full-model horizon metrics](external_horizon_metrics.csv)",
            "- [No-forward horizon metrics](no_synth_horizon_metrics.csv)",
            "- [Phase17 horizon metrics](phase17_horizon_metrics.csv)",
            "- [Full-model per-second predictions](external_online_predictions.csv)",
            "- [Dynamic station summary](dynamic_station_summary.csv)",
            "- [Publication manifest](publication_manifest.json)",
            "- Main model commit: `c7c1736`",
            "- Ablation config commit: `de2149b`",
            "- Outcome documentation commit: `9904903`",
            "",
        ]
    )
    return "\n".join(lines)


def write_manifest(
    *,
    path: Path,
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
) -> None:
    payload = {
        "inputs": {
            name: {"path": str(source), "sha256": _sha256(source)}
            for name, source in inputs.items()
        },
        "outputs": {
            name: {"path": str(output), "sha256": _sha256(output)}
            for name, output in outputs.items()
        },
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def generate_bundle(
    *,
    full_run: Path,
    no_synth_run: Path,
    phase17_run: Path,
    output_dir: Path,
) -> dict[str, Path]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    full_summary = _read_json(full_run / "summary.json")
    no_synth_summary = _read_json(no_synth_run / "summary.json")
    phase17_summary = _read_json(phase17_run / "summary.json")
    if _read_json(full_run / "selection.json") != full_summary["selection"]:
        raise ValueError("full selection artifact differs from summary")
    if _read_json(no_synth_run / "selection.json") != no_synth_summary["selection"]:
        raise ValueError("ablation selection artifact differs from summary")
    full_logs = load_training_logs(full_run)
    no_synth_logs = load_training_logs(no_synth_run)
    seed_rows = build_seed_selection_rows(full_summary)
    final_rows = load_final_event_rows(
        full_run / "external_final_event_predictions.csv"
    )
    online = load_extended_online_rows(full_run / "external_online_predictions.csv")
    full_horizons = load_horizon_rows(full_run / "external_horizon_metrics.csv")
    no_synth_horizons = load_horizon_rows(
        no_synth_run / "external_horizon_metrics.csv"
    )
    phase17_horizons = load_horizon_rows(
        phase17_run / "external_horizon_metrics.csv"
    )
    validate_inputs(
        full_summary=full_summary,
        no_synth_summary=no_synth_summary,
        phase17_summary=phase17_summary,
        final_rows=final_rows,
        online=online,
        full_horizons=full_horizons,
        no_synth_horizons=no_synth_horizons,
        phase17_horizons=phase17_horizons,
    )
    ablation_rows = build_ablation_rows(
        full_summary=full_summary,
        no_synth_summary=no_synth_summary,
        full_logs=full_logs,
        no_synth_logs=no_synth_logs,
    )
    station_rows = build_dynamic_station_rows(online, final_rows)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True)
    outputs: dict[str, Path] = {}
    plotters = (
        (
            "01_training_dynamics",
            lambda stem: plot_training_dynamics(
                logs=full_logs, summary=full_summary, output_stem=stem
            ),
        ),
        (
            "02_seed_selection",
            lambda stem: plot_seed_selection(rows=seed_rows, output_stem=stem),
        ),
        (
            "03_online_convergence",
            lambda stem: plot_convergence_comparison(
                full_rows=full_horizons,
                no_synth_rows=no_synth_horizons,
                phase17_rows=phase17_horizons,
                output_stem=stem,
            ),
        ),
        (
            "04_forward_loss_ablation",
            lambda stem: plot_forward_ablation(
                rows=ablation_rows,
                full_horizons=full_horizons,
                no_synth_horizons=no_synth_horizons,
                phase17_horizons=phase17_horizons,
                output_stem=stem,
            ),
        ),
        (
            "05_event_trajectories",
            lambda stem: plot_event_trajectories(
                online=online, final_rows=final_rows, output_stem=stem
            ),
        ),
        (
            "06_dynamic_station_selection",
            lambda stem: plot_dynamic_station_selection(
                online=online, station_rows=station_rows, output_stem=stem
            ),
        ),
    )
    for name, plotter in plotters:
        png, pdf = plotter(figures_dir / name)
        outputs[f"{name}.png"] = png
        outputs[f"{name}.pdf"] = pdf

    flattened_online = [row for label in EVENT_ORDER for row in online[label]]
    csv_outputs: tuple[tuple[str, Sequence[Mapping[str, Any]]], ...] = (
        ("seed_selection.csv", seed_rows),
        ("forward_loss_ablation.csv", ablation_rows),
        ("external_final_event_predictions.csv", final_rows),
        ("external_horizon_metrics.csv", full_horizons),
        ("no_synth_horizon_metrics.csv", no_synth_horizons),
        ("phase17_horizon_metrics.csv", phase17_horizons),
        ("external_online_predictions.csv", flattened_online),
        ("dynamic_station_summary.csv", station_rows),
    )
    for name, rows in csv_outputs:
        path = output_dir / name
        _write_csv(path, rows)
        outputs[name] = path
    readme = output_dir / "README.md"
    readme.write_text(
        _readme_text(
            summary=full_summary,
            seed_rows=seed_rows,
            final_rows=final_rows,
            full_horizons=full_horizons,
            no_synth_horizons=no_synth_horizons,
            phase17_horizons=phase17_horizons,
            ablation_rows=ablation_rows,
            station_rows=station_rows,
        ),
        encoding="utf-8",
    )
    outputs["README.md"] = readme

    inputs: dict[str, Path] = {
        "full_config": full_run / "config.yaml",
        "full_summary": full_run / "summary.json",
        "full_selection": full_run / "selection.json",
        "full_final_events": full_run / "external_final_event_predictions.csv",
        "full_online_predictions": full_run / "external_online_predictions.csv",
        "full_horizon_metrics": full_run / "external_horizon_metrics.csv",
        "no_synth_config": no_synth_run / "config.yaml",
        "no_synth_summary": no_synth_run / "summary.json",
        "no_synth_selection": no_synth_run / "selection.json",
        "no_synth_horizon_metrics": no_synth_run / "external_horizon_metrics.csv",
        "phase17_summary": phase17_run / "summary.json",
        "phase17_horizon_metrics": phase17_run / "external_horizon_metrics.csv",
    }
    for seed in SEEDS:
        inputs[f"full_seed_{seed}_training_log"] = (
            full_run / f"seed_{seed}" / "training_log.csv"
        )
        inputs[f"no_synth_seed_{seed}_training_log"] = (
            no_synth_run / f"seed_{seed}" / "training_log.csv"
        )
    manifest = output_dir / "publication_manifest.json"
    write_manifest(path=manifest, inputs=inputs, outputs=outputs)
    outputs["publication_manifest.json"] = manifest
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the phase19 causal forward-guided result gallery"
    )
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--no-synth-run", type=Path, required=True)
    parser.add_argument("--phase17-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = generate_bundle(
        full_run=args.full_run.resolve(),
        no_synth_run=args.no_synth_run.resolve(),
        phase17_run=args.phase17_run.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(f"generated {len(artifacts)} phase19 publication artifacts")


if __name__ == "__main__":
    main()
