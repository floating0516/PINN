from __future__ import annotations

import argparse
import csv
import hashlib
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


SEEDS = (17, 42, 73)
SEED_COLORS = {
    17: "#0072B2",
    42: "#D55E00",
    73: "#009E73",
}
EVENT_MATCHERS = (
    ("Iquique", "Iquique 2014"),
    ("Chiniak", "Kodiak 2018"),
    ("Kangding", "Luding 2022"),
    ("Mandalay", "Mandalay 2025"),
    ("Nepal", "Nepal 2015"),
    ("Greece", "Samos 2020"),
    ("Sand Point", "Sand Point 2025"),
    ("Tibetan Plateau", "Xizang 2025"),
)
EVENT_ORDER = tuple(label for _, label in EVENT_MATCHERS)
TARGET_MAE = 0.15


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _event_label(event: str) -> str:
    matches = [label for needle, label in EVENT_MATCHERS if needle in event]
    if len(matches) != 1:
        raise ValueError(f"event does not have one publication label: {event}")
    return matches[0]


def _sort_event_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    labeled = {_event_label(str(row["event"])): dict(row) for row in rows}
    if set(labeled) != set(EVENT_ORDER):
        raise ValueError("publication rows do not contain the fixed eight events")
    return [labeled[label] for label in EVENT_ORDER]


def _panel_label(
    axis: Any,
    label: str,
    *,
    x: float = -0.18,
    y: float = 1.10,
) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def _save_figure(fig: Any, output_stem: Path) -> tuple[Path, Path]:
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def load_training_logs(
    run_root: Path,
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    columns = (
        "loss",
        "fit_loss",
        "penalty",
        "validation_mae",
        "learning_rate",
    )
    logs: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    expected_lengths: dict[str, int] = {}
    for seed in SEEDS:
        rows = _read_csv(run_root / f"seed_{seed}" / "training_log.csv")
        if {row["phase"] for row in rows} != {"anchor", "prefix"}:
            raise ValueError(f"training phases differ for seed {seed}")
        seed_log: dict[str, dict[str, np.ndarray]] = {}
        for phase in ("anchor", "prefix"):
            phase_rows = [row for row in rows if row["phase"] == phase]
            epochs = np.asarray(
                [int(row["epoch"]) for row in phase_rows],
                dtype=np.int64,
            )
            if not np.array_equal(epochs, np.arange(1, len(phase_rows) + 1)):
                raise ValueError(f"{phase} epochs are not consecutive for seed {seed}")
            if phase in expected_lengths and len(phase_rows) != expected_lengths[phase]:
                raise ValueError(f"{phase} logs have different lengths")
            expected_lengths.setdefault(phase, len(phase_rows))
            values = {
                column: np.asarray(
                    [float(row[column]) for row in phase_rows],
                    dtype=np.float64,
                )
                for column in columns
            }
            if any(not value.shape[0] or not np.isfinite(value).all() for value in values.values()):
                raise ValueError(f"training log contains invalid values: seed {seed}")
            seed_log[phase] = {"epoch": epochs, **values}
        logs[seed] = seed_log
    return logs


def _parse_prediction_row(source: Mapping[str, str]) -> dict[str, Any]:
    row = {
        "event": source["event"],
        "horizon_sec": float(source["horizon_sec"]),
        "mw_pred": float(source["mw_pred"]),
        "mw_reference": float(source["mw_reference"]),
        "error": float(source["error"]),
        "abs_error": float(source["abs_error"]),
        "active_station_count": int(source["active_station_count"]),
        "used_station_count": int(source["used_station_count"]),
        "used_stations": source["used_stations"],
    }
    numeric = [value for key, value in row.items() if key not in {"event", "used_stations"}]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError(f"prediction contains non-finite values: {row['event']}")
    if row["active_station_count"] < row["used_station_count"] or row["used_station_count"] < 1:
        raise ValueError(f"prediction has invalid station counts: {row['event']}")
    stations = str(row["used_stations"]).split("|")
    if len(stations) != row["used_station_count"] or len(stations) != len(set(stations)):
        raise ValueError(f"prediction has invalid used stations: {row['event']}")
    if not math.isclose(
        row["mw_pred"] - row["mw_reference"],
        row["error"],
        abs_tol=1.0e-10,
    ) or not math.isclose(abs(row["error"]), row["abs_error"], abs_tol=1.0e-10):
        raise ValueError(f"prediction error is inconsistent: {row['event']}")
    return row


def load_final_event_rows(path: Path) -> list[dict[str, Any]]:
    rows = [_parse_prediction_row(source) for source in _read_csv(path)]
    if len(rows) != 8 or len({str(row["event"]) for row in rows}) != 8:
        raise ValueError("final publication table requires eight unique events")
    horizons = {float(row["horizon_sec"]) for row in rows}
    if len(horizons) != 1:
        raise ValueError("final events do not share one horizon")
    return _sort_event_rows(rows)


def load_online_event_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {label: [] for label in EVENT_ORDER}
    for source in _read_csv(path):
        row = _parse_prediction_row(source)
        grouped[_event_label(str(row["event"]))].append(row)
    if any(not rows for rows in grouped.values()):
        raise ValueError("online publication data do not cover eight events")
    for label, rows in grouped.items():
        rows.sort(key=lambda row: float(row["horizon_sec"]))
        horizons = np.asarray([float(row["horizon_sec"]) for row in rows])
        if not np.allclose(np.diff(horizons), 1.0, atol=0.0):
            raise ValueError(f"online horizons are not consecutive: {label}")
        references = {float(row["mw_reference"]) for row in rows}
        if len(references) != 1:
            raise ValueError(f"online reference magnitude changes: {label}")
        predictions = np.asarray([float(row["mw_pred"]) for row in rows])
        if np.any(np.diff(predictions) == 0.0):
            raise ValueError(f"online prediction does not update every second: {label}")
    return grouped


def load_horizon_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in _read_csv(path):
        row = {
            "horizon_sec": float(source["horizon_sec"]),
            "event_count": int(source["event_count"]),
            "event_mae": float(source["event_mae"]),
            "event_rmse": float(source["event_rmse"]),
            "event_bias": float(source["event_bias"]),
        }
        if row["event_count"] < 1 or row["event_count"] > 8:
            raise ValueError("horizon metrics have invalid event coverage")
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise ValueError("horizon metrics contain non-finite values")
        rows.append(row)
    horizons = np.asarray([float(row["horizon_sec"]) for row in rows])
    if not np.allclose(np.diff(horizons), 1.0, atol=0.0):
        raise ValueError("horizon metrics are not consecutive")
    return rows


def build_seed_selection_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    selection = summary["selection"]
    if bool(selection["ensemble_used"]) or summary.get("uses_ensemble") is not False:
        raise ValueError("phase17 publication requires one seed and no ensemble")
    if selection["selection_metric"] != "validation_online_mae":
        raise ValueError("phase17 selection metric is not validation online MAE")
    selected_seed = int(selection["selected_seed"])
    candidates = selection["candidates"]
    rows = [
        {
            "seed": seed,
            "validation_online_mae": float(candidates[str(seed)]["validation_online_mae"]),
            "validation_final_mae": float(candidates[str(seed)]["validation_final_mae"]),
            "selected": seed == selected_seed,
        }
        for seed in SEEDS
    ]
    expected = min(
        rows,
        key=lambda row: (
            float(row["validation_online_mae"]),
            float(row["validation_final_mae"]),
            int(row["seed"]),
        ),
    )
    if int(expected["seed"]) != selected_seed:
        raise ValueError("selected seed does not minimize the declared metrics")
    return rows


def build_dynamic_station_rows(
    online: Mapping[str, Sequence[Mapping[str, Any]]],
    final_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    final_by_label = {_event_label(str(row["event"])): row for row in final_rows}
    rows: list[dict[str, Any]] = []
    for label in EVENT_ORDER:
        snapshots = list(online[label])
        station_sets = [str(row["used_stations"]) for row in snapshots]
        changes = sum(
            current != previous
            for previous, current in zip(station_sets, station_sets[1:])
        )
        final = final_by_label[label]
        rows.append(
            {
                "event": label,
                "first_horizon_sec": float(snapshots[0]["horizon_sec"]),
                "prediction_count": len(snapshots),
                "maximum_active_station_count": max(
                    int(row["active_station_count"]) for row in snapshots
                ),
                "final_active_station_count": int(final["active_station_count"]),
                "final_used_station_count": int(final["used_station_count"]),
                "distinct_used_station_sets": len(set(station_sets)),
                "station_set_change_count": changes,
                "final_abs_error": float(final["abs_error"]),
            }
        )
    return rows


def _full_coverage_horizon(rows: Sequence[Mapping[str, Any]]) -> float:
    for index, row in enumerate(rows):
        if int(row["event_count"]) == 8 and all(
            int(later["event_count"]) == 8 for later in rows[index:]
        ):
            return float(row["horizon_sec"])
    raise ValueError("external metrics never reach persistent 8/8 coverage")


def _stable_target_horizon(rows: Sequence[Mapping[str, Any]]) -> float | None:
    for index, row in enumerate(rows):
        if int(row["event_count"]) == 8 and all(
            int(later["event_count"]) == 8
            and float(later["event_mae"]) <= TARGET_MAE
            for later in rows[index:]
        ):
            return float(row["horizon_sec"])
    return None


def validate_external_artifacts(
    *,
    summary: Mapping[str, Any],
    final_rows: Sequence[Mapping[str, Any]],
    online: Mapping[str, Sequence[Mapping[str, Any]]],
    horizon_rows: Sequence[Mapping[str, Any]],
) -> None:
    if summary.get("method") != "causal_radial_event_neural_v1":
        raise ValueError("phase17 summary has the wrong method")
    if not bool(summary.get("deep_learning")) or summary.get("input_components") != ["R"]:
        raise ValueError("phase17 summary has the wrong model boundary")
    if bool(summary.get("uses_future_waveform")) or bool(
        summary.get("uses_final_peak_for_station_selection")
    ):
        raise ValueError("phase17 summary is not causal")
    if bool(summary.get("uses_ensemble")):
        raise ValueError("phase17 summary unexpectedly uses an ensemble")

    errors = np.asarray([float(row["error"]) for row in final_rows])
    metrics = summary["external"]["final_metrics"]
    reproduced = (
        float(np.mean(np.abs(errors))),
        float(np.sqrt(np.mean(np.square(errors)))),
        float(np.mean(errors)),
    )
    frozen = (
        float(metrics["event_mae"]),
        float(metrics["event_rmse"]),
        float(metrics["event_bias"]),
    )
    if any(not math.isclose(a, b, abs_tol=2.0e-7) for a, b in zip(reproduced, frozen)):
        raise ValueError("final event metrics do not reproduce the phase17 summary")

    flattened = [row for label in EVENT_ORDER for row in online[label]]
    if len(flattened) != int(summary["external"]["online_metrics"]["snapshot_count"]):
        raise ValueError("online snapshot count differs from the phase17 summary")
    final_by_label = {_event_label(str(row["event"])): row for row in final_rows}
    for label in EVENT_ORDER:
        observed = online[label][-1]
        final = final_by_label[label]
        for key in ("horizon_sec", "mw_pred", "mw_reference", "error", "abs_error"):
            if not math.isclose(float(observed[key]), float(final[key]), abs_tol=1.0e-12):
                raise ValueError(f"final online row differs for {label}")
        for key in ("active_station_count", "used_station_count", "used_stations"):
            if observed[key] != final[key]:
                raise ValueError(f"final online station state differs for {label}")

    rows_by_horizon: dict[float, list[Mapping[str, Any]]] = {}
    for row in flattened:
        rows_by_horizon.setdefault(float(row["horizon_sec"]), []).append(row)
    if len(rows_by_horizon) != len(horizon_rows):
        raise ValueError("online horizons differ from the metric table")
    for metric in horizon_rows:
        horizon = float(metric["horizon_sec"])
        snapshots = rows_by_horizon.get(horizon, [])
        errors = np.asarray([float(row["error"]) for row in snapshots])
        expected = (
            len(snapshots),
            float(np.mean(np.abs(errors))),
            float(np.sqrt(np.mean(np.square(errors)))),
            float(np.mean(errors)),
        )
        observed = (
            int(metric["event_count"]),
            float(metric["event_mae"]),
            float(metric["event_rmse"]),
            float(metric["event_bias"]),
        )
        if expected[0] != observed[0] or any(
            not math.isclose(a, b, abs_tol=1.0e-10)
            for a, b in zip(expected[1:], observed[1:])
        ):
            raise ValueError(f"horizon metrics do not reproduce at {horizon:g} s")


def plot_training_dynamics(
    *,
    logs: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    summary: Mapping[str, Any],
    output_stem: Path,
) -> tuple[Path, Path]:
    apply_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    panels = list(axes.flat)
    for axis in panels:
        style_axes(axis)

    selected_seed = int(summary["selection"]["selected_seed"])
    for seed in SEEDS:
        color = SEED_COLORS[seed]
        width = 1.45 if seed == selected_seed else 0.9
        alpha = 1.0 if seed == selected_seed else 0.78
        anchor = logs[seed]["anchor"]
        prefix = logs[seed]["prefix"]
        axes[0, 0].plot(
            anchor["epoch"],
            anchor["fit_loss"],
            color=color,
            linewidth=width,
            alpha=alpha,
            label=f"Seed {seed}" + (" (selected)" if seed == selected_seed else ""),
        )
        axes[0, 1].plot(
            anchor["epoch"],
            anchor["validation_mae"],
            color=color,
            linewidth=width,
            alpha=alpha,
        )
        axes[1, 0].plot(
            prefix["epoch"],
            prefix["fit_loss"],
            color=color,
            linewidth=width,
            alpha=alpha,
        )
        axes[1, 1].plot(
            prefix["epoch"],
            prefix["validation_mae"],
            color=color,
            linewidth=width,
            alpha=alpha,
        )
        seed_summary = summary["seed_summaries"][str(seed)]
        for axis, phase, key in (
            (axes[0, 1], anchor, "best_anchor_epoch"),
            (axes[1, 1], prefix, "best_prefix_epoch"),
        ):
            best_epoch = int(seed_summary[key])
            axis.scatter(
                [best_epoch],
                [phase["validation_mae"][best_epoch - 1]],
                color=color,
                edgecolors="white",
                linewidths=0.5,
                s=24,
                zorder=5,
            )

    axes[0, 0].set_yscale("log")
    axes[1, 0].set_yscale("log")
    axes[0, 0].set_title("Final-horizon anchor fit")
    axes[0, 1].set_title("Anchor validation final MAE")
    axes[1, 0].set_title("Causal prefix residual fit")
    axes[1, 1].set_title("Prefix validation online MAE")
    axes[0, 0].set_ylabel("Fit loss")
    axes[0, 1].set_ylabel("MAE (Mw)")
    axes[1, 0].set_ylabel("Fit loss")
    axes[1, 1].set_ylabel("MAE (Mw)")
    axes[0, 0].set_xlabel("Anchor epoch")
    axes[0, 1].set_xlabel("Anchor epoch")
    axes[1, 0].set_xlabel("Prefix epoch")
    axes[1, 1].set_xlabel("Prefix epoch")
    axes[0, 0].legend(loc="upper right")
    axes[1, 1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#555555",
                markeredgecolor="white",
                markersize=5,
                label="Best internal checkpoint",
            )
        ],
        loc="upper right",
    )
    for axis, label in zip(panels, "ABCD"):
        _panel_label(axis, label)
    fig.suptitle(
        "Phase17 two-stage causal event-network training",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.10,
        top=0.90,
        wspace=0.26,
        hspace=0.32,
    )
    return _save_figure(fig, output_stem)


def plot_seed_selection(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> tuple[Path, Path]:
    apply_pub_style()
    fig, axis = plt.subplots(figsize=(6.4, 3.7))
    style_axes(axis)
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.34
    online = np.asarray([float(row["validation_online_mae"]) for row in rows])
    final = np.asarray([float(row["validation_final_mae"]) for row in rows])
    selected_index = next(index for index, row in enumerate(rows) if bool(row["selected"]))
    axis.axvspan(
        selected_index - 0.46,
        selected_index + 0.46,
        color="#009E73",
        alpha=0.09,
        zorder=0,
    )
    online_bars = axis.bar(
        x - width / 2,
        online,
        width=width,
        color="#0072B2",
        label="Validation online MAE (selection metric)",
        zorder=3,
    )
    final_bars = axis.bar(
        x + width / 2,
        final,
        width=width,
        color="#999999",
        label="Validation final MAE (tie-break)",
        zorder=3,
    )
    for bars, values in ((online_bars, online), (final_bars, final)):
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    selected_seed = int(rows[selected_index]["seed"])
    axis.text(
        selected_index,
        max(online[selected_index], final[selected_index]) + 0.036,
        f"Selected seed {selected_seed}\n(no ensemble)",
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#006E55",
    )
    axis.set_xticks(x)
    axis.set_xticklabels([f"Seed {int(row['seed'])}" for row in rows])
    axis.set_ylabel("Internal event MAE (Mw; lower is better)")
    axis.set_ylim(0.0, max(float(online.max()), float(final.max())) + 0.10)
    axis.legend(loc="upper right")
    axis.set_title("Seed selected from internal data before external evaluation")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.15, top=0.89)
    return _save_figure(fig, output_stem)


def plot_external_convergence(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> tuple[Path, Path]:
    apply_pub_style()
    horizon = np.asarray([float(row["horizon_sec"]) for row in rows])
    mae = np.asarray([float(row["event_mae"]) for row in rows])
    rmse = np.asarray([float(row["event_rmse"]) for row in rows])
    coverage = np.asarray([int(row["event_count"]) for row in rows])
    full_horizon = _full_coverage_horizon(rows)
    stable_horizon = _stable_target_horizon(rows)

    fig, (ax_error, ax_coverage) = plt.subplots(
        2,
        1,
        figsize=(7.2, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.55, 0.65], "hspace": 0.08},
    )
    for axis in (ax_error, ax_coverage):
        style_axes(axis)
        axis.axvspan(horizon[0], full_horizon, color="#999999", alpha=0.10)
        axis.axvline(full_horizon, color="#777777", linestyle=":", linewidth=0.9)
        if stable_horizon is not None:
            axis.axvline(stable_horizon, color="#009E73", linestyle="--", linewidth=0.9)
    ax_error.plot(horizon, mae, color="#0072B2", label="Event MAE")
    ax_error.plot(horizon, rmse, color="#D55E00", label="Event RMSE", alpha=0.85)
    ax_error.axhline(TARGET_MAE, color="#555555", linestyle="--", linewidth=0.9)
    ax_error.set_yscale("log")
    ticks = [0.1, 0.15, 0.3, 0.6, 1.2]
    ax_error.yaxis.set_major_locator(FixedLocator(ticks))
    ax_error.yaxis.set_major_formatter(FixedFormatter([str(value) for value in ticks]))
    ax_error.minorticks_off()
    ax_error.set_ylim(0.085, max(1.6, float(rmse.max()) * 1.08))
    ax_error.set_ylabel("Error (Mw; log scale)")
    ax_error.legend(loc="upper right")
    ax_error.text(
        horizon[0] + 2,
        0.158,
        "0.15 target",
        color="#555555",
        fontsize=7,
        va="bottom",
    )
    ax_error.scatter([horizon[-1]], [mae[-1]], color="#0072B2", s=25, zorder=5)
    ax_error.annotate(
        f"{mae[-1]:.3f} at {horizon[-1]:.0f} s",
        (horizon[-1], mae[-1]),
        xytext=(-8, 14),
        textcoords="offset points",
        ha="right",
        fontsize=7,
    )
    if stable_horizon is not None:
        ax_error.text(
            stable_horizon + 2,
            0.82,
            f"MAE stays <=0.15\nfrom {stable_horizon:.0f} s",
            color="#006E55",
            fontsize=7,
            va="top",
        )

    ax_coverage.step(horizon, coverage, where="post", color="#333333")
    ax_coverage.fill_between(horizon, coverage, step="post", color="#56B4E9", alpha=0.35)
    ax_coverage.set_yticks([1, 2, 4, 6, 8])
    ax_coverage.set_ylim(0.5, 8.5)
    ax_coverage.set_ylabel("Events")
    ax_coverage.set_xlabel("Global horizon after origin time (s)")
    ax_coverage.text(
        full_horizon + 2,
        1.05,
        f"8/8 continuously from {full_horizon:.0f} s",
        fontsize=7,
        va="bottom",
    )
    ax_coverage.set_xlim(0.0, float(horizon[-1]))
    _panel_label(ax_error, "A")
    _panel_label(ax_coverage, "B")
    fig.suptitle(
        "Selected seed 17: causal external error convergence",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.91)
    return _save_figure(fig, output_stem)


def plot_event_trajectories(
    *,
    online: Mapping[str, Sequence[Mapping[str, Any]]],
    final_rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> tuple[Path, Path]:
    apply_pub_style()
    final_by_label = {_event_label(str(row["event"])): row for row in final_rows}
    final_horizon = max(float(row["horizon_sec"]) for row in final_rows)
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 7.8), sharex=True)
    panels = list(axes.flat)
    for index, (axis, label) in enumerate(zip(panels, EVENT_ORDER)):
        style_axes(axis)
        rows = online[label]
        horizon = np.asarray([float(row["horizon_sec"]) for row in rows])
        prediction = np.asarray([float(row["mw_pred"]) for row in rows])
        reference = float(rows[0]["mw_reference"])
        final_error = float(final_by_label[label]["abs_error"])
        axis.axhspan(
            reference - TARGET_MAE,
            reference + TARGET_MAE,
            color="#009E73",
            alpha=0.12,
            zorder=0,
        )
        axis.axhline(reference, color="#222222", linestyle="--", linewidth=0.9)
        axis.plot(horizon, prediction, color="#0072B2", linewidth=1.05)
        axis.scatter(
            [horizon[-1]],
            [prediction[-1]],
            color="#009E73" if final_error <= TARGET_MAE else "#D55E00",
            edgecolors="white",
            linewidths=0.5,
            s=24,
            zorder=5,
        )
        low = min(float(prediction.min()), reference - TARGET_MAE)
        high = max(float(prediction.max()), reference + TARGET_MAE)
        margin = max(0.08, 0.06 * (high - low))
        axis.set_ylim(low - margin, high + margin)
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.set_title(f"{label}   final |error|={final_error:.3f}", fontsize=8.5)
        axis.text(
            0.02,
            0.05,
            f"starts {horizon[0]:.0f} s",
            transform=axis.transAxes,
            fontsize=6.5,
            va="bottom",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 1.0,
            },
        )
        axis.set_xlim(0.0, final_horizon)
        if index % 2 == 0:
            axis.set_ylabel("Magnitude (Mw)")
        if index >= 6:
            axis.set_xlabel("Global horizon (s)")
        _panel_label(axis, chr(ord("A") + index))
    fig.legend(
        handles=[
            Line2D([0], [0], color="#0072B2", label="Causal prediction"),
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
        ncol=3,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.suptitle(
        "Eight-event causal magnitude trajectories",
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


def plot_dynamic_station_selection(
    *,
    online: Mapping[str, Sequence[Mapping[str, Any]]],
    station_rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> tuple[Path, Path]:
    apply_pub_style()
    final_horizon = max(
        float(row["horizon_sec"])
        for label in EVENT_ORDER
        for row in online[label]
    )
    first_horizon = min(
        float(row["horizon_sec"])
        for label in EVENT_ORDER
        for row in online[label]
    )
    horizons = np.arange(first_horizon, final_horizon + 1.0)
    matrix = np.full((len(EVENT_ORDER), len(horizons)), np.nan, dtype=np.float64)
    changes_by_event: list[list[float]] = []
    for event_index, label in enumerate(EVENT_ORDER):
        rows = online[label]
        previous_set: str | None = None
        changes: list[float] = []
        for row in rows:
            horizon = float(row["horizon_sec"])
            column = int(round(horizon - first_horizon))
            matrix[event_index, column] = int(row["used_station_count"])
            station_set = str(row["used_stations"])
            if previous_set is not None and station_set != previous_set:
                changes.append(horizon)
            previous_set = station_set
        changes_by_event.append(changes)

    fig, (ax_heat, ax_error) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.08},
    )
    cmap = plt.get_cmap("cividis", 5).with_extremes(bad="#F0F0F0")
    image = ax_heat.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="none",
        origin="upper",
        extent=(first_horizon - 0.5, final_horizon + 0.5, 7.5, -0.5),
        cmap=cmap,
        vmin=0.5,
        vmax=5.5,
    )
    for event_index, changes in enumerate(changes_by_event):
        ax_heat.scatter(
            changes,
            [event_index] * len(changes),
            marker="|",
            color="#000000",
            s=18,
            linewidths=0.55,
            zorder=4,
        )
    ax_heat.set_yticks(np.arange(len(EVENT_ORDER)))
    ax_heat.set_yticklabels(EVENT_ORDER)
    ax_heat.set_xlabel("Global horizon (s)")
    ax_heat.set_title("Stations used at each second")
    ax_heat.text(
        0.01,
        -0.19,
        "Black ticks mark a change in top-k station membership",
        transform=ax_heat.transAxes,
        fontsize=6.7,
        va="top",
    )
    colorbar = fig.colorbar(image, ax=ax_heat, pad=0.02, fraction=0.045)
    colorbar.set_ticks([1, 2, 3, 4, 5])
    colorbar.ax.set_title("Used", fontsize=7, pad=4)

    errors = np.asarray([float(row["final_abs_error"]) for row in station_rows])
    y = np.arange(len(station_rows), dtype=np.float64)
    bars = ax_error.barh(
        y,
        errors,
        height=0.58,
        color=np.where(errors <= TARGET_MAE, "#009E73", "#D55E00"),
    )
    ax_error.axvline(
        TARGET_MAE,
        color="#555555",
        linestyle="--",
        linewidth=0.9,
        zorder=1,
    )
    for bar, row in zip(bars, station_rows):
        value = float(row["final_abs_error"])
        ax_error.text(
            0.225,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}; {int(row['distinct_used_station_sets'])} sets; "
            f"{int(row['final_active_station_count'])}/{int(row['final_used_station_count'])} final",
            va="center",
            fontsize=6.2,
            zorder=3,
        )
    ax_error.set_title("Final error and station turnover")
    ax_error.set_xlabel("Absolute error (Mw)")
    ax_error.tick_params(axis="y", labelleft=False)
    ax_error.set_xlim(0.0, max(0.41, float(errors.max()) + 0.19))
    style_axes(ax_error)
    ax_error.grid(True, axis="x", linestyle=":", linewidth=0.5, color="#CCCCCC")
    ax_error.grid(False, axis="y")
    _panel_label(ax_heat, "A")
    _panel_label(ax_error, "B", x=-0.08)
    fig.suptitle(
        "Dynamic top-five station selection and final event error",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.19, top=0.87)
    return _save_figure(fig, output_stem)


def _readme_text(
    *,
    summary: Mapping[str, Any],
    seed_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    horizon_rows: Sequence[Mapping[str, Any]],
    station_rows: Sequence[Mapping[str, Any]],
) -> str:
    final_metrics = summary["external"]["final_metrics"]
    selected_seed = int(summary["selection"]["selected_seed"])
    final_horizon = int(float(final_rows[0]["horizon_sec"]))
    full_horizon = _full_coverage_horizon(horizon_rows)
    stable_horizon = _stable_target_horizon(horizon_rows)
    headline_passes = sum(float(row["abs_error"]) <= TARGET_MAE for row in final_rows)
    horizon_by_second = {
        int(float(row["horizon_sec"])): row for row in horizon_rows
    }
    key_horizons = [
        second
        for second in (30, 60, 90, 120, 150, 180, 200)
        if second in horizon_by_second
    ]
    if final_horizon not in key_horizons:
        key_horizons.append(final_horizon)
    lines = [
        "# Phase17 Causal R-only Single-seed Event Neural Model",
        "",
        "> Fixed eight-event development validation. These events influenced earlier feature development and are not an unbiased final paper test set.",
        "",
        "## Headline result",
        "",
        f"| Selected seed | Ensemble | Coverage at {final_horizon} s | Event MAE | Event RMSE | Bias | Events with abs. error <= 0.15 |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
        f"| {selected_seed} | no | 8/8 | {float(final_metrics['event_mae']):.6f} | "
        f"{float(final_metrics['event_rmse']):.6f} | "
        f"{float(final_metrics['event_bias']):+.6f} | {headline_passes}/8 |",
        "",
        "The input is radial R only. At second `t`, the model uses only waveform samples available through `t`, after a conservative 6-second processing delay. Running peaks and the top-five station set are recomputed every second; no full-window peak or final station ranking is used.",
        "",
        "## 1. Two-stage training dynamics",
        "",
        "![Two-stage training dynamics](figures/01_training_dynamics.png)",
        "",
        "[Download PDF](figures/01_training_dynamics.pdf)",
        "",
        f"The {final_horizon}-second amplitude anchor is trained first. A two-layer GELU prefix residual is then trained for evolving estimates and is gated to exactly zero at {final_horizon} seconds.",
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
            "All three seeds were trained. Seed 17 was selected once using internal validation online MAE before external waveforms were loaded. External predictions use only seed 17; there is no averaging and no per-event seed choice.",
            "",
            "## 3. External error convergence",
            "",
            "![External error convergence](figures/03_external_convergence.png)",
            "",
            "[Download PDF](figures/03_external_convergence.pdf)",
            "",
            f"All eight events have predictions continuously from {full_horizon:.0f} seconds onward. "
            + (
                f"Event MAE first reaches and then remains <=0.15 Mw at {stable_horizon:.0f} seconds."
                if stable_horizon is not None
                else "Event MAE does not remain <=0.15 Mw over the available horizon."
            ),
            "",
            "| Global horizon | Event coverage | Event MAE |",
            "|---:|---:|---:|",
        ]
    )
    for second in key_horizons:
        row = horizon_by_second[second]
        lines.append(
            f"| {second} s | {int(row['event_count'])}/8 | {float(row['event_mae']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 4. Eight event trajectories",
            "",
            "![Eight event trajectories](figures/04_event_trajectories.png)",
            "",
            "[Download PDF](figures/04_event_trajectories.pdf)",
            "",
            f"| Event | Reference Mw | Predicted Mw at {final_horizon} s | Absolute error | Active/used stations |",
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
            "## 5. Dynamic station selection",
            "",
            "![Dynamic station selection](figures/05_dynamic_station_selection.png)",
            "",
            "[Download PDF](figures/05_dynamic_station_selection.pdf)",
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
            "- [Final event predictions](external_final_event_predictions.csv)",
            "- [Per-second horizon metrics](external_horizon_metrics.csv)",
            "- [Per-second event predictions](external_online_predictions.csv)",
            "- [Dynamic station summary](dynamic_station_summary.csv)",
            "- [Publication manifest](publication_manifest.json)",
            "- Formal implementation commit: `a337bd4`",
            "- Outcome documentation commit: `5cd0194`",
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
    phase17_run: Path,
    output_dir: Path,
) -> dict[str, Path]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    summary_path = phase17_run / "summary.json"
    selection_path = phase17_run / "selection.json"
    final_path = phase17_run / "external_final_event_predictions.csv"
    online_path = phase17_run / "external_online_predictions.csv"
    horizon_path = phase17_run / "external_horizon_metrics.csv"
    summary = _read_json(summary_path)
    selection = _read_json(selection_path)
    if selection != summary.get("selection"):
        raise ValueError("selection.json differs from the frozen summary")
    logs = load_training_logs(phase17_run)
    seed_rows = build_seed_selection_rows(summary)
    final_rows = load_final_event_rows(final_path)
    online = load_online_event_rows(online_path)
    horizon_rows = load_horizon_rows(horizon_path)
    validate_external_artifacts(
        summary=summary,
        final_rows=final_rows,
        online=online,
        horizon_rows=horizon_rows,
    )
    station_rows = build_dynamic_station_rows(online, final_rows)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True)
    outputs: dict[str, Path] = {}
    plotters = (
        (
            "01_training_dynamics",
            lambda stem: plot_training_dynamics(
                logs=logs,
                summary=summary,
                output_stem=stem,
            ),
        ),
        (
            "02_seed_selection",
            lambda stem: plot_seed_selection(rows=seed_rows, output_stem=stem),
        ),
        (
            "03_external_convergence",
            lambda stem: plot_external_convergence(rows=horizon_rows, output_stem=stem),
        ),
        (
            "04_event_trajectories",
            lambda stem: plot_event_trajectories(
                online=online,
                final_rows=final_rows,
                output_stem=stem,
            ),
        ),
        (
            "05_dynamic_station_selection",
            lambda stem: plot_dynamic_station_selection(
                online=online,
                station_rows=station_rows,
                output_stem=stem,
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
        ("external_final_event_predictions.csv", final_rows),
        ("external_horizon_metrics.csv", horizon_rows),
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
            summary=summary,
            seed_rows=seed_rows,
            final_rows=final_rows,
            horizon_rows=horizon_rows,
            station_rows=station_rows,
        ),
        encoding="utf-8",
    )
    outputs["README.md"] = readme

    inputs = {
        "phase17_config": phase17_run / "config.yaml",
        "phase17_summary": summary_path,
        "phase17_selection": selection_path,
        "phase17_final_events": final_path,
        "phase17_online_predictions": online_path,
        "phase17_horizon_metrics": horizon_path,
        **{
            f"seed_{seed}_training_log": phase17_run / f"seed_{seed}" / "training_log.csv"
            for seed in SEEDS
        },
    }
    manifest = output_dir / "publication_manifest.json"
    write_manifest(path=manifest, inputs=inputs, outputs=outputs)
    outputs["publication_manifest.json"] = manifest
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the phase17 causal single-seed GitHub figure bundle."
    )
    parser.add_argument("--phase17-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = generate_bundle(
        phase17_run=args.phase17_run.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(f"saved {len(outputs)} phase17 publication artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
