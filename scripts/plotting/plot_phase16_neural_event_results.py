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
METHOD_LABELS = {
    "radial_pinn_event_neural_v2": "PINN neural event v2",
    "radial_event_ridge": "Radial event ridge",
    "pgd_melgar": "PGD-Melgar",
    "phase9_r_ensemble": "Phase9 R-only ensemble",
    "phase13_event_balanced": "Phase13 event-balanced",
    "pgd_crowell": "PGD-Crowell",
    "pgd_ruhl": "PGD-Ruhl",
}
METHOD_COLORS = {
    "radial_pinn_event_neural_v2": "#0072B2",
    "radial_event_ridge": "#009E73",
    "pgd_melgar": "#CC79A7",
    "phase9_r_ensemble": "#56B4E9",
    "phase13_event_balanced": "#999999",
    "pgd_crowell": "#D55E00",
    "pgd_ruhl": "#E69F00",
}


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


def _short_event_name(value: str) -> str:
    parts = value.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else value


def load_training_logs(run_root: Path) -> dict[int, dict[str, np.ndarray]]:
    columns = (
        "loss",
        "fit_loss",
        "linear_penalty",
        "nonlinear_penalty",
        "train_event_mae",
        "validation_event_mae",
        "learning_rate",
    )
    logs: dict[int, dict[str, np.ndarray]] = {}
    epoch_count: int | None = None
    for seed in SEEDS:
        path = run_root / f"seed_{seed}" / "training_log.csv"
        rows = _read_csv(path)
        epochs = np.asarray([int(row["epoch"]) for row in rows], dtype=np.int64)
        if not np.array_equal(epochs, np.arange(1, len(rows) + 1)):
            raise ValueError(f"training epochs are not consecutive for seed {seed}")
        if epoch_count is None:
            epoch_count = len(rows)
        elif len(rows) != epoch_count:
            raise ValueError("training logs have different epoch counts")
        values = {
            column: np.asarray(
                [float(row[column]) for row in rows],
                dtype=np.float64,
            )
            for column in columns
        }
        if any(not np.isfinite(array).all() for array in values.values()):
            raise ValueError(f"training log contains non-finite values: seed {seed}")
        logs[seed] = {"epoch": epochs, **values}
    return logs


def load_event_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in _read_csv(path):
        event = source["event"]
        if event in seen:
            raise ValueError(f"duplicate phase16 event: {event}")
        seen.add(event)
        row = {
            "event": event,
            "mw_pred": float(source["mw_pred"]),
            "mw_reference": float(source["mw_reference"]),
            "error": float(source["error"]),
            "abs_error": float(source["abs_error"]),
            "station_count_available": int(source["station_count_available"]),
            "station_count_used": int(source["station_count_used"]),
            "head_prediction_std": float(source["head_prediction_std"]),
            "nonlinear_delta_mw_mean": float(source["nonlinear_delta_mw_mean"]),
        }
        numeric = [value for key, value in row.items() if key != "event"]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError(f"phase16 event contains non-finite values: {event}")
        if row["station_count_used"] > row["station_count_available"]:
            raise ValueError(f"used station count exceeds available count: {event}")
        if not math.isclose(
            row["mw_pred"] - row["mw_reference"],
            row["error"],
            abs_tol=1.0e-10,
        ):
            raise ValueError(f"phase16 event error is inconsistent: {event}")
        rows.append(row)
    if len(rows) != 8:
        raise ValueError("phase16 publication requires exactly eight events")
    return rows


def build_method_metrics(
    *,
    summary: Mapping[str, Any],
    comparison_metrics_csv: Path,
) -> list[dict[str, Any]]:
    external = summary["external_ensemble"]
    rows: list[dict[str, Any]] = [
        {
            "method": "radial_pinn_event_neural_v2",
            "event_count": int(external["event_count"]),
            "event_mae": float(external["event_mae"]),
            "event_rmse": float(external["event_rmse"]),
            "event_bias": float(external["event_bias"]),
        }
    ]
    seen = {rows[0]["method"]}
    for source in _read_csv(comparison_metrics_csv):
        method = source["method"]
        if method in seen:
            raise ValueError(f"duplicate comparison method: {method}")
        seen.add(method)
        rows.append(
            {
                "method": method,
                "event_count": int(source["event_count"]),
                "event_mae": float(source["event_mae"]),
                "event_rmse": float(source["event_rmse"]),
                "event_bias": float(source["event_bias"]),
            }
        )
    expected = set(METHOD_LABELS)
    if set(seen) != expected:
        raise ValueError(
            f"comparison methods differ: missing={sorted(expected - seen)}, "
            f"extra={sorted(seen - expected)}"
        )
    if any(row["event_count"] != 8 for row in rows):
        raise ValueError("all comparison methods must cover eight events")
    return sorted(rows, key=lambda row: float(row["event_mae"]))


def _panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.11,
        1.04,
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


def plot_training_dynamics(
    *,
    logs: Mapping[int, Mapping[str, np.ndarray]],
    summary: Mapping[str, Any],
    output_stem: Path,
) -> tuple[Path, Path]:
    apply_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    axes_flat = list(axes.flat)
    for axis in axes_flat:
        style_axes(axis)
    warmups = {
        int(summary["head_summaries"][str(seed)]["linear_warmup_epochs"])
        for seed in SEEDS
    }
    if len(warmups) != 1:
        raise ValueError("head warmup epochs differ")
    warmup = warmups.pop()

    for seed in SEEDS:
        data = logs[seed]
        epoch = data["epoch"]
        color = SEED_COLORS[seed]
        axes[0, 0].plot(epoch, data["loss"], color=color, label=f"Seed {seed}")
        axes[0, 1].plot(epoch, data["fit_loss"], color=color)
        axes[1, 0].plot(
            epoch,
            data["train_event_mae"],
            color=color,
            alpha=0.72,
            linewidth=0.9,
        )
        axes[1, 0].plot(
            epoch,
            data["validation_event_mae"],
            color=color,
            linestyle="--",
            linewidth=1.1,
        )
        best_epoch = int(summary["head_summaries"][str(seed)]["best_epoch"])
        best_index = best_epoch - 1
        axes[1, 0].scatter(
            [best_epoch],
            [data["validation_event_mae"][best_index]],
            s=22,
            color=color,
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )
    first = logs[SEEDS[0]]
    axes[1, 1].plot(
        first["epoch"],
        first["learning_rate"],
        color="#333333",
    )
    for axis in axes_flat:
        axis.axvline(warmup, color="#777777", linestyle=":", linewidth=0.9)
        axis.set_xlim(1, int(first["epoch"][-1]))

    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 0].set_title("Total objective")
    axes[0, 1].set_title("Magnitude fit loss")
    axes[1, 0].set_title("Event-level magnitude error")
    axes[1, 1].set_title("Learning-rate schedule")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_ylabel("MAE ($M_w$)")
    axes[1, 1].set_ylabel("Learning rate")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[0, 0].legend(loc="upper right", ncol=1)
    axes[1, 0].legend(
        handles=[
            Line2D([0], [0], color="#555555", linewidth=1.0, label="Train"),
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle="--",
                linewidth=1.0,
                label="Validation",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#555555",
                markeredgecolor="white",
                markersize=5,
                label="Selected checkpoint",
            ),
        ],
        loc="upper right",
    )
    axes[0, 1].text(
        warmup + 35,
        axes[0, 1].get_ylim()[1] / 1.8,
        "PINN residual enabled",
        color="#555555",
        fontsize=7,
        va="top",
    )
    for axis, label in zip(axes_flat, "ABCD"):
        _panel_label(axis, label)
    fig.suptitle(
        "Phase16 neural event-head training dynamics",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.10,
        top=0.90,
        wspace=0.25,
        hspace=0.28,
    )
    return _save_figure(fig, output_stem)


def plot_external_events(
    *,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_stem: Path,
) -> tuple[Path, Path]:
    apply_pub_style()
    labels = [_short_event_name(str(row["event"])) for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    reference = np.asarray([float(row["mw_reference"]) for row in rows])
    prediction = np.asarray([float(row["mw_pred"]) for row in rows])
    spread = np.asarray([float(row["head_prediction_std"]) for row in rows])
    errors = np.asarray([float(row["abs_error"]) for row in rows])
    colors = np.where(errors <= 0.15, "#009E73", "#D55E00")

    fig, (ax_prediction, ax_error) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.08},
    )
    for axis in (ax_prediction, ax_error):
        style_axes(axis)
    ax_prediction.plot(
        x,
        reference,
        color="#000000",
        marker="D",
        markersize=3.5,
        label="USGS reference",
        zorder=4,
    )
    ax_prediction.errorbar(
        x,
        prediction,
        yerr=spread,
        color="#0072B2",
        marker="o",
        markersize=4.5,
        linewidth=1.1,
        capsize=2.5,
        label="Three-head ensemble ± head SD",
        zorder=5,
    )
    metrics = summary["external_ensemble"]
    ax_prediction.text(
        0.01,
        0.04,
        f"8/8 events   MAE={float(metrics['event_mae']):.3f}   "
        f"RMSE={float(metrics['event_rmse']):.3f}   "
        f"bias={float(metrics['event_bias']):+.3f}",
        transform=ax_prediction.transAxes,
        fontsize=7.5,
        ha="left",
        va="bottom",
    )
    ax_prediction.set_ylabel("Magnitude ($M_w$)")
    ax_prediction.set_ylim(6.45, 8.05)
    ax_prediction.legend(loc="upper right")

    bars = ax_error.bar(x, errors, color=colors, width=0.64, zorder=3)
    ax_error.axhline(0.15, color="#666666", linestyle="--", linewidth=0.9)
    ax_error.text(
        -0.55,
        0.158,
        "0.15 target",
        color="#555555",
        fontsize=7,
        ha="left",
        va="bottom",
    )
    for bar, value in zip(bars, errors):
        ax_error.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.009,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax_error.set_ylabel("Absolute error ($M_w$)")
    ax_error.set_xlabel("Development-validation event (available/used stations)")
    ax_error.set_ylim(0.0, max(0.28, float(errors.max()) + 0.04))
    ax_error.set_xticks(x)
    ax_error.set_xticklabels(
        [
            f"{label}\n{row['station_count_available']}/{row['station_count_used']}"
            for label, row in zip(labels, rows)
        ],
        rotation=22,
        ha="right",
    )
    _panel_label(ax_prediction, "A")
    _panel_label(ax_error, "B")
    fig.suptitle(
        "Phase16 external event-magnitude validation",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    return _save_figure(fig, output_stem)


def plot_method_comparison(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> tuple[Path, Path]:
    apply_pub_style()
    labels = [METHOD_LABELS[str(row["method"])] for row in rows]
    y = np.arange(len(rows), dtype=np.float64)
    mae = np.asarray([float(row["event_mae"]) for row in rows])
    rmse = np.asarray([float(row["event_rmse"]) for row in rows])
    bias = np.asarray([float(row["event_bias"]) for row in rows])
    colors = [METHOD_COLORS[str(row["method"])] for row in rows]

    fig, (ax_error, ax_bias) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.1),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.34},
    )
    for axis in (ax_error, ax_bias):
        style_axes(axis)
        axis.grid(True, axis="x", linestyle=":", linewidth=0.5, color="#CCCCCC")
        axis.grid(False, axis="y")
    height = 0.34
    ax_error.barh(y - height / 2, mae, height=height, color=colors, label="MAE")
    ax_error.barh(
        y + height / 2,
        rmse,
        height=height,
        color=colors,
        alpha=0.42,
        label="RMSE",
    )
    for index, value in enumerate(mae):
        ax_error.text(value + 0.007, index - height / 2, f"{value:.3f}", va="center", fontsize=6.5)
    ax_error.set_yticks(y)
    ax_error.set_yticklabels(labels)
    ax_error.invert_yaxis()
    ax_error.set_xlabel("Event error ($M_w$; lower is better)")
    ax_error.legend(loc="lower right")
    ax_error.set_xlim(0.0, max(rmse) + 0.08)

    ax_bias.barh(y, bias, color=colors, height=0.54)
    ax_bias.axvline(0.0, color="#333333", linewidth=0.8)
    for index, value in enumerate(bias):
        ax_bias.text(
            value + (0.008 if value >= 0 else -0.008),
            index,
            f"{value:+.3f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=6.5,
        )
    ax_bias.set_yticks(y)
    ax_bias.set_yticklabels([])
    ax_bias.invert_yaxis()
    ax_bias.set_xlabel("Event bias ($M_w$)")
    limit = max(abs(bias)) + 0.06
    ax_bias.set_xlim(-limit, limit)
    _panel_label(ax_error, "A")
    _panel_label(ax_bias, "B")
    fig.suptitle(
        "Fixed eight-event method comparison",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    return _save_figure(fig, output_stem)


def plot_station_and_contribution(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> tuple[Path, Path]:
    apply_pub_style()
    labels = [_short_event_name(str(row["event"])) for row in rows]
    available = np.asarray([int(row["station_count_available"]) for row in rows])
    used = np.asarray([int(row["station_count_used"]) for row in rows])
    errors = np.asarray([float(row["abs_error"]) for row in rows])
    head_std = 1000.0 * np.asarray([float(row["head_prediction_std"]) for row in rows])
    nonlinear = 1000.0 * np.abs(
        np.asarray([float(row["nonlinear_delta_mw_mean"]) for row in rows])
    )
    x = np.arange(len(rows), dtype=np.float64)

    fig, (ax_station, ax_contribution) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.2),
        gridspec_kw={"width_ratios": [1.0, 1.45], "wspace": 0.32},
    )
    for axis in (ax_station, ax_contribution):
        style_axes(axis)
    scatter = ax_station.scatter(
        available,
        errors,
        s=22.0 + 10.0 * used,
        c=np.where(errors <= 0.15, "#009E73", "#D55E00"),
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
    )
    del scatter
    for index, (count, error, label) in enumerate(zip(available, errors, labels)):
        right_edge = count >= 40
        if right_edge:
            offset = (-6, 18 if error < 0.03 else -10)
        else:
            offset = (4, 5 if index % 2 == 0 else -10)
        ax_station.annotate(
            label,
            (count, error),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.2,
            ha="right" if right_edge else "left",
        )
    ax_station.axhline(0.15, color="#666666", linestyle="--", linewidth=0.9)
    ax_station.set_xscale("log")
    ax_station.set_xticks([3, 5, 10, 20, 50])
    ax_station.set_xticklabels(["3", "5", "10", "20", "50"])
    ax_station.minorticks_off()
    ax_station.set_title("Stations and event error")
    ax_station.set_xlabel("Available stations (log scale)")
    ax_station.set_ylabel("Absolute error ($M_w$)")
    ax_station.set_ylim(0.0, max(0.28, float(errors.max()) + 0.04))
    ax_station.text(
        0.03,
        0.96,
        "Marker size = stations used (3 or 5)",
        transform=ax_station.transAxes,
        fontsize=6.7,
        va="top",
    )

    width = 0.36
    ax_contribution.bar(
        x - width / 2,
        head_std,
        width=width,
        color="#56B4E9",
        label="Head-to-head SD",
    )
    ax_contribution.bar(
        x + width / 2,
        nonlinear,
        width=width,
        color="#CC79A7",
        label="|PINN nonlinear residual|",
    )
    ax_contribution.set_xticks(x)
    ax_contribution.set_xticklabels(labels, rotation=25, ha="right")
    ax_contribution.set_xlabel("Development-validation event")
    ax_contribution.set_ylabel("Scale (10$^{-3}$ $M_w$)")
    ax_contribution.set_title("Head spread vs. neural residual")
    ax_contribution.legend(loc="upper left")
    _panel_label(ax_station, "A")
    _panel_label(ax_contribution, "B")
    fig.suptitle(
        "Station coverage and learned PINN contribution",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    return _save_figure(fig, output_stem)


def _readme_text(
    *,
    summary: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
) -> str:
    external = summary["external_ensemble"]
    lines = [
        "# Phase16 R-only PINN Event Neural v2",
        "",
        "> Fixed eight-event development validation. These events participated in earlier model/feature selection and are not an unbiased final test set.",
        "",
        "## Headline result",
        "",
        "| Coverage | Event MAE | Event RMSE | Bias | Events with abs. error <= 0.15 |",
        "|---:|---:|---:|---:|---:|",
        f"| 8/8 | {float(external['event_mae']):.6f} | {float(external['event_rmse']):.6f} | {float(external['event_bias']):+.6f} | {sum(float(row['abs_error']) <= 0.15 for row in event_rows)}/8 |",
        "",
        "## 1. Training dynamics",
        "",
        "![Training dynamics](figures/01_training_dynamics.png)",
        "",
        "[Download PDF](figures/01_training_dynamics.pdf)",
        "",
        "The dotted vertical line marks the 1000-epoch amplitude-only warm-up. The nonlinear PINN residual is enabled after this point; selected checkpoints occur near epochs 1003-1009.",
        "",
        "## 2. External event predictions",
        "",
        "![External event predictions](figures/02_external_event_performance.png)",
        "",
        "[Download PDF](figures/02_external_event_performance.pdf)",
        "",
        "| Event | Stations available/used | Reference Mw | Predicted Mw | Absolute error |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in event_rows:
        lines.append(
            f"| {_short_event_name(str(row['event']))} | "
            f"{int(row['station_count_available'])}/{int(row['station_count_used'])} | "
            f"{float(row['mw_reference']):.1f} | {float(row['mw_pred']):.3f} | "
            f"{float(row['abs_error']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 3. Method comparison",
            "",
            "![Method comparison](figures/03_method_comparison.png)",
            "",
            "[Download PDF](figures/03_method_comparison.pdf)",
            "",
            "| Method | Event MAE | Event RMSE | Bias |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in method_rows:
        lines.append(
            f"| {METHOD_LABELS[str(row['method'])]} | "
            f"{float(row['event_mae']):.3f} | {float(row['event_rmse']):.3f} | "
            f"{float(row['event_bias']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## 4. Station coverage and neural contribution",
            "",
            "![Station coverage and neural contribution](figures/04_station_and_neural_contribution.png)",
            "",
            "[Download PDF](figures/04_station_and_neural_contribution.pdf)",
            "",
            "The learned nonlinear PINN residual is only about 0.001 Mw on average. The achieved accuracy is driven mainly by the top-five radial amplitude/distance trunk, so this bundle must not be used to claim that PINN deep features provide the dominant gain.",
            "",
            "## Data and provenance",
            "",
            "- [Event predictions](event_predictions.csv)",
            "- [Method metrics](method_metrics.csv)",
            "- [Publication manifest](publication_manifest.json)",
            "- Formal implementation commit: `fd40706`",
            "- Outcome documentation commit: `dee07e2`",
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
    phase16_run: Path,
    comparison_metrics_csv: Path,
    output_dir: Path,
) -> dict[str, Path]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    summary_path = phase16_run / "summary.json"
    event_path = phase16_run / "ensemble_external_event_predictions.csv"
    summary = _read_json(summary_path)
    if summary.get("method") != "radial_pinn_event_neural_v2":
        raise ValueError("phase16 summary has the wrong method")
    if not bool(summary.get("deep_learning")) or bool(summary.get("uses_ridge_prediction")):
        raise ValueError("phase16 summary has an invalid method boundary")
    logs = load_training_logs(phase16_run)
    events = load_event_rows(event_path)
    metrics = build_method_metrics(
        summary=summary,
        comparison_metrics_csv=comparison_metrics_csv,
    )
    errors = np.asarray([float(row["error"]) for row in events])
    frozen = summary["external_ensemble"]
    if not (
        math.isclose(float(np.mean(np.abs(errors))), float(frozen["event_mae"]), abs_tol=1.0e-12)
        and math.isclose(float(np.sqrt(np.mean(np.square(errors)))), float(frozen["event_rmse"]), abs_tol=1.0e-12)
        and math.isclose(float(np.mean(errors)), float(frozen["event_bias"]), abs_tol=1.0e-12)
    ):
        raise ValueError("phase16 event metrics do not reproduce the summary")

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
            "02_external_event_performance",
            lambda stem: plot_external_events(
                rows=events,
                summary=summary,
                output_stem=stem,
            ),
        ),
        (
            "03_method_comparison",
            lambda stem: plot_method_comparison(rows=metrics, output_stem=stem),
        ),
        (
            "04_station_and_neural_contribution",
            lambda stem: plot_station_and_contribution(rows=events, output_stem=stem),
        ),
    )
    for name, plotter in plotters:
        png, pdf = plotter(figures_dir / name)
        outputs[f"{name}.png"] = png
        outputs[f"{name}.pdf"] = pdf

    event_output = output_dir / "event_predictions.csv"
    metrics_output = output_dir / "method_metrics.csv"
    readme = output_dir / "README.md"
    _write_csv(event_output, events)
    _write_csv(metrics_output, metrics)
    readme.write_text(
        _readme_text(summary=summary, event_rows=events, method_rows=metrics),
        encoding="utf-8",
    )
    outputs.update(
        {
            "event_predictions.csv": event_output,
            "method_metrics.csv": metrics_output,
            "README.md": readme,
        }
    )
    inputs = {
        "phase16_summary": summary_path,
        "phase16_events": event_path,
        "comparison_metrics": comparison_metrics_csv,
        **{
            f"seed_{seed}_training_log": phase16_run / f"seed_{seed}" / "training_log.csv"
            for seed in SEEDS
        },
    }
    manifest = output_dir / "publication_manifest.json"
    write_manifest(path=manifest, inputs=inputs, outputs=outputs)
    outputs["publication_manifest.json"] = manifest
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the phase16 neural-event GitHub figure bundle."
    )
    parser.add_argument("--phase16-run", type=Path, required=True)
    parser.add_argument("--comparison-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = generate_bundle(
        phase16_run=args.phase16_run.resolve(),
        comparison_metrics_csv=args.comparison_metrics.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(f"saved {len(outputs)} phase16 publication artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
