from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _style import apply_pub_style, style_axes  # noqa: E402


COLORS = {
    "radial_event_ridge": "#009E73",
    "phase9_r_ensemble": "#0072B2",
    "pgd_melgar": "#CC79A7",
    "reference": "#000000",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _by_event(rows: list[dict[str, str]], *, source: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        event = row["event"]
        if event in result:
            raise ValueError(f"duplicate event in {source}: {event}")
        result[event] = row
    return result


def _mechanism_from_rake(value: str) -> str:
    try:
        rake = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(rake):
        return "unknown"
    rake = ((rake + 180.0) % 360.0) - 180.0
    if abs(rake) <= 30.0 or abs(rake) >= 150.0:
        return "strike-slip"
    if 30.0 <= rake <= 150.0:
        return "reverse"
    return "normal"


def build_comparison_rows(
    *,
    ridge_csv: Path,
    phase9_ensemble_csv: Path,
    phase13_ensemble_csv: Path,
    pgd_event_csv: Path,
) -> list[dict[str, Any]]:
    ridge = _by_event(_read_csv(ridge_csv), source="ridge")
    phase9 = _by_event(_read_csv(phase9_ensemble_csv), source="phase9")
    phase13 = _by_event(_read_csv(phase13_ensemble_csv), source="phase13")
    pgd_rows = [row for row in _read_csv(pgd_event_csv) if row["seed"] == "17"]
    pgd = _by_event(pgd_rows, source="pgd")
    event_sets = [set(rows) for rows in (ridge, phase9, phase13, pgd)]
    if any(events != event_sets[0] for events in event_sets[1:]):
        raise ValueError("method inputs do not contain identical event sets")
    if len(event_sets[0]) != 8:
        raise ValueError("publication comparison requires exactly eight events")

    rows: list[dict[str, Any]] = []
    for event in ridge:
        reference_values = {
            float(ridge[event]["mw_reference"]),
            float(phase9[event]["mw_selected"]),
            float(phase13[event]["mw_selected"]),
            float(pgd[event]["mw_selected"]),
        }
        if len(reference_values) != 1:
            raise ValueError(f"reference magnitude differs across methods: {event}")
        reference = reference_values.pop()
        predictions = {
            "radial_event_ridge": float(ridge[event]["mw_pred"]),
            "phase9_r_ensemble": float(phase9[event]["mw_pred_ensemble"]),
            "phase13_event_balanced": float(
                phase13[event]["mw_pred_ensemble"]
            ),
            "pgd_crowell": float(pgd[event]["pgd_crowell_mw_pred_median"]),
            "pgd_ruhl": float(pgd[event]["pgd_ruhl_mw_pred_median"]),
            "pgd_melgar": float(pgd[event]["pgd_melgar_mw_pred_median"]),
        }
        row: dict[str, Any] = {
            "event": event,
            "mechanism": _mechanism_from_rake(pgd[event]["rake"]),
            "station_count": int(pgd[event]["n_stations"]),
            "mw_reference": reference,
            "ridge_station_count_used": int(ridge[event]["station_count_used"]),
        }
        for method, prediction in predictions.items():
            row[f"{method}_mw"] = prediction
            row[f"{method}_error"] = prediction - reference
            row[f"{method}_abs_error"] = abs(prediction - reference)
        rows.append(row)
    return rows


def summarize_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = (
        "radial_event_ridge",
        "phase9_r_ensemble",
        "phase13_event_balanced",
        "pgd_crowell",
        "pgd_ruhl",
        "pgd_melgar",
    )
    summaries: list[dict[str, Any]] = []
    for method in methods:
        errors = np.asarray(
            [float(row[f"{method}_error"]) for row in rows],
            dtype=np.float64,
        )
        summaries.append(
            {
                "method": method,
                "event_count": len(errors),
                "event_mae": float(np.mean(np.abs(errors))),
                "event_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "event_bias": float(np.mean(errors)),
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty publication CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_publication_manifest(
    *,
    path: Path,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
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


def _short_event_name(value: str) -> str:
    parts = value.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else value


def plot_comparison(
    *,
    rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    output_png: Path,
    output_pdf: Path,
) -> None:
    apply_pub_style()
    metric_by_method = {row["method"]: row for row in metrics}
    labels = [_short_event_name(str(row["event"])) for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    methods = (
        ("radial_event_ridge", "Radial event head"),
        ("phase9_r_ensemble", "R-only ensemble"),
        ("pgd_melgar", "PGD-Melgar"),
    )
    offsets = (-0.22, 0.0, 0.22)

    fig, (ax_prediction, ax_error) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.08},
    )
    for axis in (ax_prediction, ax_error):
        style_axes(axis)

    reference = np.asarray(
        [float(row["mw_reference"]) for row in rows], dtype=np.float64
    )
    ax_prediction.plot(
        x,
        reference,
        color=COLORS["reference"],
        marker="D",
        markersize=3.5,
        linewidth=1.0,
        label="USGS reference",
        zorder=5,
    )
    for offset, (method, label) in zip(offsets, methods):
        predictions = np.asarray(
            [float(row[f"{method}_mw"]) for row in rows], dtype=np.float64
        )
        mae = float(metric_by_method[method]["event_mae"])
        ax_prediction.scatter(
            x + offset,
            predictions,
            s=24,
            color=COLORS[method],
            edgecolors="white",
            linewidths=0.4,
            label=f"{label} (MAE {mae:.3f})",
            zorder=4,
        )
    ax_prediction.set_ylabel("Estimated $M_w$")
    ax_prediction.set_ylim(6.45, 8.1)
    ax_prediction.legend(
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=0.9,
        handletextpad=0.4,
    )
    ax_prediction.text(
        -0.08,
        1.03,
        "A",
        transform=ax_prediction.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )

    width = 0.22
    for offset, (method, label) in zip(offsets, methods):
        errors = np.asarray(
            [float(row[f"{method}_abs_error"]) for row in rows],
            dtype=np.float64,
        )
        ax_error.bar(
            x + offset,
            errors,
            width=width,
            color=COLORS[method],
            label=label,
            zorder=3,
        )
    ax_error.axhline(0.15, color="#666666", linestyle="--", linewidth=0.8)
    ax_error.text(
        len(rows) - 0.55,
        0.16,
        "0.15",
        color="#555555",
        fontsize=7,
        ha="right",
        va="bottom",
    )
    ax_error.set_ylabel("Absolute error ($M_w$)")
    ax_error.set_xlabel("Unseen event")
    ax_error.set_xticks(x)
    ax_error.set_xticklabels(labels, rotation=24, ha="right")
    ax_error.set_ylim(0.0, 0.58)
    ax_error.text(
        -0.08,
        1.04,
        "B",
        transform=ax_error.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build phase14 publication tables and comparison figure."
    )
    parser.add_argument("--ridge-csv", type=Path, required=True)
    parser.add_argument("--phase9-ensemble-csv", type=Path, required=True)
    parser.add_argument("--phase13-ensemble-csv", type=Path, required=True)
    parser.add_argument("--pgd-event-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    rows = build_comparison_rows(
        ridge_csv=args.ridge_csv.resolve(),
        phase9_ensemble_csv=args.phase9_ensemble_csv.resolve(),
        phase13_ensemble_csv=args.phase13_ensemble_csv.resolve(),
        pgd_event_csv=args.pgd_event_csv.resolve(),
    )
    metrics = summarize_methods(rows)
    output_dir.mkdir(parents=True)
    _write_csv(output_dir / "event_method_comparison.csv", rows)
    _write_csv(output_dir / "method_metrics.csv", metrics)
    plot_comparison(
        rows=rows,
        metrics=metrics,
        output_png=output_dir / "unseen_event_method_comparison.png",
        output_pdf=output_dir / "unseen_event_method_comparison.pdf",
    )
    inputs = {
        "ridge": args.ridge_csv.resolve(),
        "phase9_ensemble": args.phase9_ensemble_csv.resolve(),
        "phase13_ensemble": args.phase13_ensemble_csv.resolve(),
        "pgd_events": args.pgd_event_csv.resolve(),
    }
    outputs = {
        name: output_dir / name
        for name in (
            "event_method_comparison.csv",
            "method_metrics.csv",
            "unseen_event_method_comparison.png",
            "unseen_event_method_comparison.pdf",
        )
    }
    write_publication_manifest(
        path=output_dir / "publication_manifest.json",
        inputs=inputs,
        outputs=outputs,
    )
    print(f"saved publication outputs: {output_dir}")


if __name__ == "__main__":
    main()
