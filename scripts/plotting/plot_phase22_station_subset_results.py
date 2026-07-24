from __future__ import annotations

import argparse
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
    EVENT_ORDER,
    SEEDS,
    SEED_COLORS,
    _event_label,
    _read_csv,
    _read_json,
    _save_figure,
    _sha256,
    _write_csv,
)


BASELINE_METHOD = "causal_forward_guided_event_neural_v2"
CANDIDATE_METHOD = "causal_forward_guided_event_neural_v3"
TARGET_MAE = 0.15
LOSS_WEIGHTS = {
    "lambda_MSE": 1.0,
    "lambda_synth": 0.5,
    "lambda_mag": 1.0,
    "lambda_shape": 0.1,
}


def _panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.15,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def _validate_summary(summary: Mapping[str, Any], *, method: str) -> None:
    if summary.get("status") != "complete" or summary.get("method") != method:
        raise ValueError(f"summary does not describe a complete {method} run")
    if summary.get("input_components") != ["R"]:
        raise ValueError("publication input is not R-only")
    if summary.get("loss") != LOSS_WEIGHTS:
        raise ValueError("four-term loss weights differ from the frozen contract")
    if bool(summary.get("uses_ensemble")):
        raise ValueError("publication input unexpectedly uses an ensemble")
    if bool(summary.get("uses_future_waveform")) or bool(
        summary.get("uses_final_peak_for_station_selection")
    ):
        raise ValueError("publication input violates the causal contract")
    for flag in (
        "uses_original_four_term_loss",
        "uses_tcn",
        "uses_transformer",
        "uses_shared_event_stf",
    ):
        if not bool(summary.get(flag)):
            raise ValueError(f"publication input does not preserve {flag}")
    selection = summary.get("selection", {})
    if selection.get("selection_metric") != "validation_online_mae" or bool(
        selection.get("ensemble_used")
    ):
        raise ValueError("seed selection contract differs from the frozen rule")


def build_seed_metric_rows(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        key = str(seed)
        baseline_seed = baseline["seed_summaries"][key]
        candidate_seed = candidate["seed_summaries"][key]
        if baseline_seed["split_assignment_sha256"] != candidate_seed[
            "split_assignment_sha256"
        ]:
            raise ValueError(f"split changed for seed {seed}")
        row: dict[str, Any] = {
            "seed": seed,
            "selected": seed == int(candidate["selection"]["selected_seed"]),
            "split_assignment_sha256": candidate_seed["split_assignment_sha256"],
        }
        for split in ("validation", "test"):
            baseline_online = float(
                baseline_seed["split_online_metrics"][split][
                    "event_equal_online_mae"
                ]
            )
            candidate_online = float(
                candidate_seed["split_online_metrics"][split][
                    "event_equal_online_mae"
                ]
            )
            baseline_final = float(
                baseline_seed["split_final_metrics"][split]["event_mae"]
            )
            candidate_final = float(
                candidate_seed["split_final_metrics"][split]["event_mae"]
            )
            for metric, old, new in (
                ("online", baseline_online, candidate_online),
                ("final", baseline_final, candidate_final),
            ):
                row[f"baseline_{split}_{metric}_mae"] = old
                row[f"candidate_{split}_{metric}_mae"] = new
                row[f"improvement_{split}_{metric}_mae"] = old - new
        rows.append(row)
    return rows


def _load_internal_final_rows(path: Path) -> dict[str, dict[str, Any]]:
    final: dict[str, dict[str, Any]] = {}
    for source in _read_csv(path):
        if not math.isclose(float(source["horizon_sec"]), 200.0, abs_tol=1.0e-12):
            continue
        row = {
            "event": source["event"],
            "horizon_sec": float(source["horizon_sec"]),
            "mw_reference": float(source["mw_reference"]),
            "mw_pred": float(source["mw_pred"]),
            "error": float(source["error"]),
            "abs_error": float(source["abs_error"]),
            "neural_residual_mw": float(source["neural_residual_mw"]),
            "active_station_count": int(source["active_station_count"]),
            "used_station_count": int(source["used_station_count"]),
            "used_stations": source["used_stations"],
        }
        if row["event"] in final:
            raise ValueError(f"duplicate final row for {row['event']}")
        if not all(
            math.isfinite(float(value))
            for key, value in row.items()
            if key not in {"event", "used_stations"}
        ):
            raise ValueError("internal prediction contains a non-finite value")
        if not math.isclose(
            abs(float(row["error"])), float(row["abs_error"]), abs_tol=2.0e-7
        ):
            raise ValueError("internal prediction error fields are inconsistent")
        if abs(float(row["neural_residual_mw"])) > 1.0e-12:
            raise ValueError("final neural residual is not zero")
        final[str(row["event"])] = row
    if len(final) != 31:
        raise ValueError("internal final rows do not cover 31 events")
    return final


def build_internal_event_rows(
    baseline_path: Path, candidate_path: Path
) -> list[dict[str, Any]]:
    baseline = _load_internal_final_rows(baseline_path)
    candidate = _load_internal_final_rows(candidate_path)
    if set(baseline) != set(candidate):
        raise ValueError("internal event sets differ")
    rows: list[dict[str, Any]] = []
    for event in sorted(candidate):
        old = baseline[event]
        new = candidate[event]
        for key in (
            "mw_reference",
            "active_station_count",
            "used_station_count",
            "used_stations",
        ):
            if old[key] != new[key]:
                raise ValueError(f"internal final station contract changed for {event}")
        rows.append(
            {
                "event": event,
                "test_station_count": new["active_station_count"],
                "used_station_count": new["used_station_count"],
                "used_stations": new["used_stations"],
                "mw_reference": new["mw_reference"],
                "baseline_mw_pred": old["mw_pred"],
                "baseline_abs_error": old["abs_error"],
                "candidate_mw_pred": new["mw_pred"],
                "candidate_abs_error": new["abs_error"],
                "improvement": old["abs_error"] - new["abs_error"],
                "improved": new["abs_error"] < old["abs_error"],
            }
        )
    return rows


def _load_external_final_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for source in _read_csv(path):
        label = _event_label(source["event"])
        row = {
            "event": label,
            "mw_reference": float(source["mw_reference"]),
            "mw_pred": float(source["mw_pred"]),
            "error": float(source["error"]),
            "abs_error": float(source["abs_error"]),
            "active_station_count": int(source["active_station_count"]),
            "used_station_count": int(source["used_station_count"]),
            "used_stations": source["used_stations"],
            "neural_residual_mw": float(source["neural_residual_mw"]),
        }
        if abs(row["neural_residual_mw"]) > 1.0e-12:
            raise ValueError("external final residual is not zero")
        rows[label] = row
    if set(rows) != set(EVENT_ORDER):
        raise ValueError("external final rows do not cover the frozen eight events")
    return rows


def build_external_event_rows(
    baseline_path: Path, candidate_path: Path
) -> list[dict[str, Any]]:
    baseline = _load_external_final_rows(baseline_path)
    candidate = _load_external_final_rows(candidate_path)
    rows: list[dict[str, Any]] = []
    for event in EVENT_ORDER:
        old = baseline[event]
        new = candidate[event]
        if not math.isclose(old["mw_reference"], new["mw_reference"], abs_tol=0.0):
            raise ValueError(f"external reference magnitude changed for {event}")
        rows.append(
            {
                "event": event,
                "available_station_count": new["active_station_count"],
                "used_station_count": new["used_station_count"],
                "used_stations": new["used_stations"],
                "mw_reference": new["mw_reference"],
                "baseline_mw_pred": old["mw_pred"],
                "baseline_abs_error": old["abs_error"],
                "candidate_mw_pred": new["mw_pred"],
                "candidate_abs_error": new["abs_error"],
                "improvement": old["abs_error"] - new["abs_error"],
                "improved": new["abs_error"] < old["abs_error"],
            }
        )
    return rows


def plot_internal_metric_comparison(
    *, rows: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any], output_stem: Path
) -> tuple[Path, Path]:
    apply_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    panels = (
        (axes[0, 0], "validation", "online", "Validation: all seconds"),
        (axes[0, 1], "test", "online", "Test: all seconds"),
        (axes[1, 0], "validation", "final", "Validation: 200 s"),
        (axes[1, 1], "test", "final", "Test: 200 s"),
    )
    selected_seed = int(candidate["selection"]["selected_seed"])
    gate = candidate["internal_test_gate"]
    for index, (axis, split, metric, title) in enumerate(panels):
        style_axes(axis)
        for row in rows:
            seed = int(row["seed"])
            old = float(row[f"baseline_{split}_{metric}_mae"])
            new = float(row[f"candidate_{split}_{metric}_mae"])
            selected = seed == selected_seed
            axis.plot(
                [0.0, 1.0],
                [old, new],
                color=SEED_COLORS[seed],
                linewidth=1.8 if selected else 1.0,
                alpha=1.0 if selected else 0.72,
                marker="o",
                markersize=5.2 if selected else 4.0,
                zorder=3,
            )
            axis.text(
                1.04,
                new,
                str(seed),
                color="#333333",
                fontsize=7,
                va="center",
            )
        if split == "test":
            threshold = float(
                gate[f"maximum_{'online' if metric == 'online' else 'final'}_mae"]
            )
            axis.axhline(
                threshold,
                color="#666666",
                linestyle="--",
                linewidth=0.9,
                zorder=1,
            )
            axis.text(
                0.02,
                threshold,
                f" gate {threshold:.3f}",
                fontsize=6.5,
                color="#555555",
                va="bottom",
            )
        axis.set_xlim(-0.1, 1.22)
        axis.set_xticks([0.0, 1.0], ["Phase19\nbaseline", "Phase22\nstation subsets"])
        axis.set_ylabel("Event MAE (Mw)")
        axis.set_title(title)
        axis.yaxis.set_major_locator(plt.MaxNLocator(5))
        _panel_label(axis, chr(ord("A") + index))
    legend = [
        Line2D(
            [0],
            [0],
            color=SEED_COLORS[seed],
            marker="o",
            linewidth=1.4,
            label=f"Seed {seed}" + (" (selected)" if seed == selected_seed else ""),
        )
        for seed in SEEDS
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.53, 0.99),
    )
    fig.subplots_adjust(left=0.11, right=0.94, bottom=0.11, top=0.86, wspace=0.34, hspace=0.42)
    return _save_figure(fig, output_stem)


def plot_internal_event_errors(
    *, rows: Sequence[Mapping[str, Any]], output_stem: Path
) -> tuple[Path, Path]:
    apply_pub_style()
    ordered = sorted(rows, key=lambda row: float(row["candidate_abs_error"]), reverse=True)
    y = np.arange(len(ordered), dtype=np.float64)
    baseline = np.asarray([float(row["baseline_abs_error"]) for row in ordered])
    candidate = np.asarray([float(row["candidate_abs_error"]) for row in ordered])
    labels = [
        f"{row['event']} ({int(row['test_station_count'])})" for row in ordered
    ]
    improved = sum(bool(row["improved"]) for row in rows)
    baseline_mae = float(np.mean(baseline))
    candidate_mae = float(np.mean(candidate))

    fig, axis = plt.subplots(figsize=(7.2, 8.0))
    style_axes(axis)
    for index, (old, new) in enumerate(zip(baseline, candidate)):
        axis.plot([old, new], [index, index], color="#B7B7B7", linewidth=1.0, zorder=1)
    axis.scatter(
        baseline,
        y,
        s=22,
        color="#999999",
        marker="o",
        label="Phase19 baseline",
        zorder=2,
    )
    axis.scatter(
        candidate,
        y,
        s=25,
        color="#0072B2",
        marker="D",
        label="Phase22 station subsets",
        zorder=3,
    )
    axis.axvline(
        TARGET_MAE,
        color="#D55E00",
        linestyle="--",
        linewidth=1.0,
        label="0.15 Mw reference",
        zorder=1,
    )
    for index, value in enumerate(candidate):
        if value >= 0.4:
            axis.text(value + 0.012, index, f"{value:.3f}", fontsize=6.5, va="center")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, max(0.95, float(max(baseline.max(), candidate.max())) + 0.08))
    axis.set_xlabel("Absolute magnitude error (Mw)")
    axis.set_ylabel("Internal test event (station count at 200 s)")
    axis.set_title(
        "Selected seed 73 internal test at 200 s\n"
        f"MAE {baseline_mae:.4f} to {candidate_mae:.4f}; "
        f"{improved}/31 events improved"
    )
    axis.legend(loc="lower right")
    axis.xaxis.set_major_locator(plt.MaxNLocator(6))
    fig.subplots_adjust(left=0.27, right=0.96, bottom=0.08, top=0.91)
    return _save_figure(fig, output_stem)


def _readme_text(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    seed_rows: Sequence[Mapping[str, Any]],
    internal_rows: Sequence[Mapping[str, Any]],
    external_rows: Sequence[Mapping[str, Any]],
) -> str:
    gate = candidate["internal_test_gate"]
    selected = int(candidate["selection"]["selected_seed"])
    baseline_selected = baseline["seed_summaries"][str(selected)]
    candidate_selected = candidate["seed_summaries"][str(selected)]
    improved_events = sum(bool(row["improved"]) for row in internal_rows)
    external = candidate["external"]
    lines = [
        "# Phase22 Station-Subset Robustness Result",
        "",
        "> The fixed eight external events remain a development validation set. The internal station-random test was gated before external files were loaded.",
        "",
        "## Headline result",
        "",
        "| Metric | Phase19 baseline | Phase22 station subsets | Improvement |",
        "|---|---:|---:|---:|",
        f"| Internal test all-second MAE | {gate['baseline_online_mae']:.6f} | {gate['candidate_online_mae']:.6f} | {gate['online_improvement']:.6f} |",
        f"| Internal test 200 s MAE | {gate['baseline_final_mae']:.6f} | {gate['candidate_final_mae']:.6f} | {gate['final_improvement']:.6f} |",
        f"| External all-second MAE | {baseline['external']['online_metrics']['event_equal_online_mae']:.6f} | {external['online_metrics']['event_equal_online_mae']:.6f} | {baseline['external']['online_metrics']['event_equal_online_mae'] - external['online_metrics']['event_equal_online_mae']:.6f} |",
        f"| External 200 s MAE | {baseline['external']['final_metrics']['event_mae']:.6f} | {external['final_metrics']['event_mae']:.6f} | {baseline['external']['final_metrics']['event_mae'] - external['final_metrics']['event_mae']:.6f} |",
        "",
        f"Seed {selected} was selected only by internal validation all-second MAE. There is no seed averaging. The internal gate passed before the external eight-event directory was hashed or loaded.",
        "",
        "The model remains R-only and causal. It retains the causal TCN, masked Transformer, shared STF, and the original `1.0 L_MSE + 0.5 L_synth + 1.0 L_mag + 0.1 L_shape` objective. The only scientific change is training-time exposure to one canonical station pool plus three deterministic 25% station subsets per event.",
        "",
        "## 1. Same-split internal metrics",
        "",
        "![Same-split internal metrics](figures/01_internal_metrics.png)",
        "",
        "[Download PDF](figures/01_internal_metrics.pdf)",
        "",
        "| Seed | Validation online | Test online | Test final | Selected |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for row in seed_rows:
        lines.append(
            f"| {int(row['seed'])} | {float(row['candidate_validation_online_mae']):.6f} | "
            f"{float(row['candidate_test_online_mae']):.6f} | "
            f"{float(row['candidate_test_final_mae']):.6f} | "
            f"{'yes' if bool(row['selected']) else 'no'} |"
        )
    lines.extend(
        [
            "",
            "All three Phase22 splits have the same assignment SHA as Phase19. Validation selected seed 73 without consulting test or external metrics.",
            "",
            "Seed 42 happens to have the lowest Phase22 test final MAE (0.193354), but switching to it after viewing test would turn the test split into a model-selection set. The frozen validation rule is therefore retained.",
            "",
            "## 2. Internal event residuals",
            "",
            "![Internal event residual comparison](figures/02_internal_event_errors.png)",
            "",
            "[Download PDF](figures/02_internal_event_errors.pdf)",
            "",
            f"The 200-second absolute error improved for {improved_events}/31 internal test events. The mean improved from {float(baseline_selected['split_final_metrics']['test']['event_mae']):.6f} to {float(candidate_selected['split_final_metrics']['test']['event_mae']):.6f} Mw. The remaining large errors, especially Napa, Miyagi2011B, Iquique, Melinka, and Ibaraki, show that the internal result is materially better but not yet a <=0.15 Mw result.",
            "",
            "## External development check",
            "",
            "| Event | Stations available/used | Reference Mw | Predicted Mw | Absolute error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in external_rows:
        lines.append(
            f"| {row['event']} | {int(row['available_station_count'])}/{int(row['used_station_count'])} | "
            f"{float(row['mw_reference']):.1f} | {float(row['candidate_mw_pred']):.6f} | "
            f"{float(row['candidate_abs_error']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Coverage is 8/8 events and 159 accepted stations. The final external MAE is {float(external['final_metrics']['event_mae']):.6f} Mw, while the all-second external MAE changes from {float(baseline['external']['online_metrics']['event_equal_online_mae']):.6f} to {float(external['online_metrics']['event_equal_online_mae']):.6f} Mw. This check was executed once after the internal gate passed; it is not an unbiased paper test.",
            "",
            "## Data and provenance",
            "",
            "- [Seed metrics](seed_metrics.csv)",
            "- [Internal final-event comparison](internal_test_final_event_comparison.csv)",
            "- [External final-event comparison](external_final_event_comparison.csv)",
            "- [Publication manifest](publication_manifest.json)",
            "- Implementation commit: `8884db2`",
            "- Formal run: `phase22-forward-guided-station-subset-20260724T113401Z-8884db2`",
            "- Selected checkpoint SHA-256: `61969f71eff384288de22af0d826fd2f03d181e3743c4894d4ce8dacc8baed6b`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_manifest(
    *, path: Path, inputs: Mapping[str, Path], outputs: Mapping[str, Path]
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
    *, baseline_run: Path, candidate_run: Path, output_dir: Path
) -> dict[str, Path]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    baseline = _read_json(baseline_run / "summary.json")
    candidate = _read_json(candidate_run / "summary.json")
    _validate_summary(baseline, method=BASELINE_METHOD)
    _validate_summary(candidate, method=CANDIDATE_METHOD)
    if _read_json(baseline_run / "selection.json") != baseline["selection"]:
        raise ValueError("baseline selection artifact differs from summary")
    if _read_json(candidate_run / "selection.json") != candidate["selection"]:
        raise ValueError("candidate selection artifact differs from summary")
    gate = candidate.get("internal_test_gate")
    if not isinstance(gate, Mapping) or not bool(gate.get("passed")):
        raise ValueError("candidate did not pass the frozen internal gate")
    if not bool(candidate.get("external_loaded_after_seed_selection")):
        raise ValueError("candidate external evaluation boundary is not recorded")

    selected = int(candidate["selection"]["selected_seed"])
    seed_rows = build_seed_metric_rows(baseline, candidate)
    internal_rows = build_internal_event_rows(
        baseline_run / f"seed_{selected}" / "test_online_predictions.csv",
        candidate_run / f"seed_{selected}" / "test_online_predictions.csv",
    )
    external_rows = build_external_event_rows(
        baseline_run / "external_final_event_predictions.csv",
        candidate_run / "external_final_event_predictions.csv",
    )

    candidate_internal_mae = float(
        np.mean([float(row["candidate_abs_error"]) for row in internal_rows])
    )
    candidate_external_mae = float(
        np.mean([float(row["candidate_abs_error"]) for row in external_rows])
    )
    if not math.isclose(
        candidate_internal_mae, float(gate["candidate_final_mae"]), abs_tol=2.0e-7
    ):
        raise ValueError("internal event table does not reproduce the gate")
    if not math.isclose(
        candidate_external_mae,
        float(candidate["external"]["final_metrics"]["event_mae"]),
        abs_tol=2.0e-7,
    ):
        raise ValueError("external event table does not reproduce the summary")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True)
    outputs: dict[str, Path] = {}
    for name, plotter in (
        (
            "01_internal_metrics",
            lambda stem: plot_internal_metric_comparison(
                rows=seed_rows, candidate=candidate, output_stem=stem
            ),
        ),
        (
            "02_internal_event_errors",
            lambda stem: plot_internal_event_errors(
                rows=internal_rows, output_stem=stem
            ),
        ),
    ):
        png, pdf = plotter(figures_dir / name)
        outputs[f"{name}.png"] = png
        outputs[f"{name}.pdf"] = pdf

    for name, rows in (
        ("seed_metrics.csv", seed_rows),
        ("internal_test_final_event_comparison.csv", internal_rows),
        ("external_final_event_comparison.csv", external_rows),
    ):
        path = output_dir / name
        _write_csv(path, rows)
        outputs[name] = path
    readme = output_dir / "README.md"
    readme.write_text(
        _readme_text(
            baseline=baseline,
            candidate=candidate,
            seed_rows=seed_rows,
            internal_rows=internal_rows,
            external_rows=external_rows,
        ),
        encoding="utf-8",
    )
    outputs["README.md"] = readme

    inputs = {
        "baseline_summary": baseline_run / "summary.json",
        "baseline_selection": baseline_run / "selection.json",
        "baseline_selected_test_predictions": baseline_run
        / f"seed_{selected}"
        / "test_online_predictions.csv",
        "baseline_external_final_predictions": baseline_run
        / "external_final_event_predictions.csv",
        "candidate_config": candidate_run / "config.yaml",
        "candidate_summary": candidate_run / "summary.json",
        "candidate_selection": candidate_run / "selection.json",
        "candidate_selected_test_predictions": candidate_run
        / f"seed_{selected}"
        / "test_online_predictions.csv",
        "candidate_external_final_predictions": candidate_run
        / "external_final_event_predictions.csv",
    }
    for seed in SEEDS:
        inputs[f"candidate_seed_{seed}_summary"] = (
            candidate_run / f"seed_{seed}" / "summary.json"
        )
        inputs[f"candidate_seed_{seed}_split"] = (
            candidate_run / f"seed_{seed}" / "split.json"
        )
    manifest = output_dir / "publication_manifest.json"
    _write_manifest(path=manifest, inputs=inputs, outputs=outputs)
    outputs["publication_manifest.json"] = manifest
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the phase22 station-subset robustness result gallery"
    )
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = generate_bundle(
        baseline_run=args.baseline_run.resolve(),
        candidate_run=args.candidate_run.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(f"generated {len(artifacts)} phase22 publication artifacts")


if __name__ == "__main__":
    main()
