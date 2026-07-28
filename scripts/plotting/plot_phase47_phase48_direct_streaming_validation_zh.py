#!/usr/bin/env python3
"""Publish the Phase47/48 direct Phase39 streaming validation report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
PROJECT_HOME = REPO_ROOT.parent.parent
DEFAULT_PHASE47_RUN = (
    PROJECT_HOME
    / "runs"
    / "phase47-direct-phase39-streaming-20260728T141807Z-31a3271"
)
DEFAULT_PHASE48_RUN = (
    PROJECT_HOME
    / "runs"
    / "phase48-joint-phase39-streaming-20260728T143620Z-3ff2449"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "docs"
    / "reports"
    / "phase47-phase48-direct-streaming-validation-zh"
)
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
SEEDS = (17, 42, 73)
COLORS = {
    "phase39": "#20262E",
    "phase47": "#D55E00",
    "phase48": "#0072B2",
    "green": "#009E73",
    "purple": "#8E5AA9",
    "yellow": "#E69F00",
    "red": "#B42318",
    "gray": "#67717E",
    "grid": "#D8DEE6",
    "ink": "#20262E",
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_plotting() -> None:
    if FONT_PATH.is_file():
        font_manager.fontManager.addfont(str(FONT_PATH))
        family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
        mpl.rcParams["font.family"] = family
    mpl.rcParams.update(
        {
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titleweight": "normal",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
        }
    )


def _style_axis(axis: Any) -> None:
    axis.grid(color=COLORS["grid"], linewidth=0.7)
    axis.set_axisbelow(True)


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _float(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def _load_run(run_root: Path, phase: str) -> dict[str, Any]:
    campaign = _read_json(run_root / "campaign_summary.json")
    seeds: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        seed_root = run_root / f"seed_{seed}"
        seeds[seed] = {
            "baseline": _read_json(seed_root / "baseline_validation_metrics.json"),
            "epochs": _read_csv(seed_root / "epoch_metrics.csv"),
            "summary": _read_json(seed_root / "summary.json"),
        }
    return {"phase": phase, "campaign": campaign, "seeds": seeds}


def _gate_targets(baseline: Mapping[str, Any]) -> dict[str, float]:
    return {
        "endpoint_event_mae": float(baseline["endpoint_event_mae"]) + 0.005,
        "endpoint_station_mae": float(baseline["endpoint_station_mae"]) + 0.005,
        "streaming_event_mae_mean": (
            0.95 * float(baseline["streaming_event_mae_mean"])
        ),
        "late_event_abs_step_p95_mw": (
            0.8 * float(baseline["late_event_abs_step_p95_mw"])
        ),
        "late_station_abs_step_p95_mw": float(
            baseline["late_station_abs_step_p95_mw"]
        ),
        "late_confirmed_cumulative_log10_l1_p95": (
            0.8
            * float(baseline["late_confirmed_cumulative_log10_l1_p95"])
        ),
    }


def _closest_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(rows, key=lambda row: float(row["selection_score"]))


def _combined_epoch_rows(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for source in run["seeds"][seed]["epochs"]:
            rows.append({"phase": run["phase"], "seed": seed, **source})
    return rows


def _baseline_rows(*runs: Mapping[str, Any]) -> list[dict[str, Any]]:
    keys = (
        "endpoint_event_mae",
        "endpoint_station_mae",
        "streaming_event_mae_mean",
        "late_event_abs_step_p95_mw",
        "late_station_abs_step_p95_mw",
        "late_confirmed_cumulative_log10_l1_p95",
    )
    rows: list[dict[str, Any]] = []
    for run in runs:
        for seed in SEEDS:
            baseline = run["seeds"][seed]["baseline"]
            rows.append(
                {
                    "phase": run["phase"],
                    "seed": seed,
                    **{key: baseline[key] for key in keys},
                }
            )
    return rows


def _closest_rows(*runs: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "epoch",
        "endpoint_event_mae",
        "endpoint_station_mae",
        "endpoint_preserved",
        "streaming_event_mae_mean",
        "late_event_abs_step_p95_mw",
        "late_station_abs_step_p95_mw",
        "late_confirmed_cumulative_log10_l1_p95",
        "selection_score",
    )
    rows: list[dict[str, Any]] = []
    for run in runs:
        for seed in SEEDS:
            closest = _closest_row(run["seeds"][seed]["epochs"])
            rows.append(
                {
                    "phase": run["phase"],
                    "seed": seed,
                    **{field: closest[field] for field in fields},
                }
            )
    return rows


def _plot_tradeoff(
    phase47: Mapping[str, Any],
    phase48: Mapping[str, Any],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), sharey=True)
    for axis, seed in zip(axes, SEEDS, strict=True):
        baseline = phase48["seeds"][seed]["baseline"]
        targets = _gate_targets(baseline)
        for run, color, label, marker in (
            (phase47, COLORS["phase47"], "Phase47 checkpoint 微调", "o"),
            (phase48, COLORS["phase48"], "Phase48 联合从头训练", "."),
        ):
            rows = run["seeds"][seed]["epochs"]
            axis.scatter(
                [_float(row, "endpoint_event_mae") for row in rows],
                [_float(row, "streaming_event_mae_mean") for row in rows],
                color=color,
                s=22 if run is phase47 else 10,
                alpha=0.75 if run is phase47 else 0.42,
                marker=marker,
                label=label,
            )
        axis.scatter(
            [float(baseline["endpoint_event_mae"])],
            [float(baseline["streaming_event_mae_mean"])],
            color=COLORS["phase39"],
            marker="*",
            s=105,
            label="Phase39 基线",
            zorder=4,
        )
        axis.axvline(
            targets["endpoint_event_mae"],
            color=COLORS["red"],
            linestyle="--",
            linewidth=1.1,
        )
        axis.axhline(
            targets["streaming_event_mae_mean"],
            color=COLORS["green"],
            linestyle="--",
            linewidth=1.1,
        )
        axis.set_title(f"seed {seed}")
        axis.set_xlabel("200 秒 Event MAE")
        _style_axis(axis)
    axes[0].set_ylabel("流式锚点平均 Event MAE")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("同一 Phase39 架构：流式学习与终点精度的 validation 权衡", y=0.99)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    _save_figure(figure, output_dir / "01_endpoint_streaming_tradeoff")


def _plot_seed73_gate_ratios(
    phase48: Mapping[str, Any],
    output_dir: Path,
) -> None:
    baseline = phase48["seeds"][73]["baseline"]
    targets = _gate_targets(baseline)
    rows = [
        row
        for row in phase48["seeds"][73]["epochs"]
        if int(row["epoch"]) >= 60
    ]
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    closest = _closest_row(rows)
    closest_epoch = int(closest["epoch"])
    panels = (
        (
            "终点 gate",
            (
                ("endpoint_event_mae", "Event MAE", COLORS["phase48"]),
                ("endpoint_station_mae", "Station MAE", COLORS["phase47"]),
            ),
        ),
        (
            "流式与稳定性 gate",
            (
                ("streaming_event_mae_mean", "流式 Event MAE", COLORS["green"]),
                ("late_event_abs_step_p95_mw", "晚期 Event 跳变", COLORS["purple"]),
                ("late_station_abs_step_p95_mw", "晚期 Station 跳变", COLORS["yellow"]),
                (
                    "late_confirmed_cumulative_log10_l1_p95",
                    "确认历史改写",
                    COLORS["red"],
                ),
            ),
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.5))
    for axis, (title, series) in zip(axes, panels, strict=True):
        for key, label, color in series:
            values = np.asarray([_float(row, key) for row in rows]) / targets[key]
            axis.plot(epochs, values, color=color, linewidth=1.5, label=label)
        axis.axhline(1.0, color=COLORS["phase39"], linestyle="--", linewidth=1.2)
        axis.axvline(
            closest_epoch,
            color=COLORS["gray"],
            linestyle=":",
            linewidth=1.2,
            label=f"最接近 epoch {closest_epoch}",
        )
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel("实际值 / 冻结上限")
        axis.set_ylim(0.35, 1.65)
        axis.legend(frameon=False, fontsize=8)
        _style_axis(axis)
    figure.suptitle("Phase48 seed73：最接近通过，但没有同时满足全部 gate", y=1.03)
    figure.tight_layout()
    _save_figure(figure, output_dir / "02_seed73_gate_ratios")


def _plot_phase48_training(
    phase48: Mapping[str, Any],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12.8, 10.2), sharex="col")
    for row_index, seed in enumerate(SEEDS):
        rows = phase48["seeds"][seed]["epochs"]
        baseline = phase48["seeds"][seed]["baseline"]
        targets = _gate_targets(baseline)
        epochs = np.asarray([int(row["epoch"]) for row in rows])
        axes[row_index, 0].plot(
            epochs,
            [_float(row, "train_online_full_science_loss") for row in rows],
            color=COLORS["phase48"],
            linewidth=1.25,
            label="完整 200 秒科学损失",
        )
        axes[row_index, 0].plot(
            epochs,
            [_float(row, "train_online_prefix_science_loss") for row in rows],
            color=COLORS["phase47"],
            linewidth=1.1,
            alpha=0.85,
            label="随机前缀科学损失",
        )
        axes[row_index, 0].set_ylabel(f"seed {seed}\n在线 loss")
        _style_axis(axes[row_index, 0])

        axes[row_index, 1].plot(
            epochs,
            [_float(row, "endpoint_event_mae") for row in rows],
            color=COLORS["phase48"],
            linewidth=1.2,
            label="200 秒 Event MAE",
        )
        axes[row_index, 1].plot(
            epochs,
            [_float(row, "streaming_event_mae_mean") for row in rows],
            color=COLORS["green"],
            linewidth=1.2,
            label="流式锚点平均 Event MAE",
        )
        axes[row_index, 1].axhline(
            targets["endpoint_event_mae"],
            color=COLORS["red"],
            linestyle="--",
            linewidth=0.9,
        )
        _style_axis(axes[row_index, 1])
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[-1, 0].set_xlabel("epoch")
    axes[-1, 1].set_xlabel("epoch")
    figure.suptitle("Phase48 联合从头训练：前缀任务学会了，但终点校准仍有代价", y=1.01)
    figure.tight_layout()
    _save_figure(figure, output_dir / "03_phase48_training_curves")


def _summary(
    phase47: Mapping[str, Any],
    phase48: Mapping[str, Any],
    *,
    phase47_run: Path,
    phase48_run: Path,
) -> dict[str, Any]:
    closest_73 = _closest_row(phase48["seeds"][73]["epochs"])
    baseline_73 = phase48["seeds"][73]["baseline"]
    targets_73 = _gate_targets(baseline_73)
    return {
        "phase47": {
            "status": phase47["campaign"]["status"],
            "selected_seed": phase47["campaign"]["selected_seed"],
            "run_root": str(phase47_run),
        },
        "phase48": {
            "status": phase48["campaign"]["status"],
            "selected_seed": phase48["campaign"]["selected_seed"],
            "run_root": str(phase48_run),
        },
        "hidden_data": {
            "internal_test_iterated": False,
            "external_eight_events_loaded": False,
            "grouped_test_loaded": False,
        },
        "phase48_seed73_closest_epoch": {
            "epoch": int(closest_73["epoch"]),
            "selection_score": float(closest_73["selection_score"]),
            "endpoint_event_mae": float(closest_73["endpoint_event_mae"]),
            "endpoint_station_mae": float(closest_73["endpoint_station_mae"]),
            "streaming_event_mae_mean": float(
                closest_73["streaming_event_mae_mean"]
            ),
            "late_event_abs_step_p95_mw": float(
                closest_73["late_event_abs_step_p95_mw"]
            ),
            "late_station_abs_step_p95_mw": float(
                closest_73["late_station_abs_step_p95_mw"]
            ),
            "late_confirmed_cumulative_log10_l1_p95": float(
                closest_73["late_confirmed_cumulative_log10_l1_p95"]
            ),
            "endpoint_event_gate": targets_73["endpoint_event_mae"],
            "endpoint_station_gate": targets_73["endpoint_station_mae"],
        },
    }


def _manifest(output_dir: Path) -> None:
    files = [
        path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "publication_manifest.json"
    ]
    _write_json(
        output_dir / "publication_manifest.json",
        {
            "files": {
                str(path.relative_to(output_dir)): {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            }
        },
    )


def publish(
    *,
    phase47_run: Path,
    phase48_run: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    phase47 = _load_run(phase47_run, "Phase47")
    phase48 = _load_run(phase48_run, "Phase48")
    _write_csv(output_dir / "phase47_epoch_metrics.csv", _combined_epoch_rows(phase47))
    _write_csv(output_dir / "phase48_epoch_metrics.csv", _combined_epoch_rows(phase48))
    _write_csv(output_dir / "baseline_metrics.csv", _baseline_rows(phase47, phase48))
    _write_csv(output_dir / "closest_candidates.csv", _closest_rows(phase47, phase48))
    shutil.copy2(phase47_run / "campaign_summary.json", output_dir / "phase47_campaign_summary.json")
    shutil.copy2(phase48_run / "campaign_summary.json", output_dir / "phase48_campaign_summary.json")
    _write_json(
        output_dir / "summary.json",
        _summary(
            phase47,
            phase48,
            phase47_run=phase47_run,
            phase48_run=phase48_run,
        ),
    )
    _configure_plotting()
    figures = output_dir / "figures"
    _plot_tradeoff(phase47, phase48, figures)
    _plot_seed73_gate_ratios(phase48, figures)
    _plot_phase48_training(phase48, figures)
    _manifest(output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase47-run", type=Path, default=DEFAULT_PHASE47_RUN)
    parser.add_argument("--phase48-run", type=Path, default=DEFAULT_PHASE48_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    publish(
        phase47_run=args.phase47_run.resolve(),
        phase48_run=args.phase48_run.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
