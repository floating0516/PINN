from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path("/home/lihe/PINN_Mag")
DEFAULT_RUN_ROOT = (
    WORKSPACE_ROOT / "runs/phase39-expanded-fixed-split-20260831T0740Z"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "docs/results/phase39-expanded-fixed-split"
)
EXPECTED_ASSIGNMENT_SHA256 = (
    "e4807aa1e6b5b389caf23974f62ff9da6b8add7f7887ec23a59d5d35a455eba7"
)
EXPECTED_ROLE_COUNTS = {
    "train": (24, 1798),
    "validation": (6, 446),
    "test": (9, 450),
}
EXPECTED_SEEDS = (17, 42, 73)
MAP_EVENTS = ("Napa2014", "us7000i9bw", "Ridgecrest2019")
WAVEFORM_EVENT = "Ridgecrest2019"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments import run_phase39_confirmatory_grouped_cv as grouped
from scripts.plotting.plot_phase39_moment_scaling_explainer import (
    iter_selected_records,
)
from src.baseline.scaling_laws import AVAILABLE_SCALING_LAWS, predict_mw
from src.utils.provenance import sha256_file, utc_now_iso


COLORS = {
    "train": "#3B6EA8",
    "validation": "#E9A23B",
    "test": "#2A9D8F",
    "legacy": "#3B6EA8",
    "new": "#2A9D8F",
    "selected": "#D1495B",
    "positive": "#C94C4C",
    "negative": "#3B6EA8",
    "truth": "#202124",
    "grid": "#D7DCE2",
    "target": "#E9A23B",
    "station": "#6C757D",
}
SEED_COLORS = {
    17: "#4C78A8",
    42: "#7A6F9B",
    73: "#D1495B",
}
PGD_LAWS = ("crowell", "ruhl", "melgar")
METHOD_ORDER = ("phase39", *PGD_LAWS)
METHOD_LABELS = {
    "phase39": "Phase 39",
    "crowell": "Crowell 2013",
    "ruhl": "Ruhl 2019",
    "melgar": "Melgar 2015",
}
METHOD_COLORS = {
    "phase39": "#D1495B",
    "crowell": "#4C78A8",
    "ruhl": "#59A14F",
    "melgar": "#B279A2",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_figure(figure: plt.Figure, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    generated = []
    for suffix, dpi in ((".png", 220), (".pdf", 300)):
        path = output_stem.with_suffix(suffix)
        metadata = {"CreationDate": None, "ModDate": None} if suffix == ".pdf" else None
        figure.savefig(path, dpi=dpi, bbox_inches="tight", metadata=metadata)
        generated.append(path)
    plt.close(figure)
    return generated


def identity_limits(*series: pd.Series, padding: float = 0.16) -> tuple[float, float]:
    values = np.concatenate([item.to_numpy(dtype=float) for item in series])
    return float(np.nanmin(values) - padding), float(np.nanmax(values) + padding)


def _candidate_inputs(run_root: Path) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    rows = []
    logs: dict[int, pd.DataFrame] = {}
    for seed in EXPECTED_SEEDS:
        candidate_path = run_root / f"candidates/seed_{seed}/candidate_summary.json"
        candidate = load_json(candidate_path)
        if candidate.get("status") != "complete":
            raise ValueError(f"candidate is not complete: seed={seed}")
        if candidate.get("test_evaluated") is not False:
            raise ValueError(f"candidate opened test inference: seed={seed}")
        if candidate.get("split_assignment_sha256") != EXPECTED_ASSIGNMENT_SHA256:
            raise ValueError(f"candidate split hash changed: seed={seed}")
        log_path = Path(str(candidate["training_log_path"]))
        if sha256_file(log_path) != candidate["training_log_sha256"]:
            raise ValueError(f"candidate training log hash changed: seed={seed}")
        log = pd.read_csv(log_path)
        if log.empty:
            raise ValueError(f"candidate training log is empty: seed={seed}")
        log["seed"] = seed
        logs[seed] = log
        rows.append(
            {
                "seed": seed,
                "best_epoch": int(candidate["best_epoch"]),
                "best_validation_event_mae_mw": float(
                    candidate["best_validation_event_mae_mw"]
                ),
                "checkpoint_sha256": str(candidate["checkpoint_sha256"]),
                "epochs_completed": int(log["Epoch"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("seed"), logs


def load_inputs(
    run_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[int, pd.DataFrame],
]:
    if not (run_root / "COMPLETE").is_file():
        raise RuntimeError(f"formal campaign is incomplete: {run_root}")
    campaign = load_json(run_root / "campaign_summary.json")
    split = load_json(run_root / "fixed_split_manifest.json")
    protocol = load_json(run_root / "protocol.json")
    if campaign.get("status") != "complete":
        raise ValueError("campaign summary is not complete")
    if split.get("assignment_sha256") != EXPECTED_ASSIGNMENT_SHA256:
        raise ValueError("fixed split assignment hash changed")
    if protocol.get("split_assignment_sha256") != EXPECTED_ASSIGNMENT_SHA256:
        raise ValueError("protocol split assignment hash changed")
    if float(protocol.get("lambda_synth", -1.0)) != 0.5:
        raise ValueError("lambda_synth changed")
    if protocol.get("test_policy") != "evaluate_selected_seed_once":
        raise ValueError("test policy changed")

    for role, (events, records) in EXPECTED_ROLE_COUNTS.items():
        payload = split["roles"][role]
        if (int(payload["event_count"]), int(payload["record_count"])) != (
            events,
            records,
        ):
            raise ValueError(f"fixed split role count changed: {role}")

    events = pd.read_csv(run_root / "selected_test_event_predictions.csv")
    stations = pd.read_csv(run_root / "selected_test_station_predictions.csv")
    if len(events) != EXPECTED_ROLE_COUNTS["test"][0]:
        raise ValueError("test event coverage changed")
    if len(stations) != EXPECTED_ROLE_COUNTS["test"][1]:
        raise ValueError("test station coverage changed")
    if sha256_file(run_root / "selected_test_event_predictions.csv") != campaign[
        "test"
    ]["event_predictions_sha256"]:
        raise ValueError("test event predictions hash changed")
    if sha256_file(run_root / "selected_test_station_predictions.csv") != campaign[
        "test"
    ]["station_predictions_sha256"]:
        raise ValueError("test station predictions hash changed")

    legacy_events = set(split["legacy_test_events"])
    new_events = set(split["new_test_events"])
    if set(events["event"]) != legacy_events | new_events:
        raise ValueError("test event identities changed")
    events["test_class"] = [
        "new" if event in new_events else "legacy" for event in events["event"]
    ]
    stations["test_class"] = [
        "new" if event in new_events else "legacy"
        for event in stations["event"]
    ]
    events["absolute_error"] = events["error_vs_catalog"].abs()
    stations["absolute_error"] = stations["error_vs_catalog"].abs()
    candidates, logs = _candidate_inputs(run_root)
    selected_seed = int(campaign["selection"]["selected_seed"])
    if selected_seed != 73:
        raise ValueError(f"selected seed changed: {selected_seed}")
    if candidates.loc[
        candidates["best_validation_event_mae_mw"].idxmin(), "seed"
    ] != selected_seed:
        raise ValueError("selected seed is not the validation minimum")
    return campaign, split, protocol, events, stations, candidates, logs


def load_phase39_samples(config_path: Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    arm = grouped.build_arm_configs(config)["phase39"]
    return grouped._load_dataset_samples(arm)


def compute_pgd_baselines(
    samples: Sequence[dict[str, Any]],
    stations: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if stations.duplicated(["event", "station"]).any():
        raise ValueError("fixed-test station keys are not unique")
    station_lookup = stations.set_index(["event", "station"])
    station_keys = set(station_lookup.index)
    found_keys: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for sample in samples:
        key = (str(sample["event"]), str(sample["station"]))
        if key not in station_keys:
            continue
        found_keys.add(key)
        metadata = station_lookup.loc[key]
        radial = np.asarray(sample["radial"], dtype=float)
        tangential = np.asarray(sample["tangential"], dtype=float)
        vertical = np.asarray(sample["vertical"], dtype=float)
        pgd_3d_m = float(
            np.max(np.sqrt(radial**2 + tangential**2 + vertical**2))
        )
        source_distance_km = float(sample["source_distance_m"]) / 1000.0
        if not np.isfinite(pgd_3d_m) or pgd_3d_m <= 0.0:
            raise ValueError(f"invalid 3-component PGD for {key}")
        if not np.isfinite(source_distance_km) or source_distance_km <= 0.0:
            raise ValueError(f"invalid source distance for {key}")
        for method in PGD_LAWS:
            mw_pred = predict_mw(
                law_name=method,
                pgd_m=pgd_3d_m,
                source_distance_km=source_distance_km,
            )
            rows.append(
                {
                    "event": key[0],
                    "station": key[1],
                    "test_class": str(metadata["test_class"]),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "mw_catalog": float(metadata["mw_catalog"]),
                    "mw_pred": float(mw_pred),
                    "error_vs_catalog": float(mw_pred - metadata["mw_catalog"]),
                    "absolute_error": float(abs(mw_pred - metadata["mw_catalog"])),
                    "pgd_3d_m": pgd_3d_m,
                    "source_distance_km": source_distance_km,
                }
            )
    if found_keys != station_keys:
        missing = sorted(station_keys - found_keys)
        raise ValueError(f"PGD baseline sample coverage changed: missing={missing[:5]}")
    station_baselines = pd.DataFrame(rows).sort_values(
        ["method", "event", "station"]
    ).reset_index(drop=True)
    if len(station_baselines) != len(stations) * len(PGD_LAWS):
        raise ValueError("PGD station prediction count changed")

    event_baselines = (
        station_baselines.groupby(["method", "method_label", "event"], as_index=False)
        .agg(
            test_class=("test_class", "first"),
            mw_catalog=("mw_catalog", "median"),
            mw_pred_median=("mw_pred", "median"),
            prediction_iqr_mw=(
                "mw_pred",
                lambda values: float(
                    np.percentile(values, 75) - np.percentile(values, 25)
                ),
            ),
            prediction_std_mw=("mw_pred", "std"),
            n_stations=("station", "size"),
            pgd_median_cm=("pgd_3d_m", lambda values: float(np.median(values) * 100.0)),
        )
    )
    event_baselines["error_vs_catalog"] = (
        event_baselines["mw_pred_median"] - event_baselines["mw_catalog"]
    )
    event_baselines["absolute_error"] = event_baselines[
        "error_vs_catalog"
    ].abs()
    if len(event_baselines) != len(events) * len(PGD_LAWS):
        raise ValueError("PGD event prediction count changed")

    summary_rows = []
    for method in METHOD_ORDER:
        if method == "phase39":
            event_errors = events["error_vs_catalog"].to_numpy(dtype=float)
            station_errors = stations["error_vs_catalog"].to_numpy(dtype=float)
            input_channels = "radial waveform"
        else:
            event_errors = event_baselines.loc[
                event_baselines["method"] == method, "error_vs_catalog"
            ].to_numpy(dtype=float)
            station_errors = station_baselines.loc[
                station_baselines["method"] == method, "error_vs_catalog"
            ].to_numpy(dtype=float)
            input_channels = "3-component PGD"
        summary_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "input_channels": input_channels,
                "event_count": int(event_errors.size),
                "station_count": int(station_errors.size),
                "event_mae_mw": float(np.mean(np.abs(event_errors))),
                "event_rmse_mw": float(np.sqrt(np.mean(np.square(event_errors)))),
                "event_bias_mw": float(np.mean(event_errors)),
                "event_within_0_2_fraction": float(
                    np.mean(np.abs(event_errors) <= 0.2)
                ),
                "station_mae_mw": float(np.mean(np.abs(station_errors))),
                "station_rmse_mw": float(np.sqrt(np.mean(np.square(station_errors)))),
                "station_bias_mw": float(np.mean(station_errors)),
            }
        )
    method_summary = pd.DataFrame(summary_rows)
    return station_baselines, event_baselines, method_summary


def split_event_frame(split: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for role in ("train", "validation", "test"):
        for event in split["roles"][role]["events"]:
            rows.append({"role": role, **event})
    result = pd.DataFrame(rows)
    if len(result) != 39:
        raise ValueError("split event table changed")
    return result


def plot_split_overview(split: dict[str, Any], output_stem: Path) -> list[Path]:
    roles = ("train", "validation", "test")
    labels = ("Train", "Validation", "Test")
    events = [int(split["roles"][role]["event_count"]) for role in roles]
    records = [int(split["roles"][role]["record_count"]) for role in roles]
    frame = split_event_frame(split)
    figure, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.8), constrained_layout=True,
        gridspec_kw={"width_ratios": [0.8, 0.9, 2.1]},
    )
    colors = [COLORS[role] for role in roles]

    axes[0].bar(labels, events, color=colors, width=0.65)
    for index, value in enumerate(events):
        axes[0].text(index, value + 0.7, str(value), ha="center", va="bottom")
    axes[0].set_ylim(0, max(events) * 1.2)
    axes[0].set_ylabel("Number of events")
    axes[0].set_title("A. Event allocation", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color=COLORS["grid"], linewidth=0.6)

    axes[1].bar(labels, records, color=colors, width=0.65)
    for index, value in enumerate(records):
        axes[1].text(index, value + 45, f"{value:,}", ha="center", va="bottom")
    axes[1].set_ylim(0, max(records) * 1.2)
    axes[1].set_ylabel("Accepted station records")
    axes[1].set_title("B. Station allocation", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color=COLORS["grid"], linewidth=0.6)

    y_positions = {"train": 2, "validation": 1, "test": 0}
    for role in roles:
        rows = frame[frame["role"] == role].sort_values("magnitude_catalog")
        jitter = np.linspace(-0.16, 0.16, len(rows)) if len(rows) > 1 else [0.0]
        sizes = 28 + 150 * np.sqrt(rows["n_stations"] / frame["n_stations"].max())
        axes[2].scatter(
            rows["magnitude_catalog"],
            y_positions[role] + jitter,
            s=sizes,
            color=COLORS[role],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.86,
            label=role.title(),
        )
    axes[2].set_yticks([0, 1, 2], ["Test", "Validation", "Train"])
    axes[2].set_xlabel("Catalog magnitude (Mw)")
    axes[2].set_title(
        "C. Magnitude coverage (marker size = station count)",
        loc="left",
        fontweight="bold",
    )
    axes[2].grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    figure.suptitle(
        "Phase 39 expanded dataset: frozen 24 / 6 / 9 event split",
        fontweight="bold",
    )
    return save_figure(figure, output_stem)


def plot_seed_selection(
    candidates: pd.DataFrame,
    logs: dict[int, pd.DataFrame],
    selected_seed: int,
    output_stem: Path,
) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    bar_colors = [SEED_COLORS[int(seed)] for seed in candidates["seed"]]
    axes[0].bar(
        [str(seed) for seed in candidates["seed"]],
        candidates["best_validation_event_mae_mw"],
        color=bar_colors,
        width=0.62,
    )
    for index, row in enumerate(candidates.itertuples(index=False)):
        axes[0].text(
            index,
            float(row.best_validation_event_mae_mw) + 0.004,
            f"{row.best_validation_event_mae_mw:.3f}\nepoch {row.best_epoch}",
            ha="center",
            va="bottom",
        )
    axes[0].axhline(0.20, color=COLORS["target"], linewidth=1.1, linestyle="--")
    axes[0].axhline(0.17, color=COLORS["truth"], linewidth=1.0, linestyle=":")
    axes[0].text(2.45, 0.202, "0.20 target", ha="right", va="bottom")
    axes[0].text(2.45, 0.172, "0.17 target", ha="right", va="bottom")
    axes[0].set_ylim(0.14, 0.215)
    axes[0].set_xlabel("Random seed")
    axes[0].set_ylabel("Best validation event MAE (Mw)")
    axes[0].set_title("A. Validation-only model selection", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color=COLORS["grid"], linewidth=0.6)

    for seed in EXPECTED_SEEDS:
        log = logs[seed]
        selected = seed == selected_seed
        color = SEED_COLORS[seed]
        axes[1].plot(
            log["Epoch"],
            log["validation_event_mae_catalog"],
            color=color,
            alpha=0.95 if selected else 0.55,
            linewidth=1.8 if selected else 1.0,
            label=f"seed {seed}" + (" (selected)" if selected else ""),
        )
        row = candidates[candidates["seed"] == seed].iloc[0]
        axes[1].scatter(
            [row["best_epoch"]],
            [row["best_validation_event_mae_mw"]],
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=54 if selected else 38,
            zorder=3,
        )
    axes[1].axhspan(0.0, 0.2, color=COLORS["target"], alpha=0.10)
    axes[1].set_ylim(0.14, 0.85)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation event MAE (Mw)")
    axes[1].set_title("B. Validation trajectory", loc="left", fontweight="bold")
    axes[1].grid(True, color=COLORS["grid"], linewidth=0.55)
    axes[1].legend(frameon=False, loc="upper right")
    figure.suptitle(
        "Seed 73 selected before the test cohort was evaluated",
        fontweight="bold",
    )
    return save_figure(figure, output_stem)


def plot_test_event_scatter(events: pd.DataFrame, output_stem: Path) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    lower = float(min(events["mw_catalog"].min(), events["mw_pred_median"].min()) - 0.18)
    upper = float(max(events["mw_catalog"].max(), events["mw_pred_median"].max()) + 0.18)
    line = np.linspace(lower, upper, 200)
    axis.fill_between(
        line,
        line - 0.2,
        line + 0.2,
        color=COLORS["target"],
        alpha=0.13,
        label="+/-0.20 Mw target",
    )
    axis.plot(line, line, color=COLORS["truth"], linewidth=1.25)
    for category, marker, label in (
        ("legacy", "o", "Six legacy test events"),
        ("new", "^", "Three newly added test events"),
    ):
        rows = events[events["test_class"] == category]
        sizes = 52 + 160 * np.sqrt(rows["n_stations"] / events["n_stations"].max())
        axis.scatter(
            rows["mw_catalog"],
            rows["mw_pred_median"],
            s=sizes,
            marker=marker,
            color=COLORS[category],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            label=label,
            zorder=3,
        )
    offsets = {
        "2016p661332": (7, 7),
        "Ecuador2016": (7, -14),
        "Iquique2014": (-70, -8),
        "Napa2014": (7, 7),
        "Puebla2017": (-62, -8),
        "Ridgecrest2019": (7, 7),
        "Tokachi2003": (-68, 7),
        "ak014cbigci8": (7, 7),
        "us7000i9bw": (7, -14),
    }
    for row in events.itertuples(index=False):
        axis.annotate(
            str(row.event),
            (float(row.mw_catalog), float(row.mw_pred_median)),
            xytext=offsets.get(str(row.event), (6, 6)),
            textcoords="offset points",
            fontsize=8,
        )
    metrics = events["absolute_error"]
    axis.text(
        0.03,
        0.97,
        "Test event MAE = 0.350 Mw\n"
        f"Within +/-0.20 Mw = {(metrics <= 0.2).sum()}/9",
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": COLORS["grid"], "alpha": 0.92},
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Catalog magnitude (Mw)")
    axis.set_ylabel("Estimated magnitude (Mw)")
    axis.set_title(
        "One-time evaluation of the nine held-out test events",
        loc="left",
        fontweight="bold",
    )
    axis.grid(True, color=COLORS["grid"], linewidth=0.6)
    axis.legend(frameon=False, loc="lower right", fontsize=9)
    return save_figure(figure, output_stem)


def plot_event_errors(events: pd.DataFrame, output_stem: Path) -> list[Path]:
    rows = events.sort_values("absolute_error", ascending=True).reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(10.5, 6.4), constrained_layout=True)
    y = np.arange(len(rows))
    colors = [COLORS[value] for value in rows["test_class"]]
    bars = axis.barh(y, rows["absolute_error"], color=colors, alpha=0.92)
    axis.axvline(0.2, color=COLORS["target"], linewidth=1.5, linestyle="--")
    axis.set_yticks(y, rows["event"])
    label_limit = max(0.45, float(rows["absolute_error"].max()) * 1.18)
    axis.set_xlim(0.0, label_limit)
    for bar, row in zip(bars, rows.itertuples(index=False)):
        axis.text(
            float(row.absolute_error) + label_limit * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{row.absolute_error:.3f} (signed {row.error_vs_catalog:+.3f})",
            va="center",
            ha="left",
            fontsize=8.2,
        )
    axis.set_xlabel("Absolute endpoint magnitude error (Mw)")
    axis.set_title(
        "Absolute error for the nine held-out test earthquakes",
        loc="left",
        fontweight="bold",
    )
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    legend = [
        Line2D([0], [0], color=COLORS["legacy"], linewidth=7, label="Legacy test event"),
        Line2D([0], [0], color=COLORS["new"], linewidth=7, label="New test event"),
        Line2D([0], [0], color=COLORS["target"], linestyle="--", label="0.20 Mw target"),
    ]
    axis.legend(handles=legend, frameon=False, loc="lower right")
    return save_figure(figure, output_stem)


def plot_station_predictions(
    stations: pd.DataFrame,
    events: pd.DataFrame,
    output_stem: Path,
) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.7), constrained_layout=True)
    lower, upper = identity_limits(stations["mw_catalog"], stations["mw_pred"])
    for axis in axes:
        axis.plot([lower, upper], [lower, upper], color=COLORS["truth"], linewidth=1.1)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_xlabel("Catalog magnitude (Mw)")
        axis.set_ylabel("Endpoint estimated magnitude (Mw)")
        axis.grid(True, color=COLORS["grid"], linewidth=0.55)
    density = axes[0].hexbin(
        stations["mw_catalog"],
        stations["mw_pred"],
        gridsize=42,
        mincnt=1,
        cmap="viridis",
        linewidths=0.0,
    )
    figure.colorbar(density, ax=axes[0], pad=0.02, label="Station count")
    axes[0].set_title("A. All 450 station estimates", loc="left", fontweight="bold")
    axes[1].scatter(
        stations["mw_catalog"],
        stations["mw_pred"],
        s=9,
        color=COLORS["station"],
        alpha=0.16,
        linewidth=0,
        rasterized=True,
        label="Stations",
    )
    for category, marker, size, label in (
        ("legacy", "o", 52, "Legacy event median"),
        ("new", "^", 78, "New event median"),
    ):
        rows = events[events["test_class"] == category]
        axes[1].scatter(
            rows["mw_catalog"],
            rows["mw_pred_median"],
            marker=marker,
            s=size,
            color=COLORS[category],
            edgecolor="white",
            linewidth=0.7,
            label=label,
            zorder=3,
        )
    axes[1].set_title(
        "B. Station estimates with event medians",
        loc="left",
        fontweight="bold",
    )
    axes[1].legend(frameon=False, loc="lower right", fontsize=8.5)
    figure.suptitle(
        "Station-level endpoint magnitude estimates on the held-out test cohort",
        fontweight="bold",
    )
    return save_figure(figure, output_stem)


def plot_selected_training(
    selected_log: pd.DataFrame,
    best_epoch: int,
    best_mae: float,
    output_stem: Path,
) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), constrained_layout=True)
    axes[0].plot(selected_log["Epoch"], selected_log["Train_Loss"], label="Total loss", linewidth=1.5)
    axes[0].plot(selected_log["Epoch"], selected_log["Train_Data_Loss"], label="Data loss", linewidth=1.2)
    axes[0].plot(selected_log["Epoch"], selected_log["Train_Phys_Loss"], label="Synth/physics loss", linewidth=1.2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss (log scale)")
    axes[0].set_title("A. Loss decomposition", loc="left", fontweight="bold")
    axes[0].grid(True, color=COLORS["grid"], linewidth=0.55)
    axes[0].legend(frameon=False, fontsize=8.5)

    axes[1].plot(
        selected_log["Epoch"],
        selected_log["validation_event_mae_catalog"],
        color=COLORS["selected"],
        label="Event MAE",
        linewidth=1.5,
    )
    axes[1].plot(
        selected_log["Epoch"],
        selected_log["validation_station_mae_catalog"],
        color=COLORS["legacy"],
        label="Station MAE",
        linewidth=1.1,
        alpha=0.85,
    )
    axes[1].axhspan(0.0, 0.2, color=COLORS["target"], alpha=0.11)
    axes[1].scatter(
        [best_epoch], [best_mae], color=COLORS["selected"], edgecolor="white", s=70, zorder=3
    )
    axes[1].annotate(
        f"selected epoch {best_epoch}\n{best_mae:.3f} Mw",
        (best_epoch, best_mae),
        xytext=(14, 34),
        textcoords="offset points",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": COLORS["grid"], "alpha": 0.92},
        arrowprops={"arrowstyle": "-", "color": COLORS["selected"], "linewidth": 0.8},
    )
    axes[1].set_ylim(0.14, 0.85)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation MAE (Mw)")
    axes[1].set_title("B. Validation metrics", loc="left", fontweight="bold")
    axes[1].grid(True, color=COLORS["grid"], linewidth=0.55)
    axes[1].legend(frameon=False, fontsize=8.5)

    axes[2].plot(selected_log["Epoch"], selected_log["LR"], color=COLORS["test"], linewidth=1.4)
    axes[2].axvline(best_epoch, color=COLORS["selected"], linewidth=1.0, linestyle="--")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_title("C. Warm-restart schedule", loc="left", fontweight="bold")
    axes[2].grid(True, color=COLORS["grid"], linewidth=0.55)
    figure.suptitle(
        "Selected seed 73 training history (lambda_synth = 0.5)",
        fontweight="bold",
    )
    return save_figure(figure, output_stem)


def coordinate_frame(
    data_path: Path,
    station_keys: set[tuple[str, str]],
    selected_events: set[str],
) -> pd.DataFrame:
    rows = []
    with np.load(data_path, allow_pickle=True) as data:
        for record in iter_selected_records(data, selected_events):
            key = (record.event, record.station)
            if key not in station_keys:
                continue
            rows.append(
                {
                    "event": record.event,
                    "station": record.station,
                    "event_lon": record.event_lon,
                    "event_lat": record.event_lat,
                    "station_lon": record.station_lon,
                    "station_lat": record.station_lat,
                }
            )
    result = pd.DataFrame(rows).drop_duplicates(["event", "station"])
    expected = sum(event in selected_events for event, _station in station_keys)
    if len(result) != expected:
        raise ValueError(
            f"coordinate join changed: expected={expected}, actual={len(result)}"
        )
    return result


def map_region(frame: pd.DataFrame) -> tuple[float, float, float, float]:
    longitudes = np.concatenate(
        [frame["station_lon"].to_numpy(), frame["event_lon"].to_numpy()]
    )
    latitudes = np.concatenate(
        [frame["station_lat"].to_numpy(), frame["event_lat"].to_numpy()]
    )
    lon_span = max(float(np.ptp(longitudes)), 0.1)
    lat_span = max(float(np.ptp(latitudes)), 0.1)
    return (
        float(np.min(longitudes) - max(0.08, 0.10 * lon_span)),
        float(np.max(longitudes) + max(0.08, 0.10 * lon_span)),
        float(np.min(latitudes) - max(0.08, 0.10 * lat_span)),
        float(np.max(latitudes) + max(0.08, 0.10 * lat_span)),
    )


def coastline_segments(region: tuple[float, float, float, float]) -> list[np.ndarray]:
    west, east, south, north = region
    with tempfile.TemporaryDirectory(prefix="phase39-fixed-map-") as temp_name:
        result = subprocess.run(
            [
                "gmt",
                "coast",
                f"-R{west:.6f}/{east:.6f}/{south:.6f}/{north:.6f}",
                "-M",
                "-W0.5p",
                "-Dh",
            ],
            cwd=temp_name,
            check=True,
            text=True,
            capture_output=True,
        )
    segments: list[np.ndarray] = []
    current: list[tuple[float, float]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            if len(current) >= 2:
                segments.append(np.asarray(current, dtype=float))
            current = []
            continue
        tokens = stripped.split()
        if len(tokens) >= 2:
            current.append((float(tokens[0]), float(tokens[1])))
    if len(current) >= 2:
        segments.append(np.asarray(current, dtype=float))
    return segments


def plot_selected_maps(
    station_predictions: pd.DataFrame,
    data_path: Path,
    output_stem: Path,
) -> list[Path]:
    if subprocess.run(["gmt", "--version"], capture_output=True).returncode != 0:
        raise RuntimeError("GMT is required for station maps")
    selected = station_predictions[
        station_predictions["event"].isin(MAP_EVENTS)
    ].copy()
    selected["residual_mw"] = selected["mw_pred"] - selected["mw_catalog"]
    keys = set(zip(selected["event"], selected["station"]))
    coordinates = coordinate_frame(data_path, keys, set(MAP_EVENTS))
    joined = selected.merge(
        coordinates, on=["event", "station"], validate="one_to_one"
    )
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.7), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=0.8)
    station_marks = None
    for axis, event in zip(axes, MAP_EVENTS):
        frame = joined[joined["event"] == event]
        if frame.empty:
            raise ValueError(f"selected map event has no stations: {event}")
        west, east, south, north = map_region(frame)
        axis.set_facecolor("#EAF4FB")
        for segment in coastline_segments((west, east, south, north)):
            axis.plot(segment[:, 0], segment[:, 1], color="#59636E", linewidth=0.75)
        size = 11 if len(frame) > 200 else 23 if len(frame) > 50 else 48
        station_marks = axis.scatter(
            frame["station_lon"],
            frame["station_lat"],
            c=frame["residual_mw"].clip(-0.8, 0.8),
            cmap="coolwarm",
            norm=norm,
            s=size,
            edgecolor="#263238",
            linewidth=0.25,
            alpha=0.88,
            rasterized=True,
            zorder=3,
        )
        axis.scatter(
            [float(frame["event_lon"].iloc[0])],
            [float(frame["event_lat"].iloc[0])],
            marker="*",
            s=220,
            color="#F2C94C",
            edgecolor="#202124",
            linewidth=0.8,
            zorder=4,
        )
        axis.set_xlim(west, east)
        axis.set_ylim(south, north)
        mean_latitude = 0.5 * (south + north)
        axis.set_aspect(1.0 / max(math.cos(math.radians(mean_latitude)), 0.2))
        axis.set_xlabel("Longitude (deg)")
        axis.set_ylabel("Latitude (deg)")
        event_error = abs(
            float(frame["mw_pred"].median()) - float(frame["mw_catalog"].iloc[0])
        )
        axis.set_title(
            f"{event}\n{len(frame)} stations | event error {event_error:.3f} Mw",
            fontweight="bold",
        )
        axis.grid(True, color="#C9D3DC", linewidth=0.5, alpha=0.75)
    if station_marks is None:
        raise ValueError("selected map cohort is empty")
    colorbar = figure.colorbar(
        station_marks, ax=axes, orientation="horizontal", pad=0.08
    )
    colorbar.set_label(
        "Station residual: endpoint estimated Mw - catalog Mw (clipped at +/-0.8)"
    )
    figure.suptitle(
        "Epicenters, GNSS stations, and endpoint residuals for selected test events",
        fontsize=14,
        fontweight="bold",
    )
    return save_figure(figure, output_stem)


def load_waveform_sample(
    samples: Sequence[dict[str, Any]],
    station_predictions: pd.DataFrame,
) -> tuple[dict[str, Any], str]:
    event_rows = station_predictions[
        station_predictions["event"] == WAVEFORM_EVENT
    ].copy()
    if event_rows.empty:
        raise ValueError(f"waveform event is absent from test predictions: {WAVEFORM_EVENT}")
    event_rows["absolute_error"] = event_rows["error_vs_catalog"].abs()
    target_error = float(event_rows["absolute_error"].median())
    event_rows["selection_distance"] = (
        event_rows["absolute_error"] - target_error
    ).abs()
    selected_station = str(
        event_rows.sort_values(["selection_distance", "station"]).iloc[0]["station"]
    )
    for sample in samples:
        if sample["event"] == WAVEFORM_EVENT and sample["station"] == selected_station:
            return sample, selected_station
    raise ValueError(f"waveform sample not found: {WAVEFORM_EVENT}/{selected_station}")


def plot_waveform_example(
    sample: dict[str, Any],
    selected_station: str,
    stations: pd.DataFrame,
    events: pd.DataFrame,
    output_stem: Path,
) -> list[Path]:
    event_stations = stations[stations["event"] == WAVEFORM_EVENT].sort_values(
        "epicentral_distance_km"
    )
    event_row = events[events["event"] == WAVEFORM_EVENT].iloc[0]
    radial_cm = np.asarray(sample["radial"], dtype=float) * 100.0
    tangential_cm = np.asarray(sample["tangential"], dtype=float) * 100.0
    vertical_cm = np.asarray(sample["vertical"], dtype=float) * 100.0
    time_sec = float(sample["waveform_start_sec"]) + np.arange(radial_cm.size) * float(
        sample["waveform_dt_sec"]
    )
    norm_cm = np.sqrt(radial_cm**2 + tangential_cm**2 + vertical_cm**2)
    pgd_cm = np.maximum.accumulate(norm_cm)
    catalog = float(event_row["mw_catalog"])

    figure, axes = plt.subplots(3, 1, figsize=(12.5, 9.3), constrained_layout=True)
    axes[0].plot(time_sec, radial_cm, color=COLORS["legacy"], linewidth=1.2)
    axes[0].axhline(0.0, color=COLORS["grid"], linewidth=0.8)
    axes[0].set_ylabel("Radial displacement (cm)")
    axes[0].set_title(
        "A. Radial waveform used by the R-only model",
        loc="left",
        fontweight="bold",
    )

    axes[1].plot(
        time_sec,
        norm_cm,
        color="#8D99A6",
        linewidth=1.0,
        label="3-component norm",
    )
    axes[1].plot(
        time_sec,
        pgd_cm,
        color=COLORS["new"],
        linewidth=2.0,
        label="Cumulative PGD",
    )
    axes[1].set_ylabel("Displacement / PGD (cm)")
    axes[1].set_xlabel("Time since origin (s)")
    axes[1].set_title("B. Peak ground displacement", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)

    axes[2].scatter(
        event_stations["epicentral_distance_km"],
        event_stations["mw_pred"],
        s=15,
        color=COLORS["station"],
        alpha=0.42,
        linewidth=0,
        rasterized=True,
        label="Station endpoint estimates",
    )
    selected = event_stations[event_stations["station"] == selected_station].iloc[0]
    axes[2].scatter(
        [float(selected["epicentral_distance_km"])],
        [float(selected["mw_pred"])],
        marker="D",
        s=75,
        color=COLORS["new"],
        edgecolor="white",
        linewidth=0.7,
        label="Waveform station",
        zorder=4,
    )
    axes[2].axhline(
        catalog,
        color=COLORS["truth"],
        linewidth=1.4,
        label=f"Catalog Mw {catalog:.2f}",
    )
    axes[2].axhline(
        float(event_row["mw_pred_median"]),
        color=COLORS["selected"],
        linewidth=1.5,
        linestyle="--",
        label=f"Test event endpoint estimate {float(event_row['mw_pred_median']):.2f}",
    )
    axes[2].set_xlabel("Epicentral distance (km)")
    axes[2].set_ylabel("Endpoint estimated magnitude (Mw)")
    axes[2].set_title(
        f"C. Seed-73 endpoint estimates; station {selected_station} = {float(selected['mw_pred']):.2f} Mw",
        loc="left",
        fontweight="bold",
    )
    axes[2].legend(frameon=False, fontsize=8, ncol=2)

    for axis in axes:
        axis.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    figure.suptitle(
        f"{WAVEFORM_EVENT} representative station {selected_station}: waveform, PGD, and endpoint estimates",
        fontsize=14,
        fontweight="bold",
    )
    return save_figure(figure, output_stem)


def method_event_frame(
    events: pd.DataFrame,
    pgd_events: pd.DataFrame,
) -> pd.DataFrame:
    phase39 = events[
        [
            "event",
            "test_class",
            "mw_catalog",
            "mw_pred_median",
            "n_stations",
            "error_vs_catalog",
            "absolute_error",
        ]
    ].copy()
    phase39["method"] = "phase39"
    phase39["method_label"] = METHOD_LABELS["phase39"]
    return pd.concat([phase39, pgd_events], ignore_index=True, sort=False)


def plot_pgd_method_scatter(
    events: pd.DataFrame,
    pgd_events: pd.DataFrame,
    method_summary: pd.DataFrame,
    output_stem: Path,
) -> list[Path]:
    comparison = method_event_frame(events, pgd_events)
    lower, upper = identity_limits(
        comparison["mw_catalog"], comparison["mw_pred_median"], padding=0.20
    )
    identity = np.linspace(lower, upper, 200)
    summary_lookup = method_summary.set_index("method")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.8, 10.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panel_labels = ("A", "B", "C", "D")
    for axis, panel, method in zip(axes.flat, panel_labels, METHOD_ORDER):
        rows = comparison[comparison["method"] == method]
        axis.fill_between(
            identity,
            identity - 0.2,
            identity + 0.2,
            color=COLORS["target"],
            alpha=0.12,
        )
        axis.plot(identity, identity, color=COLORS["truth"], linewidth=1.15)
        for test_class, marker in (("legacy", "o"), ("new", "^")):
            subset = rows[rows["test_class"] == test_class]
            sizes = 48 + 110 * np.sqrt(
                subset["n_stations"] / comparison["n_stations"].max()
            )
            axis.scatter(
                subset["mw_catalog"],
                subset["mw_pred_median"],
                s=sizes,
                marker=marker,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.92,
                zorder=3,
            )
        metrics = summary_lookup.loc[method]
        axis.text(
            0.04,
            0.96,
            f"Event MAE = {float(metrics['event_mae_mw']):.3f} Mw\n"
            f"Within +/-0.20 = {float(metrics['event_within_0_2_fraction']):.0%}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "edgecolor": COLORS["grid"],
                "alpha": 0.90,
            },
        )
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"{panel}. {METHOD_LABELS[method]}", loc="left", fontweight="bold"
        )
        axis.grid(True, color=COLORS["grid"], linewidth=0.55)
        axis.set_xlabel("Catalog magnitude (Mw)")
        axis.set_ylabel("Endpoint estimated magnitude (Mw)")
    axes[0, 0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=METHOD_COLORS["phase39"],
                markeredgecolor="white",
                markersize=8,
                label="Legacy test event",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor=METHOD_COLORS["phase39"],
                markeredgecolor="white",
                markersize=9,
                label="New test event",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["target"],
                linewidth=7,
                alpha=0.35,
                label="+/-0.20 Mw",
            ),
        ],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )
    figure.suptitle(
        "Phase 39 versus three empirical PGD magnitude relations\n"
        "Same nine held-out events and 450 accepted stations",
        fontsize=14,
        fontweight="bold",
    )
    return save_figure(figure, output_stem)


def plot_pgd_method_errors(
    events: pd.DataFrame,
    pgd_events: pd.DataFrame,
    method_summary: pd.DataFrame,
    output_stem: Path,
) -> list[Path]:
    summary = method_summary.set_index("method").loc[list(METHOD_ORDER)].reset_index()
    comparison = method_event_frame(events, pgd_events)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15.0, 6.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.9, 1.45]},
    )

    x = np.arange(len(summary))
    width = 0.34
    event_bars = axes[0].bar(
        x - width / 2,
        summary["event_mae_mw"],
        width,
        color=[METHOD_COLORS[method] for method in summary["method"]],
        alpha=0.95,
        label="Event MAE",
    )
    station_bars = axes[0].bar(
        x + width / 2,
        summary["station_mae_mw"],
        width,
        color=[METHOD_COLORS[method] for method in summary["method"]],
        alpha=0.45,
        hatch="//",
        label="Station MAE",
    )
    axes[0].axhline(
        0.2,
        color=COLORS["target"],
        linewidth=1.4,
        linestyle="--",
        label="0.20 Mw target",
    )
    axes[0].set_xticks(x, summary["method_label"], rotation=18, ha="right")
    axes[0].set_ylabel("Mean absolute error (Mw)")
    axes[0].set_title("A. Overall test errors", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color=COLORS["grid"], linewidth=0.55)
    axes[0].legend(frameon=False, fontsize=8.5)
    upper = max(
        float(summary["event_mae_mw"].max()),
        float(summary["station_mae_mw"].max()),
    ) * 1.25
    axes[0].set_ylim(0.0, upper)
    for bars in (event_bars, station_bars):
        for bar in bars:
            value = float(bar.get_height())
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + upper * 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )

    phase_order = events.sort_values("absolute_error", ascending=False)["event"].tolist()
    matrix = (
        comparison.pivot(index="event", columns="method", values="absolute_error")
        .loc[phase_order, list(METHOD_ORDER)]
        .to_numpy(dtype=float)
    )
    image = axes[1].imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.0)
    axes[1].set_xticks(
        np.arange(len(METHOD_ORDER)),
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        rotation=18,
        ha="right",
    )
    axes[1].set_yticks(np.arange(len(phase_order)), phase_order)
    axes[1].set_title(
        "B. Absolute error by held-out event", loc="left", fontweight="bold"
    )
    threshold = float(np.nanmax(matrix)) * 0.58
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = float(matrix[row_index, column_index])
            axes[1].text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= threshold else COLORS["truth"],
                fontsize=8.5,
            )
    colorbar = figure.colorbar(image, ax=axes[1], pad=0.02)
    colorbar.set_label("Absolute event error (Mw)")
    figure.suptitle(
        "Fixed-test error comparison: Phase 39 and empirical PGD baselines",
        fontsize=14,
        fontweight="bold",
    )
    return save_figure(figure, output_stem)


def event_station_summary(stations: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    station_summary = (
        stations.groupby("event", as_index=False)
        .agg(
            station_count=("station", "size"),
            station_mae_mw=("absolute_error", "mean"),
            station_rmse_mw=(
                "error_vs_catalog",
                lambda values: float(np.sqrt(np.mean(np.square(values)))),
            ),
            station_bias_mw=("error_vs_catalog", "mean"),
            station_prediction_std_mw=("mw_pred", "std"),
        )
    )
    return events.merge(station_summary, on="event", validate="one_to_one")


def build_public_summary(
    campaign: dict[str, Any],
    split: dict[str, Any],
    protocol: dict[str, Any],
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    method_summary: pd.DataFrame,
) -> dict[str, Any]:
    diagnostic_rows = events[~events["event"].isin(["Napa2014", "ak014cbigci8"])]
    return {
        "status": "complete",
        "created_at_utc": utc_now_iso(),
        "method": "Phase 39 expanded fixed split with synthesized-waveform loss",
        "lambda_synth": float(protocol["lambda_synth"]),
        "split_assignment_sha256": split["assignment_sha256"],
        "role_counts": {
            role: {
                "events": int(split["roles"][role]["event_count"]),
                "stations": int(split["roles"][role]["record_count"]),
            }
            for role in ("train", "validation", "test")
        },
        "selection": {
            "metric": campaign["selection"]["selection_metric"],
            "selected_seed": int(campaign["selection"]["selected_seed"]),
            "selected_validation_event_mae_mw": float(
                campaign["selection"]["selected_validation_event_mae_mw"]
            ),
            "selected_checkpoint_sha256": campaign["selection"][
                "selected_checkpoint_sha256"
            ],
            "candidates": candidates.to_dict(orient="records"),
        },
        "test": campaign["test"]["test_event_metrics"],
        "test_subgroups": {
            "legacy": campaign["test"]["legacy_test_event_metrics"],
            "new": campaign["test"]["new_test_event_metrics"],
        },
        "diagnostic_excluding_two_largest_outliers": {
            "excluded_events": ["Napa2014", "ak014cbigci8"],
            "event_count": int(len(diagnostic_rows)),
            "event_mae_mw": float(diagnostic_rows["absolute_error"].mean()),
            "headline_metric": False,
        },
        "pgd_comparison": {
            "contract": {
                "events": 9,
                "stations": 450,
                "window_seconds": 200,
                "event_aggregation": "station_prediction_median",
                "phase39_input": "radial waveform",
                "empirical_input": "3-component PGD",
                "distance": "hypocentral",
            },
            "methods": method_summary.to_dict(orient="records"),
            "scaling_laws": {
                method: {
                    "citation_key": AVAILABLE_SCALING_LAWS[method].citation_key,
                    "pgd_unit": AVAILABLE_SCALING_LAWS[method].pgd_unit,
                    "a": AVAILABLE_SCALING_LAWS[method].a,
                    "b": AVAILABLE_SCALING_LAWS[method].b,
                    "c": AVAILABLE_SCALING_LAWS[method].c,
                }
                for method in PGD_LAWS
            },
        },
    }


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    test = summary["test"]
    selection = summary["selection"]
    legacy = summary["test_subgroups"]["legacy"]
    new = summary["test_subgroups"]["new"]
    diagnostic = summary["diagnostic_excluding_two_largest_outliers"]
    pgd_methods = {
        row["method"]: row for row in summary["pgd_comparison"]["methods"]
    }
    content = f"""# Phase 39 Expanded Dataset: Fixed-Split Evaluation

This page publishes the validation-selected, one-time held-out test result for
the expanded Phase 39 dataset. The model retains the synthesized-waveform loss
with `lambda_synth = 0.5`.

## Headline Result

| Metric | Result |
|---|---:|
| Train / validation / test events | 24 / 6 / 9 |
| Accepted station records | 1,798 / 446 / 450 |
| Selected seed | {selection['selected_seed']} |
| Fixed-validation event MAE | **{selection['selected_validation_event_mae_mw']:.6f} Mw** |
| Held-out test event MAE | **{test['event_mae']:.6f} Mw** |
| Held-out test event RMSE | {test['event_rmse']:.6f} Mw |
| Held-out test station MAE | {test['station_mae']:.6f} Mw |
| Held-out test station RMSE | {test['station_rmse']:.6f} Mw |

Seeds 17, 42, and 73 were trained independently. The seed and checkpoint were
selected only by the six fixed validation events. The nine-event test cohort
was evaluated once after seed 73 had been selected; test results were not
averaged across seeds and were not used for model selection.

## Event-Level Test Estimates

![Test event estimates](figures/en/01_test_event_scatter.png)

[PDF](figures/en/01_test_event_scatter.pdf) |
[Event prediction table](selected_test_event_predictions.csv)

## Station-Level Test Estimates

![Station-level test estimates](figures/en/02_test_station_scatter.png)

[PDF](figures/en/02_test_station_scatter.pdf) |
[Station predictions](selected_test_station_predictions.csv) |
[Per-event station summary](analysis/test_event_station_summary.csv)

## Event Errors

![Absolute test event errors](figures/en/03_test_event_absolute_errors.png)

[PDF](figures/en/03_test_event_absolute_errors.pdf) |
[Event error analysis](analysis/test_event_error_analysis.csv)

The six legacy test events have MAE `{legacy['mae_mw']:.6f} Mw`; the three newly
added test events have MAE `{new['mae_mw']:.6f} Mw`. The two largest failures
are Napa2014 and `ak014cbigci8`. Excluding them only for diagnosis, the remaining
seven-event MAE is `{diagnostic['event_mae_mw']:.6f} Mw`; this is not the headline metric.

## Selected Training Run

![Selected seed training history](figures/en/04_selected_seed_training_curves.png)

[PDF](figures/en/04_selected_seed_training_curves.pdf) |
[Seed 73 training log](analysis/training_logs/seed_73.csv)

## Selected Test-Event Maps

![Selected test-event maps](figures/en/05_selected_test_event_maps.png)

[PDF](figures/en/05_selected_test_event_maps.pdf)

The maps show Napa2014, `us7000i9bw`, and Ridgecrest2019. Station color is the
seed-73 endpoint residual relative to catalog magnitude; the star is the
catalog epicenter.

## Representative Waveform And PGD

![Representative waveform and PGD](figures/en/06_ridgecrest_waveform_and_predictions.png)

[PDF](figures/en/06_ridgecrest_waveform_and_predictions.pdf)

This figure shows one Ridgecrest2019 station waveform, the three-component norm
and cumulative PGD, and the selected model's station/event endpoint estimates.
It does not show a second-by-second magnitude prediction.

## Comparison With Three Empirical PGD Relations

![Phase 39 and PGD method scatter comparison](figures/en/07_phase39_vs_pgd_method_scatter.png)

[PDF](figures/en/07_phase39_vs_pgd_method_scatter.pdf) |
[PGD event predictions](analysis/pgd_event_predictions.csv)

![Phase 39 and PGD error comparison](figures/en/08_phase39_vs_pgd_method_errors.png)

[PDF](figures/en/08_phase39_vs_pgd_method_errors.pdf) |
[PGD method summary](analysis/pgd_method_summary.csv) |
[PGD station predictions](analysis/pgd_station_predictions.csv)

| Method | Event MAE | Event RMSE | Event bias | Station MAE |
|---|---:|---:|---:|---:|
| Phase 39 | {pgd_methods['phase39']['event_mae_mw']:.6f} | {pgd_methods['phase39']['event_rmse_mw']:.6f} | {pgd_methods['phase39']['event_bias_mw']:+.6f} | {pgd_methods['phase39']['station_mae_mw']:.6f} |
| Crowell 2013 | {pgd_methods['crowell']['event_mae_mw']:.6f} | {pgd_methods['crowell']['event_rmse_mw']:.6f} | {pgd_methods['crowell']['event_bias_mw']:+.6f} | {pgd_methods['crowell']['station_mae_mw']:.6f} |
| Ruhl 2019 | {pgd_methods['ruhl']['event_mae_mw']:.6f} | {pgd_methods['ruhl']['event_rmse_mw']:.6f} | {pgd_methods['ruhl']['event_bias_mw']:+.6f} | {pgd_methods['ruhl']['station_mae_mw']:.6f} |
| Melgar 2015 | {pgd_methods['melgar']['event_mae_mw']:.6f} | {pgd_methods['melgar']['event_rmse_mw']:.6f} | {pgd_methods['melgar']['event_bias_mw']:+.6f} | {pgd_methods['melgar']['station_mae_mw']:.6f} |

The comparison uses the same nine test events, all 450 accepted stations, the
same 200-second preprocessed waveforms, hypocentral distance, and station-median
event aggregation. The empirical formulas use three-component PGD, while Phase
39 remains an R-only waveform model. Their published coefficients are applied
without fitting on this dataset. Crowell has the lowest event MAE on this small
fixed test cohort; Phase 39 is not the best method under this endpoint metric.

## Supplementary Protocol Diagnostics

![Frozen data split](figures/en/09_split_overview_supplement.png)

[PDF](figures/en/09_split_overview_supplement.pdf) |
[Fixed split manifest](fixed_split_manifest.json)

![Seed validation selection](figures/en/10_seed_validation_selection_supplement.png)

[PDF](figures/en/10_seed_validation_selection_supplement.pdf) |
[Seed selection table](analysis/seed_selection.csv)

Seed 73 reached the lowest validation event MAE,
`{selection['selected_validation_event_mae_mw']:.6f} Mw`, at epoch 61. This
validation number is not the independent test result.

## Interpretation

The selected checkpoint performs well on the fixed validation cohort but does
not maintain the same error on the independent test events. The gap is dominated
by Napa2014 (`+0.836831 Mw`) and `ak014cbigci8` (`+0.986244 Mw`), showing that
the current synth-constrained model remains sensitive to event distribution and
sparse M6-class cases.

This is an endpoint magnitude experiment. It is not a causal, second-by-second
prefix experiment.

## Reproducibility

- Synthesized-waveform loss weight: `lambda_synth = 0.5`
- Split assignment SHA-256: `{summary['split_assignment_sha256']}`
- Selected checkpoint SHA-256: `{selection['selected_checkpoint_sha256']}`
- Test policy: evaluate the validation-selected seed once
- Large NPZ and checkpoint files are intentionally not committed

Published evidence:

- [Machine-readable summary](summary.json)
- [Publication manifest](publication_manifest.json)
- [Selection record](selection.json)
- [Formal protocol](protocol.json)

Reproduction code:

- [Fixed-split runner](../../../scripts/experiments/run_phase39_expanded_fixed_split.py)
- [Plotting and publication script](../../../scripts/plotting/plot_phase39_expanded_fixed_split.py)
- [Expanded-data configuration](../../../configs/experiments/phase39_expanded_grouped_cv.yaml)

## 中文摘要

本实验固定使用 24 个训练事件、6 个验证事件和 9 个独立测试事件。三个随机
种子只根据固定验证集选择，最终 seed 73 的验证事件 MAE 为
`{selection['selected_validation_event_mae_mw']:.6f} Mw`。选模结束后只评估一次测试集，
九事件测试 MAE 为 `{test['event_mae']:.6f} Mw`，450 个台站的 MAE 为
`{test['station_mae']:.6f} Mw`。

测试误差主要由 Napa2014 和 `ak014cbigci8` 拉高，因此目前不能用验证集的
`0.1713 Mw` 代替独立测试结果。正式测试结论仍然是九事件 MAE
`{test['event_mae']:.6f} Mw`。

本页主图已经恢复为前一版论文式图鉴：事件散点、台站密度与事件中位数、
事件绝对误差、训练曲线、事件地图以及代表性波形/PGD。数据划分和随机种子
选择图保留为补充诊断图。所有震级结果均为固定 200 秒输入后的终点估计，
不是逐秒因果预测。

新增的经验公式比较严格使用同一批 9 个测试事件、450 个台站和 200 秒窗口。
Crowell 2013、Ruhl 2019 和 Melgar 2015 使用三分量 PGD 与震源距，Phase 39
仍使用径向波形。本测试集上 Crowell 的事件 MAE 为
`{pgd_methods['crowell']['event_mae_mw']:.6f} Mw`，低于 Phase 39 的
`{pgd_methods['phase39']['event_mae_mw']:.6f} Mw`；该结果在图表中按原值报告。
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def copy_public_artifacts(run_root: Path, output_dir: Path) -> None:
    mapping = {
        "fixed_split_manifest.json": "fixed_split_manifest.json",
        "protocol.json": "protocol.json",
        "selection.json": "selection.json",
        "selected_test_event_predictions.csv": "selected_test_event_predictions.csv",
        "selected_test_station_predictions.csv": "selected_test_station_predictions.csv",
    }
    for source_name, destination_name in mapping.items():
        source = run_root / source_name
        destination = output_dir / destination_name
        if source.suffix == ".csv":
            destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            shutil.copy2(source, destination)


def write_manifest(output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "publication_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "status": "complete",
        "created_at_utc": utc_now_iso(),
        "split_assignment_sha256": summary["split_assignment_sha256"],
        "selected_checkpoint_sha256": summary["selection"][
            "selected_checkpoint_sha256"
        ],
        "file_count": len(files),
        "files": files,
    }
    write_json(output_dir / "publication_manifest.json", manifest)
    return manifest


def generate_publication(run_root: Path, output_dir: Path) -> dict[str, Any]:
    (
        campaign,
        split,
        protocol,
        events,
        stations,
        candidates,
        logs,
    ) = load_inputs(run_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures/en"
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = output_dir / "analysis"
    training_logs_dir = analysis_dir / "training_logs"
    training_logs_dir.mkdir(parents=True, exist_ok=True)

    candidates.to_csv(analysis_dir / "seed_selection.csv", index=False)
    events.sort_values("event").to_csv(
        analysis_dir / "test_event_error_analysis.csv", index=False
    )
    event_station_summary(stations, events).sort_values("event").to_csv(
        analysis_dir / "test_event_station_summary.csv", index=False
    )
    for seed, log in logs.items():
        log.to_csv(training_logs_dir / f"seed_{seed}.csv", index=False)

    selected_seed = int(campaign["selection"]["selected_seed"])
    selected = candidates[candidates["seed"] == selected_seed].iloc[0]
    data_path = Path(str(protocol["dataset_path"]))
    config_path = Path(str(protocol["config_path"]))
    samples = load_phase39_samples(config_path)
    pgd_stations, pgd_events, pgd_summary = compute_pgd_baselines(
        samples, stations, events
    )
    pgd_stations.to_csv(analysis_dir / "pgd_station_predictions.csv", index=False)
    pgd_events.to_csv(analysis_dir / "pgd_event_predictions.csv", index=False)
    pgd_summary.to_csv(analysis_dir / "pgd_method_summary.csv", index=False)
    waveform_sample, waveform_station = load_waveform_sample(samples, stations)
    generated = []
    generated.extend(
        plot_test_event_scatter(events, figures_dir / "01_test_event_scatter")
    )
    generated.extend(
        plot_station_predictions(
            stations,
            events,
            figures_dir / "02_test_station_scatter",
        )
    )
    generated.extend(
        plot_event_errors(events, figures_dir / "03_test_event_absolute_errors")
    )
    generated.extend(
        plot_selected_training(
            logs[selected_seed],
            int(selected["best_epoch"]),
            float(selected["best_validation_event_mae_mw"]),
            figures_dir / "04_selected_seed_training_curves",
        )
    )
    generated.extend(
        plot_selected_maps(
            stations,
            data_path,
            figures_dir / "05_selected_test_event_maps",
        )
    )
    generated.extend(
        plot_waveform_example(
            waveform_sample,
            waveform_station,
            stations,
            events,
            figures_dir / "06_ridgecrest_waveform_and_predictions",
        )
    )
    generated.extend(
        plot_pgd_method_scatter(
            events,
            pgd_events,
            pgd_summary,
            figures_dir / "07_phase39_vs_pgd_method_scatter",
        )
    )
    generated.extend(
        plot_pgd_method_errors(
            events,
            pgd_events,
            pgd_summary,
            figures_dir / "08_phase39_vs_pgd_method_errors",
        )
    )
    generated.extend(
        plot_split_overview(split, figures_dir / "09_split_overview_supplement")
    )
    generated.extend(
        plot_seed_selection(
            candidates,
            logs,
            selected_seed,
            figures_dir / "10_seed_validation_selection_supplement",
        )
    )
    if len(generated) != 20:
        raise ValueError(f"unexpected generated figure count: {len(generated)}")

    summary = build_public_summary(
        campaign, split, protocol, events, candidates, pgd_summary
    )
    write_json(output_dir / "summary.json", summary)
    copy_public_artifacts(run_root, output_dir)
    write_readme(output_dir, summary)
    manifest = write_manifest(output_dir, summary)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot and publish the Phase 39 expanded fixed-split result."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_matplotlib()
    manifest = generate_publication(
        args.run_root.resolve(), args.output_dir.resolve()
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
