"""Publication figures for the Phase 39 released-moment (Direction 1) validation report.

Inputs are frozen validation replays produced by
``scripts/evaluation/evaluate_phase39_causal_released_moment.py``.  The script
reads only replay artefacts (no model inference), writes figures under
``docs/results/phase39-causal-released-moment/figures`` and copies the
per-event objective tables into ``analysis/``.

Plot style (fonts, panel lettering, shared axes, PNG+PDF export) is reused from
``plot_phase39_causal_event_trajectories.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plotting.plot_phase39_causal_event_trajectories import (  # noqa: E402
    configure_matplotlib,
    load_json,
)
from src.utils.provenance import sha256_file  # noqa: E402


RUNS_ROOT = Path("/home/lihe/PINN_Mag/runs")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs/results/phase39-causal-released-moment"
EXPECTED_SPLIT_SHA256 = (
    "e4807aa1e6b5b389caf23974f62ff9da6b8add7f7887ec23a59d5d35a455eba7"
)

# Ordered variants: key -> (replay dir, short label, expected checkpoint sha256).
VARIANTS: dict[str, dict[str, Any]] = {
    "old": {
        "replay": RUNS_ROOT / "phase39-causal-4a25-released-replay-validation-20260907-v2",
        "label": "Causal Phase 39 (final-Mw target)",
        "short": "OLD",
        "checkpoint_sha256": "4a2540241fcbd9182edd05fb5fdb24641b1c836fa1ccd2ad12b101f91fa3e1ab",
        "color": "#7F7F7F",
        "ls": "-",
    },
    "new1": {
        "replay": RUNS_ROOT
        / "phase39-causal-released-moment-seed73-20260907-v1-replay-validation-v2",
        "label": "Released-moment target",
        "short": "NEW-1",
        "checkpoint_sha256": "2206b94b54074609",
        "color": "#F28E2B",
        "ls": "-",
    },
    "new2": {
        "replay": RUNS_ROOT
        / "phase39-causal-released-moment-monotone-seed73-20260907-v1-replay-validation",
        "label": "+ monotone pair term",
        "short": "NEW-2",
        "checkpoint_sha256": "662714db28cd14e7",
        "color": "#1F9E89",
        "ls": "-",
    },
    "new3": {
        "replay": RUNS_ROOT
        / "phase39-causal-released-moment-scale1p0-seed73-20260907-v1-replay-validation",
        "label": "+ scaling floor -1.0 Mw",
        "short": "NEW-3",
        "checkpoint_sha256": "56d55b1ad65388ae",
        "color": "#D1495B",
        "ls": "-",
    },
}
PRIMARY = "new3"
EVENTS = (
    "Anchorage2018",
    "Maule2010",
    "Noto2024",
    "Parkfield2004",
    "RatIslands2014",
    "SandPoint2020",
)
PGD_METHODS = ("crowell", "ruhl", "melgar")
PGD_LABELS = {"crowell": "Crowell 2013", "ruhl": "Ruhl 2019", "melgar": "Melgar 2015"}
PGD_COLORS = {"crowell": "#4C78A8", "ruhl": "#59A14F", "melgar": "#B279A2"}
PGD_STYLES = {"crowell": "-", "ruhl": "--", "melgar": "-."}
B_STAR = "released_constrained"
B_ALL = "released"
A_FINAL = "final"
B_REF_STAR = "released_ref_constrained"
GATES = (
    ("event_mae_200s", "200 s event MAE < 0.1270 (causal P39)", lambda v: v < 0.127016),
    ("events_overshooting_catalog_at_1s", "no event above catalog at 1 s (A)", lambda v: v == 0),
    ("mean_sign_changes", "mean sign changes (B*) < 10", lambda v: v < 10),
    ("entry_before_crowell", "stable entry before Crowell on >= 4/6 events", lambda v: v >= 4),
    ("spearman_vs_catalog_030s", "rho(B*, catalog) at 30 s >= 0.94 (old A)", lambda v: v >= 0.94),
)


def load_variant(key: str) -> dict[str, Any]:
    spec = VARIANTS[key]
    root = Path(spec["replay"])
    summary = load_json(root / "summary.json")
    if summary.get("status") != "complete":
        raise ValueError(f"replay incomplete: {root}")
    if summary.get("split_assignment_sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError(f"split assignment changed: {root}")
    if not str(summary.get("checkpoint_sha256", "")).startswith(spec["checkpoint_sha256"]):
        raise ValueError(f"checkpoint changed: {root}")
    if summary.get("cohort") != "validation":
        raise ValueError(f"only validation replays are allowed here: {root}")
    events = pd.read_csv(root / "validation_event_predictions.csv")
    horizon = pd.read_csv(root / "validation_horizon_metrics.csv")
    objective = pd.read_csv(root / "validation_objective_per_event.csv")
    if set(events["event"]) != set(EVENTS):
        raise ValueError(f"event coverage changed: {root}")
    return {
        "key": key,
        "summary": summary,
        "events": events,
        "horizon": horizon,
        "objective": objective,
        "root": root,
        **spec,
    }


def _curve(frame: pd.DataFrame, method: str, event: str) -> pd.DataFrame:
    rows = frame[(frame["method"] == method) & (frame["event"] == event)]
    return rows.sort_values("observation_horizon_sec")


def _save(figure: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for suffix, dpi in ((".png", 220), (".pdf", 300)):
        path = stem.with_suffix(suffix)
        meta = {"CreationDate": None, "ModDate": None} if suffix == ".pdf" else None
        figure.savefig(path, dpi=dpi, bbox_inches="tight", metadata=meta)
        out.append(path)
    plt.close(figure)
    return out


def _grid(title: str) -> tuple[plt.Figure, np.ndarray]:
    figure, axes = plt.subplots(3, 2, figsize=(15.8, 13.2), sharex=True, sharey=True)
    figure.subplots_adjust(
        left=0.075, right=0.985, bottom=0.065, top=0.885, hspace=0.17, wspace=0.045
    )
    figure.suptitle(title, fontsize=14, fontweight="bold", y=0.985)
    return figure, axes


def _finish_grid(
    figure: plt.Figure,
    axes: np.ndarray,
    handles: list,
    labels: list[str],
    ylim: tuple[float, float],
) -> None:
    for index, axis in enumerate(axes.flat):
        axis.set_xlim(1, 200)
        axis.set_ylim(*ylim)
        axis.set_xticks([1, 50, 100, 150, 200])
        axis.grid(True, color="#D7DCE2", linewidth=0.55, alpha=0.75)
        if index // 2 == 2:
            axis.set_xlabel("Observed causal prefix (s since origin)")
        if index % 2 == 0:
            axis.set_ylabel("Event-median magnitude (Mw)")
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=min(len(labels), 5),
        frameon=False,
        fontsize=9.5,
    )


def figure_event_trajectories(primary: dict[str, Any], old: dict[str, Any], stem: Path) -> list[Path]:
    """Fig 1: primary model B*/A vs SCARDEC released label vs PGD, per event."""
    figure, axes = _grid(
        "Event-wise causal magnitude evolution: released-moment model, SCARDEC label and PGD"
    )
    events = primary["events"]
    handles: list = []
    labels: list[str] = []
    for index, (axis, event) in enumerate(zip(axes.flat, EVENTS)):
        catalog = float(events.loc[events["event"] == event, "mw_catalog"].median())
        endpoint = _curve(events, B_STAR, event)
        n_st = int(endpoint.iloc[-1]["n_stations"])
        series = [
            (B_STAR, events, dict(color=primary["color"], lw=2.3, ls="-", zorder=5),
             "B* released Mw, NEW-3 (P-arrived stations)"),
            (A_FINAL, events, dict(color=primary["color"], lw=1.3, ls="--", alpha=0.85, zorder=4),
             "A final Mw, NEW-3"),
            (A_FINAL, old["events"], dict(color=old["color"], lw=1.4, ls="-", alpha=0.9, zorder=3),
             "A final Mw, causal Phase 39 (old)"),
            (B_REF_STAR, events, dict(color="#2CA02C", lw=1.5, ls=":", zorder=4),
             "B* released Mw from SCARDEC label"),
        ]
        for method, frame, style, label in series:
            rows = _curve(frame, method, event)
            line = axis.plot(rows["observation_horizon_sec"], rows["mw_pred_median"], **style)[0]
            if index == 0:
                handles.append(line)
                labels.append(label)
        for method in PGD_METHODS:
            rows = _curve(events, method, event)
            line = axis.plot(
                rows["observation_horizon_sec"],
                rows["mw_pred_median"],
                color=PGD_COLORS[method],
                ls=PGD_STYLES[method],
                lw=1.55,
                alpha=0.9,
                zorder=3,
            )[0]
            if index == 0:
                handles.append(line)
                labels.append(f"{PGD_LABELS[method]} PGD")
        cat = axis.axhline(catalog, color="#202124", lw=1.2, ls=":", zorder=2)
        axis.axhspan(catalog - 0.3, catalog + 0.3, color="#202124", alpha=0.05, lw=0)
        if index == 0:
            handles.append(cat)
            labels.append("Catalog Mw (band: +/-0.3)")
        axis.set_title(
            f"{'ABCDEF'[index]}. {event} | catalog Mw {catalog:.2f} | {n_st} stations",
            loc="left",
            fontweight="bold",
            fontsize=10.5,
        )
    _finish_grid(figure, axes, handles, labels, (4.6, 9.4))
    return _save(figure, stem)


def figure_variant_comparison(variants: dict[str, dict[str, Any]], stem: Path) -> list[Path]:
    """Fig 2: B* of the four checkpoints per event, Crowell as reference."""
    figure, axes = _grid("Released magnitude B* across the four checkpoints (validation events)")
    handles: list = []
    labels: list[str] = []
    for index, (axis, event) in enumerate(zip(axes.flat, EVENTS)):
        ref = variants[PRIMARY]["events"]
        catalog = float(ref.loc[ref["event"] == event, "mw_catalog"].median())
        for key in ("old", "new1", "new2", "new3"):
            var = variants[key]
            rows = _curve(var["events"], B_STAR, event)
            line = axis.plot(
                rows["observation_horizon_sec"],
                rows["mw_pred_median"],
                color=var["color"],
                lw=2.3 if key == PRIMARY else 1.4,
                alpha=1.0 if key == PRIMARY else 0.85,
                zorder=5 if key == PRIMARY else 3,
            )[0]
            if index == 0:
                handles.append(line)
                labels.append(f"{var['short']}: {var['label']}")
        rows = _curve(ref, "crowell", event)
        line = axis.plot(
            rows["observation_horizon_sec"], rows["mw_pred_median"],
            color=PGD_COLORS["crowell"], lw=1.55, zorder=3,
        )[0]
        cat = axis.axhline(catalog, color="#202124", lw=1.2, ls=":", zorder=2)
        if index == 0:
            handles += [line, cat]
            labels += ["Crowell 2013 PGD", "Catalog Mw"]
        axis.set_title(
            f"{'ABCDEF'[index]}. {event} | catalog Mw {catalog:.2f}",
            loc="left", fontweight="bold", fontsize=10.5,
        )
    _finish_grid(figure, axes, handles, labels, (4.6, 9.4))
    return _save(figure, stem)


def figure_horizon_metrics(variants: dict[str, dict[str, Any]], table: pd.DataFrame, stem: Path) -> list[Path]:
    """Fig 3: (A) event MAE vs horizon, (B) objective metrics per variant."""
    configure_matplotlib()
    figure = plt.figure(figsize=(16.0, 5.4))
    grid = figure.add_gridspec(1, 5, width_ratios=[2.3, 0.6, 0.6, 0.6, 0.6], wspace=0.42,
                               left=0.05, right=0.99, bottom=0.14, top=0.86)
    ax = figure.add_subplot(grid[0, 0])
    for key in ("old", "new1", "new2", "new3"):
        var = variants[key]
        rows = var["horizon"]
        rows = rows[(rows["method"] == B_STAR) & (rows["reference"] == "catalog")].sort_values(
            "observation_horizon_sec"
        )
        ax.plot(
            rows["observation_horizon_sec"], rows["event_mae"],
            color=var["color"], lw=2.3 if key == PRIMARY else 1.4,
            label=f"{var['short']}: {var['label']} (B*)",
        )
    rows = variants[PRIMARY]["horizon"]
    for method in PGD_METHODS:
        pgd = rows[(rows["method"] == method) & (rows["reference"] == "catalog")].sort_values(
            "observation_horizon_sec"
        )
        ax.plot(
            pgd["observation_horizon_sec"], pgd["event_mae"],
            color=PGD_COLORS[method], ls=PGD_STYLES[method], lw=1.4, label=f"{PGD_LABELS[method]} PGD",
        )
    ax.axhline(0.171305, color="#202124", lw=0.9, ls=":", label="Endpoint Phase 39 gate 0.1713")
    ax.set_xlim(1, 200)
    ax.set_ylim(0.0, 1.6)
    ax.set_xlabel("Observed causal prefix (s since origin)")
    ax.set_ylabel("Event MAE vs catalog (Mw)")
    ax.set_title("A. Event MAE against horizon (6 validation events)", loc="left", fontweight="bold")
    ax.grid(True, color="#D7DCE2", linewidth=0.55, alpha=0.75)
    ax.legend(frameon=False, fontsize=8.3, ncol=2, loc="upper right")

    metrics = [
        ("event_mae_200s", "200 s event MAE (Mw)", "{:.3f}", 0.127016, "gate < 0.127"),
        ("mean_sign_changes", "mean sign changes", "{:.1f}", 10.0, "gate < 10"),
        ("median_stable_entry_sec", "median stable entry (s)", "{:.0f}", None, None),
        ("spearman_vs_catalog_030s", "Spearman rho @ 30 s", "{:.2f}", 0.94, "gate >= 0.94"),
    ]
    keys = ("old", "new1", "new2", "new3")
    crow = table[(table["variant"] == "NEW-3") & (table["method"] == "crowell")].iloc[0]
    for column, (metric, label, fmt, gate, gate_label) in enumerate(metrics, start=1):
        ax = figure.add_subplot(grid[0, column])
        values = []
        for offset, key in enumerate(keys):
            row = table[(table["variant"] == variants[key]["short"]) & (table["method"] == B_STAR)].iloc[0]
            value = float(row[metric])
            values.append(value)
            ax.bar(offset, value, 0.72, color=variants[key]["color"])
            ax.text(offset, value, fmt.format(value), ha="center", va="bottom", fontsize=7.8)
        crow_value = float(crow[metric]) if pd.notna(crow[metric]) else np.nan
        if np.isfinite(crow_value):
            ax.axhline(crow_value, color=PGD_COLORS["crowell"], lw=1.4, ls="-",
                       label=f"Crowell {fmt.format(crow_value)}")
        if gate is not None:
            ax.axhline(gate, color="#202124", lw=0.9, ls=":", label=gate_label)
        top = max(values + ([crow_value] if np.isfinite(crow_value) else []) + ([gate] if gate else []))
        ax.set_ylim(0, top * 1.28)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([variants[k]["short"] for k in keys], fontsize=8, rotation=35, ha="right")
        ax.set_title(f"{'BCDE'[column - 1]}. {label}", loc="left", fontweight="bold", fontsize=9.5)
        ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.55, alpha=0.75)
        ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    return _save(figure, stem)


def figure_released_moment_illustration(primary: dict[str, Any], stem: Path, event: str = "Maule2010") -> list[Path]:
    """Fig 4: what the released-moment target means, on one near-source station."""
    configure_matplotlib()
    root = primary["root"]
    npz = np.load(root / "validation_prefix_stf.npz")
    stations = pd.read_csv(root / "validation_anchor_station_predictions.csv")
    anchor = stations[
        (stations["method"] == B_ALL)
        & (stations["event"] == event)
        & (stations["observation_horizon_sec"] == 30)
    ].sort_values("constrained_window_sec", ascending=False)
    station = str(anchor.iloc[0]["station"])
    tau = 30.0 - float(anchor.iloc[0]["constrained_window_sec"])
    catalog = float(anchor.iloc[0]["mw_catalog"])
    native = float(anchor.iloc[0]["mw_stf_native"])
    idx = int(np.flatnonzero((npz["events"] == event) & (npz["stations"] == station))[0])
    rate = npz["stf_over_m_ref"][idx].astype(np.float64) * float(npz["m_ref_nm"])  # (200 h, 200 t)
    horizons = npz["horizons_sec"].astype(int)
    t = np.arange(rate.shape[1]) + 0.5
    floor = 1.0e15

    def mag(moment: np.ndarray) -> np.ndarray:
        return (2.0 / 3.0) * (np.log10(np.maximum(moment, floor)) - 9.1)

    window = np.clip(horizons - tau, 0.0, 200.0)
    mask = np.clip(window[:, None] - np.arange(rate.shape[1])[None, :], 0.0, 1.0)
    b_curve = mag((rate * mask).sum(axis=1))
    a_curve = mag(rate.sum(axis=1))

    figure, axes = plt.subplots(1, 2, figsize=(15.0, 5.2), gridspec_kw={"width_ratios": [1.25, 1.0]})
    figure.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.84, wspace=0.2)
    ax = axes[0]
    show = (20, 40, 60, 90, 120, 200)
    cmap = plt.get_cmap("viridis")
    for k, h in enumerate(show):
        color = cmap(k / (len(show) - 1))
        ax.plot(t, rate[h - 1] / 1e18, color=color, lw=1.6, label=f"h = {h} s  (window {max(h - tau, 0):.0f} s)")
        ax.axvline(max(h - tau, 0), color=color, lw=0.9, ls=":")
    ax.set_xlim(0, 200)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Source time since origin (s)")
    ax.set_ylabel("Predicted moment rate (1e18 N m / s)")
    ax.set_title(
        f"A. {event} station {station}: predicted STF at six prefixes (P arrival {tau:.1f} s)",
        loc="left", fontweight="bold",
    )
    ax.grid(True, color="#D7DCE2", linewidth=0.55, alpha=0.75)
    ax.legend(frameon=False, fontsize=8.5, title="dotted line = end of constrained window h - tau_P", title_fontsize=8.5)

    ax = axes[1]
    ax.plot(horizons, b_curve, color=primary["color"], lw=2.3, label="B(h): integral of STF over [0, h - tau_P]")
    ax.plot(horizons, a_curve, color=primary["color"], lw=1.3, ls="--", label="A(h): integral over the full 200 s")
    ax.axhline(catalog, color="#202124", lw=1.2, ls=":", label=f"Catalog Mw {catalog:.2f}")
    ax.axhline(native, color="#2CA02C", lw=1.0, ls="-.", label=f"SCARDEC 200 s Mw {native:.2f}")
    ax.axvline(tau, color="#4C78A8", lw=1.0, ls=":")
    ax.text(tau + 2, 4.3, "P arrival", color="#4C78A8", fontsize=8.5)
    ax.set_xlim(1, 200)
    ax.set_ylim(3.8, 9.4)
    ax.set_xlabel("Observed causal prefix h (s since origin)")
    ax.set_ylabel("Magnitude (Mw)")
    ax.set_title("B. Released (B) and final (A) magnitude for the same station", loc="left", fontweight="bold")
    ax.grid(True, color="#D7DCE2", linewidth=0.55, alpha=0.75)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    figure.suptitle("What the released-moment target supervises at each prefix", fontsize=14, fontweight="bold", y=0.975)
    return _save(figure, stem)


def build_tables(variants: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    gate_rows = []
    per_event = []
    crowell_entry = None
    for key in ("old", "new1", "new2", "new3"):
        var = variants[key]
        obj = var["summary"]["objective_by_method"]
        pe = var["objective"].copy()
        pe.insert(0, "variant", var["short"])
        per_event.append(pe)
        if crowell_entry is None:
            crowell_entry = pe[pe["method"] == "crowell"].set_index("event")["stable_entry_pm0p3_sec"]
        star = pe[pe["method"] == B_STAR].set_index("event")["stable_entry_pm0p3_sec"]
        before = 0
        for event in EVENTS:
            model_t = star.get(event, np.nan)
            crow_t = crowell_entry.get(event, np.nan)
            if np.isfinite(model_t) and (not np.isfinite(crow_t) or model_t < crow_t):
                before += 1
        for method in (B_STAR, A_FINAL, "crowell", "ruhl", "melgar"):
            entry = dict(variant=var["short"], method=method, checkpoint=var["checkpoint_sha256"][:8])
            entry.update({k: obj[method].get(k) for k in (
                "event_mae_200s", "event_rmse_200s", "event_bias_200s", "mean_mw_1s",
                "events_overshooting_catalog_at_1s", "mean_sign_changes", "mean_abs_step_mw_per_sec",
                "mean_post120_range_mw", "events_with_stable_entry", "median_stable_entry_sec",
                "spearman_vs_catalog_030s", "spearman_vs_catalog_060s",
            )})
            rows.append(entry)
        values = {
            "event_mae_200s": obj[B_STAR]["event_mae_200s"],
            "events_overshooting_catalog_at_1s": obj[A_FINAL]["events_overshooting_catalog_at_1s"],
            "mean_sign_changes": obj[B_STAR]["mean_sign_changes"],
            "entry_before_crowell": before,
            "spearman_vs_catalog_030s": obj[B_STAR]["spearman_vs_catalog_030s"],
        }
        for metric, label, test in GATES:
            value = values[metric]
            gate_rows.append(dict(variant=var["short"], gate=label, value=value,
                                  passed=bool(value is not None and test(float(value)))))
    return pd.DataFrame(rows), pd.DataFrame(gate_rows), pd.concat(per_event, ignore_index=True)


# ----------------------------------------------------------------- cohort / scatter figures

SPLIT_MANIFEST = PROJECT_ROOT / "docs/results/phase39-expanded-fixed-split/fixed_split_manifest.json"
TRAIN_REPLAYS = {
    "new3": RUNS_ROOT / "phase39-causal-released-moment-scale1p0-seed73-20260907-v1-replay-train",
    "old": RUNS_ROOT / "phase39-causal-4a25-released-replay-train-20260907",
}
COHORT_LABEL = {"train": "Training (24 events)", "validation": "Validation (6 events)"}
ANCHORS = (30, 60, 90, 120, 160, 200)


def load_cohort_replay(root: Path, cohort: str, checkpoint_prefix: str) -> dict[str, Any]:
    summary = load_json(root / "summary.json")
    if summary.get("status") != "complete" or summary.get("cohort") != cohort:
        raise ValueError(f"unexpected replay: {root}")
    if summary.get("split_assignment_sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError(f"split assignment changed: {root}")
    if not str(summary["checkpoint_sha256"]).startswith(checkpoint_prefix):
        raise ValueError(f"checkpoint changed: {root}")
    return {
        "summary": summary,
        "events": pd.read_csv(root / f"{cohort}_event_predictions.csv"),
        "stations": pd.read_csv(root / f"{cohort}_anchor_station_predictions.csv"),
        "root": root,
    }


def endpoint_events(replay: dict[str, Any], method: str = A_FINAL) -> pd.DataFrame:
    frame = replay["events"]
    rows = frame[(frame["method"] == method) & (frame["observation_horizon_sec"] == 200)].copy()
    rows["absolute_error"] = rows["error_vs_catalog"].abs()
    return rows.sort_values("mw_catalog").reset_index(drop=True)


def endpoint_stations(replay: dict[str, Any], method: str = A_FINAL) -> pd.DataFrame:
    frame = replay["stations"]
    return frame[(frame["method"] == method) & (frame["observation_horizon_sec"] == 200)].copy()


def _identity_axis(axis: plt.Axes, lower: float, upper: float, band: float = 0.2) -> None:
    line = np.linspace(lower, upper, 200)
    axis.fill_between(line, line - band, line + band, color="#E9A23B", alpha=0.13, lw=0)
    axis.plot(line, line, color="#202124", lw=1.2)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, color="#D7DCE2", linewidth=0.6)
    axis.set_xlabel("Catalog magnitude (Mw)")
    axis.set_ylabel("Estimated magnitude at 200 s (Mw)")


def figure_split_overview(stem: Path) -> tuple[list[Path], pd.DataFrame]:
    from scripts.plotting.plot_phase39_expanded_fixed_split import plot_split_overview, split_event_frame

    split = load_json(SPLIT_MANIFEST)
    configure_matplotlib()
    paths = plot_split_overview(split, stem)
    return paths, split_event_frame(split)


def figure_endpoint_event_scatter(cohorts: dict[str, dict[str, dict[str, Any]]], stem: Path) -> list[Path]:
    """Two panels (train / validation): 200 s event medians of NEW-3 (filled) and OLD (hollow)."""
    configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 7.4))
    figure.subplots_adjust(left=0.06, right=0.99, bottom=0.09, top=0.88, wspace=0.18)
    for axis, cohort, letter in zip(axes, ("train", "validation"), "AB"):
        new = endpoint_events(cohorts[cohort]["new3"])
        old = endpoint_events(cohorts[cohort]["old"])
        lower = float(min(new["mw_catalog"].min(), new["mw_pred_median"].min(), old["mw_pred_median"].min()) - 0.25)
        upper = float(max(new["mw_catalog"].max(), new["mw_pred_median"].max(), old["mw_pred_median"].max()) + 0.25)
        _identity_axis(axis, lower, upper)
        sizes = 50 + 170 * np.sqrt(new["n_stations"] / new["n_stations"].max())
        axis.scatter(old["mw_catalog"], old["mw_pred_median"], s=sizes, facecolor="none",
                     edgecolor="#7F7F7F", lw=1.3, label="Causal Phase 39 (old, 4a25)", zorder=3)
        axis.scatter(new["mw_catalog"], new["mw_pred_median"], s=sizes, color=VARIANTS[PRIMARY]["color"],
                     edgecolor="white", lw=0.8, alpha=0.92, label="Released-moment NEW-3 (56d5)", zorder=4)
        for row in new.itertuples(index=False):
            if cohort == "validation" or row.absolute_error > 0.15:
                axis.annotate(str(row.event), (float(row.mw_catalog), float(row.mw_pred_median)),
                              xytext=(6, 5), textcoords="offset points", fontsize=8)
        n = len(new)
        axis.text(0.03, 0.97,
                  f"NEW-3: event MAE {new['absolute_error'].mean():.3f} Mw, within +/-0.20 = {(new['absolute_error'] <= 0.2).sum()}/{n}\n"
                  f"OLD:   event MAE {old['absolute_error'].mean():.3f} Mw, within +/-0.20 = {(old['absolute_error'] <= 0.2).sum()}/{n}",
                  transform=axis.transAxes, ha="left", va="top", fontsize=9.5,
                  bbox={"facecolor": "white", "edgecolor": "#D7DCE2", "alpha": 0.92})
        axis.set_title(f"{letter}. {COHORT_LABEL[cohort]} | marker size = station count", loc="left", fontweight="bold")
        axis.legend(frameon=False, loc="lower right", fontsize=9)
    figure.suptitle("Endpoint (200 s) event-median magnitude on the training and validation cohorts",
                    fontsize=14, fontweight="bold", y=0.965)
    return _save(figure, stem)


def figure_endpoint_station_scatter(cohorts: dict[str, dict[str, dict[str, Any]]], stem: Path) -> list[Path]:
    """2 x 2: hexbin density of all station estimates and station cloud with event medians (NEW-3)."""
    configure_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 12.0))
    figure.subplots_adjust(left=0.06, right=0.97, bottom=0.06, top=0.92, wspace=0.28, hspace=0.24)
    for row_axes, cohort, letters in zip(axes, ("train", "validation"), ("AB", "CD")):
        stations = endpoint_stations(cohorts[cohort]["new3"])
        events = endpoint_events(cohorts[cohort]["new3"])
        lower = float(min(stations["mw_catalog"].min(), stations["mw_pred"].quantile(0.005)) - 0.3)
        upper = float(max(stations["mw_catalog"].max(), stations["mw_pred"].quantile(0.995)) + 0.3)
        for axis in row_axes:
            axis.plot([lower, upper], [lower, upper], color="#202124", lw=1.1)
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_xlabel("Catalog magnitude (Mw)")
            axis.set_ylabel("Station estimate at 200 s (Mw)")
            axis.grid(True, color="#D7DCE2", linewidth=0.55)
        density = row_axes[0].hexbin(stations["mw_catalog"], stations["mw_pred"], gridsize=42, mincnt=1,
                                     cmap="viridis", linewidths=0.0, extent=(lower, upper, lower, upper))
        figure.colorbar(density, ax=row_axes[0], pad=0.02, label="Station count")
        mae = float((stations["mw_pred"] - stations["mw_catalog"]).abs().mean())
        row_axes[0].set_title(f"{letters[0]}. {COHORT_LABEL[cohort]}: all {len(stations)} station estimates "
                              f"(station MAE {mae:.3f})", loc="left", fontweight="bold", fontsize=10.5)
        row_axes[1].scatter(stations["mw_catalog"], stations["mw_pred"], s=9, color="#6C757D", alpha=0.18,
                            linewidth=0, rasterized=True, label="Stations")
        row_axes[1].scatter(events["mw_catalog"], events["mw_pred_median"], s=58, color=VARIANTS[PRIMARY]["color"],
                            edgecolor="white", lw=0.7, zorder=3, label="Event median (NEW-3)")
        row_axes[1].set_title(f"{letters[1]}. {COHORT_LABEL[cohort]}: station cloud with event medians",
                              loc="left", fontweight="bold", fontsize=10.5)
        row_axes[1].legend(frameon=False, loc="lower right", fontsize=8.5)
    figure.suptitle("Station-level endpoint estimates of the released-moment model (NEW-3)",
                    fontsize=14, fontweight="bold", y=0.965)
    return _save(figure, stem)


def figure_validation_vs_pgd_scatter(validation: dict[str, Any], stem: Path) -> list[Path]:
    """NEW-3 and the three PGD relations on the same six validation events."""
    configure_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 11.0))
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.90, wspace=0.22, hspace=0.26)
    panels = (
        (A_FINAL, "Released-moment model (NEW-3)", VARIANTS[PRIMARY]["color"]),
        ("crowell", "Crowell 2013 PGD", PGD_COLORS["crowell"]),
        ("ruhl", "Ruhl 2019 PGD", PGD_COLORS["ruhl"]),
        ("melgar", "Melgar 2015 PGD", PGD_COLORS["melgar"]),
    )
    all_rows = pd.concat([endpoint_events(validation, m) for m, _, _ in panels])
    lower = float(min(all_rows["mw_catalog"].min(), all_rows["mw_pred_median"].min()) - 0.25)
    upper = float(max(all_rows["mw_catalog"].max(), all_rows["mw_pred_median"].max()) + 0.25)
    for axis, (method, label, color), letter in zip(axes.flat, panels, "ABCD"):
        rows = endpoint_events(validation, method)
        _identity_axis(axis, lower, upper)
        sizes = 50 + 170 * np.sqrt(rows["n_stations"] / rows["n_stations"].max())
        axis.scatter(rows["mw_catalog"], rows["mw_pred_median"], s=sizes, color=color, edgecolor="white",
                     lw=0.8, alpha=0.92, zorder=3)
        for row in rows.itertuples(index=False):
            axis.annotate(str(row.event), (float(row.mw_catalog), float(row.mw_pred_median)),
                          xytext=(6, 5), textcoords="offset points", fontsize=8)
        axis.text(0.03, 0.97,
                  f"Event MAE = {rows['absolute_error'].mean():.3f} Mw\n"
                  f"Within +/-0.20 = {(rows['absolute_error'] <= 0.2).sum()}/{len(rows)}",
                  transform=axis.transAxes, ha="left", va="top", fontsize=9.5,
                  bbox={"facecolor": "white", "edgecolor": "#D7DCE2", "alpha": 0.92})
        axis.set_title(f"{letter}. {label}", loc="left", fontweight="bold")
    figure.suptitle("Released-moment model versus three empirical PGD relations\n"
                    "Same six validation events and 446 accepted stations, 200 s endpoint",
                    fontsize=13.5, fontweight="bold", y=0.975)
    return _save(figure, stem)


def figure_event_errors(cohorts: dict[str, dict[str, dict[str, Any]]], stem: Path) -> list[Path]:
    """Per-event absolute endpoint error: NEW-3 vs OLD (vs Crowell on validation)."""
    configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 8.2), gridspec_kw={"width_ratios": [1.6, 1.0]})
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.88, wspace=0.42)
    for axis, cohort, letter in zip(axes, ("train", "validation"), "AB"):
        new = endpoint_events(cohorts[cohort]["new3"]).set_index("event")
        old = endpoint_events(cohorts[cohort]["old"]).set_index("event")
        order = new["absolute_error"].sort_values().index
        series = [("NEW-3 released-moment", new.loc[order, "absolute_error"], VARIANTS[PRIMARY]["color"]),
                  ("OLD causal Phase 39", old.loc[order, "absolute_error"], "#7F7F7F")]
        if cohort == "validation":
            crow = endpoint_events(cohorts[cohort]["new3"], "crowell").set_index("event")
            series.append(("Crowell 2013 PGD", crow.loc[order, "absolute_error"], PGD_COLORS["crowell"]))
        height = 0.8 / len(series)
        y = np.arange(len(order))
        for k, (label, values, color) in enumerate(series):
            axis.barh(y + (k - (len(series) - 1) / 2) * height, values.to_numpy(), height, color=color,
                      alpha=0.92, label=label)
        axis.axvline(0.2, color="#E9A23B", lw=1.5, ls="--")
        labels = [f"{e} ({new.loc[e, 'mw_catalog']:.2f}, n={int(new.loc[e, 'n_stations'])})" for e in order]
        axis.set_yticks(y, labels, fontsize=8.3)
        axis.set_xlim(0, max(0.45, float(max(s[1].max() for s in series)) * 1.1))
        axis.set_xlabel("Absolute endpoint error vs catalog (Mw)")
        axis.set_title(f"{letter}. {COHORT_LABEL[cohort]} | label: event (catalog Mw, stations)",
                       loc="left", fontweight="bold", fontsize=10.5)
        axis.grid(axis="x", color="#D7DCE2", linewidth=0.6)
        axis.legend(frameon=False, loc="lower right", fontsize=8.5)
    figure.suptitle("Absolute endpoint error per event (dashed line: 0.20 Mw)", fontsize=14, fontweight="bold", y=0.96)
    return _save(figure, stem)


def figure_station_convergence(validation: dict[str, Any], old: dict[str, Any], stem: Path) -> list[Path]:
    """Station-level B (released) estimates at anchor horizons for the six validation events."""
    configure_matplotlib()
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.2))
    figure.subplots_adjust(left=0.05, right=0.99, bottom=0.08, top=0.88, wspace=0.22, hspace=0.3)
    stations = validation["stations"]
    old_stations = old["stations"]
    handles: list = []
    labels: list[str] = []
    for index, (axis, event) in enumerate(zip(axes.flat, EVENTS)):
        rows = stations[(stations["event"] == event) & (stations["method"] == B_ALL)]
        rows = rows[rows["constrained_window_sec"] >= 1.0]  # P-arrived stations only (B*)
        catalog = float(rows["mw_catalog"].iloc[0])
        n_all = int(stations[(stations["event"] == event) & (stations["method"] == A_FINAL)
                             & (stations["observation_horizon_sec"] == 200)].shape[0])
        rng = np.random.default_rng(0)
        for h in ANCHORS:
            sub = rows[rows["observation_horizon_sec"] == h]
            jitter = rng.uniform(-4.0, 4.0, len(sub))
            sc = axis.scatter(np.full(len(sub), h) + jitter, sub["mw_pred"], s=16, color=VARIANTS[PRIMARY]["color"],
                              alpha=0.5, linewidth=0, rasterized=True)
        med = rows.groupby("observation_horizon_sec")["mw_pred"].median()
        line_new = axis.plot(med.index, med.values, color=VARIANTS[PRIMARY]["color"], lw=2.2, marker="o", ms=5.5, zorder=4)[0]
        old_rows = old_stations[(old_stations["event"] == event) & (old_stations["method"] == A_FINAL)]
        old_med = old_rows.groupby("observation_horizon_sec")["mw_pred"].median()
        line_old = axis.plot(old_med.index, old_med.values, color="#5F6368", lw=1.5, ls="--", marker="s", ms=4.5, zorder=3)[0]
        cat = axis.axhline(catalog, color="#202124", lw=1.1)
        if index == 0:
            handles += [sc, line_new, line_old, cat]
            labels += ["NEW-3 station B (P-arrived)", "NEW-3 median B*", "OLD median A", "Catalog Mw"]
        axis.set_xticks(ANCHORS)
        axis.set_xlabel("Observed causal prefix (s)")
        axis.set_ylabel("Station magnitude estimate (Mw)")
        axis.set_title(f"{'ABCDEF'[index]}. {event} | Mw {catalog:.2f} | {n_all} stations", loc="left",
                       fontweight="bold", fontsize=10.5)
        axis.grid(True, color="#D7DCE2", linewidth=0.55)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=4, frameon=False, fontsize=9.5)
    figure.suptitle("Station-level released magnitude across causal horizons (validation events)",
                    fontsize=14, fontweight="bold", y=0.985)
    return _save(figure, stem)


DATASET_NPZ = Path(
    "/home/lihe/PINN_Mag/data/magnitude-label-snapshots/"
    "phase39-expanded-20260831T035810Z-2e1fa4c1/gnss_events_matched.phase39_expanded.npz"
)
MECHANISM_ORDER = ("Reverse", "Strike slip", "Normal")
MECHANISM_COLORS = {"Reverse": "#3B6EA8", "Strike slip": "#E9A23B", "Normal": "#2A9D8F"}


def event_metadata() -> pd.DataFrame:
    """Per-event focal mechanism / depth from the frozen dataset snapshot."""
    npz = np.load(DATASET_NPZ, allow_pickle=True)
    frame = pd.DataFrame({
        "event": [str(e) for e in npz["events"]],
        "mechanism": [str(m) for m in npz["mechanism"]],
        "strike": npz["strike"].astype(float),
        "dip": npz["dip"].astype(float),
        "rake": npz["rake"].astype(float),
        "depth_km": npz["depth_km"].astype(float),
    })
    if set(frame["mechanism"]) - set(MECHANISM_ORDER):
        raise ValueError(f"unexpected mechanism labels: {sorted(set(frame['mechanism']))}")
    return frame


def figure_error_by_mechanism(cohort_table: pd.DataFrame, meta: pd.DataFrame, stem: Path) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    """Signed 200 s error grouped by focal mechanism (A), against rake (B) and depth (C)."""
    configure_matplotlib()
    frame = cohort_table if "mechanism" in cohort_table.columns else cohort_table.merge(meta, on="event", how="left")
    frame = frame[frame["cohort"].isin(("train", "validation"))].copy()
    if frame["mechanism"].isna().any():
        raise ValueError("mechanism missing for some evaluated events")
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    figure.subplots_adjust(left=0.05, right=0.99, bottom=0.13, top=0.84, wspace=0.28)
    markers = {"train": "o", "validation": "^"}

    ax = axes[0]
    rng = np.random.default_rng(1)
    for k, mech in enumerate(MECHANISM_ORDER):
        for model, offset, color, alpha in (("old_error", -0.25, "#7F7F7F", 0.75), ("new3_error", 0.25, None, 0.95)):
            sub = frame[frame["mechanism"] == mech]
            values = sub[model].to_numpy()
            x = k + offset + rng.uniform(-0.07, 0.07, len(sub))
            for cohort, marker in markers.items():
                sel = (sub["cohort"] == cohort).to_numpy()
                ax.scatter(x[sel], values[sel], marker=marker, s=46,
                           color=color or MECHANISM_COLORS[mech], edgecolor="white", lw=0.6, alpha=alpha, zorder=3)
            ax.hlines(np.mean(values), k + offset - 0.14, k + offset + 0.14, color="#202124", lw=2.0, zorder=4)
            ax.text(k + offset, -0.485, f"{'OLD' if model == 'old_error' else 'NEW-3'}\nMAE {np.mean(np.abs(values)):.3f}\nbias {np.mean(values):+.3f}",
                    ha="center", va="bottom", fontsize=7.0)
    ax.axhline(0.0, color="#202124", lw=0.9)
    ax.axhspan(-0.2, 0.2, color="#E9A23B", alpha=0.10, lw=0)
    ax.set_xticks(range(len(MECHANISM_ORDER)))
    ax.set_xticklabels([f"{m}\n(n={int((frame['mechanism'] == m).sum())})" for m in MECHANISM_ORDER])
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(-0.6, len(MECHANISM_ORDER) - 0.4)
    ax.set_ylabel("Signed 200 s event error vs catalog (Mw)")
    ax.set_title("A. Error by focal mechanism | left grey = OLD, right = NEW-3", loc="left", fontweight="bold", fontsize=10.5)
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.55)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", ls="", color="#5F6368", label="train event"),
        Line2D([0], [0], marker="^", ls="", color="#5F6368", label="validation event"),
        Line2D([0], [0], color="#202124", lw=2.0, label="group mean"),
    ], frameon=False, fontsize=8.5, loc="upper left")

    for ax, column, label, letter in ((axes[1], "rake", "Rake (deg)", "B"), (axes[2], "depth_km", "Hypocentral depth (km)", "C")):
        for mech in MECHANISM_ORDER:
            sub = frame[frame["mechanism"] == mech]
            for cohort, marker in markers.items():
                s = sub[sub["cohort"] == cohort]
                ax.scatter(s[column], s["new3_error"], marker=marker, s=48 + 120 * np.sqrt(s["n_stations"] / frame["n_stations"].max()),
                           color=MECHANISM_COLORS[mech], edgecolor="white", lw=0.6, alpha=0.92, zorder=3,
                           label=f"{mech} ({cohort})" if letter == "B" else None)
        for row in frame[(frame["new3_error"].abs() > 0.18) | (frame["cohort"] == "validation")].itertuples(index=False):
            ax.annotate(row.event, (getattr(row, column), row.new3_error), xytext=(5, 4), textcoords="offset points", fontsize=7.5)
        ax.axhline(0.0, color="#202124", lw=0.9)
        ax.axhspan(-0.2, 0.2, color="#E9A23B", alpha=0.10, lw=0)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel(label)
        ax.set_ylabel("NEW-3 signed 200 s error (Mw)")
        ax.grid(True, color="#D7DCE2", linewidth=0.55)
        ax.set_title(f"{letter}. NEW-3 error against {label.split(' (')[0].lower()} | size = station count",
                     loc="left", fontweight="bold", fontsize=10.5)
    axes[1].set_xticks([-180, -90, 0, 90, 180])
    axes[1].legend(frameon=False, fontsize=7.6, ncol=2, loc="lower left")
    axes[2].axvspan(47.4, 115, color="#6C757D", alpha=0.08, lw=0)
    axes[2].text(50, 0.43, "deeper than any\ntraining event", fontsize=8, color="#5F6368")
    figure.suptitle("Endpoint error by focal mechanism, rake and depth (training + validation events; test not evaluated)",
                    fontsize=13.5, fontweight="bold", y=0.965)
    paths = _save(figure, stem)
    summary = (frame.groupby(["mechanism", "cohort"])
               .agg(n_events=("event", "size"), n_stations=("n_stations", "sum"),
                    new3_mae=("new3_error", lambda v: float(np.mean(np.abs(v)))),
                    new3_bias=("new3_error", "mean"),
                    old_mae=("old_error", lambda v: float(np.mean(np.abs(v)))),
                    old_bias=("old_error", "mean"),
                    crowell_mae=("crowell_error", lambda v: float(np.mean(np.abs(v)))))
               .reset_index())
    return paths, frame, summary


def cohort_summary_table(split_frame: pd.DataFrame, cohorts: dict[str, dict[str, dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for cohort in ("train", "validation"):
        new = endpoint_events(cohorts[cohort]["new3"]).set_index("event")
        old = endpoint_events(cohorts[cohort]["old"]).set_index("event")
        crow = endpoint_events(cohorts[cohort]["new3"], "crowell").set_index("event")
        for event in new.index:
            rows.append(dict(
                cohort=cohort, event=event, mw_catalog=float(new.loc[event, "mw_catalog"]),
                n_stations=int(new.loc[event, "n_stations"]),
                new3_mw_200s=float(new.loc[event, "mw_pred_median"]),
                new3_error=float(new.loc[event, "error_vs_catalog"]),
                old_mw_200s=float(old.loc[event, "mw_pred_median"]),
                old_error=float(old.loc[event, "error_vs_catalog"]),
                crowell_mw_200s=float(crow.loc[event, "mw_pred_median"]) if event in crow.index else np.nan,
                crowell_error=float(crow.loc[event, "error_vs_catalog"]) if event in crow.index else np.nan,
            ))
    for row in split_frame[split_frame["role"] == "test"].itertuples(index=False):
        rows.append(dict(cohort="test", event=row.event, mw_catalog=float(row.magnitude_catalog),
                         n_stations=int(row.n_stations)))
    return pd.DataFrame(rows)


def generate(output_dir: Path) -> dict[str, Any]:
    variants = {key: load_variant(key) for key in VARIANTS}
    table, gates, per_event = build_tables(variants)
    figures_dir = output_dir / "figures"
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    generated += figure_event_trajectories(variants[PRIMARY], variants["old"], figures_dir / "01_event_trajectories")
    generated += figure_variant_comparison(variants, figures_dir / "02_variant_comparison")
    generated += figure_horizon_metrics(variants, table, figures_dir / "03_horizon_mae_and_objectives")
    generated += figure_released_moment_illustration(variants[PRIMARY], figures_dir / "04_released_moment_illustration")

    cohorts = {
        "train": {
            "new3": load_cohort_replay(TRAIN_REPLAYS["new3"], "train", VARIANTS["new3"]["checkpoint_sha256"]),
            "old": load_cohort_replay(TRAIN_REPLAYS["old"], "train", VARIANTS["old"]["checkpoint_sha256"]),
        },
        "validation": {
            "new3": load_cohort_replay(VARIANTS["new3"]["replay"], "validation", VARIANTS["new3"]["checkpoint_sha256"]),
            "old": load_cohort_replay(VARIANTS["old"]["replay"], "validation", VARIANTS["old"]["checkpoint_sha256"]),
        },
    }
    split_paths, split_frame = figure_split_overview(figures_dir / "05_split_overview")
    generated += split_paths
    generated += figure_endpoint_event_scatter(cohorts, figures_dir / "06_endpoint_event_scatter")
    generated += figure_endpoint_station_scatter(cohorts, figures_dir / "07_endpoint_station_scatter")
    generated += figure_validation_vs_pgd_scatter(cohorts["validation"]["new3"], figures_dir / "08_validation_vs_pgd_scatter")
    generated += figure_event_errors(cohorts, figures_dir / "09_event_absolute_errors")
    generated += figure_station_convergence(cohorts["validation"]["new3"], cohorts["validation"]["old"],
                                            figures_dir / "10_station_convergence")
    cohort_table = cohort_summary_table(split_frame, cohorts).merge(event_metadata(), on="event", how="left")
    cohort_table.to_csv(analysis_dir / "cohort_event_summary.csv", index=False, lineterminator="\n")
    mech_paths, _, mech_summary = figure_error_by_mechanism(cohort_table, event_metadata(), figures_dir / "11_error_by_mechanism")
    generated += mech_paths
    mech_summary.to_csv(analysis_dir / "error_by_mechanism.csv", index=False, lineterminator="\n")
    for key, replay in cohorts["train"].items():
        shutil.copyfile(replay["root"] / "summary.json", analysis_dir / f"replay_summary_train_{key}.json")
        shutil.copyfile(replay["root"] / "train_event_predictions.csv", analysis_dir / f"event_trajectories_train_{key}.csv")

    table.to_csv(analysis_dir / "validation_objective_by_variant.csv", index=False, lineterminator="\n")
    gates.to_csv(analysis_dir / "validation_gates.csv", index=False, lineterminator="\n")
    per_event.to_csv(analysis_dir / "validation_objective_per_event_all_variants.csv", index=False, lineterminator="\n")
    for key, var in variants.items():
        shutil.copyfile(var["root"] / "summary.json", analysis_dir / f"replay_summary_{var['short'].lower()}.json")
        shutil.copyfile(var["root"] / "validation_event_predictions.csv",
                        analysis_dir / f"event_trajectories_{var['short'].lower()}.csv")
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "figure_manifest.json":
            files.append({"path": path.relative_to(output_dir).as_posix(),
                          "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    manifest = {
        "status": "complete",
        "cohort": "validation",
        "split_assignment_sha256": EXPECTED_SPLIT_SHA256,
        "variants": {k: {"short": v["short"], "label": v["label"], "replay": str(v["replay"]),
                         "checkpoint_sha256": v["summary"]["checkpoint_sha256"]} for k, v in variants.items()},
        "files": files,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "complete", "figures": [str(p) for p in generated],
            "gates": gates.to_dict(orient="records")}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = generate(args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
