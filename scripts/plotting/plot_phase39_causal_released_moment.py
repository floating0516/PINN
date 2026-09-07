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
