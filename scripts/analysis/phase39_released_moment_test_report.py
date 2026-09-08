"""One-time held-out test report for the released-moment final candidate (NEW-3).

Reads two frozen test replays produced by
``scripts/evaluation/evaluate_phase39_causal_released_moment.py --cohort test``:

* NEW-3 (checkpoint 56d55b1a..., declared final candidate on 2026-09-08)
* OLD   (causal Phase 39 checkpoint 4a254024..., already tested on 2026-08-31)

and writes ``figures/16_*``, ``figures/17_*`` and ``analysis/test_*`` under the
report directory. No inference.

Usage: venv/bin/python scripts/analysis/phase39_released_moment_test_report.py
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

from scripts.plotting import plot_phase39_causal_released_moment as base  # noqa: E402
from scripts.plotting.plot_phase39_causal_event_trajectories import configure_matplotlib, load_json  # noqa: E402

RUNS = base.RUNS_ROOT
TEST_REPLAYS = {
    "new3": {
        "replay": RUNS / "phase39-causal-released-moment-scale1p0-seed73-20260907-v1-replay-test",
        "short": "NEW-3", "label": "NEW-3 (final candidate)",
        "checkpoint_sha256": "56d55b1ad65388ae", "color": "#D1495B",
    },
    "old": {
        "replay": RUNS / "phase39-causal-4a25-released-replay-test-20260908",
        "short": "OLD", "label": "OLD causal Phase 39",
        "checkpoint_sha256": "4a2540241fcbd918", "color": "#7F7F7F",
    },
}
TEST_EVENTS = ("Napa2014", "ak014cbigci8", "Puebla2017", "2016p661332", "Ridgecrest2019",
               "us7000i9bw", "Ecuador2016", "Tokachi2003", "Iquique2014")  # by catalog Mw
TRAIN_MIN_MW = 6.4


def load_test(key: str) -> dict[str, Any]:
    spec = TEST_REPLAYS[key]
    root = spec["replay"]
    summary = load_json(root / "summary.json")
    if summary.get("status") != "complete" or summary.get("cohort") != "test":
        raise ValueError(f"not a complete test replay: {root}")
    if summary.get("split_assignment_sha256") != base.EXPECTED_SPLIT_SHA256:
        raise ValueError(f"split assignment changed: {root}")
    if not str(summary["checkpoint_sha256"]).startswith(spec["checkpoint_sha256"]):
        raise ValueError(f"checkpoint changed: {root}")
    events = pd.read_csv(root / "test_event_predictions.csv")
    if set(events["event"]) != set(TEST_EVENTS):
        raise ValueError(f"test event coverage changed: {root}")
    return {"key": key, "root": root, "summary": summary, "events": events,
            "objective": pd.read_csv(root / "test_objective_per_event.csv"), **spec}


def figure_test_trajectories(new: dict[str, Any], old: dict[str, Any], stem: Path) -> list[Path]:
    """Fig 16: 9 test events, NEW-3 B*/A, OLD A, SCARDEC label B*, PGD, catalog."""
    configure_matplotlib()
    figure, axes = plt.subplots(3, 3, figsize=(16.5, 13.2), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.06, right=0.985, bottom=0.06, top=0.885, hspace=0.2, wspace=0.05)
    figure.suptitle("Held-out test events (evaluated once): released-moment model, SCARDEC label and PGD",
                    fontsize=14, fontweight="bold", y=0.985)
    events = new["events"]
    handles: list = []
    labels: list[str] = []
    for index, (axis, event) in enumerate(zip(axes.flat, TEST_EVENTS)):
        catalog = float(events.loc[events["event"] == event, "mw_catalog"].median())
        n_st = int(base._curve(events, base.A_FINAL, event).iloc[-1]["n_stations"])
        series = [
            (base.B_STAR, events, dict(color=new["color"], lw=2.3, ls="-", zorder=5), "B* released Mw, NEW-3"),
            (base.A_FINAL, events, dict(color=new["color"], lw=1.3, ls="--", alpha=0.85, zorder=4), "A final Mw, NEW-3"),
            (base.A_FINAL, old["events"], dict(color=old["color"], lw=1.4, ls="-", alpha=0.9, zorder=3), "A final Mw, OLD"),
            (base.B_REF_STAR, events, dict(color="#2CA02C", lw=1.5, ls=":", zorder=4), "B* from SCARDEC label"),
        ]
        for method, frame, style, label in series:
            rows = base._curve(frame, method, event)
            line = axis.plot(rows["observation_horizon_sec"], rows["mw_pred_median"], **style)[0]
            if index == 0:
                handles.append(line)
                labels.append(label)
        for method in base.PGD_METHODS:
            rows = base._curve(events, method, event)
            line = axis.plot(rows["observation_horizon_sec"], rows["mw_pred_median"], color=base.PGD_COLORS[method],
                             ls=base.PGD_STYLES[method], lw=1.55, alpha=0.9, zorder=3)[0]
            if index == 0:
                handles.append(line)
                labels.append(f"{base.PGD_LABELS[method]} PGD")
        cat = axis.axhline(catalog, color="#202124", lw=1.2, ls=":", zorder=2)
        axis.axhspan(catalog - 0.3, catalog + 0.3, color="#202124", alpha=0.05, lw=0)
        if index == 0:
            handles.append(cat)
            labels.append("Catalog Mw (band: +/-0.3)")
        tag = "  [below train min M6.4]" if catalog < TRAIN_MIN_MW else ""
        axis.set_title(f"{'ABCDEFGHI'[index]}. {event} | Mw {catalog:.2f} | {n_st} st{tag}",
                       loc="left", fontweight="bold", fontsize=10)
        axis.set_xlim(1, 200)
        axis.set_ylim(4.6, 9.4)
        axis.set_xticks([1, 50, 100, 150, 200])
        axis.grid(True, color="#D7DCE2", linewidth=0.55, alpha=0.75)
        if index // 3 == 2:
            axis.set_xlabel("Observed causal prefix (s since origin)")
        if index % 3 == 0:
            axis.set_ylabel("Event-median magnitude (Mw)")
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=5, frameon=False, fontsize=9.5)
    return base._save(figure, stem)


def endpoint_table(new: dict[str, Any], old: dict[str, Any]) -> pd.DataFrame:
    def pick(var: dict[str, Any], method: str, col: str) -> pd.Series:
        obj = var["objective"]
        return obj[obj["method"] == method].set_index("event")[col]

    n_st = new["events"].query("method == 'final' and observation_horizon_sec == 200").set_index("event")["n_stations"]
    table = pd.DataFrame({
        "event": TEST_EVENTS,
        "mw_catalog": pick(new, base.B_STAR, "mw_catalog").loc[list(TEST_EVENTS)].values,
        "n_stations": n_st.loc[list(TEST_EVENTS)].values,
        "new3_mw_200s": pick(new, base.B_STAR, "mw_200s").loc[list(TEST_EVENTS)].values,
        "new3_error": pick(new, base.B_STAR, "error_200s").loc[list(TEST_EVENTS)].values,
        "old_mw_200s": pick(old, base.B_STAR, "mw_200s").loc[list(TEST_EVENTS)].values,
        "old_error": pick(old, base.B_STAR, "error_200s").loc[list(TEST_EVENTS)].values,
        "crowell_error": pick(new, "crowell", "error_200s").loc[list(TEST_EVENTS)].values,
        "new3_stable_entry_sec": pick(new, base.B_STAR, "stable_entry_pm0p3_sec").loc[list(TEST_EVENTS)].values,
        "old_stable_entry_sec": pick(old, base.B_STAR, "stable_entry_pm0p3_sec").loc[list(TEST_EVENTS)].values,
        "crowell_stable_entry_sec": pick(new, "crowell", "stable_entry_pm0p3_sec").loc[list(TEST_EVENTS)].values,
        "new3_sign_changes": pick(new, base.B_STAR, "sign_changes").loc[list(TEST_EVENTS)].values,
    })
    table["below_train_min"] = table["mw_catalog"] < TRAIN_MIN_MW
    return table


def figure_test_endpoint(table: pd.DataFrame, new: dict[str, Any], old: dict[str, Any], stem: Path) -> list[Path]:
    """Fig 17: (A) 200 s endpoint vs catalog, (B) signed error per event."""
    configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 5.4), gridspec_kw={"width_ratios": [1.0, 1.35]})
    figure.subplots_adjust(left=0.06, right=0.99, bottom=0.2, top=0.86, wspace=0.18)
    ax = axes[0]
    base._identity_axis(ax, 5.6, 8.8)
    ax.scatter(table["mw_catalog"], table["mw_catalog"] + table["crowell_error"], s=55, marker="s",
               facecolor="none", edgecolor=base.PGD_COLORS["crowell"], lw=1.4, label="Crowell 2013 PGD", zorder=4)
    ax.scatter(table["mw_catalog"], table["old_mw_200s"], s=60, facecolor="none", edgecolor=old["color"], lw=1.4,
               label="OLD causal Phase 39", zorder=4)
    ax.scatter(table["mw_catalog"], table["new3_mw_200s"], s=60, color=new["color"], label="NEW-3 (final candidate)", zorder=5)
    ax.axvspan(5.6, TRAIN_MIN_MW, color="#F28E2B", alpha=0.08, lw=0)
    ax.text(5.65, 8.6, "below training\nminimum (M6.4)", fontsize=8.5, color="#B5651D", va="top")
    for _, r in table.iterrows():
        ax.annotate(r["event"], (r["mw_catalog"], r["new3_mw_200s"]), textcoords="offset points",
                    xytext=(6, -3 if r["new3_error"] < 0 else 4), fontsize=7.5, color="#444")
    ax.set_xlabel("Catalog Mw")
    ax.set_ylabel("200 s event-median estimate (Mw)")
    ax.set_title("A. Held-out test: 200 s endpoint vs catalog (9 events, 450 stations)", loc="left", fontweight="bold", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax = axes[1]
    x = np.arange(len(table))
    w = 0.27
    ax.bar(x - w, table["new3_error"], w, color=new["color"], edgecolor="black", lw=0.4, label="NEW-3")
    ax.bar(x, table["old_error"], w, color=old["color"], edgecolor="black", lw=0.4, label="OLD")
    ax.bar(x + w, table["crowell_error"], w, color=base.PGD_COLORS["crowell"], edgecolor="black", lw=0.4, label="Crowell 2013")
    ax.axhline(0, color="#202124", lw=1.0)
    for y in (-0.3, 0.3):
        ax.axhline(y, color="#7F7F7F", lw=0.8, ls="--")
    for i, r in table.iterrows():
        if r["below_train_min"]:
            ax.axvspan(i - 0.5, i + 0.5, color="#F28E2B", alpha=0.08, lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}\nMw {m:.2f} | {n} st" for e, m, n in zip(table["event"], table["mw_catalog"], table["n_stations"])],
                       fontsize=7.5, rotation=20, ha="right", rotation_mode="anchor")
    ax.set_ylabel("200 s error, estimate - catalog (Mw)")
    ax.set_title("B. Signed 200 s error per test event (dashed: +/-0.3; shaded: below train minimum)",
                 loc="left", fontweight="bold", fontsize=10.5)
    ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.55, alpha=0.75)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    return base._save(figure, stem)


def objective_table(new: dict[str, Any], old: dict[str, Any], table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var, method, label in ((new, base.B_STAR, "NEW-3 B*"), (new, base.A_FINAL, "NEW-3 A"),
                               (old, base.B_STAR, "OLD B*"), (old, base.A_FINAL, "OLD A"),
                               (new, "crowell", "Crowell 2013"), (new, "ruhl", "Ruhl 2019"), (new, "melgar", "Melgar 2015")):
        o = var["summary"]["objective_by_method"][method]
        obj = var["objective"]
        sub = obj[obj["method"] == method].set_index("event")
        in_range = [e for e in TEST_EVENTS if not table.set_index("event").loc[e, "below_train_min"]]
        err = sub.loc[in_range, "error_200s"]
        crowell = var["objective"].query("method == 'crowell'").set_index("event").loc[in_range, "error_200s"].abs()
        rows.append({
            "series": label,
            "event_mae_200s": o["event_mae_200s"], "event_rmse_200s": o["event_rmse_200s"], "event_bias_200s": o["event_bias_200s"],
            # Applicability range M >= 6.4 (training minimum): the report's main table.
            "in_range_event_count": len(in_range),
            "in_range_event_mae_200s": float(err.abs().mean()),
            "in_range_event_rmse_200s": float(np.sqrt((err ** 2).mean())),
            "in_range_event_bias_200s": float(err.mean()),
            "in_range_events_within_0p3": int((err.abs() <= 0.3).sum()),
            "in_range_wins_vs_crowell": int(((err.abs() - crowell) < -0.02).sum()),
            "in_range_ties_vs_crowell": int(((err.abs() - crowell).abs() <= 0.02).sum()),
            "in_range_losses_vs_crowell": int(((err.abs() - crowell) > 0.02).sum()),
            "mean_mw_1s": o.get("mean_mw_1s"), "events_overshooting_catalog_at_1s": o["events_overshooting_catalog_at_1s"],
            "mean_sign_changes": o["mean_sign_changes"],
            "events_with_stable_entry": o["events_with_stable_entry"], "median_stable_entry_sec": o["median_stable_entry_sec"],
            "spearman_vs_catalog_030s": o["spearman_vs_catalog_030s"], "spearman_vs_catalog_060s": o["spearman_vs_catalog_060s"],
        })
    return pd.DataFrame(rows)


def run(output_dir: Path) -> dict[str, Any]:
    new, old = load_test("new3"), load_test("old")
    figures_dir, analysis_dir = output_dir / "figures", output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    table = endpoint_table(new, old)
    objective = objective_table(new, old, table)
    paths = figure_test_trajectories(new, old, figures_dir / "16_test_event_trajectories")
    paths += figure_test_endpoint(table, new, old, figures_dir / "17_test_endpoint")
    table.to_csv(analysis_dir / "test_event_summary.csv", index=False, lineterminator="\n")
    objective.to_csv(analysis_dir / "test_objective_by_series.csv", index=False, lineterminator="\n")
    for var in (new, old):
        shutil.copyfile(var["root"] / "summary.json", analysis_dir / f"replay_summary_test_{var['key']}.json")
        shutil.copyfile(var["root"] / "test_event_predictions.csv", analysis_dir / f"event_trajectories_test_{var['key']}.csv")
        shutil.copyfile(var["root"] / "test_objective_per_event.csv", analysis_dir / f"test_objective_per_event_{var['key']}.csv")
    return {"figures": [str(p) for p in paths], "objective": objective.round(3).to_dict(orient="records"),
            "events": table.round(3).to_dict(orient="records")}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=base.DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(run(args.output_dir), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
