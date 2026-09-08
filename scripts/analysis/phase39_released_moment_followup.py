"""Follow-up figures/tables for the released-moment report: seed check and sampling knobs.

Compares four frozen validation replays that share the NEW-3 recipe:

* NEW-3        seed 73 (report primary)
* NEW-3 s42    same recipe, training RNG seed 42          (step 4: seed reproducibility)
* 5a           + event balance estimator -> replacement sampling (was inverse-count weights)
* 5b           + distance-stratified counterfactual scaling for far stations

Reads replay artefacts only (no inference). Writes ``figures/14_*``, ``figures/15_*``
and ``analysis/followup_*.csv`` under the report directory.

Usage: venv/bin/python scripts/analysis/phase39_released_moment_followup.py
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
from scripts.plotting.plot_phase39_causal_event_trajectories import configure_matplotlib  # noqa: E402

RUNS = base.RUNS_ROOT
FOLLOWUP: dict[str, dict[str, Any]] = {
    "new3": {
        "replay": RUNS / "phase39-causal-released-moment-scale1p0-seed73-20260907-v1-replay-validation",
        "short": "NEW-3",
        "label": "NEW-3 (seed 73)",
        "checkpoint_sha256": "56d55b1ad65388ae",
        "color": "#D1495B",
        "ls": "-",
    },
    "seed42": {
        "replay": RUNS / "phase39-causal-released-moment-scale1p0-seed42-20260907-v1-replay-validation",
        "short": "NEW-3 s42",
        "label": "NEW-3 recipe, seed 42",
        "checkpoint_sha256": "25dca4bb",
        "color": "#E6A0AA",
        "ls": "-",
    },
    "balanced": {
        "replay": RUNS / "phase39-causal-released-moment-scale1p0-balanced-seed73-20260907-v1-replay-validation",
        "short": "5a",
        "label": "5a: event resampling",
        "checkpoint_sha256": "0cde6daa",
        "color": "#1F77B4",
        "ls": "-",
    },
    "farscale": {
        "replay": RUNS / "phase39-causal-released-moment-scale1p0-farscale-seed73-20260907-v1-replay-validation",
        "short": "5b",
        "label": "5b: far-station down-scaling",
        "checkpoint_sha256": "c06ab33f",
        "color": "#9467BD",
        "ls": "-",
    },
}
OUTPUT_DIR = base.DEFAULT_OUTPUT_DIR


def load(key: str) -> dict[str, Any]:
    base.VARIANTS[key] = FOLLOWUP[key]  # reuse the validated loader
    return base.load_variant(key)


def figure_curves(variants: dict[str, dict[str, Any]], stem: Path) -> list[Path]:
    """Fig 14: B* per event for the four NEW-3-recipe checkpoints, Crowell as reference."""
    configure_matplotlib()
    figure, axes = base._grid(
        "Released magnitude B*: seed check and sampling knobs on the NEW-3 recipe (validation)"
    )
    handles: list = []
    labels: list[str] = []
    ref = variants["new3"]["events"]
    for index, (axis, event) in enumerate(zip(axes.flat, base.EVENTS)):
        catalog = float(ref.loc[ref["event"] == event, "mw_catalog"].median())
        for key, var in variants.items():
            rows = base._curve(var["events"], base.B_STAR, event)
            line = axis.plot(
                rows["observation_horizon_sec"], rows["mw_pred_median"],
                color=var["color"], lw=2.3 if key == "new3" else 1.5,
                alpha=1.0 if key == "new3" else 0.9, zorder=5 if key == "new3" else 3,
            )[0]
            if index == 0:
                handles.append(line)
                labels.append(var["label"])
        rows = base._curve(ref, "crowell", event)
        line = axis.plot(rows["observation_horizon_sec"], rows["mw_pred_median"],
                         color=base.PGD_COLORS["crowell"], lw=1.4, ls="--", zorder=2)[0]
        cat = axis.axhline(catalog, color="#202124", lw=1.2, ls=":", zorder=2)
        axis.axhspan(catalog - 0.3, catalog + 0.3, color="#202124", alpha=0.05, lw=0)
        if index == 0:
            handles += [line, cat]
            labels += ["Crowell 2013 PGD", "Catalog Mw (band: +/-0.3)"]
        axis.set_title(f"{'ABCDEF'[index]}. {event} | catalog Mw {catalog:.2f}",
                       loc="left", fontweight="bold", fontsize=10.5)
    base._finish_grid(figure, axes, handles, labels, (4.6, 9.4))
    return base._save(figure, stem)


def endpoint_table(variants: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for key, var in variants.items():
        obj = var["objective"]
        obj = obj[obj["method"] == base.B_STAR]
        for _, r in obj.iterrows():
            rows.append({"variant": var["short"], "event": r["event"],
                         "mw_catalog": r["mw_catalog"], "mw_200s": r["mw_200s"],
                         "error_200s": r["error_200s"]})
    return pd.DataFrame(rows)


def figure_errors(per_event: pd.DataFrame, variants: dict[str, dict[str, Any]], stem: Path) -> list[Path]:
    """Fig 15: signed 200 s endpoint error per validation event and variant."""
    configure_matplotlib()
    figure, axis = plt.subplots(figsize=(11.5, 4.6))
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.86)
    events = list(base.EVENTS)
    order = ["new3", "seed42", "balanced", "farscale"]
    width = 0.19
    x = np.arange(len(events))
    for i, key in enumerate(order):
        var = variants[key]
        sub = per_event[per_event["variant"] == var["short"]].set_index("event").loc[events]
        axis.bar(x + (i - 1.5) * width, sub["error_200s"], width, color=var["color"],
                 edgecolor="black", linewidth=0.4, label=var["label"])
    axis.axhline(0, color="#202124", lw=1.0)
    for y in (-0.2, 0.2):
        axis.axhline(y, color="#7F7F7F", lw=0.8, ls="--")
    axis.set_xticks(x)
    axis.set_xticklabels([f"{e}\nMw {per_event[per_event['event'] == e]['mw_catalog'].iloc[0]:.2f}"
                          for e in events], fontsize=9.5)
    axis.set_ylabel("200 s event error, B* - catalog (Mw)")
    axis.set_ylim(-0.25, 0.5)
    axis.grid(True, axis="y", color="#D7DCE2", linewidth=0.55, alpha=0.75)
    axis.legend(ncol=4, frameon=False, fontsize=9.5, loc="upper left")
    axis.set_title("Signed 200 s endpoint error per validation event (dashed: +/-0.2 Mw)",
                   loc="left", fontweight="bold", fontsize=11)
    return base._save(figure, stem)


def objective_table(variants: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for key, var in variants.items():
        s = var["summary"]
        b = s["objective_by_method"][base.B_STAR]
        a = s["objective_by_method"][base.A_FINAL]
        rows.append({
            "variant": var["short"],
            "replay": var["replay"].name,
            "checkpoint_sha256": s["checkpoint_sha256"],
            "seed": s.get("seed"),
            "event_mae_200s": b["event_mae_200s"],
            "event_rmse_200s": b["event_rmse_200s"],
            "event_bias_200s": b["event_bias_200s"],
            "mean_sign_changes_Bstar": b["mean_sign_changes"],
            "events_overshooting_catalog_at_1s_A": a["events_overshooting_catalog_at_1s"],
            "spearman_A_030s": a["spearman_vs_catalog_030s"],
            "spearman_Bstar_060s": b["spearman_vs_catalog_060s"],
            "events_with_stable_entry_Bstar": b["events_with_stable_entry"],
            "median_stable_entry_sec_Bstar": b["median_stable_entry_sec"],
        })
    return pd.DataFrame(rows)


def run(output_dir: Path) -> dict[str, Any]:
    variants = {key: load(key) for key in FOLLOWUP}
    figures_dir = output_dir / "figures"
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    per_event = endpoint_table(variants)
    table = objective_table(variants)
    paths = figure_curves(variants, figures_dir / "14_followup_seed_and_sampling_curves")
    paths += figure_errors(per_event, variants, figures_dir / "15_followup_endpoint_errors")
    table.to_csv(analysis_dir / "followup_objective_by_variant.csv", index=False, lineterminator="\n")
    per_event.to_csv(analysis_dir / "followup_objective_per_event.csv", index=False, lineterminator="\n")
    for key in ("seed42", "balanced", "farscale"):
        shutil.copyfile(variants[key]["root"] / "summary.json", analysis_dir / f"replay_summary_{key}.json")
        shutil.copyfile(variants[key]["root"] / "validation_event_predictions.csv",
                        analysis_dir / f"event_trajectories_{key}.csv")
    refresh_manifest(output_dir, variants)
    return {"figures": [str(p) for p in paths], "table": table.to_dict(orient="records")}


def refresh_manifest(output_dir: Path, variants: dict[str, dict[str, Any]]) -> None:
    """Re-hash every report file; keep the header written by the main plotting script."""
    manifest_path = output_dir / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.setdefault("variants", {})
    for key, var in variants.items():
        if key == "new3":
            continue
        manifest["variants"][key] = {
            "short": var["short"], "label": var["label"], "replay": str(var["replay"]),
            "checkpoint_sha256": var["summary"]["checkpoint_sha256"],
        }
    manifest["files"] = [
        {"path": p.relative_to(output_dir).as_posix(), "sha256": base.sha256_file(p),
         "size_bytes": p.stat().st_size}
        for p in sorted(output_dir.rglob("*")) if p.is_file() and p.name != "figure_manifest.json"
    ]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
