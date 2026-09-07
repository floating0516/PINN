"""Why is Noto 2024 over-estimated?  Read-only diagnostics on frozen replays.

Three hypotheses are checked from replay artefacts and the dataset snapshot:

1. label: the SCARDEC 200 s magnitude used as the training target vs catalog;
2. near-field saturation: station error at 200 s against epicentral distance;
3. geometry prior: the model's *final* estimate ``A`` at ``h = 1 s`` (before any
   signal has arrived) against epicentral distance, on every validation and
   training record -- a strong distance dependence means the network has
   learned "a record accepted far from the source implies a large event".
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_NPZ = Path(
    "/home/lihe/PINN_Mag/data/magnitude-label-snapshots/"
    "phase39-expanded-20260831T035810Z-2e1fa4c1/gnss_events_matched.phase39_expanded.npz"
)
RUNS_ROOT = Path("/home/lihe/PINN_Mag/runs")
REPLAYS = {
    "validation": {
        "NEW-3": RUNS_ROOT / "phase39-causal-released-moment-scale1p0-seed73-20260907-v1-replay-validation",
        "OLD": RUNS_ROOT / "phase39-causal-4a25-released-replay-validation-20260907-v2",
    },
    "train": {
        "NEW-3": RUNS_ROOT / "phase39-causal-released-moment-scale1p0-seed73-20260907-v1-replay-train",
    },
}
EVENT = "Noto2024"
DISTANCE_BINS = [0, 50, 100, 150, 200, 300, 500, 3000]
FLOOR_NM = 1.0e15


class Geometry:
    def __init__(self) -> None:
        npz = np.load(DATASET_NPZ, allow_pickle=True)
        self.events = [str(e) for e in npz["events"]]
        self.lat = npz["latitude"].astype(float)
        self.lon = npz["longitude"].astype(float)
        self.station_info = npz["station_info"]
        self.catalog = {str(e): float(m) for e, m in zip(npz["events"], npz["magnitude_selected"])}
        self.stf_native = {str(e): float(m) for e, m in zip(npz["events"], npz["magnitude_stf_native"])}

    def distance_km(self, event: str, station: str) -> float:
        i = self.events.index(event)
        info = self.station_info[i][station]
        la0, lo0 = np.radians(self.lat[i]), np.radians(self.lon[i])
        la, lo = np.radians(float(info["lat"])), np.radians(float(info["lon"]))
        h = np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2
        return float(2 * 6371.0 * np.arcsin(np.sqrt(h)))


def station_errors(root: Path, cohort: str, geometry: Geometry) -> pd.DataFrame:
    frame = pd.read_csv(root / f"{cohort}_anchor_station_predictions.csv")
    rows = frame[(frame["method"] == "final") & (frame["observation_horizon_sec"] == 200)].copy()
    rows["dist_km"] = [geometry.distance_km(e, s) for e, s in zip(rows["event"], rows["station"])]
    rows["error"] = rows["mw_pred"] - rows["mw_catalog"]
    return rows


def prior_at_one_second(root: Path, cohort: str, geometry: Geometry) -> pd.DataFrame:
    npz = np.load(root / f"{cohort}_prefix_stf.npz")
    rate = npz["stf_over_m_ref"].astype(np.float64) * float(npz["m_ref_nm"])
    a1 = (2.0 / 3.0) * (np.log10(np.maximum(rate[:, 0, :].sum(axis=1), FLOOR_NM)) - 9.1)
    frame = pd.DataFrame({"event": npz["events"], "station": npz["stations"], "A_1s": a1})
    frame["dist_km"] = [geometry.distance_km(e, s) for e, s in zip(frame["event"], frame["station"])]
    frame["mw_catalog"] = frame["event"].map(geometry.catalog)
    return frame


def run(output_dir: Path) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = Geometry()
    new_val = station_errors(REPLAYS["validation"]["NEW-3"], "validation", geometry)
    old_val = station_errors(REPLAYS["validation"]["OLD"], "validation", geometry)
    prior_val = prior_at_one_second(REPLAYS["validation"]["NEW-3"], "validation", geometry)
    prior_train = prior_at_one_second(REPLAYS["train"]["NEW-3"], "train", geometry)

    noto_new = new_val[new_val["event"] == EVENT].copy()
    noto_old = old_val[old_val["event"] == EVENT].copy()
    noto_new["bin"] = pd.cut(noto_new["dist_km"], DISTANCE_BINS)
    noto_old["bin"] = pd.cut(noto_old["dist_km"], DISTANCE_BINS)
    by_bin = pd.DataFrame({
        "n_stations": noto_new.groupby("bin", observed=True)["error"].size(),
        "new3_median_error": noto_new.groupby("bin", observed=True)["error"].median(),
        "old_median_error": noto_old.groupby("bin", observed=True)["error"].median(),
    }).reset_index()
    by_bin.to_csv(output_dir / "noto_station_error_by_distance.csv", index=False, lineterminator="\n")

    prior_all = pd.concat([prior_train.assign(cohort="train"), prior_val.assign(cohort="validation")])
    prior_all["bin"] = pd.cut(prior_all["dist_km"], DISTANCE_BINS)
    prior_table = prior_all.groupby(["cohort", "bin"], observed=True)["A_1s"].agg(["size", "median"]).reset_index()
    prior_table.to_csv(output_dir / "prior_A1s_by_distance.csv", index=False, lineterminator="\n")
    corr_dist = float(np.corrcoef(prior_train["A_1s"], prior_train["dist_km"])[0, 1])
    corr_cat = float(np.corrcoef(prior_train["A_1s"], prior_train["mw_catalog"])[0, 1])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    figure.subplots_adjust(left=0.05, right=0.99, bottom=0.13, top=0.84, wspace=0.27)

    ax = axes[0]
    ax.scatter(noto_old["dist_km"], noto_old["error"], s=16, color="#7F7F7F", alpha=0.5, label="OLD 4a25")
    ax.scatter(noto_new["dist_km"], noto_new["error"], s=16, color="#D1495B", alpha=0.6, label="NEW-3")
    centers = [(a + b) / 2 for a, b in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:])]
    ax.plot(centers[: len(by_bin)], by_bin["new3_median_error"], color="#D1495B", lw=2.2, marker="o", label="NEW-3 bin median")
    ax.axhline(0.0, color="#202124", lw=0.9)
    ax.axhspan(-0.2, 0.2, color="#E9A23B", alpha=0.10, lw=0)
    ax.set_xscale("log")
    ax.set_xlim(3, 1000)
    ax.set_ylim(-1.0, 1.4)
    ax.set_xlabel("Epicentral distance (km)")
    ax.set_ylabel("Station error at 200 s vs catalog (Mw)")
    ax.set_title(f"A. {EVENT}: 200 s station error vs distance ({len(noto_new)} stations)", loc="left", fontweight="bold", fontsize=10.5)
    ax.grid(True, color="#D7DCE2", linewidth=0.55, which="both")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.text(0.03, 0.97, f"SCARDEC 200 s Mw {geometry.stf_native[EVENT]:.2f} vs catalog {geometry.catalog[EVENT]:.2f}\n"
            f"near field (<50 km, n=3): median {by_bin['new3_median_error'].iloc[0]:+.2f}\n"
            f"50-500 km: median {noto_new[(noto_new['dist_km'] > 50) & (noto_new['dist_km'] <= 500)]['error'].median():+.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.8,
            bbox={"facecolor": "white", "edgecolor": "#D7DCE2", "alpha": 0.92})

    ax = axes[1]
    ax.scatter(prior_train["dist_km"], prior_train["A_1s"], s=7, color="#3B6EA8", alpha=0.25, linewidth=0, rasterized=True, label="training records")
    ax.scatter(prior_val["dist_km"], prior_val["A_1s"], s=9, color="#E9A23B", alpha=0.5, linewidth=0, rasterized=True, label="validation records")
    noto_prior = prior_val[prior_val["event"] == EVENT]
    ax.scatter(noto_prior["dist_km"], noto_prior["A_1s"], s=9, color="#D1495B", alpha=0.6, linewidth=0, rasterized=True, label=f"{EVENT} records")
    ax.set_xscale("log")
    ax.set_xlim(3, 3000)
    ax.set_xlabel("Epicentral distance (km)")
    ax.set_ylabel("Final estimate A at h = 1 s (Mw)")
    ax.set_title("B. Prior before any signal: A(1 s) vs distance, NEW-3", loc="left", fontweight="bold", fontsize=10.5)
    ax.grid(True, color="#D7DCE2", linewidth=0.55, which="both")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(0.97, 0.05, f"train: corr(A(1 s), distance) = {corr_dist:.2f}\ncorr(A(1 s), catalog Mw) = {corr_cat:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.8,
            bbox={"facecolor": "white", "edgecolor": "#D7DCE2", "alpha": 0.92})

    ax = axes[2]
    train_far = prior_train[prior_train["dist_km"] > 200]
    counts = train_far["event"].value_counts()
    top = counts.head(8)
    ax.barh(range(len(top)), top.values, color="#3B6EA8")
    ax.set_yticks(range(len(top)), [f"{e} (Mw {geometry.catalog[e]:.1f})" for e in top.index], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Training records with epicentral distance > 200 km")
    ax.set_title(f"C. Who teaches the far field? ({len(train_far)} of {len(prior_train)} training records)",
                 loc="left", fontweight="bold", fontsize=10.5)
    ax.grid(axis="x", color="#D7DCE2", linewidth=0.55)
    for i, v in enumerate(top.values):
        ax.text(v + 5, i, f"{v}  ({100 * v / len(train_far):.0f}%)", va="center", fontsize=8.5)
    ax.set_xlim(0, top.values.max() * 1.3)

    figure.suptitle("Noto 2024 over-estimate: label, distance dependence and the learned geometry prior",
                    fontsize=13.5, fontweight="bold", y=0.965)
    for suffix, dpi in ((".png", 220), (".pdf", 300)):
        figure.savefig(output_dir / f"13_noto_diagnostic{suffix}", dpi=dpi, bbox_inches="tight",
                       metadata={"CreationDate": None, "ModDate": None} if suffix == ".pdf" else None)
    plt.close(figure)
    return {
        "noto_scardec_mw": geometry.stf_native[EVENT],
        "noto_catalog_mw": geometry.catalog[EVENT],
        "corr_prior_distance_train": corr_dist,
        "corr_prior_catalog_train": corr_cat,
        "far_records_train": int(len(train_far)),
        "far_records_top": {str(k): int(v) for k, v in top.items()},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args.output_dir.resolve())
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
