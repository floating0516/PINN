"""Manuscript figures (SRL layout) for the released-moment causal magnitude paper.

Reads only frozen artefacts (fixed-split manifest, dataset snapshot metadata,
validation / training / one-time test replays, published endpoint tables) and
writes vector PDF + 600 dpi PNG figures sized for the SSA two-column layout
(full width 7.0 in, single column 3.4 in) with Liberation Sans (Arial metrics)
at 7-9 pt.

Usage: venv/bin/python scripts/plotting/plot_srl_released_moment_figures.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.provenance import sha256_file  # noqa: E402

RUNS = Path("/home/lihe/PINN_Mag/runs")
DATASET_NPZ = Path(
    "/home/lihe/PINN_Mag/data/magnitude-label-snapshots/"
    "phase39-expanded-20260831T035810Z-2e1fa4c1/gnss_events_matched.phase39_expanded.npz"
)
FIXED_SPLIT_DIR = PROJECT_ROOT / "docs/results/phase39-expanded-fixed-split"
OUTPUT_DIR = PROJECT_ROOT / "paper/srl-released-moment/figures"
EXPECTED_SPLIT_SHA256 = "e4807aa1e6b5b389caf23974f62ff9da6b8add7f7887ec23a59d5d35a455eba7"

REPLAY = {
    "val_new": RUNS / "phase39-causal-released-moment-scale1p0-seed73-20260907-v1-replay-validation",
    "val_old": RUNS / "phase39-causal-4a25-released-replay-validation-20260907-v2",
    "train_new": RUNS / "phase39-causal-released-moment-scale1p0-seed73-20260907-v1-replay-train",
    "train_old": RUNS / "phase39-causal-4a25-released-replay-train-20260907",
    "test_old_causal": RUNS / "phase39-causal-expanded-fixed-test-20260831-v1",
    # One-time held-out test replay of the final candidate (decided 2026-09-08) and the
    # same-frame replay of the old causal checkpoint (already tested on 2026-08-31).
    "test_new": RUNS / "phase39-causal-released-moment-scale1p0-seed73-20260907-v1-replay-test",
    "test_old": RUNS / "phase39-causal-4a25-released-replay-test-20260908",
}
CHECKPOINTS = {
    "new": "56d55b1ad65388ae",
    "old": "4a2540241fcbd918",
    "endpoint": "5905ccaabcbcfc15",
}
VALIDATION_EVENTS = ("Anchorage2018", "Maule2010", "Noto2024", "Parkfield2004", "RatIslands2014", "SandPoint2020")
TEST_EVENTS = ("Napa2014", "ak014cbigci8", "2016p661332", "Puebla2017", "Ridgecrest2019",
               "us7000i9bw", "Ecuador2016", "Tokachi2003", "Iquique2014")

# ----------------------------------------------------------------------------- style
FULL_W = 7.0
COL_W = 3.4
C = {
    "new": "#C0392B",        # released-moment model
    "old": "#5F6A72",        # causal Phase 39 (final-Mw target)
    "endpoint": "#8E6C8A",   # endpoint Phase 39
    "label": "#2E8B57",      # SCARDEC released label
    "crowell": "#2E6FB0",
    "ruhl": "#4CA64C",
    "melgar": "#9467BD",
    "truth": "#1B1F23",
    "band": "#F2C57C",
    "grid": "#D9DEE3",
    "train": "#3B6EA8",
    "validation": "#E9A23B",
    "test": "#2A9D8F",
    "sea": "#EEF4F8",
    "land": "#FFFFFF",
    "coast": "#7B8794",
}
PGD = {"crowell": ("Crowell et al. (2013)", "-"), "ruhl": ("Ruhl et al. (2019)", "--"), "melgar": ("Melgar et al. (2015)", "-.")}
MECH_MARKER = {"Reverse": "o", "Strike slip": "s", "Normal": "^"}
MECH_LABEL = {"Reverse": "reverse", "Strike slip": "strike-slip", "Normal": "normal"}


def style() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "figure.dpi": 110,
    })


def panel_label(ax: plt.Axes, text: str, x: float = -0.14, y: float = 1.04) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def inside_label(ax: plt.Axes, text: str, x: float = 0.03, y: float = 0.97) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=10)


def save(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for suffix, dpi in ((".pdf", 300), (".png", 600)):
        path = stem.with_suffix(suffix)
        meta = {"CreationDate": None, "ModDate": None} if suffix == ".pdf" else None
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, metadata=meta)
        out.append(path)
    plt.close(fig)
    return out


def grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=C["grid"], linewidth=0.45, alpha=0.9)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- data
def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def event_table() -> pd.DataFrame:
    """39 events: role, catalog Mw, station count, mechanism, coordinates, depth."""
    split = load_json(FIXED_SPLIT_DIR / "fixed_split_manifest.json")
    if split["assignment_sha256"] != EXPECTED_SPLIT_SHA256:
        raise ValueError("split assignment changed")
    rows = []
    for role in ("train", "validation", "test"):
        for e in split["roles"][role]["events"]:
            rows.append({"event": e["event"], "role": role, "mw": float(e["magnitude_catalog"]), "n_stations": int(e["n_stations"])})
    frame = pd.DataFrame(rows)
    npz = np.load(DATASET_NPZ, allow_pickle=True)
    meta = pd.DataFrame({
        "event": [str(e) for e in npz["events"]],
        "lat": npz["latitude"].astype(float),
        "lon": npz["longitude"].astype(float),
        "depth_km": npz["depth_km"].astype(float),
        "mechanism": [str(m) for m in npz["mechanism"]],
        "rake": npz["rake"].astype(float),
        "origin_time": [str(t) for t in npz["origin_time"]],
    })
    frame = frame.merge(meta, on="event", how="left")
    if frame["lat"].isna().any() or len(frame) != 39:
        raise ValueError("event metadata join failed")
    return frame


def read_replay(key: str, cohort: str) -> dict[str, pd.DataFrame]:
    root = REPLAY[key]
    summary = load_json(root / "summary.json")
    if summary.get("status") != "complete":
        raise ValueError(f"incomplete replay {root}")
    if summary.get("split_assignment_sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError(f"split changed {root}")
    prefix = cohort
    out = {
        "events": pd.read_csv(root / f"{prefix}_event_predictions.csv"),
        "horizon": pd.read_csv(root / f"{prefix}_horizon_metrics.csv"),
        "summary": summary,
    }
    station_path = root / f"{prefix}_anchor_station_predictions.csv"
    if station_path.exists():
        out["stations"] = pd.read_csv(station_path)
    return out


def curve(frame: pd.DataFrame, method: str, event: str) -> pd.DataFrame:
    return frame[(frame["method"] == method) & (frame["event"] == event)].sort_values("observation_horizon_sec")


def endpoint(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = frame[(frame["method"] == method) & (frame["observation_horizon_sec"] == 200)].copy()
    rows["abs_error"] = rows["error_vs_catalog"].abs()
    return rows.sort_values("mw_catalog").reset_index(drop=True)


def coastlines() -> list[np.ndarray]:
    result = subprocess.run(["gmt", "coast", "-Rg", "-M", "-W", "-Dl", "-A5000"], check=True, text=True, capture_output=True)
    segments, current = [], []
    for line in result.stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(">"):
            if len(current) >= 2:
                segments.append(np.asarray(current))
            current = []
            continue
        parts = s.split()
        if len(parts) >= 2:
            lon = float(parts[0])
            current.append((lon - 360.0 if lon > 180.0 else lon, float(parts[1])))
    if len(current) >= 2:
        segments.append(np.asarray(current))
    return segments


# ----------------------------------------------------------------------------- Fig 1
LABEL_OFFSETS = {  # manual label placement (points) for the validation / test events on the map
    "ak014cbigci8": (-4, 8), "SandPoint2020": (-4, -10), "Anchorage2018": (6, -2),
    "Napa2014": (-42, 4), "Ridgecrest2019": (5, -8), "Parkfield2004": (-52, -2),
    "Puebla2017": (7, -11), "us7000i9bw": (-48, 6), "Ecuador2016": (5, -2),
    "Iquique2014": (5, 2), "Maule2010": (5, -8), "Tokachi2003": (5, 5), "Noto2024": (5, -8),
    "RatIslands2014": (-58, 6), "2016p661332": (-50, -9),
}


def fig1_dataset(events: pd.DataFrame, stem: Path) -> list[Path]:
    fig = plt.figure(figsize=(FULL_W, 4.75))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], width_ratios=[1.6, 1.0], hspace=0.28, wspace=0.24,
                          left=0.065, right=0.99, top=0.98, bottom=0.09)
    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor(C["sea"])
    for seg in coastlines():
        # break segments that wrap across the date line
        jumps = np.flatnonzero(np.abs(np.diff(seg[:, 0])) > 180.0)
        for piece in np.split(seg, jumps + 1):
            if len(piece) >= 2:
                ax.plot(piece[:, 0], piece[:, 1], color=C["coast"], lw=0.35, zorder=1)
    for role in ("train", "validation", "test"):
        for mech, marker in MECH_MARKER.items():
            sub = events[(events["role"] == role) & (events["mechanism"] == mech)]
            if sub.empty:
                continue
            ax.scatter(sub["lon"], sub["lat"], s=6 + 9 * (sub["mw"] - 5.5) ** 2, marker=marker, color=C[role],
                       edgecolor="white", lw=0.4, alpha=0.95, zorder=3)
    for row in events[events["role"] != "train"].itertuples(index=False):
        ax.annotate(row.event, (row.lon, row.lat), xytext=LABEL_OFFSETS.get(row.event, (4, 2)),
                    textcoords="offset points", fontsize=5.2, color=C[row.role], zorder=5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-62, 75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-60, 76, 30))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    grid(ax)
    handles = [Line2D([0], [0], marker="o", ls="", ms=5, color=C[r], label=f"{lab}") for r, lab in
               (("train", "training (24)"), ("validation", "validation (6)"), ("test", "test (9)"))]
    handles += [Line2D([0], [0], marker=m, ls="", ms=5, color="#444444", markerfacecolor="none", label=MECH_LABEL[k])
                for k, m in MECH_MARKER.items()]
    handles += [Line2D([0], [0], marker="o", ls="", ms=s, color="#444444", markerfacecolor="none", label=f"Mw {m}")
                for m, s in ((6.0, 2.6), (7.5, 5.0), (9.0, 7.6))]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.66, 0.02), ncol=3, fontsize=6.2, handletextpad=0.3,
              columnspacing=0.9, borderaxespad=0.3, facecolor="white", framealpha=0.9, frameon=True, edgecolor="none")
    inside_label(ax, "(a)", x=0.008, y=0.975)

    ax = fig.add_subplot(gs[1, 0])
    ypos = {"train": 2, "validation": 1, "test": 0}
    for role in ("train", "validation", "test"):
        sub = events[events["role"] == role].sort_values("mw").reset_index(drop=True)
        # stack near-coincident magnitudes vertically so every event stays visible
        offsets = np.zeros(len(sub))
        for i in range(1, len(sub)):
            if sub.loc[i, "mw"] - sub.loc[i - 1, "mw"] < 0.08:
                offsets[i] = offsets[i - 1] + 0.16
        for (_, row), off in zip(sub.iterrows(), offsets):
            ax.scatter(row["mw"], ypos[role] + off, s=8 + 60 * np.sqrt(row["n_stations"] / events["n_stations"].max()),
                       marker=MECH_MARKER[row["mechanism"]], color=C[role], edgecolor="white", lw=0.4, alpha=0.92)
    ax.axvline(6.4, color=C["truth"], lw=0.6, ls=":")
    ax.text(6.44, -0.52, "smallest training event, Mw 6.4", fontsize=6.0, ha="left", va="center", color="#4A5568")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Test", "Validation", "Training"])
    ax.set_xlabel("Catalog magnitude (Mw)")
    ax.set_xlim(5.8, 9.3)
    ax.set_ylim(-0.7, 2.75)
    grid(ax, "x")
    inside_label(ax, "(b)", x=0.01, y=0.97)

    ax = fig.add_subplot(gs[1, 1])
    counts = events.groupby("role")["n_stations"].sum()
    nev = events.groupby("role")["event"].size()
    order = ("train", "validation", "test")
    ax.bar(range(3), [counts[r] for r in order], color=[C[r] for r in order], width=0.62)
    for i, r in enumerate(order):
        ax.text(i, counts[r] + 30, f"{counts[r]:,}\n({nev[r]} events)", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Training", "Validation", "Test"])
    ax.set_ylabel("Accepted station records")
    ax.set_ylim(0, 2300)
    grid(ax, "y")
    inside_label(ax, "(c)", x=0.03, y=0.97)
    return save(fig, stem)


# ----------------------------------------------------------------------------- Fig 2
def _box(ax, x, y, w, h, text, fc="#F7F9FB", ec="#4A5568", fs=6.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015", fc=fc, ec=ec, lw=0.7, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            linespacing=1.25, zorder=3)


def _arrow(ax, p, q, color="#4A5568", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.75, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=0, shrinkB=0))


def fig2_method(val_new: dict[str, pd.DataFrame], stem: Path, event: str = "Maule2010") -> list[Path]:
    fig = plt.figure(figsize=(FULL_W, 4.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], hspace=0.5, wspace=0.3, left=0.005, right=0.99, top=0.95, bottom=0.11)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    grey, red, gold = "#4A5568", C["new"], "#B7791F"
    # column 1: inputs and the observation used by L_synth
    _box(ax, 0.02, 0.74, 0.29, 0.13, "Radial GNSS displacement\nfirst $h$ s observed,\nzero-padded to 200 s", fc="#EAF1FB")
    _box(ax, 0.02, 0.55, 0.29, 0.12, "Station geometry\n$\\ln r,\\ \\sin\\theta,\\ \\cos\\theta,$\n$\\sin\\varphi,\\ \\cos\\varphi$", fc="#EAF1FB")
    _box(ax, 0.02, 0.12, 0.29, 0.17, "$\\mathcal{L}_\\mathrm{synth}$ ($\\lambda_\\mathrm{synth}=0.5$)\nobserved $u$ vs synthetic $\\hat u$\namplitude-normalised,\npolarity-tolerant", fc="#FFF8E6", ec=gold, fs=6.0)
    # column 2: network, forward operator, synthetic waveform
    _box(ax, 0.38, 0.60, 0.25, 0.22, "TCN + Transformer\nencoder\n(1.01 M parameters,\nunchanged Phase 39)", fc="#FFFFFF", bold=True)
    _box(ax, 0.38, 0.33, 0.25, 0.16, "Double-couple forward\noperator (far-field P, S)\n$\\hat u = F(\\dot M;\\,r,\\theta,\\varphi)$", fc="#FFFFFF")
    _box(ax, 0.38, 0.12, 0.25, 0.10, "Synthetic radial\ndisplacement $\\hat u(t)$", fc="#EAF1FB")
    # column 3: STF and the two magnitudes
    _box(ax, 0.70, 0.66, 0.28, 0.14, "Non-negative moment rate\n$\\dot M(t) = M_0\\,p(t)$\n$0 \\leq t \\leq 200$ s", fc="#FDECEA", ec=red)
    _box(ax, 0.70, 0.44, 0.28, 0.13, "Released magnitude $B(h)$\n$\\int_0^{\\,h-\\tau_P}\\dot M\\,dt \\rightarrow M_\\mathrm{w}$", fc="#FDECEA", ec=red, fs=6.1, bold=True)
    _box(ax, 0.70, 0.27, 0.28, 0.10, "Final magnitude $A(h)$\n$\\int_0^{\\,200}\\dot M\\,dt \\rightarrow M_\\mathrm{w}$", fc="#FFFFFF", ec=red)
    _box(ax, 0.70, 0.04, 0.28, 0.15, "$\\mathcal{L}^{h}_\\mathrm{MSE}+\\mathcal{L}^{h}_\\mathrm{mag}$\non $[0,\\,h-\\tau_P]$ only\ntargets: SCARDEC $\\dot M$,\nreleased moment", fc="#FFF8E6", ec=gold)
    # arrows
    _arrow(ax, (0.31, 0.805), (0.38, 0.74), color=grey)
    _arrow(ax, (0.31, 0.61), (0.38, 0.67), color=grey)
    _arrow(ax, (0.63, 0.71), (0.70, 0.73), color=grey)
    _arrow(ax, (0.84, 0.66), (0.84, 0.57), color=red)
    _arrow(ax, (0.84, 0.44), (0.84, 0.37), color=red)
    _arrow(ax, (0.70, 0.66), (0.63, 0.47), color=grey)
    _arrow(ax, (0.505, 0.33), (0.505, 0.22), color=grey)
    _arrow(ax, (0.38, 0.17), (0.31, 0.20), color=gold)
    _arrow(ax, (0.165, 0.74), (0.165, 0.29), color=gold, ls=":")
    _arrow(ax, (0.95, 0.44), (0.95, 0.19), color=gold, rad=-0.35)
    ax.text(0.09, 0.985, "Prefixes $h\\in[5,199]$ s and the 200 s endpoint are trained jointly. Counterfactual moment scaling\n"
                         "$a = 10^{1.5\\,\\Delta M_\\mathrm{w}}$, $\\Delta M_\\mathrm{w}\\in[-1.0,\\,0.5]$ rescales waveform, target STF and target moment.",
            fontsize=5.9, va="top", color=grey, linespacing=1.3)
    ax.text(0.02, 0.02, "$\\tau_P$: station P-wave arrival;  $h-\\tau_P$: causally observable source window", fontsize=6.0, va="bottom", color=grey)
    inside_label(ax, "(a)", x=0.0, y=1.0)

    # right column: illustration on one near-source station
    root = REPLAY["val_new"]
    npz = np.load(root / "validation_prefix_stf.npz")
    stations = val_new["stations"]
    anchor = stations[(stations["method"] == "released") & (stations["event"] == event) & (stations["observation_horizon_sec"] == 30)]
    anchor = anchor.sort_values("constrained_window_sec", ascending=False).iloc[0]
    station = str(anchor["station"])
    tau = 30.0 - float(anchor["constrained_window_sec"])
    catalog = float(anchor["mw_catalog"])
    idx = int(np.flatnonzero((npz["events"] == event) & (npz["stations"] == station))[0])
    rate = npz["stf_over_m_ref"][idx].astype(np.float64) * float(npz["m_ref_nm"])
    horizons = npz["horizons_sec"].astype(int)
    t = np.arange(rate.shape[1]) + 0.5
    mag = lambda m: (2.0 / 3.0) * (np.log10(np.maximum(m, 1.0e15)) - 9.1)  # noqa: E731
    window = np.clip(horizons - tau, 0.0, 200.0)
    mask = np.clip(window[:, None] - np.arange(rate.shape[1])[None, :], 0.0, 1.0)
    b_curve = mag((rate * mask).sum(axis=1))
    a_curve = mag(rate.sum(axis=1))

    ax = fig.add_subplot(gs[0, 1])
    show = (20, 40, 80, 120, 200)
    cmap = plt.get_cmap("viridis")
    for k, h in enumerate(show):
        color = cmap(k / (len(show) - 1))
        ax.plot(t, rate[h - 1] / 1e18, color=color, lw=0.9, label=f"$h$ = {h} s")
        ax.axvline(max(h - tau, 0), color=color, lw=0.6, ls=":")
    ax.set_xlim(0, 200)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Source time since origin (s)")
    ax.set_ylabel("$\\dot M(t)$ ($10^{18}$ N m s$^{-1}$)")
    ax.legend(ncol=2, loc="upper right", handlelength=1.4)
    ax.set_title(f"{event}, station {station}, P arrival $\\tau_P$ = {tau:.1f} s", fontsize=7, fontweight="normal", loc="left")
    grid(ax)
    inside_label(ax, "(b)", x=0.02, y=0.97)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(horizons, b_curve, color=C["new"], lw=1.3, label="$B(h)$: released magnitude")
    ax.plot(horizons, a_curve, color=C["new"], lw=0.9, ls="--", label="$A(h)$: final magnitude")
    ax.axhline(catalog, color=C["truth"], lw=0.7, ls=":", label=f"catalog $M_\\mathrm{{w}}$ {catalog:.2f}")
    ax.axvline(tau, color=C["crowell"], lw=0.6, ls=":")
    ax.text(tau + 2.5, 4.15, "P arrival", fontsize=6.2, color=C["crowell"])
    ax.set_xlim(1, 200)
    ax.set_ylim(3.8, 9.4)
    ax.set_xlabel("Observed prefix $h$ (s since origin)")
    ax.set_ylabel("Magnitude ($M_\\mathrm{w}$)")
    ax.legend(loc="lower right")
    grid(ax)
    inside_label(ax, "(c)", x=0.02, y=0.97)
    return save(fig, stem)


# ----------------------------------------------------------------------------- Fig 3
def _identity(ax, lo, hi, band=0.2):
    line = np.linspace(lo, hi, 50)
    ax.fill_between(line, line - band, line + band, color=C["band"], alpha=0.25, lw=0)
    ax.plot(line, line, color=C["truth"], lw=0.7)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    grid(ax)


def _endpoint_scatter(ax, sets, lo, hi, annotate=None, size_ref=None, legend_loc="upper left"):
    """sets: list of (frame, style dict, label template). Frames come from endpoint()."""
    _identity(ax, lo, hi)
    ref = size_ref if size_ref is not None else sets[0][0]["n_stations"].max()
    for frame, kw, label in sets:
        size = 12 + 50 * np.sqrt(frame["n_stations"] / ref)
        ax.scatter(frame["mw_catalog"], frame["mw_pred_median"], s=size, label=label.format(mae=frame["abs_error"].mean()), **kw)
    if annotate is not None:
        for row in annotate.itertuples(index=False):
            dx, dy = ANNOT_OFFSETS.get(row.event, (4, 3))
            ax.annotate(row.event, (row.mw_catalog, row.mw_pred_median), xytext=(dx, dy), textcoords="offset points", fontsize=5.4)
    ax.set_xlabel("Catalog magnitude ($M_\\mathrm{w}$)")
    ax.set_ylabel("Event-median estimate at 200 s ($M_\\mathrm{w}$)")
    ax.legend(loc=legend_loc, fontsize=5.6, handletextpad=0.2, borderaxespad=0.3, labelspacing=0.35)


ANNOT_OFFSETS = {
    "RatIslands2014": (4, -9), "SandPoint2020": (4, 3), "Noto2024": (-36, 6), "Anchorage2018": (4, -8),
    "Maule2010": (-30, -9), "Parkfield2004": (-4, 7),
    "Napa2014": (4, -9), "ak014cbigci8": (4, 3), "2016p661332": (-8, 9), "Puebla2017": (4, -8), "Ridgecrest2019": (-50, 3),
    "us7000i9bw": (4, -7), "Ecuador2016": (4, -8), "Tokachi2003": (-46, -9), "Iquique2014": (4, 1),
}
STYLE_NEW = dict(color=C["new"], edgecolor="white", lw=0.4, zorder=4)
STYLE_OLD = dict(facecolor="none", edgecolor=C["old"], lw=0.8, zorder=3)
STYLE_ENDPOINT = dict(marker="s", facecolor="none", edgecolor=C["endpoint"], lw=0.8, zorder=3)
STYLE_CROWELL = dict(marker="D", facecolor="none", edgecolor=C["crowell"], lw=0.7, zorder=3)


def _mae_vs_horizon(ax, series, pgd_source, ylim):
    for rows, kw, label in series:
        rows = rows.sort_values("observation_horizon_sec")
        ax.plot(rows["observation_horizon_sec"], rows["event_mae"], label=label, **kw)
    for m, (label, ls) in PGD.items():
        rows = pgd_source[(pgd_source["method"] == m) & (pgd_source["reference"] == "catalog")].sort_values("observation_horizon_sec")
        ax.plot(rows["observation_horizon_sec"], rows["event_mae"], color=C[m], lw=0.9, ls=ls, label=f"{label} PGD")
    ax.set_xlim(1, 200)
    ax.set_ylim(*ylim)
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.set_xlabel("Observed prefix (s since origin)")
    ax.set_ylabel("Event MAE vs catalog ($M_\\mathrm{w}$)")
    ax.legend(loc="upper right", fontsize=6.0, labelspacing=0.35)
    grid(ax)


def horizon_rows(src, method):
    rows = src["horizon"]
    return rows[(rows["method"] == method) & (rows["reference"] == "catalog")]


def fig3_validation(val_new, val_old, train_new, train_old, stem: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.55), layout="constrained", gridspec_kw={"width_ratios": [1.0, 1.15, 1.0]})
    new, old, crow = endpoint(val_new["events"], "final"), endpoint(val_old["events"], "final"), endpoint(val_new["events"], "crowell")
    _endpoint_scatter(axes[0], [
        (crow, STYLE_CROWELL, "Crowell PGD (MAE {mae:.3f})"),
        (old, STYLE_OLD, "causal, final-$M_\\mathrm{{w}}$ target (MAE {mae:.3f})"),
        (new, STYLE_NEW, "released-moment target (MAE {mae:.3f})"),
    ], 5.6, 9.2, annotate=new)
    inside_label(axes[0], "(a)", x=0.86, y=0.12)
    _mae_vs_horizon(axes[1], [
        (horizon_rows(val_new, "released_constrained"), dict(color=C["new"], lw=1.4), "released-moment target, $B^*$"),
        (horizon_rows(val_old, "final"), dict(color=C["old"], lw=1.0), "causal final-$M_\\mathrm{w}$ target, $A$"),
    ], val_new["horizon"], (0, 1.5))
    inside_label(axes[1], "(b)")
    tn, to, tc = endpoint(train_new["events"], "final"), endpoint(train_old["events"], "final"), endpoint(train_new["events"], "crowell")
    _endpoint_scatter(axes[2], [
        (tc, STYLE_CROWELL, "Crowell PGD (MAE {mae:.3f})"),
        (to, STYLE_OLD, "causal, final-$M_\\mathrm{{w}}$ target (MAE {mae:.3f})"),
        (tn, STYLE_NEW, "released-moment target (MAE {mae:.3f})"),
    ], 6.0, 9.4)
    inside_label(axes[2], "(c)", x=0.86, y=0.12)
    return save(fig, stem)


# ----------------------------------------------------------------------------- Fig 4 / 6 trajectories
def trajectory_grid(frame_new, frame_old, events: Sequence[str], stem: Path, *, ncols: int, height: float,
                    show_new: bool, old_label: str, ylim=(4.6, 9.4)) -> list[Path]:
    nrows = int(np.ceil(len(events) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_W, height), sharex=True, sharey=True)
    top = 0.86 if nrows == 2 else 0.905
    fig.subplots_adjust(left=0.065, right=0.99, top=top, bottom=0.10 if nrows == 2 else 0.075, hspace=0.30, wspace=0.10)
    handles, labels = [], []
    for k, (ax, event) in enumerate(zip(np.ravel(axes), events)):
        src = frame_new if show_new else frame_old
        catalog = float(src.loc[src["event"] == event, "mw_catalog"].median())
        n_st = int(curve(src, "final" if show_new else "direct", event)["n_stations"].iloc[-1])
        series = []
        if show_new:
            series += [
                ("released_constrained", frame_new, dict(color=C["new"], lw=1.4, zorder=6), "released-moment target: B* (P-arrived stations)"),
                ("final", frame_new, dict(color=C["new"], lw=0.8, ls="--", alpha=0.9, zorder=5), "released-moment target: A (final)"),
                ("final", frame_old, dict(color=C["old"], lw=0.9, zorder=4), old_label),
                ("released_ref_constrained", frame_new, dict(color=C["label"], lw=0.9, ls=":", zorder=5), "SCARDEC released label B*"),
            ]
        else:
            series += [("direct", frame_old, dict(color=C["old"], lw=1.3, zorder=6), old_label)]
        for method, frame, kw, label in series:
            rows = curve(frame, method, event)
            (line,) = ax.plot(rows["observation_horizon_sec"], rows["mw_pred_median"], **kw)
            if k == 0:
                handles.append(line)
                labels.append(label)
        for m, (label, ls) in PGD.items():
            rows = curve(src, m, event)
            (line,) = ax.plot(rows["observation_horizon_sec"], rows["mw_pred_median"], color=C[m], lw=0.8, ls=ls, zorder=3)
            if k == 0:
                handles.append(line)
                labels.append(f"{label} PGD")
        ax.axhspan(catalog - 0.3, catalog + 0.3, color=C["truth"], alpha=0.06, lw=0)
        cat = ax.axhline(catalog, color=C["truth"], lw=0.7, ls=":")
        if k == 0:
            handles.append(cat)
            labels.append("catalog Mw (±0.3 band)")
        ax.set_xlim(1, 200)
        ax.set_ylim(*ylim)
        ax.set_xticks([1, 50, 100, 150, 200])
        ax.set_title(f"({'abcdefghi'[k]}) {event}: $M_\\mathrm{{w}}$ {catalog:.2f}, {n_st} stations", loc="left", fontsize=7.2)
        grid(ax)
        if k % ncols == 0:
            ax.set_ylabel("Event-median $M_\\mathrm{w}$")
        if k // ncols == nrows - 1:
            ax.set_xlabel("Observed prefix (s since origin)")
    for ax in np.ravel(axes)[len(events):]:
        ax.axis("off")
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 0.995), fontsize=6.4, handlelength=1.8, columnspacing=1.2)
    return save(fig, stem)


# ----------------------------------------------------------------------------- Fig 5 test
def fig5_test(test_new, test_old, stem: Path) -> list[Path]:
    ev = pd.read_csv(FIXED_SPLIT_DIR / "selected_test_event_predictions.csv")
    ev["abs_error"] = ev["error_vs_catalog"].abs()
    ne = endpoint(test_new["events"], "released_constrained")
    ca = endpoint(test_old["events"], "final")
    cr = endpoint(test_new["events"], "crowell")
    ev = ev.sort_values("mw_catalog").reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.55), layout="constrained", gridspec_kw={"width_ratios": [1.0, 1.15, 1.1]})
    # Legend MAE over the applicability range (M >= 6.4, 7 events); all 9 events are plotted.
    def in_range_mae(frame):
        return float(frame.loc[frame["mw_catalog"] >= 6.4, "abs_error"].mean())
    _endpoint_scatter(axes[0], [
        (cr, STYLE_CROWELL, f"Crowell PGD (MAE {in_range_mae(cr):.3f})"),
        (ev, STYLE_ENDPOINT, f"endpoint model (MAE {in_range_mae(ev):.3f})"),
        (ca, STYLE_OLD, f"causal, final-$M_\\mathrm{{{{w}}}}$ target (MAE {in_range_mae(ca):.3f})"),
        (ne, STYLE_NEW, f"released-moment target (MAE {in_range_mae(ne):.3f})"),
    ], 5.7, 8.7, annotate=ne, size_ref=ne["n_stations"].max(), legend_loc="upper left")
    axes[0].axvspan(5.7, 6.4, color=C["band"], alpha=0.12, lw=0)
    inside_label(axes[0], "(a)", x=0.86, y=0.12)
    _mae_vs_horizon(axes[1], [
        (horizon_rows(test_old, "final"), dict(color=C["old"], lw=1.0), "causal final-$M_\\mathrm{w}$ target: A"),
        (horizon_rows(test_new, "released_constrained"), dict(color=C["new"], lw=1.3), "released-moment target: B*"),
    ], test_new["horizon"], (0, 1.6))
    inside_label(axes[1], "(b)")

    ax = axes[2]
    merged = ne[["event", "mw_catalog", "n_stations", "abs_error"]].rename(columns={"abs_error": "new"})
    merged = merged.merge(ca[["event", "abs_error"]].rename(columns={"abs_error": "causal"}), on="event")
    merged = merged.merge(ev[["event", "abs_error"]].rename(columns={"abs_error": "endpoint"}), on="event")
    merged = merged.merge(cr[["event", "abs_error"]].rename(columns={"abs_error": "crowell"}), on="event")
    merged = merged.sort_values("mw_catalog", ascending=True)
    y = np.arange(len(merged))
    h = 0.21
    ax.barh(y + 1.5 * h, merged["endpoint"], h, color=C["endpoint"], label="endpoint model")
    ax.barh(y + 0.5 * h, merged["causal"], h, color=C["old"], label="causal final-$M_\\mathrm{w}$ target")
    ax.barh(y - 0.5 * h, merged["new"], h, color=C["new"], label="released-moment target")
    ax.barh(y - 1.5 * h, merged["crowell"], h, color=C["crowell"], label="Crowell PGD")
    ax.axvline(0.2, color=C["band"], lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{e}\n$M_\\mathrm{{w}}$ {m:.2f}, $n$={n}" for e, m, n in zip(merged["event"], merged["mw_catalog"], merged["n_stations"])],
                       fontsize=5.6, linespacing=1.0)
    ax.set_xlabel("Absolute error at 200 s ($M_\\mathrm{w}$)")
    ax.set_xlim(0, 1.45)
    ax.set_ylim(-0.6, len(merged) - 0.4)
    ax.legend(loc="upper right", fontsize=6.0, labelspacing=0.35)
    grid(ax, "x")
    inside_label(ax, "(c)", x=0.03, y=0.985)
    return save(fig, stem)


# ----------------------------------------------------------------------------- Fig S1 / S2
def figS1_mechanism(events: pd.DataFrame, val_new, val_old, train_new, train_old, stem: Path) -> list[Path]:
    rows = []
    for cohort, new, old in (("train", train_new, train_old), ("validation", val_new, val_old)):
        n = endpoint(new["events"], "final").set_index("event")
        o = endpoint(old["events"], "final").set_index("event")
        for e in n.index:
            rows.append({"event": e, "cohort": cohort, "new": float(n.loc[e, "error_vs_catalog"]), "old": float(o.loc[e, "error_vs_catalog"]),
                         "n_stations": int(n.loc[e, "n_stations"])})
    frame = pd.DataFrame(rows).merge(events[["event", "mechanism", "rake", "depth_km"]], on="event")
    train_max_depth = float(events.loc[events["role"] == "train", "depth_km"].max())
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.45), layout="constrained")
    order = ("Reverse", "Strike slip", "Normal")
    ax = axes[0]
    rng = np.random.default_rng(3)
    for k, mech in enumerate(order):
        sub = frame[frame["mechanism"] == mech]
        for col, off, color in (("old", -0.2, C["old"]), ("new", 0.2, C["new"])):
            x = k + off + rng.uniform(-0.06, 0.06, len(sub))
            for cohort, marker in (("train", "o"), ("validation", "^")):
                sel = (sub["cohort"] == cohort).to_numpy()
                ax.scatter(x[sel], sub[col].to_numpy()[sel], marker=marker, s=14, color=color, edgecolor="white", lw=0.3, zorder=3)
            ax.hlines(sub[col].mean(), k + off - 0.13, k + off + 0.13, color=C["truth"], lw=1.2, zorder=4)
    ax.axhline(0, color=C["truth"], lw=0.6)
    ax.axhspan(-0.2, 0.2, color=C["band"], alpha=0.2, lw=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{MECH_LABEL[m]}\n(n={int((frame['mechanism'] == m).sum())})" for m in order])
    ax.set_ylim(-0.5, 0.5)
    ax.set_ylabel("Signed error at 200 s ($M_\\mathrm{w}$)")
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=C["old"], label="final-$M_\\mathrm{w}$ target"),
                       Line2D([0], [0], marker="o", ls="", color=C["new"], label="released-moment target"),
                       Line2D([0], [0], marker="o", ls="", color="#888888", label="training event"),
                       Line2D([0], [0], marker="^", ls="", color="#888888", label="validation event")],
              loc="lower left", fontsize=5.4, ncol=2, labelspacing=0.3, columnspacing=0.8)
    grid(ax, "y")
    inside_label(ax, "(a)")
    for ax, col, label, letter in ((axes[1], "rake", "Rake (°)", "(b)"), (axes[2], "depth_km", "Hypocentral depth (km)", "(c)")):
        for mech in order:
            sub = frame[frame["mechanism"] == mech]
            for cohort, marker in (("train", "o"), ("validation", "^")):
                s = sub[sub["cohort"] == cohort]
                ax.scatter(s[col], s["new"], marker=marker, s=12 + 40 * np.sqrt(s["n_stations"] / frame["n_stations"].max()),
                           color={"Reverse": C["crowell"], "Strike slip": C["validation"], "Normal": C["test"]}[mech], edgecolor="white", lw=0.3, zorder=3,
                           label=f"{MECH_LABEL[mech]} ({cohort})" if letter == "(b)" else None)
        labelled = frame[(frame["new"].abs() > 0.2) | (frame["cohort"] == "validation")]
        offsets = {
            "depth_km": {"Anchorage2018": (3, -8), "Maule2010": (3, 3), "SandPoint2020": (3, 5), "Noto2024": (4, -2),
                         "Parkfield2004": (3, 3), "RatIslands2014": (-46, 4), "Chignic2021": (3, -8)},
            "rake": {"Anchorage2018": (3, 3), "Maule2010": (3, -8), "SandPoint2020": (-48, 4), "RatIslands2014": (3, -8),
                     "Noto2024": (4, 2), "Parkfield2004": (3, 3), "Chignic2021": (3, -8)},
        }[col]
        for row in labelled.itertuples(index=False):
            dx, dy = offsets.get(row.event, (3, 2))
            ax.annotate(row.event, (getattr(row, col), row.new), xytext=(dx, dy), textcoords="offset points", fontsize=5.0)
        ax.axhline(0, color=C["truth"], lw=0.6)
        ax.axhspan(-0.2, 0.2, color=C["band"], alpha=0.2, lw=0)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel(label)
        ax.set_ylabel("Released-moment model error ($M_\\mathrm{w}$)")
        grid(ax)
        inside_label(ax, letter)
    axes[1].set_xticks([-180, -90, 0, 90, 180])
    axes[1].legend(fontsize=5.0, ncol=2, loc="lower left", labelspacing=0.3, columnspacing=0.8)
    axes[2].axvspan(train_max_depth, 120, color="#6C757D", alpha=0.08, lw=0)
    axes[2].set_xlim(0, 120)
    axes[2].text(train_max_depth + 3, -0.45, f"deeper than any training\nevent (>{train_max_depth:.0f} km)", fontsize=5.6, color="#5F6368", va="bottom")
    return save(fig, stem)


def figS2_stations(val_new, train_new, stem: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.1))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.14, wspace=0.3)
    for ax, src, title, letter in ((axes[0], train_new, "training cohort, 1,798 stations", "(a)"), (axes[1], val_new, "validation cohort, 446 stations", "(b)")):
        st = src["stations"]
        st = st[(st["method"] == "final") & (st["observation_horizon_sec"] == 200)]
        ev = endpoint(src["events"], "final")
        lo, hi = 5.6, 9.6
        ax.plot([lo, hi], [lo, hi], color=C["truth"], lw=0.7)
        hb = ax.hexbin(st["mw_catalog"], st["mw_pred"], gridsize=40, mincnt=1, cmap="Blues", linewidths=0.0, extent=(lo, hi, lo, hi), bins="log")
        ax.scatter(ev["mw_catalog"], ev["mw_pred_median"], s=22, color=C["new"], edgecolor="white", lw=0.4, zorder=4, label="event median")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        mae = float((st["mw_pred"] - st["mw_catalog"]).abs().mean())
        ax.set_title(f"{title}; station MAE {mae:.3f} Mw", loc="left", fontsize=7.2, fontweight="normal")
        ax.set_xlabel("Catalog magnitude (Mw)")
        ax.set_ylabel("Station estimate at 200 s (Mw)")
        grid(ax)
        cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label("stations per bin", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
        ax.legend(loc="lower right")
        panel_label(ax, letter, x=-0.2)
    return save(fig, stem)


# ----------------------------------------------------------------------------- tables
def write_tables(events: pd.DataFrame, val_new, val_old, train_new, train_old, test_new, test_old, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ev_test = pd.read_csv(FIXED_SPLIT_DIR / "selected_test_event_predictions.csv").set_index("event")
    ne_t = endpoint(test_new["events"], "released_constrained").set_index("event")
    ca = endpoint(test_old["events"], "final").set_index("event")
    cr_t = endpoint(test_new["events"], "crowell").set_index("event")
    vn = endpoint(val_new["events"], "final").set_index("event")
    vo = endpoint(val_old["events"], "final").set_index("event")
    vc = endpoint(val_new["events"], "crowell").set_index("event")
    tn = endpoint(train_new["events"], "final").set_index("event")
    to = endpoint(train_old["events"], "final").set_index("event")
    tc = endpoint(train_new["events"], "crowell").set_index("event")
    rows = []
    for row in events.sort_values(["role", "mw"]).itertuples(index=False):
        e = row.event
        rec = dict(event=e, role=row.role, mw=row.mw, n_stations=row.n_stations, mechanism=row.mechanism, depth_km=row.depth_km,
                   lat=row.lat, lon=row.lon, origin_time=row.origin_time)
        if row.role == "train":
            rec.update(new_err=tn.loc[e, "error_vs_catalog"], old_err=to.loc[e, "error_vs_catalog"], crowell_err=tc.loc[e, "error_vs_catalog"])
        elif row.role == "validation":
            rec.update(new_err=vn.loc[e, "error_vs_catalog"], old_err=vo.loc[e, "error_vs_catalog"], crowell_err=vc.loc[e, "error_vs_catalog"])
        else:
            rec.update(new_err=ne_t.loc[e, "error_vs_catalog"], endpoint_err=ev_test.loc[e, "error_vs_catalog"],
                       old_err=ca.loc[e, "error_vs_catalog"], crowell_err=cr_t.loc[e, "error_vs_catalog"])
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out_dir / "table_events.csv", index=False, lineterminator="\n")

    def mae(s):
        return float(np.mean(np.abs(s)))
    summary = {
        "validation": {
            "released_moment": {"mae": mae(vn["error_vs_catalog"]), "rmse": float(np.sqrt(np.mean(vn["error_vs_catalog"] ** 2))),
                                "bias": float(vn["error_vs_catalog"].mean())},
            "causal_final_target": {"mae": mae(vo["error_vs_catalog"]), "rmse": float(np.sqrt(np.mean(vo["error_vs_catalog"] ** 2))),
                                    "bias": float(vo["error_vs_catalog"].mean())},
            "crowell": {"mae": mae(vc["error_vs_catalog"])},
            "ruhl": {"mae": mae(endpoint(val_new["events"], "ruhl")["error_vs_catalog"])},
            "melgar": {"mae": mae(endpoint(val_new["events"], "melgar")["error_vs_catalog"])},
            "objective_released_moment": val_new["summary"]["objective_by_method"]["released_constrained"],
            "objective_causal_final": val_old["summary"]["objective_by_method"]["final"],
            "objective_crowell": val_new["summary"]["objective_by_method"]["crowell"],
        },
        "train": {
            "released_moment": {"mae": mae(tn["error_vs_catalog"]), "bias": float(tn["error_vs_catalog"].mean())},
            "causal_final_target": {"mae": mae(to["error_vs_catalog"]), "bias": float(to["error_vs_catalog"].mean())},
            "crowell": {"mae": mae(tc["error_vs_catalog"])},
        },
        "test": {
            "evaluated_once": "NEW-3 declared final candidate 2026-09-08; single replay, no further tuning",
            "released_moment": {"mae": mae(ne_t["error_vs_catalog"]), "rmse": float(np.sqrt(np.mean(ne_t["error_vs_catalog"] ** 2))),
                                "bias": float(ne_t["error_vs_catalog"].mean())},
            "endpoint_model": {"mae": mae(ev_test["error_vs_catalog"]), "rmse": float(np.sqrt(np.mean(ev_test["error_vs_catalog"] ** 2)))},
            "causal_final_target": {"mae": mae(ca["error_vs_catalog"]), "rmse": float(np.sqrt(np.mean(ca["error_vs_catalog"] ** 2)))},
            "crowell": {"mae": mae(cr_t["error_vs_catalog"])},
            "ruhl": {"mae": mae(endpoint(test_new["events"], "ruhl")["error_vs_catalog"])},
            "melgar": {"mae": mae(endpoint(test_new["events"], "melgar")["error_vs_catalog"])},
            "objective_released_moment": test_new["summary"]["objective_by_method"]["released_constrained"],
            "objective_causal_final": test_old["summary"]["objective_by_method"]["final"],
            "objective_crowell": test_new["summary"]["objective_by_method"]["crowell"],
        },
    }
    (out_dir / "results_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")

    # LaTeX rows for the supplementary event table (one row per event, grouped by cohort)
    def fmt(value: float | None) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "--"
        return f"{value:+.2f}"

    lines = []
    role_name = {"train": "Training", "validation": "Validation", "test": "Test"}
    table = pd.DataFrame(rows)
    for role in ("train", "validation", "test"):
        sub = table[table["role"] == role].sort_values("mw")
        lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{role_name[role]} cohort}} ({len(sub)} events, {int(sub['n_stations'].sum()):,} records)}} \\\\")
        for r in sub.itertuples(index=False):
            name = r.event.replace("_", "\\_")
            date = r.origin_time[:10]
            first = fmt(r.new_err)
            lines.append(
                f"{name} & {date} & {r.mw:.2f} & {MECH_LABEL[r.mechanism]} & {r.depth_km:.0f} & {r.n_stations} & "
                f"{first} & {fmt(r.old_err)} & {fmt(r.crowell_err)} \\\\"
            )
    (out_dir / "events_table_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Self-contained longtable for the manuscript (\multicolumn cannot follow \input inside a row).
    header = (
        "Event & Origin date & $M_{\\mathrm{w}}$ & Mechanism & Depth (km) & Stations & NEW-3 & FINAL & Crowell \\\\\n\\midrule"
    )
    longtable = "\n".join([
        "\\begin{longtable}{@{}lccclrrrr@{}}",
        "\\caption*{\\textbf{Table S1.} Events, cohorts, faulting style, depth, accepted stations and signed 200-s "
        "event error (estimate $-$ catalog) of the released-moment model (NEW-3), the final-magnitude model (FINAL) "
        "and Crowell PGD.}\\\\",
        "\\toprule", header, "\\endfirsthead", "\\toprule", header, "\\endhead", "\\bottomrule", "\\endfoot",
        *lines,
        "\\end{longtable}",
    ])
    (out_dir / "events_table.tex").write_text(longtable + "\n", encoding="utf-8")


def generate(output_dir: Path) -> dict[str, Any]:
    style()
    events = event_table()
    val_new = read_replay("val_new", "validation")
    val_old = read_replay("val_old", "validation")
    train_new = read_replay("train_new", "train")
    train_old = read_replay("train_old", "train")
    test_new = read_replay("test_new", "test")
    test_old = read_replay("test_old", "test")
    for key, prefix in (("val_new", "new"), ("train_new", "new"), ("val_old", "old"), ("train_old", "old"), ("test_new", "new"), ("test_old", "old")):
        sha = read_replay(key, "validation" if key.startswith("val") else "train" if key.startswith("train") else "test")["summary"]["checkpoint_sha256"]
        if not str(sha).startswith(CHECKPOINTS[prefix]):
            raise ValueError(f"unexpected checkpoint for {key}: {sha}")
    figs: list[Path] = []
    figs += fig1_dataset(events, output_dir / "fig1_dataset")
    figs += fig2_method(val_new, output_dir / "fig2_method")
    figs += fig3_validation(val_new, val_old, train_new, train_old, output_dir / "fig3_validation_endpoint")
    figs += trajectory_grid(val_new["events"], val_old["events"], VALIDATION_EVENTS, output_dir / "fig4_validation_trajectories",
                            ncols=3, height=4.3, show_new=True, old_label="causal final-Mw target: A")
    figs += fig5_test(test_new, test_old, output_dir / "fig5_test_endpoint")
    figs += trajectory_grid(test_new["events"], test_old["events"], TEST_EVENTS, output_dir / "fig6_test_trajectories",
                            ncols=3, height=6.0, show_new=True, old_label="causal final-Mw target: A", ylim=(4.6, 9.4))
    figs += figS1_mechanism(events, val_new, val_old, train_new, train_old, output_dir / "figS1_mechanism")
    figs += figS2_stations(val_new, train_new, output_dir / "figS2_station_scatter")
    write_tables(events, val_new, val_old, train_new, train_old, test_new, test_old, output_dir.parent / "tables")
    manifest = {"files": [{"path": p.relative_to(output_dir.parent).as_posix(), "sha256": sha256_file(p)} for p in figs]}
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"figures": [str(p) for p in figs]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.output_dir.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
