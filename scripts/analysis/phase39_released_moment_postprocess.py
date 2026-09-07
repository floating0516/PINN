"""Causal post-processing of released-magnitude event curves (no model inference).

Reads frozen validation replays of the released-moment checkpoints, applies
causal filters to the per-second event-median released magnitude ``B*`` and
recomputes the objective metrics, so that the effect of smoothing can be
separated from the effect of the model.

Filters (all strictly causal; ``h`` is the observation horizon in seconds):

* ``raw``        : ``B*(h)`` as reported by the replay.
* ``runmax``     : ``max_{k<=h} B*(k)`` -- released moment cannot decrease.
* ``envelope``   : ``E(h) = max(B*(h), E(h-1) - delta)`` -- non-increasing only
                   at a bounded rate ``delta`` (Mw/s), so late downward
                   corrections are allowed but slowly.
* ``ema_envelope``: exponential moving average (time constant ``tau`` s) of
                   ``B*`` followed by ``envelope``.

Also reports the 30 s Spearman rank correlation of the *final* estimate ``A``
(the model's guess of the endpoint), which is the quantity that should carry
early ranking information; ``B*`` at 30 s is by construction the moment
released so far and is not expected to rank events by final size.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.evaluate_phase39_causal_released_moment import (  # noqa: E402
    _sign_changes,
    _spearman,
    _stable_entry,
)

HORIZONS = np.arange(1, 201)
RUNS_ROOT = Path("/home/lihe/PINN_Mag/runs")
REPLAYS = {
    "OLD": RUNS_ROOT / "phase39-causal-4a25-released-replay-validation-20260907-v2",
    "NEW-1": RUNS_ROOT / "phase39-causal-released-moment-seed73-20260907-v1-replay-validation-v2",
    "NEW-2": RUNS_ROOT / "phase39-causal-released-moment-monotone-seed73-20260907-v1-replay-validation",
    "NEW-3": RUNS_ROOT / "phase39-causal-released-moment-scale1p0-seed73-20260907-v1-replay-validation",
}
B_STAR = "released_constrained"
A_FINAL = "final"
DEFAULT_DELTA = 0.005  # Mw per second
DEFAULT_TAU = 5.0  # seconds


def running_max(curve: np.ndarray) -> np.ndarray:
    out = np.full_like(curve, np.nan)
    current = -math.inf
    for i, value in enumerate(curve):
        if np.isfinite(value):
            current = max(current, value)
            out[i] = current
        elif np.isfinite(current):
            out[i] = current
    return out


def envelope(curve: np.ndarray, delta: float) -> np.ndarray:
    if delta < 0.0:
        raise ValueError("delta must be nonnegative")
    out = np.full_like(curve, np.nan)
    current = math.nan
    for i, value in enumerate(curve):
        if not np.isfinite(value):
            if np.isfinite(current):
                out[i] = current
            continue
        current = value if not np.isfinite(current) else max(value, current - delta)
        out[i] = current
    return out


def ema(curve: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    alpha = 1.0 - math.exp(-1.0 / tau)
    out = np.full_like(curve, np.nan)
    state = math.nan
    for i, value in enumerate(curve):
        if not np.isfinite(value):
            out[i] = state
            continue
        state = value if not np.isfinite(state) else state + alpha * (value - state)
        out[i] = state
    return out


FILTERS = {
    "raw": lambda c, delta, tau: c,
    "runmax": lambda c, delta, tau: running_max(c),
    "envelope": lambda c, delta, tau: envelope(c, delta),
    "ema_envelope": lambda c, delta, tau: envelope(ema(c, tau), delta),
}


def event_curves(frame: pd.DataFrame, method: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    rows = frame[frame["method"] == method]
    curves: dict[str, np.ndarray] = {}
    catalog: dict[str, float] = {}
    for event, group in rows.groupby("event"):
        arr = np.full(len(HORIZONS), np.nan)
        arr[group["observation_horizon_sec"].to_numpy(dtype=int) - 1] = group["mw_pred_median"].to_numpy(dtype=float)
        curves[str(event)] = arr
        catalog[str(event)] = float(group["mw_catalog"].iloc[0])
    return curves, catalog


def score(curves: dict[str, np.ndarray], catalog: dict[str, float], crowell_entry: dict[str, int | None],
          a_curves: dict[str, np.ndarray]) -> dict[str, Any]:
    events = sorted(curves)
    err200 = np.array([curves[e][-1] - catalog[e] for e in events])
    sign = [_sign_changes(curves[e]) for e in events]
    entry = {e: _stable_entry(curves[e], catalog[e], 0.3) for e in events}
    before = sum(
        1 for e in events
        if entry[e] is not None and (crowell_entry.get(e) is None or entry[e] < crowell_entry[e])
    )
    cat = np.array([catalog[e] for e in events])

    def rho(cs: dict[str, np.ndarray], h: int) -> float:
        return _spearman(np.array([cs[e][h - 1] for e in events]), cat)

    return {
        "event_mae_200s": float(np.mean(np.abs(err200))),
        "event_rmse_200s": float(np.sqrt(np.mean(err200**2))),
        "mean_sign_changes": float(np.mean(sign)),
        "max_sign_changes": int(np.max(sign)),
        "events_with_stable_entry": int(sum(v is not None for v in entry.values())),
        "median_stable_entry_sec": float(np.median([v for v in entry.values() if v is not None])) if any(
            v is not None for v in entry.values()) else float("nan"),
        "entry_before_crowell": int(before),
        "rho_Bstar_30s": rho(curves, 30),
        "rho_Bstar_60s": rho(curves, 60),
        "rho_A_30s": rho(a_curves, 30),
        "rho_A_60s": rho(a_curves, 60),
        "stable_entry_by_event": {e: entry[e] for e in events},
        "sign_changes_by_event": dict(zip(events, sign)),
    }


def run(output_dir: Path, delta: float, tau: float) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    filtered_curves: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    catalog_all: dict[str, float] = {}
    crowell_curves: dict[str, np.ndarray] = {}
    for variant, root in REPLAYS.items():
        frame = pd.read_csv(root / "validation_event_predictions.csv")
        b_curves, catalog = event_curves(frame, B_STAR)
        a_curves, _ = event_curves(frame, A_FINAL)
        crowell, _ = event_curves(frame, "crowell")
        crowell_curves = crowell
        catalog_all = catalog
        crowell_entry = {e: _stable_entry(crowell[e], catalog[e], 0.3) for e in crowell}
        for name, fn in FILTERS.items():
            curves = {e: fn(c, delta, tau) for e, c in b_curves.items()}
            filtered_curves[(variant, name)] = curves
            result = score(curves, catalog, crowell_entry, a_curves)
            rows.append({"variant": variant, "filter": name, "delta_mw_per_s": delta, "ema_tau_s": tau,
                         **{k: v for k, v in result.items() if not isinstance(v, dict)},
                         "stable_entry_by_event": json.dumps(result["stable_entry_by_event"]),
                         "sign_changes_by_event": json.dumps(result["sign_changes_by_event"])})
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "postprocess_objective.csv", index=False, lineterminator="\n")

    # Figure: NEW-3 raw vs ema_envelope per event, Crowell for reference.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    events = sorted(catalog_all)
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), sharex=True)
    figure.subplots_adjust(left=0.05, right=0.99, bottom=0.08, top=0.87, wspace=0.18, hspace=0.28)
    handles: list = []
    labels: list[str] = []
    for index, (axis, event) in enumerate(zip(axes.flat, events)):
        raw = filtered_curves[("NEW-3", "raw")][event]
        env = filtered_curves[("NEW-3", "ema_envelope")][event]
        rmx = filtered_curves[("NEW-3", "runmax")][event]
        l1 = axis.plot(HORIZONS, raw, color="#D1495B", lw=1.0, alpha=0.45)[0]
        l2 = axis.plot(HORIZONS, env, color="#D1495B", lw=2.3)[0]
        l3 = axis.plot(HORIZONS, rmx, color="#7A6F9B", lw=1.2, ls="--")[0]
        l4 = axis.plot(HORIZONS, crowell_curves[event], color="#4C78A8", lw=1.5)[0]
        cat = axis.axhline(catalog_all[event], color="#202124", lw=1.1, ls=":")
        axis.axhspan(catalog_all[event] - 0.3, catalog_all[event] + 0.3, color="#202124", alpha=0.05, lw=0)
        if index == 0:
            handles += [l1, l2, l3, l4, cat]
            labels += ["B* raw (NEW-3)", f"B* EMA({tau:.0f} s) + envelope ({delta:g} Mw/s)", "B* running max",
                       "Crowell 2013 PGD", "Catalog Mw"]
        axis.set_title(f"{'ABCDEF'[index]}. {event} | Mw {catalog_all[event]:.2f}", loc="left", fontweight="bold")
        axis.set_xlim(1, 200)
        axis.set_ylim(4.6, 9.4)
        axis.grid(True, color="#D7DCE2", linewidth=0.55)
        if index >= 3:
            axis.set_xlabel("Observed causal prefix (s since origin)")
        if index % 3 == 0:
            axis.set_ylabel("Event-median magnitude (Mw)")
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=5, frameon=False, fontsize=9.5)
    figure.suptitle("Causal post-processing of the released magnitude B* (validation events, NEW-3)",
                    fontsize=14, fontweight="bold", y=0.985)
    for suffix, dpi in ((".png", 220), (".pdf", 300)):
        figure.savefig(output_dir / f"12_causal_envelope{suffix}", dpi=dpi, bbox_inches="tight",
                       metadata={"CreationDate": None, "ModDate": None} if suffix == ".pdf" else None)
    plt.close(figure)
    return table


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    table = run(args.output_dir.resolve(), args.delta, args.tau)
    cols = ["variant", "filter", "event_mae_200s", "event_rmse_200s", "mean_sign_changes", "max_sign_changes",
            "events_with_stable_entry", "median_stable_entry_sec", "entry_before_crowell", "rho_Bstar_30s",
            "rho_Bstar_60s", "rho_A_30s", "rho_A_60s"]
    pd.set_option("display.width", 250)
    print(table[cols].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
