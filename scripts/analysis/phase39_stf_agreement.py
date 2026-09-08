"""Reference (SCARDEC) STF for the validation cohort and its agreement with predicted STFs.

Step 1 rebuilds the validation cohort exactly as the frozen replay did (same
config snapshot, same fixed split) and stores the label STF per station in
``analysis/validation_reference_stf.npz``. No model is loaded.

Step 2 compares, per station and horizon, the predicted STF saved by the replay
(``validation_prefix_stf.npz``) with the label inside the constrained window
``[0, h - tau_P]``:

* ``shape_r``   Pearson correlation of the two moment-rate curves in the window
* ``delta_mw``  Mw(predicted released moment) - Mw(label released moment)

and, at h = 200 s, the same two numbers over the full 200 s STF. Results go to
``analysis/stf_agreement_stations.csv`` and ``analysis/stf_agreement_summary.csv``.

Usage: venv/bin/python scripts/analysis/phase39_stf_agreement.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation import evaluate_phase39_causal_released_moment as evaluator  # noqa: E402
from scripts.experiments import run_phase39_causal_direct as direct  # noqa: E402
from scripts.plotting import plot_phase39_causal_released_moment as base  # noqa: E402
from src.training.train import _build_stf_rate_criterion, _prepare_v2_batch  # noqa: E402
from src.utils.config_v2 import validate_config_on_startup  # noqa: E402
from src.utils.provenance import sha256_file  # noqa: E402

FLOOR_NM = 1.0e15
ANCHORS = (30, 60, 90, 120, 160, 200)
MIN_WINDOW_SEC = 5.0
REPLAYS = {"new3": base.VARIANTS["new3"]["replay"], "old": base.VARIANTS["old"]["replay"]}


def mag(moment_nm: np.ndarray) -> np.ndarray:
    return (2.0 / 3.0) * (np.log10(np.maximum(moment_nm, FLOOR_NM)) - 9.1)


def extract_reference(replay_root: Path, output_path: Path) -> dict[str, np.ndarray]:
    """Rebuild the validation cohort from the replay's config snapshot and dump the label STF."""
    summary = direct._read_json(replay_root / "summary.json")
    config_path = Path(str(summary["config_path"]))
    if sha256_file(config_path) != str(summary["config_sha256"]):
        raise ValueError("config snapshot hash changed")
    if summary["split_assignment_sha256"] != base.EXPECTED_SPLIT_SHA256:
        raise ValueError("split assignment changed")
    config = direct._read_yaml(config_path)
    validate_config_on_startup(config)
    loader, _, _ = evaluator._load_cohort(config, base.EXPECTED_SPLIT_SHA256, "validation")
    device = torch.device("cpu")
    criterion = _build_stf_rate_criterion(config, device)
    events, stations, rates, taus, dts = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            prepared = _prepare_v2_batch(batch, config, device)
            rates.append(prepared.stf_true.cpu().numpy().astype(np.float64))
            taus.append(criterion.travel_time.delays(prepared.source_distance_m).p_sec.reshape(-1).cpu().numpy())
            dts.append(prepared.source_dt_sec.reshape(-1).cpu().numpy())
            events.extend(str(e) for e in batch["event"])
            stations.extend(str(s) for s in batch["station"])
    result = {
        "events": np.array(events),
        "stations": np.array(stations),
        "stf_ref_over_m_ref": (np.concatenate(rates) / float(criterion.stf_m_ref)).astype(np.float32),
        "m_ref_nm": np.float64(criterion.stf_m_ref),
        "p_arrival_sec": np.concatenate(taus).astype(np.float32),
        "source_dt_sec": np.concatenate(dts).astype(np.float32),
        "config_sha256": np.array(str(summary["config_sha256"])),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)
    return result


def _window_metrics(pred: np.ndarray, ref: np.ndarray, window_sec: float, dt: float) -> tuple[float, float]:
    steps = pred.shape[0]
    mask = np.clip(window_sec / dt - np.arange(steps), 0.0, 1.0)
    dmw = float(mag((pred * mask).sum() * dt) - mag((ref * mask).sum() * dt))
    n = int(np.ceil(window_sec / dt))
    if n < MIN_WINDOW_SEC / dt or ref[:n].std() == 0 or pred[:n].std() == 0:
        return float("nan"), dmw
    return float(np.corrcoef(pred[:n], ref[:n])[0, 1]), dmw


def agreement(reference: dict[str, np.ndarray], replay_root: Path, variant: str) -> pd.DataFrame:
    npz = np.load(replay_root / "validation_prefix_stf.npz")
    pred_all = npz["stf_over_m_ref"].astype(np.float64) * float(npz["m_ref_nm"])
    horizons = npz["horizons_sec"].astype(int)
    ref_all = reference["stf_ref_over_m_ref"].astype(np.float64) * float(reference["m_ref_nm"])
    ref_index = {(e, s): i for i, (e, s) in enumerate(zip(reference["events"], reference["stations"]))}
    rows = []
    for i, (event, station) in enumerate(zip(npz["events"], npz["stations"])):
        j = ref_index[(str(event), str(station))]
        ref = ref_all[j]
        tau = float(reference["p_arrival_sec"][j])
        dt = float(reference["source_dt_sec"][j])
        for h in ANCHORS:
            pred = pred_all[i, int(np.flatnonzero(horizons == h)[0])]
            window = float(np.clip(h - tau, 0.0, pred.shape[0] * dt))
            r, dmw = _window_metrics(pred, ref, window, dt)
            row = {"variant": variant, "event": str(event), "station": str(station), "horizon_sec": h,
                   "p_arrival_sec": tau, "constrained_window_sec": window,
                   "shape_r_window": r, "delta_mw_window": dmw}
            if h == 200:
                r_full, dmw_full = _window_metrics(pred, ref, pred.shape[0] * dt, dt)
                row.update({"shape_r_full": r_full, "delta_mw_full": dmw_full})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(stations: pd.DataFrame) -> pd.DataFrame:
    valid = stations[stations["constrained_window_sec"] >= MIN_WINDOW_SEC]
    grouped = valid.groupby(["variant", "horizon_sec", "event"], as_index=False).agg(
        n_stations=("station", "size"),
        median_shape_r=("shape_r_window", "median"),
        median_delta_mw=("delta_mw_window", "median"),
        mae_delta_mw=("delta_mw_window", lambda x: float(np.abs(x).mean())),
    )
    overall = valid.groupby(["variant", "horizon_sec"], as_index=False).agg(
        n_stations=("station", "size"),
        median_shape_r=("shape_r_window", "median"),
        median_delta_mw=("delta_mw_window", "median"),
        mae_delta_mw=("delta_mw_window", lambda x: float(np.abs(x).mean())),
    )
    overall["event"] = "ALL"  # station-weighted: Noto (397/446) dominates
    per_event = grouped.groupby(["variant", "horizon_sec"], as_index=False).agg(
        n_stations=("n_stations", "sum"),
        median_shape_r=("median_shape_r", "median"),
        median_delta_mw=("median_delta_mw", "median"),
        mae_delta_mw=("median_delta_mw", lambda x: float(np.abs(x).mean())),
    )
    per_event["event"] = "EVENT_MEDIAN"  # one vote per event
    return pd.concat([grouped, overall, per_event], ignore_index=True)[
        ["variant", "horizon_sec", "event", "n_stations", "median_shape_r", "median_delta_mw", "mae_delta_mw"]]


def run(output_dir: Path) -> dict[str, Any]:
    analysis_dir = output_dir / "analysis"
    ref_path = analysis_dir / "validation_reference_stf.npz"
    reference = extract_reference(REPLAYS["new3"], ref_path)
    frames = [agreement(reference, root, variant) for variant, root in REPLAYS.items()]
    stations = pd.concat(frames, ignore_index=True)
    summary = summarize(stations)
    stations.to_csv(analysis_dir / "stf_agreement_stations.csv", index=False, lineterminator="\n")
    summary.to_csv(analysis_dir / "stf_agreement_summary.csv", index=False, lineterminator="\n")
    return {"reference": str(ref_path), "records": int(len(reference["events"])),
            "summary_all": summary[summary["event"] == "ALL"].to_dict(orient="records")}


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
