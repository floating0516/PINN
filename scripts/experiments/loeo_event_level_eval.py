#!/usr/bin/env python
"""LOEO 事件级汇总：加载每个 fold 已训练的 best_model.pth，对留出事件做纯推断，
保存逐台站预测 (station_predictions.csv) 与事件级汇总 (loeo_event_summary.csv)。

用法:
    python scripts/experiments/loeo_event_level_eval.py \
        --loeo-root ./outputs_experiments/e1_loeo_faronly_lp100 [--folds 5 33]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import get_data_loaders_loeo, list_event_indices
from src.models.model import PINNModel
from src.training.physics import PhysicsLoss
from src.utils.device import get_preferred_device


def _ensure_time_steps(x: torch.Tensor, time_steps: int) -> torch.Tensor:
    if x.size(-1) == time_steps:
        return x
    if x.size(-1) > time_steps:
        return x[..., :time_steps]
    pad = time_steps - x.size(-1)
    return torch.nn.functional.pad(x, (0, pad))


def eval_fold(fold_dir: Path, ev_idx: int, device) -> list[dict]:
    run_dirs = sorted((fold_dir / "models").iterdir())
    run_dir = run_dirs[-1]
    with open(run_dir / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ds_cfg = config.get("dataset", {}) or {}
    stf_m_ref = float(ds_cfg.get("stf_m_ref", 1.0e18))
    time_steps = int(config["training"].get("time_steps", 250))

    _, _, test_loader = get_data_loaders_loeo(config, leave_out_event_index=ev_idx)

    model = PINNModel(config).to(device)
    state = torch.load(run_dir / "best_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    criterion = PhysicsLoss(config).to(device)

    rows: list[dict] = []
    with torch.no_grad():
        for batch in test_loader:
            radial = _ensure_time_steps(batch["radial"].to(device), time_steps)
            B = int(radial.size(0))
            distance_m = batch.get("distance", None)
            theta_deg = batch.get("theta_deg", torch.tensor(45.0)).to(device)
            phi_deg = batch.get("phi_deg", torch.tensor(0.0)).to(device)
            dt_val = batch["dt"].mean().item()
            if distance_m is not None:
                dist_log = torch.log(distance_m.to(device).view(-1).clamp(min=1.0))
            else:
                dist_log = torch.zeros(B, device=device)
            theta_r = torch.deg2rad(theta_deg.view(-1))
            phi_r = torch.deg2rad(phi_deg.view(-1))
            meta = torch.stack(
                [dist_log, torch.sin(theta_r), torch.cos(theta_r), torch.sin(phi_r), torch.cos(phi_r)],
                dim=1,
            )
            rate_log = model(radial, meta=meta)
            dot_m0 = torch.clamp(stf_m_ref * (torch.pow(10.0, rate_log) - 1.0), min=0.0)
            mw_pred = criterion.utils.magnitude_from_rate(dot_m0, dt_val).cpu().numpy().flatten()

            stf_true = batch.get("stf", None)
            mw_stf = None
            if stf_true is not None and torch.is_tensor(stf_true):
                mw_stf = criterion.utils.magnitude_from_rate(stf_true.to(device), dt_val).cpu().numpy().flatten()
            has_stf = batch.get("has_stf", None)
            magnitude = batch.get("magnitude", None)
            events = batch.get("event", ["?"] * B)
            stations = batch.get("station", ["?"] * B)
            mechanism = batch.get("mechanism", ["?"] * B)

            for i in range(B):
                mw_cat = float(magnitude.view(-1)[i].item()) if torch.is_tensor(magnitude) else float("nan")
                use_stf = bool(has_stf.view(-1)[i].item()) if torch.is_tensor(has_stf) else (mw_stf is not None)
                mw_true = float(mw_stf[i]) if (use_stf and mw_stf is not None) else mw_cat
                rows.append(
                    {
                        "event": str(events[i]) if not torch.is_tensor(events) else str(events[i].item()),
                        "station": str(stations[i]) if not torch.is_tensor(stations) else str(stations[i].item()),
                        "mechanism": str(mechanism[i]) if not torch.is_tensor(mechanism) else str(mechanism[i].item()),
                        "distance_km": float(distance_m.view(-1)[i].item()) / 1000.0 if torch.is_tensor(distance_m) else float("nan"),
                        "mw_pred": float(mw_pred[i]),
                        "mw_true_stf": float(mw_stf[i]) if mw_stf is not None else float("nan"),
                        "mw_catalog": mw_cat,
                        "mw_true_used": mw_true,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loeo-root", type=str, required=True)
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="only these event indices")
    args = parser.parse_args()

    root = Path(args.loeo_root)
    device = get_preferred_device()
    print(f"device: {device}")

    fold_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("fold_"))
    summary_rows = []
    sp_path = root / "station_predictions.csv"
    ev_path = root / "loeo_event_summary.csv"
    sp_f = open(sp_path, "w", newline="", encoding="utf-8")
    sp_writer = None

    for fold_dir in fold_dirs:
        m = re.match(r"fold_(\d+)_(.+)", fold_dir.name)
        ev_idx = int(m.group(1))
        ev_name = m.group(2)
        if args.folds is not None and ev_idx not in args.folds:
            continue
        print(f"=== fold {ev_idx} {ev_name}")
        try:
            rows = eval_fold(fold_dir, ev_idx, device)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            summary_rows.append({"event_index": ev_idx, "event": ev_name, "error": str(exc)})
            continue
        for r in rows:
            r["event_index"] = ev_idx
            if sp_writer is None:
                sp_writer = csv.DictWriter(sp_f, fieldnames=list(r.keys()))
                sp_writer.writeheader()
            sp_writer.writerow(r)
        sp_f.flush()

        preds = np.array([r["mw_pred"] for r in rows])
        mw_cat = np.nanmedian([r["mw_catalog"] for r in rows])
        med = float(np.median(preds))
        err = med - float(mw_cat)
        station_mae = float(np.mean(np.abs(preds - mw_cat)))
        summary_rows.append(
            {
                "event_index": ev_idx,
                "event": ev_name,
                "mw_catalog": float(mw_cat),
                "mw_pred_median": med,
                "event_error": err,
                "abs_event_error": abs(err),
                "n_stations": len(rows),
                "pred_std": float(np.std(preds)),
                "pred_iqr": float(np.percentile(preds, 75) - np.percentile(preds, 25)),
                "station_mae_vs_catalog": station_mae,
                "error": "",
            }
        )
        print(f"  n={len(rows)} mw_cat={mw_cat:.2f} pred_med={med:.3f} err={err:+.3f}")

    sp_f.close()
    with open(ev_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["event_index", "event", "mw_catalog", "mw_pred_median", "event_error",
                      "abs_event_error", "n_stations", "pred_std", "pred_iqr",
                      "station_mae_vs_catalog", "error"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    ok = [r for r in summary_rows if r.get("abs_event_error") is not None and not r.get("error")]
    if ok:
        errs = np.array([r["abs_event_error"] for r in ok])
        print(f"\nfolds ok: {len(ok)}  event-level MAE={errs.mean():.4f} median={np.median(errs):.4f} max={errs.max():.4f}")
    print(f"saved: {sp_path}\nsaved: {ev_path}")


if __name__ == "__main__":
    main()
