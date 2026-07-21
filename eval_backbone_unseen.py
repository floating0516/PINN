#!/usr/bin/env python
"""Evaluate the 3 backbone-ablation models on EXACTLY the 8 unseen events
at cm0/cm1/cm2. Produces PINN + PGD (crowell/melgar/ruhl) per-event errors so
we can build both the backbone-ablation table and the same-cohort PGD table.
"""
import sys, csv, math
from pathlib import Path

ROOT = Path("/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/PINN_Mag/demo")
sys.path.insert(0, str(ROOT))
from src.evaluation.evaluate_unseen import evaluate_unseen_events

GNSS = Path("/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/GNSS_EQDATA")
UNSEEN8 = [
    GNSS / "iquique-aftershock-2014-chile",
    GNSS / "nepal-aftershock-2015",
    GNSS / "kodiak-2018-alaska",
    GNSS / "samos-2020-greece",
    GNSS / "luding-2022-china",
    GNSS / "xizang-2025-southern-tibetan-plateau",
    GNSS / "myanmar-2025-mandalay",
    GNSS / "sand-point-2025-alaska",
]
MODELS = {
    "hybrid":           ROOT / "outputs_experiments/e1_backbone/models/20260623_100405",
    "tcn_only":         ROOT / "outputs_experiments/e1_backbone/models/20260623_102522",
    "transformer_only": ROOT / "outputs_experiments/e1_backbone/models/20260623_103803",
}
THRESH = {"cm0": 0.0, "cm1": 1.0, "cm2": 2.0}

def mae(xs): return sum(abs(x) for x in xs) / len(xs)
def rmse(xs): return math.sqrt(sum(x*x for x in xs) / len(xs))

out_root = ROOT / "outputs_experiments/e1_backbone/unseen_eval"
rows = []
for mname, mdir in MODELS.items():
    for tlabel, tcm in THRESH.items():
        odir = out_root / mname / tlabel
        odir.mkdir(parents=True, exist_ok=True)
        evaluate_unseen_events(event_dirs=[str(d) for d in UNSEEN8],
                               model_dir=str(mdir), output_dir=str(odir),
                               radial_peak_min_cm_override=tcm)
        summ = odir / "event_summary.csv"
        rr = list(csv.DictReader(open(summ)))
        pinn = [float(r["error"]) for r in rr]
        rec = {"model": mname, "thr": tlabel, "n": len(rr),
               "pinn_mae": mae(pinn), "pinn_rmse": rmse(pinn)}
        for m in ("crowell", "melgar", "ruhl"):
            errs = [float(r[f"pgd_{m}_error"]) for r in rr if r.get(f"pgd_{m}_error") not in (None, "")]
            rec[f"{m}_mae"] = mae(errs) if errs else float("nan")
        rows.append(rec)
        print(f"DONE {mname:16s} {tlabel} n={len(rr)} pinn_mae={rec['pinn_mae']:.4f}")

print("\n==== BACKBONE ABLATION (unseen MAE) ====")
print(f"{'model':16s} {'cm0':>7s} {'cm1':>7s} {'cm2':>7s}")
for mname in MODELS:
    vals = {r["thr"]: r["pinn_mae"] for r in rows if r["model"] == mname}
    print(f"{mname:16s} {vals['cm0']:7.4f} {vals['cm1']:7.4f} {vals['cm2']:7.4f}")

print("\n==== PINN (hybrid) vs PGD same-cohort (unseen MAE) ====")
print(f"{'thr':5s} {'PINN':>7s} {'Crowell':>8s} {'Melgar':>8s} {'Ruhl':>8s}")
for tlabel in THRESH:
    r = next(x for x in rows if x["model"] == "hybrid" and x["thr"] == tlabel)
    print(f"{tlabel:5s} {r['pinn_mae']:7.4f} {r['crowell_mae']:8.4f} {r['melgar_mae']:8.4f} {r['ruhl_mae']:8.4f}")

with open(out_root / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\nwrote {out_root/'summary.csv'}")
