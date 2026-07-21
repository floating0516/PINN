"""最终评估：排除重复事件 Iwate2011，使用最优参数"""
import sys
import math
sys.path.insert(0, "f:\\more_eq")
import numpy as np
from scripts.experiments.sweep_baseline_params import load_data, evaluate_config


def evaluate_with_exclusion(records, exclude_events, **kwargs):
    """在 evaluate_config 外部过滤事件"""
    filtered = [r for r in records if r["event"] not in exclude_events]
    return evaluate_config(filtered, **kwargs)


records = load_data()
exclude = {"Iwate2011"}  # 与 Tohoku2011 是同一事件

configs = [
    ("200s / 5cm / dual", dict(max_window_sec=200, min_peak_cm=5.0, use_dual_sign=True)),
    ("200s / 3cm / dual", dict(max_window_sec=200, min_peak_cm=3.0, use_dual_sign=True)),
    ("200s / 8cm / dual", dict(max_window_sec=200, min_peak_cm=8.0, use_dual_sign=True)),
    ("300s / 5cm / dual", dict(max_window_sec=300, min_peak_cm=5.0, use_dual_sign=True)),
]

print("Excluding Iwate2011 (duplicate of Tohoku2011)")
print(f"{'Config':<25} {'MAE':>6} {'RMSE':>6} {'Mean':>7} {'N':>3}")
print("-" * 55)
for label, kw in configs:
    mae, rmse, mean_err, n, _ = evaluate_with_exclusion(records, exclude, **kw)
    print(f"{label:<25} {mae:6.3f} {rmse:6.3f} {mean_err:+7.3f} {n:3d}")

# Detail for best
print("\n" + "=" * 80)
print("Detail: 200s / 5cm / dual (excluding Iwate2011)")
print("=" * 80)
mae, rmse, mean_err, n, evr = evaluate_with_exclusion(
    records, exclude, max_window_sec=200, min_peak_cm=5.0, use_dual_sign=True
)
print(f"MAE={mae:.3f}, RMSE={rmse:.3f}, Mean={mean_err:+.3f}, N={n}\n")

header = f"{'Event':<25} {'Mw':>5} {'Mwg':>6} {'Error':>7} {'N':>4} {'IQR':>5}"
print(header)
print("-" * 60)

within_02, within_03, within_05 = 0, 0, 0
for ev in sorted(evr.keys()):
    d = evr[ev]
    med = float(np.median(d["vals"]))
    err = med - d["mw"]
    iqr = (
        float(np.percentile(d["vals"], 75) - np.percentile(d["vals"], 25))
        if len(d["vals"]) > 1
        else 0.0
    )
    tag = "" if abs(err) <= 0.3 else ("*" if abs(err) <= 0.5 else "**")
    if abs(err) <= 0.2: within_02 += 1
    if abs(err) <= 0.3: within_03 += 1
    if abs(err) <= 0.5: within_05 += 1
    print(f"{ev:<25} {d['mw']:5.1f} {med:6.2f} {err:+7.2f} {len(d['vals']):4d} {iqr:5.2f} {tag}")

print(f"\nWithin +/-0.2: {within_02}/{n} ({100*within_02/n:.0f}%)")
print(f"Within +/-0.3: {within_03}/{n} ({100*within_03/n:.0f}%)")
print(f"Within +/-0.5: {within_05}/{n} ({100*within_05/n:.0f}%)")
