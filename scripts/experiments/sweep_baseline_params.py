"""
参数扫描：寻找最优的 window / peak_threshold 组合。
评估指标：per-event MAE（无辐射花型修正）。
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.baseline import Baseline


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def azimuth_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def load_data(npz_path=r"F:\dataset_catalog\gnss_events_matched.gcmt.npz"):
    """预加载所有台站数据，加速后续参数扫描"""
    data = np.load(npz_path, allow_pickle=True)
    blacklist = {"N.Honshu2011", "N.Honshu2012", "N.Honshu2013", "E.Fukushima2011"}

    rho, alpha, beta = 3400.0, 7900.0, 4500.0
    records = []  # list of dicts per station

    for idx in range(len(data["events"])):
        ev_name = str(data["events"][idx])
        if ev_name in blacklist:
            continue
        ev_mag = float(data["magnitude"][idx])
        ev_lat = float(data["latitude"][idx])
        ev_lon = float(data["longitude"][idx])
        ev_depth = float(data["depth_km"][idx])

        enu_ev = data["enu"][idx]
        st_info = data["station_info"][idx]

        for st_name, waveforms in enu_ev.items():
            meta = st_info.get(st_name, {})
            st_lat = float(meta.get("lat", float("nan")))
            st_lon = float(meta.get("lon", float("nan")))
            if not math.isfinite(st_lat) or not math.isfinite(st_lon):
                continue

            dist_m = haversine_m(ev_lat, ev_lon, st_lat, st_lon)
            dist_km = dist_m / 1000.0
            if dist_km > 800:
                continue
            az = azimuth_deg(ev_lat, ev_lon, st_lat, st_lon)

            t = waveforms.get("t", None)
            e_mm = waveforms.get("E", None)
            n_mm = waveforms.get("N", None)
            if t is None or e_mm is None or n_mm is None:
                continue
            if len(t) < 10:
                continue

            mask = np.isfinite(t) & np.isfinite(e_mm) & np.isfinite(n_mm)
            if not np.all(mask):
                t, e_mm, n_mm = t[mask], e_mm[mask], n_mm[mask]
            if len(t) < 10:
                continue

            dt = float(np.median(np.diff(t)))
            if dt <= 0 or not math.isfinite(dt):
                dt = 1.0

            # Post-origin
            if float(np.min(t)) < 0:
                m = t >= 0
            else:
                m = np.ones(len(t), dtype=bool)
            t_post, e_post, n_post = t[m], e_mm[m], n_mm[m]

            # Project to radial and convert mm→m
            az_rad = math.radians(az)
            radial_m = (e_post * math.sin(az_rad) + n_post * math.cos(az_rad)) * 0.001

            # Pre-P baseline subtraction
            p_arr = dist_m / alpha
            pre_p = t_post < p_arr
            if np.sum(pre_p) >= 3:
                bl = float(np.mean(radial_m[pre_p]))
            else:
                bl = float(np.mean(radial_m[:max(5, int(len(radial_m)*0.05))]))
            radial_m = radial_m - bl

            # Hypocentral distance and theta
            if math.isfinite(ev_depth) and ev_depth > 0:
                hypo = math.sqrt(dist_m**2 + (ev_depth*1000)**2)
                theta = math.degrees(math.atan2(dist_m, ev_depth*1000))
            else:
                hypo = dist_m
                theta = 80.0

            records.append({
                "event": ev_name,
                "mw": ev_mag,
                "radial_m": radial_m,
                "t_post": t_post,
                "dt": dt,
                "hypo_m": hypo,
                "theta_deg": theta,
                "phi_deg": az,
                "depth_km": ev_depth,
            })

    print(f"Loaded {len(records)} station records")
    return records


def evaluate_config(
    records,
    max_window_sec=600.0,
    min_peak_cm=2.0,
    use_dual_sign=True,
    use_final_m0=False,  # 如果 True，用最终 M0 而非 peak
):
    """用给定参数评估所有台站"""
    baseline = Baseline(rho=3400.0, alpha=7900.0, beta=4500.0)
    event_mwg = {}  # event -> list of mwg

    for rec in records:
        # Window
        t_post = rec["t_post"]
        radial_m = rec["radial_m"]
        win = (t_post >= 0) & (t_post <= max_window_sec)
        r_win = radial_m[win]
        if len(r_win) < 10:
            continue

        # Peak check
        peak_cm = float(np.max(np.abs(r_win))) * 100.0
        if peak_cm < min_peak_cm:
            continue

        u_t = torch.tensor(r_win, dtype=torch.float32).unsqueeze(0)
        r_t = torch.tensor([rec["hypo_m"]], dtype=torch.float32)
        theta_t = torch.tensor([rec["theta_deg"]], dtype=torch.float32)
        phi_t = torch.tensor([rec["phi_deg"]], dtype=torch.float32)
        dt = rec["dt"]

        if use_dual_sign:
            # Let calculate_mwg handle dual sign
            mwg = baseline.calculate_mwg(u_t, r_t, theta_t, phi_t, dt,
                                         apply_radiation_pattern=False)
        else:
            m0 = baseline.calculate_seismic_moment(u_t, r_t, theta_t, phi_t, dt,
                                                   apply_radiation_pattern=False)
            mwg = baseline.calculate_moment_magnitude(m0)

        v = float(mwg[0].item())
        if math.isfinite(v):
            ev = rec["event"]
            if ev not in event_mwg:
                event_mwg[ev] = {"mw": rec["mw"], "vals": []}
            event_mwg[ev]["vals"].append(v)

    # Per-event metrics
    errors = []
    for ev, d in event_mwg.items():
        median_mwg = float(np.median(d["vals"]))
        errors.append(median_mwg - d["mw"])
    errors = np.array(errors)
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mean_err = float(np.mean(errors))
    return mae, rmse, mean_err, len(errors), event_mwg


def main():
    records = load_data()
    print()

    # Parameter grid
    windows = [200, 300, 400, 500, 600]
    peaks = [2.0, 3.0, 5.0, 8.0]
    dual_signs = [True, False]

    print(f"{'Window':>6} {'Peak':>5} {'Dual':>5} {'MAE':>6} {'RMSE':>6} {'Mean':>7} {'N':>3}")
    print("-" * 50)

    best_mae = 999
    best_cfg = None

    for win in windows:
        for pk in peaks:
            for ds in dual_signs:
                mae, rmse, mean_err, n, _ = evaluate_config(
                    records,
                    max_window_sec=win,
                    min_peak_cm=pk,
                    use_dual_sign=ds,
                )
                tag = "*" if mae < best_mae else " "
                if mae < best_mae:
                    best_mae = mae
                    best_cfg = (win, pk, ds)
                print(f"{win:6d} {pk:5.1f} {str(ds):>5} {mae:6.3f} {rmse:6.3f} {mean_err:+7.3f} {n:3d} {tag}")

    print(f"\nBest: window={best_cfg[0]}s, peak={best_cfg[1]}cm, dual_sign={best_cfg[2]}, MAE={best_mae:.3f}")

    # Print per-event detail for best config
    print(f"\n{'='*90}")
    print(f"Best config detail: window={best_cfg[0]}s, peak={best_cfg[1]}cm, dual_sign={best_cfg[2]}")
    print(f"{'='*90}")
    _, _, _, _, evr = evaluate_config(
        records, max_window_sec=best_cfg[0], min_peak_cm=best_cfg[1], use_dual_sign=best_cfg[2]
    )
    print(f"{'Event':<25} {'Mw':>5} {'Mwg':>6} {'Error':>7} {'N':>4}")
    print("-" * 50)
    for ev in sorted(evr.keys()):
        d = evr[ev]
        med = float(np.median(d["vals"]))
        err = med - d["mw"]
        print(f"{ev:<25} {d['mw']:5.1f} {med:6.2f} {err:+7.2f} {len(d['vals']):4d}")


if __name__ == "__main__":
    main()
