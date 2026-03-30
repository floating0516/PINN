"""
正确的 EEW_0012 基线评估：直接从原始 NPZ 加载数据，最小化预处理。

关键差异（vs 当前 evaluate.py 中的基线）：
1. 使用原始 mm 位移数据，转换为 m（不做中值居中）
2. 减去 P 波到达前的均值作为基线校正（而非整体中值）
3. 使用完整时间窗口（不截断为 200 步）
4. 对每个事件取台站中值（与论文一致）
5. 与目录震级比较（而非 STF 积分震级）
"""
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.baseline import Baseline


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def azimuth_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def evaluate_baseline_raw(
    npz_path: str = r"F:\dataset_catalog\gnss_events_matched.gcmt.npz",
    rho: float = 3400.0,
    alpha: float = 7900.0,
    beta: float = 4533.0,
    max_window_sec: float = 200.0,
    max_dist_km: float = 800.0,
    min_peak_cm: float = 5.0,
    apply_rp: bool = False,
    include_intermediate_field: bool = True,
    blacklist: list[str] | None = None,
):
    """评估基线方法，直接从原始 NPZ 读取并做最小预处理"""
    if blacklist is None:
        blacklist = [
            "N.Honshu2011", "N.Honshu2012", "N.Honshu2013", "E.Fukushima2011",
            "Iwate2011",  # 与 Tohoku2011 是同一事件（相同坐标和震级）
        ]
    blacklist_set = set(blacklist)

    data = np.load(npz_path, allow_pickle=True)
    events = data["events"]
    magnitudes = data["magnitude"]
    event_lats = data["latitude"]
    event_lons = data["longitude"]
    depths_km = data["depth_km"]
    mechanisms = data.get("mechanism", None)
    strikes = data.get("strike", None)
    dips = data.get("dip", None)
    rakes = data.get("rake", None)
    enu_all = data["enu"]
    station_info_all = data["station_info"]

    baseline = Baseline(rho=rho, alpha=alpha, beta=beta)

    # Per-event results
    event_results = {}
    total_stations = 0
    skipped_stations = 0

    for idx in range(len(events)):
        ev_name = str(events[idx])
        if ev_name in blacklist_set:
            continue
        ev_mag = float(magnitudes[idx])
        ev_lat = float(event_lats[idx])
        ev_lon = float(event_lons[idx])
        ev_depth = float(depths_km[idx])
        ev_mech = str(mechanisms[idx]) if mechanisms is not None else ""
        ev_strike = float(strikes[idx]) if strikes is not None else float("nan")
        ev_dip = float(dips[idx]) if dips is not None else float("nan")
        ev_rake = float(rakes[idx]) if rakes is not None else float("nan")

        enu_ev = enu_all[idx]
        st_info = station_info_all[idx]

        station_mwg_list = []

        for st_name, waveforms in enu_ev.items():
            total_stations += 1

            # Get station coordinates
            meta = st_info.get(st_name, {})
            st_lat = float(meta.get("lat", float("nan")))
            st_lon = float(meta.get("lon", float("nan")))
            if not math.isfinite(st_lat) or not math.isfinite(st_lon):
                skipped_stations += 1
                continue

            # Distance and azimuth
            dist_m = haversine_m(ev_lat, ev_lon, st_lat, st_lon)
            dist_km = dist_m / 1000.0
            if dist_km > max_dist_km:
                skipped_stations += 1
                continue
            az = azimuth_deg(ev_lat, ev_lon, st_lat, st_lon)

            # Get waveform data (raw, in mm)
            t = waveforms.get("t", None)
            e_mm = waveforms.get("E", None)
            n_mm = waveforms.get("N", None)
            if t is None or e_mm is None or n_mm is None:
                skipped_stations += 1
                continue
            if len(t) < 10 or len(e_mm) < 10 or len(n_mm) < 10:
                skipped_stations += 1
                continue

            # Check for NaN/Inf
            mask_ok = np.isfinite(t) & np.isfinite(e_mm) & np.isfinite(n_mm)
            if not np.all(mask_ok):
                t = t[mask_ok]
                e_mm = e_mm[mask_ok]
                n_mm = n_mm[mask_ok]
            if len(t) < 10:
                skipped_stations += 1
                continue

            # Estimate dt
            dt = float(np.median(np.diff(t)))
            if dt <= 0 or not math.isfinite(dt):
                dt = 1.0

            # Detect origin: find where t crosses 0 or use t_min
            t_min = float(np.min(t))
            if t_min < 0:
                origin_mask = t >= 0
            else:
                origin_mask = np.ones(len(t), dtype=bool)
            
            t_post = t[origin_mask]
            e_post = e_mm[origin_mask]
            n_post = n_mm[origin_mask]

            # Window: 0 to max_window_sec
            win_mask = (t_post >= 0) & (t_post <= max_window_sec)
            t_win = t_post[win_mask]
            e_win = e_post[win_mask]
            n_win = n_post[win_mask]

            if len(t_win) < 10:
                skipped_stations += 1
                continue

            # Project to radial (mm)
            az_rad = math.radians(az)
            radial_mm = e_win * math.sin(az_rad) + n_win * math.cos(az_rad)

            # Convert mm → m
            radial_m = radial_mm * 0.001

            # Baseline correction: subtract pre-P-wave mean
            # Estimate P arrival time
            p_arrival_sec = dist_m / alpha
            pre_p_mask = t_win < p_arrival_sec
            if np.sum(pre_p_mask) >= 3:
                baseline_val = float(np.mean(radial_m[pre_p_mask]))
            else:
                # Use first 5 seconds or 10% of data
                n_pre = max(int(len(radial_m) * 0.05), 5)
                baseline_val = float(np.mean(radial_m[:n_pre]))
            radial_m = radial_m - baseline_val

            # Check peak displacement (in cm)
            peak_cm = float(np.max(np.abs(radial_m))) * 100.0
            if peak_cm < min_peak_cm:
                skipped_stations += 1
                continue

            # Compute theta (zenith angle) and phi
            if math.isfinite(ev_depth) and ev_depth > 0:
                # Hypocentral distance
                hypo_dist_m = math.sqrt(dist_m**2 + (ev_depth * 1000)**2)
                theta_deg = math.degrees(math.atan2(dist_m, ev_depth * 1000))
            else:
                hypo_dist_m = dist_m
                theta_deg = 80.0  # Default for shallow events

            # phi = angle between slip direction and station (paper definition)
            # Compute slip azimuth from strike/dip/rake (horizontal projection of slip vector)
            if apply_rp and math.isfinite(ev_strike) and math.isfinite(ev_dip) and math.isfinite(ev_rake):
                strike_rad = math.radians(ev_strike)
                dip_rad = math.radians(ev_dip)
                rake_rad = math.radians(ev_rake)
                # Slip vector horizontal projection azimuth
                slip_h_east = math.cos(rake_rad) * math.sin(strike_rad) - math.sin(rake_rad) * math.cos(dip_rad) * math.cos(strike_rad)
                slip_h_north = math.cos(rake_rad) * math.cos(strike_rad) + math.sin(rake_rad) * math.cos(dip_rad) * math.sin(strike_rad)
                slip_azimuth = math.degrees(math.atan2(slip_h_east, slip_h_north)) % 360.0
                phi_deg = (az - slip_azimuth) % 360.0
            else:
                phi_deg = 0.0  # no-RP mode: phi is unused (all coefficients=1)

            # Apply EEW_0012 formula
            u_hr = torch.tensor(radial_m, dtype=torch.float32).unsqueeze(0)  # (1, T)
            r_t = torch.tensor([hypo_dist_m], dtype=torch.float32)
            theta_t = torch.tensor([theta_deg], dtype=torch.float32)
            phi_t = torch.tensor([phi_deg], dtype=torch.float32)

            mwg = baseline.calculate_mwg(
                u_hr=u_hr,
                r_m=r_t,
                theta_deg=theta_t,
                phi_deg=phi_t,
                dt=dt,
                apply_radiation_pattern=apply_rp,
                include_intermediate_field=include_intermediate_field,
            )
            mwg_val = float(mwg[0].item())
            if math.isfinite(mwg_val):
                station_mwg_list.append(mwg_val)

        if len(station_mwg_list) > 0:
            median_mwg = float(np.median(station_mwg_list))
            iqr = float(np.percentile(station_mwg_list, 75) - np.percentile(station_mwg_list, 25)) if len(station_mwg_list) > 1 else 0.0
            event_results[ev_name] = {
                "mw": ev_mag,
                "mwg": median_mwg,
                "iqr": iqr,
                "n_stations": len(station_mwg_list),
                "mechanism": ev_mech,
                "depth_km": ev_depth,
                "all_mwg": station_mwg_list,
            }

    # Print results
    print("=" * 100)
    print(f"EEW_0012 Baseline Evaluation (RP={'ON' if apply_rp else 'OFF'})")
    print(f"Total stations: {total_stations}, Skipped: {skipped_stations}")
    print("=" * 100)
    print(f"{'Event':<25} {'Mech':<12} {'Mw':>5} {'Mwg':>6} {'±IQR':>6} {'Error':>7} {'N':>4}")
    print("-" * 100)

    all_errors = []
    for ev_name in sorted(event_results.keys()):
        r = event_results[ev_name]
        error = r["mwg"] - r["mw"]
        all_errors.append(error)
        print(
            f"{ev_name:<25} {r['mechanism']:<12} "
            f"{r['mw']:5.1f} {r['mwg']:6.2f} ±{r['iqr']:5.2f} "
            f"{error:+7.2f} {r['n_stations']:4d}"
        )

    all_errors = np.array(all_errors)
    print("-" * 100)
    print(f"{'OVERALL (per-event)':<25} {'':12} {'':5} {'':6} {'':6} "
          f"MAE={np.mean(np.abs(all_errors)):.3f} RMSE={np.sqrt(np.mean(all_errors**2)):.3f} "
          f"N={len(all_errors)}")
    print(f"{'':25} {'':12} {'':5} {'':6} {'':6} "
          f"Mean={np.mean(all_errors):+.3f} Std={np.std(all_errors):.3f}")
    
    # Also compute per-sample metrics
    all_sample_errors = []
    for ev_name, r in event_results.items():
        for mwg_val in r["all_mwg"]:
            all_sample_errors.append(mwg_val - r["mw"])
    all_sample_errors = np.array(all_sample_errors)
    print(f"\n{'OVERALL (per-sample)':<25} N={len(all_sample_errors)} "
          f"MAE={np.mean(np.abs(all_sample_errors)):.3f} "
          f"RMSE={np.sqrt(np.mean(all_sample_errors**2)):.3f} "
          f"Mean={np.mean(all_sample_errors):+.3f}")

    return event_results


if __name__ == "__main__":
    print("\n=== WITHOUT Radiation Pattern ===")
    results_no_rp = evaluate_baseline_raw(apply_rp=False)
    
    print("\n\n=== WITH Radiation Pattern ===")
    results_rp = evaluate_baseline_raw(apply_rp=True)
