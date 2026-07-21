"""Catalog-wide QC scan: flag stations whose peak radial displacement is
implausibly large for the event magnitude and epicentral distance.

For each event/station:
  - project E/N to the radial component (source-to-station azimuth)
  - debias by the pre-P median (P travel time ~ r/alpha), low-pass 0.2 Hz FIR
  - peak |radial| within the first 200 s after origin
  - expected PGD from the Melgar et al. (2015) scaling law
  - flag if peak >= 2 cm (training threshold) AND peak >= RATIO x expected
Also reports pre-event noise level (std of samples before P).

Output: outputs_experiments/qc_station_noise_scan.csv (all stations) and a
console summary of flagged stations per event.
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np

DATA = "/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/PINN_Mag/dataset/gnss_events_matched.gcmt.npz"
OUT = "outputs_experiments/qc_station_noise_scan.csv"
ALPHA_KM_S = 7.9
WINDOW_S = 200.0
RATIO = 5.0
PEAK_MIN_M = 0.02

MELGAR = (-4.434, 1.047, -0.138)  # log10(PGD_cm) = a + b*Mw + c*Mw*log10(R_km)


def expected_pgd_m(mw: float, r_km: float) -> float:
    a, b, c = MELGAR
    return 10.0 ** (a + b * mw + c * mw * math.log10(max(r_km, 1.0))) / 100.0


def dist_az(lat1, lon1, lat2, lon2):
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = rlat2 - rlat1, rlon2 - rlon1
    h = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    d_km = 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))
    az = math.atan2(
        math.sin(dlon) * math.cos(rlat2),
        math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon),
    )
    return d_km, az


def lowpass(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import filtfilt, firwin

        taps = firwin(7, 0.2, window="hamming", fs=1.0)
        return filtfilt(taps, [1.0], x)
    except Exception:
        return x


def main() -> None:
    d = np.load(DATA, allow_pickle=True)
    events = list(d["events"])
    rows = []
    flagged_by_event: dict[str, list] = {}
    for i, ev in enumerate(events):
        mw = float(d["magnitude"][i])
        elat, elon = float(d["latitude"][i]), float(d["longitude"][i])
        enu = d["enu"][i]
        si = d["station_info"][i]
        if hasattr(si, "item"):
            si = si.item()
        for sta, wf in enu.items():
            info = si.get(sta)
            if info is None:
                continue
            r_km, az = dist_az(elat, elon, float(info["lat"]), float(info["lon"]))
            t = np.asarray(wf["t"], dtype=float)
            e = np.asarray(wf["E"], dtype=float) * 0.001  # archive is mm (configs: units: mm)
            n = np.asarray(wf["N"], dtype=float) * 0.001
            rad = e * math.sin(az) + n * math.cos(az)
            ok = np.isfinite(rad) & np.isfinite(t)
            t, rad = t[ok], rad[ok]
            if t.size < 10:
                continue
            tp = r_km / ALPHA_KM_S
            pre = rad[t < tp]
            if pre.size >= 3:
                rad = rad - np.median(pre)
                pre_noise = float(np.std(pre - np.median(pre)))
            else:
                rad = rad - rad[0]
                pre_noise = float("nan")
            rad_f = lowpass(rad)
            win = (t >= 0) & (t <= WINDOW_S)
            if not win.any():
                continue
            peak = float(np.max(np.abs(rad_f[win])))
            exp = expected_pgd_m(mw, r_km)
            flag = peak >= PEAK_MIN_M and peak >= RATIO * exp
            rows.append(
                dict(
                    event=ev,
                    mw=round(mw, 2),
                    station=sta,
                    distance_km=round(r_km, 1),
                    peak_radial_m=round(peak, 4),
                    expected_pgd_m=round(exp, 4),
                    ratio=round(peak / exp, 1) if exp > 0 else float("inf"),
                    pre_event_noise_m=round(pre_noise, 4) if math.isfinite(pre_noise) else "",
                    flagged=int(flag),
                )
            )
            if flag:
                flagged_by_event.setdefault(ev, []).append(rows[-1])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"scanned {len(rows)} station records across {len(events)} events")
    print(f"flagged {sum(len(v) for v in flagged_by_event.values())} records "
          f"(peak>={PEAK_MIN_M*100:.0f} cm and >= {RATIO}x Melgar-expected PGD)\n")
    for ev in sorted(flagged_by_event, key=lambda e: -len(flagged_by_event[e])):
        recs = sorted(flagged_by_event[ev], key=lambda r: -r["ratio"])
        print(f"== {ev} (Mw {recs[0]['mw']}): {len(recs)} flagged")
        for r in recs[:8]:
            print(f"   {r['station']:8s} r={r['distance_km']:7.1f} km  "
                  f"peak={r['peak_radial_m']*100:6.2f} cm  exp={r['expected_pgd_m']*100:6.2f} cm  "
                  f"x{r['ratio']:.0f}  noise={r['pre_event_noise_m']}")
    print(f"\nfull table: {OUT}")


if __name__ == "__main__":
    main()
