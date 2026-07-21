import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = "/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/PINN_Mag/dataset/gnss_events_matched.gcmt.npz"
STAS = ["P095", "MODB", "P340", "P127", "P217", "P200"]

d = np.load(DATA, allow_pickle=True)
i = list(d["events"]).index("Napa2014")
enu, si = d["enu"][i], d["station_info"][i]
elat, elon = float(d["latitude"][i]), float(d["longitude"][i])

def dist_az(lat2, lon2):
    r1, l1, r2, l2 = map(math.radians, (elat, elon, lat2, lon2))
    dlon = l2 - l1
    h = math.sin((r2 - r1) / 2) ** 2 + math.cos(r1) * math.cos(r2) * math.sin(dlon / 2) ** 2
    dkm = 2 * 6371 * math.asin(math.sqrt(h))
    az = math.atan2(math.sin(dlon) * math.cos(r2), math.cos(r1) * math.sin(r2) - math.sin(r1) * math.cos(r2) * math.cos(dlon))
    return dkm, az

fig, axes = plt.subplots(len(STAS), 1, figsize=(8, 2 * len(STAS)), sharex=True)
for ax, sta in zip(axes, STAS):
    wf = enu[sta]
    r_km, az = dist_az(float(si[sta]["lat"]), float(si[sta]["lon"]))
    t = np.asarray(wf["t"], float)
    rad = (np.asarray(wf["E"], float) * math.sin(az) + np.asarray(wf["N"], float) * math.cos(az)) * 0.1  # mm->cm
    pre = rad[t < r_km / 7.9]
    if pre.size >= 3:
        rad = rad - np.median(pre)
    ax.plot(t, rad, lw=0.7)
    ax.axhline(2, color="r", ls="--", lw=0.5)
    ax.axhline(-2, color="r", ls="--", lw=0.5)
    ax.set_ylabel("cm")
    ax.set_title(f"{sta}  r={r_km:.0f} km", fontsize=9, loc="left")
axes[-1].set_xlabel("time since origin (s)")
fig.suptitle("Napa 2014 (Mw 6.1) radial displacement, suspect stations")
fig.tight_layout()
fig.savefig("outputs_experiments/qc_napa_suspect_waveforms.png", dpi=130)
print("saved")
