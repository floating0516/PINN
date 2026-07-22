"""Generate real-data waveform panels for Figure 2 using one real earthquake.

Runs a real event (Illapel 2015, Chile, Mw 8.3, 38 stations) through the exact
preprocessing + trained hybrid model + physics forward pipeline and produces
five compact panels matching the Figure 2 schematic insets:
  (a) multi-station three-component GNSS displacement (E/N/U)
  (b) radial-projected displacement u_r(t)
  (c) predicted STF vs SCARDEC reference STF
  (d) predicted Mw vs catalog Mw
  (e) observed vs physics-synthesized radial waveform

Usage:
  python scripts/plotting/fig2_real_event_panels.py
Outputs go to paper/srl/figure_sources/fig2_real_panels/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate_unseen import (  # noqa: E402
    _station_sample_from_bundle,
    load_event_bundle,
)
from src.data.metadata import build_metadata_tensor  # noqa: E402
from src.data.waveform import waveform_config_from_v2  # noqa: E402
from src.evaluation.evaluate import _ensure_time_steps  # noqa: E402
from src.models.model import PINNModel  # noqa: E402
from src.training.physics import PhysicsLoss  # noqa: E402
from src.training.loss_stf_rate import (  # noqa: E402
    STFRateWaveformLoss,
    compute_physical_coefficients,
    compute_radiation_coefficients,
    forward_displacement_from_rate,
)
from src.utils.device import get_preferred_device  # noqa: E402

GNSS = Path("/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/GNSS_EQDATA")
EVENT_DIR = GNSS / "illapel-2015-chile"
MODEL_DIR = ROOT / "outputs_experiments/e1_backbone/models/20260623_100405"
SCARDEC = Path(
    "/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/PINN_Mag/dataset/STF_SCARDEC/Illapel2015.stf"
)
OUT = ROOT / "paper/srl/figure_sources/fig2_real_panels"
OUT.mkdir(parents=True, exist_ok=True)

# muted palette matching current fig2
C_BLUE = "#5B7FA6"
C_PURPLE = "#7B68A6"
C_ORANGE = "#D9913F"
C_GREEN = "#55A87D"
C_RED = "#BB6B60"
C_GREY = "#98A2B3"


def load_scardec_stf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    rows = []
    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) >= 2:
            rows.append((float(parts[0]), float(parts[1])))
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1]


def mini_axes(figsize=(2.2, 1.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def main() -> None:
    with (MODEL_DIR / "config.yaml").open() as f:
        config = yaml.safe_load(f)
    waveform_config = waveform_config_from_v2(config)
    device = get_preferred_device()
    model = PINNModel(config).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=device))
    model.eval()
    criterion = PhysicsLoss(config).to(device)
    synth_loss = STFRateWaveformLoss(config).to(device)
    time_steps = int(config.get("training", {}).get("time_steps", 200))
    dataset_config = config.get("dataset", {}) or {}
    stf_m_ref = float(
        (dataset_config.get("stf", {}) or {}).get(
            "m_ref",
            dataset_config.get("stf_m_ref", 1.0e18),
        )
    )

    bundle = load_event_bundle(EVENT_DIR)
    print(f"Event: {bundle.event_name}  Mw={bundle.magnitude}  stations={len(bundle.stations)}")

    results = []
    with torch.no_grad():
        for st in bundle.stations:
            sample = _station_sample_from_bundle(
                bundle,
                st,
                config,
                waveform_config=waveform_config,
            )
            if sample is None:
                continue
            radial = torch.tensor(sample["radial"], dtype=torch.float32, device=device)[None, None, :]
            radial = _ensure_time_steps(radial, time_steps)
            meta = build_metadata_tensor(
                torch.tensor([sample["source_distance_m"]], dtype=torch.float32, device=device),
                torch.tensor([sample["theta_deg"]], dtype=torch.float32, device=device),
                torch.tensor([sample["azimuth_deg"]], dtype=torch.float32, device=device),
            )
            rate_log = model(radial, meta=meta)
            dot_m0 = torch.clamp(stf_m_ref * (torch.pow(10.0, rate_log) - 1.0), min=0.0)
            sample_dt = float(sample["waveform_dt_sec"])
            mw_pred = float(criterion.utils.magnitude_from_rate(dot_m0, sample_dt)[0].item())
            r_m = torch.tensor([sample["source_distance_m"]], dtype=torch.float32, device=device)
            th = torch.tensor([sample["theta_deg"]], dtype=torch.float32, device=device)
            ph = torch.tensor([sample["phi_slip_deg"]], dtype=torch.float32, device=device)
            a_ip, a_is, a_fp, a_fs = compute_radiation_coefficients(
                th, ph, mode=synth_loss.radiation_mode
            )
            c_ip, c_is, c_fp, c_fs = compute_physical_coefficients(
                r_m, synth_loss.rho, synth_loss.alpha, synth_loss.beta,
                a_ip, a_is, a_fp, a_fs,
                geom=synth_loss.geom,
                free_surface=synth_loss.free_surface,
                attenuation=synth_loss.attenuation,
            )
            dt_b = torch.tensor([sample_dt], dtype=torch.float32, device=device)
            synth = forward_displacement_from_rate(
                dot_m0.view(1, -1), dt_b, r_m,
                synth_loss.alpha, synth_loss.beta,
                c_ip, c_is, c_fp, c_fs,
                include_intermediate=synth_loss.include_intermediate,
                skip_delays=synth_loss.skip_travel_delays,
                include_far_P=synth_loss.include_far_P,
                include_far_S=synth_loss.include_far_S,
                include_intermediate_P=synth_loss.include_intermediate_P,
                include_intermediate_S=synth_loss.include_intermediate_S,
            )[0].cpu().numpy()
            results.append(
                dict(
                    station=st.station,
                    st_obj=st,
                    sample=sample,
                    rate=dot_m0[0].view(-1).cpu().numpy(),
                    mw_pred=mw_pred,
                    synth=synth,
                )
            )
            print(f"  {st.station}: dist={sample['source_distance_m']/1e3:.0f} km  "
                  f"peak={sample['radial_peak_cm']:.1f} cm  Mw_pred={mw_pred:.2f}")

    if not results:
        raise SystemExit("no usable stations")

    mw_preds = np.array([r["mw_pred"] for r in results])
    mw_event = float(np.median(mw_preds))
    print(f"Event-level Mw_pred (median of {len(results)} stations) = {mw_event:.3f}  "
          f"catalog = {bundle.magnitude}")

    # representative station: largest radial peak
    rep = max(results, key=lambda r: r["sample"]["radial_peak_cm"])
    print(f"Representative station: {rep['station']}")

    # best physics-fit station: highest correlation between obs and synth
    def _fit_corr(r):
        o = np.asarray(r["sample"]["radial"], dtype=float)
        s = np.asarray(r["synth"], dtype=float)[: len(o)]
        n = min(len(o), len(s))
        if n < 10 or np.std(o[:n]) == 0 or np.std(s[:n]) == 0:
            return -np.inf
        return float(np.corrcoef(o[:n], s[:n])[0, 1])
    fit = max(results, key=_fit_corr)
    print(f"Best physics-fit station: {fit['station']} (corr={_fit_corr(fit):.2f})")

    # ---- combined preview figure ----
    figc, axs = plt.subplots(1, 5, figsize=(16, 2.6), dpi=200)

    # (a) E/N/U of representative station (raw, cm)
    st = rep["st_obj"]
    t = np.asarray(st.t, dtype=float)
    m = (t >= 0) & (t <= 300)
    for arr, lab, off in [(st.e_m, "E", 2), (st.n_m, "N", 0), (st.u_m, "U", -2)]:
        a = np.asarray(arr, dtype=float) * 100.0
        a = a - np.nanmedian(a[m][: max(int(10 / max(np.median(np.diff(t)), 1e-6)), 1)])
        axs[0].plot(t[m], a[m] + off * np.nanmax(np.abs(a[m]) + 1e-9), lw=0.9,
                    label=lab, color={"E": C_BLUE, "N": C_PURPLE, "U": C_GREY}[lab])
    axs[0].set_title(f"(a) E/N/U — {rep['station']}")
    axs[0].legend(fontsize=7, loc="upper right")

    # (b) processed radial
    dt = float(rep["sample"]["waveform_dt_sec"])
    rr = rep["sample"]["radial"]
    tr = np.arange(len(rr)) * dt
    axs[1].plot(tr, rr * 100.0, color=C_BLUE, lw=1.1)
    axs[1].set_title("(b) radial u_r(t) [cm]")

    # (c) predicted STF vs SCARDEC
    rate = rep["rate"]
    ts = np.arange(len(rate)) * dt
    t_ref, f_ref = load_scardec_stf(SCARDEC)
    axs[2].plot(ts, rate / 1e18, color=C_ORANGE, lw=1.2, label="predicted")
    axs[2].plot(t_ref, f_ref / 1e18, color="k", lw=1.0, ls="--", label="SCARDEC")
    axs[2].set_xlim(0, 100)
    axs[2].set_title("(c) STF [1e18 N·m/s]")
    axs[2].legend(fontsize=7)

    # (d) magnitude comparison
    axs[3].bar([0, 1], [bundle.magnitude, mw_event], width=0.55,
               color=[C_GREY, C_ORANGE])
    axs[3].set_xticks([0, 1], ["catalog", "predicted"])
    lo = min(bundle.magnitude, mw_event) - 0.4
    hi = max(bundle.magnitude, mw_event) + 0.3
    axs[3].set_ylim(lo, hi)
    for x, v in [(0, bundle.magnitude), (1, mw_event)]:
        axs[3].text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axs[3].set_title(f"(d) Mw (err {mw_event - bundle.magnitude:+.2f})")

    # (e) observed vs physics-synth radial
    obs = np.asarray(fit["sample"]["radial"], dtype=float)
    syn = np.asarray(fit["synth"], dtype=float)[: len(obs)]
    dt_fit = float(fit["sample"]["waveform_dt_sec"])
    t_fit = np.arange(len(obs)) * dt_fit
    obs_n = obs / (np.nanmax(np.abs(obs)) + 1e-12)
    syn_n = syn / (np.nanmax(np.abs(syn)) + 1e-12)
    axs[4].plot(t_fit, obs_n, color="k", lw=1.0, label="observed")
    axs[4].plot(np.arange(len(syn)) * dt_fit, syn_n, color=C_GREEN, lw=1.1,
                ls="--", label="physics synth")
    axs[4].set_title(f"(e) waveform fit — {fit['station']} (normalized)")
    axs[4].legend(fontsize=7)

    figc.suptitle(
        f"{bundle.event_name} — real data through trained hybrid model "
        f"({len(results)} stations)", fontsize=10)
    figc.tight_layout(rect=[0, 0, 1, 0.93])
    figc.savefig(OUT / "fig2_real_panels_preview.png", bbox_inches="tight")
    figc.savefig(OUT / "fig2_real_panels_preview.pdf", bbox_inches="tight")

    # ---- individual clean insets (no axes) for drawio embedding ----
    # inset A: E/N/U stacked
    fig, ax = mini_axes()
    for arr, off, col in [(st.e_m, 1.0, C_BLUE), (st.n_m, 0.0, C_PURPLE), (st.u_m, -1.0, C_GREY)]:
        a = np.asarray(arr, dtype=float)[m]
        a = (a - np.nanmedian(a[:20]))
        a = a / (np.nanmax(np.abs(a)) + 1e-12)
        ax.plot(t[m], a * 0.45 + off, lw=1.2, color=col)
    fig.savefig(OUT / "inset_enu.png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, ax = mini_axes()
    ax.plot(tr, rr, color=C_BLUE, lw=1.4)
    fig.savefig(OUT / "inset_radial.png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, ax = mini_axes()
    ax.plot(ts, rate, color=C_ORANGE, lw=1.4)
    ref_i = np.interp(ts, t_ref, f_ref, left=0, right=0)
    ax.plot(ts, ref_i, color="k", lw=1.1, ls="--")
    ax.set_xlim(0, 100)
    fig.savefig(OUT / "inset_stf.png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, ax = mini_axes(figsize=(1.4, 1.2))
    ax.bar([0, 1], [bundle.magnitude, mw_event], width=0.6, color=[C_GREY, C_ORANGE])
    ax.set_ylim(lo, hi)
    fig.savefig(OUT / "inset_mag.png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # inset: cumulative Mw(t) curve converging to prediction, catalog as dashed line
    from src.evaluation.evaluate import magnitude_series_from_rate
    rate_t = torch.tensor(rate, dtype=torch.float32)
    mw_series = np.asarray(magnitude_series_from_rate(rate_t, dt), dtype=float)
    mw_series = np.clip(mw_series, 0.0, None)
    fig, ax = mini_axes(figsize=(1.8, 1.2))
    ax.plot(ts, mw_series, color=C_ORANGE, lw=1.5)
    ax.axhline(bundle.magnitude, color="k", lw=1.0, ls="--")
    ax.set_ylim(max(mw_series.min(), bundle.magnitude - 3.0), hi)
    fig.savefig(OUT / "inset_mag_curve.png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, ax = mini_axes()
    ax.plot(t_fit, obs_n, color="k", lw=1.1)
    ax.plot(np.arange(len(syn)) * dt_fit, syn_n, color=C_GREEN, lw=1.3, ls="--")
    fig.savefig(OUT / "inset_fit.png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    np.savez(
        OUT / "fig2_real_panels_data.npz",
        event=bundle.event_name,
        mw_catalog=bundle.magnitude,
        mw_pred_event=mw_event,
        mw_pred_stations=mw_preds,
        stations=np.array([r["station"] for r in results]),
        rep_station=rep["station"],
        fit_station=fit["station"],
        dt=dt,
        radial_obs=obs,
        radial_synth=syn,
        pred_rate=rate,
        scardec_t=t_ref,
        scardec_rate=f_ref,
    )
    print(f"Saved outputs to {OUT}")


if __name__ == "__main__":
    main()
