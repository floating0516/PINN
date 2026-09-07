"""Causally constrained ("released") moment targets for prefix training.

Definitions (all times are seconds since origin; the waveform window, the
reference STF and the predicted STF share this origin-aligned axis):

* ``tau_i``      : P arrival at station ``i`` (``source_distance_m / alpha``).
* ``h``          : observation horizon; the model has seen waveform samples
                   ``[0, h)`` only.
* ``h_src,i``    : ``clip(h - tau_i, 0, S * dt)`` -- the part of source time
                   whose radiation can already have reached station ``i``.
* ``B(h)``       : ``Mw( integral_0^{h_src,i} STF dt )`` -- released moment
                   magnitude. Applied to the reference STF it is the label,
                   applied to the predicted STF it is the prediction.
* ``A(h)``       : ``Mw( integral_0^{S*dt} STF dt )`` -- the full-window
                   integral, i.e. the "final magnitude" quantity that the
                   earlier causal Phase 39 reported.

The mask is a fixed per-sample coverage vector over source steps: step ``k``
covers ``[k*dt, (k+1)*dt)`` and receives fractional weight for a partially
covered last step. It carries no gradient.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


DEFAULT_MOMENT_FLOOR_NM = 1.0e15  # Mw ~ 3.93


def _as_batch_vector(value: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch_size)
    tensor = tensor.reshape(-1)
    if tensor.numel() != batch_size:
        raise ValueError(f"{name} must have one value per sample")
    return tensor


def constrained_window_sec(
    horizon_sec: float,
    p_arrival_sec: torch.Tensor,
    *,
    max_sec: float,
) -> torch.Tensor:
    """``clip(h - tau, 0, max_sec)`` per sample."""
    if not math.isfinite(float(horizon_sec)) or float(horizon_sec) < 0.0:
        raise ValueError("horizon_sec must be finite and nonnegative")
    if not math.isfinite(float(max_sec)) or float(max_sec) <= 0.0:
        raise ValueError("max_sec must be finite and positive")
    tau = torch.as_tensor(p_arrival_sec, dtype=torch.float32)
    if not bool(torch.isfinite(tau).all()) or bool((tau < 0.0).any()):
        raise ValueError("p_arrival_sec must be finite and nonnegative")
    return torch.clamp(float(horizon_sec) - tau, min=0.0, max=float(max_sec))


def constrained_source_mask(
    horizon_sec: float,
    p_arrival_sec: torch.Tensor,
    *,
    source_steps: int,
    source_dt_sec: torch.Tensor,
) -> torch.Tensor:
    """Coverage mask ``(batch, source_steps)`` for source time ``< h - tau``."""
    if source_steps < 1:
        raise ValueError("source_steps must be positive")
    tau = torch.as_tensor(p_arrival_sec, dtype=torch.float32)
    batch_size = tau.reshape(-1).numel()
    dt = _as_batch_vector(source_dt_sec, batch_size, "source_dt_sec").to(
        device=tau.device,
        dtype=torch.float32,
    )
    if bool((dt <= 0.0).any()):
        raise ValueError("source_dt_sec must be positive")
    window = constrained_window_sec(
        horizon_sec,
        tau.reshape(-1),
        max_sec=float(source_steps) * float(dt.max()),
    ).to(device=tau.device)
    steps_covered = (window / dt).reshape(-1, 1)
    index = torch.arange(
        source_steps,
        device=tau.device,
        dtype=torch.float32,
    ).reshape(1, -1)
    return torch.clamp(steps_covered - index, min=0.0, max=1.0)


def zero_pad_prefix(model_input: torch.Tensor, horizon_sec: int) -> torch.Tensor:
    """Keep the first ``horizon_sec`` samples on the absolute axis, zero the rest."""
    if model_input.ndim != 3:
        raise ValueError("model_input must have shape (batch, channels, time)")
    steps = int(model_input.shape[-1])
    if horizon_sec < 1 or horizon_sec > steps:
        raise ValueError("horizon_sec must lie within the input window")
    if horizon_sec == steps:
        return model_input
    keep = torch.zeros(
        steps,
        device=model_input.device,
        dtype=model_input.dtype,
    )
    keep[:horizon_sec] = 1.0
    return model_input * keep.reshape(1, 1, -1)


def moment_from_rate(
    rate_nm_per_s: torch.Tensor,
    mask: torch.Tensor,
    source_dt_sec: torch.Tensor,
) -> torch.Tensor:
    if rate_nm_per_s.ndim != 2 or mask.shape != rate_nm_per_s.shape:
        raise ValueError("rate and mask must share shape (batch, source_time)")
    batch_size = rate_nm_per_s.shape[0]
    dt = _as_batch_vector(source_dt_sec, batch_size, "source_dt_sec").to(
        device=rate_nm_per_s.device,
        dtype=rate_nm_per_s.dtype,
    ).reshape(-1, 1)
    return torch.sum(torch.clamp(rate_nm_per_s, min=0.0) * mask * dt, dim=1)


def magnitude_from_moment(
    moment_nm: torch.Tensor,
    *,
    floor_nm: float = DEFAULT_MOMENT_FLOOR_NM,
) -> torch.Tensor:
    if not math.isfinite(float(floor_nm)) or float(floor_nm) <= 0.0:
        raise ValueError("floor_nm must be finite and positive")
    return (2.0 / 3.0) * (torch.log10(moment_nm.clamp_min(float(floor_nm))) - 9.1)


def released_magnitude(
    rate_nm_per_s: torch.Tensor,
    mask: torch.Tensor,
    source_dt_sec: torch.Tensor,
    *,
    floor_nm: float = DEFAULT_MOMENT_FLOOR_NM,
) -> torch.Tensor:
    return magnitude_from_moment(
        moment_from_rate(rate_nm_per_s, mask, source_dt_sec),
        floor_nm=floor_nm,
    )


def masked_encoded_mse(
    pred_encoded: torch.Tensor,
    ref_encoded: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sample MSE over covered source steps only (``0`` when nothing is covered)."""
    if pred_encoded.shape != ref_encoded.shape or mask.shape != pred_encoded.shape:
        raise ValueError("encoded rates and mask must share shape")
    squared = F.mse_loss(pred_encoded, ref_encoded, reduction="none") * mask
    coverage = mask.sum(dim=1)
    return squared.sum(dim=1) / coverage.clamp_min(1.0e-6) * (coverage > 0.0)


def released_prefix_losses(
    *,
    pred_encoded: torch.Tensor,
    rate_hat: torch.Tensor,
    ref_encoded: torch.Tensor,
    stf_true: torch.Tensor,
    mask: torch.Tensor,
    source_dt_sec: torch.Tensor,
    sample_weights: torch.Tensor | None,
    min_constrained_window_sec: float,
    floor_nm: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return ``(L_MSE^h, L_mag^h, metrics)`` for one prefix.

    Samples whose constrained window is shorter than
    ``min_constrained_window_sec`` (essentially: P has not arrived yet) are
    excluded from both terms; they still contribute L_synth outside.
    """
    batch_size = rate_hat.shape[0]
    dt = _as_batch_vector(source_dt_sec, batch_size, "source_dt_sec").to(
        device=rate_hat.device,
        dtype=rate_hat.dtype,
    )
    covered_sec = mask.sum(dim=1) * dt
    included = covered_sec >= float(min_constrained_window_sec)
    if sample_weights is None:
        weights = torch.ones(batch_size, device=rate_hat.device, dtype=rate_hat.dtype)
    else:
        weights = sample_weights.reshape(-1).to(
            device=rate_hat.device,
            dtype=rate_hat.dtype,
        )
        if weights.shape[0] != batch_size:
            raise ValueError("sample_weights must match the batch")
    b_pred = released_magnitude(rate_hat, mask, source_dt_sec, floor_nm=floor_nm)
    b_ref = released_magnitude(stf_true, mask, source_dt_sec, floor_nm=floor_nm)
    mse_per_sample = masked_encoded_mse(pred_encoded, ref_encoded, mask)
    mag_per_sample = (b_pred - b_ref).square()
    gate = included.to(rate_hat.dtype) * weights
    # Mean over the whole batch (excluded samples count as zero), which keeps the
    # scale comparable to the criterion's own weighted means.
    l_mse = (gate * mse_per_sample).mean()
    l_mag = (gate * mag_per_sample).mean()
    metrics = {
        "included_fraction": float(included.float().mean().detach().cpu()),
        "mean_constrained_window_sec": float(covered_sec.mean().detach().cpu()),
        "released_abs_error_mw": (
            float((b_pred - b_ref).abs()[included].mean().detach().cpu())
            if bool(included.any())
            else 0.0
        ),
    }
    return l_mse, l_mag, metrics


def released_monotone_loss(
    early_released_mw: torch.Tensor,
    late_released_mw: torch.Tensor,
    *,
    slack_mw: float,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalise released magnitude decreasing from an earlier to a later prefix."""
    if slack_mw < 0.0 or not math.isfinite(slack_mw):
        raise ValueError("slack_mw must be finite and nonnegative")
    violation = F.relu(
        early_released_mw.reshape(-1) - late_released_mw.reshape(-1) - float(slack_mw)
    ).square()
    if sample_weights is None:
        return violation.mean()
    weights = sample_weights.reshape(-1).to(
        device=violation.device,
        dtype=violation.dtype,
    )
    return (weights * violation).mean()


def distance_stratified_delta_bounds(
    source_distance_m: torch.Tensor,
    *,
    near_km: float,
    far_km: float,
    near_min_mw: float,
    near_max_mw: float,
    far_min_mw: float,
    far_max_mw: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly interpolate the ΔMw sampling interval from near to far stations.

    ``t = clip((r_km - near_km) / (far_km - near_km), 0, 1)``.
    Near stations keep the baseline interval; far stations shift toward
    down-scaling so that a Tohoku-type far record can become an M7 far record.
    """
    if not math.isfinite(near_km) or not math.isfinite(far_km) or far_km <= near_km:
        raise ValueError("far_km must be finite and greater than near_km")
    for name, lo, hi in (
        ("near", near_min_mw, near_max_mw),
        ("far", far_min_mw, far_max_mw),
    ):
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            raise ValueError(f"{name} ΔMw interval must be finite with min < max")
    distance_km = torch.as_tensor(source_distance_m, dtype=torch.float32).reshape(-1) / 1000.0
    if bool((distance_km < 0.0).any()) or not bool(torch.isfinite(distance_km).all()):
        raise ValueError("source_distance_m must be finite and nonnegative")
    t = torch.clamp((distance_km - float(near_km)) / (float(far_km) - float(near_km)), 0.0, 1.0)
    lo = (1.0 - t) * float(near_min_mw) + t * float(far_min_mw)
    hi = (1.0 - t) * float(near_max_mw) + t * float(far_max_mw)
    return lo, hi


def sample_distance_stratified_delta_mw(
    source_distance_m: torch.Tensor,
    *,
    near_km: float,
    far_km: float,
    near_min_mw: float,
    near_max_mw: float,
    far_min_mw: float,
    far_max_mw: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one ΔMw per sample from the distance-dependent interval."""
    lo, hi = distance_stratified_delta_bounds(
        source_distance_m,
        near_km=near_km,
        far_km=far_km,
        near_min_mw=near_min_mw,
        near_max_mw=near_max_mw,
        far_min_mw=far_min_mw,
        far_max_mw=far_max_mw,
    )
    unit = torch.rand(lo.shape, device=lo.device, dtype=lo.dtype, generator=generator)
    return lo + unit * (hi - lo)
