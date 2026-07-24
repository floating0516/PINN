from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.physics.travel_time import (
    ConstantVelocityTravelTime,
    travel_time_from_config,
)
from src.training.time_sampling import sample_source_history
from src.utils.config_v2 import stf_m_ref_from_config, validate_config_v2


def _batch_vector(
    value: torch.Tensor,
    *,
    batch_size: int,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    result = torch.as_tensor(
        value,
        device=reference.device,
        dtype=reference.dtype,
    )
    if result.ndim == 0:
        result = result.expand(batch_size)
    if result.shape != (batch_size,):
        raise ValueError(f"{name} must be scalar or have shape (batch,)")
    return result


def _validated_sample_weights(
    sample_weights: torch.Tensor | None,
    *,
    batch_size: int,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if sample_weights is None:
        return None
    weights = torch.as_tensor(
        sample_weights,
        device=reference.device,
        dtype=reference.dtype,
    )
    if weights.shape != (batch_size,):
        raise ValueError("sample_weights must have shape (batch,)")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("sample_weights must be finite")
    if not bool(torch.all(weights > 0.0)):
        raise ValueError("sample_weights must be positive")
    return weights


def compute_physical_coefficients(
    source_distance_m: torch.Tensor,
    rho: float,
    alpha: float,
    beta: float,
    A_IP: torch.Tensor,
    A_IS: torch.Tensor,
    A_FP: torch.Tensor,
    A_FS: torch.Tensor,
    amplitude_gain: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return near- and far-field coefficients using source distance R."""
    if rho <= 0.0 or alpha <= 0.0 or beta <= 0.0:
        raise ValueError("rho, alpha, and beta must be positive")
    distance = torch.clamp(source_distance_m, min=1.0)
    scale = float(amplitude_gain)
    four_pi = 4.0 * math.pi
    C_int_P = scale * A_IP / (four_pi * rho * alpha**2 * distance**2)
    C_int_S = scale * A_IS / (four_pi * rho * beta**2 * distance**2)
    C_far_P = scale * A_FP / (four_pi * rho * alpha**3 * distance)
    C_far_S = scale * A_FS / (four_pi * rho * beta**3 * distance)
    return C_int_P, C_int_S, C_far_P, C_far_S


def forward_displacement_from_rate(
    rate_hat: torch.Tensor,
    source_dt_sec: torch.Tensor,
    observation_dt_sec: torch.Tensor,
    observation_steps: int,
    source_distance_m: torch.Tensor,
    travel_time: ConstantVelocityTravelTime,
    C_int_P: torch.Tensor,
    C_int_S: torch.Tensor,
    C_far_P: torch.Tensor,
    C_far_S: torch.Tensor,
    *,
    include_intermediate: bool,
    include_far_P: bool,
    include_far_S: bool,
    include_intermediate_P: bool,
    include_intermediate_S: bool,
    origin_aligned: bool = False,
) -> torch.Tensor:
    if rate_hat.ndim != 2 or not rate_hat.is_floating_point():
        raise ValueError("rate_hat must have shape (batch, source_time)")
    batch_size = rate_hat.shape[0]
    source_dt = _batch_vector(
        source_dt_sec,
        batch_size=batch_size,
        reference=rate_hat,
        name="source_dt_sec",
    )
    observation_dt = _batch_vector(
        observation_dt_sec,
        batch_size=batch_size,
        reference=rate_hat,
        name="observation_dt_sec",
    )
    source_distance = _batch_vector(
        source_distance_m,
        batch_size=batch_size,
        reference=rate_hat,
        name="source_distance_m",
    )
    if torch.any(source_distance < 0.0):
        raise ValueError("source_distance_m must be nonnegative")

    moment_hat = torch.cumsum(
        rate_hat * source_dt.reshape(-1, 1),
        dim=1,
    )
    delays = travel_time.delays(source_distance)
    if origin_aligned:
        p_delay_sec = delays.p_sec
        s_delay_sec = delays.s_sec
    else:
        p_delay_sec = torch.zeros_like(delays.p_sec)
        s_delay_sec = delays.s_after_p_sec
    rate_p = sample_source_history(
        rate_hat,
        source_dt,
        observation_dt,
        p_delay_sec,
        observation_steps,
    )
    rate_s = sample_source_history(
        rate_hat,
        source_dt,
        observation_dt,
        s_delay_sec,
        observation_steps,
    )
    moment_p = sample_source_history(
        moment_hat,
        source_dt,
        observation_dt,
        p_delay_sec,
        observation_steps,
    )
    moment_s = sample_source_history(
        moment_hat,
        source_dt,
        observation_dt,
        s_delay_sec,
        observation_steps,
    )

    C_int_P_batch = _batch_vector(
        C_int_P,
        batch_size=batch_size,
        reference=rate_hat,
        name="C_int_P",
    ).reshape(-1, 1)
    C_int_S_batch = _batch_vector(
        C_int_S,
        batch_size=batch_size,
        reference=rate_hat,
        name="C_int_S",
    ).reshape(-1, 1)
    C_far_P_batch = _batch_vector(
        C_far_P,
        batch_size=batch_size,
        reference=rate_hat,
        name="C_far_P",
    ).reshape(-1, 1)
    C_far_S_batch = _batch_vector(
        C_far_S,
        batch_size=batch_size,
        reference=rate_hat,
        name="C_far_S",
    ).reshape(-1, 1)

    displacement = torch.zeros_like(rate_p)
    if include_far_P:
        displacement = displacement + C_far_P_batch * rate_p
    if include_far_S:
        displacement = displacement + C_far_S_batch * rate_s
    if include_intermediate and include_intermediate_P:
        displacement = displacement + C_int_P_batch * moment_p
    if include_intermediate and include_intermediate_S:
        displacement = displacement + C_int_S_batch * moment_s
    return displacement


def forward_displacement_from_origin_rate(
    rate_hat: torch.Tensor,
    source_dt_sec: torch.Tensor,
    observation_dt_sec: torch.Tensor,
    observation_steps: int,
    source_distance_m: torch.Tensor,
    travel_time: ConstantVelocityTravelTime,
    C_int_P: torch.Tensor,
    C_int_S: torch.Tensor,
    C_far_P: torch.Tensor,
    C_far_S: torch.Tensor,
    *,
    include_intermediate: bool,
    include_far_P: bool,
    include_far_S: bool,
    include_intermediate_P: bool,
    include_intermediate_S: bool,
) -> torch.Tensor:
    """Forward an origin-aligned source history with absolute P/S delays."""
    return forward_displacement_from_rate(
        rate_hat,
        source_dt_sec,
        observation_dt_sec,
        observation_steps,
        source_distance_m,
        travel_time,
        C_int_P,
        C_int_S,
        C_far_P,
        C_far_S,
        include_intermediate=include_intermediate,
        include_far_P=include_far_P,
        include_far_S=include_far_S,
        include_intermediate_P=include_intermediate_P,
        include_intermediate_S=include_intermediate_S,
        origin_aligned=True,
    )


def compute_radiation_coefficients(
    theta_deg: torch.Tensor,
    phi_slip_deg: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if theta_deg.shape != phi_slip_deg.shape or theta_deg.ndim != 1:
        raise ValueError("radiation angles must have matching (batch,) shapes")
    if mode == "none":
        zeros = torch.zeros_like(theta_deg)
        return zeros, zeros, zeros, zeros
    if mode == "simplified":
        ones = torch.ones_like(theta_deg)
        return ones, ones, ones, ones
    if mode != "full":
        raise ValueError(f"unsupported radiation mode: {mode}")

    theta = torch.deg2rad(theta_deg)
    phi = torch.deg2rad(phi_slip_deg)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_2theta = torch.sin(2.0 * theta)
    cos_2theta = torch.cos(2.0 * theta)
    cos_phi = torch.cos(phi)
    A_IP = cos_phi * (
        4.0 * sin_2theta * sin_theta
        - 2.0 * cos_2theta * cos_theta
    )
    A_IS = cos_phi * (
        -3.0 * sin_2theta * sin_theta
        + 3.0 * cos_2theta * cos_theta
    )
    A_FP = cos_phi * sin_2theta * sin_theta
    A_FS = cos_phi * cos_2theta * cos_theta
    return A_IP, A_IS, A_FP, A_FS


def moment_magnitude_from_rate(
    rate_nm_per_s: torch.Tensor,
    dt_sec: torch.Tensor,
) -> torch.Tensor:
    if rate_nm_per_s.ndim != 2:
        raise ValueError("rate_nm_per_s must have shape (batch, source_time)")
    dt = _batch_vector(
        dt_sec,
        batch_size=rate_nm_per_s.shape[0],
        reference=rate_nm_per_s,
        name="dt_sec",
    ).reshape(-1, 1)
    if torch.any(dt <= 0.0):
        raise ValueError("dt_sec must be positive")
    moment = torch.sum(
        torch.clamp(rate_nm_per_s, min=0.0) * dt,
        dim=1,
    ).clamp_min(1.0e10)
    return (2.0 / 3.0) * (torch.log10(moment) - 9.1)


def masked_normalized_waveform_mse(
    u_hat: torch.Tensor,
    u_obs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if u_hat.shape != u_obs.shape or valid_mask.shape != u_obs.shape:
        raise ValueError("waveform prediction, observation, and mask must match")
    mask_bool = valid_mask.to(device=u_obs.device, dtype=torch.bool)
    mask = mask_bool.to(dtype=u_obs.dtype)
    observed_abs = torch.where(
        mask_bool,
        u_obs.abs(),
        torch.zeros_like(u_obs),
    )
    scale = observed_abs.amax(dim=1, keepdim=True).clamp_min(1.0e-12)
    squared = ((u_hat - u_obs) / scale).pow(2) * mask
    return squared.sum() / mask.sum().clamp_min(1.0)


def _per_sample_masked_normalized_waveform_mse(
    u_hat: torch.Tensor,
    u_obs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if u_hat.shape != u_obs.shape or valid_mask.shape != u_obs.shape:
        raise ValueError("waveform prediction, observation, and mask must match")
    mask_bool = valid_mask.to(device=u_obs.device, dtype=torch.bool)
    mask = mask_bool.to(dtype=u_obs.dtype)
    observed_abs = torch.where(
        mask_bool,
        u_obs.abs(),
        torch.zeros_like(u_obs),
    )
    scale = observed_abs.amax(dim=1, keepdim=True).clamp_min(1.0e-12)
    squared = ((u_hat - u_obs) / scale).pow(2) * mask
    return squared.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _shape_loss(predicted: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    predicted_nonnegative = torch.clamp(predicted, min=0.0)
    reference_nonnegative = torch.clamp(reference, min=0.0)
    predicted_normalized = predicted_nonnegative / predicted_nonnegative.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0e-12)
    reference_normalized = reference_nonnegative / reference_nonnegative.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0e-12)
    return F.mse_loss(predicted_normalized, reference_normalized)


def _per_sample_shape_loss(
    predicted: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    predicted_nonnegative = torch.clamp(predicted, min=0.0)
    reference_nonnegative = torch.clamp(reference, min=0.0)
    predicted_normalized = predicted_nonnegative / predicted_nonnegative.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0e-12)
    reference_normalized = reference_nonnegative / reference_nonnegative.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0e-12)
    return (predicted_normalized - reference_normalized).pow(2).mean(dim=1)


def pinn_loss_stf_rate_v2(
    *,
    rate_hat: torch.Tensor,
    pred_rate_encoded: torch.Tensor,
    rate_ref_encoded: torch.Tensor | None,
    rate_ref_physical: torch.Tensor | None,
    u_obs: torch.Tensor,
    source_dt_sec: torch.Tensor,
    observation_dt_sec: torch.Tensor,
    source_distance_m: torch.Tensor,
    waveform_valid_mask: torch.Tensor,
    travel_time: ConstantVelocityTravelTime,
    rho: float,
    theta_deg: torch.Tensor,
    phi_slip_deg: torch.Tensor,
    amplitude_gain: float,
    lambda_MSE: float,
    lambda_synth: float,
    lambda_mag: float,
    lambda_shape: float,
    has_ref: torch.Tensor | None,
    true_mag: torch.Tensor | None,
    pred_catalog_mw: torch.Tensor | None,
    include_intermediate: bool,
    radiation_mode: str,
    include_far_P: bool,
    include_far_S: bool,
    include_intermediate_P: bool,
    include_intermediate_S: bool,
    origin_aligned: bool = False,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size = rate_hat.shape[0]
    if pred_rate_encoded.shape != rate_hat.shape:
        raise ValueError("encoded and physical predicted rates must match")
    if u_obs.ndim != 2 or u_obs.shape[0] != batch_size:
        raise ValueError("u_obs must have shape (batch, observation_time)")
    weights = _validated_sample_weights(
        sample_weights,
        batch_size=batch_size,
        reference=rate_hat,
    )
    angles_theta = _batch_vector(
        theta_deg,
        batch_size=batch_size,
        reference=rate_hat,
        name="theta_deg",
    )
    angles_phi = _batch_vector(
        phi_slip_deg,
        batch_size=batch_size,
        reference=rate_hat,
        name="phi_slip_deg",
    )
    radiation = compute_radiation_coefficients(
        angles_theta,
        angles_phi,
        mode=radiation_mode,
    )
    coefficients = compute_physical_coefficients(
        _batch_vector(
            source_distance_m,
            batch_size=batch_size,
            reference=rate_hat,
            name="source_distance_m",
        ),
        rho,
        travel_time.alpha_m_per_s,
        travel_time.beta_m_per_s,
        *radiation,
        amplitude_gain=amplitude_gain,
    )
    u_hat = forward_displacement_from_rate(
        rate_hat,
        source_dt_sec,
        observation_dt_sec,
        u_obs.shape[1],
        source_distance_m,
        travel_time,
        *coefficients,
        include_intermediate=include_intermediate,
        include_far_P=include_far_P,
        include_far_S=include_far_S,
        include_intermediate_P=include_intermediate_P,
        include_intermediate_S=include_intermediate_S,
        origin_aligned=origin_aligned,
    )
    if weights is None:
        L_synth = masked_normalized_waveform_mse(
            u_hat,
            u_obs,
            waveform_valid_mask,
        )
    else:
        synth_per_sample = _per_sample_masked_normalized_waveform_mse(
            u_hat,
            u_obs,
            waveform_valid_mask,
        )
        # Do not self-normalize a minibatch of globally normalized weights.
        L_synth = (weights * synth_per_sample).mean()

    L_MSE = rate_hat.new_tensor(0.0)
    L_shape = rate_hat.new_tensor(0.0)
    reference_mask = (
        torch.ones(batch_size, device=rate_hat.device, dtype=torch.bool)
        if has_ref is None
        else has_ref.to(device=rate_hat.device, dtype=torch.bool).reshape(-1)
    )
    if rate_ref_encoded is not None and torch.any(reference_mask):
        if weights is None:
            L_MSE = F.mse_loss(
                pred_rate_encoded[reference_mask],
                rate_ref_encoded[reference_mask],
            )
        else:
            mse_per_sample = F.mse_loss(
                pred_rate_encoded[reference_mask],
                rate_ref_encoded[reference_mask],
                reduction="none",
            ).mean(dim=1)
            L_MSE = (weights[reference_mask] * mse_per_sample).mean()
    if rate_ref_physical is not None and torch.any(reference_mask):
        if weights is None:
            L_shape = _shape_loss(
                rate_hat[reference_mask],
                rate_ref_physical[reference_mask],
            )
        else:
            shape_per_sample = _per_sample_shape_loss(
                rate_hat[reference_mask],
                rate_ref_physical[reference_mask],
            )
            L_shape = (weights[reference_mask] * shape_per_sample).mean()

    window_magnitude = moment_magnitude_from_rate(
        rate_hat,
        source_dt_sec,
    )
    L_mag = rate_hat.new_tensor(0.0)
    if true_mag is not None:
        predicted_magnitude = (
            window_magnitude
            if pred_catalog_mw is None
            else _batch_vector(
                pred_catalog_mw,
                batch_size=batch_size,
                reference=rate_hat,
                name="pred_catalog_mw",
            )
        )
        finite = torch.isfinite(true_mag.reshape(-1))
        if torch.any(finite):
            if weights is None:
                L_mag = F.mse_loss(
                    predicted_magnitude[finite],
                    true_mag.reshape(-1)[finite],
                )
            else:
                magnitude_per_sample = F.mse_loss(
                    predicted_magnitude[finite],
                    true_mag.reshape(-1)[finite],
                    reduction="none",
                )
                L_mag = (weights[finite] * magnitude_per_sample).mean()

    total_loss = (
        float(lambda_MSE) * L_MSE
        + float(lambda_synth) * L_synth
        + float(lambda_mag) * L_mag
        + float(lambda_shape) * L_shape
    )
    metrics = {
        "L_total": float(total_loss.detach().cpu()),
        "L_MSE": float(L_MSE.detach().cpu()),
        "L_synth": float(L_synth.detach().cpu()),
        "L_mag": float(L_mag.detach().cpu()),
        "L_shape": float(L_shape.detach().cpu()),
        "window_mw_mean": float(window_magnitude.detach().mean().cpu()),
    }
    return total_loss, metrics


def causal_event_stf_rate_loss_v2(
    *,
    rate_hat: torch.Tensor,
    pred_rate_encoded: torch.Tensor,
    rate_ref_encoded: torch.Tensor,
    rate_ref_physical: torch.Tensor,
    pred_catalog_mw: torch.Tensor,
    true_mag: torch.Tensor,
    radial_obs: torch.Tensor,
    source_distance_m: torch.Tensor,
    theta_deg: torch.Tensor,
    phi_slip_deg: torch.Tensor,
    source_dt_sec: torch.Tensor,
    observation_dt_sec: torch.Tensor,
    waveform_valid_mask: torch.Tensor,
    station_mask: torch.Tensor,
    travel_time: ConstantVelocityTravelTime,
    rho: float,
    amplitude_gain: float,
    lambda_MSE: float,
    lambda_synth: float,
    lambda_mag: float,
    lambda_shape: float,
    include_intermediate: bool,
    radiation_mode: str,
    include_far_P: bool,
    include_far_S: bool,
    include_intermediate_P: bool,
    include_intermediate_S: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the original four losses to a shared origin-aligned event STF."""
    if rate_hat.ndim != 2 or pred_rate_encoded.shape != rate_hat.shape:
        raise ValueError("event STF predictions must have shape (batch, source_time)")
    batch_size = rate_hat.shape[0]
    if rate_ref_encoded.shape != rate_hat.shape:
        raise ValueError("encoded STF reference must match the prediction")
    if rate_ref_physical.shape != rate_hat.shape:
        raise ValueError("physical STF reference must match the prediction")
    if radial_obs.ndim != 3 or radial_obs.shape[0] != batch_size:
        raise ValueError("radial_obs must have shape (batch, station, time)")
    station_shape = radial_obs.shape[:2]
    station_tensors = {
        "source_distance_m": source_distance_m,
        "theta_deg": theta_deg,
        "phi_slip_deg": phi_slip_deg,
        "observation_dt_sec": observation_dt_sec,
        "station_mask": station_mask,
    }
    for name, value in station_tensors.items():
        if value.shape != station_shape:
            raise ValueError(f"{name} must have shape (batch, station)")
    if waveform_valid_mask.shape != radial_obs.shape:
        raise ValueError("waveform_valid_mask must match radial_obs")

    event_indices, station_indices = torch.nonzero(
        station_mask.to(dtype=torch.bool),
        as_tuple=True,
    )
    if event_indices.numel() == 0:
        raise ValueError("at least one station is required for the forward loss")
    selected_rate = rate_hat[event_indices]
    selected_distance = source_distance_m[event_indices, station_indices]
    selected_theta = theta_deg[event_indices, station_indices]
    selected_phi = phi_slip_deg[event_indices, station_indices]
    radiation = compute_radiation_coefficients(
        selected_theta,
        selected_phi,
        mode=radiation_mode,
    )
    coefficients = compute_physical_coefficients(
        selected_distance,
        rho,
        travel_time.alpha_m_per_s,
        travel_time.beta_m_per_s,
        *radiation,
        amplitude_gain=amplitude_gain,
    )
    source_dt = _batch_vector(
        source_dt_sec,
        batch_size=batch_size,
        reference=rate_hat,
        name="source_dt_sec",
    )[event_indices]
    selected_observation_dt = observation_dt_sec[event_indices, station_indices]
    selected_observed = radial_obs[event_indices, station_indices]
    selected_valid = waveform_valid_mask[event_indices, station_indices]
    u_hat = forward_displacement_from_origin_rate(
        selected_rate,
        source_dt,
        selected_observation_dt,
        radial_obs.shape[-1],
        selected_distance,
        travel_time,
        *coefficients,
        include_intermediate=include_intermediate,
        include_far_P=include_far_P,
        include_far_S=include_far_S,
        include_intermediate_P=include_intermediate_P,
        include_intermediate_S=include_intermediate_S,
    )

    L_MSE = F.mse_loss(pred_rate_encoded, rate_ref_encoded)
    L_synth = masked_normalized_waveform_mse(
        u_hat,
        selected_observed,
        selected_valid,
    )
    L_mag = F.mse_loss(
        pred_catalog_mw.reshape(batch_size),
        true_mag.reshape(batch_size),
    )
    L_shape = _shape_loss(rate_hat, rate_ref_physical)
    total_loss = (
        float(lambda_MSE) * L_MSE
        + float(lambda_synth) * L_synth
        + float(lambda_mag) * L_mag
        + float(lambda_shape) * L_shape
    )
    window_magnitude = moment_magnitude_from_rate(rate_hat, source_dt_sec)
    metrics = {
        "L_total": float(total_loss.detach().cpu()),
        "L_MSE": float(L_MSE.detach().cpu()),
        "L_synth": float(L_synth.detach().cpu()),
        "L_mag": float(L_mag.detach().cpu()),
        "L_shape": float(L_shape.detach().cpu()),
        "window_mw_mean": float(window_magnitude.detach().mean().cpu()),
    }
    return total_loss, metrics


class CausalEventSTFRateWaveformLossV2(nn.Module):
    """Original V2 loss system for a causal, event-level shared STF model."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        validate_config_v2(config)
        loss_config = config["training"]["stf_rate_loss"]
        self.lambda_MSE = float(loss_config["lambda_MSE"])
        self.lambda_synth = float(loss_config["lambda_synth"])
        self.lambda_mag = float(loss_config["lambda_mag"])
        self.lambda_shape = float(loss_config["lambda_shape"])
        self.rate_representation = str(
            config["training"]["rate_representation"]
        ).lower()
        self.stf_m_ref = stf_m_ref_from_config(config)
        physics = config["physics"]
        self.rho = float(physics["rho"])
        self.travel_time = travel_time_from_config(config)
        self.amplitude_gain = float(physics["amplitude_gain"])
        self.include_intermediate = bool(
            loss_config.get("include_intermediate_field", True)
        )
        self.radiation_mode = str(
            loss_config.get("radiation_pattern_mode", "simplified")
        ).lower()
        self.include_far_P = bool(loss_config.get("include_far_field_P", True))
        self.include_far_S = bool(loss_config.get("include_far_field_S", True))
        self.include_intermediate_P = bool(
            loss_config.get("include_intermediate_field_P", True)
        )
        self.include_intermediate_S = bool(
            loss_config.get("include_intermediate_field_S", True)
        )

    def _decode_rate(self, pred_rate: torch.Tensor) -> torch.Tensor:
        safe = torch.nan_to_num(
            pred_rate,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if self.rate_representation == "log1p":
            safe = torch.clamp(safe, min=-20.0, max=6.0)
            return torch.clamp(
                self.stf_m_ref * (torch.pow(10.0, safe) - 1.0),
                min=0.0,
            )
        if self.rate_representation == "linear":
            return torch.clamp(safe, min=0.0)
        raise ValueError(
            f"unsupported rate_representation: {self.rate_representation}"
        )

    def forward(
        self,
        pred_rate: torch.Tensor,
        *,
        pred_catalog_mw: torch.Tensor,
        rate_ref_encoded: torch.Tensor,
        rate_ref_physical: torch.Tensor,
        true_mag: torch.Tensor,
        radial_obs: torch.Tensor,
        source_distance_m: torch.Tensor,
        theta_deg: torch.Tensor,
        phi_slip_deg: torch.Tensor,
        source_dt_sec: torch.Tensor,
        observation_dt_sec: torch.Tensor,
        waveform_valid_mask: torch.Tensor,
        station_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return causal_event_stf_rate_loss_v2(
            rate_hat=self._decode_rate(pred_rate),
            pred_rate_encoded=pred_rate,
            rate_ref_encoded=rate_ref_encoded,
            rate_ref_physical=rate_ref_physical,
            pred_catalog_mw=pred_catalog_mw,
            true_mag=true_mag,
            radial_obs=radial_obs,
            source_distance_m=source_distance_m,
            theta_deg=theta_deg,
            phi_slip_deg=phi_slip_deg,
            source_dt_sec=source_dt_sec,
            observation_dt_sec=observation_dt_sec,
            waveform_valid_mask=waveform_valid_mask,
            station_mask=station_mask,
            travel_time=self.travel_time,
            rho=self.rho,
            amplitude_gain=self.amplitude_gain,
            lambda_MSE=self.lambda_MSE,
            lambda_synth=self.lambda_synth,
            lambda_mag=self.lambda_mag,
            lambda_shape=self.lambda_shape,
            include_intermediate=self.include_intermediate,
            radiation_mode=self.radiation_mode,
            include_far_P=self.include_far_P,
            include_far_S=self.include_far_S,
            include_intermediate_P=self.include_intermediate_P,
            include_intermediate_S=self.include_intermediate_S,
        )


class STFRateWaveformLossV2(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        validate_config_v2(config)
        self.config = config
        self.active_workflow = (
            config.get("workflow") == "station_random_shifted_stf"
        )
        loss_config = config["training"]["stf_rate_loss"]
        self.lambda_MSE = float(loss_config["lambda_MSE"])
        self.lambda_synth = float(loss_config["lambda_synth"])
        self.lambda_mag = float(loss_config["lambda_mag"])
        self.lambda_shape = float(loss_config["lambda_shape"])
        self.rate_representation = str(
            config["training"]["rate_representation"]
        ).lower()
        self.stf_m_ref = stf_m_ref_from_config(config)
        physics = config["physics"]
        self.origin_aligned = str(physics["delay_mode"]) == "absolute"
        self.rho = float(physics["rho"])
        self.alpha = float(physics["alpha"])
        self.beta = float(physics["beta"])
        self.travel_time = (
            travel_time_from_config(config)
            if physics.get("travel_time_model") == "constant_velocity"
            else ConstantVelocityTravelTime(self.alpha, self.beta)
        )
        self.amplitude_gain = float(physics["amplitude_gain"])
        self.include_intermediate = bool(
            loss_config.get("include_intermediate_field", True)
        )
        self.radiation_mode = str(
            loss_config.get("radiation_pattern_mode", "simplified")
        ).lower()
        self.include_far_P = bool(loss_config.get("include_far_field_P", True))
        self.include_far_S = bool(loss_config.get("include_far_field_S", True))
        self.include_intermediate_P = bool(
            loss_config.get("include_intermediate_field_P", True)
        )
        self.include_intermediate_S = bool(
            loss_config.get("include_intermediate_field_S", True)
        )

    def _decode_rate(self, pred_rate: torch.Tensor) -> torch.Tensor:
        safe = torch.nan_to_num(
            pred_rate,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if self.rate_representation == "log1p":
            safe = torch.clamp(safe, min=-20.0, max=6.0)
            return torch.clamp(
                self.stf_m_ref * (torch.pow(10.0, safe) - 1.0),
                min=0.0,
            )
        if self.rate_representation == "linear":
            return torch.clamp(safe, min=0.0)
        raise ValueError(
            f"unsupported rate_representation: {self.rate_representation}"
        )

    def forward(
        self,
        pred_rate: torch.Tensor,
        *,
        pred_catalog_mw: torch.Tensor | None = None,
        radial_obs: torch.Tensor,
        source_distance_m: torch.Tensor,
        theta_deg: torch.Tensor,
        phi_slip_deg: torch.Tensor,
        source_dt_sec: torch.Tensor,
        observation_dt_sec: torch.Tensor,
        waveform_valid_mask: torch.Tensor,
        stf_true: torch.Tensor | None = None,
        has_stf: torch.Tensor | None = None,
        true_mag: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.active_workflow and pred_catalog_mw is None:
            raise ValueError(
                "active station workflow requires pred_catalog_mw"
            )
        if radial_obs.ndim == 3 and radial_obs.shape[1] == 1:
            u_obs = radial_obs[:, 0, :]
        elif radial_obs.ndim == 2:
            u_obs = radial_obs
        else:
            raise ValueError("radial_obs must have shape (B,T) or (B,1,T)")
        rate_hat = self._decode_rate(pred_rate)
        rate_ref_encoded = None
        if stf_true is not None:
            rate_ref_encoded = torch.log10(
                1.0 + torch.clamp(stf_true, min=0.0) / self.stf_m_ref
            )
        return pinn_loss_stf_rate_v2(
            rate_hat=rate_hat,
            pred_rate_encoded=pred_rate,
            rate_ref_encoded=rate_ref_encoded,
            rate_ref_physical=stf_true,
            u_obs=u_obs,
            source_dt_sec=source_dt_sec,
            observation_dt_sec=observation_dt_sec,
            source_distance_m=source_distance_m,
            waveform_valid_mask=waveform_valid_mask,
            travel_time=self.travel_time,
            rho=self.rho,
            theta_deg=theta_deg,
            phi_slip_deg=phi_slip_deg,
            amplitude_gain=self.amplitude_gain,
            lambda_MSE=self.lambda_MSE,
            lambda_synth=self.lambda_synth,
            lambda_mag=self.lambda_mag,
            lambda_shape=self.lambda_shape,
            has_ref=has_stf,
            true_mag=true_mag,
            pred_catalog_mw=pred_catalog_mw,
            include_intermediate=self.include_intermediate,
            radiation_mode=self.radiation_mode,
            include_far_P=self.include_far_P,
            include_far_S=self.include_far_S,
            include_intermediate_P=self.include_intermediate_P,
            include_intermediate_S=self.include_intermediate_S,
            origin_aligned=self.origin_aligned,
            sample_weights=sample_weights,
        )
