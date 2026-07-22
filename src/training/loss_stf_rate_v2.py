from __future__ import annotations

import math

import torch

from src.training.time_sampling import sample_source_history


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


def compute_physical_coefficients(
    source_distance_m: torch.Tensor,
    rho: float,
    alpha: float,
    beta: float,
    A_IP: torch.Tensor,
    A_IS: torch.Tensor,
    A_FP: torch.Tensor,
    A_FS: torch.Tensor,
    geom: float = 1.0,
    free_surface: float = 1.0,
    attenuation: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return near- and far-field coefficients using source distance R."""
    if rho <= 0.0 or alpha <= 0.0 or beta <= 0.0:
        raise ValueError("rho, alpha, and beta must be positive")
    distance = torch.clamp(source_distance_m, min=1.0)
    scale = float(geom) * float(free_surface) * float(attenuation)
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
    alpha: float,
    beta: float,
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
    if rate_hat.ndim != 2 or not rate_hat.is_floating_point():
        raise ValueError("rate_hat must have shape (batch, source_time)")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")
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
    p_delay_sec = source_distance / float(alpha)
    s_delay_sec = source_distance / float(beta)
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
