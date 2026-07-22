from __future__ import annotations

from numbers import Integral

import torch


def _batch_vector(
    value: torch.Tensor,
    *,
    batch_size: int,
    source: torch.Tensor,
    name: str,
) -> torch.Tensor:
    result = torch.as_tensor(
        value,
        device=source.device,
        dtype=source.dtype,
    )
    if result.ndim == 0:
        result = result.expand(batch_size)
    if result.shape != (batch_size,):
        raise ValueError(f"{name} must be scalar or have shape (batch,)")
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def sample_source_history(
    source: torch.Tensor,
    source_dt_sec: torch.Tensor,
    observation_dt_sec: torch.Tensor,
    delay_sec: torch.Tensor,
    observation_steps: int,
) -> torch.Tensor:
    if source.ndim != 2 or not source.is_floating_point():
        raise ValueError("source must be a floating tensor with shape (batch, time)")
    batch_size, source_steps = source.shape
    if source_steps < 1:
        raise ValueError("source must contain at least one time step")
    if (
        isinstance(observation_steps, bool)
        or not isinstance(observation_steps, Integral)
        or observation_steps < 0
    ):
        raise ValueError("observation_steps must be a nonnegative integer")

    source_dt = _batch_vector(
        source_dt_sec,
        batch_size=batch_size,
        source=source,
        name="source_dt_sec",
    )
    observation_dt = _batch_vector(
        observation_dt_sec,
        batch_size=batch_size,
        source=source,
        name="observation_dt_sec",
    )
    delay = _batch_vector(
        delay_sec,
        batch_size=batch_size,
        source=source,
        name="delay_sec",
    )
    if torch.any(source_dt <= 0.0) or torch.any(observation_dt <= 0.0):
        raise ValueError("source and observation time steps must be positive")
    if torch.any(delay < 0.0):
        raise ValueError("delay_sec must be nonnegative")

    observation_index = torch.arange(
        int(observation_steps),
        device=source.device,
        dtype=source.dtype,
    ).view(1, -1)
    observation_time = observation_index * observation_dt.reshape(-1, 1)
    source_position = (
        observation_time - delay.reshape(-1, 1)
    ) / source_dt.reshape(-1, 1)
    left = torch.floor(source_position).to(torch.long)
    right = left + 1
    weight_right = source_position - left.to(source.dtype)
    valid = (source_position >= 0.0) & (
        source_position <= float(source_steps - 1)
    )
    left_safe = left.clamp(0, source_steps - 1)
    right_safe = right.clamp(0, source_steps - 1)
    left_value = torch.gather(source, 1, left_safe)
    right_value = torch.gather(source, 1, right_safe)
    interpolated = (
        left_value * (1.0 - weight_right)
        + right_value * weight_right
    )
    return torch.where(valid, interpolated, torch.zeros_like(interpolated))
