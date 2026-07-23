from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from src.utils.config_v2 import waveform_input_components_from_config


def _single_channel(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value.unsqueeze(1)
    if value.ndim == 3 and value.shape[1] == 1:
        return value
    raise ValueError(
        f"{name} must have shape (batch,time) or (batch,1,time)"
    )


def assemble_model_input(
    values: Mapping[str, torch.Tensor],
    config: dict[str, Any],
) -> torch.Tensor:
    channels = [
        _single_channel(name, values[name])
        for name in waveform_input_components_from_config(config)
    ]
    reference_shape = channels[0].shape
    if any(
        channel.shape != reference_shape for channel in channels[1:]
    ):
        raise ValueError("configured waveform component shapes differ")
    output = torch.cat(channels, dim=1)
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError("model waveform input is non-finite")
    return output
