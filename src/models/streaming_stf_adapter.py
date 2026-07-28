from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class StreamingSTFAdapter(nn.Module):
    """Causal learned retention gate over frozen per-prefix STF predictions."""

    def __init__(
        self,
        *,
        source_steps: int = 200,
        stf_m_ref: float = 1.0e18,
        local_channels: int = 4,
        hidden_size: int = 8,
        support_ramp_sec: float = 6.0,
        initial_gate_logit: float = -4.0,
    ) -> None:
        super().__init__()
        if source_steps < 1:
            raise ValueError("source_steps must be positive")
        if not math.isfinite(stf_m_ref) or stf_m_ref <= 0.0:
            raise ValueError("stf_m_ref must be positive and finite")
        if local_channels < 1 or hidden_size < 1:
            raise ValueError("adapter channel sizes must be positive")
        if not math.isfinite(support_ramp_sec) or support_ramp_sec <= 0.0:
            raise ValueError("support_ramp_sec must be positive and finite")
        if not math.isfinite(initial_gate_logit):
            raise ValueError("initial_gate_logit must be finite")

        self.source_steps = int(source_steps)
        self.stf_m_ref = float(stf_m_ref)
        self.support_ramp_sec = float(support_ramp_sec)
        self.local_context = nn.Conv1d(
            1,
            int(local_channels),
            kernel_size=5,
            padding=2,
        )
        feature_count = 5 + int(local_channels)
        self.temporal_gate = nn.GRU(
            input_size=feature_count,
            hidden_size=int(hidden_size),
            num_layers=1,
            batch_first=True,
        )
        self.gate_head = nn.Linear(int(hidden_size), 1)
        nn.init.normal_(self.gate_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.gate_head.bias, float(initial_gate_logit))

    def _validate_inputs(
        self,
        raw_rate_nm_per_s: torch.Tensor,
        horizons_sec: torch.Tensor | Sequence[int],
        source_distance_m: torch.Tensor,
        source_dt_sec: torch.Tensor,
        beta_m_per_s: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = raw_rate_nm_per_s
        if raw.ndim != 3 or raw.shape[2] != self.source_steps:
            raise ValueError(
                "raw_rate_nm_per_s must have shape "
                f"(batch, issue_time, {self.source_steps})"
            )
        if not raw.is_floating_point() or not bool(torch.isfinite(raw).all()):
            raise ValueError("raw STF rates must be finite floating-point values")
        if bool(torch.any(raw < 0.0)):
            raise ValueError("raw STF rates must be nonnegative")
        horizons = torch.as_tensor(
            horizons_sec,
            device=raw.device,
            dtype=raw.dtype,
        ).reshape(-1)
        if horizons.shape[0] != raw.shape[1]:
            raise ValueError("horizons must match the issue-time dimension")
        if horizons.numel() > 1 and not bool(
            torch.allclose(
                horizons[1:] - horizons[:-1],
                torch.ones_like(horizons[1:]),
                rtol=0.0,
                atol=0.0,
            )
        ):
            raise ValueError("horizons must be consecutive one-second steps")
        distance = torch.as_tensor(
            source_distance_m,
            device=raw.device,
            dtype=raw.dtype,
        ).reshape(-1)
        source_dt = torch.as_tensor(
            source_dt_sec,
            device=raw.device,
            dtype=raw.dtype,
        ).reshape(-1)
        if distance.shape != (raw.shape[0],):
            raise ValueError("source_distance_m must have shape (batch,)")
        if source_dt.shape != (raw.shape[0],):
            raise ValueError("source_dt_sec must have shape (batch,)")
        if bool(torch.any(distance < 0.0)) or bool(torch.any(source_dt <= 0.0)):
            raise ValueError("distance must be nonnegative and source dt positive")
        if not math.isfinite(beta_m_per_s) or beta_m_per_s <= 0.0:
            raise ValueError("beta_m_per_s must be positive and finite")
        return horizons, distance, source_dt

    def forward(
        self,
        raw_rate_nm_per_s: torch.Tensor,
        *,
        horizons_sec: torch.Tensor | Sequence[int],
        source_distance_m: torch.Tensor,
        source_dt_sec: torch.Tensor,
        beta_m_per_s: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = raw_rate_nm_per_s
        horizons, distance, source_dt = self._validate_inputs(
            raw,
            horizons_sec,
            source_distance_m,
            source_dt_sec,
            beta_m_per_s,
        )
        batch_size, issue_steps, source_steps = raw.shape
        encoded = torch.log10(1.0 + raw / self.stf_m_ref)
        delta = torch.zeros_like(encoded)
        if issue_steps > 1:
            delta[:, 1:] = encoded[:, 1:] - encoded[:, :-1]

        local = self.local_context(
            encoded.reshape(batch_size * issue_steps, 1, source_steps)
        )
        local = F.gelu(local).reshape(
            batch_size,
            issue_steps,
            -1,
            source_steps,
        )
        local = local.permute(0, 3, 1, 2)

        source_index = torch.arange(
            source_steps,
            device=raw.device,
            dtype=raw.dtype,
        ).reshape(1, 1, source_steps)
        support_age = (
            horizons.reshape(1, issue_steps, 1)
            - distance.reshape(batch_size, 1, 1) / float(beta_m_per_s)
            - source_index
        )
        support_feature = torch.clamp(support_age / 30.0, min=-1.0, max=1.0)
        horizon_feature = (
            horizons / float(max(self.source_steps, 1))
        ).reshape(1, issue_steps, 1).expand(batch_size, -1, source_steps)

        raw_moment = torch.sum(
            raw * source_dt.reshape(batch_size, 1, 1),
            dim=2,
        ).clamp_min(1.0e10)
        raw_moment_feature = torch.clamp(
            (torch.log10(raw_moment) - 20.0) / 3.0,
            min=-2.0,
            max=2.0,
        ).unsqueeze(-1).expand(-1, -1, source_steps)

        scalar_features = torch.stack(
            (
                encoded,
                delta,
                support_feature,
                horizon_feature,
                raw_moment_feature,
            ),
            dim=-1,
        ).permute(0, 2, 1, 3)
        features = torch.cat((scalar_features, local), dim=-1)
        recurrent_input = features.reshape(
            batch_size * source_steps,
            issue_steps,
            features.shape[-1],
        )
        hidden, _ = self.temporal_gate(recurrent_input)
        logits = self.gate_head(hidden).reshape(
            batch_size,
            source_steps,
            issue_steps,
        ).permute(0, 2, 1)
        confidence = torch.clamp(
            support_age / self.support_ramp_sec,
            min=0.0,
            max=1.0,
        )
        gates = torch.sigmoid(logits) * confidence
        gates = gates.clone()
        gates[:, 0] = 0.0

        states = [raw[:, 0]]
        state = raw[:, 0]
        for issue_index in range(1, issue_steps):
            gate = gates[:, issue_index]
            state = gate * state + (1.0 - gate) * raw[:, issue_index]
            states.append(state)
        return torch.stack(states, dim=1), gates
