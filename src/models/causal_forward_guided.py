from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.models.causal_event_magnitude import CausalRadialEventSpec
from src.models.model import SinusoidalPositionalEncoding


@dataclass(frozen=True)
class CausalForwardGuidedSpec:
    top_k: int = 5
    distance_exponents: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25)
    upper_quantile: float = 0.9
    duration_sec: int = 200
    sample_rate_hz: float = 1.0
    causal_latency_samples: int = 6
    hidden_dim: int = 64
    num_tcn_blocks: int = 4
    transformer_num_layers: int = 2
    transformer_num_heads: int = 4
    dropout: float = 0.1
    magnitude_residual_scale: float = 0.25
    minimum_peak_cm: float = 1.0e-5

    def __post_init__(self) -> None:
        if (
            isinstance(self.hidden_dim, bool)
            or not isinstance(self.hidden_dim, Integral)
            or self.hidden_dim < 8
        ):
            raise ValueError("hidden_dim must be an integer of at least eight")
        integer_fields = {
            "num_tcn_blocks": self.num_tcn_blocks,
            "transformer_num_layers": self.transformer_num_layers,
            "transformer_num_heads": self.transformer_num_heads,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.transformer_num_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_num_heads")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if (
            not math.isfinite(self.magnitude_residual_scale)
            or self.magnitude_residual_scale <= 0.0
        ):
            raise ValueError("magnitude_residual_scale must be positive and finite")
        self.event_spec

    @property
    def event_spec(self) -> CausalRadialEventSpec:
        return CausalRadialEventSpec(
            top_k=self.top_k,
            distance_exponents=self.distance_exponents,
            upper_quantile=self.upper_quantile,
            duration_sec=self.duration_sec,
            sample_rate_hz=self.sample_rate_hz,
            causal_latency_samples=self.causal_latency_samples,
            hidden_dim=max(4, self.hidden_dim // 2),
            dropout=self.dropout,
            prefix_residual_scale=self.magnitude_residual_scale,
            minimum_peak_cm=self.minimum_peak_cm,
        )

    @property
    def total_steps(self) -> int:
        return self.event_spec.total_steps

    @property
    def anchor_feature_count(self) -> int:
        return self.event_spec.anchor_feature_count

    @property
    def feature_count(self) -> int:
        return self.event_spec.feature_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": int(self.top_k),
            "distance_exponents": list(self.distance_exponents),
            "upper_quantile": float(self.upper_quantile),
            "duration_sec": int(self.duration_sec),
            "sample_rate_hz": float(self.sample_rate_hz),
            "causal_latency_samples": int(self.causal_latency_samples),
            "hidden_dim": int(self.hidden_dim),
            "num_tcn_blocks": int(self.num_tcn_blocks),
            "transformer_num_layers": int(self.transformer_num_layers),
            "transformer_num_heads": int(self.transformer_num_heads),
            "dropout": float(self.dropout),
            "magnitude_residual_scale": float(self.magnitude_residual_scale),
            "minimum_peak_cm": float(self.minimum_peak_cm),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CausalForwardGuidedSpec:
        values = dict(payload)
        values["distance_exponents"] = tuple(
            float(value) for value in values["distance_exponents"]
        )
        return cls(**values)


@dataclass(frozen=True)
class CausalForwardGuidedPrediction:
    catalog_mw: torch.Tensor
    stf_encoded: torch.Tensor
    anchor_mw: torch.Tensor
    magnitude_residual: torch.Tensor


class _CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class _ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class _CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
        )
        self.norm1 = _ChannelLayerNorm(channels)
        self.conv2 = _CausalConv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=2 * dilation,
        )
        self.norm2 = _ChannelLayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = F.gelu(self.norm1(self.conv1(values)))
        update = F.gelu(self.norm2(self.conv2(update)))
        return values + self.dropout(update)


def _normalization_tensor(
    values: Sequence[float] | torch.Tensor,
    *,
    expected_size: int,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.shape != (expected_size,) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be a finite vector of size {expected_size}")
    return tensor


class CausalForwardGuidedEventNet(nn.Module):
    """Causal R-only TCN/Transformer with shared STF and event-Mw heads."""

    def __init__(
        self,
        *,
        anchor_feature_mean: Sequence[float] | torch.Tensor,
        anchor_feature_scale: Sequence[float] | torch.Tensor,
        online_feature_mean: Sequence[float] | torch.Tensor,
        online_feature_scale: Sequence[float] | torch.Tensor,
        metadata_mean: Sequence[float] | torch.Tensor,
        metadata_scale: Sequence[float] | torch.Tensor,
        target_mean: float,
        target_scale: float,
        stf_encoded_mean: Sequence[float] | torch.Tensor,
        spec: CausalForwardGuidedSpec,
    ) -> None:
        super().__init__()
        anchor_mean = _normalization_tensor(
            anchor_feature_mean,
            expected_size=spec.anchor_feature_count,
            name="anchor_feature_mean",
        )
        anchor_scale = _normalization_tensor(
            anchor_feature_scale,
            expected_size=spec.anchor_feature_count,
            name="anchor_feature_scale",
        )
        online_mean = _normalization_tensor(
            online_feature_mean,
            expected_size=spec.feature_count,
            name="online_feature_mean",
        )
        online_scale = _normalization_tensor(
            online_feature_scale,
            expected_size=spec.feature_count,
            name="online_feature_scale",
        )
        meta_mean = _normalization_tensor(
            metadata_mean,
            expected_size=5,
            name="metadata_mean",
        )
        meta_scale = _normalization_tensor(
            metadata_scale,
            expected_size=5,
            name="metadata_scale",
        )
        stf_mean = _normalization_tensor(
            stf_encoded_mean,
            expected_size=spec.total_steps,
            name="stf_encoded_mean",
        )
        if bool((anchor_scale <= 0.0).any()) or bool((online_scale <= 0.0).any()):
            raise ValueError("feature scales must be positive")
        if bool((meta_scale <= 0.0).any()):
            raise ValueError("metadata scales must be positive")
        if not math.isfinite(target_mean) or not math.isfinite(target_scale):
            raise ValueError("target normalization must be finite")
        if target_scale <= 0.0:
            raise ValueError("target_scale must be positive")

        self.spec = spec
        self.register_buffer("anchor_feature_mean", anchor_mean)
        self.register_buffer("anchor_feature_scale", anchor_scale)
        self.register_buffer("online_feature_mean", online_mean)
        self.register_buffer("online_feature_scale", online_scale)
        self.register_buffer("metadata_mean", meta_mean)
        self.register_buffer("metadata_scale", meta_scale)
        self.register_buffer(
            "target_mean", torch.tensor(float(target_mean), dtype=torch.float32)
        )
        self.register_buffer(
            "target_scale", torch.tensor(float(target_scale), dtype=torch.float32)
        )
        self.register_buffer("stf_encoded_mean", stf_mean)
        causal_mask = torch.triu(
            torch.ones(spec.total_steps, spec.total_steps, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_attention_mask", causal_mask, persistent=False)

        hidden = spec.hidden_dim
        self.waveform_embed = nn.Sequential(
            _CausalConv1d(1, hidden, kernel_size=7),
            _ChannelLayerNorm(hidden),
            nn.GELU(),
        )
        self.tcn_blocks = nn.ModuleList(
            _CausalResidualBlock(
                hidden,
                dilation=2**index,
                dropout=spec.dropout,
            )
            for index in range(spec.num_tcn_blocks)
        )
        self.metadata_embed = nn.Sequential(
            nn.Linear(5, hidden),
            nn.GELU(),
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            hidden,
            max_len=spec.total_steps,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=spec.transformer_num_heads,
            dim_feedforward=2 * hidden,
            dropout=spec.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=spec.transformer_num_layers,
            enable_nested_tensor=False,
        )
        self.post_transformer_norm = nn.LayerNorm(hidden)
        self.online_feature_embed = nn.Sequential(
            nn.Linear(spec.feature_count, hidden),
            nn.GELU(),
        )
        self.event_fusion = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(spec.dropout),
        )
        self.anchor_branch = nn.Linear(spec.anchor_feature_count, 1)
        self.magnitude_residual_head = nn.Sequential(
            nn.Linear(hidden, max(16, hidden // 2)),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(max(16, hidden // 2), 1),
        )
        self.stf_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(hidden, spec.total_steps),
        )
        nn.init.zeros_(self.magnitude_residual_head[-1].weight)
        nn.init.zeros_(self.magnitude_residual_head[-1].bias)
        nn.init.zeros_(self.stf_head[-1].weight)
        nn.init.zeros_(self.stf_head[-1].bias)

    def standardized_anchor(self, online_features: torch.Tensor) -> torch.Tensor:
        if online_features.ndim != 2 or online_features.shape[1] != self.spec.feature_count:
            raise ValueError("online_features has the wrong shape")
        anchor_features = online_features[:, : self.spec.anchor_feature_count]
        standardized = (
            anchor_features - self.anchor_feature_mean
        ) / self.anchor_feature_scale
        return self.anchor_branch(standardized).squeeze(-1)

    def _station_summary(
        self,
        radial_m: torch.Tensor,
        waveform_valid_mask: torch.Tensor,
        station_metadata: torch.Tensor,
        station_mask: torch.Tensor,
        observed_steps: torch.Tensor,
    ) -> torch.Tensor:
        if radial_m.ndim != 3:
            raise ValueError("radial_m must have shape (batch, station, time)")
        batch_size, station_count, time_steps = radial_m.shape
        if time_steps != self.spec.total_steps:
            raise ValueError("radial_m time axis differs from the configured duration")
        if waveform_valid_mask.shape != radial_m.shape:
            raise ValueError("waveform_valid_mask must match radial_m")
        if station_metadata.shape != (batch_size, station_count, 5):
            raise ValueError("station_metadata has the wrong shape")
        if station_mask.shape != (batch_size, station_count):
            raise ValueError("station_mask has the wrong shape")
        if observed_steps.shape != (batch_size,):
            raise ValueError("observed_steps must have shape (batch,)")
        if bool((observed_steps < 1).any()) or bool((observed_steps > time_steps).any()):
            raise ValueError("observed_steps is outside the waveform")

        time_index = torch.arange(time_steps, device=radial_m.device)
        released = time_index.reshape(1, 1, -1) < observed_steps.reshape(-1, 1, 1)
        prefix_mask = (
            waveform_valid_mask.to(dtype=torch.bool)
            & released
            & station_mask.to(dtype=torch.bool).unsqueeze(-1)
        )
        usable_station = prefix_mask.any(dim=-1)
        if bool((~usable_station.any(dim=1)).any()):
            raise ValueError("every event must have a station with a released prefix")
        masked_waveform = torch.where(prefix_mask, radial_m, torch.zeros_like(radial_m))
        flattened = (masked_waveform * 100.0).reshape(
            batch_size * station_count,
            1,
            time_steps,
        )
        features = self.waveform_embed(flattened)
        for block in self.tcn_blocks:
            features = block(features)
        sequence = features.transpose(1, 2)
        flat_metadata = station_metadata.reshape(-1, 5)
        standardized_metadata = (
            flat_metadata - self.metadata_mean
        ) / self.metadata_scale
        sequence = sequence + self.metadata_embed(standardized_metadata).unsqueeze(1)
        sequence = self.positional_encoding(sequence)
        flat_prefix = prefix_mask.reshape(-1, time_steps)
        attention_valid = flat_prefix.clone()
        # A zero-valued causal sentinel prevents all-masked early attention rows.
        attention_valid[:, 0] = True
        sequence = self.transformer(
            sequence,
            mask=self.causal_attention_mask,
            src_key_padding_mask=~attention_valid,
        )
        sequence = self.post_transformer_norm(sequence)
        weights = flat_prefix.to(dtype=sequence.dtype).unsqueeze(-1)
        pooled = torch.sum(sequence * weights, dim=1) / weights.sum(dim=1).clamp_min(1.0)
        station_features = pooled.reshape(batch_size, station_count, -1)
        usable_float = usable_station.to(dtype=station_features.dtype).unsqueeze(-1)
        mean_feature = torch.sum(station_features * usable_float, dim=1) / (
            usable_float.sum(dim=1).clamp_min(1.0)
        )
        maximum_feature = station_features.masked_fill(
            ~usable_station.unsqueeze(-1),
            -torch.inf,
        ).amax(dim=1)
        return torch.cat((mean_feature, maximum_feature), dim=1)

    def forward(
        self,
        *,
        radial_m: torch.Tensor,
        waveform_valid_mask: torch.Tensor,
        station_metadata: torch.Tensor,
        station_mask: torch.Tensor,
        observed_steps: torch.Tensor,
        online_features: torch.Tensor,
    ) -> CausalForwardGuidedPrediction:
        station_summary = self._station_summary(
            radial_m,
            waveform_valid_mask,
            station_metadata,
            station_mask,
            observed_steps,
        )
        if online_features.ndim != 2 or online_features.shape[1] != self.spec.feature_count:
            raise ValueError("online_features has the wrong shape")
        online_standardized = (
            online_features - self.online_feature_mean
        ) / self.online_feature_scale
        engineered = self.online_feature_embed(online_standardized)
        event_hidden = self.event_fusion(torch.cat((station_summary, engineered), dim=1))
        anchor_standardized = self.standardized_anchor(online_features)
        residual_standardized = float(self.spec.magnitude_residual_scale) * torch.tanh(
            self.magnitude_residual_head(event_hidden).squeeze(-1)
        )
        anchor_mw = self.target_mean + self.target_scale * anchor_standardized
        magnitude_residual = self.target_scale * residual_standardized
        catalog_mw = anchor_mw + magnitude_residual
        stf_encoded = torch.relu(self.stf_encoded_mean + self.stf_head(event_hidden))
        return CausalForwardGuidedPrediction(
            catalog_mw=catalog_mw,
            stf_encoded=stf_encoded,
            anchor_mw=anchor_mw,
            magnitude_residual=magnitude_residual,
        )
