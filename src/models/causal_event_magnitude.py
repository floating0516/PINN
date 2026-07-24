from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class CausalRadialStationObservation:
    event: str
    station: str
    running_peak_cm: np.ndarray
    source_distance_km: float
    p_arrival_sec: float
    magnitude: float | None = None

    def __post_init__(self) -> None:
        peak = np.asarray(self.running_peak_cm, dtype=np.float64)
        if not self.event or not self.station:
            raise ValueError("event and station must be non-empty")
        if peak.ndim != 1 or peak.size < 1:
            raise ValueError("running_peak_cm must be a non-empty vector")
        if not np.isfinite(peak).all() or bool((peak < 0.0).any()):
            raise ValueError("running_peak_cm must be finite and nonnegative")
        if bool((np.diff(peak) < 0.0).any()):
            raise ValueError("running_peak_cm must be nondecreasing")
        if (
            not math.isfinite(self.source_distance_km)
            or self.source_distance_km <= 0.0
        ):
            raise ValueError("source_distance_km must be positive and finite")
        if not math.isfinite(self.p_arrival_sec) or self.p_arrival_sec < 0.0:
            raise ValueError("p_arrival_sec must be nonnegative and finite")
        if self.magnitude is not None and not math.isfinite(self.magnitude):
            raise ValueError("magnitude must be finite when provided")
        object.__setattr__(self, "running_peak_cm", peak.copy())


@dataclass(frozen=True)
class CausalRadialEventSpec:
    top_k: int = 5
    distance_exponents: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25)
    upper_quantile: float = 0.9
    duration_sec: int = 200
    sample_rate_hz: float = 1.0
    causal_latency_samples: int = 6
    hidden_dim: int = 32
    dropout: float = 0.05
    prefix_residual_scale: float = 0.25
    minimum_peak_cm: float = 1.0e-5

    def __post_init__(self) -> None:
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, Integral)
            or self.top_k < 1
        ):
            raise ValueError("top_k must be a positive integer")
        if not self.distance_exponents or not all(
            math.isfinite(value) for value in self.distance_exponents
        ):
            raise ValueError("distance_exponents must contain finite values")
        if not 0.5 < self.upper_quantile < 1.0:
            raise ValueError("upper_quantile must be between 0.5 and 1.0")
        if (
            isinstance(self.duration_sec, bool)
            or not isinstance(self.duration_sec, Integral)
            or self.duration_sec < 2
        ):
            raise ValueError("duration_sec must be an integer of at least two")
        if not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive and finite")
        if not math.isclose(self.sample_rate_hz, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("causal event model currently requires 1 Hz waveforms")
        if (
            isinstance(self.causal_latency_samples, bool)
            or not isinstance(self.causal_latency_samples, Integral)
            or not 0 <= self.causal_latency_samples < self.total_steps
        ):
            raise ValueError("causal_latency_samples is outside the event window")
        if (
            isinstance(self.hidden_dim, bool)
            or not isinstance(self.hidden_dim, Integral)
            or self.hidden_dim < 4
        ):
            raise ValueError("hidden_dim must be an integer of at least four")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if (
            not math.isfinite(self.prefix_residual_scale)
            or self.prefix_residual_scale <= 0.0
        ):
            raise ValueError("prefix_residual_scale must be positive and finite")
        if not math.isfinite(self.minimum_peak_cm) or self.minimum_peak_cm <= 0.0:
            raise ValueError("minimum_peak_cm must be positive and finite")

    @property
    def total_steps(self) -> int:
        return int(round(self.duration_sec * self.sample_rate_hz))

    @property
    def anchor_feature_count(self) -> int:
        return 2 * len(self.distance_exponents) + 2

    @property
    def feature_count(self) -> int:
        return self.anchor_feature_count + 5

    @property
    def time_fraction_index(self) -> int:
        return self.anchor_feature_count + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": int(self.top_k),
            "distance_exponents": list(self.distance_exponents),
            "upper_quantile": float(self.upper_quantile),
            "duration_sec": int(self.duration_sec),
            "sample_rate_hz": float(self.sample_rate_hz),
            "causal_latency_samples": int(self.causal_latency_samples),
            "hidden_dim": int(self.hidden_dim),
            "dropout": float(self.dropout),
            "prefix_residual_scale": float(self.prefix_residual_scale),
            "minimum_peak_cm": float(self.minimum_peak_cm),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CausalRadialEventSpec:
        values = dict(payload)
        values["distance_exponents"] = tuple(
            float(value) for value in values["distance_exponents"]
        )
        return cls(**values)


@dataclass(frozen=True)
class CausalEventSnapshot:
    event: str
    horizon_step: int
    horizon_sec: float
    features: np.ndarray
    active_station_count: int
    used_stations: tuple[str, ...]
    magnitude: float | None


def causal_running_peak_cm(
    radial_m: Sequence[float] | np.ndarray,
    valid_mask: Sequence[bool] | np.ndarray,
    *,
    latency_samples: int,
) -> np.ndarray:
    radial = np.asarray(radial_m, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if radial.ndim != 1 or radial.size < 1 or mask.shape != radial.shape:
        raise ValueError("radial_m and valid_mask must be equal non-empty vectors")
    if not np.isfinite(radial).all():
        raise ValueError("radial_m must be finite")
    if (
        isinstance(latency_samples, bool)
        or not isinstance(latency_samples, Integral)
        or not 0 <= latency_samples < radial.size
    ):
        raise ValueError("latency_samples is outside the waveform")

    delayed = np.zeros_like(radial)
    delayed_mask = np.zeros_like(mask)
    if latency_samples == 0:
        delayed[:] = radial
        delayed_mask[:] = mask
    else:
        delayed[latency_samples:] = radial[:-latency_samples]
        delayed_mask[latency_samples:] = mask[:-latency_samples]
    observed_cm = np.where(delayed_mask, np.abs(delayed) * 100.0, 0.0)
    return np.maximum.accumulate(observed_cm)


def causal_event_feature_names(
    spec: CausalRadialEventSpec,
) -> tuple[str, ...]:
    exponent_names = [f"{value:g}" for value in spec.distance_exponents]
    names = [
        f"running_corrected_median_gamma_{value}" for value in exponent_names
    ] + [
        f"running_corrected_q{int(round(spec.upper_quantile * 100)):02d}_gamma_{value}"
        for value in exponent_names
    ]
    names.extend(
        (
            "median_log10_distance_km",
            "log1p_used_station_count",
            "log1p_active_station_count",
            "time_fraction",
            "log_time_fraction",
            "median_post_arrival_age_fraction",
            "max_post_arrival_age_fraction",
        )
    )
    return tuple(names)


def build_causal_event_snapshot(
    observations: Sequence[CausalRadialStationObservation],
    *,
    horizon_step: int,
    spec: CausalRadialEventSpec,
) -> CausalEventSnapshot | None:
    if not observations:
        raise ValueError("at least one station observation is required")
    if (
        isinstance(horizon_step, bool)
        or not isinstance(horizon_step, Integral)
        or not 1 <= horizon_step <= spec.total_steps
    ):
        raise ValueError("horizon_step is outside the event window")
    events = {row.event for row in observations}
    if len(events) != 1:
        raise ValueError("event snapshot requires observations from one event")
    if any(row.running_peak_cm.size < spec.total_steps for row in observations):
        raise ValueError("station running peaks are shorter than the event window")
    magnitudes = {
        float(row.magnitude)
        for row in observations
        if row.magnitude is not None
    }
    if len(magnitudes) > 1:
        raise ValueError("station observations disagree on event magnitude")

    horizon_sec = float(horizon_step) / float(spec.sample_rate_hz)
    active = [
        row for row in observations if float(row.p_arrival_sec) <= horizon_sec
    ]
    if not active:
        return None
    peak_index = horizon_step - 1
    selected = sorted(
        active,
        key=lambda row: (-float(row.running_peak_cm[peak_index]), row.station),
    )[: int(spec.top_k)]
    peaks = np.maximum(
        np.asarray(
            [row.running_peak_cm[peak_index] for row in selected],
            dtype=np.float64,
        ),
        float(spec.minimum_peak_cm),
    )
    log_peak = np.log10(peaks)
    log_distance = np.log10(
        np.asarray([row.source_distance_km for row in selected], dtype=np.float64)
    )
    corrected = [
        log_peak + exponent * log_distance
        for exponent in spec.distance_exponents
    ]
    features = (
        [float(np.median(values)) for values in corrected]
        + [
            float(np.quantile(values, spec.upper_quantile))
            for values in corrected
        ]
        + [
            float(np.median(log_distance)),
            math.log1p(len(selected)),
            math.log1p(len(active)),
            horizon_step / spec.total_steps,
            math.log1p(horizon_step) / math.log1p(spec.total_steps),
        ]
    )
    post_arrival_age = np.asarray(
        [max(0.0, horizon_sec - row.p_arrival_sec) for row in selected],
        dtype=np.float64,
    )
    features.extend(
        (
            float(np.median(post_arrival_age)) / spec.duration_sec,
            float(np.max(post_arrival_age)) / spec.duration_sec,
        )
    )
    feature_array = np.asarray(features, dtype=np.float64)
    if (
        feature_array.shape != (spec.feature_count,)
        or not np.isfinite(feature_array).all()
    ):
        raise ValueError("causal event features have an invalid shape or value")
    return CausalEventSnapshot(
        event=next(iter(events)),
        horizon_step=int(horizon_step),
        horizon_sec=horizon_sec,
        features=feature_array,
        active_station_count=len(active),
        used_stations=tuple(row.station for row in selected),
        magnitude=(next(iter(magnitudes)) if magnitudes else None),
    )


class CausalRadialEventNet(nn.Module):
    """Event-Mw network with a final anchor and a causal prefix residual."""

    def __init__(
        self,
        *,
        anchor_feature_mean: Sequence[float],
        anchor_feature_scale: Sequence[float],
        online_feature_mean: Sequence[float],
        online_feature_scale: Sequence[float],
        target_mean: float,
        target_scale: float,
        spec: CausalRadialEventSpec,
    ) -> None:
        super().__init__()
        anchor_mean = torch.as_tensor(anchor_feature_mean, dtype=torch.float32)
        anchor_scale = torch.as_tensor(anchor_feature_scale, dtype=torch.float32)
        online_mean = torch.as_tensor(online_feature_mean, dtype=torch.float32)
        online_scale = torch.as_tensor(online_feature_scale, dtype=torch.float32)
        if anchor_mean.shape != (spec.anchor_feature_count,):
            raise ValueError("anchor feature normalization has the wrong shape")
        if anchor_scale.shape != anchor_mean.shape:
            raise ValueError("anchor feature scales have the wrong shape")
        if online_mean.shape != (spec.feature_count,):
            raise ValueError("online feature normalization has the wrong shape")
        if online_scale.shape != online_mean.shape:
            raise ValueError("online feature scales have the wrong shape")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (anchor_mean, anchor_scale, online_mean, online_scale)
        ):
            raise ValueError("feature normalization must be finite")
        if bool((anchor_scale <= 0.0).any()) or bool((online_scale <= 0.0).any()):
            raise ValueError("feature scales must be positive")
        if not math.isfinite(target_mean) or not math.isfinite(target_scale):
            raise ValueError("target normalization must be finite")
        if target_scale <= 0.0:
            raise ValueError("target_scale must be positive")

        self.spec = spec
        self.register_buffer("anchor_feature_mean", anchor_mean)
        self.register_buffer("anchor_feature_scale", anchor_scale)
        self.register_buffer("online_feature_mean", online_mean)
        self.register_buffer("online_feature_scale", online_scale)
        self.register_buffer(
            "target_mean",
            torch.tensor(float(target_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "target_scale",
            torch.tensor(float(target_scale), dtype=torch.float32),
        )
        self.anchor_branch = nn.Linear(spec.anchor_feature_count, 1)
        self.prefix_branch = nn.Sequential(
            nn.Linear(spec.feature_count, spec.hidden_dim),
            nn.LayerNorm(spec.hidden_dim),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_dim, spec.hidden_dim),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_dim, 1),
        )
        nn.init.zeros_(self.prefix_branch[-1].weight)
        nn.init.zeros_(self.prefix_branch[-1].bias)

    def standardized_components(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.spec.feature_count:
            raise ValueError("features must have shape (batch, feature_count)")
        anchor_features = features[:, : self.spec.anchor_feature_count]
        anchor_standardized = (
            anchor_features - self.anchor_feature_mean
        ) / self.anchor_feature_scale
        online_standardized = (
            features - self.online_feature_mean
        ) / self.online_feature_scale
        anchor = self.anchor_branch(anchor_standardized).squeeze(-1)
        gate = torch.clamp(
            1.0 - features[:, self.spec.time_fraction_index],
            min=0.0,
            max=1.0,
        )
        prefix_residual = (
            float(self.spec.prefix_residual_scale)
            * gate
            * self.prefix_branch(online_standardized).squeeze(-1)
        )
        return anchor, prefix_residual

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        anchor, prefix_residual = self.standardized_components(features)
        return self.target_mean + self.target_scale * (anchor + prefix_residual)


def select_single_seed(summaries: Mapping[int, Mapping[str, Any]]) -> int:
    if not summaries:
        raise ValueError("at least one seed summary is required")
    candidates: list[tuple[float, float, int]] = []
    for raw_seed, summary in summaries.items():
        seed = int(raw_seed)
        if "validation_online_mae" not in summary:
            raise ValueError(f"seed {seed} lacks validation_online_mae")
        if "validation_final_mae" not in summary:
            raise ValueError(f"seed {seed} lacks validation_final_mae")
        online = float(summary["validation_online_mae"])
        final = float(summary["validation_final_mae"])
        if not math.isfinite(online) or not math.isfinite(final):
            raise ValueError(f"seed {seed} selection metrics must be finite")
        candidates.append((online, final, seed))
    return min(candidates)[2]
