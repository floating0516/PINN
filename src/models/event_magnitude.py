from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class RadialPINNStationObservation:
    event: str
    station: str
    radial_peak_cm: float
    source_distance_km: float
    pinn_mw: tuple[float, ...]
    magnitude: float | None = None

    def __post_init__(self) -> None:
        if not self.event or not self.station:
            raise ValueError("event and station must be non-empty")
        if not math.isfinite(self.radial_peak_cm) or self.radial_peak_cm <= 0.0:
            raise ValueError("radial_peak_cm must be positive and finite")
        if (
            not math.isfinite(self.source_distance_km)
            or self.source_distance_km <= 0.0
        ):
            raise ValueError("source_distance_km must be positive and finite")
        if not self.pinn_mw or not all(math.isfinite(value) for value in self.pinn_mw):
            raise ValueError("pinn_mw must contain finite predictions")
        if self.magnitude is not None and not math.isfinite(self.magnitude):
            raise ValueError("magnitude must be finite when provided")


@dataclass(frozen=True)
class RadialPINNEventSpec:
    top_k: int = 5
    distance_exponents: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25)
    upper_quantile: float = 0.9
    pinn_view_names: tuple[str, ...] = ("seed_17", "seed_42", "seed_73")
    hidden_dim: int = 32
    dropout: float = 0.05
    nonlinear_scale: float = 0.25

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
        if not self.pinn_view_names or any(not name for name in self.pinn_view_names):
            raise ValueError("pinn_view_names must be non-empty")
        if len(set(self.pinn_view_names)) != len(self.pinn_view_names):
            raise ValueError("pinn_view_names must be unique")
        if (
            isinstance(self.hidden_dim, bool)
            or not isinstance(self.hidden_dim, Integral)
            or self.hidden_dim < 4
        ):
            raise ValueError("hidden_dim must be an integer of at least four")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.nonlinear_scale) or self.nonlinear_scale <= 0.0:
            raise ValueError("nonlinear_scale must be positive and finite")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["distance_exponents"] = list(self.distance_exponents)
        payload["pinn_view_names"] = list(self.pinn_view_names)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RadialPINNEventSpec:
        values = dict(payload)
        values["distance_exponents"] = tuple(
            float(value) for value in values["distance_exponents"]
        )
        values["pinn_view_names"] = tuple(
            str(value) for value in values["pinn_view_names"]
        )
        return cls(**values)


def radial_pinn_event_feature_names(
    spec: RadialPINNEventSpec,
) -> tuple[str, ...]:
    exponent_names = [f"{value:g}" for value in spec.distance_exponents]
    amplitude_names = [
        f"corrected_median_gamma_{value}" for value in exponent_names
    ] + [
        f"corrected_q{int(round(spec.upper_quantile * 100)):02d}_gamma_{value}"
        for value in exponent_names
    ] + ["median_log10_distance_km", "log1p_station_count"]
    prediction_names: list[str] = []
    for view_name in spec.pinn_view_names:
        prediction_names.extend(
            f"pinn_{view_name}_{statistic}"
            for statistic in ("median", "q90", "mean", "std")
        )
    prediction_names.extend(
        f"pinn_ensemble_{statistic}"
        for statistic in ("median", "q90", "mean", "std")
    )
    prediction_names.extend(
        ("pinn_disagreement_mean", "pinn_disagreement_q90")
    )
    return tuple(amplitude_names + prediction_names)


def _selected_observations(
    observations: Sequence[RadialPINNStationObservation],
    spec: RadialPINNEventSpec,
) -> list[RadialPINNStationObservation]:
    if not observations:
        raise ValueError("at least one station observation is required")
    if len({row.event for row in observations}) != 1:
        raise ValueError("event features require observations from one event")
    if any(len(row.pinn_mw) != len(spec.pinn_view_names) for row in observations):
        raise ValueError("station PINN views do not match the event specification")
    return sorted(
        observations,
        key=lambda row: (-row.radial_peak_cm, row.station),
    )[: spec.top_k]


def _summary_features(values: np.ndarray, quantile: float) -> list[float]:
    return [
        float(np.median(values)),
        float(np.quantile(values, quantile)),
        float(np.mean(values)),
        float(np.std(values)),
    ]


def build_radial_pinn_event_features(
    observations: Sequence[RadialPINNStationObservation],
    spec: RadialPINNEventSpec,
) -> np.ndarray:
    selected = _selected_observations(observations, spec)
    log_peak = np.log10(
        np.asarray([row.radial_peak_cm for row in selected], dtype=np.float64)
    )
    log_distance = np.log10(
        np.asarray(
            [row.source_distance_km for row in selected],
            dtype=np.float64,
        )
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
        + [float(np.median(log_distance)), math.log1p(len(selected))]
    )

    predictions = np.asarray([row.pinn_mw for row in selected], dtype=np.float64)
    for view_index in range(predictions.shape[1]):
        features.extend(
            _summary_features(
                predictions[:, view_index],
                spec.upper_quantile,
            )
        )
    ensemble_prediction = predictions.mean(axis=1)
    features.extend(
        _summary_features(ensemble_prediction, spec.upper_quantile)
    )
    disagreement = predictions.std(axis=1)
    features.extend(
        [
            float(np.mean(disagreement)),
            float(np.quantile(disagreement, spec.upper_quantile)),
        ]
    )
    result = np.asarray(features, dtype=np.float64)
    expected = len(radial_pinn_event_feature_names(spec))
    if result.shape != (expected,) or not np.isfinite(result).all():
        raise ValueError("event features have an invalid shape or value")
    return result


class RadialPINNEventNet(nn.Module):
    """Neural event head over R-only amplitude and frozen PINN predictions."""

    def __init__(
        self,
        *,
        feature_mean: Sequence[float],
        feature_scale: Sequence[float],
        target_mean: float,
        target_scale: float,
        spec: RadialPINNEventSpec,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(feature_mean, dtype=torch.float32)
        scale = torch.as_tensor(feature_scale, dtype=torch.float32)
        if mean.ndim != 1 or scale.shape != mean.shape or mean.numel() < 1:
            raise ValueError("feature normalization tensors must be one-dimensional")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
            raise ValueError("feature normalization must be finite")
        if bool((scale <= 0.0).any()):
            raise ValueError("feature_scale must be positive")
        if not math.isfinite(target_mean) or not math.isfinite(target_scale):
            raise ValueError("target normalization must be finite")
        if target_scale <= 0.0:
            raise ValueError("target_scale must be positive")

        self.spec = spec
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale)
        self.register_buffer(
            "target_mean",
            torch.tensor(float(target_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "target_scale",
            torch.tensor(float(target_scale), dtype=torch.float32),
        )
        feature_count = int(mean.numel())
        self.amplitude_feature_count = 2 * len(spec.distance_exponents) + 2
        if self.amplitude_feature_count >= feature_count:
            raise ValueError("event features must include PINN residual inputs")
        hidden_dim = int(spec.hidden_dim)
        self.linear_branch = nn.Linear(self.amplitude_feature_count, 1)
        self.nonlinear_branch = nn.Sequential(
            nn.Linear(feature_count, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(spec.dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(spec.dropout)),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.nonlinear_branch[-1].weight)
        nn.init.zeros_(self.nonlinear_branch[-1].bias)

    def standardized_components(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.feature_mean.numel():
            raise ValueError("features must have shape (batch, feature_count)")
        standardized = (features - self.feature_mean) / self.feature_scale
        linear = self.linear_branch(
            standardized[:, : self.amplitude_feature_count]
        ).squeeze(-1)
        nonlinear = (
            float(self.spec.nonlinear_scale)
            * self.nonlinear_branch(standardized).squeeze(-1)
        )
        return linear, nonlinear

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        linear, nonlinear = self.standardized_components(features)
        return self.target_mean + self.target_scale * (linear + nonlinear)
