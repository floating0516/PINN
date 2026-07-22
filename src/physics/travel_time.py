from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class TravelTimeDelays:
    p_sec: Any
    s_sec: Any
    s_after_p_sec: Any


@dataclass(frozen=True)
class ConstantVelocityTravelTime:
    alpha_m_per_s: float
    beta_m_per_s: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.alpha_m_per_s)
            or self.alpha_m_per_s <= 0.0
        ):
            raise ValueError("alpha_m_per_s must be positive and finite")
        if (
            not math.isfinite(self.beta_m_per_s)
            or self.beta_m_per_s <= 0.0
        ):
            raise ValueError("beta_m_per_s must be positive and finite")

    def delays(self, source_distance_m: Any) -> TravelTimeDelays:
        p_sec = source_distance_m / self.alpha_m_per_s
        s_sec = source_distance_m / self.beta_m_per_s
        return TravelTimeDelays(
            p_sec=p_sec,
            s_sec=s_sec,
            s_after_p_sec=s_sec - p_sec,
        )


def travel_time_from_config(
    config: dict[str, Any],
) -> ConstantVelocityTravelTime:
    physics = config.get("physics")
    if not isinstance(physics, dict):
        raise ValueError("config.physics must be a mapping")
    if physics.get("travel_time_model") != "constant_velocity":
        raise ValueError(
            "physics.travel_time_model must be constant_velocity"
        )
    return ConstantVelocityTravelTime(
        alpha_m_per_s=float(physics["alpha"]),
        beta_m_per_s=float(physics["beta"]),
    )
