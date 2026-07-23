from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from numbers import Integral
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RadialStationObservation:
    event: str
    station: str
    radial_peak_cm: float
    source_distance_km: float
    magnitude: float | None = None

    def __post_init__(self) -> None:
        if not self.event:
            raise ValueError("event must be non-empty")
        if not math.isfinite(self.radial_peak_cm) or self.radial_peak_cm <= 0.0:
            raise ValueError("radial_peak_cm must be positive and finite")
        if (
            not math.isfinite(self.source_distance_km)
            or self.source_distance_km <= 0.0
        ):
            raise ValueError("source_distance_km must be positive and finite")
        if self.magnitude is not None and not math.isfinite(self.magnitude):
            raise ValueError("magnitude must be finite when provided")


@dataclass(frozen=True)
class RadialEventRidgeSpec:
    top_k: int = 5
    distance_exponents: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25)
    upper_quantile: float = 0.9
    ridge_alpha: float = 10.0

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
        if not math.isfinite(self.ridge_alpha) or self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be nonnegative and finite")


def radial_event_feature_names(spec: RadialEventRidgeSpec) -> tuple[str, ...]:
    exponent_names = [f"{value:g}" for value in spec.distance_exponents]
    return tuple(
        [f"corrected_median_gamma_{value}" for value in exponent_names]
        + [
            f"corrected_q{int(round(spec.upper_quantile * 100)):02d}_gamma_{value}"
            for value in exponent_names
        ]
        + ["median_log10_distance_km", "log1p_station_count"]
    )


def _top_observations(
    observations: Sequence[RadialStationObservation],
    spec: RadialEventRidgeSpec,
) -> list[RadialStationObservation]:
    if not observations:
        raise ValueError("at least one station observation is required")
    events = {observation.event for observation in observations}
    if len(events) != 1:
        raise ValueError("event features require observations from one event")
    return sorted(
        observations,
        key=lambda observation: (
            -observation.radial_peak_cm,
            observation.station,
        ),
    )[: spec.top_k]


def build_radial_event_features(
    observations: Sequence[RadialStationObservation],
    spec: RadialEventRidgeSpec,
) -> np.ndarray:
    selected = _top_observations(observations, spec)
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
    features = np.asarray(
        [float(np.median(values)) for values in corrected]
        + [
            float(np.quantile(values, spec.upper_quantile))
            for values in corrected
        ]
        + [float(np.median(log_distance)), math.log1p(len(selected))],
        dtype=np.float64,
    )
    if not np.isfinite(features).all():
        raise ValueError("event features must be finite")
    return features


def _group_observations(
    observations: Iterable[RadialStationObservation],
) -> dict[str, list[RadialStationObservation]]:
    grouped: dict[str, list[RadialStationObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.event].append(observation)
    if not grouped:
        raise ValueError("no station observations supplied")
    return dict(grouped)


def _event_magnitude(observations: Sequence[RadialStationObservation]) -> float:
    magnitudes = {
        float(row.magnitude) for row in observations if row.magnitude is not None
    }
    if len(magnitudes) != 1:
        raise ValueError("each training event must have one finite magnitude")
    return magnitudes.pop()


@dataclass(frozen=True)
class RadialEventRidgeModel:
    spec: RadialEventRidgeSpec
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    target_mean: float
    coefficients: tuple[float, ...]
    training_events: tuple[str, ...]
    training_event_magnitudes: tuple[float, ...]
    training_observation_count: int

    def predict_event(
        self,
        observations: Sequence[RadialStationObservation],
    ) -> float:
        features = build_radial_event_features(observations, self.spec)
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if features.shape != mean.shape or mean.shape != coefficients.shape:
            raise ValueError("serialized model feature dimensions are inconsistent")
        prediction = self.target_mean + float(
            ((features - mean) / scale) @ coefficients
        )
        if not math.isfinite(prediction):
            raise ValueError("event prediction is not finite")
        return prediction

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "radial_event_ridge_v1",
            "spec": {
                **asdict(self.spec),
                "distance_exponents": list(self.spec.distance_exponents),
            },
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "target_mean": self.target_mean,
            "coefficients": list(self.coefficients),
            "training_events": list(self.training_events),
            "training_event_magnitudes": list(
                self.training_event_magnitudes
            ),
            "training_observation_count": self.training_observation_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RadialEventRidgeModel:
        if payload.get("model_type") != "radial_event_ridge_v1":
            raise ValueError("unsupported radial event model type")
        spec_payload = dict(payload["spec"])
        spec_payload["distance_exponents"] = tuple(
            float(value) for value in spec_payload["distance_exponents"]
        )
        return cls(
            spec=RadialEventRidgeSpec(**spec_payload),
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            feature_mean=tuple(float(value) for value in payload["feature_mean"]),
            feature_scale=tuple(float(value) for value in payload["feature_scale"]),
            target_mean=float(payload["target_mean"]),
            coefficients=tuple(float(value) for value in payload["coefficients"]),
            training_events=tuple(str(value) for value in payload["training_events"]),
            training_event_magnitudes=tuple(
                float(value) for value in payload["training_event_magnitudes"]
            ),
            training_observation_count=int(payload["training_observation_count"]),
        )


def fit_radial_event_ridge(
    observations: Iterable[RadialStationObservation],
    spec: RadialEventRidgeSpec | None = None,
) -> RadialEventRidgeModel:
    active_spec = spec or RadialEventRidgeSpec()
    observation_list = list(observations)
    grouped = _group_observations(observation_list)
    events = tuple(sorted(grouped))
    feature_names = radial_event_feature_names(active_spec)
    feature_matrix = np.stack(
        [build_radial_event_features(grouped[event], active_spec) for event in events]
    )
    targets = np.asarray(
        [_event_magnitude(grouped[event]) for event in events],
        dtype=np.float64,
    )
    if feature_matrix.shape[0] < 2:
        raise ValueError("at least two training events are required")
    feature_mean = feature_matrix.mean(axis=0)
    feature_scale = feature_matrix.std(axis=0)
    feature_scale[feature_scale < 1.0e-8] = 1.0
    standardized = (feature_matrix - feature_mean) / feature_scale
    target_mean = float(targets.mean())
    centered_targets = targets - target_mean
    penalty = np.eye(standardized.shape[1], dtype=np.float64)
    coefficients = np.linalg.pinv(
        standardized.T @ standardized + active_spec.ridge_alpha * penalty
    ) @ standardized.T @ centered_targets
    return RadialEventRidgeModel(
        spec=active_spec,
        feature_names=feature_names,
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_scale=tuple(float(value) for value in feature_scale),
        target_mean=target_mean,
        coefficients=tuple(float(value) for value in coefficients),
        training_events=events,
        training_event_magnitudes=tuple(float(value) for value in targets),
        training_observation_count=len(observation_list),
    )


def predict_grouped_events(
    model: RadialEventRidgeModel,
    observations: Iterable[RadialStationObservation],
) -> list[dict[str, Any]]:
    grouped = _group_observations(observations)
    rows: list[dict[str, Any]] = []
    for event in sorted(grouped):
        event_observations = grouped[event]
        magnitudes = {
            float(row.magnitude)
            for row in event_observations
            if row.magnitude is not None
        }
        reference = magnitudes.pop() if len(magnitudes) == 1 else None
        prediction = model.predict_event(event_observations)
        row: dict[str, Any] = {
            "event": event,
            "mw_pred": prediction,
            "station_count_available": len(event_observations),
            "station_count_used": min(len(event_observations), model.spec.top_k),
        }
        if reference is not None:
            row.update(
                {
                    "mw_reference": reference,
                    "error": prediction - reference,
                    "abs_error": abs(prediction - reference),
                }
            )
        rows.append(row)
    return rows


def leave_one_event_out_predictions(
    observations: Iterable[RadialStationObservation],
    spec: RadialEventRidgeSpec | None = None,
) -> list[dict[str, Any]]:
    active_spec = spec or RadialEventRidgeSpec()
    observation_list = list(observations)
    grouped = _group_observations(observation_list)
    rows: list[dict[str, Any]] = []
    for event in sorted(grouped):
        training = [row for row in observation_list if row.event != event]
        model = fit_radial_event_ridge(training, active_spec)
        prediction = model.predict_event(grouped[event])
        reference = _event_magnitude(grouped[event])
        rows.append(
            {
                "event": event,
                "mw_pred": prediction,
                "mw_reference": reference,
                "error": prediction - reference,
                "abs_error": abs(prediction - reference),
                "station_count_available": len(grouped[event]),
                "station_count_used": min(len(grouped[event]), active_spec.top_k),
            }
        )
    return rows
