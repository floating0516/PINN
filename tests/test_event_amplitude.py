import numpy as np
import pytest

from src.baseline.event_amplitude import (
    RadialEventRidgeModel,
    RadialEventRidgeSpec,
    RadialStationObservation,
    build_radial_event_features,
    fit_radial_event_ridge,
)


def _event(event: str, magnitude: float, scale: float):
    return [
        RadialStationObservation(
            event=event,
            station=f"S{index}",
            radial_peak_cm=scale * (index + 1),
            source_distance_km=20.0 * (index + 1),
            magnitude=magnitude,
        )
        for index in range(7)
    ]


def test_event_features_use_top_five_stations_and_fixed_schema():
    observations = _event("A", 7.0, 1.0)
    spec = RadialEventRidgeSpec()

    features = build_radial_event_features(observations, spec)

    assert features.shape == (10,)
    selected = observations[2:]
    corrected = np.asarray(
        [
            np.log10(row.radial_peak_cm)
            + 0.5 * np.log10(row.source_distance_km)
            for row in selected
        ]
    )
    assert features[0] == pytest.approx(float(np.median(corrected)))
    assert features[4] == pytest.approx(float(np.quantile(corrected, 0.9)))
    assert features[-1] == pytest.approx(np.log1p(5))


def test_fitted_event_ridge_round_trips_and_predicts_finite_value():
    observations = (
        _event("A", 6.5, 0.5)
        + _event("B", 7.0, 1.0)
        + _event("C", 7.5, 2.0)
        + _event("D", 8.0, 4.0)
    )
    model = fit_radial_event_ridge(observations)
    restored = RadialEventRidgeModel.from_dict(model.to_dict())

    assert restored.training_events == ("A", "B", "C", "D")
    assert restored.training_observation_count == 28
    assert restored.predict_event(_event("X", 7.2, 1.5)) == pytest.approx(
        model.predict_event(_event("X", 7.2, 1.5)),
        abs=1.0e-12,
    )
