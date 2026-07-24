import numpy as np
import pytest
import torch

from src.models.event_magnitude import (
    RadialPINNEventNet,
    RadialPINNEventSpec,
    RadialPINNStationObservation,
    build_radial_pinn_event_features,
    radial_pinn_event_feature_names,
)


def _observations(event: str = "A") -> list[RadialPINNStationObservation]:
    return [
        RadialPINNStationObservation(
            event=event,
            station=f"S{index}",
            radial_peak_cm=float(index + 1),
            source_distance_km=float(20 * (index + 1)),
            pinn_mw=(6.8 + index / 20, 6.9 + index / 20, 7.0 + index / 20),
            magnitude=7.2,
        )
        for index in range(7)
    ]


def test_radial_pinn_event_features_are_top_five_and_deterministic() -> None:
    spec = RadialPINNEventSpec()
    observations = _observations()

    features = build_radial_pinn_event_features(observations, spec)
    observations[0] = RadialPINNStationObservation(
        event="A",
        station="S0",
        radial_peak_cm=0.01,
        source_distance_km=999.0,
        pinn_mw=(4.0, 4.0, 4.0),
        magnitude=7.2,
    )
    unchanged = build_radial_pinn_event_features(observations, spec)

    assert features.shape == (28,)
    assert len(radial_pinn_event_feature_names(spec)) == 28
    np.testing.assert_array_equal(features, unchanged)
    assert features[9] == pytest.approx(np.log1p(5))


def test_radial_pinn_event_net_is_nonlinear_and_backpropagates() -> None:
    spec = RadialPINNEventSpec(hidden_dim=16, dropout=0.0)
    feature_count = len(radial_pinn_event_feature_names(spec))
    model = RadialPINNEventNet(
        feature_mean=np.zeros(feature_count),
        feature_scale=np.ones(feature_count),
        target_mean=7.5,
        target_scale=0.5,
        spec=spec,
    )
    features = torch.randn(4, feature_count)

    predictions = model(features)
    linear, nonlinear = model.standardized_components(features)
    predictions.square().mean().backward()

    assert predictions.shape == (4,)
    assert linear.shape == nonlinear.shape == (4,)
    assert not torch.equal(nonlinear, torch.zeros_like(nonlinear))
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_radial_pinn_event_checkpoint_round_trip() -> None:
    spec = RadialPINNEventSpec(hidden_dim=8, dropout=0.0)
    feature_count = len(radial_pinn_event_feature_names(spec))
    model = RadialPINNEventNet(
        feature_mean=np.arange(feature_count),
        feature_scale=np.linspace(1.0, 2.0, feature_count),
        target_mean=7.4,
        target_scale=0.6,
        spec=spec,
    ).eval()
    restored_spec = RadialPINNEventSpec.from_dict(spec.to_dict())
    restored = RadialPINNEventNet(
        feature_mean=np.zeros(feature_count),
        feature_scale=np.ones(feature_count),
        target_mean=0.0,
        target_scale=1.0,
        spec=restored_spec,
    ).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    features = torch.randn(3, feature_count)

    with torch.no_grad():
        expected = model(features)
        actual = restored(features)

    assert torch.equal(actual, expected)
