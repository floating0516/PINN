from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.causal_event_magnitude import (
    CausalRadialEventNet,
    CausalRadialEventSpec,
    CausalRadialStationObservation,
    build_causal_event_snapshot,
    causal_running_peak_cm,
    select_single_seed,
)


def _observation(
    station: str,
    radial_m: list[float],
    *,
    latency_samples: int = 2,
) -> CausalRadialStationObservation:
    radial = np.asarray(radial_m, dtype=np.float64)
    return CausalRadialStationObservation(
        event="Event",
        station=station,
        running_peak_cm=causal_running_peak_cm(
            radial,
            np.ones_like(radial, dtype=bool),
            latency_samples=latency_samples,
        ),
        source_distance_km=100.0,
        p_arrival_sec=0.0,
        magnitude=7.0,
    )


def test_current_snapshot_ignores_future_and_changes_dynamic_top_station() -> None:
    spec = CausalRadialEventSpec(
        top_k=1,
        duration_sec=6,
        sample_rate_hz=1.0,
        causal_latency_samples=2,
    )
    first = _observation("A", [0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
    future_small = _observation("B", [0.01, 0.01, 0.01, 0.01, 0.01, 1.0])
    future_large = _observation("B", [0.01, 0.01, 0.01, 100.0, 0.01, 1.0])

    snapshot_small = build_causal_event_snapshot(
        [first, future_small],
        horizon_step=4,
        spec=spec,
    )
    snapshot_large = build_causal_event_snapshot(
        [first, future_large],
        horizon_step=4,
        spec=spec,
    )

    assert snapshot_small is not None
    assert snapshot_large is not None
    assert snapshot_small.used_stations == ("A",)
    assert snapshot_large.used_stations == ("A",)
    np.testing.assert_array_equal(snapshot_small.features, snapshot_large.features)

    final_snapshot = build_causal_event_snapshot(
        [first, future_large],
        horizon_step=6,
        spec=spec,
    )
    assert final_snapshot is not None
    assert final_snapshot.used_stations == ("B",)


def test_prefix_residual_is_exactly_zero_at_final_second() -> None:
    spec = CausalRadialEventSpec(
        duration_sec=4,
        sample_rate_hz=1.0,
        causal_latency_samples=1,
        hidden_dim=8,
        dropout=0.0,
    )
    feature_count = spec.feature_count
    model = CausalRadialEventNet(
        anchor_feature_mean=np.zeros(spec.anchor_feature_count),
        anchor_feature_scale=np.ones(spec.anchor_feature_count),
        online_feature_mean=np.zeros(feature_count),
        online_feature_scale=np.ones(feature_count),
        target_mean=7.0,
        target_scale=1.0,
        spec=spec,
    )
    with torch.no_grad():
        model.prefix_branch[-1].weight.fill_(0.1)
        model.prefix_branch[-1].bias.fill_(0.2)

    early = torch.zeros(1, feature_count)
    early[:, spec.time_fraction_index] = 0.5
    final = early.clone()
    final[:, spec.time_fraction_index] = 1.0
    _, early_residual = model.standardized_components(early)
    _, final_residual = model.standardized_components(final)

    assert early_residual.abs().item() > 0.0
    assert final_residual.item() == 0.0
    model(final).sum().backward()
    assert model.anchor_branch.weight.grad is not None
    assert model.prefix_branch[-1].weight.grad is not None
    assert torch.count_nonzero(model.prefix_branch[-1].weight.grad) == 0


def test_single_seed_selection_never_averages_predictions() -> None:
    summaries = {
        17: {"validation_online_mae": 0.21, "validation_final_mae": 0.18},
        42: {"validation_online_mae": 0.19, "validation_final_mae": 0.20},
        73: {"validation_online_mae": 0.19, "validation_final_mae": 0.17},
    }

    assert select_single_seed(summaries) == 73

    with pytest.raises(ValueError, match="validation_online_mae"):
        select_single_seed({17: {"validation_final_mae": 0.1}})
