from __future__ import annotations

import pytest
import torch

from scripts.experiments.run_phase39_causal_direct import (
    augment_prepared_with_moment_scaling,
    causal_curriculum_scale,
    direct_causal_objective,
    error_descent_loss,
    event_aggregated_magnitude_loss,
    initialize_direct_model,
    moment_scale_from_delta_mw,
    ordered_prefix_horizons_for_step,
    prefix_horizon_for_step,
)
from src.training.train import _PreparedV2Batch


def test_prefix_schedule_covers_every_nonendpoint_horizon() -> None:
    values = [
        prefix_horizon_for_step(
            step,
            seed=73,
            minimum=5,
            maximum=199,
            multiplier=73,
        )
        for step in range(195)
    ]

    assert len(set(values)) == 195
    assert min(values) == 5
    assert max(values) == 199


def test_ordered_prefix_pairs_cover_range_with_fixed_gap() -> None:
    pairs = [
        ordered_prefix_horizons_for_step(
            step,
            seed=73,
            minimum=5,
            maximum=199,
            multiplier=73,
            gap_sec=30,
        )
        for step in range(165)
    ]

    assert len(set(pairs)) == 165
    assert min(early for early, _ in pairs) == 5
    assert max(late for _, late in pairs) == 199
    assert all(late - early == 30 for early, late in pairs)


def test_causal_curriculum_ramps_to_full_strength() -> None:
    assert causal_curriculum_scale(
        10,
        start_epoch=10,
        full_epoch=30,
    ) == pytest.approx(0.0)
    assert causal_curriculum_scale(
        20,
        start_epoch=10,
        full_epoch=30,
    ) == pytest.approx(0.5)
    assert causal_curriculum_scale(
        30,
        start_epoch=10,
        full_epoch=30,
    ) == pytest.approx(1.0)


def test_error_descent_updates_endpoint_without_degrading_prefix() -> None:
    prefix = torch.tensor([7.1], requires_grad=True)
    endpoint = torch.tensor([7.5], requires_grad=True)
    target = torch.tensor([7.0])

    loss, metrics = error_descent_loss(
        prefix,
        endpoint,
        target,
        slack_mw=0.03,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert metrics["violation_fraction"] == pytest.approx(1.0)
    assert prefix.grad is None
    assert endpoint.grad is not None
    assert endpoint.grad.item() > 0.0


def test_direct_objective_contains_only_three_declared_terms() -> None:
    prefix = torch.tensor(2.0)
    endpoint = torch.tensor(3.0)
    descent = torch.tensor(4.0)

    result = direct_causal_objective(
        prefix,
        endpoint,
        descent,
        prefix_weight=0.5,
        endpoint_weight=1.0,
        descent_weight=0.5,
    )

    assert result.item() == pytest.approx(6.0)


def test_direct_model_can_start_from_scratch_or_phase39() -> None:
    source = torch.nn.Linear(2, 1)
    scratch = torch.nn.Linear(2, 1)
    scratch_before = {key: value.clone() for key, value in scratch.state_dict().items()}

    initialize_direct_model(
        scratch,
        source.state_dict(),
        initialization="scratch",
    )
    assert all(
        torch.equal(scratch.state_dict()[key], value)
        for key, value in scratch_before.items()
    )

    initialize_direct_model(
        scratch,
        source.state_dict(),
        initialization="phase39_checkpoint",
    )
    assert all(
        torch.equal(scratch.state_dict()[key], value)
        for key, value in source.state_dict().items()
    )


def test_event_magnitude_loss_weights_events_equally() -> None:
    predicted = torch.tensor([7.0, 7.4, 8.0], requires_grad=True)
    target = torch.tensor([7.1, 7.1, 7.5])

    loss, metrics = event_aggregated_magnitude_loss(
        predicted,
        target,
        ["event-a", "event-a", "event-b"],
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.3)
    assert metrics["event_count"] == pytest.approx(2.0)
    assert predicted.grad is not None
    assert predicted.grad.tolist() == pytest.approx([0.25, 0.25, 0.5])


def test_moment_scaling_preserves_the_physical_amplitude_contract() -> None:
    prepared = _PreparedV2Batch(
        radial=torch.tensor([[2.0, 4.0]]),
        model_input=torch.tensor([[[2.0, 4.0]]]),
        source_distance_m=torch.tensor([10.0]),
        theta_deg=torch.tensor([20.0]),
        phi_slip_deg=torch.tensor([30.0]),
        source_dt_sec=torch.tensor([1.0]),
        observation_dt_sec=torch.tensor([1.0]),
        waveform_valid_mask=torch.tensor([[True, True]]),
        stf_true=torch.tensor([[3.0, 6.0]]),
        has_stf=torch.tensor([True]),
        true_mag=torch.tensor([7.0]),
        metadata=torch.tensor([[1.0, 2.0]]),
    )
    delta = torch.tensor([-2.0 / 3.0])

    augmented, weights = augment_prepared_with_moment_scaling(
        prepared,
        torch.tensor([1.0]),
        delta_mw=delta,
        augmentation_weight=0.5,
    )

    assert moment_scale_from_delta_mw(delta).item() == pytest.approx(0.1)
    torch.testing.assert_close(
        augmented.radial,
        torch.tensor([[2.0, 4.0], [0.2, 0.4]]),
    )
    torch.testing.assert_close(
        augmented.model_input[:, 0],
        torch.tensor([[2.0, 4.0], [0.2, 0.4]]),
    )
    torch.testing.assert_close(
        augmented.stf_true,
        torch.tensor([[3.0, 6.0], [0.3, 0.6]]),
    )
    assert augmented.true_mag.tolist() == pytest.approx([7.0, 7.0 - 2.0 / 3.0])
    assert augmented.source_distance_m.tolist() == pytest.approx([10.0, 10.0])
    torch.testing.assert_close(
        augmented.metadata,
        torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
    )
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights.tolist() == pytest.approx([4.0 / 3.0, 2.0 / 3.0])
