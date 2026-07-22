import inspect

import torch

from src.training.loss_stf_rate_v2 import (
    compute_physical_coefficients,
    forward_displacement_from_rate,
)
from src.training.time_sampling import sample_source_history


def test_fractional_delay_uses_linear_interpolation() -> None:
    source = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    sampled = sample_source_history(
        source,
        source_dt_sec=torch.tensor([1.0]),
        observation_dt_sec=torch.tensor([1.0]),
        delay_sec=torch.tensor([1.5]),
        observation_steps=5,
    )

    assert sampled.shape == (1, 5)
    assert torch.allclose(
        sampled[0],
        torch.tensor([0.0, 0.0, 0.5, 0.5, 0.0]),
        atol=1.0e-6,
    )


def test_delay_larger_than_window_returns_zeros() -> None:
    sampled = sample_source_history(
        torch.ones(1, 4),
        source_dt_sec=torch.tensor([1.0]),
        observation_dt_sec=torch.tensor([1.0]),
        delay_sec=torch.tensor([10.0]),
        observation_steps=5,
    )

    assert torch.count_nonzero(sampled) == 0


def test_exact_last_source_sample_is_not_dropped() -> None:
    sampled = sample_source_history(
        torch.tensor([[0.0, 1.0, 2.0]]),
        source_dt_sec=torch.tensor([1.0]),
        observation_dt_sec=torch.tensor([1.0]),
        delay_sec=torch.tensor([0.0]),
        observation_steps=3,
    )

    assert torch.equal(sampled, torch.tensor([[0.0, 1.0, 2.0]]))


def test_fractional_sampling_preserves_source_gradient() -> None:
    source = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)

    sampled = sample_source_history(
        source,
        source_dt_sec=torch.tensor([1.0]),
        observation_dt_sec=torch.tensor([1.0]),
        delay_sec=torch.tensor([0.5]),
        observation_steps=4,
    )
    sampled.sum().backward()

    assert source.grad is not None
    assert torch.count_nonzero(source.grad) > 0


def test_forward_operator_applies_absolute_fractional_p_delay() -> None:
    rate = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    zeros = torch.zeros(1)
    ones = torch.ones(1)

    displacement = forward_displacement_from_rate(
        rate,
        source_dt_sec=torch.tensor([1.0]),
        observation_dt_sec=torch.tensor([1.0]),
        observation_steps=5,
        source_distance_m=torch.tensor([3.0]),
        alpha=2.0,
        beta=1.0,
        C_int_P=zeros,
        C_int_S=zeros,
        C_far_P=ones,
        C_far_S=zeros,
        include_intermediate=False,
        include_far_P=True,
        include_far_S=False,
        include_intermediate_P=False,
        include_intermediate_S=False,
    )

    assert torch.allclose(
        displacement,
        torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0]]),
        atol=1.0e-6,
    )


def test_v2_forward_signatures_use_source_distance_and_no_skip_branch() -> None:
    forward_parameters = set(
        inspect.signature(forward_displacement_from_rate).parameters
    )
    coefficient_parameters = set(
        inspect.signature(compute_physical_coefficients).parameters
    )

    assert "source_distance_m" in forward_parameters
    assert "source_distance_m" in coefficient_parameters
    assert "r_m" not in forward_parameters | coefficient_parameters
    assert "skip_delays" not in forward_parameters
