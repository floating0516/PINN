from __future__ import annotations

import pytest
import torch

from src.models.streaming_stf_adapter import StreamingSTFAdapter


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.tensor(
        [
            [
                [1.0e18, 2.0e18, 3.0e18, 4.0e18],
                [2.0e18, 3.0e18, 4.0e18, 5.0e18],
                [3.0e18, 4.0e18, 5.0e18, 6.0e18],
            ]
        ]
    )
    horizons = torch.tensor([20.0, 21.0, 22.0])
    distance = torch.tensor([0.0])
    dt = torch.tensor([1.0])
    return raw, horizons, distance, dt


def test_adapter_parameter_count_is_frozen() -> None:
    torch.manual_seed(42)
    adapter = StreamingSTFAdapter(source_steps=200)

    assert sum(parameter.numel() for parameter in adapter.parameters()) == 489


def test_adapter_output_is_nonnegative_and_convex_at_each_update() -> None:
    torch.manual_seed(42)
    adapter = StreamingSTFAdapter(source_steps=4)
    raw, horizons, distance, dt = _inputs()

    states, gates = adapter(
        raw,
        horizons_sec=horizons,
        source_distance_m=distance,
        source_dt_sec=dt,
        beta_m_per_s=4.0,
    )

    assert states.shape == raw.shape
    assert gates.shape == raw.shape
    assert bool(torch.all(states >= 0.0))
    assert bool(torch.all((gates >= 0.0) & (gates <= 1.0)))
    assert torch.equal(states[:, 0], raw[:, 0])
    for index in range(1, raw.shape[1]):
        lower = torch.minimum(states[:, index - 1], raw[:, index])
        upper = torch.maximum(states[:, index - 1], raw[:, index])
        assert bool(torch.all(states[:, index] >= lower))
        assert bool(torch.all(states[:, index] <= upper))


def test_adapter_gate_is_zero_before_s_support() -> None:
    torch.manual_seed(42)
    adapter = StreamingSTFAdapter(source_steps=4)
    raw, horizons, _, dt = _inputs()

    _, gates = adapter(
        raw,
        horizons_sec=horizons,
        source_distance_m=torch.tensor([1000.0]),
        source_dt_sec=dt,
        beta_m_per_s=4.0,
    )

    assert torch.equal(gates, torch.zeros_like(gates))


def test_adapter_is_causal_across_issue_time() -> None:
    torch.manual_seed(42)
    adapter = StreamingSTFAdapter(source_steps=4)
    raw, horizons, distance, dt = _inputs()
    changed = raw.clone()
    changed[:, 2] *= 100.0

    baseline_states, baseline_gates = adapter(
        raw,
        horizons_sec=horizons,
        source_distance_m=distance,
        source_dt_sec=dt,
        beta_m_per_s=4.0,
    )
    changed_states, changed_gates = adapter(
        changed,
        horizons_sec=horizons,
        source_distance_m=distance,
        source_dt_sec=dt,
        beta_m_per_s=4.0,
    )

    assert torch.equal(baseline_states[:, :2], changed_states[:, :2])
    assert torch.equal(baseline_gates[:, :2], changed_gates[:, :2])


def test_adapter_gradients_reach_every_parameter() -> None:
    torch.manual_seed(42)
    adapter = StreamingSTFAdapter(source_steps=4)
    raw, horizons, distance, dt = _inputs()

    states, gates = adapter(
        raw,
        horizons_sec=horizons,
        source_distance_m=distance,
        source_dt_sec=dt,
        beta_m_per_s=4.0,
    )
    (states[:, -1].log().mean() + gates.mean()).backward()

    assert all(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0.0))
        for parameter in adapter.parameters()
    )


def test_adapter_rejects_nonconsecutive_horizons() -> None:
    adapter = StreamingSTFAdapter(source_steps=4)
    raw, _, distance, dt = _inputs()

    with pytest.raises(ValueError, match="consecutive"):
        adapter(
            raw,
            horizons_sec=torch.tensor([20.0, 22.0, 23.0]),
            source_distance_m=distance,
            source_dt_sec=dt,
            beta_m_per_s=4.0,
        )
