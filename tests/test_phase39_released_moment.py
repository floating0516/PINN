from __future__ import annotations

import math

import pytest
import torch

from src.training.released_moment import (
    DEFAULT_MOMENT_FLOOR_NM,
    constrained_source_mask,
    constrained_window_sec,
    magnitude_from_moment,
    masked_encoded_mse,
    moment_from_rate,
    released_magnitude,
    released_monotone_loss,
    released_prefix_losses,
    zero_pad_prefix,
)


STEPS = 200


def _reference_stf(total_moment_nm: float, duration_sec: float) -> torch.Tensor:
    """Triangular STF over ``[0, duration]`` integrating to ``total_moment_nm``."""
    t = torch.arange(STEPS, dtype=torch.float32) + 0.5
    tri = torch.clamp(1.0 - torch.abs(t - duration_sec / 2.0) / (duration_sec / 2.0), min=0.0)
    return (tri / tri.sum() * total_moment_nm).reshape(1, -1)


def test_constrained_window_clips_to_zero_before_p_arrival() -> None:
    tau = torch.tensor([3.0, 10.0, 40.0])
    window = constrained_window_sec(8.0, tau, max_sec=200.0)
    assert torch.allclose(window, torch.tensor([5.0, 0.0, 0.0]))
    assert torch.allclose(
        constrained_window_sec(500.0, tau, max_sec=200.0),
        torch.full((3,), 200.0),
    )


def test_mask_has_fractional_last_step_and_no_gradient() -> None:
    tau = torch.tensor([2.5])
    mask = constrained_source_mask(
        6.0,
        tau,
        source_steps=STEPS,
        source_dt_sec=torch.tensor([1.0]),
    )
    assert mask.shape == (1, STEPS)
    assert torch.allclose(mask[0, :3], torch.ones(3))
    assert math.isclose(float(mask[0, 3]), 0.5)
    assert float(mask[0, 4:].sum()) == 0.0
    assert not mask.requires_grad


def test_released_magnitude_is_monotone_and_matches_endpoint() -> None:
    total = 10.0 ** (1.5 * 7.0 + 9.1)  # Mw 7.0
    stf = _reference_stf(total, duration_sec=30.0)
    tau = torch.tensor([5.0])
    dt = torch.tensor([1.0])
    previous = -math.inf
    for horizon in range(1, STEPS + 1):
        mask = constrained_source_mask(horizon, tau, source_steps=STEPS, source_dt_sec=dt)
        value = float(released_magnitude(stf, mask, dt)[0])
        assert value >= previous - 1.0e-6
        previous = value
    endpoint_mask = constrained_source_mask(STEPS, tau, source_steps=STEPS, source_dt_sec=dt)
    endpoint = float(released_magnitude(stf, endpoint_mask, dt)[0])
    assert math.isclose(endpoint, 7.0, abs_tol=1.0e-4)
    # Once the rupture is over (30 s) plus P arrival, the released value equals the endpoint.
    settled_mask = constrained_source_mask(40, tau, source_steps=STEPS, source_dt_sec=dt)
    assert math.isclose(float(released_magnitude(stf, settled_mask, dt)[0]), endpoint, abs_tol=1.0e-4)


def test_released_magnitude_before_p_arrival_sits_on_floor() -> None:
    stf = _reference_stf(1.0e20, duration_sec=30.0)
    tau = torch.tensor([12.0])
    dt = torch.tensor([1.0])
    mask = constrained_source_mask(10, tau, source_steps=STEPS, source_dt_sec=dt)
    value = float(released_magnitude(stf, mask, dt)[0])
    expected = float(magnitude_from_moment(torch.tensor([DEFAULT_MOMENT_FLOOR_NM]))[0])
    assert math.isclose(value, expected, abs_tol=1.0e-6)


def test_moment_scaling_shifts_released_magnitude_by_delta() -> None:
    stf = _reference_stf(10.0 ** (1.5 * 7.4 + 9.1), duration_sec=40.0)
    tau = torch.tensor([5.0])
    dt = torch.tensor([1.0])
    delta = -0.6
    scale = 10.0 ** (1.5 * delta)
    for horizon in (20, 35, 120):
        mask = constrained_source_mask(horizon, tau, source_steps=STEPS, source_dt_sec=dt)
        base = float(released_magnitude(stf, mask, dt)[0])
        scaled = float(released_magnitude(stf * scale, mask, dt)[0])
        assert math.isclose(scaled - base, delta, abs_tol=1.0e-4)


def test_zero_pad_prefix_keeps_absolute_axis() -> None:
    x = torch.arange(1.0, 2.0 * STEPS + 1.0).reshape(1, 2, STEPS)
    padded = zero_pad_prefix(x, 37)
    assert padded.shape == x.shape
    assert torch.equal(padded[..., :37], x[..., :37])
    assert float(padded[..., 37:].abs().sum()) == 0.0
    assert zero_pad_prefix(x, STEPS) is x
    with pytest.raises(ValueError):
        zero_pad_prefix(x, 0)


def test_masked_encoded_mse_ignores_uncovered_steps() -> None:
    pred = torch.zeros(2, STEPS)
    ref = torch.ones(2, STEPS)
    mask = torch.zeros(2, STEPS)
    mask[0, :10] = 1.0
    per_sample = masked_encoded_mse(pred, ref, mask)
    assert math.isclose(float(per_sample[0]), 1.0)
    assert float(per_sample[1]) == 0.0


def test_released_prefix_losses_gate_samples_without_constraint() -> None:
    stf = torch.cat(
        [
            _reference_stf(10.0 ** (1.5 * 7.0 + 9.1), 30.0),
            _reference_stf(10.0 ** (1.5 * 6.0 + 9.1), 10.0),
        ]
    )
    tau = torch.tensor([5.0, 60.0])  # second station has not seen P at h=20
    dt = torch.tensor([1.0, 1.0])
    mask = constrained_source_mask(20, tau, source_steps=STEPS, source_dt_sec=dt)
    ref_encoded = torch.log10(1.0 + stf / 1.0e18)
    l_mse, l_mag, metrics = released_prefix_losses(
        pred_encoded=ref_encoded,
        rate_hat=stf,
        ref_encoded=ref_encoded,
        stf_true=stf,
        mask=mask,
        source_dt_sec=dt,
        sample_weights=None,
        min_constrained_window_sec=1.0,
        floor_nm=DEFAULT_MOMENT_FLOOR_NM,
    )
    assert float(l_mse) == 0.0 and float(l_mag) == 0.0
    assert math.isclose(metrics["included_fraction"], 0.5)
    # Prediction 0.3 Mw too high on the constrained sample only.
    high = stf.clone()
    high[0] *= 10.0 ** (1.5 * 0.3)
    _, l_mag_high, _ = released_prefix_losses(
        pred_encoded=ref_encoded,
        rate_hat=high,
        ref_encoded=ref_encoded,
        stf_true=stf,
        mask=mask,
        source_dt_sec=dt,
        sample_weights=None,
        min_constrained_window_sec=1.0,
        floor_nm=DEFAULT_MOMENT_FLOOR_NM,
    )
    assert math.isclose(float(l_mag_high), 0.5 * 0.3 ** 2, abs_tol=1.0e-4)


def test_moment_from_rate_uses_fractional_mask() -> None:
    rate = torch.full((1, STEPS), 2.0)
    mask = torch.zeros(1, STEPS)
    mask[0, :3] = 1.0
    mask[0, 3] = 0.5
    assert math.isclose(float(moment_from_rate(rate, mask, torch.tensor([1.0]))[0]), 7.0)


def test_released_monotone_loss_only_penalises_decrease() -> None:
    early = torch.tensor([6.0, 6.5, 7.0])
    late = torch.tensor([6.5, 6.5, 6.8])
    loss = released_monotone_loss(early, late, slack_mw=0.03)
    assert math.isclose(float(loss), ((0.2 - 0.03) ** 2) / 3.0, abs_tol=1.0e-6)
