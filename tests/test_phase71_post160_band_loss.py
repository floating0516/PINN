from __future__ import annotations

import pytest
import torch

from scripts.experiments import run_phase67_pgd_guided_stateful as campaign
from scripts.experiments.run_phase43_streaming_adapter import HORIZONS
from scripts.experiments.run_phase71_post160_band_loss import (
    PHASE71_BAND_LOSS_WEIGHT,
    PHASE71_BAND_NORMALIZER_FLOOR,
    configure_phase71,
)


def test_phase71_configuration_enables_only_the_band_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutable_names = (
        "EXPERIMENT_PHASE",
        "STATE_HIDDEN_SIZE",
        "USE_LATE_PROPOSAL_ASSIMILATION",
        "LATE_PROPOSAL_ASSIMILATION_START_SEC",
        "EXPECTED_TRANSITION_PARAMETER_COUNT",
        "EXPECTED_TOTAL_PARAMETER_COUNT",
        "USE_PLATEAU_BAND_LOSS",
        "LOSS_WEIGHTS",
        "NORMALIZER_FLOORS",
    )
    original = {name: getattr(campaign, name) for name in mutable_names}

    with monkeypatch.context() as restore:
        for name, value in original.items():
            restore.setattr(campaign, name, value)
        configure_phase71()

        assert campaign.EXPERIMENT_PHASE == "Phase71"
        assert campaign.STATE_HIDDEN_SIZE == 16
        assert campaign.USE_LATE_PROPOSAL_ASSIMILATION is True
        assert campaign.LATE_PROPOSAL_ASSIMILATION_START_SEC == 140
        assert campaign.USE_PLATEAU_BAND_LOSS is True
        assert campaign.LOSS_WEIGHTS["plateau_band"] == pytest.approx(
            PHASE71_BAND_LOSS_WEIGHT
        )
        assert campaign.NORMALIZER_FLOORS["plateau_band"] == pytest.approx(
            PHASE71_BAND_NORMALIZER_FLOOR
        )

    for name, value in original.items():
        assert getattr(campaign, name) is value


def test_plateau_band_component_is_a_zero_below_gate_hinge() -> None:
    start = HORIZONS.index(campaign.PLATEAU_BAND_START_SEC)
    tail_size = len(HORIZONS) - start
    constant = torch.full((len(HORIZONS),), 6.0)
    at_gate = torch.cat(
        (
            torch.full((start,), 6.0),
            torch.linspace(6.0, 6.3, tail_size),
        )
    )
    above_gate = torch.cat(
        (
            torch.full((start,), 6.0),
            torch.linspace(6.0, 6.4, tail_size),
        )
    )
    state_mw = torch.stack((constant, at_gate, above_gate)).requires_grad_()
    sample_weights = torch.ones(3)

    loss, violation = campaign._plateau_band_component(
        state_mw,
        sample_weights,
    )

    torch.testing.assert_close(
        violation,
        torch.tensor([0.0, 0.0, 0.1 - campaign.PLATEAU_BAND_EPS_MW]),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert state_mw.grad is not None
    assert torch.count_nonzero(state_mw.grad[:2]) == 0
    assert torch.count_nonzero(state_mw.grad[2]) > 0
