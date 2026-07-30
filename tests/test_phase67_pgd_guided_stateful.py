from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.experiments.run_phase43_streaming_adapter import HORIZONS
from scripts.experiments.run_phase67_pgd_guided_stateful import (
    EPOCHS,
    EXPECTED_TRANSITION_PARAMETER_COUNT,
    LOSS_WEIGHTS,
    NORMALIZER_FLOORS,
    PGD_EVENT_MAE_BASELINE,
    SEEDS,
    VALIDATION_GATES,
    _assert_backbone_unchanged,
    build_arg_parser,
    closest_candidate_rank,
    freeze_transition_scope,
    normalizer_training_indices,
    normalized_loss,
    validation_gate,
)
from src.models.model import PGDGuidedSTFTransition, PINNModel
from src.utils.config_v2 import (
    stateful_streaming_config_from_config,
    validate_config_v2,
)


def _base_config() -> dict:
    config = yaml.safe_load(
        Path("configs/experiments_v2/V2-BASE.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["model"].update(
        {
            "hidden_dim": 32,
            "num_layers": 1,
            "num_tcn_blocks": 2,
            "transformer_num_layers": 1,
            "dropout": 0.0,
            "input_components": ["radial"],
            "stf_output_parameterization": "moment_shape_factorized",
        }
    )
    return config


def _pgd_config() -> dict:
    config = _base_config()
    config["model"]["stateful_streaming"] = {
        "mode": "pgd_residual_gru",
        "local_channels": 4,
        "hidden_size": 8,
        "support_ramp_sec": 6.0,
        "initial_gate_logit": -4.0,
        "max_proposal_correction_log10": 1.0,
        "shape_identity_start_sec": 160,
        "max_initial_residual_mw": 1.5,
        "max_residual_mw": 2.0,
        "max_early_revision_mw_per_step": 0.04,
        "max_late_revision_mw_per_step": 0.008,
        "confidence_step_max": 0.05,
        "initial_absorption_logit": -4.0,
        "initial_confidence_logit": -4.0,
        "proposal_assimilation_scale": 1.0,
        "pgd_hint_law": "crowell",
    }
    return config


def test_pgd_residual_config_parses_and_validates() -> None:
    parsed = stateful_streaming_config_from_config(_pgd_config())

    assert parsed["mode"] == "pgd_residual_gru"
    assert parsed["proposal_semantics"] == "complete_forecast"
    assert parsed["pgd_hint_law"] == "crowell"
    assert parsed["max_early_revision_mw_per_step"] == 0.04
    assert parsed["max_late_revision_mw_per_step"] == 0.008
    assert parsed["proposal_assimilation_scale"] == 1.0
    validate_config_v2(_pgd_config())


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("max_initial_residual_mw", 3.0, "must not exceed"),
        ("max_late_revision_mw_per_step", 0.05, "must not exceed"),
        ("confidence_step_max", 1.0, "must be below one"),
        ("proposal_assimilation_scale", 2.5, "must not exceed"),
        ("pgd_hint_law", "melgar", "must be crowell"),
    ],
)
def test_pgd_residual_config_rejects_invalid_values(
    key: str,
    value: object,
    message: str,
) -> None:
    config = _pgd_config()
    config["model"]["stateful_streaming"][key] = value

    with pytest.raises(ValueError, match=message):
        stateful_streaming_config_from_config(config)


def test_phase39_checkpoint_only_misses_pgd_transition_keys() -> None:
    source = PINNModel(_base_config())
    stateful = PINNModel(_pgd_config())

    incompatible = stateful.load_state_dict(source.state_dict(), strict=False)

    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(
        key.startswith("released_stf_transition.")
        for key in incompatible.missing_keys
    )


def test_pgd_stream_requires_hints_and_returns_diagnostics() -> None:
    torch.manual_seed(67)
    model = PINNModel(_pgd_config()).eval()
    raw = torch.full((2, 3, 200), 2.0e18)
    raw[:, 1] *= 1.1
    raw[:, 2] *= 1.2
    hints = torch.tensor([[6.0, 6.1, 6.2], [6.4, 6.5, 6.6]])
    valid = torch.tensor([[False, True, True], [True, True, True]])
    distance = torch.tensor([45_330.0, 20_000.0])
    source_dt = torch.ones(2)

    with pytest.raises(ValueError, match="requires PGD"):
        model.stream_sequence_from_rates(
            raw,
            horizons_sec=(20, 21, 22),
            source_distance_m=distance,
            source_dt_sec=source_dt,
            beta_m_per_s=4_533.0,
        )

    released, encoded, magnitudes, gates, diagnostics = (
        model.stream_sequence_from_rates(
            raw,
            horizons_sec=(20, 21, 22),
            source_distance_m=distance,
            source_dt_sec=source_dt,
            beta_m_per_s=4_533.0,
            pgd_mw_hint=hints,
            pgd_valid_hint=valid,
            return_diagnostics=True,
        )
    )

    assert released.shape == encoded.shape == (2, 3, 200)
    assert magnitudes.shape == (2, 3)
    assert gates.shape == (2, 3, 200)
    assert set(diagnostics) == {
        "absorption_gate",
        "revision_mw",
        "plateau_confidence",
        "pgd_anchor_mw",
        "pgd_residual_mw",
    }
    assert all(value.shape == (2, 3) for value in diagnostics.values())
    torch.testing.assert_close(
        magnitudes,
        diagnostics["pgd_anchor_mw"] + diagnostics["pgd_residual_mw"],
    )
    assert torch.all(released >= 0.0)
    assert torch.all(
        diagnostics["plateau_confidence"][:, 1:]
        >= diagnostics["plateau_confidence"][:, :-1]
    )
    assert torch.all(
        diagnostics["pgd_anchor_mw"][:, 2:]
        >= diagnostics["pgd_anchor_mw"][:, 1:-1]
    )


def test_absorption_gate_can_hold_output_while_pgd_grows() -> None:
    model = PINNModel(_pgd_config()).eval()
    transition = model.released_stf_transition
    assert isinstance(transition, PGDGuidedSTFTransition)
    with torch.no_grad():
        transition.moment_update_head.weight.zero_()
        transition.moment_update_head.bias.copy_(
            torch.tensor([0.0, 20.0, 0.0, -20.0])
        )
    raw = torch.full((1, 200), 2.0e18)
    metadata = {
        "source_distance_m": torch.tensor([20_000.0]),
        "source_dt_sec": torch.ones(1),
        "beta_m_per_s": 4_533.0,
    }
    first = model.stream_step_from_rate(
        raw,
        state=None,
        horizon_sec=20,
        pgd_mw_hint=torch.tensor([6.0]),
        pgd_valid_hint=torch.tensor([True]),
        **metadata,
    )
    second = model.stream_step_from_rate(
        raw,
        state=first.state,
        horizon_sec=21,
        pgd_mw_hint=torch.tensor([6.3]),
        pgd_valid_hint=torch.tensor([True]),
        **metadata,
    )

    assert second.absorption_gate is not None
    assert float(second.absorption_gate.item()) > 0.999
    torch.testing.assert_close(
        second.released_mw,
        first.released_mw,
        atol=1.0e-5,
        rtol=0.0,
    )


def test_high_confidence_negative_revision_is_bounded() -> None:
    model = PINNModel(_pgd_config()).eval()
    transition = model.released_stf_transition
    assert isinstance(transition, PGDGuidedSTFTransition)
    with torch.no_grad():
        transition.moment_update_head.weight.zero_()
        transition.moment_update_head.bias.copy_(
            torch.tensor([0.0, -20.0, -20.0, -20.0])
        )
    raw = torch.full((1, 200), 2.0e18)
    metadata = {
        "source_distance_m": torch.tensor([20_000.0]),
        "source_dt_sec": torch.ones(1),
        "beta_m_per_s": 4_533.0,
    }
    first = model.stream_step_from_rate(
        raw,
        state=None,
        horizon_sec=20,
        pgd_mw_hint=torch.tensor([6.0]),
        pgd_valid_hint=torch.tensor([True]),
        **metadata,
    )
    high_confidence = replace(
        first.state,
        plateau_confidence=torch.ones(1),
    )
    second = model.stream_step_from_rate(
        raw,
        state=high_confidence,
        horizon_sec=21,
        pgd_mw_hint=torch.tensor([6.0]),
        pgd_valid_hint=torch.tensor([True]),
        **metadata,
    )

    drop = float((first.released_mw - second.released_mw).item())
    assert 0.0079 <= drop <= 0.0081
    assert second.revision_mw is not None
    assert float(second.revision_mw.item()) == pytest.approx(-0.008, abs=1.0e-5)


def test_late_proposal_assimilation_scale_changes_only_late_update() -> None:
    base_config = _pgd_config()
    base_config["model"]["stateful_streaming"].update(
        {
            "use_late_proposal_assimilation": True,
            "late_proposal_assimilation_start_sec": 199,
            "initial_proposal_assimilation_logit": 20.0,
            "proposal_assimilation_scale": 1.0,
        }
    )
    scaled_config = _pgd_config()
    scaled_config["model"]["stateful_streaming"].update(
        {
            "use_late_proposal_assimilation": True,
            "late_proposal_assimilation_start_sec": 199,
            "initial_proposal_assimilation_logit": 20.0,
            "proposal_assimilation_scale": 1.5,
        }
    )
    base = PINNModel(base_config).eval()
    transition = base.released_stf_transition
    assert isinstance(transition, PGDGuidedSTFTransition)
    with torch.no_grad():
        transition.moment_update_head.weight.zero_()
        transition.moment_update_head.bias.copy_(
            torch.tensor([0.0, -20.0, 0.0, -20.0])
        )
        assert transition.proposal_assimilation_head is not None
        transition.proposal_assimilation_head.weight.zero_()
        transition.proposal_assimilation_head.bias.fill_(20.0)
    scaled = PINNModel(scaled_config).eval()
    scaled.load_state_dict(base.state_dict(), strict=True)

    raw = torch.full((1, 200), 2.0e18)
    metadata = {
        "source_distance_m": torch.tensor([20_000.0]),
        "source_dt_sec": torch.ones(1),
        "beta_m_per_s": 4_533.0,
        "pgd_mw_hint": torch.tensor([6.0]),
        "pgd_valid_hint": torch.tensor([True]),
    }
    base_first = base.stream_step_from_rate(
        raw,
        state=None,
        horizon_sec=199,
        **metadata,
    )
    scaled_first = scaled.stream_step_from_rate(
        raw,
        state=None,
        horizon_sec=199,
        **metadata,
    )
    torch.testing.assert_close(base_first.released_mw, scaled_first.released_mw)

    base_late = base.stream_step_from_rate(
        raw,
        state=base_first.state,
        horizon_sec=200,
        **metadata,
    )
    scaled_late = scaled.stream_step_from_rate(
        raw,
        state=scaled_first.state,
        horizon_sec=200,
        **metadata,
    )
    assert base_late.proposal_assimilation_mw is not None
    assert scaled_late.proposal_assimilation_mw is not None
    assert float(base_late.proposal_assimilation_mw.item()) > 0.0
    torch.testing.assert_close(
        scaled_late.proposal_assimilation_mw,
        1.5 * base_late.proposal_assimilation_mw,
    )


def test_pgd_state_checkpoint_round_trip() -> None:
    torch.manual_seed(73)
    model = PINNModel(_pgd_config()).eval()
    clone = PINNModel(_pgd_config()).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    raw = torch.full((1, 2, 200), 2.0e18)
    hints = torch.tensor([[6.2, 6.3]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    kwargs = {
        "horizons_sec": (20, 21),
        "source_distance_m": torch.tensor([20_000.0]),
        "source_dt_sec": torch.ones(1),
        "beta_m_per_s": 4_533.0,
        "pgd_mw_hint": hints,
        "pgd_valid_hint": valid,
        "return_diagnostics": True,
    }

    original = model.stream_sequence_from_rates(raw, **kwargs)
    restored = clone.stream_sequence_from_rates(raw, **kwargs)

    for left, right in zip(original[:4], restored[:4]):
        torch.testing.assert_close(left, right)
    for name in original[4]:
        torch.testing.assert_close(original[4][name], restored[4][name])


def test_phase67_training_runner_contracts_are_enforced_together() -> None:
    model = PINNModel(_pgd_config()).train()
    trainable = freeze_transition_scope(model)
    assert sum(parameter.numel() for parameter in trainable) == (
        EXPECTED_TRANSITION_PARAMETER_COUNT
    )
    assert all(
        parameter.requires_grad == name.startswith("released_stf_transition.")
        for name, parameter in model.named_parameters()
    )

    training_indices = np.arange(1_788, dtype=np.int64)
    audit_indices = normalizer_training_indices(training_indices)
    assert len(audit_indices) == 256
    assert len(np.unique(audit_indices)) == len(audit_indices)
    assert np.array_equal(
        audit_indices,
        normalizer_training_indices(training_indices),
    )
    assert not np.array_equal(audit_indices, training_indices[:256])

    frozen_source = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("released_stf_transition.")
    }
    _assert_backbone_unchanged(model, frozen_source)

    components = {
        name: torch.tensor(float(index + 1), requires_grad=True)
        for index, name in enumerate(LOSS_WEIGHTS)
    }
    normalizers = {
        name: max(float(value.detach()), NORMALIZER_FLOORS[name])
        for name, value in components.items()
    }
    total = normalized_loss(components, normalizers)
    assert total.item() == pytest.approx(sum(LOSS_WEIGHTS.values()))
    total.backward()
    assert all(
        value.grad is not None and torch.isfinite(value.grad)
        for value in components.values()
    )

    passing = {
        "endpoint_event_mae": (
            VALIDATION_GATES["endpoint_event_mae_max"] * 0.9
        ),
        "endpoint_station_mae": (
            VALIDATION_GATES["endpoint_station_mae_max"] * 0.9
        ),
        "event_start_to_end_increase_fraction": 0.95,
        "event_20s_not_below_minus_0_05_fraction": 0.99,
        "event_post120_abs_step_p95_mw": 0.018,
        "event_post120_abs_step_max_mw": 0.135,
        "event_post120_excess_variation_p95_mw": 0.135,
        "event_post120_sign_changes_p95": 1.8,
        "event_peak_to_final_p95_mw": 0.135,
        "event_post160_band_width_p95_mw": 0.27,
        "mean_key_event_mae": 0.10,
    }
    for horizon in PGD_EVENT_MAE_BASELINE:
        passing[f"pgd_event_mae_improvement_{horizon}s"] = 0.011
    passing_gate = validation_gate(passing)
    assert passing_gate["passed"] is True
    assert closest_candidate_rank(passing, passing_gate)[0] <= 1.0

    maximum_metrics = {
        "endpoint_event_mae": "endpoint_event_mae_max",
        "endpoint_station_mae": "endpoint_station_mae_max",
        "event_post120_abs_step_p95_mw": (
            "event_post120_abs_step_p95_mw_max"
        ),
        "event_post120_abs_step_max_mw": (
            "event_post120_abs_step_max_mw_max"
        ),
        "event_post120_excess_variation_p95_mw": (
            "event_post120_excess_variation_p95_mw_max"
        ),
        "event_post120_sign_changes_p95": (
            "event_post120_sign_changes_p95_max"
        ),
        "event_peak_to_final_p95_mw": "event_peak_to_final_p95_mw_max",
        "event_post160_band_width_p95_mw": (
            "event_post160_band_width_p95_mw_max"
        ),
    }
    for metric, gate_name in maximum_metrics.items():
        failing = dict(passing)
        failing[metric] = VALIDATION_GATES[gate_name] * 1.01
        assert validation_gate(failing)["passed"] is False

    minimum_metrics = {
        "event_start_to_end_increase_fraction": (
            "event_start_to_end_increase_fraction_min"
        ),
        "event_20s_not_below_minus_0_05_fraction": (
            "event_20s_not_below_minus_0_05_fraction_min"
        ),
    }
    for metric, gate_name in minimum_metrics.items():
        failing = dict(passing)
        failing[metric] = VALIDATION_GATES[gate_name] * 0.99
        assert validation_gate(failing)["passed"] is False

    for horizon in PGD_EVENT_MAE_BASELINE:
        failing = dict(passing)
        failing[f"pgd_event_mae_improvement_{horizon}s"] = 0.009
        assert validation_gate(failing)["passed"] is False

    help_text = " ".join(build_arg_parser().format_help().lower().split())
    assert "--hint-cache-root" in help_text
    hint_action = next(
        action
        for action in build_arg_parser()._actions
        if action.dest == "hint_cache_root"
    )
    assert hint_action.required is True
    assert "hint, not an external adapter" in help_text
    assert SEEDS == (17, 42, 73)
    assert EPOCHS == 30
    assert tuple(PGD_EVENT_MAE_BASELINE) == (30, 60, 90, 120, 200)
    assert "convergence" not in VALIDATION_GATES
    assert LOSS_WEIGHTS["pgd_advantage"] == pytest.approx(0.05)
    assert len(HORIZONS) == 181
