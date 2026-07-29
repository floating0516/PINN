from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.experiments.run_phase43_streaming_adapter import HORIZONS
from scripts.experiments.run_phase50_stateful_incremental_model import (
    LOSS_WEIGHTS,
    VALIDATION_GATES,
    _assert_backbone_unchanged,
    build_arg_parser,
    freeze_transition_scope,
    normalizer_training_indices,
    normalized_loss,
    stateful_loss_components,
    validation_gate,
)
from src.models.model import PINNModel, PINNStreamingState
from src.utils.config_v2 import (
    stateful_streaming_config_from_config,
    validate_config_v2,
)


def _config() -> dict:
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


def _stateful_config() -> dict:
    config = _config()
    config["model"]["stateful_streaming"] = {
        "mode": "released_stf_gru",
        "local_channels": 4,
        "hidden_size": 8,
        "support_ramp_sec": 6.0,
        "initial_gate_logit": -4.0,
    }
    return config


def test_stateful_config_defaults_to_disabled() -> None:
    config = _config()

    assert stateful_streaming_config_from_config(config) == {"mode": "none"}
    validate_config_v2(config)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"mode": "unknown"}, "mode"),
        ({"mode": "none", "hidden_size": 8}, "require"),
        (
            {"mode": "released_stf_gru", "hidden_size": 0},
            "positive integer",
        ),
        (
            {"mode": "released_stf_gru", "support_ramp_sec": 0.0},
            "positive",
        ),
    ],
)
def test_stateful_config_rejects_invalid_values(
    update: dict,
    message: str,
) -> None:
    config = _config()
    config["model"]["stateful_streaming"] = update

    with pytest.raises(ValueError, match=message):
        stateful_streaming_config_from_config(config)


def test_default_phase39_state_dict_is_unchanged_by_disabled_mode() -> None:
    implicit = _config()
    explicit = copy.deepcopy(implicit)
    explicit["model"]["stateful_streaming"] = {"mode": "none"}

    torch.manual_seed(50)
    implicit_model = PINNModel(implicit)
    torch.manual_seed(50)
    explicit_model = PINNModel(explicit)

    assert implicit_model.state_dict().keys() == explicit_model.state_dict().keys()
    assert not any(
        key.startswith("released_stf_transition.")
        for key in implicit_model.state_dict()
    )
    explicit_model.load_state_dict(implicit_model.state_dict(), strict=True)


def test_phase39_checkpoint_loads_with_only_transition_keys_missing() -> None:
    source = PINNModel(_config())
    stateful = PINNModel(_stateful_config())

    incompatible = stateful.load_state_dict(source.state_dict(), strict=False)

    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(
        key.startswith("released_stf_transition.")
        for key in incompatible.missing_keys
    )
    transition_parameters = sum(
        parameter.numel()
        for name, parameter in stateful.named_parameters()
        if name.startswith("released_stf_transition.")
    )
    assert transition_parameters == 537


def test_stream_sequence_carries_state_and_matches_manual_steps() -> None:
    torch.manual_seed(50)
    model = PINNModel(_stateful_config()).eval()
    raw_rates = torch.full((2, 3, 200), 2.0e18)
    raw_rates[:, 1] *= 1.2
    raw_rates[:, 2] *= 0.8
    horizons = (20, 21, 22)
    distance = torch.tensor([0.0, 45_330.0])
    source_dt = torch.ones(2)

    released, encoded, magnitudes, gates = model.stream_sequence_from_rates(
        raw_rates,
        horizons_sec=horizons,
        source_distance_m=distance,
        source_dt_sec=source_dt,
        beta_m_per_s=4_533.0,
    )

    state: PINNStreamingState | None = None
    manual = []
    for index, horizon in enumerate(horizons):
        output = model.stream_step_from_rate(
            raw_rates[:, index],
            state=state,
            horizon_sec=horizon,
            source_distance_m=distance,
            source_dt_sec=source_dt,
            beta_m_per_s=4_533.0,
        )
        state = output.state
        manual.append(output.released_rate)
    torch.testing.assert_close(released, torch.stack(manual, dim=1))
    assert released.shape == encoded.shape == (2, 3, 200)
    assert magnitudes.shape == (2, 3)
    assert gates.shape == (2, 3, 200)
    assert torch.isfinite(encoded).all()
    assert torch.all(released >= 0.0)
    assert torch.all((gates >= 0.0) & (gates <= 1.0))
    assert state is not None and state.horizon_sec == 22


def test_first_stream_step_is_causally_supported_proposal() -> None:
    model = PINNModel(_stateful_config()).eval()
    raw = torch.full((1, 200), 3.0e18)

    output = model.stream_step_from_rate(
        raw,
        state=None,
        horizon_sec=20,
        source_distance_m=torch.tensor([45_330.0]),
        source_dt_sec=torch.ones(1),
        beta_m_per_s=4_533.0,
    )

    torch.testing.assert_close(
        output.released_rate,
        raw * output.support_fraction,
    )
    assert torch.equal(output.retention_gate, torch.zeros_like(raw))
    assert output.support_fraction[0, 0] == pytest.approx(1.0)
    assert output.support_fraction[0, 10] == pytest.approx(0.0)


def test_stream_step_rejects_nonconsecutive_state() -> None:
    model = PINNModel(_stateful_config()).eval()
    raw = torch.ones(1, 200)
    first = model.stream_step_from_rate(
        raw,
        state=None,
        horizon_sec=20,
        source_distance_m=torch.zeros(1),
        source_dt_sec=torch.ones(1),
        beta_m_per_s=4_533.0,
    )

    with pytest.raises(ValueError, match="exactly one second"):
        model.stream_step_from_rate(
            raw,
            state=first.state,
            horizon_sec=22,
            source_distance_m=torch.zeros(1),
            source_dt_sec=torch.ones(1),
            beta_m_per_s=4_533.0,
        )


def test_stateful_transition_receives_finite_gradients() -> None:
    model = PINNModel(_stateful_config()).train()
    raw_rates = torch.rand(2, 3, 200) * 1.0e19

    released, encoded, magnitudes, gates = model.stream_sequence_from_rates(
        raw_rates,
        horizons_sec=(20, 21, 22),
        source_distance_m=torch.zeros(2),
        source_dt_sec=torch.ones(2),
        beta_m_per_s=4_533.0,
    )
    (released.mean() / 1.0e19 + encoded.mean() + magnitudes.mean()).backward()

    missing = [
        name
        for name, parameter in model.named_parameters()
        if name.startswith("released_stf_transition.")
        and parameter.grad is None
    ]
    assert missing == []
    assert all(
        torch.isfinite(parameter.grad).all()
        for name, parameter in model.named_parameters()
        if name.startswith("released_stf_transition.")
    )


def test_phase50_training_runner_contracts_are_enforced_together() -> None:
    torch.manual_seed(50)
    config = _stateful_config()
    model = PINNModel(config).train()
    trainable = freeze_transition_scope(model)
    assert sum(parameter.numel() for parameter in trainable) == 537
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

    batch_size = 1
    raw_rate = torch.full(
        (batch_size, len(HORIZONS), 200),
        2.0e18,
    )
    raw_rate[:, 80:120] *= 0.8
    raw_rate[:, 120:] *= 1.1
    batch = {
        "raw_rate": raw_rate,
        "source_distance_m": torch.zeros(batch_size),
        "source_dt_sec": torch.ones(batch_size),
        "sample_weight": torch.ones(batch_size),
        "stf": torch.full((batch_size, 200), 2.0e18),
        "radial": torch.zeros(batch_size, 200),
        "theta_deg": torch.zeros(batch_size),
        "phi_slip_deg": torch.zeros(batch_size),
        "observation_dt_sec": torch.ones(batch_size),
        "waveform_valid_mask": torch.ones(batch_size, 200, dtype=torch.bool),
        "magnitude_catalog": torch.full((batch_size,), 8.0),
    }

    def criterion(predicted: torch.Tensor, **_kwargs: object):
        loss = predicted.pow(2).mean()
        value = float(loss.detach())
        return loss, {"L_MSE": value, "L_synth": value, "L_mag": value}

    components, diagnostics = stateful_loss_components(
        model,
        batch,
        criterion=criterion,
        config=config,
    )
    assert set(components) == set(LOSS_WEIGHTS)
    assert all(torch.isfinite(value) for value in components.values())
    assert all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values())
    normalizers = {
        name: max(float(value.detach()), 1.0e-8)
        for name, value in components.items()
    }
    total = normalized_loss(components, normalizers)
    assert total.item() == pytest.approx(sum(LOSS_WEIGHTS.values()))
    total.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trainable
    )
    _assert_backbone_unchanged(model, frozen_source)

    passing = {
        "endpoint_event_mae": VALIDATION_GATES["endpoint_event_mae_max"] * 0.9,
        "endpoint_station_mae": VALIDATION_GATES["endpoint_station_mae_max"] * 0.9,
        "late_event_abs_step_p95_mw": (
            VALIDATION_GATES["late_event_abs_step_p95_mw_max"] * 0.9
        ),
        "late_station_abs_step_p95_mw": (
            VALIDATION_GATES["late_station_abs_step_p95_mw_max"] * 0.9
        ),
        "late_confirmed_cumulative_log10_l1_p95": (
            VALIDATION_GATES[
                "late_confirmed_cumulative_log10_l1_p95_max"
            ]
            * 0.9
        ),
        "event_downward_step_p95_mw": (
            VALIDATION_GATES["event_downward_step_p95_mw_max"] * 0.9
        ),
        "event_peak_to_final_p95_mw": (
            VALIDATION_GATES["event_peak_to_final_p95_mw_max"] * 0.9
        ),
    }
    assert validation_gate(passing)["passed"] is True
    metric_to_gate = {
        "endpoint_event_mae": "endpoint_event_mae_max",
        "endpoint_station_mae": "endpoint_station_mae_max",
        "late_event_abs_step_p95_mw": "late_event_abs_step_p95_mw_max",
        "late_station_abs_step_p95_mw": "late_station_abs_step_p95_mw_max",
        "late_confirmed_cumulative_log10_l1_p95": (
            "late_confirmed_cumulative_log10_l1_p95_max"
        ),
        "event_downward_step_p95_mw": "event_downward_step_p95_mw_max",
        "event_peak_to_final_p95_mw": "event_peak_to_final_p95_mw_max",
    }
    for metric, gate_name in metric_to_gate.items():
        failing = dict(passing)
        failing[metric] = VALIDATION_GATES[gate_name] * 1.01
        assert validation_gate(failing)["passed"] is False

    help_text = " ".join(build_arg_parser().format_help().lower().split())
    assert "--stage" in help_text
    assert "--normalizer-path" in help_text
    assert "no external adapter" in help_text
