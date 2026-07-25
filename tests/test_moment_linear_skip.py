from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from src.models.model import PINNModel
from src.utils.config_v2 import (
    moment_linear_skip_from_config,
    validate_config_v2,
)


_SKIP_STATE_KEY = "moment_linear_skip.weight"


def _model_config(
    enabled: bool | None,
    *,
    hidden_dim: int = 32,
    dropout: float = 0.0,
) -> dict:
    config = yaml.safe_load(
        Path("configs/experiments_v2/V2-BASE.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["model"].update(
        {
            "hidden_dim": hidden_dim,
            "num_layers": 1,
            "num_tcn_blocks": 2,
            "transformer_num_layers": 1,
            "dropout": dropout,
            "stf_output_parameterization": "moment_shape_factorized",
            "predict_catalog_mw": False,
        }
    )
    if enabled is None:
        config["model"].pop("moment_linear_skip", None)
    else:
        config["model"]["moment_linear_skip"] = enabled
    return config


def _metadata(batch_size: int, device: torch.device) -> torch.Tensor:
    takeoff = torch.linspace(0.2, 0.8, batch_size, device=device)
    azimuth = torch.linspace(-1.0, 1.0, batch_size, device=device)
    return torch.stack(
        (
            torch.linspace(
                math.log(10_000.0),
                math.log(100_000.0),
                batch_size,
                device=device,
            ),
            torch.sin(takeoff),
            torch.cos(takeoff),
            torch.sin(azimuth),
            torch.cos(azimuth),
        ),
        dim=1,
    )


def _common_state(model: PINNModel) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if key != _SKIP_STATE_KEY
    }


def test_zero_skip_preserves_state_rng_output_and_common_gradients() -> None:
    baseline_config = _model_config(False, dropout=0.2)
    candidate_config = _model_config(True, dropout=0.2)

    torch.manual_seed(2718)
    baseline = PINNModel(baseline_config).train()
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(2718)
    candidate = PINNModel(candidate_config).train()
    candidate_rng = torch.random.get_rng_state().clone()

    assert torch.equal(baseline_rng, candidate_rng)
    assert _SKIP_STATE_KEY not in baseline.state_dict()
    assert set(candidate.state_dict()) == set(baseline.state_dict()) | {
        _SKIP_STATE_KEY
    }
    assert list(candidate.state_dict())[-1] == _SKIP_STATE_KEY
    baseline_state = baseline.state_dict()
    for key, value in _common_state(candidate).items():
        assert torch.equal(value, baseline_state[key]), key

    waveform = 0.02 * torch.randn(2, 1, 40)
    metadata = _metadata(2, torch.device("cpu"))
    forward_rng = torch.random.get_rng_state().clone()
    baseline_prediction = baseline.predict_heads(waveform, meta=metadata)
    baseline_after_forward = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(forward_rng)
    candidate_prediction = candidate.predict_heads(waveform, meta=metadata)
    candidate_after_forward = torch.random.get_rng_state().clone()

    assert torch.equal(baseline_after_forward, candidate_after_forward)
    torch.testing.assert_close(
        candidate_prediction.stf_encoded,
        baseline_prediction.stf_encoded,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        candidate_prediction.catalog_mw,
        baseline_prediction.catalog_mw,
        rtol=0.0,
        atol=0.0,
    )

    time_weights = torch.linspace(
        0.5,
        1.5,
        baseline_prediction.stf_encoded.size(1),
    )
    baseline_loss = (
        baseline_prediction.stf_encoded * time_weights.unsqueeze(0)
    ).mean() + baseline_prediction.catalog_mw.square().mean()
    candidate_loss = (
        candidate_prediction.stf_encoded * time_weights.unsqueeze(0)
    ).mean() + candidate_prediction.catalog_mw.square().mean()
    baseline_loss.backward()
    candidate_loss.backward()

    candidate_parameters = dict(candidate.named_parameters())
    for name, parameter in baseline.named_parameters():
        candidate_parameter = candidate_parameters[name]
        assert parameter.grad is not None, name
        assert candidate_parameter.grad is not None, name
        torch.testing.assert_close(
            candidate_parameter.grad,
            parameter.grad,
            rtol=0.0,
            atol=0.0,
        )
    assert candidate.moment_linear_skip is not None
    skip_gradient = candidate.moment_linear_skip.weight.grad
    assert skip_gradient is not None
    assert torch.isfinite(skip_gradient).all()
    assert float(skip_gradient.abs().sum()) > 0.0


def test_default_false_keeps_legacy_checkpoint_and_rng_compatible() -> None:
    default_config = _model_config(None)
    explicit_config = _model_config(False)

    torch.manual_seed(1618)
    legacy = PINNModel(default_config)
    legacy_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(1618)
    explicit = PINNModel(explicit_config)
    explicit_rng = torch.random.get_rng_state().clone()

    assert moment_linear_skip_from_config(default_config) is False
    assert torch.equal(legacy_rng, explicit_rng)
    assert legacy.state_dict().keys() == explicit.state_dict().keys()
    incompatible = explicit.load_state_dict(legacy.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_skip_has_exact_shape_count_and_bias_free_placement() -> None:
    baseline = PINNModel(_model_config(False, hidden_dim=128))
    candidate = PINNModel(_model_config(True, hidden_dim=128))
    skip = candidate.moment_linear_skip

    assert skip is not None
    assert list(skip.named_parameters())[0][0] == "weight"
    assert len(list(skip.parameters())) == 1
    assert skip.weight.shape == (1, 128)
    assert skip.weight.numel() == 128
    assert torch.count_nonzero(skip.weight) == 0
    assert list(candidate.state_dict())[-1] == _SKIP_STATE_KEY
    assert (
        sum(parameter.numel() for parameter in candidate.parameters())
        - sum(parameter.numel() for parameter in baseline.parameters())
        == 128
    )


def test_nonzero_skip_is_an_exact_log_moment_residual_and_preserves_integral() -> None:
    baseline_config = _model_config(False)
    candidate_config = _model_config(True)
    torch.manual_seed(314)
    baseline = PINNModel(baseline_config).eval()
    torch.manual_seed(314)
    candidate = PINNModel(candidate_config).eval()
    assert candidate.moment_linear_skip is not None
    with torch.no_grad():
        candidate.moment_linear_skip.weight.copy_(
            torch.linspace(-0.02, 0.03, candidate.hidden_dim).unsqueeze(0)
        )

    waveform = 0.02 * torch.randn(3, 1, 40)
    metadata = _metadata(3, torch.device("cpu"))
    with torch.no_grad():
        sequence = candidate._encode_sequence(waveform, metadata)
        expected_delta_mw = (2.0 / 3.0) * F.linear(
            sequence.mean(dim=1),
            candidate.moment_linear_skip.weight,
            bias=None,
        ).squeeze(-1)
        baseline_prediction = baseline.predict_heads(waveform, meta=metadata)
        candidate_prediction = candidate.predict_heads(waveform, meta=metadata)

    torch.testing.assert_close(
        candidate_prediction.catalog_mw - baseline_prediction.catalog_mw,
        expected_delta_mw,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    m_ref = float(candidate_config["dataset"]["stf"]["m_ref"])
    source_dt = 1.0 / float(candidate_config["dataset"]["sample_rate_hz"])
    physical_rate = m_ref * torch.expm1(
        candidate_prediction.stf_encoded * math.log(10.0)
    )
    integrated_moment = physical_rate.sum(dim=1) * source_dt
    expected_moment = torch.pow(
        10.0,
        1.5 * candidate_prediction.catalog_mw + 9.1,
    )
    torch.testing.assert_close(
        integrated_moment,
        expected_moment,
        rtol=2.0e-5,
        atol=0.0,
    )


def test_candidate_checkpoint_strictly_round_trips() -> None:
    config = _model_config(True)
    source = PINNModel(config).eval()
    assert source.moment_linear_skip is not None
    with torch.no_grad():
        source.moment_linear_skip.weight.copy_(
            torch.linspace(-0.01, 0.01, source.hidden_dim).unsqueeze(0)
        )

    reloaded = PINNModel(copy.deepcopy(config)).eval()
    incompatible = reloaded.load_state_dict(source.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    waveform = 0.02 * torch.randn(2, 1, 40)
    metadata = _metadata(2, torch.device("cpu"))
    with torch.no_grad():
        expected = source.predict_heads(waveform, meta=metadata)
        actual = reloaded.predict_heads(waveform, meta=metadata)
    torch.testing.assert_close(actual.stf_encoded, expected.stf_encoded)
    torch.testing.assert_close(actual.catalog_mw, expected.catalog_mw)


def test_enabled_skip_rejects_direct_stf_parameterization() -> None:
    config = _model_config(True)
    config["model"]["stf_output_parameterization"] = "direct"

    with pytest.raises(ValueError, match="requires.*moment_shape_factorized"):
        moment_linear_skip_from_config(config)
    with pytest.raises(ValueError, match="requires.*moment_shape_factorized"):
        validate_config_v2(config)
    with pytest.raises(ValueError, match="requires.*moment_shape_factorized"):
        PINNModel(config)


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable",
            ),
        ),
    ],
)
def test_skip_factorized_forward_backward_is_finite(device_name: str) -> None:
    device = torch.device(device_name)
    model = PINNModel(_model_config(True)).to(device).train()
    waveform = 0.02 * torch.randn(3, 1, 40, device=device)
    prediction = model.predict_heads(waveform, meta=_metadata(3, device))
    loss = prediction.stf_encoded.mean() + prediction.catalog_mw.square().mean()
    loss.backward()

    assert torch.isfinite(prediction.stf_encoded).all()
    assert torch.isfinite(prediction.catalog_mw).all()
    assert model.moment_linear_skip is not None
    assert model.moment_linear_skip.weight.grad is not None
    assert torch.isfinite(model.moment_linear_skip.weight.grad).all()
    assert float(model.moment_linear_skip.weight.grad.abs().sum()) > 0.0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
