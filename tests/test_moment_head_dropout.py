from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from src.models.model import PINNModel
from src.utils.config_v2 import (
    moment_head_dropout_from_config,
    validate_config_v2,
)


def _model_config(
    enabled: bool | None,
    *,
    hidden_dim: int = 32,
    dropout: float = 0.2,
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
            "moment_linear_skip": False,
        }
    )
    if enabled is None:
        config["model"].pop("moment_head_dropout", None)
    else:
        config["model"]["moment_head_dropout"] = enabled
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


def test_config_defaults_true_and_false_requires_factorized_head() -> None:
    default_config = _model_config(None)
    assert moment_head_dropout_from_config(default_config) is True
    validate_config_v2(default_config)

    candidate_config = _model_config(False)
    assert moment_head_dropout_from_config(candidate_config) is False
    validate_config_v2(candidate_config)

    direct_config = _model_config(False)
    direct_config["model"]["stf_output_parameterization"] = "direct"
    with pytest.raises(ValueError, match="requires.*moment_shape_factorized"):
        moment_head_dropout_from_config(direct_config)
    with pytest.raises(ValueError, match="requires.*moment_shape_factorized"):
        validate_config_v2(direct_config)


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, []])
def test_config_requires_a_strict_boolean(value: object) -> None:
    config = _model_config(True)
    config["model"]["moment_head_dropout"] = value

    with pytest.raises(ValueError, match="moment_head_dropout.*boolean"):
        moment_head_dropout_from_config(config)
    with pytest.raises(ValueError, match="moment_head_dropout.*boolean"):
        validate_config_v2(config)


def test_variants_preserve_state_parameter_count_and_construction_rng() -> None:
    torch.manual_seed(2718)
    baseline = PINNModel(_model_config(True))
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(2718)
    candidate = PINNModel(_model_config(False))
    candidate_rng = torch.random.get_rng_state().clone()

    assert torch.equal(baseline_rng, candidate_rng)
    assert isinstance(baseline.log10_moment_head, nn.Sequential)
    assert isinstance(candidate.log10_moment_head, nn.Sequential)
    assert baseline.state_dict().keys() == candidate.state_dict().keys()
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key]), key
    assert sum(parameter.numel() for parameter in baseline.parameters()) == sum(
        parameter.numel() for parameter in candidate.parameters()
    )


def test_train_forward_consumes_same_dropout_rng_and_initial_gradients() -> None:
    torch.manual_seed(314)
    baseline = PINNModel(_model_config(True)).train()
    torch.manual_seed(314)
    candidate = PINNModel(_model_config(False)).train()
    waveform = 0.02 * torch.randn(3, 1, 40)
    metadata = _metadata(3, torch.device("cpu"))
    dropout_calls = {"baseline": 0, "candidate": 0}

    baseline_hook = baseline.log10_moment_head[2].register_forward_hook(
        lambda *_: dropout_calls.__setitem__(
            "baseline", dropout_calls["baseline"] + 1
        )
    )
    candidate_hook = candidate.log10_moment_head[2].register_forward_hook(
        lambda *_: dropout_calls.__setitem__(
            "candidate", dropout_calls["candidate"] + 1
        )
    )
    forward_rng = torch.random.get_rng_state().clone()
    baseline_prediction = baseline.predict_heads(waveform, meta=metadata)
    baseline_after_forward = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(forward_rng)
    candidate_prediction = candidate.predict_heads(waveform, meta=metadata)
    candidate_after_forward = torch.random.get_rng_state().clone()
    baseline_hook.remove()
    candidate_hook.remove()

    assert dropout_calls == {"baseline": 1, "candidate": 1}
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

    expected_difference = "log10_moment_head.3.weight"
    candidate_parameters = dict(candidate.named_parameters())
    observed_differences: list[str] = []
    for name, parameter in baseline.named_parameters():
        candidate_parameter = candidate_parameters[name]
        assert parameter.grad is not None, name
        assert candidate_parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.isfinite(candidate_parameter.grad).all(), name
        if not torch.equal(parameter.grad, candidate_parameter.grad):
            observed_differences.append(name)
    assert observed_differences == [expected_difference]


def test_legacy_checkpoint_strict_loads_in_both_variants() -> None:
    legacy = PINNModel(_model_config(None)).eval()
    checkpoint = legacy.state_dict()

    for enabled in (True, False):
        reloaded = PINNModel(_model_config(enabled)).eval()
        incompatible = reloaded.load_state_dict(checkpoint, strict=True)
        assert incompatible.missing_keys == []
        assert incompatible.unexpected_keys == []


@pytest.mark.parametrize("enabled", [True, False])
def test_eval_output_and_stf_integral_mw_identity(enabled: bool) -> None:
    config = _model_config(enabled)
    model = PINNModel(config).eval()
    waveform = 0.02 * torch.randn(3, 1, 40)

    with torch.no_grad():
        prediction = model.predict_heads(
            waveform,
            meta=_metadata(3, torch.device("cpu")),
        )
    m_ref = float(config["dataset"]["stf"]["m_ref"])
    source_dt = 1.0 / float(config["dataset"]["sample_rate_hz"])
    physical_rate = m_ref * torch.expm1(
        prediction.stf_encoded * math.log(10.0)
    )
    integrated_moment = physical_rate.sum(dim=1) * source_dt
    expected_moment = torch.pow(
        10.0,
        1.5 * prediction.catalog_mw + 9.1,
    )
    torch.testing.assert_close(
        integrated_moment,
        expected_moment,
        rtol=2.0e-5,
        atol=0.0,
    )


@pytest.mark.parametrize("enabled", [True, False])
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
def test_factorized_forward_backward_is_finite(
    enabled: bool,
    device_name: str,
) -> None:
    device = torch.device(device_name)
    model = PINNModel(_model_config(enabled)).to(device).train()
    waveform = 0.02 * torch.randn(3, 1, 40, device=device)
    prediction = model.predict_heads(waveform, meta=_metadata(3, device))
    loss = prediction.stf_encoded.mean() + prediction.catalog_mw.square().mean()
    loss.backward()

    assert torch.isfinite(prediction.stf_encoded).all()
    assert torch.isfinite(prediction.catalog_mw).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
