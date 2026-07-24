from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from src.data.waveform import WaveformConfig
from src.evaluation.delayed_prefix import preprocess_and_release_delayed_prefix
from src.models.model import PINNModel
from src.utils.config_v2 import (
    radial_dynamic_range_stem_from_config,
    validate_config_v2,
)


_RESIDUAL_STATE_KEY = "radial_asinh_zero_conv.weight"


def _model_config(
    stem: str | None = "none",
    *,
    parameterization: str = "direct",
    hidden_dim: int = 32,
    num_tcn_blocks: int = 2,
    dropout: float = 0.0,
) -> dict:
    config = yaml.safe_load(
        Path("configs/experiments/manuscript_station_stf_usgs.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["model"].update(
        {
            "hidden_dim": hidden_dim,
            "num_layers": 1,
            "num_tcn_blocks": num_tcn_blocks,
            "transformer_num_layers": 1,
            "dropout": dropout,
            "stf_output_parameterization": parameterization,
        }
    )
    if stem is None:
        config["model"].pop("radial_dynamic_range_stem", None)
    else:
        config["model"]["radial_dynamic_range_stem"] = stem
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


def _public_state(model: PINNModel) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if key != _RESIDUAL_STATE_KEY
    }


@pytest.mark.parametrize(
    "parameterization",
    ["direct", "moment_shape_factorized"],
)
def test_zero_asinh_stem_preserves_state_rng_output_and_common_gradients(
    parameterization: str,
) -> None:
    baseline_config = _model_config(
        "none",
        parameterization=parameterization,
        dropout=0.2,
    )
    candidate_config = _model_config(
        "asinh_residual",
        parameterization=parameterization,
        dropout=0.2,
    )

    torch.manual_seed(2718)
    baseline = PINNModel(baseline_config).train()
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(2718)
    candidate = PINNModel(candidate_config).train()
    candidate_rng = torch.random.get_rng_state().clone()

    assert torch.equal(baseline_rng, candidate_rng)
    assert _RESIDUAL_STATE_KEY not in baseline.state_dict()
    assert set(candidate.state_dict()) == set(baseline.state_dict()) | {
        _RESIDUAL_STATE_KEY
    }
    assert list(candidate.state_dict())[-1] == _RESIDUAL_STATE_KEY
    baseline_state = baseline.state_dict()
    for key, value in _public_state(candidate).items():
        assert torch.equal(value, baseline_state[key]), key

    waveform = 0.02 * torch.randn(2, 1, 24)
    metadata = _metadata(2, torch.device("cpu"))
    forward_rng = torch.random.get_rng_state().clone()
    baseline_output = baseline(waveform, meta=metadata)
    baseline_after_forward = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(forward_rng)
    candidate_output = candidate(waveform, meta=metadata)
    candidate_after_forward = torch.random.get_rng_state().clone()

    assert torch.equal(baseline_after_forward, candidate_after_forward)
    torch.testing.assert_close(
        candidate_output,
        baseline_output,
        rtol=0.0,
        atol=0.0,
    )

    baseline_output.sum().backward()
    candidate_output.sum().backward()
    baseline_parameters = dict(baseline.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    for name, parameter in baseline_parameters.items():
        candidate_parameter = candidate_parameters[name]
        assert parameter.grad is not None, name
        assert candidate_parameter.grad is not None, name
        torch.testing.assert_close(
            candidate_parameter.grad,
            parameter.grad,
            rtol=0.0,
            atol=0.0,
        )


def test_none_stem_keeps_the_original_embed_path_exact() -> None:
    model = PINNModel(_model_config("none")).eval()
    waveform = 0.02 * torch.randn(3, 1, 20)

    with torch.no_grad():
        original = model.embed(waveform)
        split_path = model._embed_backbone_input(waveform)

    torch.testing.assert_close(split_path, original, rtol=0.0, atol=0.0)


def test_asinh_stem_has_exact_shape_count_initialization_and_placement() -> None:
    baseline = PINNModel(_model_config("none", hidden_dim=128))
    candidate = PINNModel(_model_config("asinh_residual", hidden_dim=128))
    residual = candidate.radial_asinh_zero_conv

    assert residual is not None
    assert list(residual.named_parameters())[0][0] == "weight"
    assert len(list(residual.parameters())) == 1
    assert residual.weight.shape == (128, 1, 7)
    assert residual.weight.numel() == 896
    assert torch.count_nonzero(residual.weight) == 0
    assert (
        sum(parameter.numel() for parameter in candidate.parameters())
        - sum(parameter.numel() for parameter in baseline.parameters())
        == 896
    )

    waveform = 0.02 * torch.randn(2, 1, 18)
    with torch.no_grad():
        residual.weight[:, 0, 3].fill_(0.01)
        expected_residual = F.conv1d(
            torch.asinh(waveform / 0.01),
            residual.weight,
            bias=None,
            padding=3,
        )
        expected = candidate.embed[2](
            candidate.embed[1](candidate.embed[0](waveform) + expected_residual)
        )
        actual = candidate._embed_backbone_input(waveform)

    assert expected_residual.shape == (2, 128, 18)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def _initial_factorized_backward(
    model: PINNModel,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    waveform = 0.02 * torch.randn(3, 1, 32, device=device)
    prediction = model.predict_heads(
        waveform,
        meta=_metadata(3, device),
    )
    time_weights = torch.linspace(
        0.5,
        1.5,
        prediction.stf_encoded.size(1),
        device=device,
    )
    loss = (
        prediction.stf_encoded * time_weights.unsqueeze(0)
    ).mean() + prediction.catalog_mw.square().mean()
    loss.backward()
    return prediction.stf_encoded, prediction.catalog_mw


def test_zero_asinh_stem_receives_nonzero_gradient_on_first_backward() -> None:
    model = PINNModel(
        _model_config(
            "asinh_residual",
            parameterization="moment_shape_factorized",
        )
    ).train()
    residual = model.radial_asinh_zero_conv

    assert residual is not None
    assert torch.count_nonzero(residual.weight) == 0
    _initial_factorized_backward(model, device=torch.device("cpu"))

    assert residual.weight.grad is not None
    assert torch.isfinite(residual.weight.grad).all()
    assert float(residual.weight.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("value", ["", "asinh", "dual", "ASINH_RESIDUAL", 1, None])
def test_config_rejects_unknown_radial_dynamic_range_stem(value: object) -> None:
    config = _model_config("none")
    config["model"]["radial_dynamic_range_stem"] = value

    with pytest.raises(ValueError, match="radial_dynamic_range_stem"):
        radial_dynamic_range_stem_from_config(config)
    with pytest.raises(ValueError, match="radial_dynamic_range_stem"):
        validate_config_v2(config)
    with pytest.raises(ValueError, match="radial_dynamic_range_stem"):
        PINNModel(config)


def test_asinh_stem_rejects_non_radial_model_input() -> None:
    config = _model_config("asinh_residual")
    config["model"]["input_components"] = ["radial", "tangential"]

    with pytest.raises(ValueError, match="requires R-only input"):
        radial_dynamic_range_stem_from_config(config)
    with pytest.raises(ValueError, match="requires R-only input"):
        validate_config_v2(config)
    with pytest.raises(ValueError, match="requires R-only input"):
        PINNModel(config)


def test_default_none_keeps_legacy_checkpoint_strictly_compatible() -> None:
    default_config = _model_config(None)
    explicit_config = _model_config("none")
    torch.manual_seed(1618)
    legacy_model = PINNModel(default_config)
    torch.manual_seed(1618)
    explicit_model = PINNModel(explicit_config)

    assert radial_dynamic_range_stem_from_config(default_config) == "none"
    assert legacy_model.state_dict().keys() == explicit_model.state_dict().keys()
    incompatible = explicit_model.load_state_dict(
        legacy_model.state_dict(),
        strict=True,
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


@pytest.mark.parametrize(
    "parameterization",
    ["direct", "moment_shape_factorized"],
)
def test_asinh_stem_checkpoint_strict_reload_supports_both_stf_heads(
    parameterization: str,
) -> None:
    config = _model_config(
        "asinh_residual",
        parameterization=parameterization,
    )
    source = PINNModel(config).eval()
    assert source.radial_asinh_zero_conv is not None
    with torch.no_grad():
        source.radial_asinh_zero_conv.weight.fill_(0.001)

    reloaded = PINNModel(copy.deepcopy(config)).eval()
    incompatible = reloaded.load_state_dict(source.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    waveform = 0.02 * torch.randn(2, 1, 30)
    metadata = _metadata(2, torch.device("cpu"))
    with torch.no_grad():
        expected = source(waveform, meta=metadata)
        actual = reloaded(waveform, meta=metadata)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.isfinite(actual).all()

    if parameterization == "moment_shape_factorized":
        with torch.no_grad():
            prediction = reloaded.predict_heads(waveform, meta=metadata)
        m_ref = float(config["dataset"]["stf"]["m_ref"])
        source_dt = 1.0 / float(config["dataset"]["sample_rate_hz"])
        rate = m_ref * torch.expm1(prediction.stf_encoded * math.log(10.0))
        integrated_moment = rate.sum(dim=1) * source_dt
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


def _waveform_config() -> WaveformConfig:
    return WaveformConfig(
        sample_rate_hz=1.0,
        start_sec=0.0,
        duration_sec=12.0,
        min_valid_fraction=0.8,
        max_interpolation_gap_sec=0.0,
        baseline_method="median",
        pre_event_start_sec=-3.0,
        pre_event_end_sec=0.0,
        baseline_fallback="pre_p",
        baseline_fallback_max_sec=3.0,
        baseline_min_samples=3,
        filter_type="lowpass",
        cutoff_hz=0.1,
        num_taps=7,
        filter_window="hamming",
    )


def test_asinh_stem_delayed_prefix_cannot_read_unavailable_future() -> None:
    waveform_config = _waveform_config()
    time_sec = np.arange(-3.0, 12.0)
    values = np.where(time_sec < 0.0, 0.0, 0.01 * time_sec)
    changed = values.copy()
    changed[time_sec > 9.0] = -1.0e6
    releases = [
        preprocess_and_release_delayed_prefix(
            time_sec,
            item,
            units="m",
            p_arrival_sec=5.0,
            prefix_steps=4,
            config=waveform_config,
        )
        for item in (values, changed)
    ]
    np.testing.assert_array_equal(
        releases[0].masked_waveform_m,
        releases[1].masked_waveform_m,
    )
    assert releases[0].issue_time_sec == 9.0
    assert np.all(releases[0].masked_waveform_m[4:] == 0.0)

    model_config = _model_config(
        "asinh_residual",
        parameterization="moment_shape_factorized",
        num_tcn_blocks=1,
    )
    model_config["dataset"]["waveform"]["duration_sec"] = 12.0
    model_config["dataset"]["stf"]["duration_sec"] = 12.0
    model = PINNModel(model_config).eval()
    assert model.radial_asinh_zero_conv is not None
    assert model.log10_moment_head is not None
    with torch.no_grad():
        model.radial_asinh_zero_conv.weight.fill_(0.01)
        model.log10_moment_head[-1].weight.fill_(0.01)
        predictions = [
            model.predict_heads(
                torch.from_numpy(release.masked_waveform_m)
                .float()
                .reshape(1, 1, -1),
                meta=torch.zeros(1, 5),
            )
            for release in releases
        ]

    torch.testing.assert_close(
        predictions[0].stf_encoded,
        predictions[1].stf_encoded,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        predictions[0].catalog_mw,
        predictions[1].catalog_mw,
        rtol=0.0,
        atol=0.0,
    )


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
def test_asinh_stem_factorized_forward_backward_is_finite(
    device_name: str,
) -> None:
    device = torch.device(device_name)
    model = PINNModel(
        _model_config(
            "asinh_residual",
            parameterization="moment_shape_factorized",
            dropout=0.0,
        )
    ).to(device).train()

    stf_encoded, catalog_mw = _initial_factorized_backward(
        model,
        device=device,
    )

    assert torch.isfinite(stf_encoded).all()
    assert torch.isfinite(catalog_mw).all()
    assert model.radial_asinh_zero_conv is not None
    assert model.radial_asinh_zero_conv.weight.grad is not None
    assert torch.isfinite(model.radial_asinh_zero_conv.weight.grad).all()
    assert float(model.radial_asinh_zero_conv.weight.grad.abs().sum()) > 0.0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
