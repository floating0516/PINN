from pathlib import Path

import pytest
import torch
import yaml

from src.models.causal_forward_guided import (
    CausalForwardGuidedEventNet,
    CausalForwardGuidedSpec,
)
from src.training.loss_stf_rate_v2 import CausalEventSTFRateWaveformLossV2
from scripts.experiments.run_causal_forward_guided_event_neural import (
    _load_yaml,
    _validate_experiment_config,
)


def _spec() -> CausalForwardGuidedSpec:
    return CausalForwardGuidedSpec(
        top_k=2,
        duration_sec=8,
        sample_rate_hz=1.0,
        causal_latency_samples=1,
        hidden_dim=8,
        num_tcn_blocks=1,
        transformer_num_layers=1,
        transformer_num_heads=2,
        dropout=0.0,
    )


def _model(spec: CausalForwardGuidedSpec) -> CausalForwardGuidedEventNet:
    return CausalForwardGuidedEventNet(
        anchor_feature_mean=torch.zeros(spec.anchor_feature_count),
        anchor_feature_scale=torch.ones(spec.anchor_feature_count),
        online_feature_mean=torch.zeros(spec.feature_count),
        online_feature_scale=torch.ones(spec.feature_count),
        metadata_mean=torch.zeros(5),
        metadata_scale=torch.ones(5),
        target_mean=7.0,
        target_scale=1.0,
        stf_encoded_mean=torch.full((spec.total_steps,), 0.1),
        spec=spec,
    )


def test_model_keeps_tcn_transformer_and_ignores_unreleased_future() -> None:
    torch.manual_seed(3)
    spec = _spec()
    model = _model(spec).eval()
    radial = torch.randn(1, 2, spec.total_steps) * 0.01
    changed = radial.clone()
    changed[:, :, 4:] = torch.randn_like(changed[:, :, 4:]) * 1000.0
    valid_mask = torch.ones_like(radial, dtype=torch.bool)
    valid_mask[:, 0, 0] = False
    arguments = {
        "waveform_valid_mask": valid_mask,
        "station_metadata": torch.zeros(1, 2, 5),
        "station_mask": torch.ones(1, 2, dtype=torch.bool),
        "observed_steps": torch.tensor([4]),
        "online_features": torch.zeros(1, spec.feature_count),
    }

    first = model(radial_m=radial, **arguments)
    second = model(radial_m=changed, **arguments)

    assert len(model.tcn_blocks) == 1
    assert model.transformer.num_layers == 1
    torch.testing.assert_close(first.catalog_mw, second.catalog_mw, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.stf_encoded, second.stf_encoded, rtol=0.0, atol=0.0)


def test_shared_event_loss_preserves_all_four_terms_and_gradients() -> None:
    config = yaml.safe_load(Path("configs/config_v2.yaml").read_text(encoding="utf-8"))
    config["training"]["stf_rate_loss"].update(
        lambda_MSE=1.0,
        lambda_synth=0.5,
        lambda_mag=1.0,
        lambda_shape=0.1,
        radiation_pattern_mode="full",
        include_intermediate_field=False,
    )
    criterion = CausalEventSTFRateWaveformLossV2(config)
    pred_rate = torch.full((1, 8), 0.2, requires_grad=True)
    pred_mw = torch.tensor([7.2], requires_grad=True)
    radial = torch.linspace(0.0, 0.01, 8).reshape(1, 1, 8)

    loss, metrics = criterion(
        pred_rate,
        pred_catalog_mw=pred_mw,
        rate_ref_encoded=torch.full((1, 8), 0.1),
        rate_ref_physical=torch.full((1, 8), 1.0e18),
        true_mag=torch.tensor([7.0]),
        radial_obs=radial,
        source_distance_m=torch.tensor([[7900.0]]),
        theta_deg=torch.tensor([[45.0]]),
        phi_slip_deg=torch.tensor([[30.0]]),
        source_dt_sec=torch.ones(1),
        observation_dt_sec=torch.ones(1, 1),
        waveform_valid_mask=torch.ones(1, 1, 8, dtype=torch.bool),
        station_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    loss.backward()

    assert set(metrics) == {
        "L_total",
        "L_MSE",
        "L_synth",
        "L_mag",
        "L_shape",
        "window_mw_mean",
    }
    assert metrics["L_MSE"] > 0.0
    assert metrics["L_synth"] > 0.0
    assert metrics["L_mag"] > 0.0
    assert metrics["L_shape"] >= 0.0
    assert metrics["L_total"] == pytest.approx(
        metrics["L_MSE"]
        + 0.5 * metrics["L_synth"]
        + metrics["L_mag"]
        + 0.1 * metrics["L_shape"]
    )
    assert pred_rate.grad is not None and torch.isfinite(pred_rate.grad).all()
    assert pred_mw.grad is not None and torch.isfinite(pred_mw.grad).all()


def test_magnitude_residual_is_nonzero_early_and_exactly_zero_at_final() -> None:
    spec = _spec()
    model = _model(spec).eval()
    with torch.no_grad():
        model.magnitude_residual_head[-1].bias.fill_(1.0)
    radial = torch.ones(2, 2, spec.total_steps) * 0.01
    online_features = torch.zeros(2, spec.feature_count)
    online_features[0, spec.event_spec.time_fraction_index] = 0.5
    online_features[1, spec.event_spec.time_fraction_index] = 1.0

    prediction = model(
        radial_m=radial,
        waveform_valid_mask=torch.ones_like(radial, dtype=torch.bool),
        station_metadata=torch.zeros(2, 2, 5),
        station_mask=torch.ones(2, 2, dtype=torch.bool),
        observed_steps=torch.tensor([4, spec.total_steps]),
        online_features=online_features,
    )

    assert prediction.magnitude_residual[0].abs().item() > 0.0
    assert prediction.magnitude_residual[1].item() == 0.0


def test_full_and_no_synth_configs_differ_only_in_forward_weight() -> None:
    full = _load_yaml(
        Path("configs/experiments/causal_forward_guided_event_neural.yaml")
    )
    no_synth = _load_yaml(
        Path(
            "configs/experiments/"
            "causal_forward_guided_event_neural_no_synth.yaml"
        )
    )
    _validate_experiment_config(full)
    _validate_experiment_config(no_synth)

    assert full["loss"]["lambda_synth"] == 0.5
    assert no_synth["loss"]["lambda_synth"] == 0.0
    full["loss"]["lambda_synth"] = 0.0
    assert full == no_synth
