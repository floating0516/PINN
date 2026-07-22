import copy
import inspect
from pathlib import Path

import pytest
import torch
import yaml

from scripts.experiments.run_experiment import collect_summary_row
from scripts.plotting.plot_training_curves import build_loss_summary
from src.training.loss_stf_rate_v2 import (
    STFRateWaveformLossV2,
    compute_physical_coefficients,
    masked_normalized_waveform_mse,
    moment_magnitude_from_rate,
)
from src.training.train import _build_stf_rate_criterion, _prepare_v2_batch


def _v2_config() -> dict:
    return yaml.safe_load(
        Path("configs/config_v2.yaml").read_text(encoding="utf-8")
    )


def test_per_sample_source_dt_changes_only_its_own_moment() -> None:
    rate = torch.ones(2, 4) * 1.0e18

    magnitude = moment_magnitude_from_rate(
        rate,
        torch.tensor([1.0, 2.0]),
    )

    expected_delta = (2.0 / 3.0) * torch.log10(torch.tensor(2.0))
    assert torch.allclose(
        magnitude[1] - magnitude[0],
        expected_delta,
        atol=1.0e-6,
    )


def test_invalid_waveform_samples_do_not_enter_synth_loss() -> None:
    predicted = torch.tensor([[1.0, 100.0]])
    observed = torch.tensor([[1.0, 0.0]])
    valid_mask = torch.tensor([[True, False]])

    loss = masked_normalized_waveform_mse(
        predicted,
        observed,
        valid_mask,
    )

    assert loss.item() == 0.0


def test_v2_physical_coefficients_use_only_explicit_amplitude_gain() -> None:
    parameters = set(inspect.signature(compute_physical_coefficients).parameters)

    assert "amplitude_gain" in parameters
    assert {"geom", "free_surface", "attenuation"}.isdisjoint(parameters)


def test_v2_loss_is_finite_differentiable_and_has_no_nonnegative_term() -> None:
    config = _v2_config()
    criterion = STFRateWaveformLossV2(config)
    pred_rate = torch.full((2, 4), 0.3, requires_grad=True)
    pred_catalog_mw = torch.tensor([7.0, 8.0], requires_grad=True)
    stf_true = torch.ones(2, 4) * 1.0e18
    source_dt = torch.tensor([1.0, 2.0])
    true_magnitude = moment_magnitude_from_rate(stf_true, source_dt)

    loss, metrics = criterion(
        pred_rate,
        pred_catalog_mw=pred_catalog_mw,
        radial_obs=torch.zeros(2, 1, 6),
        source_distance_m=torch.tensor([1000.0, 2000.0]),
        theta_deg=torch.tensor([30.0, 45.0]),
        phi_slip_deg=torch.tensor([10.0, 20.0]),
        source_dt_sec=source_dt,
        observation_dt_sec=torch.tensor([1.0, 1.0]),
        waveform_valid_mask=torch.tensor(
            [
                [True, True, True, True, False, False],
                [True, True, True, True, True, True],
            ]
        ),
        stf_true=stf_true,
        has_stf=torch.tensor([True, True]),
        true_mag=true_magnitude,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert pred_rate.grad is not None
    assert torch.isfinite(pred_rate.grad).all()
    assert pred_catalog_mw.grad is not None
    assert torch.isfinite(pred_catalog_mw.grad).all()
    assert set(metrics) == {
        "L_total",
        "L_MSE",
        "L_synth",
        "L_mag",
        "L_shape",
        "window_mw_mean",
    }
    assert not hasattr(criterion, "lambda_nonneg")
    assert criterion.amplitude_gain == 1.0


def test_magnitude_loss_uses_scalar_head_not_window_integral() -> None:
    criterion = STFRateWaveformLossV2(_v2_config())
    pred_stf = torch.zeros(2, 300, requires_grad=True)
    pred_catalog_mw = torch.tensor([7.0, 8.0], requires_grad=True)

    loss, metrics = criterion(
        pred_stf,
        pred_catalog_mw=pred_catalog_mw,
        radial_obs=torch.zeros(2, 1, 200),
        source_distance_m=torch.tensor([1000.0, 2000.0]),
        theta_deg=torch.tensor([30.0, 45.0]),
        phi_slip_deg=torch.tensor([10.0, 20.0]),
        source_dt_sec=torch.ones(2),
        observation_dt_sec=torch.ones(2),
        waveform_valid_mask=torch.ones(2, 200, dtype=torch.bool),
        stf_true=torch.zeros(2, 300),
        has_stf=torch.tensor([True, True]),
        true_mag=torch.tensor([7.5, 8.5]),
    )
    loss.backward()

    assert pred_catalog_mw.grad is not None
    assert metrics["L_mag"] == pytest.approx(0.25)
    assert metrics["window_mw_mean"] == pytest.approx(
        float(
            moment_magnitude_from_rate(pred_stf, torch.ones(2))
            .mean()
            .detach()
        )
    )


def test_legacy_v2_integral_magnitude_loss_keeps_rate_gradient() -> None:
    config = _v2_config()
    config.pop("workflow")
    config["physics"]["delay_mode"] = "absolute"
    config["training"]["stf_rate_loss"].update(
        lambda_MSE=0.0,
        lambda_synth=0.0,
        lambda_mag=1.0,
        lambda_shape=0.0,
    )
    criterion = STFRateWaveformLossV2(config)
    pred_stf = torch.full((1, 4), 0.3, requires_grad=True)

    loss, _ = criterion(
        pred_stf,
        radial_obs=torch.zeros(1, 1, 6),
        source_distance_m=torch.tensor([1000.0]),
        theta_deg=torch.tensor([30.0]),
        phi_slip_deg=torch.tensor([10.0]),
        source_dt_sec=torch.ones(1),
        observation_dt_sec=torch.ones(1),
        waveform_valid_mask=torch.ones(1, 6, dtype=torch.bool),
        true_mag=torch.tensor([7.5]),
    )
    loss.backward()

    assert pred_stf.grad is not None
    assert torch.count_nonzero(pred_stf.grad) > 0


def test_v2_batch_preparation_uses_explicit_fields() -> None:
    config = _v2_config()
    batch = {
        "radial": torch.zeros(2, 1, 200),
        "vertical": torch.zeros(2, 200),
        "source_distance_m": torch.tensor([10_000.0, 20_000.0]),
        "theta_deg": torch.tensor([30.0, 40.0]),
        "azimuth_deg": torch.tensor([90.0, 180.0]),
        "phi_slip_deg": torch.tensor([5.0, 15.0]),
        "stf": torch.ones(2, 200),
        "stf_log": torch.zeros(2, 200),
        "stf_dt_sec": torch.tensor([1.0, 2.0]),
        "waveform_dt_sec": torch.tensor([1.0, 1.0]),
        "waveform_valid_mask": torch.ones(2, 200, dtype=torch.bool),
        "has_stf": torch.tensor([True, True]),
        "mw_stf_native": torch.tensor([7.0, 7.1]),
        "magnitude_catalog": torch.tensor([6.9, 7.2]),
    }

    prepared = _prepare_v2_batch(batch, config, torch.device("cpu"))

    assert torch.equal(
        prepared.source_distance_m,
        batch["source_distance_m"],
    )
    assert torch.equal(prepared.source_dt_sec, batch["stf_dt_sec"])
    assert torch.equal(
        prepared.observation_dt_sec,
        batch["waveform_dt_sec"],
    )
    assert torch.equal(prepared.true_mag, batch["magnitude_catalog"])
    assert torch.allclose(prepared.metadata[:, 0], torch.log(batch["source_distance_m"]))
    assert torch.allclose(
        prepared.metadata[:, 3],
        torch.sin(torch.deg2rad(batch["azimuth_deg"])),
    )


def test_v2_dispatch_and_reporting_exclude_nonnegative_loss() -> None:
    config = _v2_config()

    criterion = _build_stf_rate_criterion(config, torch.device("cpu"))
    summary = build_loss_summary(config)
    row = collect_summary_row(
        experiment_name="v2",
        config=config,
        train_result=None,
        eval_result=None,
        unseen_result=None,
        overrides=[],
    )

    assert isinstance(criterion, STFRateWaveformLossV2)
    assert "nonneg" not in summary.lower()
    assert "lambda_nonneg" not in row

    legacy = copy.deepcopy(config)
    legacy["pipeline_version"] = 1
    legacy["training"]["stf_rate_loss"]["lambda_nonneg"] = 0.5
    legacy_row = collect_summary_row(
        experiment_name="legacy",
        config=legacy,
        train_result=None,
        eval_result=None,
        unseen_result=None,
        overrides=[],
    )
    assert legacy_row["lambda_nonneg"] == 0.5
