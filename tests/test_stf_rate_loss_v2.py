import copy
import inspect
from pathlib import Path

import pytest
import torch
import yaml

from scripts.experiments.run_experiment import collect_summary_row
from scripts.plotting.plot_training_curves import build_loss_summary
from src.physics.travel_time import ConstantVelocityTravelTime
from src.training.loss_stf_rate_v2 import (
    STFRateWaveformLossV2,
    compute_physical_coefficients,
    forward_displacement_from_origin_rate,
    masked_normalized_waveform_mse,
    moment_magnitude_from_rate,
    pinn_loss_stf_rate_v2,
)
from src.training.train import _build_stf_rate_criterion, _prepare_v2_batch


def _v2_config() -> dict:
    return yaml.safe_load(
        Path("configs/config_v2.yaml").read_text(encoding="utf-8")
    )


def _analytic_weighted_loss_arguments() -> dict[str, object]:
    return {
        "rate_hat": torch.ones(2, 2, requires_grad=True),
        "pred_rate_encoded": torch.zeros(2, 2, requires_grad=True),
        "rate_ref_encoded": torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        "rate_ref_physical": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "u_obs": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "source_dt_sec": torch.ones(2),
        "observation_dt_sec": torch.ones(2),
        "source_distance_m": torch.ones(2),
        "waveform_valid_mask": torch.ones(2, 2, dtype=torch.bool),
        "travel_time": ConstantVelocityTravelTime(2.0, 1.0),
        "rho": 1.0,
        "theta_deg": torch.zeros(2),
        "phi_slip_deg": torch.zeros(2),
        "amplitude_gain": 1.0,
        "lambda_MSE": 1.0,
        "lambda_synth": 0.5,
        "lambda_mag": 1.0,
        "lambda_shape": 0.1,
        "has_ref": torch.ones(2, dtype=torch.bool),
        "true_mag": torch.tensor([1.0, 3.0]),
        "pred_catalog_mw": torch.zeros(2, requires_grad=True),
        "include_intermediate": False,
        "radiation_mode": "simplified",
        "include_far_P": False,
        "include_far_S": False,
        "include_intermediate_P": False,
        "include_intermediate_S": False,
    }


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


def test_weighted_four_term_loss_uses_batch_mean_not_weight_sum() -> None:
    arguments = _analytic_weighted_loss_arguments()

    loss, metrics = pinn_loss_stf_rate_v2(
        **arguments,
        sample_weights=torch.tensor([2.0, 4.0]),
    )
    loss.backward()

    assert metrics["L_MSE"] == pytest.approx(9.0)
    assert metrics["L_synth"] == pytest.approx(2.5)
    assert metrics["L_mag"] == pytest.approx(19.0)
    assert metrics["L_shape"] == pytest.approx(0.25)
    assert metrics["L_total"] == pytest.approx(29.275)
    assert loss.item() == pytest.approx(
        metrics["L_MSE"]
        + 0.5 * metrics["L_synth"]
        + metrics["L_mag"]
        + 0.1 * metrics["L_shape"]
    )
    for name in ("rate_hat", "pred_rate_encoded", "pred_catalog_mw"):
        tensor = arguments[name]
        assert isinstance(tensor, torch.Tensor)
        gradient = tensor.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_unit_sample_weights_match_unweighted_four_term_loss() -> None:
    arguments = _analytic_weighted_loss_arguments()

    unweighted_loss, unweighted_metrics = pinn_loss_stf_rate_v2(**arguments)
    weighted_loss, weighted_metrics = pinn_loss_stf_rate_v2(
        **arguments,
        sample_weights=torch.ones(2),
    )

    assert weighted_loss.item() == unweighted_loss.item()
    assert weighted_metrics == unweighted_metrics
    assert unweighted_metrics["L_MSE"] == pytest.approx(2.5)
    assert unweighted_metrics["L_synth"] == pytest.approx(0.75)
    assert unweighted_metrics["L_mag"] == pytest.approx(5.0)
    assert unweighted_metrics["L_shape"] == pytest.approx(0.125)


def test_weighted_four_term_loss_respects_generic_target_masks() -> None:
    arguments = _analytic_weighted_loss_arguments()
    arguments["has_ref"] = torch.tensor([True, False])
    arguments["true_mag"] = torch.tensor([1.0, float("nan")])
    arguments["waveform_valid_mask"] = torch.tensor(
        [[True, False], [True, True]]
    )

    _, metrics = pinn_loss_stf_rate_v2(
        **arguments,
        sample_weights=torch.tensor([2.0, 4.0]),
    )

    assert metrics["L_MSE"] == pytest.approx(2.0)
    assert metrics["L_synth"] == pytest.approx(3.0)
    assert metrics["L_mag"] == pytest.approx(2.0)
    assert metrics["L_shape"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("sample_weights", "message"),
    [
        (torch.tensor(1.0), "shape"),
        (torch.tensor([1.0]), "shape"),
        (torch.tensor([1.0, float("nan")]), "finite"),
        (torch.tensor([1.0, float("inf")]), "finite"),
        (torch.tensor([1.0, 0.0]), "positive"),
        (torch.tensor([1.0, -1.0]), "positive"),
    ],
)
def test_sample_weights_must_be_a_finite_positive_batch_vector(
    sample_weights: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pinn_loss_stf_rate_v2(
            **_analytic_weighted_loss_arguments(),
            sample_weights=sample_weights,
        )


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
        sample_weights=torch.tensor([0.5, 1.5]),
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
    assert criterion.origin_aligned is False


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


def test_source_stf_loss_uses_absolute_p_and_s_delays() -> None:
    config = _v2_config()
    config.pop("workflow")
    config["physics"].update(
        {
            "alpha": 3.0,
            "beta": 1.5,
            "delay_mode": "absolute",
        }
    )
    config["training"]["stf_rate_loss"].update(
        {
            "lambda_MSE": 0.0,
            "lambda_synth": 1.0,
            "lambda_mag": 0.0,
            "lambda_shape": 0.0,
            "include_intermediate_field": False,
            "include_far_field_P": True,
            "include_far_field_S": True,
            "radiation_pattern_mode": "simplified",
        }
    )
    criterion = STFRateWaveformLossV2(config)
    physical_rate = torch.tensor([[0.0, 1.0e18, 0.0, 0.0]])
    encoded_rate = torch.log10(1.0 + physical_rate / criterion.stf_m_ref)
    distance = torch.tensor([3.0])
    zeros = torch.zeros(1)
    ones = torch.ones(1)
    coefficients = compute_physical_coefficients(
        distance,
        criterion.rho,
        criterion.alpha,
        criterion.beta,
        ones,
        ones,
        ones,
        ones,
        amplitude_gain=criterion.amplitude_gain,
    )
    observed = forward_displacement_from_origin_rate(
        physical_rate,
        source_dt_sec=torch.ones(1),
        observation_dt_sec=torch.ones(1),
        observation_steps=6,
        source_distance_m=distance,
        travel_time=criterion.travel_time,
        C_int_P=coefficients[0],
        C_int_S=coefficients[1],
        C_far_P=coefficients[2],
        C_far_S=coefficients[3],
        include_intermediate=False,
        include_far_P=True,
        include_far_S=True,
        include_intermediate_P=True,
        include_intermediate_S=True,
    )

    loss, metrics = criterion(
        encoded_rate,
        radial_obs=observed,
        source_distance_m=distance,
        theta_deg=zeros,
        phi_slip_deg=zeros,
        source_dt_sec=torch.ones(1),
        observation_dt_sec=torch.ones(1),
        waveform_valid_mask=torch.ones(1, 6, dtype=torch.bool),
    )

    assert criterion.origin_aligned is True
    assert metrics["L_synth"] == pytest.approx(0.0, abs=1.0e-12)
    assert loss.item() == pytest.approx(0.0, abs=1.0e-12)


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
