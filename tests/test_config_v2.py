from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.evaluation.evaluate import evaluate
from src.evaluation.evaluate_unseen import evaluate_unseen_events
from src.training.train import train
from src.utils.config_v2 import (
    magnitude_penalty_from_config,
    moment_linear_skip_from_config,
    stf_m_ref_from_config,
    stf_output_steps_from_config,
    validate_config_on_startup,
    validate_config_v2,
    waveform_input_components_from_config,
)


def _minimal_v2() -> dict:
    return {
        "pipeline_version": 2,
        "dataset": {
            "sample_rate_hz": 1.0,
            "waveform": {
                "start_sec": 0.0,
                "duration_sec": 200.0,
                "min_valid_fraction": 0.99,
            },
            "baseline": {
                "method": "median",
                "pre_event_start_sec": -60.0,
                "pre_event_end_sec": 0.0,
                "fallback": "pre_p",
                "fallback_max_sec": 30.0,
                "min_samples": 10,
            },
            "filter": {
                "type": "lowpass",
                "cutoff_hz": 0.1,
                "num_taps": 7,
            },
            "stf": {
                "start_sec": 0.0,
                "duration_sec": 200.0,
                "min_retained_moment_fraction": 0.995,
                "preserve_integral": True,
                "m_ref": 1.0e18,
            },
        },
        "physics": {
            "rho": 3400.0,
            "alpha": 7900.0,
            "beta": 4533.0,
            "distance_mode": "hypocentral",
            "delay_mode": "absolute",
            "amplitude_gain": 1.0,
        },
        "training": {
            "rate_representation": "log1p",
            "stf_rate_loss": {
                "lambda_MSE": 1.0,
                "lambda_synth": 0.5,
                "lambda_mag": 1.0,
                "lambda_shape": 0.1,
            },
        },
        "evaluation": {
            "primary_reference": "catalog",
            "aggregation": "event_median",
        },
    }


def _active_station_v2() -> dict:
    config = _minimal_v2()
    config["workflow"] = "station_random_shifted_stf"
    config["dataset"]["stf"].update(
        {
            "station_window_duration_sec": 300.0,
            "station_alignment": "p_arrival",
            "station_preserve_integral": False,
            "magnitude_target": "stf_native",
        }
    )
    config["physics"].update(
        {
            "travel_time_model": "constant_velocity",
            "delay_mode": "p_aligned_relative",
        }
    )
    config["model"] = {
        "predict_catalog_mw": True,
        "catalog_mw_initial_bias": 8.0,
    }
    config["training"].update(
        {
            "split_protocol": "within_event_station",
            "validation_event_fraction": 0.15,
            "test_event_fraction": 0.15,
            "event_balanced_sampling": False,
            "early_stop_patience": 0,
            "checkpoint_metric": "station_mae_catalog",
        }
    )
    config["evaluation"]["external_role"] = "sanity"
    return config


def test_valid_v2_config_passes() -> None:
    validate_config_v2(_minimal_v2())


def test_magnitude_penalty_defaults_and_validates() -> None:
    config = _minimal_v2()
    assert magnitude_penalty_from_config(config) == "squared"

    for value in ("squared", "absolute"):
        config["training"]["stf_rate_loss"]["magnitude_penalty"] = value
        validate_config_v2(config)
        assert magnitude_penalty_from_config(config) == value

    for value in ("mae", "SQUARED", True, 1):
        config["training"]["stf_rate_loss"]["magnitude_penalty"] = value
        with pytest.raises(ValueError, match="magnitude_penalty"):
            validate_config_v2(config)


def test_moment_linear_skip_defaults_and_accepts_factorized_candidate() -> None:
    config = _minimal_v2()
    assert moment_linear_skip_from_config(config) is False
    validate_config_v2(config)

    config["model"] = {
        "stf_output_parameterization": "moment_shape_factorized",
        "moment_linear_skip": True,
    }
    assert moment_linear_skip_from_config(config) is True
    validate_config_v2(config)


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, []])
def test_moment_linear_skip_requires_a_strict_boolean(value: object) -> None:
    config = _minimal_v2()
    config["model"] = {
        "stf_output_parameterization": "moment_shape_factorized",
        "moment_linear_skip": value,
    }

    with pytest.raises(ValueError, match="moment_linear_skip.*boolean"):
        moment_linear_skip_from_config(config)
    with pytest.raises(ValueError, match="moment_linear_skip.*boolean"):
        validate_config_v2(config)


def test_moment_linear_skip_rejects_direct_parameterization() -> None:
    config = _minimal_v2()
    config["model"] = {
        "stf_output_parameterization": "direct",
        "moment_linear_skip": True,
    }

    with pytest.raises(ValueError, match="requires.*moment_shape_factorized"):
        validate_config_v2(config)


def test_manuscript_point_two_hz_filter_passes() -> None:
    config = _minimal_v2()
    config["dataset"]["filter"]["cutoff_hz"] = 0.2

    validate_config_v2(config)


def test_source_aligned_workflow_rejects_string_event_balanced_flag() -> None:
    config = _minimal_v2()
    config["training"]["event_balanced_sampling"] = "false"

    with pytest.raises(ValueError, match="event_balanced_sampling"):
        validate_config_v2(config)


def test_event_balance_estimator_is_validated_with_its_enable_flag() -> None:
    config = _minimal_v2()
    config["training"]["event_balance_estimator"] = "unknown"
    with pytest.raises(ValueError, match="event_balance_estimator"):
        validate_config_v2(config)

    config["training"]["event_balance_estimator"] = "inverse_count_full_data"
    with pytest.raises(ValueError, match="event_balanced_sampling=true"):
        validate_config_v2(config)

    config["training"]["event_balanced_sampling"] = True
    validate_config_v2(config)


def test_event_balance_exponent_defaults_accepts_tempering_and_validates() -> None:
    config = _minimal_v2()
    validate_config_v2(config)

    config["training"].update(
        event_balanced_sampling=True,
        event_balance_estimator="inverse_count_full_data",
        event_balance_exponent=0.5,
    )
    validate_config_v2(config)

    for value in (True, "0.5", float("nan"), float("inf"), -0.1, 1.1):
        config["training"]["event_balance_exponent"] = value
        with pytest.raises(ValueError, match="event_balance_exponent"):
            validate_config_v2(config)


@pytest.mark.parametrize(
    "training_update",
    [
        {
            "event_balanced_sampling": False,
            "event_balance_exponent": 0.5,
        },
        {
            "event_balanced_sampling": True,
            "event_balance_estimator": "replacement_sampling",
            "event_balance_exponent": 0.5,
        },
    ],
)
def test_tempered_event_balance_exponent_requires_full_data_inverse_count(
    training_update: dict[str, object],
) -> None:
    config = _minimal_v2()
    config["training"].update(training_update)

    with pytest.raises(ValueError, match="requires.*inverse_count_full_data"):
        validate_config_v2(config)


def test_explicit_unit_event_balance_exponent_is_legacy_compatible() -> None:
    config = _minimal_v2()
    config["training"]["event_balance_exponent"] = 1.0

    validate_config_v2(config)


def test_waveform_components_default_to_radial() -> None:
    config = _minimal_v2()

    validate_config_v2(config)

    assert waveform_input_components_from_config(config) == ("radial",)


def test_waveform_components_accept_canonical_rt() -> None:
    config = _minimal_v2()
    config.setdefault("model", {})["input_components"] = [
        "radial",
        "tangential",
    ]

    validate_config_v2(config)

    assert waveform_input_components_from_config(config) == (
        "radial",
        "tangential",
    )


@pytest.mark.parametrize(
    "value",
    [
        "radial",
        [],
        ["tangential"],
        ["tangential", "radial"],
        ["radial", "radial"],
        ["radial", "vertical"],
    ],
)
def test_waveform_components_reject_ambiguous_values(value: object) -> None:
    config = _minimal_v2()
    config.setdefault("model", {})["input_components"] = value

    with pytest.raises(ValueError, match="input_components"):
        validate_config_v2(config)


def test_active_station_workflow_has_fixed_three_hundred_step_contract() -> None:
    config = _active_station_v2()

    validate_config_v2(config)

    assert stf_output_steps_from_config(config) == 300


def test_active_station_workflow_accepts_event_checkpoint_metric() -> None:
    config = _active_station_v2()
    config["training"]["checkpoint_metric"] = "event_mae_catalog"

    validate_config_v2(config)


def test_active_station_workflow_accepts_event_balanced_sampling() -> None:
    config = _active_station_v2()
    config["training"]["event_balanced_sampling"] = True

    validate_config_v2(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("dataset", "stf", "station_window_duration_sec"), 299.0, "station_window_duration_sec"),
        (("dataset", "stf", "station_alignment"), "source", "station_alignment"),
        (("dataset", "stf", "station_preserve_integral"), True, "station_preserve_integral"),
        (("dataset", "stf", "magnitude_target"), "catalog", "magnitude_target"),
        (("physics", "travel_time_model"), "crust1", "travel_time_model"),
        (("physics", "delay_mode"), "absolute", "delay_mode"),
        (("model", "predict_catalog_mw"), False, "predict_catalog_mw"),
        (("training", "split_protocol"), "grouped_event", "split_protocol"),
        (("training", "event_balanced_sampling"), "true", "event_balanced_sampling"),
        (("training", "early_stop_patience"), 50, "early_stop_patience"),
        (("training", "checkpoint_metric"), "val_loss", "checkpoint_metric"),
        (("evaluation", "external_role"), "validation", "external_role"),
    ],
)
def test_active_station_workflow_rejects_conflicting_values(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    config = _active_station_v2()
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        validate_config_v2(config)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("dataset", "p_velocity_mps"),
        ("physics", "attenuation"),
        ("physics", "geometrical_spreading_factor"),
        ("physics", "free_surface_factor"),
    ],
)
def test_v2_rejects_legacy_top_level_keys(section: str, key: str) -> None:
    config = _minimal_v2()
    config[section][key] = 1.0

    with pytest.raises(ValueError, match=key):
        validate_config_v2(config)


def test_v2_rejects_skip_delays_and_nonnegative_loss() -> None:
    config = _minimal_v2()
    config["training"]["stf_rate_loss"]["skip_travel_delays"] = True
    config["training"]["stf_rate_loss"]["lambda_nonneg"] = 0.5

    with pytest.raises(ValueError, match="skip_travel_delays|lambda_nonneg"):
        validate_config_v2(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("pipeline_version",), 1, "pipeline_version"),
        (("dataset", "sample_rate_hz"), 2.0, "sample_rate_hz"),
        (("dataset", "filter", "num_taps"), 6, "num_taps"),
        (
            ("dataset", "stf", "min_retained_moment_fraction"),
            0.99,
            "min_retained_moment_fraction",
        ),
        (("dataset", "stf", "m_ref"), 0.0, "m_ref"),
        (("training", "rate_representation"), "auto", "rate_representation"),
        (("physics", "distance_mode"), "epicentral", "distance_mode"),
        (("physics", "delay_mode"), "none", "delay_mode"),
    ],
)
def test_v2_rejects_invalid_fixed_semantics(
    path: tuple[str, ...], value: object, message: str
) -> None:
    config = _minimal_v2()
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        validate_config_v2(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("dataset", "filter", "type"), "highpass", "filter.type"),
        (("dataset", "filter", "cutoff_hz"), 0.15, "filter.cutoff_hz"),
        (("dataset", "filter", "num_taps"), 5, "filter.num_taps"),
        (("dataset", "stf", "start_sec"), 1.0, "stf.start_sec"),
        (("dataset", "stf", "duration_sec"), 0.0, "stf.duration_sec"),
        (
            ("dataset", "stf", "preserve_integral"),
            False,
            "stf.preserve_integral",
        ),
    ],
)
def test_v2_rejects_noncanonical_filter_and_stf_semantics(
    path: tuple[str, ...], value: object, message: str
) -> None:
    config = _minimal_v2()
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        validate_config_v2(config)


def test_v2_requires_stf_duration() -> None:
    config = _minimal_v2()
    del config["dataset"]["stf"]["duration_sec"]

    with pytest.raises(ValueError, match="stf.duration_sec"):
        validate_config_v2(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("pipeline_version",), 2.5, "pipeline_version"),
        (("pipeline_version",), 2.0, "pipeline_version"),
        (("pipeline_version",), True, "pipeline_version"),
        (("dataset", "sample_rate_hz"), "1.0", "sample_rate_hz"),
        (("dataset", "sample_rate_hz"), True, "sample_rate_hz"),
        (("dataset", "filter", "cutoff_hz"), float("nan"), "cutoff_hz"),
        (("dataset", "filter", "num_taps"), 7.0, "num_taps"),
        (("dataset", "filter", "num_taps"), "7", "num_taps"),
        (("dataset", "filter", "num_taps"), True, "num_taps"),
        (("dataset", "stf", "start_sec"), float("nan"), "start_sec"),
        (("dataset", "stf", "duration_sec"), float("inf"), "duration_sec"),
        (
            ("dataset", "stf", "min_retained_moment_fraction"),
            float("nan"),
            "min_retained_moment_fraction",
        ),
        (
            ("dataset", "stf", "min_retained_moment_fraction"),
            float("inf"),
            "min_retained_moment_fraction",
        ),
        (("dataset", "stf", "m_ref"), float("nan"), "m_ref"),
        (("dataset", "stf", "m_ref"), float("inf"), "m_ref"),
    ],
)
def test_v2_rejects_malformed_numeric_values(
    path: tuple[str, ...], value: object, message: str
) -> None:
    config = _minimal_v2()
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        validate_config_v2(config)


def test_v2_reports_missing_required_section_as_value_error() -> None:
    config = _minimal_v2()
    del config["dataset"]["stf"]

    with pytest.raises(ValueError, match="dataset.stf"):
        validate_config_v2(config)


def test_v2_stf_helpers_use_the_validated_contract() -> None:
    config = _minimal_v2()

    assert stf_m_ref_from_config(config) == 1.0e18
    assert stf_output_steps_from_config(config) == 200


def _invalid_entry_config() -> dict:
    config = deepcopy(_minimal_v2())
    config["dataset"]["p_velocity_mps"] = 4500.0
    return config


@pytest.mark.parametrize(
    "legacy_config",
    [
        {},
        {"pipeline_version": 1},
    ],
)
def test_startup_validation_preserves_legacy_configs(legacy_config: dict) -> None:
    validate_config_on_startup(legacy_config)


@pytest.mark.parametrize(
    "pipeline_version",
    [2.0, 2.5, True, "2", None, 3],
)
def test_startup_validation_rejects_malformed_explicit_versions(
    pipeline_version: object,
) -> None:
    config = _minimal_v2()
    config["pipeline_version"] = pipeline_version

    with pytest.raises(ValueError, match="pipeline_version"):
        validate_config_on_startup(config)


def test_train_validates_v2_before_runtime_setup() -> None:
    with pytest.raises(ValueError, match="p_velocity_mps"):
        train(config=_invalid_entry_config(), data_loaders=((), (), ()))


def test_evaluate_validates_v2_before_output_setup() -> None:
    with pytest.raises(ValueError, match="p_velocity_mps"):
        evaluate(
            config=_invalid_entry_config(),
            test_loader=[],
            save_plots=False,
            save_metrics=False,
        )


def test_unseen_evaluation_validates_v2_before_checkpoint_load(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump(_invalid_entry_config(), sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="p_velocity_mps"):
        evaluate_unseen_events(
            event_dirs=[],
            model_dir=model_dir,
            output_dir=tmp_path / "results",
        )


def _malformed_version_entry_config() -> dict:
    config = _minimal_v2()
    config["pipeline_version"] = True
    return config


def test_train_rejects_malformed_version_before_runtime_setup() -> None:
    with pytest.raises(ValueError, match="pipeline_version"):
        train(config=_malformed_version_entry_config(), data_loaders=((), (), ()))


def test_evaluate_rejects_malformed_version_before_output_setup() -> None:
    with pytest.raises(ValueError, match="pipeline_version"):
        evaluate(
            config=_malformed_version_entry_config(),
            test_loader=[],
            save_plots=False,
            save_metrics=False,
        )


def test_unseen_evaluation_rejects_malformed_version_before_checkpoint_load(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump(_malformed_version_entry_config(), sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pipeline_version"):
        evaluate_unseen_events(
            event_dirs=[],
            model_dir=model_dir,
            output_dir=tmp_path / "results",
        )


def test_tracked_v2_config_is_portable_and_valid() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config_v2.yaml"
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)

    assert "/Users/" not in config_text
    assert "/home/" not in config_text
    assert not Path(config["paths"]["data_path"]).is_absolute()
    assert not Path(config["dataset"]["stf"]["path"]).is_absolute()
    validate_config_v2(config)
