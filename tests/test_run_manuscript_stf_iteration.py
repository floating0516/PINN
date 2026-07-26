from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from scripts.experiments import run_manuscript_stf_iteration as campaign
from src.utils.provenance import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE23_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "manuscript_station_stf_usgs.yaml"
)
PHASE24_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "manuscript_station_stf_usgs_monotonic_cosine.yaml"
)
PHASE25_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "manuscript_station_stf_usgs_dual_range_stem.yaml"
)
PHASE26_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "manuscript_station_stf_usgs_event_balanced.yaml"
)
PHASE27_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "manuscript_station_stf_usgs_event_loss_weighted.yaml"
)
PHASE28_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "manuscript_station_stf_usgs_absolute_magnitude.yaml"
)
PHASE29_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "manuscript_station_stf_usgs_tempered_event_loss.yaml"
)
PHASE30_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "manuscript_station_stf_usgs_moment_linear_skip.yaml"
)
PHASE31_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "manuscript_station_stf_usgs_moment_head_dropout.yaml"
)
PHASE32_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "manuscript_station_stf_usgs_loss_weight_search.yaml"
)
PHASE33_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "manuscript_station_stf_usgs_no_shape.yaml"
)
PHASE34_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "manuscript_station_stf_usgs_three_loss_ablation.yaml"
)
PHASE38_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "manuscript_station_stf_usgs_polarity_robust.yaml"
)
CONFIG_PATH = PHASE23_CONFIG_PATH


def _config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _nested_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        value = value[key]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mark_stage(output_root: Path, stage: str, payload: dict[str, Any]) -> None:
    stage_dir = output_root / stage
    _write_json(stage_dir / "summary.json", {"stage": stage, **payload})
    (stage_dir / "COMPLETE").write_text("\n", encoding="utf-8")


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _phase27_sampling_manifest(seed: int) -> dict[str, Any]:
    normalization = 1788 / 31
    return {
        "schema_version": 2,
        "mode": "event_equal_inverse_count_full_data",
        "event_balanced_sampling": True,
        "event_balance_estimator": campaign.INVERSE_COUNT_FULL_DATA_ESTIMATOR,
        "sampler_class": "RandomSampler",
        "replacement": False,
        "draw_count": 1788,
        "record_count": 1788,
        "event_count": 31,
        "event_record_count_minimum": 1,
        "event_record_count_maximum": 482,
        "expected_unique_record_count": 1788.0,
        "expected_unique_record_fraction": 1.0,
        "optimizer_step_count": 28,
        "loss_weights_applied": True,
        "objective_weight_formula": "N/(E*n_event)",
        "objective_reduction": "mean(sample_weight * per_sample_loss)",
        "objective_weight_minimum": normalization / 482,
        "objective_weight_maximum": normalization,
        "event_objective_mass_minimum": normalization,
        "event_objective_mass_maximum": normalization,
        "objective_weight_sha256": (
            campaign.PHASE27_INCUMBENT_OBJECTIVE_WEIGHT_SHA256_BY_SEED[seed]
        ),
        "sample_weight_sha256": (
            campaign.PHASE27_INCUMBENT_SAMPLE_WEIGHT_SHA256_BY_SEED[seed]
        ),
        "sampling_weight_sha256": (
            campaign.PHASE27_INCUMBENT_SAMPLE_WEIGHT_SHA256_BY_SEED[seed]
        ),
    }


def _selected_train_summary(
    *,
    baseline: float,
    candidate: float,
) -> dict[str, Any]:
    def variant(name: str, value: float) -> dict[str, Any]:
        values = {
            "17": {
                "seed": 17,
                campaign.VALIDATION_METRIC: value,
            },
            "42": {
                "seed": 42,
                campaign.VALIDATION_METRIC: value + 0.1,
            },
            "73": {
                "seed": 73,
                campaign.VALIDATION_METRIC: value + 0.2,
            },
        }
        return {
            "parameterization": campaign.VARIANTS[name],
            "seeds": values,
            "selection": {
                "selected_seed": 17,
                "selection_metric": campaign.VALIDATION_METRIC,
                "ensemble_used": False,
            },
        }

    return {
        "stage": "train",
        "status": "complete",
        "variants": {
            "baseline": variant("baseline", baseline),
            "candidate": variant("candidate", candidate),
        },
    }


def _run_phase33_train_with_deltas(
    *,
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_deltas: dict[int, float],
) -> tuple[dict[str, Any], list[tuple[str, int]]]:
    config_path = output_root / "frozen_config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(PHASE33_CONFIG_PATH), sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = output_root / "dataset_manifest.csv"
    manifest_path.write_text("event,station\n", encoding="utf-8")
    splits: dict[str, Any] = {}
    for seed in campaign.SEEDS:
        split_path = output_root / f"split_{seed}.json"
        _write_json(split_path, {"seed": seed})
        splits[str(seed)] = {"manifest": _artifact(split_path)}
    _mark_stage(
        output_root,
        "preflight",
        {
            "status": "complete",
            "source_data": {"sha256": campaign.EXPECTED_SOURCE_SHA256},
            "frozen_config": _artifact(config_path),
            "dataset_manifest": _artifact(manifest_path),
            "splits": splits,
        },
    )
    _mark_stage(output_root, "smoke", {"status": "complete"})
    calls: list[tuple[str, int]] = []

    def fake_train_one_seed(**kwargs: Any) -> dict[str, Any]:
        variant = str(kwargs["variant"])
        seed = int(kwargs["seed"])
        calls.append((variant, seed))
        metric = campaign.PHASE27_INCUMBENT_VALIDATION_BY_SEED[seed]
        if variant == "candidate":
            metric += candidate_deltas[seed]
        return {
            "variant": variant,
            "seed": seed,
            "loss_weights": campaign.loss_weights_from_config(kwargs["config"]),
            campaign.VALIDATION_METRIC: metric,
            "checkpoint": {
                "sha256": (
                    campaign.PHASE27_INCUMBENT_CHECKPOINT_SHA256_BY_SEED[seed]
                    if variant == "baseline"
                    else f"candidate-{seed}"
                )
            },
        }

    monkeypatch.setattr(campaign, "_train_one_seed", fake_train_one_seed)
    return campaign.run_train(output_root=output_root, resume=False), calls


def _complete_phase33_internal(
    *,
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], list[tuple[int, bool]]]:
    candidate_deltas = {
        17: 0.0,
        42: 0.0,
        73: 0.10 - campaign.PHASE27_INCUMBENT_VALIDATION_BY_SEED[73],
    }
    train_summary, _calls = _run_phase33_train_with_deltas(
        output_root=output_root,
        monkeypatch=monkeypatch,
        candidate_deltas=candidate_deltas,
    )
    assert train_summary["shape_loss_validation_compatibility"]["passed"] is True
    calls: list[tuple[int, bool]] = []

    def fake_locked_test(**kwargs: Any) -> dict[str, Any]:
        seed = int(kwargs["seed_summary"]["seed"])
        delayed = bool(kwargs["include_delayed_prefix"])
        calls.append((seed, delayed))
        output_dir = Path(kwargs["output_dir"])
        metrics = {
            "station_mae": 0.10,
            "station_rmse": 0.12,
            "station_bias": -0.01,
            "event_mae": 0.14 if delayed else 0.137,
            "event_rmse": 0.16,
            "event_bias": -0.02,
        }
        metrics_path = output_dir / "metrics.json"
        _write_json(metrics_path, metrics)
        station_path = output_dir / "station_predictions.csv"
        station_path.parent.mkdir(parents=True, exist_ok=True)
        station_path.write_text("event,station\n", encoding="utf-8")
        event_path = output_dir / "event_predictions.csv"
        event_path.write_text("event,mw_pred\n", encoding="utf-8")
        registry_path = output_dir / "result_registry.json"
        _write_json(registry_path, {"seed": seed})
        result: dict[str, Any] = {
            "selected_seed": seed,
            "metrics": metrics,
            "artifacts": {
                "metrics": _artifact(metrics_path),
                "station_predictions": _artifact(station_path),
                "event_predictions": _artifact(event_path),
                "result_registry": _artifact(registry_path),
            },
        }
        if delayed:
            delayed_dir = output_dir / "delayed_prefix"
            delayed_station_path = delayed_dir / "station_predictions.csv"
            delayed_station_path.parent.mkdir(parents=True, exist_ok=True)
            delayed_station_path.write_text("event,station\n", encoding="utf-8")
            delayed_event_path = delayed_dir / "event_predictions.csv"
            delayed_event_path.write_text("event,mw_pred\n", encoding="utf-8")
            unavailable_path = delayed_dir / "unavailable_stations.csv"
            unavailable_path.write_text("event,station\n", encoding="utf-8")
            horizon_metrics = [
                {
                    "observation_horizon_sec": float(horizon),
                    "event_equal_mae": 0.14,
                }
                for horizon in campaign.DEFAULT_HORIZONS_SEC
            ]
            horizon_path = delayed_dir / "horizon_metrics.json"
            _write_json(horizon_path, horizon_metrics)
            cohort = {
                "waveform_prefix_causal": True,
                "station_selection_causal": False,
            }
            cohort_path = delayed_dir / "cohort_contract.json"
            _write_json(cohort_path, cohort)
            result["delayed_prefix"] = {
                "horizons_sec": list(campaign.DEFAULT_HORIZONS_SEC),
                "horizon_metrics": horizon_metrics,
                "cohort": cohort,
                "artifacts": {
                    "station_predictions": _artifact(delayed_station_path),
                    "event_predictions": _artifact(delayed_event_path),
                    "unavailable_stations": _artifact(unavailable_path),
                    "horizon_metrics": _artifact(horizon_path),
                    "cohort_contract": _artifact(cohort_path),
                },
            }
        return result

    monkeypatch.setattr(campaign, "_evaluate_locked_test", fake_locked_test)
    return campaign.run_internal(output_root=output_root, resume=False), calls


@pytest.mark.parametrize(
    ("config_path", "expected_axis", "baseline_value", "candidate_value"),
    [
        (
            PHASE23_CONFIG_PATH,
            campaign.STF_OUTPUT_PARAMETERIZATION_AXIS,
            "direct",
            "moment_shape_factorized",
        ),
        (
            PHASE24_CONFIG_PATH,
            campaign.SCHEDULER_T0_AXIS,
            15,
            195,
        ),
        (
            PHASE25_CONFIG_PATH,
            campaign.RADIAL_DYNAMIC_RANGE_STEM_AXIS,
            "none",
            "asinh_residual",
        ),
        (
            PHASE26_CONFIG_PATH,
            campaign.EVENT_BALANCED_SAMPLING_AXIS,
            False,
            True,
        ),
        (
            PHASE27_CONFIG_PATH,
            campaign.EVENT_BALANCE_ESTIMATOR_AXIS,
            campaign.REPLACEMENT_SAMPLING_ESTIMATOR,
            campaign.INVERSE_COUNT_FULL_DATA_ESTIMATOR,
        ),
        (
            PHASE28_CONFIG_PATH,
            campaign.MAGNITUDE_PENALTY_AXIS,
            "squared",
            "absolute",
        ),
        (
            PHASE29_CONFIG_PATH,
            campaign.EVENT_BALANCE_EXPONENT_AXIS,
            1.0,
            0.5,
        ),
        (
            PHASE30_CONFIG_PATH,
            campaign.MOMENT_LINEAR_SKIP_AXIS,
            False,
            True,
        ),
        (
            PHASE31_CONFIG_PATH,
            campaign.MOMENT_HEAD_DROPOUT_AXIS,
            True,
            False,
        ),
        (
            PHASE33_CONFIG_PATH,
            campaign.SHAPE_LOSS_WEIGHT_AXIS,
            0.1,
            0.0,
        ),
    ],
)
def test_formal_configs_and_variants_have_one_scientific_difference(
    config_path: Path,
    expected_axis: str,
    baseline_value: Any,
    candidate_value: Any,
) -> None:
    config = _config(config_path)

    campaign.validate_formal_config(config)
    variants = campaign.build_variant_configs(config)

    assert campaign.variant_axis_from_config(config) == expected_axis
    path = campaign.VARIANT_AXIS_PATHS[expected_axis]
    assert _nested_value(variants["baseline"], path) == baseline_value
    assert _nested_value(variants["candidate"], path) == candidate_value
    assert campaign._config_diff_paths(
        variants["baseline"],
        variants["candidate"],
    ) == {".".join(campaign.VARIANT_AXIS_PATHS[expected_axis])}
    if expected_axis in {
        campaign.SCHEDULER_T0_AXIS,
        campaign.RADIAL_DYNAMIC_RANGE_STEM_AXIS,
        campaign.EVENT_BALANCED_SAMPLING_AXIS,
        campaign.EVENT_BALANCE_ESTIMATOR_AXIS,
        campaign.MAGNITUDE_PENALTY_AXIS,
        campaign.EVENT_BALANCE_EXPONENT_AXIS,
        campaign.MOMENT_LINEAR_SKIP_AXIS,
        campaign.MOMENT_HEAD_DROPOUT_AXIS,
        campaign.SHAPE_LOSS_WEIGHT_AXIS,
    }:
        assert {
            variant["model"]["stf_output_parameterization"]
            for variant in variants.values()
        } == {"moment_shape_factorized"}
        assert {
            variant["training"]["scheduler_T_mult"]
            for variant in variants.values()
        } == {2}
    if expected_axis in {
        campaign.RADIAL_DYNAMIC_RANGE_STEM_AXIS,
        campaign.EVENT_BALANCED_SAMPLING_AXIS,
        campaign.EVENT_BALANCE_ESTIMATOR_AXIS,
        campaign.MAGNITUDE_PENALTY_AXIS,
        campaign.EVENT_BALANCE_EXPONENT_AXIS,
        campaign.MOMENT_LINEAR_SKIP_AXIS,
        campaign.MOMENT_HEAD_DROPOUT_AXIS,
        campaign.SHAPE_LOSS_WEIGHT_AXIS,
    }:
        assert {
            variant["training"]["scheduler_T0"] for variant in variants.values()
        } == {15}


@pytest.mark.parametrize(
    ("config_path", "expected_axis"),
    [
        (PHASE23_CONFIG_PATH, campaign.STF_OUTPUT_PARAMETERIZATION_AXIS),
        (PHASE24_CONFIG_PATH, campaign.SCHEDULER_T0_AXIS),
        (PHASE25_CONFIG_PATH, campaign.RADIAL_DYNAMIC_RANGE_STEM_AXIS),
        (PHASE26_CONFIG_PATH, campaign.EVENT_BALANCED_SAMPLING_AXIS),
        (PHASE27_CONFIG_PATH, campaign.EVENT_BALANCE_ESTIMATOR_AXIS),
        (PHASE28_CONFIG_PATH, campaign.MAGNITUDE_PENALTY_AXIS),
        (PHASE29_CONFIG_PATH, campaign.EVENT_BALANCE_EXPONENT_AXIS),
        (PHASE30_CONFIG_PATH, campaign.MOMENT_LINEAR_SKIP_AXIS),
        (PHASE31_CONFIG_PATH, campaign.MOMENT_HEAD_DROPOUT_AXIS),
        (PHASE33_CONFIG_PATH, campaign.SHAPE_LOSS_WEIGHT_AXIS),
    ],
)
def test_formal_config_rejects_a_second_scientific_change(
    config_path: Path,
    expected_axis: str,
) -> None:
    variants = campaign.build_variant_configs(_config(config_path))
    changed = copy.deepcopy(variants["candidate"])
    changed["model"]["dropout"] = 0.1

    assert campaign._config_diff_paths(variants["baseline"], changed) == {
        "model.dropout",
        ".".join(campaign.VARIANT_AXIS_PATHS[expected_axis]),
    }


@pytest.mark.parametrize(
    "config_path",
    [
        PHASE26_CONFIG_PATH,
        PHASE27_CONFIG_PATH,
        PHASE28_CONFIG_PATH,
        PHASE29_CONFIG_PATH,
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_prior_formal_axes_reject_a_moment_skip_marker(
    config_path: Path,
    enabled: bool,
) -> None:
    config = _config(config_path)
    config["model"]["moment_linear_skip"] = enabled

    with pytest.raises(ValueError):
        campaign.validate_formal_config(config)


@pytest.mark.parametrize(
    "config_path",
    [
        PHASE23_CONFIG_PATH,
        PHASE24_CONFIG_PATH,
        PHASE25_CONFIG_PATH,
        PHASE26_CONFIG_PATH,
        PHASE27_CONFIG_PATH,
        PHASE28_CONFIG_PATH,
        PHASE29_CONFIG_PATH,
        PHASE30_CONFIG_PATH,
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_prior_formal_axes_reject_a_moment_head_dropout_marker(
    config_path: Path,
    enabled: bool,
) -> None:
    config = _config(config_path)
    config["model"]["moment_head_dropout"] = enabled

    with pytest.raises(ValueError):
        campaign.validate_formal_config(config)


def test_phase24_axis_requires_scheduler_t0_15_baseline() -> None:
    config = _config(PHASE24_CONFIG_PATH)
    config["training"]["scheduler_T0"] = 195

    with pytest.raises(ValueError, match="Phase24 factorized/T0=15 baseline"):
        campaign.validate_formal_config(config)


def test_phase25_axis_uses_an_explicit_none_baseline_marker() -> None:
    config = _config(PHASE25_CONFIG_PATH)

    assert campaign._config_diff_paths(
        _config(PHASE24_CONFIG_PATH),
        config,
    ) == {"model.radial_dynamic_range_stem"}
    assert (
        campaign.variant_axis_from_config(config)
        == campaign.RADIAL_DYNAMIC_RANGE_STEM_AXIS
    )

    without_marker = copy.deepcopy(config)
    del without_marker["model"]["radial_dynamic_range_stem"]
    assert (
        campaign.variant_axis_from_config(without_marker)
        == campaign.SCHEDULER_T0_AXIS
    )

    candidate_as_baseline = copy.deepcopy(config)
    candidate_as_baseline["model"]["radial_dynamic_range_stem"] = "asinh_residual"
    with pytest.raises(ValueError, match="Phase25 formal baseline requires explicit"):
        campaign.validate_formal_config(candidate_as_baseline)

    phase23_with_marker = _config(PHASE23_CONFIG_PATH)
    phase23_with_marker["model"]["radial_dynamic_range_stem"] = "none"
    with pytest.raises(ValueError, match="formal config must describe"):
        campaign.validate_formal_config(phase23_with_marker)


def test_phase26_axis_requires_an_explicit_campaign_marker() -> None:
    config = _config(PHASE26_CONFIG_PATH)

    assert (
        campaign.variant_axis_from_config(config)
        == campaign.EVENT_BALANCED_SAMPLING_AXIS
    )
    assert "radial_dynamic_range_stem" not in config["model"]

    without_marker = copy.deepcopy(config)
    del without_marker["campaign"]
    assert (
        campaign.variant_axis_from_config(without_marker)
        == campaign.SCHEDULER_T0_AXIS
    )

    candidate_as_baseline = copy.deepcopy(config)
    candidate_as_baseline["training"]["event_balanced_sampling"] = True
    with pytest.raises(ValueError, match="event_balanced_sampling=False"):
        campaign.validate_formal_config(candidate_as_baseline)

    unknown_axis = copy.deepcopy(config)
    unknown_axis["campaign"]["variant_axis"] = "unknown"
    with pytest.raises(ValueError, match="unsupported formal campaign axis"):
        campaign.validate_formal_config(unknown_axis)


def test_phase27_axis_requires_the_phase26_candidate_as_baseline() -> None:
    config = _config(PHASE27_CONFIG_PATH)

    assert (
        campaign.variant_axis_from_config(config)
        == campaign.EVENT_BALANCE_ESTIMATOR_AXIS
    )
    assert config["training"]["event_balanced_sampling"] is True
    assert (
        config["training"]["event_balance_estimator"]
        == campaign.REPLACEMENT_SAMPLING_ESTIMATOR
    )
    phase26_incumbent = campaign.build_variant_configs(
        _config(PHASE26_CONFIG_PATH)
    )["candidate"]
    phase27_baseline = campaign.build_variant_configs(config)["baseline"]
    phase26_incumbent.pop("campaign")
    phase27_baseline.pop("campaign")
    phase27_baseline["training"].pop("event_balance_estimator")
    assert phase27_baseline == phase26_incumbent

    candidate_as_baseline = copy.deepcopy(config)
    candidate_as_baseline["training"]["event_balance_estimator"] = (
        campaign.INVERSE_COUNT_FULL_DATA_ESTIMATOR
    )
    with pytest.raises(ValueError, match="Phase27 full-data objective baseline"):
        campaign.validate_formal_config(candidate_as_baseline)

    disabled = copy.deepcopy(config)
    disabled["training"]["event_balanced_sampling"] = False
    with pytest.raises(ValueError, match="Phase27 full-data objective baseline"):
        campaign.validate_formal_config(disabled)


def test_phase28_axis_requires_the_phase27_candidate_as_baseline() -> None:
    config = _config(PHASE28_CONFIG_PATH)

    assert (
        campaign.variant_axis_from_config(config)
        == campaign.MAGNITUDE_PENALTY_AXIS
    )
    phase27_incumbent = campaign.build_variant_configs(
        _config(PHASE27_CONFIG_PATH)
    )["candidate"]
    phase28_baseline = campaign.build_variant_configs(config)["baseline"]
    phase27_incumbent.pop("campaign")
    phase28_baseline.pop("campaign")
    phase28_baseline["training"]["stf_rate_loss"].pop(
        "magnitude_penalty"
    )
    assert phase28_baseline == phase27_incumbent

    absolute_as_baseline = copy.deepcopy(config)
    absolute_as_baseline["training"]["stf_rate_loss"][
        "magnitude_penalty"
    ] = "absolute"
    with pytest.raises(ValueError, match="Phase28 magnitude-penalty baseline"):
        campaign.validate_formal_config(absolute_as_baseline)

    replacement_estimator = copy.deepcopy(config)
    replacement_estimator["training"]["event_balance_estimator"] = (
        campaign.REPLACEMENT_SAMPLING_ESTIMATOR
    )
    with pytest.raises(ValueError, match="Phase28 magnitude-penalty baseline"):
        campaign.validate_formal_config(replacement_estimator)


def test_phase29_axis_is_exactly_the_phase27_incumbent_plus_exponent() -> None:
    config = _config(PHASE29_CONFIG_PATH)

    assert (
        campaign.variant_axis_from_config(config)
        == campaign.EVENT_BALANCE_EXPONENT_AXIS
    )
    variants = campaign.build_variant_configs(config)
    assert variants["baseline"]["training"]["event_balance_exponent"] == 1.0
    assert variants["candidate"]["training"]["event_balance_exponent"] == 0.5
    assert "magnitude_penalty" not in config["training"]["stf_rate_loss"]

    phase27_incumbent = campaign.build_variant_configs(
        _config(PHASE27_CONFIG_PATH)
    )["candidate"]
    phase29_baseline = copy.deepcopy(variants["baseline"])
    phase27_incumbent.pop("campaign")
    phase29_baseline.pop("campaign")
    phase29_baseline["training"].pop("event_balance_exponent")
    assert phase29_baseline == phase27_incumbent

    tempered_as_baseline = copy.deepcopy(config)
    tempered_as_baseline["training"]["event_balance_exponent"] = 0.5
    with pytest.raises(ValueError, match="Phase29 tempered inverse-count baseline"):
        campaign.validate_formal_config(tempered_as_baseline)

    explicit_magnitude_penalty = copy.deepcopy(config)
    explicit_magnitude_penalty["training"]["stf_rate_loss"][
        "magnitude_penalty"
    ] = "squared"
    with pytest.raises(ValueError, match="Phase29 tempered inverse-count baseline"):
        campaign.validate_formal_config(explicit_magnitude_penalty)

    changed_learning_rate = copy.deepcopy(config)
    changed_learning_rate["training"]["learning_rate"] = 2.0e-4
    with pytest.raises(ValueError, match="differs from the frozen Phase27"):
        campaign.validate_formal_config(changed_learning_rate)


def test_phase30_axis_is_exactly_the_phase27_incumbent_plus_skip_marker() -> None:
    config = _config(PHASE30_CONFIG_PATH)

    assert (
        campaign.variant_axis_from_config(config)
        == campaign.MOMENT_LINEAR_SKIP_AXIS
    )
    variants = campaign.build_variant_configs(config)
    assert variants["baseline"]["model"]["moment_linear_skip"] is False
    assert variants["candidate"]["model"]["moment_linear_skip"] is True
    assert "event_balance_exponent" not in config["training"]
    assert "magnitude_penalty" not in config["training"]["stf_rate_loss"]
    assert "radial_dynamic_range_stem" not in config["model"]

    phase27_incumbent = campaign.build_variant_configs(
        _config(PHASE27_CONFIG_PATH)
    )["candidate"]
    phase30_baseline = copy.deepcopy(variants["baseline"])
    phase27_incumbent.pop("campaign")
    phase30_baseline.pop("campaign")
    phase30_baseline["model"].pop("moment_linear_skip")
    assert phase30_baseline == phase27_incumbent

    enabled_as_baseline = copy.deepcopy(config)
    enabled_as_baseline["model"]["moment_linear_skip"] = True
    with pytest.raises(ValueError, match="Phase30 moment-linear-skip baseline"):
        campaign.validate_formal_config(enabled_as_baseline)

    explicit_exponent = copy.deepcopy(config)
    explicit_exponent["training"]["event_balance_exponent"] = 1.0
    with pytest.raises(ValueError, match="Phase30 moment-linear-skip baseline"):
        campaign.validate_formal_config(explicit_exponent)

    changed_learning_rate = copy.deepcopy(config)
    changed_learning_rate["training"]["learning_rate"] = 2.0e-4
    with pytest.raises(ValueError, match="differs from the frozen Phase27"):
        campaign.validate_formal_config(changed_learning_rate)


def test_phase31_axis_is_exactly_the_phase27_incumbent_plus_dropout_marker() -> None:
    config = _config(PHASE31_CONFIG_PATH)

    assert (
        campaign.variant_axis_from_config(config)
        == campaign.MOMENT_HEAD_DROPOUT_AXIS
    )
    variants = campaign.build_variant_configs(config)
    assert variants["baseline"]["model"]["moment_head_dropout"] is True
    assert variants["candidate"]["model"]["moment_head_dropout"] is False
    assert "moment_linear_skip" not in config["model"]
    assert "event_balance_exponent" not in config["training"]
    assert "magnitude_penalty" not in config["training"]["stf_rate_loss"]
    assert "radial_dynamic_range_stem" not in config["model"]

    phase27_incumbent = campaign.build_variant_configs(
        _config(PHASE27_CONFIG_PATH)
    )["candidate"]
    phase31_baseline = copy.deepcopy(variants["baseline"])
    phase27_incumbent.pop("campaign")
    phase31_baseline.pop("campaign")
    phase31_baseline["model"].pop("moment_head_dropout")
    assert phase31_baseline == phase27_incumbent

    disabled_as_baseline = copy.deepcopy(config)
    disabled_as_baseline["model"]["moment_head_dropout"] = False
    with pytest.raises(ValueError, match="Phase31 moment-head-dropout baseline"):
        campaign.validate_formal_config(disabled_as_baseline)

    explicit_skip = copy.deepcopy(config)
    explicit_skip["model"]["moment_linear_skip"] = False
    with pytest.raises(ValueError, match="Phase31 moment-head-dropout baseline"):
        campaign.validate_formal_config(explicit_skip)

    changed_learning_rate = copy.deepcopy(config)
    changed_learning_rate["training"]["learning_rate"] = 2.0e-4
    with pytest.raises(ValueError, match="differs from the frozen Phase27"):
        campaign.validate_formal_config(changed_learning_rate)


def test_phase32_profiles_and_hash_are_frozen_on_phase27_incumbent() -> None:
    config = _config(PHASE32_CONFIG_PATH)

    campaign.validate_formal_config(config)
    assert campaign.variant_axis_from_config(config) == campaign.LOSS_WEIGHT_PROFILE_AXIS
    profiles = campaign.loss_weight_profiles_from_config(config)
    assert profiles == campaign.LOSS_WEIGHT_PROFILES
    assert campaign._candidate_profile_sha256(profiles) == (
        "2235d225dd480f675ab3bdd4b044fae420f4420a9a0f40c26fc70bfd9399df64"
    )
    variants = campaign.build_variant_configs(config)
    assert list(variants) == ["baseline", "w01", "w10"]
    expected_paths = {
        ".".join(campaign.LOSS_WEIGHT_PATHS[name])
        for name in campaign.LOSS_WEIGHT_KEYS
    }
    for profile_id in ("w01", "w10"):
        assert campaign._config_diff_paths(
            variants["baseline"], variants[profile_id]
        ) == expected_paths
        assert campaign.variant_axis_from_config(variants[profile_id]) == (
            campaign.LOSS_WEIGHT_PROFILE_AXIS
        )

    phase27_incumbent = campaign.build_variant_configs(
        _config(PHASE27_CONFIG_PATH)
    )["candidate"]
    phase32_baseline = copy.deepcopy(variants["baseline"])
    phase27_incumbent.pop("campaign")
    phase32_baseline.pop("campaign")
    assert phase32_baseline == phase27_incumbent


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_phase32_rejects_nonpositive_or_nonfinite_profile_weights(
    invalid: float,
) -> None:
    config = _config(PHASE32_CONFIG_PATH)
    config["campaign"]["loss_weight_search"]["profiles"]["w01"][
        "lambda_shape"
    ] = invalid

    with pytest.raises(ValueError, match="finite and positive"):
        campaign.loss_weight_profiles_from_config(config)


def test_phase32_rejects_profile_table_or_hash_tampering() -> None:
    changed_hash = _config(PHASE32_CONFIG_PATH)
    changed_hash["campaign"]["loss_weight_search"][
        "candidate_profile_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="candidate_profile_sha256 changed"):
        campaign.loss_weight_profiles_from_config(changed_hash)

    changed_table = _config(PHASE32_CONFIG_PATH)
    changed_table["campaign"]["loss_weight_search"]["profiles"]["w10"][
        "lambda_mag"
    ] += 1.0e-9
    with pytest.raises(ValueError, match="w10.lambda_mag changed"):
        campaign.loss_weight_profiles_from_config(changed_table)


def test_phase32_profile_selection_uses_paired_median_and_profile_id_tie_break() -> None:
    baseline = {17: 0.10, 42: 0.20, 73: 0.30}
    deltas = {
        "w01": {17: -0.02, 42: -0.01, 73: 0.02},
        "w10": {17: -0.03, 42: -0.01, 73: 0.03},
    }
    variants: dict[str, Any] = {
        "baseline": {
            "seeds": {
                str(seed): {campaign.VALIDATION_METRIC: value}
                for seed, value in baseline.items()
            }
        }
    }
    for profile_id in ("w01", "w10"):
        variants[profile_id] = {
            "seeds": {
                str(seed): {
                    campaign.VALIDATION_METRIC: baseline[seed] + deltas[profile_id][seed]
                }
                for seed in campaign.SEEDS
            }
        }

    selection = campaign.select_loss_weight_profile(variants)

    assert selection["winner_profile"] == "w01"
    assert selection["winner_paired_median_delta"] == pytest.approx(-0.01)
    assert selection["winner_paired_mean_delta"] < 0.0
    assert selection["winner_improved_seed_count"] == 2
    assert selection["winner_improved_seed_majority"] is True


def test_phase33_is_exactly_phase27_with_only_shape_loss_removed() -> None:
    config = _config(PHASE33_CONFIG_PATH)

    campaign.validate_formal_config(config)
    assert (
        campaign.variant_axis_from_config(config)
        == campaign.SHAPE_LOSS_WEIGHT_AXIS
    )
    variants = campaign.build_variant_configs(config)
    assert list(variants) == ["baseline", "candidate"]
    assert campaign.loss_weights_from_config(variants["baseline"]) == {
        "lambda_MSE": 1.0,
        "lambda_synth": 0.5,
        "lambda_mag": 1.0,
        "lambda_shape": 0.1,
    }
    assert campaign.loss_weights_from_config(variants["candidate"]) == {
        "lambda_MSE": 1.0,
        "lambda_synth": 0.5,
        "lambda_mag": 1.0,
        "lambda_shape": 0.0,
    }
    assert campaign._config_diff_paths(
        variants["baseline"], variants["candidate"]
    ) == {"training.stf_rate_loss.lambda_shape"}

    phase27_incumbent = campaign.build_variant_configs(
        _config(PHASE27_CONFIG_PATH)
    )["candidate"]
    phase33_baseline = copy.deepcopy(variants["baseline"])
    phase27_incumbent.pop("campaign")
    phase33_baseline.pop("campaign")
    assert phase33_baseline == phase27_incumbent


def test_phase33_zero_shape_weight_is_a_valid_loss_configuration() -> None:
    candidate = campaign.build_variant_configs(_config(PHASE33_CONFIG_PATH))[
        "candidate"
    ]

    assert campaign.loss_weights_from_config(candidate)["lambda_shape"] == 0.0


@pytest.mark.parametrize("weight_name", campaign.LOSS_WEIGHT_KEYS)
@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_loss_weights_reject_negative_or_nonfinite_values(
    weight_name: str,
    invalid: float,
) -> None:
    config = _config(PHASE33_CONFIG_PATH)
    config["training"]["stf_rate_loss"][weight_name] = invalid

    with pytest.raises(ValueError):
        campaign.loss_weights_from_config(config)


def test_phase33_train_uses_three_seed_pairs_and_accepts_exact_ties(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary, calls = _run_phase33_train_with_deltas(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
        candidate_deltas={seed: 0.0 for seed in campaign.SEEDS},
    )

    assert calls == [
        (variant, seed)
        for variant in ("baseline", "candidate")
        for seed in campaign.SEEDS
    ]
    assert summary["incumbent_reproduction"]["passed"] is True
    compatibility = summary["shape_loss_validation_compatibility"]
    assert compatibility["passed"] is True
    assert compatibility["paired_deltas_by_seed"] == {
        str(seed): 0.0 for seed in campaign.SEEDS
    }
    assert compatibility["paired_mean_delta"] == 0.0
    assert compatibility["paired_median_delta"] == 0.0
    assert compatibility["improved_or_equal_seed_count"] == 3
    assert compatibility["candidate_selected_seed"] == 17
    assert compatibility["candidate_selected_validation"] == (
        campaign.PHASE27_INCUMBENT_VALIDATION
    )
    assert all(compatibility["conditions"].values())
    compatibility_path = campaign._validate_artifact(
        compatibility["artifact"],
        label="Phase33 compatibility",
    )
    persisted = json.loads(compatibility_path.read_text(encoding="utf-8"))
    assert persisted == {
        key: value for key, value in compatibility.items() if key != "artifact"
    }


def test_split_contract_is_frozen_for_all_three_seeds() -> None:
    for seed in campaign.SEEDS:
        manifest = {
            "seed": seed,
            "protocol": "within_event_station",
            "train_record_count": 1788,
            "validation_record_count": 385,
            "test_record_count": 385,
            "assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[seed],
        }
        campaign._assert_split_manifest(manifest, seed=seed)

    changed = dict(manifest)
    changed["test_record_count"] = 384
    with pytest.raises(ValueError, match="split counts changed"):
        campaign._assert_split_manifest(changed, seed=73)


def test_preflight_rejects_dirty_git_before_loading_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "git_is_dirty", lambda _root: True)
    monkeypatch.setattr(
        campaign,
        "CorrectedEarthquakeDataset",
        lambda _config: pytest.fail("dirty preflight must not load the dataset"),
    )

    with pytest.raises(ValueError, match="clean Git worktree"):
        campaign.run_preflight(
            config_path=CONFIG_PATH,
            output_root=tmp_path,
            resume=False,
        )


def test_validation_log_replays_checkpoint_minimum_delta() -> None:
    rows = [
        {"Epoch": "1", campaign.VALIDATION_METRIC: "0.30"},
        {"Epoch": "2", campaign.VALIDATION_METRIC: "0.29995"},
        {"Epoch": "3", campaign.VALIDATION_METRIC: "0.28"},
    ]

    selected = campaign.select_checkpoint_from_validation_log(
        rows,
        minimum_delta=0.0001,
    )

    assert selected == {
        "epoch": 3,
        campaign.VALIDATION_METRIC: 0.28,
    }


def test_phase23_legacy_seed_summary_without_new_axis_fields_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_root = tmp_path / "seed_17"
    references: dict[str, dict[str, str]] = {}
    for name in ("checkpoint", "config", "split", "training_log", "run_manifest"):
        artifact_path = tmp_path / f"{name}.artifact"
        artifact_path.write_text(name, encoding="utf-8")
        references[name] = _artifact(artifact_path)
    legacy_summary = {
        "variant": "baseline",
        "parameterization": "direct",
        "seed": 17,
        campaign.VALIDATION_METRIC: 0.2,
        **references,
    }
    _write_json(seed_root / "seed_summary.json", legacy_summary)
    monkeypatch.setattr(
        campaign,
        "_runtime_config",
        lambda *_args, **_kwargs: pytest.fail(
            "a valid legacy seed summary must resume without retraining"
        ),
    )

    resumed = campaign._train_one_seed(
        variant="baseline",
        config=_config(PHASE23_CONFIG_PATH),
        seed=17,
        seed_root=seed_root,
        dataset_manifest=tmp_path / "unused_manifest.csv",
        frozen_split_path=tmp_path / "unused_split.json",
        resume=True,
    )

    assert "variant_axis" not in resumed
    assert "scheduler_T0" not in resumed
    assert "radial_dynamic_range_stem" not in resumed
    assert resumed == legacy_summary


def test_phase26_seed_summary_requires_sampling_provenance(tmp_path: Path) -> None:
    references: dict[str, dict[str, str]] = {}
    for name in ("checkpoint", "config", "split", "training_log", "run_manifest"):
        artifact_path = tmp_path / f"{name}.artifact"
        artifact_path.write_text(name, encoding="utf-8")
        references[name] = _artifact(artifact_path)
    summary = {
        "seed": 17,
        campaign.VALIDATION_METRIC: 0.2,
        **references,
    }

    assert campaign._seed_summary_is_valid(summary)
    assert not campaign._seed_summary_is_valid(summary, require_sampling=True)

    sampling_path = tmp_path / "sampling.json"
    sampling_path.write_text("{}\n", encoding="utf-8")
    summary["sampling"] = _artifact(sampling_path)
    assert campaign._seed_summary_is_valid(summary, require_sampling=True)
    assert not campaign._seed_summary_is_valid(
        summary,
        require_sampling=True,
        expected_magnitude_penalty="squared",
    )
    summary["magnitude_penalty"] = "squared"
    assert campaign._seed_summary_is_valid(
        summary,
        require_sampling=True,
        expected_magnitude_penalty="squared",
    )


def test_phase28_seed_summary_resume_is_bound_to_full_context(tmp_path: Path) -> None:
    references: dict[str, dict[str, str]] = {}
    for name in (
        "checkpoint",
        "training_log",
        "run_manifest",
        "sampling",
    ):
        artifact_path = tmp_path / f"{name}.artifact"
        artifact_path.write_text(name, encoding="utf-8")
        references[name] = _artifact(artifact_path)
    split_path = tmp_path / "split.json"
    _write_json(
        split_path,
        {
            "seed": 17,
            "protocol": "within_event_station",
            "train_record_count": 1788,
            "validation_record_count": 385,
            "test_record_count": 385,
            "assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[17],
        },
    )
    references["split"] = _artifact(split_path)
    expected_config = {
        "training": {
            "random_seed": 17,
            "event_balance_estimator": "inverse_count_full_data",
            "stf_rate_loss": {"magnitude_penalty": "squared"},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(expected_config, sort_keys=False),
        encoding="utf-8",
    )
    summary = {
        "variant": "baseline",
        "seed": 17,
        "magnitude_penalty": "squared",
        "split_assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[17],
        campaign.VALIDATION_METRIC: 0.1,
        "config": _artifact(config_path),
        **references,
    }
    arguments = {
        "require_sampling": True,
        "expected_magnitude_penalty": "squared",
        "expected_variant": "baseline",
        "expected_seed": 17,
        "expected_split_assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[17],
        "expected_config": expected_config,
    }

    assert campaign._seed_summary_is_valid(summary, **arguments)
    assert not campaign._seed_summary_is_valid(
        {**summary, "variant": "candidate"},
        **arguments,
    )
    assert not campaign._seed_summary_is_valid(
        {**summary, "seed": 42},
        **arguments,
    )
    assert not campaign._seed_summary_is_valid(
        {**summary, "split_assignment_sha256": "wrong"},
        **arguments,
    )
    wrong_split_path = tmp_path / "wrong_split.json"
    _write_json(
        wrong_split_path,
        {
            "seed": 17,
            "protocol": "within_event_station",
            "train_record_count": 1788,
            "validation_record_count": 385,
            "test_record_count": 385,
            "assignment_sha256": "wrong",
        },
    )
    assert not campaign._seed_summary_is_valid(
        {**summary, "split": _artifact(wrong_split_path)},
        **arguments,
    )
    changed_config = copy.deepcopy(expected_config)
    changed_config["training"]["event_balance_estimator"] = "replacement_sampling"
    assert not campaign._seed_summary_is_valid(
        summary,
        **{**arguments, "expected_config": changed_config},
    )


def test_phase29_seed_summary_resume_rejects_exponent_or_commit_mismatch(
    tmp_path: Path,
) -> None:
    expected_git_commit = "phase29-commit"
    references: dict[str, dict[str, str]] = {}
    for name in ("checkpoint", "training_log"):
        artifact_path = tmp_path / f"{name}.artifact"
        artifact_path.write_text(name, encoding="utf-8")
        references[name] = _artifact(artifact_path)
    run_manifest_path = tmp_path / "run_manifest.json"
    _write_json(
        run_manifest_path,
        {"git_commit": expected_git_commit, "git_dirty": False},
    )
    references["run_manifest"] = _artifact(run_manifest_path)
    split_path = tmp_path / "split.json"
    _write_json(
        split_path,
        {
            "seed": 17,
            "protocol": "within_event_station",
            "train_record_count": 1788,
            "validation_record_count": 385,
            "test_record_count": 385,
            "assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[17],
        },
    )
    references["split"] = _artifact(split_path)
    normalization = 1788 / 31
    sampling_manifest = {
        "schema_version": 3,
        "mode": "event_equal_inverse_count_full_data",
        "event_balance_estimator": campaign.INVERSE_COUNT_FULL_DATA_ESTIMATOR,
        "event_balance_exponent": 1.0,
        "record_count": 1788,
        "event_count": 31,
        "event_record_count_minimum": 1,
        "event_record_count_maximum": 482,
        "objective_normalization_constant": normalization,
        "objective_weight_formula": "N/(E*n_event)",
        "objective_event_mass_formula": "N/E",
        "equal_event_objective_mass": True,
        "objective_weight_minimum": normalization / 482,
        "objective_weight_maximum": normalization,
        "event_objective_mass_minimum": normalization,
        "event_objective_mass_maximum": normalization,
        "event_objective_mass_ratio": 1.0,
        "event_objective_mass_ess": 31.0,
    }
    sampling_path = tmp_path / "sampling.json"
    _write_json(sampling_path, sampling_manifest)
    references["sampling"] = _artifact(sampling_path)
    expected_config = {
        "training": {
            "random_seed": 17,
            "event_balance_estimator": "inverse_count_full_data",
            "event_balance_exponent": 1.0,
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(expected_config, sort_keys=False),
        encoding="utf-8",
    )
    summary = {
        "variant": "baseline",
        "seed": 17,
        "git_commit": expected_git_commit,
        "event_balance_exponent": 1.0,
        "split_assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[17],
        campaign.VALIDATION_METRIC: 0.1,
        "config": _artifact(config_path),
        **references,
    }
    arguments = {
        "require_sampling": True,
        "expected_event_balance_exponent": 1.0,
        "expected_variant": "baseline",
        "expected_seed": 17,
        "expected_split_assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[17],
        "expected_config": expected_config,
        "expected_git_commit": expected_git_commit,
    }

    assert campaign._seed_summary_is_valid(summary, **arguments)
    assert not campaign._seed_summary_is_valid(
        {**summary, "event_balance_exponent": 0.5},
        **arguments,
    )
    assert not campaign._seed_summary_is_valid(
        summary,
        **{**arguments, "expected_event_balance_exponent": 0.5},
    )

    mismatched_sampling = dict(sampling_manifest)
    mismatched_sampling["event_balance_exponent"] = 0.5
    mismatched_sampling_path = tmp_path / "sampling_mismatch.json"
    _write_json(mismatched_sampling_path, mismatched_sampling)
    assert not campaign._seed_summary_is_valid(
        {**summary, "sampling": _artifact(mismatched_sampling_path)},
        **arguments,
    )

    dirty_run_manifest_path = tmp_path / "run_manifest_dirty.json"
    _write_json(
        dirty_run_manifest_path,
        {"git_commit": expected_git_commit, "git_dirty": True},
    )
    assert not campaign._seed_summary_is_valid(
        {**summary, "run_manifest": _artifact(dirty_run_manifest_path)},
        **arguments,
    )


def test_phase30_seed_summary_resume_binds_bool_and_schema2_sampling(
    tmp_path: Path,
) -> None:
    expected_git_commit = "phase30-commit"
    references: dict[str, dict[str, str]] = {}
    for name in ("checkpoint", "training_log"):
        artifact_path = tmp_path / f"{name}.artifact"
        artifact_path.write_text(name, encoding="utf-8")
        references[name] = _artifact(artifact_path)
    run_manifest_path = tmp_path / "run_manifest.json"
    _write_json(
        run_manifest_path,
        {"git_commit": expected_git_commit, "git_dirty": False},
    )
    references["run_manifest"] = _artifact(run_manifest_path)
    split_path = tmp_path / "split.json"
    _write_json(
        split_path,
        {
            "seed": 17,
            "protocol": "within_event_station",
            "train_record_count": 1788,
            "validation_record_count": 385,
            "test_record_count": 385,
            "assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[17],
        },
    )
    references["split"] = _artifact(split_path)
    normalization = 1788 / 31
    sampling_manifest = {
        "schema_version": 2,
        "mode": "event_equal_inverse_count_full_data",
        "event_balanced_sampling": True,
        "event_balance_estimator": campaign.INVERSE_COUNT_FULL_DATA_ESTIMATOR,
        "sampler_class": "RandomSampler",
        "replacement": False,
        "draw_count": 1788,
        "record_count": 1788,
        "event_count": 31,
        "event_record_count_minimum": 1,
        "event_record_count_maximum": 482,
        "expected_unique_record_count": 1788.0,
        "expected_unique_record_fraction": 1.0,
        "optimizer_step_count": 28,
        "loss_weights_applied": True,
        "objective_weight_formula": "N/(E*n_event)",
        "objective_reduction": "mean(sample_weight * per_sample_loss)",
        "objective_weight_minimum": normalization / 482,
        "objective_weight_maximum": normalization,
        "event_objective_mass_minimum": normalization,
        "event_objective_mass_maximum": normalization,
        "objective_weight_sha256": (
            campaign.PHASE27_INCUMBENT_OBJECTIVE_WEIGHT_SHA256_BY_SEED[17]
        ),
        "sample_weight_sha256": (
            campaign.PHASE27_INCUMBENT_SAMPLE_WEIGHT_SHA256_BY_SEED[17]
        ),
        "sampling_weight_sha256": (
            campaign.PHASE27_INCUMBENT_SAMPLE_WEIGHT_SHA256_BY_SEED[17]
        ),
    }
    sampling_path = tmp_path / "sampling.json"
    _write_json(sampling_path, sampling_manifest)
    references["sampling"] = _artifact(sampling_path)
    expected_config = {
        "model": {"moment_linear_skip": False},
        "training": {
            "random_seed": 17,
            "event_balance_estimator": "inverse_count_full_data",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(expected_config, sort_keys=False),
        encoding="utf-8",
    )
    summary = {
        "variant": "baseline",
        "seed": 17,
        "git_commit": expected_git_commit,
        "event_balance_exponent": 1.0,
        "moment_linear_skip": False,
        "split_assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[17],
        campaign.VALIDATION_METRIC: 0.1,
        "config": _artifact(config_path),
        **references,
    }
    arguments = {
        "require_sampling": True,
        "expected_moment_linear_skip": False,
        "expected_variant": "baseline",
        "expected_seed": 17,
        "expected_split_assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[17],
        "expected_config": expected_config,
        "expected_git_commit": expected_git_commit,
    }

    assert campaign._seed_summary_is_valid(summary, **arguments)
    assert not campaign._seed_summary_is_valid(
        {**summary, "moment_linear_skip": True},
        **arguments,
    )
    assert not campaign._seed_summary_is_valid(
        summary,
        **{**arguments, "expected_moment_linear_skip": True},
    )

    changed_sampling = dict(sampling_manifest)
    changed_sampling["objective_weight_sha256"] = "wrong"
    changed_sampling_path = tmp_path / "sampling_changed.json"
    _write_json(changed_sampling_path, changed_sampling)
    assert not campaign._seed_summary_is_valid(
        {**summary, "sampling": _artifact(changed_sampling_path)},
        **arguments,
    )

    phase31_expected_config = {
        "model": {"moment_head_dropout": True},
        "training": {
            "random_seed": 17,
            "event_balance_estimator": "inverse_count_full_data",
        },
    }
    phase31_config_path = tmp_path / "phase31_config.yaml"
    phase31_config_path.write_text(
        yaml.safe_dump(phase31_expected_config, sort_keys=False),
        encoding="utf-8",
    )
    phase31_summary = {
        **summary,
        "moment_head_dropout": True,
        "config": _artifact(phase31_config_path),
    }
    phase31_arguments = {
        "require_sampling": True,
        "expected_moment_head_dropout": True,
        "expected_variant": "baseline",
        "expected_seed": 17,
        "expected_split_assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[17],
        "expected_config": phase31_expected_config,
        "expected_git_commit": expected_git_commit,
    }

    assert campaign._seed_summary_is_valid(
        phase31_summary,
        **phase31_arguments,
    )
    assert not campaign._seed_summary_is_valid(
        {**phase31_summary, "moment_head_dropout": False},
        **phase31_arguments,
    )
    assert not campaign._seed_summary_is_valid(
        phase31_summary,
        **{**phase31_arguments, "expected_moment_head_dropout": False},
    )

    phase32_weights = dict(campaign.LOSS_WEIGHT_PROFILES["w01"])
    phase32_expected_config = {
        "training": {
            "random_seed": 17,
            "event_balance_estimator": "inverse_count_full_data",
            "stf_rate_loss": phase32_weights,
        }
    }
    phase32_config_path = tmp_path / "phase32_config.yaml"
    phase32_config_path.write_text(
        yaml.safe_dump(phase32_expected_config, sort_keys=False),
        encoding="utf-8",
    )
    phase32_summary = {
        **summary,
        "loss_weight_profile": "w01",
        "loss_weights": phase32_weights,
        "loss_weight_profile_sha256": (
            campaign.LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256
        ),
        "config": _artifact(phase32_config_path),
    }
    phase32_arguments = {
        "require_sampling": True,
        "expected_loss_weight_profile": "w01",
        "expected_loss_weights": phase32_weights,
        "expected_loss_weight_profile_sha256": (
            campaign.LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256
        ),
        "expected_variant": "baseline",
        "expected_seed": 17,
        "expected_split_assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[17],
        "expected_config": phase32_expected_config,
        "expected_git_commit": expected_git_commit,
    }

    assert campaign._seed_summary_is_valid(
        phase32_summary,
        **phase32_arguments,
    )
    assert not campaign._seed_summary_is_valid(
        {**phase32_summary, "loss_weight_profile": "w10"},
        **phase32_arguments,
    )
    assert not campaign._seed_summary_is_valid(
        {
            **phase32_summary,
            "loss_weight_profile_sha256": "wrong",
        },
        **phase32_arguments,
    )
    for weight_name in campaign.LOSS_WEIGHT_KEYS:
        changed_weights = dict(phase32_weights)
        changed_weights[weight_name] += 1.0e-9
        assert not campaign._seed_summary_is_valid(
            {**phase32_summary, "loss_weights": changed_weights},
            **phase32_arguments,
        )


def test_phase33_seed_resume_binds_weights_config_commit_split_and_sampling(
    tmp_path: Path,
) -> None:
    seed = 17
    commit = "phase33-commit"
    weights = {
        "lambda_MSE": 1.0,
        "lambda_synth": 0.5,
        "lambda_mag": 1.0,
        "lambda_shape": 0.0,
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    training_log_path = tmp_path / "training.csv"
    training_log_path.write_text("epoch\n", encoding="utf-8")
    run_manifest_path = tmp_path / "run_manifest.json"
    _write_json(run_manifest_path, {"git_commit": commit, "git_dirty": False})
    split_path = tmp_path / "split.json"
    _write_json(
        split_path,
        {
            "seed": seed,
            "protocol": "within_event_station",
            "train_record_count": 1788,
            "validation_record_count": 385,
            "test_record_count": 385,
            "assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[seed],
        },
    )
    sampling_manifest = _phase27_sampling_manifest(seed)
    sampling_path = tmp_path / "sampling.json"
    _write_json(sampling_path, sampling_manifest)
    expected_config = {
        "training": {
            "random_seed": seed,
            "event_balance_estimator": campaign.INVERSE_COUNT_FULL_DATA_ESTIMATOR,
            "stf_rate_loss": weights,
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(expected_config, sort_keys=False),
        encoding="utf-8",
    )
    summary = {
        "variant": "candidate",
        "seed": seed,
        "git_commit": commit,
        "event_balance_exponent": 1.0,
        "loss_weights": weights,
        "split_assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[seed],
        campaign.VALIDATION_METRIC: 0.1,
        "checkpoint": _artifact(checkpoint_path),
        "config": _artifact(config_path),
        "split": _artifact(split_path),
        "training_log": _artifact(training_log_path),
        "run_manifest": _artifact(run_manifest_path),
        "sampling": _artifact(sampling_path),
    }
    arguments = {
        "require_sampling": True,
        "expected_loss_weights": weights,
        "expected_variant": "candidate",
        "expected_seed": seed,
        "expected_split_assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[seed],
        "expected_config": expected_config,
        "expected_git_commit": commit,
    }

    assert campaign._seed_summary_is_valid(summary, **arguments)

    changed_weights = dict(weights)
    changed_weights["lambda_shape"] = 0.1
    assert not campaign._seed_summary_is_valid(
        {**summary, "loss_weights": changed_weights},
        **arguments,
    )

    changed_config = copy.deepcopy(expected_config)
    changed_config["training"]["stf_rate_loss"]["lambda_shape"] = 0.1
    changed_config_path = tmp_path / "changed_config.yaml"
    changed_config_path.write_text(
        yaml.safe_dump(changed_config, sort_keys=False),
        encoding="utf-8",
    )
    assert not campaign._seed_summary_is_valid(
        {**summary, "config": _artifact(changed_config_path)},
        **arguments,
    )

    assert not campaign._seed_summary_is_valid(
        {**summary, "git_commit": "other-commit"},
        **arguments,
    )

    changed_split_path = tmp_path / "changed_split.json"
    _write_json(
        changed_split_path,
        {
            "seed": seed,
            "protocol": "within_event_station",
            "train_record_count": 1788,
            "validation_record_count": 385,
            "test_record_count": 385,
            "assignment_sha256": "wrong",
        },
    )
    assert not campaign._seed_summary_is_valid(
        {**summary, "split": _artifact(changed_split_path)},
        **arguments,
    )

    changed_sampling = dict(sampling_manifest)
    changed_sampling["objective_weight_sha256"] = "wrong"
    changed_sampling_path = tmp_path / "changed_sampling.json"
    _write_json(changed_sampling_path, changed_sampling)
    assert not campaign._seed_summary_is_valid(
        {**summary, "sampling": _artifact(changed_sampling_path)},
        **arguments,
    )


def test_phase30_train_resume_passes_complete_strict_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_root = tmp_path / "seed_17"
    _write_json(seed_root / "seed_summary.json", {"existing": True})
    config = campaign.build_variant_configs(_config(PHASE30_CONFIG_PATH))[
        "baseline"
    ]
    captured: dict[str, Any] = {}

    def fake_validate(_summary: dict[str, Any], **kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr(campaign, "_seed_summary_is_valid", fake_validate)
    monkeypatch.setattr(campaign, "current_git_commit", lambda _root: "phase30")

    resumed = campaign._train_one_seed(
        variant="baseline",
        config=config,
        seed=17,
        seed_root=seed_root,
        dataset_manifest=tmp_path / "dataset.csv",
        frozen_split_path=tmp_path / "split.json",
        resume=True,
    )

    assert resumed == {"existing": True}
    assert captured["require_sampling"] is True
    assert captured["expected_moment_linear_skip"] is False
    assert captured["expected_variant"] == "baseline"
    assert captured["expected_seed"] == 17
    assert captured["expected_split_assignment_sha256"] == (
        campaign.EXPECTED_SPLIT_SHA256[17]
    )
    assert captured["expected_config"]["model"]["moment_linear_skip"] is False
    assert "event_balance_exponent" not in captured["expected_config"]["training"]
    assert captured["expected_git_commit"] == "phase30"


def test_phase31_train_resume_passes_complete_strict_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_root = tmp_path / "seed_17"
    _write_json(seed_root / "seed_summary.json", {"existing": True})
    config = campaign.build_variant_configs(_config(PHASE31_CONFIG_PATH))[
        "baseline"
    ]
    captured: dict[str, Any] = {}

    def fake_validate(_summary: dict[str, Any], **kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr(campaign, "_seed_summary_is_valid", fake_validate)
    monkeypatch.setattr(campaign, "current_git_commit", lambda _root: "phase31")

    resumed = campaign._train_one_seed(
        variant="baseline",
        config=config,
        seed=17,
        seed_root=seed_root,
        dataset_manifest=tmp_path / "dataset.csv",
        frozen_split_path=tmp_path / "split.json",
        resume=True,
    )

    assert resumed == {"existing": True}
    assert captured["require_sampling"] is True
    assert captured["expected_moment_head_dropout"] is True
    assert captured["expected_variant"] == "baseline"
    assert captured["expected_seed"] == 17
    assert captured["expected_split_assignment_sha256"] == (
        campaign.EXPECTED_SPLIT_SHA256[17]
    )
    assert captured["expected_config"]["model"]["moment_head_dropout"] is True
    assert "moment_linear_skip" not in captured["expected_config"]["model"]
    assert "event_balance_exponent" not in captured["expected_config"]["training"]
    assert captured["expected_git_commit"] == "phase31"


def test_phase33_train_resume_passes_complete_strict_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_root = tmp_path / "seed_17"
    _write_json(seed_root / "seed_summary.json", {"existing": True})
    config = campaign.build_variant_configs(_config(PHASE33_CONFIG_PATH))[
        "candidate"
    ]
    captured: dict[str, Any] = {}

    def fake_validate(_summary: dict[str, Any], **kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr(campaign, "_seed_summary_is_valid", fake_validate)
    monkeypatch.setattr(campaign, "current_git_commit", lambda _root: "phase33")

    resumed = campaign._train_one_seed(
        variant="candidate",
        config=config,
        seed=17,
        seed_root=seed_root,
        dataset_manifest=tmp_path / "dataset.csv",
        frozen_split_path=tmp_path / "split.json",
        resume=True,
    )

    assert resumed == {"existing": True}
    assert captured["require_sampling"] is True
    assert captured["expected_loss_weights"] == {
        "lambda_MSE": 1.0,
        "lambda_synth": 0.5,
        "lambda_mag": 1.0,
        "lambda_shape": 0.0,
    }
    assert captured["expected_variant"] == "candidate"
    assert captured["expected_seed"] == 17
    assert captured["expected_split_assignment_sha256"] == (
        campaign.EXPECTED_SPLIT_SHA256[17]
    )
    assert captured["expected_config"]["training"]["stf_rate_loss"][
        "lambda_shape"
    ] == 0.0
    assert captured["expected_config"]["training"]["random_seed"] == 17
    assert captured["expected_git_commit"] == "phase33"


def _logged_scheduler_learning_rates(scheduler_t0: int) -> list[float]:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=scheduler_t0,
        T_mult=2,
        eta_min=1.0e-6,
    )
    learning_rates: list[float] = []
    for epoch in range(200):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().backward()
        optimizer.step()
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        if epoch >= 5:
            scheduler.step()
    return learning_rates


def test_phase24_scheduler_has_no_restart_in_the_logged_training_window() -> None:
    baseline = _logged_scheduler_learning_rates(15)
    candidate = _logged_scheduler_learning_rates(195)

    assert candidate[:6] == pytest.approx([1.0e-4] * 6)
    assert all(next_lr <= lr for lr, next_lr in zip(candidate, candidate[1:]))
    baseline_restart_epochs = [
        epoch
        for epoch in range(2, 201)
        if baseline[epoch - 1] > baseline[epoch - 2]
    ]
    assert baseline_restart_epochs == [21, 51, 111]


def test_seed_selection_uses_validation_event_mae_and_never_ensembles() -> None:
    seed_rows = {
        17: {campaign.VALIDATION_METRIC: 0.22},
        42: {campaign.VALIDATION_METRIC: 0.18},
        73: {campaign.VALIDATION_METRIC: 0.20},
    }

    assert campaign.select_seed_by_validation(seed_rows) == 42

    tied = copy.deepcopy(seed_rows)
    tied[17][campaign.VALIDATION_METRIC] = 0.18
    assert campaign.select_seed_by_validation(tied) == 17


def test_locked_test_loader_raises_on_iteration() -> None:
    class Loader:
        dataset = [1, 2, 3]

        def __len__(self) -> int:
            return 2

    locked = campaign.LockedTestLoader(Loader())

    assert len(locked.dataset) == 3
    assert len(locked) == 2
    with pytest.raises(RuntimeError, match="forbidden"):
        iter(locked)


def test_sampling_manifest_audits_balanced_replacement_without_consuming_rng() -> None:
    class SamplingDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            events = ["E00"]
            events.extend(
                event
                for index in range(1, 30)
                for event in [f"E{index:02d}"] * 45
            )
            events.extend(["E30"] * 482)
            self.samples = [
                {"event": event, "station": f"S{index:04d}"}
                for index, event in enumerate(events)
            ]

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> int:
            return index

    dataset = SamplingDataset()
    subset = torch.utils.data.Subset(dataset, range(len(dataset)))
    counts: dict[str, int] = {}
    for row in dataset.samples:
        event = str(row["event"])
        counts[event] = counts.get(event, 0) + 1
    weights = [1.0 / counts[str(row["event"])] for row in dataset.samples]
    generator = torch.Generator().manual_seed(17)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=64,
        sampler=sampler,
        generator=generator,
    )
    state_before = generator.get_state().clone()

    manifest = campaign._training_sampling_manifest(
        loader,
        {"training": {"event_balanced_sampling": True}},
    )

    assert torch.equal(generator.get_state(), state_before)
    assert manifest["mode"] == "event_equal_with_replacement"
    assert manifest["sampler_class"] == "WeightedRandomSampler"
    assert manifest["replacement"] is True
    assert manifest["draw_count"] == 1788
    assert manifest["event_count"] == 31
    assert manifest["expected_event_draws_minimum"] == pytest.approx(1788 / 31)
    assert manifest["expected_event_draws_maximum"] == pytest.approx(1788 / 31)
    assert manifest["expected_unique_record_count"] < 1788
    assert len(manifest["sample_weight_sha256"]) == 64
    assert len(manifest["objective_weight_sha256"]) == 64

    baseline_generator = torch.Generator().manual_seed(17)
    baseline_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=64,
        shuffle=True,
        generator=baseline_generator,
    )
    baseline = campaign._training_sampling_manifest(
        baseline_loader,
        {"training": {"event_balanced_sampling": False}},
    )
    assert baseline["mode"] == "station_uniform_without_replacement"
    assert baseline["sampler_class"] == "RandomSampler"
    assert baseline["replacement"] is False
    assert baseline["expected_unique_record_count"] == pytest.approx(1788)

    baseline_loader.event_balance_weights_by_event = {
        event: 1788 / (31 * count) for event, count in counts.items()
    }
    baseline_loader.event_balance_exponent = 1.0
    full_data = campaign._training_sampling_manifest(
        baseline_loader,
        {
            "training": {
                "event_balanced_sampling": True,
                "event_balance_estimator": (
                    campaign.INVERSE_COUNT_FULL_DATA_ESTIMATOR
                ),
            }
        },
    )
    assert full_data["mode"] == "event_equal_inverse_count_full_data"
    assert full_data["sampler_class"] == "RandomSampler"
    assert full_data["replacement"] is False
    assert full_data["expected_unique_record_count"] == pytest.approx(1788)
    assert full_data["expected_unique_record_fraction"] == pytest.approx(1.0)
    assert full_data["optimizer_step_count"] == 28
    assert full_data["loss_weights_applied"] is True
    assert full_data["objective_weight_formula"] == "N/(E*n_event)"
    assert (
        full_data["objective_reduction"]
        == "mean(sample_weight * per_sample_loss)"
    )
    assert full_data["objective_weight_minimum"] == pytest.approx(1788 / (31 * 482))
    assert full_data["objective_weight_maximum"] == pytest.approx(1788 / 31)
    assert full_data["event_objective_mass_minimum"] == pytest.approx(1788 / 31)
    assert full_data["event_objective_mass_maximum"] == pytest.approx(1788 / 31)

    event_names = [str(row["event"]) for row in dataset.samples]
    tempered_exponent = 0.5
    tempered_weights = campaign.make_event_inverse_count_weights(
        event_names,
        exponent=tempered_exponent,
    )
    baseline_loader.event_balance_weights_by_event = dict(
        zip(event_names, tempered_weights, strict=True)
    )
    baseline_loader.event_balance_exponent = tempered_exponent
    tempered = campaign._training_sampling_manifest(
        baseline_loader,
        {
            "training": {
                "event_balanced_sampling": True,
                "event_balance_estimator": (
                    campaign.INVERSE_COUNT_FULL_DATA_ESTIMATOR
                ),
                "event_balance_exponent": tempered_exponent,
            }
        },
    )
    normalization = 1788 / sum(
        count ** (1.0 - tempered_exponent) for count in counts.values()
    )
    event_masses = [
        normalization * count ** (1.0 - tempered_exponent)
        for count in counts.values()
    ]
    expected_ess = sum(event_masses) ** 2 / sum(
        mass**2 for mass in event_masses
    )
    assert tempered["schema_version"] == 3
    assert tempered["mode"] == "tempered_inverse_count_full_data"
    assert tempered["event_balance_exponent"] == 0.5
    assert tempered["equal_event_objective_mass"] is False
    assert (
        tempered["objective_weight_formula"]
        == "C*n_event^(-p), C=N/sum_event(n_event^(1-p))"
    )
    assert tempered["objective_event_mass_formula"] == "C*n_event^(1-p)"
    assert tempered["objective_normalization_constant"] == pytest.approx(
        normalization
    )
    assert tempered["event_objective_mass_ratio"] == pytest.approx(482**0.5)
    assert tempered["event_objective_mass_ess"] == pytest.approx(expected_ess)
    assert tempered["event_objective_mass_minimum"] < tempered[
        "event_objective_mass_maximum"
    ]
    assert campaign._phase29_sampling_manifest_is_valid(
        tempered,
        expected_exponent=0.5,
    )


def test_phase29_smoke_probe_uses_exact_frozen_split_weight_extrema() -> None:
    counts = {"E00": 1, "E30": 482}
    counts.update({f"E{index:02d}": 45 for index in range(1, 30)})
    split_manifests = {
        seed: {
            "per_event_station_counts": {
                event: {"train": count, "validation": 0, "test": 0}
                for event, count in counts.items()
            }
        }
        for seed in campaign.SEEDS
    }
    config = campaign.build_variant_configs(_config(PHASE29_CONFIG_PATH))[
        "candidate"
    ]

    probe = campaign._formal_sample_weight_probe(split_manifests, config)

    assert probe is not None
    events = [event for event, count in counts.items() for _ in range(count)]
    expected_weights = campaign.make_event_inverse_count_weights(
        events,
        exponent=0.5,
    )
    assert probe["source"] == "frozen_train_split_event_counts"
    assert probe["event_balance_exponent"] == 0.5
    assert probe["minimum"] == pytest.approx(min(expected_weights))
    assert probe["maximum"] == pytest.approx(max(expected_weights))
    assert probe["maximum"] != pytest.approx((1788 / 31) ** 0.5)


def test_phase30_smoke_enforces_frozen_parameter_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(PHASE30_CONFIG_PATH), sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "dataset.csv"
    manifest_path.write_text("event,station\n", encoding="utf-8")
    splits: dict[str, Any] = {}
    for seed in campaign.SEEDS:
        split_path = tmp_path / f"split_{seed}.json"
        _write_json(split_path, {})
        splits[str(seed)] = {"manifest": _artifact(split_path)}
    _mark_stage(
        tmp_path,
        "preflight",
        {
            "status": "complete",
            "source_data": {"sha256": campaign.EXPECTED_SOURCE_SHA256},
            "frozen_config": _artifact(config_path),
            "dataset_manifest": _artifact(manifest_path),
            "splits": splits,
        },
    )
    monkeypatch.setattr(campaign, "CorrectedEarthquakeDataset", lambda _config: [])
    monkeypatch.setattr(campaign.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        campaign,
        "_formal_sample_weight_probe",
        lambda _splits, _config: {},
    )

    def smoke_result(config: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        enabled = campaign.moment_linear_skip_from_config(config)
        variant = "candidate" if enabled else "baseline"
        return {
            "parameter_count": campaign.PHASE30_PARAMETER_COUNT_BY_VARIANT[variant],
            "moment_linear_skip": enabled,
        }

    monkeypatch.setattr(campaign, "_smoke_one_device", smoke_result)

    summary = campaign.run_smoke(output_root=tmp_path, resume=False)

    assert summary["results"]["baseline"]["cpu"]["parameter_count"] == 1_010_850
    assert summary["results"]["candidate"]["cuda"]["parameter_count"] == 1_010_978


def test_phase30_smoke_rejects_parameter_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(PHASE30_CONFIG_PATH), sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "dataset.csv"
    manifest_path.write_text("event,station\n", encoding="utf-8")
    splits: dict[str, Any] = {}
    for seed in campaign.SEEDS:
        split_path = tmp_path / f"split_{seed}.json"
        _write_json(split_path, {})
        splits[str(seed)] = {"manifest": _artifact(split_path)}
    _mark_stage(
        tmp_path,
        "preflight",
        {
            "status": "complete",
            "source_data": {"sha256": campaign.EXPECTED_SOURCE_SHA256},
            "frozen_config": _artifact(config_path),
            "dataset_manifest": _artifact(manifest_path),
            "splits": splits,
        },
    )
    monkeypatch.setattr(campaign, "CorrectedEarthquakeDataset", lambda _config: [])
    monkeypatch.setattr(campaign.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        campaign,
        "_formal_sample_weight_probe",
        lambda _splits, _config: {},
    )
    monkeypatch.setattr(
        campaign,
        "_smoke_one_device",
        lambda _config, **_kwargs: {"parameter_count": 1},
    )

    with pytest.raises(ValueError, match="Phase30 parameter count changed"):
        campaign.run_smoke(output_root=tmp_path, resume=False)


@pytest.mark.parametrize("parameter_count", [1_010_850, 1])
def test_phase31_smoke_enforces_equal_frozen_parameter_counts(
    parameter_count: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(PHASE31_CONFIG_PATH), sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "dataset.csv"
    manifest_path.write_text("event,station\n", encoding="utf-8")
    splits: dict[str, Any] = {}
    for seed in campaign.SEEDS:
        split_path = tmp_path / f"split_{seed}.json"
        _write_json(split_path, {})
        splits[str(seed)] = {"manifest": _artifact(split_path)}
    _mark_stage(
        tmp_path,
        "preflight",
        {
            "status": "complete",
            "source_data": {"sha256": campaign.EXPECTED_SOURCE_SHA256},
            "frozen_config": _artifact(config_path),
            "dataset_manifest": _artifact(manifest_path),
            "splits": splits,
        },
    )
    monkeypatch.setattr(campaign, "CorrectedEarthquakeDataset", lambda _config: [])
    monkeypatch.setattr(campaign.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        campaign,
        "_formal_sample_weight_probe",
        lambda _splits, _config: {},
    )

    def smoke_result(config: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        return {
            "parameter_count": parameter_count,
            "moment_head_dropout": campaign.moment_head_dropout_from_config(config),
        }

    monkeypatch.setattr(campaign, "_smoke_one_device", smoke_result)

    if parameter_count != 1_010_850:
        with pytest.raises(ValueError, match="Phase31 parameter count changed"):
            campaign.run_smoke(output_root=tmp_path, resume=False)
        return

    summary = campaign.run_smoke(output_root=tmp_path, resume=False)
    assert summary["variant_axis"] == campaign.MOMENT_HEAD_DROPOUT_AXIS
    assert summary["results"]["baseline"]["cpu"] == {
        "parameter_count": 1_010_850,
        "moment_head_dropout": True,
    }
    assert summary["results"]["candidate"]["cuda"] == {
        "parameter_count": 1_010_850,
        "moment_head_dropout": False,
    }


@pytest.mark.parametrize(
    (
        "formal_config_path",
        "expected_axis",
        "expected_parameterizations",
        "expected_scheduler_t0s",
        "expected_stems",
        "expected_event_balanced",
        "expected_estimators",
    ),
    [
        (
            PHASE23_CONFIG_PATH,
            campaign.STF_OUTPUT_PARAMETERIZATION_AXIS,
            ("direct", "moment_shape_factorized"),
            (15, 15),
            ("none", "none"),
            (False, False),
            ("station_uniform", "station_uniform"),
        ),
        (
            PHASE24_CONFIG_PATH,
            campaign.SCHEDULER_T0_AXIS,
            ("moment_shape_factorized", "moment_shape_factorized"),
            (15, 195),
            ("none", "none"),
            (False, False),
            ("station_uniform", "station_uniform"),
        ),
        (
            PHASE25_CONFIG_PATH,
            campaign.RADIAL_DYNAMIC_RANGE_STEM_AXIS,
            ("moment_shape_factorized", "moment_shape_factorized"),
            (15, 15),
            ("none", "asinh_residual"),
            (False, False),
            ("station_uniform", "station_uniform"),
        ),
        (
            PHASE26_CONFIG_PATH,
            campaign.EVENT_BALANCED_SAMPLING_AXIS,
            ("moment_shape_factorized", "moment_shape_factorized"),
            (15, 15),
            ("none", "none"),
            (False, True),
            ("station_uniform", "replacement_sampling"),
        ),
        (
            PHASE27_CONFIG_PATH,
            campaign.EVENT_BALANCE_ESTIMATOR_AXIS,
            ("moment_shape_factorized", "moment_shape_factorized"),
            (15, 15),
            ("none", "none"),
            (True, True),
            ("replacement_sampling", "inverse_count_full_data"),
        ),
        (
            PHASE28_CONFIG_PATH,
            campaign.MAGNITUDE_PENALTY_AXIS,
            ("moment_shape_factorized", "moment_shape_factorized"),
            (15, 15),
            ("none", "none"),
            (True, True),
            ("inverse_count_full_data", "inverse_count_full_data"),
        ),
        (
            PHASE31_CONFIG_PATH,
            campaign.MOMENT_HEAD_DROPOUT_AXIS,
            ("moment_shape_factorized", "moment_shape_factorized"),
            (15, 15),
            ("none", "none"),
            (True, True),
            ("inverse_count_full_data", "inverse_count_full_data"),
        ),
        (
            PHASE30_CONFIG_PATH,
            campaign.MOMENT_LINEAR_SKIP_AXIS,
            ("moment_shape_factorized", "moment_shape_factorized"),
            (15, 15),
            ("none", "none"),
            (True, True),
            ("inverse_count_full_data", "inverse_count_full_data"),
        ),
        (
            PHASE29_CONFIG_PATH,
            campaign.EVENT_BALANCE_EXPONENT_AXIS,
            ("moment_shape_factorized", "moment_shape_factorized"),
            (15, 15),
            ("none", "none"),
            (True, True),
            ("inverse_count_full_data", "inverse_count_full_data"),
        ),
        (
            PHASE33_CONFIG_PATH,
            campaign.SHAPE_LOSS_WEIGHT_AXIS,
            ("moment_shape_factorized", "moment_shape_factorized"),
            (15, 15),
            ("none", "none"),
            (True, True),
            ("inverse_count_full_data", "inverse_count_full_data"),
        ),
    ],
)
def test_train_stage_selects_from_validation_without_test_or_external(
    formal_config_path: Path,
    expected_axis: str,
    expected_parameterizations: tuple[str, str],
    expected_scheduler_t0s: tuple[int, int],
    expected_stems: tuple[str, str],
    expected_event_balanced: tuple[bool, bool],
    expected_estimators: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "frozen_config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(formal_config_path), sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "dataset_manifest.csv"
    manifest_path.write_text("event,station\n", encoding="utf-8")
    splits: dict[str, Any] = {}
    for seed in campaign.SEEDS:
        split_path = tmp_path / f"split_{seed}.json"
        _write_json(split_path, {"seed": seed})
        splits[str(seed)] = {"manifest": _artifact(split_path)}
    _mark_stage(
        tmp_path,
        "preflight",
        {
            "status": "complete",
            "source_data": {"sha256": campaign.EXPECTED_SOURCE_SHA256},
            "frozen_config": _artifact(config_path),
            "dataset_manifest": _artifact(manifest_path),
            "splits": splits,
        },
    )
    _mark_stage(tmp_path, "smoke", {"status": "complete"})
    values = {
        "baseline": {17: 0.30, 42: 0.20, 73: 0.25},
        "candidate": {17: 0.19, 42: 0.21, 73: 0.18},
    }
    if expected_axis in campaign.FROZEN_PHASE27_INCUMBENT_AXES:
        values["baseline"] = dict(
            campaign.PHASE27_INCUMBENT_VALIDATION_BY_SEED
        )
    calls: list[tuple[str, int]] = []

    def fake_train_one_seed(**kwargs: Any) -> dict[str, Any]:
        variant = str(kwargs["variant"])
        seed = int(kwargs["seed"])
        calls.append((variant, seed))
        summary = {
            "variant": variant,
            "seed": seed,
            campaign.VALIDATION_METRIC: values[variant][seed],
        }
        if expected_axis in campaign.FROZEN_PHASE27_INCUMBENT_AXES:
            summary["checkpoint"] = {
                "sha256": (
                    campaign.PHASE27_INCUMBENT_CHECKPOINT_SHA256_BY_SEED[seed]
                    if variant == "baseline"
                    else f"candidate-{seed}"
                )
            }
        return summary

    monkeypatch.setattr(campaign, "_train_one_seed", fake_train_one_seed)
    monkeypatch.setattr(
        campaign,
        "_evaluate_locked_test",
        lambda **_kwargs: pytest.fail("train stage must not evaluate test"),
    )
    monkeypatch.setattr(
        campaign,
        "_evaluate_external_threshold",
        lambda **_kwargs: pytest.fail("train stage must not evaluate external data"),
    )
    monkeypatch.setattr(
        campaign,
        "evaluate_delayed_prefix",
        lambda *_args, **_kwargs: pytest.fail(
            "train stage must not evaluate delayed test prefixes"
        ),
    )

    summary = campaign.run_train(output_root=tmp_path, resume=False)

    assert calls == [
        (variant, seed)
        for variant in ("baseline", "candidate")
        for seed in campaign.SEEDS
    ]
    assert summary["test_evaluated"] is False
    assert summary["external_evaluated"] is False
    assert summary["variant_axis"] == expected_axis
    expected_baseline_seed = (
        campaign.PHASE27_INCUMBENT_SELECTED_SEED
        if expected_axis in campaign.FROZEN_PHASE27_INCUMBENT_AXES
        else 42
    )
    assert (
        summary["variants"]["baseline"]["selection"]["selected_seed"]
        == expected_baseline_seed
    )
    assert summary["variants"]["candidate"]["selection"]["selected_seed"] == 73
    variant_names = ("baseline", "candidate")
    assert tuple(
        summary["variants"][name]["parameterization"] for name in variant_names
    ) == expected_parameterizations
    assert tuple(
        summary["variants"][name]["scheduler_T0"] for name in variant_names
    ) == expected_scheduler_t0s
    assert tuple(
        summary["variants"][name]["radial_dynamic_range_stem"]
        for name in variant_names
    ) == expected_stems
    assert tuple(
        summary["variants"][name]["event_balanced_sampling"]
        for name in variant_names
    ) == expected_event_balanced
    assert tuple(
        summary["variants"][name]["event_balance_estimator"]
        for name in variant_names
    ) == expected_estimators
    expected_exponents = (
        (1.0, 0.5)
        if expected_axis == campaign.EVENT_BALANCE_EXPONENT_AXIS
        else (1.0, 1.0)
    )
    assert tuple(
        summary["variants"][name]["event_balance_exponent"]
        for name in variant_names
    ) == expected_exponents
    expected_penalties = (
        ("squared", "absolute")
        if expected_axis == campaign.MAGNITUDE_PENALTY_AXIS
        else ("squared", "squared")
    )
    assert tuple(
        summary["variants"][name]["magnitude_penalty"]
        for name in variant_names
    ) == expected_penalties
    expected_moment_skips = (
        (False, True)
        if expected_axis == campaign.MOMENT_LINEAR_SKIP_AXIS
        else (False, False)
    )
    assert tuple(
        summary["variants"][name]["moment_linear_skip"]
        for name in variant_names
    ) == expected_moment_skips
    expected_moment_dropouts = (
        (True, False)
        if expected_axis == campaign.MOMENT_HEAD_DROPOUT_AXIS
        else (True, True)
    )
    assert tuple(
        summary["variants"][name]["moment_head_dropout"]
        for name in variant_names
    ) == expected_moment_dropouts
    expected_shape_weights = (
        (0.1, 0.0)
        if expected_axis == campaign.SHAPE_LOSS_WEIGHT_AXIS
        else (0.1, 0.1)
    )
    assert tuple(
        summary["variants"][name]["loss_weights"]["lambda_shape"]
        for name in variant_names
    ) == expected_shape_weights
    assert summary["variants"]["candidate"]["scientific_diff_from_baseline"] == [
        ".".join(campaign.VARIANT_AXIS_PATHS[expected_axis])
    ]
    if expected_axis in campaign.FROZEN_PHASE27_INCUMBENT_AXES:
        assert summary["incumbent_reproduction"]["passed"] is True


def test_phase32_train_runs_baseline_then_two_profiles_for_all_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "frozen_config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(PHASE32_CONFIG_PATH), sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "dataset_manifest.csv"
    manifest_path.write_text("event,station\n", encoding="utf-8")
    splits: dict[str, Any] = {}
    for seed in campaign.SEEDS:
        split_path = tmp_path / f"split_{seed}.json"
        _write_json(split_path, {"seed": seed})
        splits[str(seed)] = {"manifest": _artifact(split_path)}
    _mark_stage(
        tmp_path,
        "preflight",
        {
            "status": "complete",
            "source_data": {"sha256": campaign.EXPECTED_SOURCE_SHA256},
            "frozen_config": _artifact(config_path),
            "dataset_manifest": _artifact(manifest_path),
            "splits": splits,
        },
    )
    _mark_stage(tmp_path, "smoke", {"status": "complete"})
    values = {
        "baseline": dict(campaign.PHASE27_INCUMBENT_VALIDATION_BY_SEED),
        "w01": {17: 0.100, 42: 0.120, 73: 0.190},
        "w10": {17: 0.120, 42: 0.130, 73: 0.180},
    }
    calls: list[tuple[str, int]] = []

    def fake_train_one_seed(**kwargs: Any) -> dict[str, Any]:
        profile_id = str(kwargs["variant"])
        seed = int(kwargs["seed"])
        calls.append((profile_id, seed))
        return {
            "variant": profile_id,
            "seed": seed,
            campaign.VALIDATION_METRIC: values[profile_id][seed],
            "checkpoint": {
                "sha256": (
                    campaign.PHASE27_INCUMBENT_CHECKPOINT_SHA256_BY_SEED[seed]
                    if profile_id == "baseline"
                    else f"{profile_id}-{seed}"
                )
            },
        }

    monkeypatch.setattr(campaign, "_train_one_seed", fake_train_one_seed)

    summary = campaign.run_train(output_root=tmp_path, resume=False)

    assert calls == [
        (profile_id, seed)
        for profile_id in ("baseline", "w01", "w10")
        for seed in campaign.SEEDS
    ]
    assert summary["incumbent_reproduction"]["passed"] is True
    assert summary["selected_candidate_variant"] == "w01"
    assert summary["loss_weight_profile_selection"]["winner_profile"] == "w01"
    assert summary["test_evaluated"] is False
    assert summary["external_evaluated"] is False


@pytest.mark.parametrize(
    "config_path",
    [
        PHASE29_CONFIG_PATH,
        PHASE30_CONFIG_PATH,
        PHASE31_CONFIG_PATH,
        PHASE32_CONFIG_PATH,
        PHASE33_CONFIG_PATH,
    ],
)
@pytest.mark.parametrize("mismatch", ["validation_metric", "checkpoint_sha256"])
def test_frozen_incumbent_control_drift_stops_before_candidate_training(
    config_path: Path,
    mismatch: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_config_path = tmp_path / "frozen_config.yaml"
    frozen_config_path.write_text(
        yaml.safe_dump(_config(config_path), sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "dataset_manifest.csv"
    manifest_path.write_text("event,station\n", encoding="utf-8")
    splits: dict[str, Any] = {}
    for seed in campaign.SEEDS:
        split_path = tmp_path / f"split_{seed}.json"
        _write_json(split_path, {"seed": seed})
        splits[str(seed)] = {"manifest": _artifact(split_path)}
    _mark_stage(
        tmp_path,
        "preflight",
        {
            "status": "complete",
            "source_data": {"sha256": campaign.EXPECTED_SOURCE_SHA256},
            "frozen_config": _artifact(frozen_config_path),
            "dataset_manifest": _artifact(manifest_path),
            "splits": splits,
        },
    )
    _mark_stage(tmp_path, "smoke", {"status": "complete"})
    calls: list[tuple[str, int]] = []

    def fake_train_one_seed(**kwargs: Any) -> dict[str, Any]:
        variant = str(kwargs["variant"])
        seed = int(kwargs["seed"])
        calls.append((variant, seed))
        metric = campaign.PHASE27_INCUMBENT_VALIDATION_BY_SEED[seed]
        checkpoint_sha256 = (
            campaign.PHASE27_INCUMBENT_CHECKPOINT_SHA256_BY_SEED[seed]
        )
        if seed == 42 and mismatch == "validation_metric":
            metric += 1.0e-12
        if seed == 42 and mismatch == "checkpoint_sha256":
            checkpoint_sha256 = "wrong"
        return {
            "variant": variant,
            "seed": seed,
            campaign.VALIDATION_METRIC: metric,
            "checkpoint": {"sha256": checkpoint_sha256},
        }

    monkeypatch.setattr(campaign, "_train_one_seed", fake_train_one_seed)

    with pytest.raises(ValueError, match="refusing to train or compare"):
        campaign.run_train(output_root=tmp_path, resume=False)

    assert calls == [("baseline", seed) for seed in campaign.SEEDS]
    reproduction = json.loads(
        (tmp_path / "train" / "baseline" / "incumbent_reproduction.json").read_text(
            encoding="utf-8"
        )
    )
    assert reproduction["passed"] is False
    assert not (tmp_path / "train" / "candidate").exists()


def test_delayed_prefix_result_persists_auditable_contract_and_outputs(
    tmp_path: Path,
) -> None:
    horizon_metrics = [
        {
            "observation_horizon_sec": float(horizon),
            "release_time_sec": float(horizon + 5),
            "event_equal_mae": 0.1,
        }
        for horizon in campaign.DEFAULT_HORIZONS_SEC
    ]
    result = campaign._persist_delayed_prefix_result(
        {
            "cohort": {
                "cohort": "processed radial peak over the full 200 s record >= 2 cm",
                "radial_peak_min_cm": 2.0,
                "waveform_prefix_causal": True,
                "station_selection_causal": False,
                "end_to_end_causal": False,
            },
            "station_rows": [{"event": "E", "station": "S", "mw_pred": 7.0}],
            "event_rows": [{"event": "E", "mw_pred_median": 7.0}],
            "unavailable_rows": [],
            "horizon_metrics": horizon_metrics,
        },
        output_dir=tmp_path,
        resume=False,
    )

    assert result["horizons_sec"] == list(campaign.DEFAULT_HORIZONS_SEC)
    assert result["cohort"]["station_selection_causal"] is False
    assert result["cohort"]["end_to_end_causal"] is False
    assert result["unavailable_station_count"] == 0
    assert set(result["artifacts"]) == {
        "station_predictions",
        "event_predictions",
        "unavailable_stations",
        "horizon_metrics",
        "cohort_contract",
    }
    for reference in result["artifacts"].values():
        assert campaign._validate_artifact(reference, label="test").is_file()
    assert (tmp_path / "unavailable_stations.csv").read_text(
        encoding="utf-8"
    ).startswith("event,station,observation_horizon_sec")


def test_internal_runs_delayed_prefix_only_for_selected_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mark_stage(
        tmp_path,
        "train",
        _selected_train_summary(baseline=0.20, candidate=0.18),
    )
    calls: list[bool] = []

    def fake_locked_test(**kwargs: Any) -> dict[str, Any]:
        include_delayed = bool(kwargs["include_delayed_prefix"])
        calls.append(include_delayed)
        result: dict[str, Any] = {
            "selected_seed": int(kwargs["seed_summary"]["seed"]),
            "metrics": {"event_mae": 0.14 if include_delayed else 0.13},
            "artifacts": {},
        }
        if include_delayed:
            result["delayed_prefix"] = {
                "horizon_metrics": [{"event_equal_mae": 9.0}],
                "artifacts": {},
            }
        return result

    monkeypatch.setattr(campaign, "_evaluate_locked_test", fake_locked_test)

    summary = campaign.run_internal(output_root=tmp_path, resume=False)

    assert calls == [False, True]
    assert summary["status"] == "complete"
    assert summary["candidate_gate"] == {
        "passed": True,
        "event_mae": 0.14,
        "maximum_exclusive": campaign.INTERNAL_EVENT_MAE_MAXIMUM,
    }
    assert summary["frozen_test_diagnostic"] == {
        "baseline_event_mae": 0.13,
        "candidate_event_mae": 0.14,
        "candidate_minus_baseline": pytest.approx(0.01),
        "candidate_improved": False,
        "used_for_selection_or_gate": False,
    }
    assert "delayed_prefix" not in summary["variants"]["baseline"]
    assert "delayed_prefix" in summary["variants"]["candidate"]


@pytest.mark.parametrize(
    ("candidate_mae", "expected_passed"),
    [(0.13, True), (0.14, False)],
)
def test_phase32_internal_evaluates_only_winner_and_applies_both_test_gates(
    candidate_mae: float,
    expected_passed: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def profile(selected_seed: int) -> dict[str, Any]:
        return {
            "selection": {"selected_seed": selected_seed},
            "seeds": {
                str(seed): {
                    "seed": seed,
                    campaign.VALIDATION_METRIC: 0.10 + seed / 1000.0,
                }
                for seed in campaign.SEEDS
            },
        }

    _mark_stage(
        tmp_path,
        "train",
        {
            "status": "complete",
            "variant_axis": campaign.LOSS_WEIGHT_PROFILE_AXIS,
            "selected_candidate_variant": "w10",
            "variants": {
                "baseline": profile(17),
                "w01": profile(42),
                "w10": profile(73),
            },
        },
    )
    profile_selection = {
        "winner_profile": "w10",
        "winner_paired_mean_delta": -0.01,
        "winner_paired_median_delta": -0.01,
        "winner_improved_seed_count": 2,
        "winner_improved_seed_majority": True,
    }
    monkeypatch.setattr(campaign, "candidate_validation_improves", lambda _s: True)
    monkeypatch.setattr(
        campaign,
        "_validated_loss_weight_profile_selection",
        lambda _s: profile_selection,
    )
    calls: list[tuple[int, bool]] = []

    def fake_locked_test(**kwargs: Any) -> dict[str, Any]:
        seed = int(kwargs["seed_summary"]["seed"])
        delayed = bool(kwargs["include_delayed_prefix"])
        calls.append((seed, delayed))
        return {
            "selected_seed": seed,
            "metrics": {
                "event_mae": candidate_mae if delayed else 0.137,
            },
            "artifacts": {},
        }

    monkeypatch.setattr(campaign, "_evaluate_locked_test", fake_locked_test)

    summary = campaign.run_internal(output_root=tmp_path, resume=False)

    assert calls == [(17, False), (73, True)]
    assert summary["selected_candidate_variant"] == "w10"
    assert summary["variants"]["candidate"]["source_variant"] == "w10"
    assert summary["candidate_gate"]["passed"] is expected_passed
    assert summary["candidate_gate"][
        "phase27_incumbent_maximum_exclusive"
    ] == campaign.PHASE27_INCUMBENT_TEST_EVENT_MAE
    assert summary["candidate_gate"]["absolute_maximum_exclusive"] == 0.15


@pytest.mark.parametrize(
    (
        "improved_majority",
        "mean_delta",
        "median_delta",
        "candidate_validation",
    ),
    [
        (False, -0.01, -0.01, 0.10),
        (True, 0.001, -0.01, 0.10),
        (True, -0.01, 0.0, 0.10),
        (True, -0.01, -0.01, campaign.PHASE27_INCUMBENT_VALIDATION),
    ],
)
def test_phase32_validation_failures_never_read_locked_test(
    improved_majority: bool,
    mean_delta: float,
    median_delta: float,
    candidate_validation: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def profile(selected_seed: int, validation: float) -> dict[str, Any]:
        return {
            "selection": {"selected_seed": selected_seed},
            "seeds": {
                str(seed): {
                    "seed": seed,
                    campaign.VALIDATION_METRIC: (
                        validation if seed == selected_seed else validation + 0.01
                    ),
                }
                for seed in campaign.SEEDS
            },
        }

    train_summary = {
        "status": "complete",
        "variant_axis": campaign.LOSS_WEIGHT_PROFILE_AXIS,
        "selected_candidate_variant": "w01",
        "variants": {
            "baseline": profile(
                campaign.PHASE27_INCUMBENT_SELECTED_SEED,
                campaign.PHASE27_INCUMBENT_VALIDATION,
            ),
            "w01": profile(17, candidate_validation),
            "w10": profile(42, 0.12),
        },
    }
    _mark_stage(tmp_path, "train", train_summary)
    selection = {
        "winner_profile": "w01",
        "winner_paired_mean_delta": mean_delta,
        "winner_paired_median_delta": median_delta,
        "winner_improved_seed_count": 2 if improved_majority else 1,
        "winner_improved_seed_majority": improved_majority,
    }
    monkeypatch.setattr(
        campaign,
        "_validated_phase27_incumbent_reproduction",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        campaign,
        "_validated_loss_weight_profile_selection",
        lambda _summary: selection,
    )
    locked_test_calls = 0

    def forbidden_locked_test(**_kwargs: Any) -> dict[str, Any]:
        nonlocal locked_test_calls
        locked_test_calls += 1
        raise AssertionError("Phase32 locked test must remain unread")

    monkeypatch.setattr(campaign, "_evaluate_locked_test", forbidden_locked_test)

    summary = campaign.run_internal(output_root=tmp_path, resume=False)

    assert locked_test_calls == 0
    assert summary["status"] == "candidate_validation_gate_failed"
    assert summary["validation_gate"]["passed"] is False
    assert summary["test_evaluated"] is False
    assert "variants" not in summary


@pytest.mark.parametrize(
    ("failed_condition", "candidate_deltas"),
    [
        (
            "improved_or_equal_seed_majority",
            {17: -0.02, 42: 0.02, 73: 0.02},
        ),
        (
            "paired_mean_nonpositive",
            {17: -0.02, 42: -0.01, 73: 0.10},
        ),
        (
            "paired_median_nonpositive",
            {17: -0.001, 42: 0.001, 73: 0.001},
        ),
        (
            "candidate_selected_at_or_below_phase27_incumbent",
            {17: 0.02, 42: -0.02, 73: -0.08},
        ),
    ],
)
def test_phase33_each_validation_condition_failure_keeps_test_locked(
    failed_condition: str,
    candidate_deltas: dict[int, float],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_summary, _calls = _run_phase33_train_with_deltas(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
        candidate_deltas=candidate_deltas,
    )
    compatibility = train_summary["shape_loss_validation_compatibility"]
    assert compatibility["passed"] is False
    assert compatibility["conditions"][failed_condition] is False
    locked_test_calls = 0

    def forbidden_locked_test(**_kwargs: Any) -> dict[str, Any]:
        nonlocal locked_test_calls
        locked_test_calls += 1
        raise AssertionError("Phase33 locked test must remain unread")

    monkeypatch.setattr(campaign, "_evaluate_locked_test", forbidden_locked_test)

    summary = campaign.run_internal(output_root=tmp_path, resume=False)

    assert locked_test_calls == 0
    assert summary["status"] == "candidate_validation_gate_failed"
    assert summary["validation_gate"]["passed"] is False
    assert summary["validation_gate"]["conditions"][failed_condition] is False
    assert summary["test_evaluated"] is False
    assert summary["external_evaluated"] is False
    assert "variants" not in summary


@pytest.mark.parametrize(
    ("candidate_event_mae", "expected_status"),
    [
        (0.14, "complete"),
        (0.15, "candidate_internal_gate_failed"),
    ],
)
def test_phase33_internal_uses_selected_seeds_and_only_the_point15_gate(
    candidate_event_mae: float,
    expected_status: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_deltas = {
        17: 0.0,
        42: 0.0,
        73: 0.10 - campaign.PHASE27_INCUMBENT_VALIDATION_BY_SEED[73],
    }
    train_summary, _calls = _run_phase33_train_with_deltas(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
        candidate_deltas=candidate_deltas,
    )
    assert train_summary["shape_loss_validation_compatibility"]["passed"] is True
    calls: list[tuple[int, bool]] = []

    def fake_locked_test(**kwargs: Any) -> dict[str, Any]:
        seed = int(kwargs["seed_summary"]["seed"])
        delayed = bool(kwargs["include_delayed_prefix"])
        calls.append((seed, delayed))
        return {
            "selected_seed": seed,
            "metrics": {
                "event_mae": candidate_event_mae if delayed else 0.137,
            },
            "artifacts": {},
        }

    monkeypatch.setattr(campaign, "_evaluate_locked_test", fake_locked_test)

    summary = campaign.run_internal(output_root=tmp_path, resume=False)

    assert calls == [(17, False), (73, True)]
    assert summary["status"] == expected_status
    assert summary["candidate_gate"] == {
        "passed": expected_status == "complete",
        "event_mae": candidate_event_mae,
        "maximum_exclusive": campaign.INTERNAL_EVENT_MAE_MAXIMUM,
    }
    assert summary["variants"]["baseline"]["source_variant"] == "baseline"
    assert summary["variants"]["baseline"]["loss_weights"] == {
        "lambda_MSE": 1.0,
        "lambda_synth": 0.5,
        "lambda_mag": 1.0,
        "lambda_shape": 0.1,
    }
    assert summary["variants"]["candidate"]["source_variant"] == "candidate"
    assert summary["variants"]["candidate"]["loss_weights"] == {
        "lambda_MSE": 1.0,
        "lambda_synth": 0.5,
        "lambda_mag": 1.0,
        "lambda_shape": 0.0,
    }
    diagnostic = summary["frozen_test_diagnostic"]
    assert diagnostic["phase27_incumbent_event_mae"] == (
        campaign.PHASE27_INCUMBENT_TEST_EVENT_MAE
    )
    assert diagnostic["used_for_selection_or_gate"] is False
    assert summary["external_evaluated"] is False


@pytest.mark.parametrize(
    ("tampered_field", "tampered_value"),
    [
        ("selected_seed", 42),
        ("source_variant", "baseline"),
        (
            "loss_weights",
            {
                "lambda_MSE": 1.0,
                "lambda_synth": 0.5,
                "lambda_mag": 1.0,
                "lambda_shape": 0.1,
            },
        ),
    ],
)
def test_phase33_internal_resume_rejects_seed_variant_or_weight_tampering(
    tampered_field: str,
    tampered_value: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_deltas = {
        17: 0.0,
        42: 0.0,
        73: 0.10 - campaign.PHASE27_INCUMBENT_VALIDATION_BY_SEED[73],
    }
    train_summary, _calls = _run_phase33_train_with_deltas(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
        candidate_deltas=candidate_deltas,
    )
    assert train_summary["shape_loss_validation_compatibility"]["passed"] is True
    results = {
        "baseline": {
            "selected_seed": 17,
            "source_variant": "baseline",
            "loss_weights": {
                "lambda_MSE": 1.0,
                "lambda_synth": 0.5,
                "lambda_mag": 1.0,
                "lambda_shape": 0.1,
            },
            "metrics": {"event_mae": 0.137},
            "artifacts": {},
        },
        "candidate": {
            "selected_seed": 73,
            "source_variant": "candidate",
            "loss_weights": {
                "lambda_MSE": 1.0,
                "lambda_synth": 0.5,
                "lambda_mag": 1.0,
                "lambda_shape": 0.0,
            },
            "metrics": {"event_mae": 0.14},
            "artifacts": {},
            "delayed_prefix": {"artifacts": {}},
        },
    }
    results["candidate"][tampered_field] = tampered_value

    def fake_load_completed(
        path: Path,
        *,
        require_delayed_prefix: bool,
    ) -> dict[str, Any]:
        assert require_delayed_prefix is (path.name == "candidate")
        return copy.deepcopy(results[path.name])

    locked_test_calls = 0

    def forbidden_locked_test(**_kwargs: Any) -> dict[str, Any]:
        nonlocal locked_test_calls
        locked_test_calls += 1
        raise AssertionError("tampered Phase33 resume must not re-evaluate test")

    monkeypatch.setattr(campaign, "_load_completed_evaluation", fake_load_completed)
    monkeypatch.setattr(campaign, "_evaluate_locked_test", forbidden_locked_test)

    with pytest.raises(ValueError, match="resumed Phase33 internal evaluation changed"):
        campaign.run_internal(output_root=tmp_path, resume=True)

    assert locked_test_calls == 0


def test_phase33_completed_internal_resume_revalidates_and_never_reevaluates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, initial_calls = _complete_phase33_internal(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert initial_calls == [(17, False), (73, True)]
    resume_evaluation_calls = 0

    def forbidden_locked_test(**_kwargs: Any) -> dict[str, Any]:
        nonlocal resume_evaluation_calls
        resume_evaluation_calls += 1
        raise AssertionError("valid completed Phase33 internal must not re-evaluate")

    monkeypatch.setattr(campaign, "_evaluate_locked_test", forbidden_locked_test)

    resumed = campaign.run_internal(output_root=tmp_path, resume=True)

    assert resumed == completed
    assert resume_evaluation_calls == 0


@pytest.mark.parametrize(
    "tamper_kind",
    [
        "train_paired",
        "validation_gate",
        "candidate_gate",
        "selected_seed",
        "source_variant",
        "loss_weights",
        "artifact",
        "status",
        "test_evaluated",
        "external_evaluated",
    ],
)
def test_phase33_completed_internal_resume_rejects_provenance_tampering(
    tamper_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _completed, _initial_calls = _complete_phase33_internal(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
    )
    internal_summary_path = tmp_path / "internal" / "summary.json"
    internal_summary = json.loads(internal_summary_path.read_text(encoding="utf-8"))

    if tamper_kind == "train_paired":
        train_summary_path = tmp_path / "train" / "summary.json"
        train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
        train_summary["shape_loss_validation_compatibility"]["conditions"][
            "paired_mean_nonpositive"
        ] = False
        _write_json(train_summary_path, train_summary)
        internal_summary["train_summary"] = _artifact(train_summary_path)
    elif tamper_kind == "validation_gate":
        internal_summary["validation_gate"]["paired_mean_delta"] = 1.0
    elif tamper_kind == "candidate_gate":
        internal_summary["candidate_gate"]["event_mae"] = 0.01
    elif tamper_kind == "selected_seed":
        internal_summary["variants"]["candidate"]["selected_seed"] = 42
    elif tamper_kind == "source_variant":
        internal_summary["variants"]["candidate"]["source_variant"] = "baseline"
    elif tamper_kind == "loss_weights":
        internal_summary["variants"]["candidate"]["loss_weights"][
            "lambda_shape"
        ] = 0.1
    elif tamper_kind == "artifact":
        internal_summary["variants"]["candidate"]["artifacts"]["metrics"][
            "sha256"
        ] = "wrong"
    elif tamper_kind == "status":
        internal_summary["status"] = "candidate_internal_gate_failed"
    elif tamper_kind == "test_evaluated":
        internal_summary["test_evaluated"] = False
    elif tamper_kind == "external_evaluated":
        internal_summary["external_evaluated"] = True
    else:  # pragma: no cover - exhaustive parametrization guard
        raise AssertionError(f"unknown tamper kind: {tamper_kind}")
    _write_json(internal_summary_path, internal_summary)
    resume_evaluation_calls = 0

    def forbidden_locked_test(**_kwargs: Any) -> dict[str, Any]:
        nonlocal resume_evaluation_calls
        resume_evaluation_calls += 1
        raise AssertionError("tampered completed Phase33 internal must be rejected")

    monkeypatch.setattr(campaign, "_evaluate_locked_test", forbidden_locked_test)

    with pytest.raises(ValueError):
        campaign.run_internal(output_root=tmp_path, resume=True)

    assert resume_evaluation_calls == 0


def test_phase33_external_stage_is_always_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mark_stage(
        tmp_path,
        "train",
        {
            "status": "complete",
            "variant_axis": campaign.SHAPE_LOSS_WEIGHT_AXIS,
        },
    )
    _mark_stage(
        tmp_path,
        "internal",
        {
            "status": "complete",
            "candidate_gate": {"passed": True},
        },
    )
    external_calls = 0

    def forbidden_external(**_kwargs: Any) -> dict[str, Any]:
        nonlocal external_calls
        external_calls += 1
        raise AssertionError("Phase33 external benchmark must remain closed")

    monkeypatch.setattr(campaign, "_evaluate_external_threshold", forbidden_external)

    with pytest.raises(RuntimeError, match="Phase33 external evaluation.*closed"):
        campaign.run_external(
            output_root=tmp_path,
            event_root=tmp_path / "external-events",
            resume=False,
        )

    assert external_calls == 0


def test_phase33_completed_external_resume_is_still_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mark_stage(
        tmp_path,
        "train",
        {
            "status": "complete",
            "variant_axis": campaign.SHAPE_LOSS_WEIGHT_AXIS,
        },
    )
    _mark_stage(
        tmp_path,
        "internal",
        {
            "status": "complete",
            "candidate_gate": {"passed": True},
        },
    )
    _mark_stage(
        tmp_path,
        "external",
        {
            "status": "complete",
            "external_waveform_grid_schema_version": (
                campaign.EXTERNAL_WAVEFORM_GRID_SCHEMA_VERSION
            ),
        },
    )
    external_calls = 0

    def forbidden_external(**_kwargs: Any) -> dict[str, Any]:
        nonlocal external_calls
        external_calls += 1
        raise AssertionError("completed Phase33 external must not be reused")

    monkeypatch.setattr(campaign, "_evaluate_external_threshold", forbidden_external)

    with pytest.raises(RuntimeError, match="Phase33 external evaluation.*closed"):
        campaign.run_external(
            output_root=tmp_path,
            event_root=tmp_path / "external-events",
            resume=True,
        )

    assert external_calls == 0


def test_internal_validation_gate_fails_before_locked_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mark_stage(
        tmp_path,
        "train",
        _selected_train_summary(baseline=0.18, candidate=0.19),
    )
    calls = 0

    def forbidden_test(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("locked test must remain unread")

    monkeypatch.setattr(campaign, "_evaluate_locked_test", forbidden_test)

    summary = campaign.run_internal(output_root=tmp_path, resume=False)

    assert calls == 0
    assert summary["status"] == "candidate_validation_gate_failed"
    assert summary["validation_gate"]["passed"] is False
    assert summary["test_evaluated"] is False


@pytest.mark.parametrize(
    "variant_axis",
    [
        campaign.EVENT_BALANCE_EXPONENT_AXIS,
        campaign.MOMENT_LINEAR_SKIP_AXIS,
        campaign.MOMENT_HEAD_DROPOUT_AXIS,
    ],
)
def test_frozen_incumbent_gate_fails_before_locked_test(
    variant_axis: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_rows = {
        str(seed): {
            "seed": seed,
            campaign.VALIDATION_METRIC: (
                campaign.PHASE27_INCUMBENT_VALIDATION_BY_SEED[seed]
            ),
            "checkpoint": {
                "sha256": (
                    campaign.PHASE27_INCUMBENT_CHECKPOINT_SHA256_BY_SEED[seed]
                )
            },
        }
        for seed in campaign.SEEDS
    }
    candidate_rows = {
        "17": {
            "seed": 17,
            campaign.VALIDATION_METRIC: campaign.PHASE27_INCUMBENT_VALIDATION,
        },
        "42": {"seed": 42, campaign.VALIDATION_METRIC: 0.13},
        "73": {"seed": 73, campaign.VALIDATION_METRIC: 0.20},
    }
    reproduction = campaign.phase29_incumbent_reproduction(
        {seed: baseline_rows[str(seed)] for seed in campaign.SEEDS}
    )
    reproduction_path = tmp_path / "incumbent_reproduction.json"
    _write_json(reproduction_path, reproduction)
    train_summary = {
        "status": "complete",
        "variant_axis": variant_axis,
        "incumbent_reproduction": {
            **reproduction,
            "artifact": _artifact(reproduction_path),
        },
        "variants": {
            "baseline": {
                "seeds": baseline_rows,
                "selection": {
                    "selected_seed": 17,
                    "selection_metric": campaign.VALIDATION_METRIC,
                    "ensemble_used": False,
                },
            },
            "candidate": {
                "seeds": candidate_rows,
                "selection": {
                    "selected_seed": 17,
                    "selection_metric": campaign.VALIDATION_METRIC,
                    "ensemble_used": False,
                },
            },
        },
    }
    _mark_stage(tmp_path, "train", train_summary)
    calls = 0

    def forbidden_test(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("Phase29 locked test must remain unread")

    monkeypatch.setattr(campaign, "_evaluate_locked_test", forbidden_test)

    summary = campaign.run_internal(output_root=tmp_path, resume=False)

    assert calls == 0
    assert summary["status"] == "candidate_validation_gate_failed"
    assert summary["validation_gate"] == {
        "passed": False,
        "baseline_reproduction": campaign.PHASE27_INCUMBENT_VALIDATION,
        "frozen_incumbent": campaign.PHASE27_INCUMBENT_VALIDATION,
        "frozen_incumbent_seed": campaign.PHASE27_INCUMBENT_SELECTED_SEED,
        "candidate": campaign.PHASE27_INCUMBENT_VALIDATION,
        "metric": campaign.VALIDATION_METRIC,
        "rule": "candidate < frozen Phase27 incumbent",
        "incumbent_reproduction_verified": True,
    }
    assert summary["test_evaluated"] is False
    assert summary["external_evaluated"] is False


def test_external_gate_fails_before_external_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mark_stage(
        tmp_path,
        "internal",
        {
            "status": "candidate_internal_gate_failed",
            "candidate_gate": {"passed": False},
        },
    )
    calls = 0

    def forbidden_external(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("external data must remain unread")

    monkeypatch.setattr(
        campaign,
        "_evaluate_external_threshold",
        forbidden_external,
    )

    with pytest.raises(RuntimeError, match="passed internal candidate gate"):
        campaign.run_external(
            output_root=tmp_path,
            event_root=tmp_path / "events",
            resume=False,
        )

    assert calls == 0


def test_external_resume_rejects_obsolete_waveform_grid_schema(
    tmp_path: Path,
) -> None:
    _mark_stage(
        tmp_path,
        "external",
        {"status": "cm0_coverage_gate_failed"},
    )

    with pytest.raises(ValueError, match="obsolete waveform-grid schema"):
        campaign.run_external(
            output_root=tmp_path,
            event_root=tmp_path / "events",
            resume=True,
        )


def test_external_coverage_gate_requires_exact_frozen_event_identity() -> None:
    labels = [{"event": "A"}, {"event": "B"}, {"event": "C"}]

    complete = campaign._external_coverage_gate(
        {"event_names": ["C", "A", "B"], "event_metrics": {"count": 3}},
        labels,
    )
    wrong_identity = campaign._external_coverage_gate(
        {"event_names": ["A", "B", "D"], "event_metrics": {"count": 3}},
        labels,
    )

    assert complete["passed"] is True
    assert complete["observed_event_names"] == ["A", "B", "C"]
    assert wrong_identity["passed"] is False
    assert wrong_identity["missing_event_names"] == ["C"]
    assert wrong_identity["unexpected_event_names"] == ["D"]


def test_external_threshold_summary_records_full_window_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_station = tmp_path / "raw_station.csv"
    raw_event = tmp_path / "raw_event.csv"
    raw_station.write_text("event,station\nE,S\n", encoding="utf-8")
    raw_event.write_text("event\nE\n", encoding="utf-8")

    from src.evaluation import evaluate_unseen

    monkeypatch.setattr(
        evaluate_unseen,
        "evaluate_unseen_events",
        lambda **_kwargs: {
            "station_rows": [
                {
                    "event": "E",
                    "station": "S",
                    "mw_pred": 7.0,
                    "waveform_start_sec": 0.4,
                    "waveform_phase_adjusted": True,
                    "waveform_max_interpolation_gap_sec": 0.0,
                    "waveform_slot_count": 200,
                    "waveform_valid_sample_count": 199,
                    "waveform_masked_sample_count": 1,
                    "waveform_valid_fraction": 0.995,
                    "waveform_raw_dt_sec": 1.0,
                    "waveform_baseline_source": "pre_event",
                    "waveform_available_before_sec": 200.0,
                }
            ],
            "event_rows": [{"event": "E", "mw_pred_median": 7.0}],
            "station_csv": raw_station,
            "event_csv": raw_event,
        },
    )
    monkeypatch.setattr(
        campaign,
        "pair_prediction_rows",
        lambda rows, *_args, **_kwargs: list(rows),
    )
    monkeypatch.setattr(
        campaign,
        "summarize_paired_rows",
        lambda rows, **_kwargs: {"selected": {"count": len(rows), "mae": 0.0}},
    )

    summary = campaign._evaluate_external_threshold(
        model_dir=tmp_path / "model",
        event_dirs=[],
        label_rows=[{"event": "E"}],
        threshold_cm=0.0,
        output_dir=tmp_path / "external",
        resume=False,
    )

    assert summary["observation_horizon_sec"] == 200.0
    assert summary["release_time_sec"] == 205.0
    assert summary["waveform_grid"] == {
        "schema_version": campaign.EXTERNAL_WAVEFORM_GRID_SCHEMA_VERSION,
        "mode": "raw_sample_phase_no_interpolation",
        "configured_start_sec": 0.0,
        "raw_input_available_before_sec": 200.0,
        "minimum_start_sec": 0.4,
        "maximum_start_sec": 0.4,
        "slot_count": 200,
        "minimum_valid_sample_count": 199,
        "maximum_masked_sample_count": 1,
        "minimum_valid_fraction": 0.995,
        "raw_dt_sec_minimum": 1.0,
        "raw_dt_sec_maximum": 1.0,
        "baseline_source_counts": {"pre_event": 1},
        "maximum_last_sample_sec": pytest.approx(199.4),
        "fir_boundary_mode": "zero_padded_same",
        "fir_nominal_lookahead_sec": 3.0,
        "nominal_maximum_filter_support_sec": pytest.approx(202.4),
        "release_margin_sec": pytest.approx(2.6),
        "max_interpolation_gap_sec": 0.0,
        "phase_adjusted_station_count": 1,
        "station_count": 1,
    }
    assert summary["event_names"] == ["E"]
    persisted = json.loads(
        (tmp_path / "external" / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted["observation_horizon_sec"] == 200.0
    assert persisted["release_time_sec"] == 205.0
    station_rows = campaign._read_csv(
        tmp_path / "external" / "station_predictions_usgs.csv"
    )
    event_rows = campaign._read_csv(
        tmp_path / "external" / "event_predictions_usgs.csv"
    )
    for row in [*station_rows, *event_rows]:
        assert float(row["observation_horizon_sec"]) == 200.0
        assert float(row["release_time_sec"]) == 205.0


def test_cli_dispatches_exactly_one_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_internal(**_kwargs: Any) -> dict[str, Any]:
        calls.append("internal")
        return {"stage": "internal", "status": "complete"}

    monkeypatch.setattr(campaign, "run_internal", fake_internal)

    exit_code = campaign.main(
        ["--stage", "internal", "--output-root", str(tmp_path)]
    )

    assert exit_code == 0
    assert calls == ["internal"]


def _mark_phase34_smoke(output_root: Path) -> None:
    results: dict[str, Any] = {}
    for profile_id, weights in campaign.LOSS_TERM_ABLATION_PROFILES.items():
        results[profile_id] = {
            device: {
                "passed": True,
                "device": device,
                "loss": 1.0,
                "metrics": {"L_total": 1.0},
                "parameter_count": 1_010_850,
                "loss_weights": weights,
                "loss_term_ablation_profile": profile_id,
                "loss_term_ablation_profile_sha256": (
                    campaign.LOSS_TERM_ABLATION_PROFILE_SHA256
                ),
                "event_balance_estimator": (
                    campaign.INVERSE_COUNT_FULL_DATA_ESTIMATOR
                ),
                "event_balance_exponent": 1.0,
                "sample_weights_exercised": True,
            }
            for device in ("cpu", "cuda")
        }
    _mark_stage(
        output_root,
        "smoke",
        {
            "status": "complete",
            "variant_axis": campaign.LOSS_TERM_ABLATION_AXIS,
            "test_evaluated": False,
            "external_evaluated": False,
            "loss_term_ablation_profiles": campaign.LOSS_TERM_ABLATION_PROFILES,
            "loss_term_ablation_profile_sha256": (
                campaign.LOSS_TERM_ABLATION_PROFILE_SHA256
            ),
            "results": results,
        },
    )


def _phase34_split_manifest(seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "protocol": "within_event_station",
        "train_record_count": 1788,
        "validation_record_count": 385,
        "test_record_count": 385,
        "assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[seed],
        "sample_keys": [],
        "per_event_station_counts": {},
    }


def _run_phase34_train_with_deltas(
    *,
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    deltas: dict[str, dict[int, float]],
) -> tuple[dict[str, Any], list[tuple[str, int]]]:
    config_path = output_root / "frozen_config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(PHASE34_CONFIG_PATH), sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = output_root / "dataset_manifest.csv"
    manifest_path.write_text("event,station\n", encoding="utf-8")
    splits: dict[str, Any] = {}
    for seed in campaign.SEEDS:
        split_path = output_root / f"split_{seed}.json"
        _write_json(split_path, _phase34_split_manifest(seed))
        splits[str(seed)] = {"manifest": _artifact(split_path)}
    _mark_stage(
        output_root,
        "preflight",
        {
            "status": "complete",
            "source_data": {"sha256": campaign.EXPECTED_SOURCE_SHA256},
            "frozen_config": _artifact(config_path),
            "dataset_manifest": _artifact(manifest_path),
            "splits": splits,
        },
    )
    _mark_phase34_smoke(output_root)
    calls: list[tuple[str, int]] = []

    def fake_train_one_seed(**kwargs: Any) -> dict[str, Any]:
        profile_id = str(kwargs["variant"])
        seed = int(kwargs["seed"])
        calls.append((profile_id, seed))
        metric = campaign.PHASE33_FULL_THREE_VALIDATION_BY_SEED[seed]
        if profile_id != "baseline":
            metric += deltas[profile_id][seed]
        seed_root = Path(kwargs["seed_root"])
        seed_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = seed_root / "best_model.pth"
        checkpoint_path.write_text(f"{profile_id}-{seed}-best", encoding="utf-8")
        last_checkpoint_path = seed_root / "last_state.pth"
        last_checkpoint_path.write_text(
            f"{profile_id}-{seed}-last", encoding="utf-8"
        )
        training_log_path = seed_root / "training.csv"
        training_log_path.write_text("epoch\n", encoding="utf-8")
        split_path = seed_root / "split.json"
        _write_json(split_path, _phase34_split_manifest(seed))
        sampling_path = seed_root / "sampling.json"
        _write_json(sampling_path, _phase27_sampling_manifest(seed))
        commit = campaign.current_git_commit(campaign.PROJECT_ROOT)
        run_manifest_path = seed_root / "run_manifest.json"
        _write_json(run_manifest_path, {"git_commit": commit, "git_dirty": False})
        runtime = campaign._runtime_config(
            kwargs["config"],
            run_root=seed_root,
            seed=seed,
            dataset_manifest=Path(kwargs["dataset_manifest"]),
        )
        config_artifact_path = seed_root / "config.yaml"
        config_artifact_path.write_text(
            yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8"
        )
        checkpoint = _artifact(checkpoint_path)
        if profile_id == "baseline":
            monkeypatch.setitem(
                campaign.PHASE33_FULL_THREE_CHECKPOINT_SHA256_BY_SEED,
                seed,
                checkpoint["sha256"],
            )
        return {
            "variant": profile_id,
            "seed": seed,
            "git_commit": commit,
            "loss_weights": campaign.loss_weights_from_config(kwargs["config"]),
            "loss_term_ablation_profile": profile_id,
            "loss_term_ablation_profile_sha256": (
                campaign.LOSS_TERM_ABLATION_PROFILE_SHA256
            ),
            "split_assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[seed],
            campaign.VALIDATION_METRIC: metric,
            "checkpoint": checkpoint,
            "last_checkpoint": _artifact(last_checkpoint_path),
            "config": _artifact(config_artifact_path),
            "split": _artifact(split_path),
            "training_log": _artifact(training_log_path),
            "run_manifest": _artifact(run_manifest_path),
            "sampling": _artifact(sampling_path),
        }

    monkeypatch.setattr(campaign, "_train_one_seed", fake_train_one_seed)
    monkeypatch.setattr(
        campaign,
        "_evaluate_locked_test",
        lambda **_kwargs: pytest.fail("Phase34 must never evaluate locked test"),
    )
    monkeypatch.setattr(
        campaign,
        "_evaluate_external_threshold",
        lambda **_kwargs: pytest.fail("Phase34 must never evaluate external data"),
    )
    return campaign.run_train(output_root=output_root, resume=False), calls


def test_phase34_config_is_exact_nested_three_loss_ablation() -> None:
    phase33_candidate = campaign.build_variant_configs(
        _config(PHASE33_CONFIG_PATH)
    )["candidate"]
    phase34_config = _config(PHASE34_CONFIG_PATH)
    campaign.validate_formal_config(phase34_config)
    variants = campaign.build_variant_configs(phase34_config)

    assert campaign.variant_axis_from_config(phase34_config) == (
        campaign.LOSS_TERM_ABLATION_AXIS
    )
    assert campaign._profile_table_sha256(
        campaign.loss_term_ablation_profiles_from_config(phase34_config)
    ) == campaign.LOSS_TERM_ABLATION_PROFILE_SHA256
    phase33_candidate.pop("campaign")
    phase34_baseline = copy.deepcopy(variants["baseline"])
    phase34_baseline.pop("campaign")
    assert phase34_baseline == phase33_candidate
    for profile_id, weight_name in campaign.LOSS_TERM_ABLATION_REMOVED_WEIGHT.items():
        assert campaign._config_diff_paths(
            variants["baseline"], variants[profile_id]
        ) == {".".join(campaign.LOSS_WEIGHT_PATHS[weight_name])}
        assert variants[profile_id]["training"]["stf_rate_loss"][weight_name] == 0.0


@pytest.mark.parametrize(
    ("tamper_kind", "match"),
    [
        ("profile_hash", "profile_sha256 changed"),
        ("baseline_shape", "baseline.lambda_shape changed"),
        ("extra_zero", "no_mse.lambda_synth changed"),
        ("negative", "finite and nonnegative"),
    ],
)
def test_phase34_profile_table_tampering_is_rejected(
    tamper_kind: str,
    match: str,
) -> None:
    config = _config(PHASE34_CONFIG_PATH)
    ablation = config["campaign"]["loss_term_ablation"]
    if tamper_kind == "profile_hash":
        ablation["profile_sha256"] = "wrong"
    elif tamper_kind == "baseline_shape":
        ablation["profiles"]["baseline"]["lambda_shape"] = 0.1
    elif tamper_kind == "extra_zero":
        ablation["profiles"]["no_mse"]["lambda_synth"] = 0.0
    elif tamper_kind == "negative":
        ablation["profiles"]["no_mag"]["lambda_mag"] = -1.0
    else:  # pragma: no cover
        raise AssertionError(tamper_kind)

    with pytest.raises(ValueError, match=match):
        campaign.loss_term_ablation_profiles_from_config(config)


def test_phase34_runs_full_three_then_each_ablation_for_all_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deltas = {
        profile_id: {17: 0.01, 42: 0.02, 73: 0.03}
        for profile_id in campaign.LOSS_TERM_ABLATION_REMOVED_WEIGHT
    }
    summary, calls = _run_phase34_train_with_deltas(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
        deltas=deltas,
    )

    assert calls == [
        (profile_id, seed)
        for profile_id in campaign.LOSS_TERM_ABLATION_PROFILES
        for seed in campaign.SEEDS
    ]
    assert summary["full_three_reproduction"]["passed"] is True
    assert summary["loss_term_ablation_evidence"][
        "all_three_terms_supported"
    ] is True
    assert summary["winning_ablation_selected"] is False
    assert summary["ensemble_used"] is False
    assert summary["test_evaluated"] is False
    assert summary["external_evaluated"] is False
    assert "selected_candidate_variant" not in summary
    validated = campaign._validated_loss_term_ablation_campaign(summary)
    assert validated["all_three_terms_supported"] is True


def test_phase34_completed_train_resume_rejects_preflight_axis_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deltas = {
        profile_id: {17: 0.01, 42: 0.02, 73: 0.03}
        for profile_id in campaign.LOSS_TERM_ABLATION_REMOVED_WEIGHT
    }
    _run_phase34_train_with_deltas(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
        deltas=deltas,
    )

    preflight_path = tmp_path / "preflight" / "summary.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    frozen_config_path = Path(preflight["frozen_config"]["path"])
    frozen_config_path.write_text(
        yaml.safe_dump(_config(PHASE33_CONFIG_PATH), sort_keys=False),
        encoding="utf-8",
    )
    preflight["frozen_config"] = _artifact(frozen_config_path)
    _write_json(preflight_path, preflight)

    with pytest.raises(ValueError, match="campaign axes differ"):
        campaign.run_train(output_root=tmp_path, resume=True)


@pytest.mark.parametrize("tamper_kind", ["config", "split"])
def test_phase34_completed_train_resume_rejects_seed_provenance_tampering(
    tamper_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deltas = {
        profile_id: {17: 0.01, 42: 0.02, 73: 0.03}
        for profile_id in campaign.LOSS_TERM_ABLATION_REMOVED_WEIGHT
    }
    _run_phase34_train_with_deltas(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
        deltas=deltas,
    )

    summary_path = tmp_path / "train" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed_row = summary["variants"]["no_mse"]["seeds"]["17"]
    artifact_path = Path(seed_row[tamper_kind]["path"])
    if tamper_kind == "config":
        persisted = yaml.safe_load(artifact_path.read_text(encoding="utf-8"))
        persisted["training"]["learning_rate"] = 0.5
        artifact_path.write_text(
            yaml.safe_dump(persisted, sort_keys=False), encoding="utf-8"
        )
    else:
        persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
        persisted["sample_keys"] = ["changed-with-frozen-hash-field"]
        _write_json(artifact_path, persisted)
    seed_row[tamper_kind] = _artifact(artifact_path)
    _write_json(summary_path, summary)

    with pytest.raises(ValueError):
        campaign.run_train(output_root=tmp_path, resume=True)


@pytest.mark.parametrize("mismatch", ["validation_metric", "checkpoint_sha256"])
def test_phase34_full_three_drift_stops_before_first_ablation(
    mismatch: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "frozen_config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(PHASE34_CONFIG_PATH), sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "dataset_manifest.csv"
    manifest_path.write_text("event,station\n", encoding="utf-8")
    splits: dict[str, Any] = {}
    for seed in campaign.SEEDS:
        split_path = tmp_path / f"split_{seed}.json"
        _write_json(split_path, {"seed": seed})
        splits[str(seed)] = {"manifest": _artifact(split_path)}
    _mark_stage(
        tmp_path,
        "preflight",
        {
            "status": "complete",
            "source_data": {"sha256": campaign.EXPECTED_SOURCE_SHA256},
            "frozen_config": _artifact(config_path),
            "dataset_manifest": _artifact(manifest_path),
            "splits": splits,
        },
    )
    _mark_phase34_smoke(tmp_path)
    calls: list[tuple[str, int]] = []

    def fake_train_one_seed(**kwargs: Any) -> dict[str, Any]:
        profile_id = str(kwargs["variant"])
        seed = int(kwargs["seed"])
        calls.append((profile_id, seed))
        metric = campaign.PHASE33_FULL_THREE_VALIDATION_BY_SEED[seed]
        checkpoint = campaign.PHASE33_FULL_THREE_CHECKPOINT_SHA256_BY_SEED[seed]
        if seed == 42 and mismatch == "validation_metric":
            metric += 1.0e-12
        if seed == 42 and mismatch == "checkpoint_sha256":
            checkpoint = "wrong"
        return {
            "variant": profile_id,
            "seed": seed,
            campaign.VALIDATION_METRIC: metric,
            "checkpoint": {"sha256": checkpoint},
        }

    monkeypatch.setattr(campaign, "_train_one_seed", fake_train_one_seed)

    with pytest.raises(ValueError, match="refusing to train any loss-term"):
        campaign.run_train(output_root=tmp_path, resume=False)

    assert calls == [("baseline", seed) for seed in campaign.SEEDS]


def test_phase34_evidence_uses_strict_positive_paired_boundaries() -> None:
    baseline = campaign.PHASE33_FULL_THREE_VALIDATION_BY_SEED
    profile_deltas = {
        "no_mse": {17: 0.0, 42: 0.0, 73: 0.0},
        "no_synth": {17: 0.01, 42: 0.02, 73: 0.03},
        "no_mag": {17: 0.01, 42: 0.02, 73: 0.03},
    }
    variants: dict[str, Any] = {
        "baseline": {
            "seeds": {
                str(seed): {
                    "seed": seed,
                    campaign.VALIDATION_METRIC: baseline[seed],
                }
                for seed in campaign.SEEDS
            }
        }
    }
    for profile_id, deltas in profile_deltas.items():
        variants[profile_id] = {
            "seeds": {
                str(seed): {
                    "seed": seed,
                    campaign.VALIDATION_METRIC: baseline[seed] + deltas[seed],
                }
                for seed in campaign.SEEDS
            }
        }

    evidence = campaign.select_loss_term_ablation_evidence(variants)

    no_mse = evidence["profiles"]["no_mse"]
    assert no_mse["paired_mean_delta"] == pytest.approx(0.0)
    assert no_mse["paired_median_delta"] == pytest.approx(0.0)
    assert no_mse["conditions"] == {
        "worsened_seed_majority": False,
        "paired_mean_positive": False,
        "paired_median_positive": False,
    }
    assert no_mse["supports_retained_term"] is False
    assert evidence["profiles"]["no_synth"]["supports_retained_term"] is True
    assert evidence["profiles"]["no_mag"]["supports_retained_term"] is True
    assert evidence["all_three_terms_supported"] is False
    assert evidence["winning_ablation_selected"] is False


def test_phase34_seed_resume_binds_profile_hash_and_weights(tmp_path: Path) -> None:
    seed = 17
    commit = "phase34-commit"
    profile_id = "no_mse"
    weights = campaign.LOSS_TERM_ABLATION_PROFILES[profile_id]
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    training_log_path = tmp_path / "training.csv"
    training_log_path.write_text("epoch\n", encoding="utf-8")
    run_manifest_path = tmp_path / "run_manifest.json"
    _write_json(run_manifest_path, {"git_commit": commit, "git_dirty": False})
    split_path = tmp_path / "split.json"
    _write_json(
        split_path,
        {
            "seed": seed,
            "protocol": "within_event_station",
            "train_record_count": 1788,
            "validation_record_count": 385,
            "test_record_count": 385,
            "assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[seed],
        },
    )
    sampling_path = tmp_path / "sampling.json"
    _write_json(sampling_path, _phase27_sampling_manifest(seed))
    expected_config = {
        "training": {
            "random_seed": seed,
            "event_balance_estimator": campaign.INVERSE_COUNT_FULL_DATA_ESTIMATOR,
            "stf_rate_loss": weights,
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(expected_config, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "variant": profile_id,
        "seed": seed,
        "git_commit": commit,
        "event_balance_exponent": 1.0,
        "loss_weights": weights,
        "loss_term_ablation_profile": profile_id,
        "loss_term_ablation_profile_sha256": (
            campaign.LOSS_TERM_ABLATION_PROFILE_SHA256
        ),
        "split_assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[seed],
        campaign.VALIDATION_METRIC: 0.2,
        "checkpoint": _artifact(checkpoint_path),
        "config": _artifact(config_path),
        "split": _artifact(split_path),
        "training_log": _artifact(training_log_path),
        "run_manifest": _artifact(run_manifest_path),
        "sampling": _artifact(sampling_path),
    }
    arguments = {
        "require_sampling": True,
        "expected_loss_weights": weights,
        "expected_loss_term_ablation_profile": profile_id,
        "expected_loss_term_ablation_profile_sha256": (
            campaign.LOSS_TERM_ABLATION_PROFILE_SHA256
        ),
        "expected_variant": profile_id,
        "expected_seed": seed,
        "expected_split_assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[seed],
        "expected_config": expected_config,
        "expected_git_commit": commit,
    }

    assert campaign._seed_summary_is_valid(summary, **arguments)
    assert not campaign._seed_summary_is_valid(
        {**summary, "loss_term_ablation_profile": "no_synth"}, **arguments
    )
    assert not campaign._seed_summary_is_valid(
        {**summary, "loss_term_ablation_profile_sha256": "wrong"}, **arguments
    )
    changed_weights = dict(weights)
    changed_weights["lambda_MSE"] = 1.0
    assert not campaign._seed_summary_is_valid(
        {**summary, "loss_weights": changed_weights}, **arguments
    )


@pytest.mark.parametrize("resume", [False, True])
def test_phase34_internal_is_always_closed(
    resume: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mark_stage(
        tmp_path,
        "train",
        {"status": "complete", "variant_axis": campaign.LOSS_TERM_ABLATION_AXIS},
    )
    if resume:
        _mark_stage(tmp_path, "internal", {"status": "complete"})
    monkeypatch.setattr(
        campaign,
        "_evaluate_locked_test",
        lambda **_kwargs: pytest.fail("Phase34 locked test must remain closed"),
    )

    with pytest.raises(RuntimeError, match="Phase34.*internal evaluation is closed"):
        campaign.run_internal(output_root=tmp_path, resume=resume)


@pytest.mark.parametrize("resume", [False, True])
def test_phase34_external_is_always_closed(
    resume: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mark_stage(
        tmp_path,
        "train",
        {"status": "complete", "variant_axis": campaign.LOSS_TERM_ABLATION_AXIS},
    )
    if resume:
        _mark_stage(
            tmp_path,
            "external",
            {
                "status": "complete",
                "external_waveform_grid_schema_version": (
                    campaign.EXTERNAL_WAVEFORM_GRID_SCHEMA_VERSION
                ),
            },
        )
    monkeypatch.setattr(
        campaign,
        "_evaluate_external_threshold",
        lambda **_kwargs: pytest.fail("Phase34 external must remain closed"),
    )

    with pytest.raises(RuntimeError, match="Phase34.*external evaluation is closed"):
        campaign.run_external(
            output_root=tmp_path,
            event_root=tmp_path / "external-events",
            resume=resume,
        )


def test_phase38_config_changes_only_station_global_synth_polarity() -> None:
    phase33_candidate = campaign.build_variant_configs(
        _config(PHASE33_CONFIG_PATH)
    )["candidate"]
    phase38_config = _config(PHASE38_CONFIG_PATH)

    campaign.validate_formal_config(phase38_config)
    variants = campaign.build_variant_configs(phase38_config)

    assert campaign.variant_axis_from_config(phase38_config) == (
        campaign.SYNTH_POLARITY_MODE_AXIS
    )
    assert campaign._config_diff_paths(
        variants["baseline"], variants["candidate"]
    ) == {"training.stf_rate_loss.synth_polarity_mode"}
    assert campaign.synth_polarity_mode_from_config(variants["baseline"]) == "signed"
    assert (
        campaign.synth_polarity_mode_from_config(variants["candidate"])
        == "global_invariant"
    )
    assert campaign.loss_weights_from_config(variants["baseline"]) == {
        "lambda_MSE": 1.0,
        "lambda_synth": 0.5,
        "lambda_mag": 1.0,
        "lambda_shape": 0.0,
    }
    phase33_candidate.pop("campaign")
    phase38_baseline = copy.deepcopy(variants["baseline"])
    phase38_baseline.pop("campaign")
    phase38_baseline["training"]["stf_rate_loss"].pop("synth_polarity_mode")
    assert phase38_baseline == phase33_candidate


def test_explicit_synth_polarity_mode_is_rejected_for_older_campaigns() -> None:
    config = _config(PHASE33_CONFIG_PATH)
    config["training"]["stf_rate_loss"][
        "synth_polarity_mode"
    ] = "global_invariant"

    with pytest.raises(ValueError, match="reserved for Phase38"):
        campaign.validate_formal_config(config)


def test_phase38_frozen_no_synth_resolver_audits_temporary_phase34_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "phase34-source"
    source_root.mkdir()
    source_summary, _calls = _run_phase34_train_with_deltas(
        output_root=source_root,
        monkeypatch=monkeypatch,
        deltas={
            "no_mse": {seed: 0.01 for seed in campaign.SEEDS},
            "no_synth": {17: 0.03, 42: -0.01, 73: -0.02},
            "no_mag": {seed: 0.10 for seed in campaign.SEEDS},
        },
    )
    no_synth_rows = source_summary["variants"]["no_synth"]["seeds"]
    for seed in campaign.SEEDS:
        row = no_synth_rows[str(seed)]
        row["epoch"] = 1
        metric = float(row[campaign.VALIDATION_METRIC])
        training_log_path = Path(row["training_log"]["path"])
        training_log_path.write_text(
            f"Epoch,{campaign.VALIDATION_METRIC}\n1,{metric!r}\n",
            encoding="utf-8",
        )
        row["training_log"] = _artifact(training_log_path)
        monkeypatch.setitem(
            campaign.PHASE34_NO_SYNTH_VALIDATION_BY_SEED,
            seed,
            metric,
        )
        monkeypatch.setitem(
            campaign.PHASE34_NO_SYNTH_CHECKPOINT_SHA256_BY_SEED,
            seed,
            row["checkpoint"]["sha256"],
        )
    source_summary_path = source_root / "train" / "summary.json"
    _write_json(source_summary_path, source_summary)
    monkeypatch.setattr(
        campaign,
        "PHASE34_NO_SYNTH_TRAIN_SUMMARY_PATH",
        source_summary_path,
    )
    monkeypatch.setattr(
        campaign,
        "PHASE34_NO_SYNTH_TRAIN_SUMMARY_SHA256",
        campaign.sha256_file(source_summary_path),
    )
    monkeypatch.setattr(
        campaign,
        "PHASE34_SOURCE_GIT_COMMIT",
        source_summary["git_commit"],
    )
    monkeypatch.setattr(campaign, "PHASE34_NO_SYNTH_SELECTED_SEED", 42)

    provenance = campaign.frozen_phase34_no_synth_provenance()

    assert provenance["not_retrained"] is True
    assert provenance["selected_seed"] == 42
    assert provenance["effective_synth_polarity_mode"] == "signed"
    assert set(provenance["seeds"]) == {"17", "42", "73"}

    selected_checkpoint = Path(no_synth_rows["42"]["checkpoint"]["path"])
    selected_checkpoint.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifacts changed"):
        campaign.frozen_phase34_no_synth_provenance()


def _mark_phase38_smoke(output_root: Path) -> None:
    modes = campaign.VARIANT_AXES[campaign.SYNTH_POLARITY_MODE_AXIS]
    results = {
        variant: {
            device: {
                "passed": True,
                "device": device,
                "loss": 1.0,
                "metrics": {"L_total": 1.0},
                "parameter_count": 1_010_850,
                "loss_weights": campaign.LOSS_TERM_ABLATION_PROFILES["baseline"],
                "synth_polarity_mode": mode,
                "event_balance_estimator": (
                    campaign.INVERSE_COUNT_FULL_DATA_ESTIMATOR
                ),
                "event_balance_exponent": 1.0,
                "sample_weights_exercised": True,
            }
            for device in ("cpu", "cuda")
        }
        for variant, mode in modes.items()
    }
    _mark_stage(
        output_root,
        "smoke",
        {
            "status": "complete",
            "created_at_utc": "2026-07-26T00:00:00+00:00",
            "git_commit": campaign.current_git_commit(campaign.PROJECT_ROOT),
            "variant_axis": campaign.SYNTH_POLARITY_MODE_AXIS,
            "test_evaluated": False,
            "external_evaluated": False,
            "synth_polarity_modes": modes,
            "results": results,
        },
    )


def _install_fake_phase38_frozen_no_synth(
    *,
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = output_root / "frozen_phase34_train_summary.json"
    _write_json(source_path, {"stage": "train", "status": "complete"})
    provenance_seeds: dict[str, Any] = {}
    source_rows: dict[int, dict[str, Any]] = {}
    for seed in campaign.SEEDS:
        checkpoint_path = output_root / f"frozen_no_synth_{seed}.pth"
        checkpoint_path.write_text(f"no-synth-{seed}", encoding="utf-8")
        checkpoint = _artifact(checkpoint_path)
        provenance_seeds[str(seed)] = {
            campaign.VALIDATION_METRIC: (
                campaign.PHASE34_NO_SYNTH_VALIDATION_BY_SEED[seed]
            ),
            "checkpoint": checkpoint,
            "split_assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[seed],
        }
        source_rows[seed] = {
            "variant": "no_synth",
            "seed": seed,
            campaign.VALIDATION_METRIC: (
                campaign.PHASE34_NO_SYNTH_VALIDATION_BY_SEED[seed]
            ),
            "checkpoint": checkpoint,
        }
    provenance = {
        "source_campaign": "Phase34 loss-term ablation",
        "source_variant": "no_synth",
        "source_train_summary": _artifact(source_path),
        "source_git_commit": campaign.PHASE34_SOURCE_GIT_COMMIT,
        "loss_weights": dict(campaign.LOSS_TERM_ABLATION_PROFILES["no_synth"]),
        "effective_synth_polarity_mode": "signed",
        "polarity_mode_is_objectively_inactive": True,
        "not_retrained": True,
        "selected_seed": campaign.PHASE34_NO_SYNTH_SELECTED_SEED,
        "selection_metric": campaign.VALIDATION_METRIC,
        "ensemble_used": False,
        "seeds": provenance_seeds,
    }
    monkeypatch.setattr(
        campaign,
        "frozen_phase34_no_synth_provenance",
        lambda: copy.deepcopy(provenance),
    )
    monkeypatch.setattr(
        campaign,
        "_frozen_phase34_no_synth_seed_summary",
        lambda seed: copy.deepcopy(source_rows[int(seed)]),
    )


def _run_phase38_train(
    *,
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_values: dict[int, float],
    baseline_metric_deltas: dict[int, float] | None = None,
    baseline_checkpoint_mismatch_seed: int | None = None,
    resume: bool = False,
) -> tuple[dict[str, Any], list[tuple[str, int]]]:
    _install_fake_phase38_frozen_no_synth(
        output_root=output_root,
        monkeypatch=monkeypatch,
    )
    baseline_metric_deltas = baseline_metric_deltas or {
        seed: 0.0 for seed in campaign.SEEDS
    }
    config_path = output_root / "frozen_config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(PHASE38_CONFIG_PATH), sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = output_root / "dataset_manifest.csv"
    manifest_path.write_text("event,station\n", encoding="utf-8")
    splits: dict[str, Any] = {}
    for seed in campaign.SEEDS:
        split_path = output_root / f"split_{seed}.json"
        _write_json(split_path, _phase34_split_manifest(seed))
        splits[str(seed)] = {"manifest": _artifact(split_path)}
    _mark_stage(
        output_root,
        "preflight",
        {
            "status": "complete",
            "source_data": {"sha256": campaign.EXPECTED_SOURCE_SHA256},
            "frozen_config": _artifact(config_path),
            "dataset_manifest": _artifact(manifest_path),
            "splits": splits,
        },
    )
    _mark_phase38_smoke(output_root)
    calls: list[tuple[str, int]] = []

    def fake_train_one_seed(**kwargs: Any) -> dict[str, Any]:
        variant = str(kwargs["variant"])
        seed = int(kwargs["seed"])
        calls.append((variant, seed))
        seed_root = Path(kwargs["seed_root"])
        seed_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = seed_root / "best_model.pth"
        checkpoint_path.write_text(f"{variant}-{seed}-best", encoding="utf-8")
        last_checkpoint_path = seed_root / "last_state.pth"
        last_checkpoint_path.write_text(f"{variant}-{seed}-last", encoding="utf-8")
        training_log_path = seed_root / "training.csv"
        training_log_path.write_text("epoch\n", encoding="utf-8")
        split_path = seed_root / "split.json"
        _write_json(split_path, _phase34_split_manifest(seed))
        sampling_path = seed_root / "sampling.json"
        _write_json(sampling_path, _phase27_sampling_manifest(seed))
        commit = campaign.current_git_commit(campaign.PROJECT_ROOT)
        run_manifest_path = seed_root / "run_manifest.json"
        _write_json(run_manifest_path, {"git_commit": commit, "git_dirty": False})
        runtime = campaign._runtime_config(
            kwargs["config"],
            run_root=seed_root,
            seed=seed,
            dataset_manifest=Path(kwargs["dataset_manifest"]),
        )
        config_artifact_path = seed_root / "config.yaml"
        config_artifact_path.write_text(
            yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8"
        )
        checkpoint = _artifact(checkpoint_path)
        if variant == "baseline" and seed != baseline_checkpoint_mismatch_seed:
            monkeypatch.setitem(
                campaign.PHASE33_FULL_THREE_CHECKPOINT_SHA256_BY_SEED,
                seed,
                checkpoint["sha256"],
            )
        metric = (
            campaign.PHASE33_FULL_THREE_VALIDATION_BY_SEED[seed]
            + baseline_metric_deltas[seed]
            if variant == "baseline"
            else candidate_values[seed]
        )
        return {
            "variant": variant,
            "seed": seed,
            "git_commit": commit,
            "loss_weights": campaign.loss_weights_from_config(kwargs["config"]),
            "synth_polarity_mode": campaign.synth_polarity_mode_from_config(
                kwargs["config"]
            ),
            "split_assignment_sha256": campaign.EXPECTED_SPLIT_SHA256[seed],
            campaign.VALIDATION_METRIC: metric,
            "checkpoint": checkpoint,
            "last_checkpoint": _artifact(last_checkpoint_path),
            "config": _artifact(config_artifact_path),
            "split": _artifact(split_path),
            "training_log": _artifact(training_log_path),
            "run_manifest": _artifact(run_manifest_path),
            "sampling": _artifact(sampling_path),
        }

    monkeypatch.setattr(campaign, "_train_one_seed", fake_train_one_seed)
    monkeypatch.setattr(
        campaign,
        "_evaluate_locked_test",
        lambda **_kwargs: pytest.fail("Phase38 train must not evaluate locked test"),
    )
    monkeypatch.setattr(
        campaign,
        "_evaluate_external_threshold",
        lambda **_kwargs: pytest.fail("Phase38 external must remain closed"),
    )
    return campaign.run_train(output_root=output_root, resume=resume), calls


def _phase38_evidence_inputs(
    *,
    candidate_values: dict[int, float],
    signed_deltas: dict[int, float],
    frozen_no_synth_deltas: dict[int, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    def rows(values: dict[int, float]) -> dict[str, Any]:
        return {
            str(seed): {
                "seed": seed,
                campaign.VALIDATION_METRIC: values[seed],
            }
            for seed in campaign.SEEDS
        }

    signed_values = {
        seed: candidate_values[seed] - signed_deltas[seed]
        for seed in campaign.SEEDS
    }
    frozen_values = {
        seed: candidate_values[seed] - frozen_no_synth_deltas[seed]
        for seed in campaign.SEEDS
    }
    return (
        {
            "baseline": {"seeds": rows(signed_values)},
            "candidate": {"seeds": rows(candidate_values)},
        },
        {"seeds": rows(frozen_values)},
    )


def test_phase38_validation_evidence_selects_seed_from_candidate_only() -> None:
    candidate_values = {17: 0.11, 42: 0.09, 73: 0.10}
    variants, frozen = _phase38_evidence_inputs(
        candidate_values=candidate_values,
        signed_deltas={seed: -0.01 for seed in campaign.SEEDS},
        frozen_no_synth_deltas={seed: -0.02 for seed in campaign.SEEDS},
    )

    evidence = campaign.select_synth_polarity_validation_evidence(
        variants, frozen
    )

    assert evidence["passed"] is True
    assert evidence["selected_candidate_seed"] == 42
    assert evidence["selected_candidate_validation"] == pytest.approx(0.09)
    assert evidence["numerical_zero_abs_tolerance"] == pytest.approx(1.0e-12)
    assert evidence["selection_uses_candidate_validation_only"] is True
    assert evidence["ensemble_used"] is False
    assert all(
        comparison["passed"]
        for comparison in evidence["comparisons"].values()
    )


@pytest.mark.parametrize("comparator", ["signed", "frozen_no_synth"])
@pytest.mark.parametrize(
    ("target_deltas", "expected_conditions"),
    [
        (
            {17: -0.10, 42: 0.0, 73: 0.0},
            {
                "improved_seed_majority": False,
                "paired_mean_negative": True,
                "paired_median_negative": False,
            },
        ),
        (
            {17: -0.01, 42: -0.01, 73: 0.10},
            {
                "improved_seed_majority": True,
                "paired_mean_negative": False,
                "paired_median_negative": True,
            },
        ),
        (
            {17: -0.125, 42: -0.125, 73: 0.25},
            {
                "improved_seed_majority": True,
                "paired_mean_negative": False,
                "paired_median_negative": True,
            },
        ),
    ],
)
def test_phase38_each_paired_validation_gate_failure_stops_the_candidate(
    comparator: str,
    target_deltas: dict[int, float],
    expected_conditions: dict[str, bool],
) -> None:
    passing_deltas = {seed: -0.10 for seed in campaign.SEEDS}
    variants, frozen = _phase38_evidence_inputs(
        candidate_values={17: 1.0, 42: 1.0, 73: 1.0},
        signed_deltas=(
            target_deltas if comparator == "signed" else passing_deltas
        ),
        frozen_no_synth_deltas=(
            target_deltas
            if comparator == "frozen_no_synth"
            else passing_deltas
        ),
    )

    evidence = campaign.select_synth_polarity_validation_evidence(
        variants, frozen
    )

    assert evidence["comparisons"][comparator]["conditions"] == (
        expected_conditions
    )
    assert evidence["comparisons"][comparator]["passed"] is False
    other = "frozen_no_synth" if comparator == "signed" else "signed"
    assert evidence["comparisons"][other]["passed"] is True
    assert evidence["passed"] is False


@pytest.mark.parametrize("comparator", ["signed", "frozen_no_synth"])
def test_phase38_paired_mean_and_median_zero_are_strict_failures(
    comparator: str,
) -> None:
    zero_deltas = {seed: 0.0 for seed in campaign.SEEDS}
    passing_deltas = {seed: -0.10 for seed in campaign.SEEDS}
    variants, frozen = _phase38_evidence_inputs(
        candidate_values={17: 1.0, 42: 1.0, 73: 1.0},
        signed_deltas=zero_deltas if comparator == "signed" else passing_deltas,
        frozen_no_synth_deltas=(
            zero_deltas
            if comparator == "frozen_no_synth"
            else passing_deltas
        ),
    )

    evidence = campaign.select_synth_polarity_validation_evidence(
        variants, frozen
    )

    comparison = evidence["comparisons"][comparator]
    assert comparison["paired_mean_delta"] == pytest.approx(0.0)
    assert comparison["paired_median_delta"] == pytest.approx(0.0)
    assert comparison["passed"] is False
    assert evidence["passed"] is False


def test_phase38_runs_only_signed_and_global_invariant_three_seed_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary, calls = _run_phase38_train(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
        candidate_values={17: 0.09, 42: 0.10, 73: 0.15},
    )

    assert calls == [
        (variant, seed)
        for variant in ("baseline", "candidate")
        for seed in campaign.SEEDS
    ]
    assert summary["full_three_reproduction"]["passed"] is True
    assert summary["frozen_no_synth_provenance"]["not_retrained"] is True
    assert summary["frozen_no_synth_provenance"]["selected_seed"] == 42
    assert summary["synth_polarity_validation_evidence"]["passed"] is True
    assert summary["selected_candidate_seed"] == 17
    assert summary["variants"]["candidate"]["selection"]["selected_seed"] == 17
    assert summary["variants"]["baseline"]["synth_polarity_mode"] == "signed"
    assert (
        summary["variants"]["candidate"]["synth_polarity_mode"]
        == "global_invariant"
    )
    assert summary["test_evaluated"] is False
    assert summary["external_evaluated"] is False
    assert summary["ensemble_used"] is False


@pytest.mark.parametrize("mismatch", ["validation_metric", "checkpoint_sha256"])
def test_phase38_signed_reproduction_drift_stops_before_candidate_training(
    mismatch: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments: dict[str, Any] = {}
    if mismatch == "validation_metric":
        arguments["baseline_metric_deltas"] = {17: 0.0, 42: 1.0e-12, 73: 0.0}
    else:
        arguments["baseline_checkpoint_mismatch_seed"] = 42

    with pytest.raises(ValueError, match="Phase38 signed baseline did not exactly"):
        _run_phase38_train(
            output_root=tmp_path,
            monkeypatch=monkeypatch,
            candidate_values={17: 0.09, 42: 0.10, 73: 0.15},
            **arguments,
        )

    assert not (tmp_path / "train" / "candidate").exists()


def test_phase38_completed_train_resume_revalidates_without_retraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary, calls = _run_phase38_train(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
        candidate_values={17: 0.09, 42: 0.10, 73: 0.15},
    )
    calls.clear()

    resumed = campaign.run_train(output_root=tmp_path, resume=True)

    assert resumed == summary
    assert calls == []


def test_phase38_completed_train_resume_rejects_preflight_axis_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_phase38_train(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
        candidate_values={17: 0.09, 42: 0.10, 73: 0.15},
    )
    preflight_path = tmp_path / "preflight" / "summary.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    frozen_config_path = Path(preflight["frozen_config"]["path"])
    frozen_config_path.write_text(
        yaml.safe_dump(_config(PHASE34_CONFIG_PATH), sort_keys=False),
        encoding="utf-8",
    )
    preflight["frozen_config"] = _artifact(frozen_config_path)
    _write_json(preflight_path, preflight)

    with pytest.raises(ValueError, match="campaign axes differ"):
        campaign.run_train(output_root=tmp_path, resume=True)


@pytest.mark.parametrize("tamper_kind", ["candidate_config", "frozen_provenance"])
def test_phase38_completed_train_resume_rejects_provenance_tampering(
    tamper_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_phase38_train(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
        candidate_values={17: 0.09, 42: 0.10, 73: 0.15},
    )
    summary_path = tmp_path / "train" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if tamper_kind == "candidate_config":
        seed_row = summary["variants"]["candidate"]["seeds"]["17"]
        config_path = Path(seed_row["config"]["path"])
        persisted_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        persisted_config["training"]["stf_rate_loss"][
            "synth_polarity_mode"
        ] = "signed"
        config_path.write_text(
            yaml.safe_dump(persisted_config, sort_keys=False), encoding="utf-8"
        )
        seed_row["config"] = _artifact(config_path)
    else:
        recorded = summary["frozen_no_synth_provenance"]
        provenance_path = Path(recorded["artifact"]["path"])
        persisted_provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        persisted_provenance["selected_seed"] = 17
        _write_json(provenance_path, persisted_provenance)
        recorded["selected_seed"] = 17
        recorded["artifact"] = _artifact(provenance_path)
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="Phase38"):
        campaign.run_train(output_root=tmp_path, resume=True)


def _phase38_fake_locked_test(
    *,
    event_mae_by_source_variant: dict[str, float],
    calls: list[tuple[str, int, bool]],
):
    def evaluate(**kwargs: Any) -> dict[str, Any]:
        seed_summary = kwargs["seed_summary"]
        source_variant = str(seed_summary["variant"])
        seed = int(seed_summary["seed"])
        include_delayed = bool(kwargs["include_delayed_prefix"])
        calls.append((source_variant, seed, include_delayed))
        output_dir = Path(kwargs["output_dir"])
        metrics = {
            "station_mae": 0.10,
            "station_rmse": 0.12,
            "station_bias": -0.01,
            "event_mae": event_mae_by_source_variant[source_variant],
            "event_rmse": 0.16,
            "event_bias": -0.02,
        }
        metrics_path = output_dir / "metrics.json"
        _write_json(metrics_path, metrics)
        station_path = output_dir / "station_predictions.csv"
        station_path.parent.mkdir(parents=True, exist_ok=True)
        station_path.write_text("event,station\n", encoding="utf-8")
        event_path = output_dir / "event_predictions.csv"
        event_path.write_text("event,mw_pred\n", encoding="utf-8")
        registry_path = output_dir / "result_registry.json"
        _write_json(registry_path, {"seed": seed})
        result: dict[str, Any] = {
            "selected_seed": seed,
            "metrics": metrics,
            "artifacts": {
                "metrics": _artifact(metrics_path),
                "station_predictions": _artifact(station_path),
                "event_predictions": _artifact(event_path),
                "result_registry": _artifact(registry_path),
            },
        }
        if include_delayed:
            delayed_dir = output_dir / "delayed_prefix"
            delayed_station_path = delayed_dir / "station_predictions.csv"
            delayed_station_path.parent.mkdir(parents=True, exist_ok=True)
            delayed_station_path.write_text("event,station\n", encoding="utf-8")
            delayed_event_path = delayed_dir / "event_predictions.csv"
            delayed_event_path.write_text("event,mw_pred\n", encoding="utf-8")
            unavailable_path = delayed_dir / "unavailable_stations.csv"
            unavailable_path.write_text("event,station\n", encoding="utf-8")
            horizon_metrics = [
                {
                    "observation_horizon_sec": float(horizon),
                    "event_equal_mae": event_mae_by_source_variant[source_variant],
                }
                for horizon in campaign.DEFAULT_HORIZONS_SEC
            ]
            horizon_path = delayed_dir / "horizon_metrics.json"
            _write_json(horizon_path, horizon_metrics)
            cohort = {
                "waveform_prefix_causal": True,
                "station_selection_causal": False,
            }
            cohort_path = delayed_dir / "cohort_contract.json"
            _write_json(cohort_path, cohort)
            result["delayed_prefix"] = {
                "horizons_sec": list(campaign.DEFAULT_HORIZONS_SEC),
                "horizon_metrics": horizon_metrics,
                "cohort": cohort,
                "artifacts": {
                    "station_predictions": _artifact(delayed_station_path),
                    "event_predictions": _artifact(delayed_event_path),
                    "unavailable_stations": _artifact(unavailable_path),
                    "horizon_metrics": _artifact(horizon_path),
                    "cohort_contract": _artifact(cohort_path),
                },
            }
        return result

    return evaluate


def test_phase38_failed_validation_gate_keeps_internal_test_locked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_phase38_train(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
        candidate_values=dict(campaign.PHASE33_FULL_THREE_VALIDATION_BY_SEED),
    )
    calls: list[tuple[str, int, bool]] = []
    monkeypatch.setattr(
        campaign,
        "_evaluate_locked_test",
        _phase38_fake_locked_test(
            event_mae_by_source_variant={
                "baseline": 0.10,
                "candidate": 0.10,
                "no_synth": 0.10,
            },
            calls=calls,
        ),
    )

    summary = campaign.run_internal(output_root=tmp_path, resume=False)

    assert calls == []
    assert summary["status"] == "candidate_validation_gate_failed"
    assert summary["validation_gate"]["passed"] is False
    assert summary["test_evaluated"] is False
    assert summary["external_evaluated"] is False


@pytest.mark.parametrize(
    ("candidate_event_mae", "expected_passed"),
    [(0.149999, True), (0.15, False)],
)
def test_phase38_internal_gate_is_frozen_absolute_threshold_only(
    candidate_event_mae: float,
    expected_passed: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_phase38_train(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
        candidate_values={17: 0.09, 42: 0.10, 73: 0.15},
    )
    calls: list[tuple[str, int, bool]] = []
    monkeypatch.setattr(
        campaign,
        "_evaluate_locked_test",
        _phase38_fake_locked_test(
            event_mae_by_source_variant={
                "baseline": 0.12,
                "candidate": candidate_event_mae,
                "no_synth": 0.11,
            },
            calls=calls,
        ),
    )

    summary = campaign.run_internal(output_root=tmp_path, resume=False)

    assert calls == [
        ("baseline", 17, False),
        ("candidate", 17, True),
        ("no_synth", campaign.PHASE34_NO_SYNTH_SELECTED_SEED, False),
    ]
    assert summary["candidate_gate"] == {
        "passed": expected_passed,
        "event_mae": candidate_event_mae,
        "maximum_exclusive": campaign.INTERNAL_EVENT_MAE_MAXIMUM,
        "comparators_used_for_selection_or_gate": False,
    }
    assert summary["status"] == (
        "complete" if expected_passed else "candidate_internal_gate_failed"
    )
    diagnostic = summary["frozen_test_diagnostic"]
    assert diagnostic["signed_event_mae"] == pytest.approx(0.12)
    assert diagnostic["frozen_no_synth_event_mae"] == pytest.approx(0.11)
    assert diagnostic["candidate_event_mae"] == pytest.approx(candidate_event_mae)
    assert diagnostic["used_for_selection_or_gate"] is False
    assert summary["test_evaluated"] is True
    assert summary["external_evaluated"] is False


@pytest.mark.parametrize("resume", [False, True])
def test_phase38_external_is_permanently_closed(
    resume: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mark_stage(
        tmp_path,
        "train",
        {"status": "complete", "variant_axis": campaign.SYNTH_POLARITY_MODE_AXIS},
    )
    if resume:
        _mark_stage(
            tmp_path,
            "external",
            {
                "status": "complete",
                "external_waveform_grid_schema_version": (
                    campaign.EXTERNAL_WAVEFORM_GRID_SCHEMA_VERSION
                ),
            },
        )
    calls = 0

    def forbidden_external(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("Phase38 external data must remain unread")

    monkeypatch.setattr(
        campaign,
        "_evaluate_external_threshold",
        forbidden_external,
    )

    with pytest.raises(RuntimeError, match="Phase38 external evaluation.*closed"):
        campaign.run_external(
            output_root=tmp_path,
            event_root=tmp_path / "external-events",
            resume=resume,
        )

    assert calls == 0


def _complete_phase38_internal(
    *,
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    _run_phase38_train(
        output_root=output_root,
        monkeypatch=monkeypatch,
        candidate_values={17: 0.09, 42: 0.10, 73: 0.15},
    )
    calls: list[tuple[str, int, bool]] = []
    monkeypatch.setattr(
        campaign,
        "_evaluate_locked_test",
        _phase38_fake_locked_test(
            event_mae_by_source_variant={
                "baseline": 0.12,
                "candidate": 0.14,
                "no_synth": 0.11,
            },
            calls=calls,
        ),
    )
    completed = campaign.run_internal(output_root=output_root, resume=False)
    assert len(calls) == 3
    return completed


def test_phase38_completed_internal_resume_revalidates_without_reevaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = _complete_phase38_internal(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
    )
    monkeypatch.setattr(
        campaign,
        "_evaluate_locked_test",
        lambda **_kwargs: pytest.fail(
            "valid completed Phase38 internal must not re-evaluate test"
        ),
    )

    resumed = campaign.run_internal(output_root=tmp_path, resume=True)

    assert resumed == completed


@pytest.mark.parametrize(
    "tamper_kind",
    ["candidate_mode", "candidate_seed", "frozen_source", "candidate_gate"],
)
def test_phase38_completed_internal_resume_rejects_coordinated_tampering(
    tamper_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _complete_phase38_internal(
        output_root=tmp_path,
        monkeypatch=monkeypatch,
    )
    summary_path = tmp_path / "internal" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if tamper_kind == "candidate_gate":
        summary["candidate_gate"]["passed"] = False
    else:
        variant = (
            "frozen_no_synth" if tamper_kind == "frozen_source" else "candidate"
        )
        evaluation_path = tmp_path / "internal" / variant / "summary.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if tamper_kind == "candidate_mode":
            evaluation["synth_polarity_mode"] = "signed"
        elif tamper_kind == "candidate_seed":
            evaluation["selected_seed"] = 42
        else:
            evaluation["frozen_source_train_summary"]["sha256"] = "wrong"
        summary["variants"][variant] = evaluation
        _write_json(evaluation_path, evaluation)
    _write_json(summary_path, summary)
    monkeypatch.setattr(
        campaign,
        "_evaluate_locked_test",
        lambda **_kwargs: pytest.fail(
            "tampered completed Phase38 internal must not re-evaluate test"
        ),
    )

    with pytest.raises(ValueError, match="completed Phase38 internal"):
        campaign.run_internal(output_root=tmp_path, resume=True)
