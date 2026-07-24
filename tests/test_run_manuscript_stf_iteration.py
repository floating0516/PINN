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
CONFIG_PATH = PHASE23_CONFIG_PATH


def _config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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
    section, key = campaign.VARIANT_AXIS_PATHS[expected_axis]
    assert variants["baseline"][section][key] == baseline_value
    assert variants["candidate"][section][key] == candidate_value
    assert campaign._config_diff_paths(
        variants["baseline"],
        variants["candidate"],
    ) == {".".join(campaign.VARIANT_AXIS_PATHS[expected_axis])}
    if expected_axis in {
        campaign.SCHEDULER_T0_AXIS,
        campaign.RADIAL_DYNAMIC_RANGE_STEM_AXIS,
        campaign.EVENT_BALANCED_SAMPLING_AXIS,
        campaign.EVENT_BALANCE_ESTIMATOR_AXIS,
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
    calls: list[tuple[str, int]] = []

    def fake_train_one_seed(**kwargs: Any) -> dict[str, Any]:
        variant = str(kwargs["variant"])
        seed = int(kwargs["seed"])
        calls.append((variant, seed))
        return {
            "variant": variant,
            "seed": seed,
            campaign.VALIDATION_METRIC: values[variant][seed],
        }

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
    assert summary["variants"]["baseline"]["selection"]["selected_seed"] == 42
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
    assert summary["variants"]["candidate"]["scientific_diff_from_baseline"] == [
        ".".join(campaign.VARIANT_AXIS_PATHS[expected_axis])
    ]


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
