from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.experiments import run_manuscript_stf_iteration as campaign
from src.utils.provenance import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "manuscript_station_stf_usgs.yaml"
)


def _config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


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


def test_formal_config_and_variants_have_one_scientific_difference() -> None:
    config = _config()

    campaign.validate_formal_config(config)
    variants = campaign.build_variant_configs(config)

    assert variants["baseline"]["model"]["stf_output_parameterization"] == "direct"
    assert (
        variants["candidate"]["model"]["stf_output_parameterization"]
        == "moment_shape_factorized"
    )
    assert campaign._config_diff_paths(
        variants["baseline"],
        variants["candidate"],
    ) == {"model.stf_output_parameterization"}


def test_formal_config_rejects_a_second_scientific_change() -> None:
    variants = campaign.build_variant_configs(_config())
    changed = copy.deepcopy(variants["candidate"])
    changed["model"]["dropout"] = 0.1

    assert campaign._config_diff_paths(variants["baseline"], changed) == {
        "model.dropout",
        "model.stf_output_parameterization",
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


def test_train_stage_selects_from_validation_without_test_or_external(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "frozen_config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(), sort_keys=False),
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
    assert summary["variants"]["baseline"]["selection"]["selected_seed"] == 42
    assert summary["variants"]["candidate"]["selection"]["selected_seed"] == 73
    assert summary["variants"]["candidate"]["scientific_diff_from_baseline"] == [
        "model.stf_output_parameterization"
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
            "station_rows": [{"event": "E", "station": "S", "mw_pred": 7.0}],
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
