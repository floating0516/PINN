from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.experiments import run_phase39_confirmatory_grouped_cv as campaign
from src.utils.provenance import sha256_file


def _samples() -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    station_counts = [8, 2, 7, 3, 6, 4, 5, 1, 9, 2]
    for event_index, station_count in enumerate(station_counts):
        for station_index in range(station_count):
            samples.append(
                {
                    "event": f"event-{event_index:02d}",
                    "station": f"station-{station_index:03d}",
                    "magnitude_catalog": 6.0 + 0.1 * event_index,
                }
            )
    return samples


def _base_config() -> dict:
    return {
        "campaign": {"variant_axis": "radiation_coefficient_contract"},
        "paths": {
            "output_dir": "old",
            "models_dir": "old/models",
            "logs_dir": "old/logs",
            "results_dir": "old/results",
        },
        "training": {
            "split_protocol": "within_event_station",
            "strict_within_event_split_audit": True,
            "random_seed": 42,
            "epochs": 200,
            "stf_rate_loss": {
                "lambda_synth": 0.5,
                "radiation_coefficient_contract": "glehman_scalar",
                "synth_polarity_mode": "global_invariant",
                "include_intermediate_field": False,
            },
        },
    }


def test_event_folds_are_deterministic_complete_and_mw_stratified() -> None:
    samples = _samples()

    first = campaign.build_event_folds(samples, n_folds=5)
    second = campaign.build_event_folds(list(reversed(samples)), n_folds=5)

    assert first == second
    assert len(first["events"]) == 10
    assert {row["event"] for row in first["events"]} == {
        f"event-{index:02d}" for index in range(10)
    }
    assert sum(row["n_stations"] for row in first["folds"]) == len(samples)
    assert len(first["assignment_sha256"]) == 64
    for start in range(0, 10, 5):
        assert {
            row["fold"] for row in first["events"][start : start + 5]
        } == set(range(5))


def test_outer_split_uses_next_fold_for_validation() -> None:
    samples = _samples()
    folds = campaign.build_event_folds(samples, n_folds=5)

    split = campaign.make_outer_split(samples, folds, outer_fold=3)
    event_by_index = [str(sample["event"]) for sample in samples]
    event_to_fold = {
        str(row["event"]): int(row["fold"]) for row in folds["events"]
    }

    assert {
        event_to_fold[event_by_index[index]] for index in split.test_indices
    } == {3}
    assert {
        event_to_fold[event_by_index[index]]
        for index in split.validation_indices
    } == {4}
    assert {
        event_to_fold[event_by_index[index]] for index in split.train_indices
    } == {0, 1, 2}


def test_arm_configs_change_only_lambda_synth() -> None:
    arms = campaign.build_arm_configs(_base_config())

    assert campaign.config_diff_paths(
        arms["phase39"], arms["no_synth"]
    ) == {"training.stf_rate_loss.lambda_synth"}
    assert arms["phase39"]["training"]["stf_rate_loss"]["lambda_synth"] == 0.5
    assert arms["no_synth"]["training"]["stf_rate_loss"]["lambda_synth"] == 0.0
    for config in arms.values():
        assert "campaign" not in config
        assert config["training"]["split_protocol"] == "grouped_event"
        assert config["training"]["strict_within_event_split_audit"] is False


def _paired_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold, event in enumerate(("A", "B", "C", "D", "E")):
        for seed in campaign.SEEDS:
            for arm, error in (("phase39", 0.1), ("no_synth", 0.2)):
                rows.append(
                    {
                        "event": event,
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "error_vs_catalog": error,
                    }
                )
    return rows


def test_paired_statistics_use_event_as_the_bootstrap_unit() -> None:
    result = campaign.paired_event_statistics(
        _paired_rows(),
        expected_events={"A", "B", "C", "D", "E"},
        seeds=campaign.SEEDS,
        n_bootstrap=2000,
        n_sign_flips=2000,
        seed=123,
    )

    assert result["n_events"] == 5
    assert result["mean_delta_mw"] == pytest.approx(-0.1)
    assert result["bootstrap_ci_upper_mw"] == pytest.approx(-0.1)
    assert set(result["seed_mean_delta_mw"]) == {"17", "42", "73"}
    assert all(value < 0.0 for value in result["seed_mean_delta_mw"].values())
    assert result["promotion_gate"]["passed"] is True
    assert len(result["event_deltas"]) == 5


def test_paired_statistics_reject_incomplete_oof_coverage() -> None:
    rows = _paired_rows()
    rows.pop()

    with pytest.raises(ValueError, match="OOF coverage"):
        campaign.paired_event_statistics(
            rows,
            expected_events={"A", "B", "C", "D", "E"},
            seeds=campaign.SEEDS,
            n_bootstrap=100,
            n_sign_flips=100,
            seed=123,
        )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_completed_run_is_skipped_only_after_hash_and_contract_validation(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "fold_0" / "phase39" / "seed_17"
    run_root.mkdir(parents=True)
    rows = [
        {
            "event": "A",
            "fold": 0,
            "seed": 17,
            "arm": "phase39",
            "error_vs_catalog": 0.1,
        }
    ]
    event_path = run_root / "event_oof.csv"
    station_path = run_root / "station_oof.csv"
    checkpoint_path = run_root / "best_model.pth"
    _write_csv(event_path, rows)
    _write_csv(station_path, rows)
    checkpoint_path.write_bytes(b"checkpoint")
    summary = {
        "status": "complete",
        "fold": 0,
        "arm": "phase39",
        "seed": 17,
        "split_assignment_sha256": "split-sha",
        "event_oof_path": str(event_path),
        "event_oof_sha256": sha256_file(event_path),
        "station_oof_path": str(station_path),
        "station_oof_sha256": sha256_file(station_path),
        "best_model_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    (run_root / "run_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    loaded = campaign.load_completed_run(
        run_root,
        fold=0,
        arm="phase39",
        seed=17,
        split_assignment_sha256="split-sha",
        expected_events={"A"},
    )
    assert loaded == rows

    checkpoint_path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        campaign.load_completed_run(
            run_root,
            fold=0,
            arm="phase39",
            seed=17,
            split_assignment_sha256="split-sha",
            expected_events={"A"},
        )
