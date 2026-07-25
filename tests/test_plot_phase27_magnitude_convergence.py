from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.plotting.plot_phase27_magnitude_convergence import (
    EXPECTED_GIT_COMMIT,
    EXPECTED_HORIZONS,
    _jittered_catalog_positions,
    build_event_convergence_rows,
    generate_bundle,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture_run(root: Path) -> Path:
    run = root / "phase27"
    selected = run / "internal" / "candidate" / "results" / "selected_seed_17"
    prefix = run / "internal" / "candidate" / "results" / "delayed_prefix"
    magnitudes = [
        *[6.0 + 0.16 * index for index in range(6)],
        *[7.0 + 0.05 * index for index in range(17)],
        *[8.1 + 0.10 * index for index in range(7)],
    ]
    station_counts = [12] * 29 + [37]
    prediction_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    for index, (mw, station_count) in enumerate(zip(magnitudes, station_counts)):
        event = f"Event{index:02d}"
        final_error = 0.20 if index in {0, 8, 24} else (-0.05 if index % 2 else 0.05)
        for horizon in EXPECTED_HORIZONS:
            early_deficit = (200.0 - horizon) / 180.0 * (0.45 + 0.30 * (mw >= 8.0))
            prediction = mw + final_error - early_deficit
            error = prediction - mw
            prediction_rows.append(
                {
                    "event": event,
                    "observation_horizon_sec": horizon,
                    "release_time_sec": horizon + 5.0,
                    "mw_pred_median": prediction,
                    "mw_catalog": mw,
                    "error": error,
                    "station_count": station_count,
                }
            )
            if horizon == 200.0:
                final_rows.append(
                    {
                        "event": event,
                        "mw_pred_median": prediction,
                        "mw_catalog": mw,
                        "error_vs_catalog": error,
                        "n_stations": station_count,
                    }
                )
    _write_csv(prefix / "event_predictions.csv", prediction_rows)
    _write_csv(selected / "event_summary.csv", final_rows)

    horizon_metrics: list[dict[str, object]] = []
    for horizon in EXPECTED_HORIZONS:
        errors = np.asarray(
            [
                float(row["error"])
                for row in prediction_rows
                if float(row["observation_horizon_sec"]) == horizon
            ]
        )
        horizon_metrics.append(
            {
                "observation_horizon_sec": horizon,
                "release_time_sec": horizon + 5.0,
                "event_count": 30,
                "event_equal_mae": float(np.mean(np.abs(errors))),
                "event_equal_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "event_equal_bias": float(np.mean(errors)),
            }
        )
    _write_json(prefix / "horizon_metrics.json", horizon_metrics)
    _write_json(
        prefix / "cohort_contract.json",
        {
            "cohort": "processed radial peak over the full 200 s record >= 2 cm",
            "end_to_end_causal": False,
            "radial_peak_min_cm": 2.0,
            "station_selection_causal": False,
            "waveform_prefix_causal": True,
        },
    )

    final_errors = np.asarray([abs(float(row["error_vs_catalog"])) for row in final_rows])
    signed_errors = np.asarray([float(row["error_vs_catalog"]) for row in final_rows])
    metrics = {
        "event_bias": float(np.mean(signed_errors)),
        "event_count": 30,
        "event_mae": float(np.mean(final_errors)),
        "event_rmse": float(np.sqrt(np.mean(np.square(signed_errors)))),
        "reference": "catalog",
        "station_bias": 0.01,
        "station_count": 385,
        "station_mae": 0.10,
        "station_rmse": 0.15,
    }
    _write_json(selected / "metrics.json", metrics)
    _write_json(
        selected / "result_registry.json",
        {
            "checkpoint": {"path": "/tmp/best_model.pth", "sha256": "a" * 64},
            "primary_reference": "catalog",
            "split_protocol": "within_event_station",
        },
    )
    _write_json(
        run / "train" / "summary.json",
        {"status": "complete", "git_commit": EXPECTED_GIT_COMMIT},
    )
    _write_json(
        run / "train" / "candidate" / "selection.json",
        {
            "candidates": {"17": 0.11, "42": 0.13, "73": 0.20},
            "ensemble_used": False,
            "selected_seed": 17,
            "selection_metric": "validation_event_mae_catalog",
        },
    )
    _write_json(
        run / "internal" / "summary.json",
        {
            "status": "complete",
            "validation_gate": {"candidate": 0.11, "passed": True},
            "candidate_gate": {"event_mae": metrics["event_mae"], "passed": True},
            "variants": {"candidate": {"selected_seed": 17}},
        },
    )
    return run


def test_event_convergence_distinguishes_first_stable_and_censored() -> None:
    rows: list[dict[str, object]] = []
    cases = {
        "returns": (7.2, (0.10, 0.20, 0.10)),
        "leaves": (8.2, (0.20, 0.10, 0.20)),
    }
    for event, (mw, errors) in cases.items():
        for horizon, error in zip((20.0, 40.0, 60.0), errors):
            rows.append(
                {
                    "event": event,
                    "observation_horizon_sec": horizon,
                    "release_time_sec": horizon + 5.0,
                    "mw_pred_median": mw + error,
                    "mw_catalog": mw,
                    "error": error,
                    "abs_error": abs(error),
                    "station_count": 2,
                    "magnitude_group": "mw_7_to_lt_8" if mw < 8 else "mw_ge_8",
                }
            )

    output = {row["event"]: row for row in build_event_convergence_rows(rows)}
    assert output["returns"]["first_within_0p15_observation_sec"] == 20.0
    assert output["returns"]["stable_within_0p15_observation_sec"] == 60.0
    assert output["returns"]["stable_accuracy_right_censored"] is False
    assert output["leaves"]["first_within_0p15_observation_sec"] == 40.0
    assert output["leaves"]["stable_within_0p15_observation_sec"] is None
    assert output["leaves"]["stable_accuracy_right_censored"] is True

    duplicate_rows = [
        {
            "event": "one",
            "mw_catalog": 7.8,
            "stable_within_0p15_observation_sec": None,
        },
        {
            "event": "two",
            "mw_catalog": 7.8,
            "stable_within_0p15_observation_sec": None,
        },
        {
            "event": "three",
            "mw_catalog": 7.8,
            "stable_within_0p15_observation_sec": None,
        },
    ]
    positions = _jittered_catalog_positions(
        duplicate_rows, field="stable_within_0p15_observation_sec"
    )
    assert len(set(positions.values())) == 3
    assert sum(positions.values()) / 3 == pytest.approx(7.8)


def test_generate_bundle_creates_auditable_github_gallery(tmp_path: Path) -> None:
    run = _fixture_run(tmp_path)
    output = tmp_path / "bundle"

    artifacts = generate_bundle(run_dir=run, output_dir=output)

    figure_names = (
        "01_magnitude_group_convergence",
        "02_high_magnitude_event_trajectories",
        "03_convergence_time_by_magnitude",
    )
    expected = {
        *(f"{name}.{suffix}" for name in figure_names for suffix in ("png", "pdf")),
        "event_predictions_by_horizon.csv",
        "magnitude_group_horizon_metrics.csv",
        "event_convergence_summary.csv",
        "README.md",
        "publication_manifest.json",
    }
    assert set(artifacts) == expected
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts.values())
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "same-event, unseen-station split" in readme
    assert "retrospective cohort" in readme
    assert "right-censored" in readme
    assert "figures/03_convergence_time_by_magnitude.png" in readme
    manifest = json.loads((output / "publication_manifest.json").read_text())
    assert manifest["analysis_contract"] == {
        "ensemble_used": False,
        "observation_horizons_sec": list(EXPECTED_HORIZONS),
        "processing_delay_sec": 5.0,
        "right_censor_horizon_sec": 200.0,
        "selected_seed": 17,
        "station_selection_causal": False,
        "target_absolute_error_mw": 0.15,
        "waveform_prefix_causal": True,
    }
    assert len(manifest["inputs"]) == 10
    assert len(manifest["outputs"]) == 10
    for name in expected:
        if name.endswith(".csv"):
            assert b"\r\n" not in (output / name).read_bytes()


def test_generate_bundle_refuses_overwrite_and_noncausal_prefix(
    tmp_path: Path,
) -> None:
    run = _fixture_run(tmp_path)
    output = tmp_path / "bundle"
    output.mkdir()
    with pytest.raises(FileExistsError):
        generate_bundle(run_dir=run, output_dir=output)

    cohort_path = (
        run
        / "internal"
        / "candidate"
        / "results"
        / "delayed_prefix"
        / "cohort_contract.json"
    )
    cohort = json.loads(cohort_path.read_text())
    cohort["waveform_prefix_causal"] = False
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    with pytest.raises(ValueError, match="cohort contract changed"):
        generate_bundle(run_dir=run, output_dir=tmp_path / "new-bundle")
