from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.plotting.plot_phase17_causal_online_results import (
    SEEDS,
    build_dynamic_station_rows,
    generate_bundle,
    load_final_event_rows,
    load_online_event_rows,
    load_training_logs,
)


EVENTS = (
    "M 7.7 - Iquique",
    "M 7.9 - Chiniak",
    "M 6.6 - Kangding",
    "M 7.7 - Mandalay",
    "M 7.3 - Nepal",
    "M 7.0 - Greece",
    "M 7.3 - Sand Point",
    "M 7.1 - Tibetan Plateau",
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _fixture_run(root: Path) -> Path:
    run = root / "phase17"
    seed_summaries: dict[str, object] = {}
    candidates: dict[str, object] = {}
    for seed in SEEDS:
        rows: list[dict[str, object]] = []
        for phase, count in (("anchor", 6), ("prefix", 4)):
            for epoch in range(1, count + 1):
                rows.append(
                    {
                        "phase": phase,
                        "epoch": epoch,
                        "loss": 1.2 / epoch + seed / 1000,
                        "fit_loss": 1.0 / epoch + seed / 1000,
                        "penalty": 0.1 / epoch,
                        "validation_mae": 0.45 / epoch + seed / 1000,
                        "learning_rate": 0.005 / epoch,
                    }
                )
        _write_csv(run / f"seed_{seed}" / "training_log.csv", rows)
        online_mae = {17: 0.25, 42: 0.31, 73: 0.29}[seed]
        final_mae = {17: 0.20, 42: 0.24, 73: 0.21}[seed]
        candidates[str(seed)] = {
            "validation_online_mae": online_mae,
            "validation_final_mae": final_mae,
        }
        seed_summaries[str(seed)] = {
            "best_anchor_epoch": 3,
            "best_prefix_epoch": 2,
        }

    references = [7.7, 7.9, 6.6, 7.7, 7.3, 7.0, 7.3, 7.1]
    final_errors = [0.05, -0.08, 0.20, 0.10, -0.17, 0.02, 0.03, 0.06]
    online_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    for horizon in range(1, 7):
        horizon_errors: list[float] = []
        for index, (event, reference, final_error) in enumerate(
            zip(EVENTS, references, final_errors)
        ):
            direction = 1.0 if index % 2 == 0 else -1.0
            error = final_error + direction * (6 - horizon) * 0.02
            used_count = 3 if "Greece" in event else 5
            stations = "|".join(
                f"S{index}_{horizon}_{station}" for station in range(used_count)
            )
            row = {
                "event": event,
                "horizon_sec": float(horizon),
                "mw_pred": reference + error,
                "mw_reference": reference,
                "error": error,
                "abs_error": abs(error),
                "active_station_count": used_count + index,
                "used_station_count": used_count,
                "used_stations": stations,
            }
            online_rows.append(row)
            horizon_errors.append(error)
            if horizon == 6:
                final_rows.append(row)
        horizon_rows.append(
            {
                "horizon_sec": float(horizon),
                "event_count": 8,
                "event_mae": sum(abs(value) for value in horizon_errors) / 8,
                "event_rmse": (sum(value * value for value in horizon_errors) / 8) ** 0.5,
                "event_bias": sum(horizon_errors) / 8,
            }
        )

    selection = {
        "candidates": candidates,
        "ensemble_used": False,
        "selected_seed": 17,
        "selection_metric": "validation_online_mae",
        "tie_break_metric": "validation_final_mae",
    }
    final_values = [float(row["error"]) for row in final_rows]
    summary = {
        "method": "causal_radial_event_neural_v1",
        "deep_learning": True,
        "input_components": ["R"],
        "uses_ensemble": False,
        "uses_future_waveform": False,
        "uses_final_peak_for_station_selection": False,
        "selection": selection,
        "seed_summaries": seed_summaries,
        "external": {
            "final_metrics": {
                "event_count": 8,
                "event_mae": sum(abs(value) for value in final_values) / 8,
                "event_rmse": (
                    sum(value * value for value in final_values) / 8
                )
                ** 0.5,
                "event_bias": sum(final_values) / 8,
            },
            "online_metrics": {"snapshot_count": len(online_rows)},
        },
        "spec": {
            "duration_sec": 6,
            "causal_latency_samples": 1,
            "top_k": 5,
        },
    }
    run.mkdir(parents=True, exist_ok=True)
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    (run / "config.yaml").write_text("method: causal_radial_event_neural_v1\n")
    _write_csv(run / "external_final_event_predictions.csv", final_rows)
    _write_csv(run / "external_online_predictions.csv", online_rows)
    _write_csv(run / "external_horizon_metrics.csv", horizon_rows)
    return run


def test_load_training_logs_separates_consecutive_phases(tmp_path: Path) -> None:
    run = _fixture_run(tmp_path)

    logs = load_training_logs(run)

    assert set(logs) == set(SEEDS)
    assert logs[17]["anchor"]["epoch"].tolist() == [1, 2, 3, 4, 5, 6]
    assert logs[42]["prefix"]["validation_mae"].shape == (4,)


def test_dynamic_station_summary_counts_membership_changes(tmp_path: Path) -> None:
    run = _fixture_run(tmp_path)
    final_rows = load_final_event_rows(run / "external_final_event_predictions.csv")
    online = load_online_event_rows(run / "external_online_predictions.csv")

    rows = build_dynamic_station_rows(online, final_rows)

    assert len(rows) == 8
    assert {row["distinct_used_station_sets"] for row in rows} == {6}
    assert {row["station_set_change_count"] for row in rows} == {5}
    assert rows[5]["final_used_station_count"] == 3


def test_generate_bundle_creates_github_gallery_and_all_figures(tmp_path: Path) -> None:
    run = _fixture_run(tmp_path)
    output = tmp_path / "bundle"

    artifacts = generate_bundle(phase17_run=run, output_dir=output)

    figure_names = (
        "01_training_dynamics",
        "02_seed_selection",
        "03_external_convergence",
        "04_event_trajectories",
        "05_dynamic_station_selection",
    )
    expected = {
        *(f"{name}.{suffix}" for name in figure_names for suffix in ("png", "pdf")),
        "seed_selection.csv",
        "external_final_event_predictions.csv",
        "external_horizon_metrics.csv",
        "external_online_predictions.csv",
        "dynamic_station_summary.csv",
        "README.md",
        "publication_manifest.json",
    }
    assert set(artifacts) == expected
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts.values())
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "figures/03_external_convergence.png" in readme
    assert "there is no averaging" in readme
    assert "not an unbiased final paper test set" in readme
    for name in (
        "seed_selection.csv",
        "external_final_event_predictions.csv",
        "external_horizon_metrics.csv",
        "external_online_predictions.csv",
        "dynamic_station_summary.csv",
    ):
        assert b"\r\n" not in (output / name).read_bytes()
    manifest = json.loads((output / "publication_manifest.json").read_text())
    assert len(manifest["inputs"]) == 9
    assert len(manifest["outputs"]) == 16


def test_generate_bundle_refuses_to_overwrite(tmp_path: Path) -> None:
    run = _fixture_run(tmp_path)
    output = tmp_path / "bundle"
    output.mkdir()

    with pytest.raises(FileExistsError):
        generate_bundle(phase17_run=run, output_dir=output)
