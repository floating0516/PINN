from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.plotting.plot_phase16_neural_event_results import (
    build_method_metrics,
    generate_bundle,
    load_training_logs,
)


SEEDS = (17, 42, 73)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture_run(root: Path) -> tuple[Path, Path]:
    run = root / "run"
    head_summaries: dict[str, object] = {}
    for seed in SEEDS:
        _write_csv(
            run / f"seed_{seed}" / "training_log.csv",
            [
                {
                    "epoch": epoch,
                    "loss": 1.0 / epoch + seed / 1000,
                    "fit_loss": 0.8 / epoch + seed / 1000,
                    "linear_penalty": 0.1,
                    "nonlinear_penalty": 0.0 if epoch <= 2 else 0.01,
                    "train_event_mae": 0.4 / epoch + seed / 1000,
                    "validation_event_mae": 0.5 / epoch + seed / 1000,
                    "learning_rate": 0.005 / epoch,
                }
                for epoch in range(1, 7)
            ],
        )
        head_summaries[str(seed)] = {
            "best_epoch": 3,
            "linear_warmup_epochs": 2,
        }
    events: list[dict[str, object]] = []
    for index in range(8):
        reference = 6.6 + index * 0.15
        error = (-1 if index % 2 else 1) * (0.03 + index * 0.02)
        events.append(
            {
                "event": f"Event{index} 202{index}",
                "mw_pred": reference + error,
                "mw_reference": reference,
                "error": error,
                "abs_error": abs(error),
                "station_count_available": index + 3,
                "station_count_used": min(5, index + 3),
                "head_prediction_std": 0.01 + index * 0.001,
                "nonlinear_delta_mw_mean": -0.001 + index * 0.0001,
            }
        )
    _write_csv(run / "ensemble_external_event_predictions.csv", events)
    errors = [float(row["error"]) for row in events]
    summary = {
        "method": "radial_pinn_event_neural_v2",
        "deep_learning": True,
        "uses_ridge_prediction": False,
        "head_summaries": head_summaries,
        "external_ensemble": {
            "event_count": 8,
            "event_mae": sum(abs(value) for value in errors) / 8,
            "event_rmse": (sum(value * value for value in errors) / 8) ** 0.5,
            "event_bias": sum(errors) / 8,
        },
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    comparison = root / "comparison.csv"
    methods = (
        "radial_event_ridge",
        "phase9_r_ensemble",
        "phase13_event_balanced",
        "pgd_crowell",
        "pgd_ruhl",
        "pgd_melgar",
    )
    _write_csv(
        comparison,
        [
            {
                "method": method,
                "event_count": 8,
                "event_mae": 0.12 + index * 0.03,
                "event_rmse": 0.15 + index * 0.04,
                "event_bias": -0.1 + index * 0.04,
            }
            for index, method in enumerate(methods)
        ],
    )
    return run, comparison


def test_load_training_logs_requires_consecutive_equal_epochs(tmp_path: Path) -> None:
    run, _ = _fixture_run(tmp_path)

    logs = load_training_logs(run)

    assert set(logs) == set(SEEDS)
    assert logs[17]["epoch"].tolist() == [1, 2, 3, 4, 5, 6]
    assert logs[42]["validation_event_mae"].shape == (6,)


def test_build_method_metrics_prepends_neural_result(tmp_path: Path) -> None:
    run, comparison = _fixture_run(tmp_path)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))

    rows = build_method_metrics(
        summary=summary,
        comparison_metrics_csv=comparison,
    )

    assert len(rows) == 7
    assert rows[0]["method"] == "radial_pinn_event_neural_v2"
    assert rows[0]["event_mae"] == pytest.approx(0.1)


def test_generate_bundle_creates_github_gallery_and_all_figures(tmp_path: Path) -> None:
    run, comparison = _fixture_run(tmp_path)
    output = tmp_path / "bundle"

    artifacts = generate_bundle(
        phase16_run=run,
        comparison_metrics_csv=comparison,
        output_dir=output,
    )

    expected = {
        *(f"0{index}_{name}.{suffix}" for index, name in (
            (1, "training_dynamics"),
            (2, "external_event_performance"),
            (3, "method_comparison"),
            (4, "station_and_neural_contribution"),
        ) for suffix in ("png", "pdf")),
        "event_predictions.csv",
        "method_metrics.csv",
        "README.md",
        "publication_manifest.json",
    }
    assert set(artifacts) == expected
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts.values())
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "figures/01_training_dynamics.png" in readme
    assert "Fixed eight-event development validation" in readme
    assert b"\r\n" not in (output / "event_predictions.csv").read_bytes()
    assert b"\r\n" not in (output / "method_metrics.csv").read_bytes()
    manifest = json.loads((output / "publication_manifest.json").read_text())
    assert len(manifest["inputs"]) == 6
    assert len(manifest["outputs"]) == 11
