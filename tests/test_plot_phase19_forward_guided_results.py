from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.plotting.plot_phase19_forward_guided_results import (
    SEEDS,
    build_ablation_rows,
    generate_bundle,
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


def _training_logs(root: Path, *, no_synth: bool) -> None:
    for seed in SEEDS:
        rows: list[dict[str, object]] = []
        for epoch in range(1, 7):
            rows.append(
                {
                    "phase": "anchor",
                    "epoch": epoch,
                    "train_total_loss": 1.0 / epoch + seed / 1000,
                    "train_L_MSE": "",
                    "train_L_synth": "",
                    "train_L_mag": 0.8 / epoch,
                    "train_L_shape": "",
                    "validation_total_loss": "",
                    "validation_L_MSE": "",
                    "validation_L_synth": "",
                    "validation_L_mag": "",
                    "validation_L_shape": "",
                    "validation_online_mae": 0.5 / epoch + seed / 1000,
                    "learning_rate": 0.005 / epoch,
                }
            )
        for epoch in range(1, 5):
            synth_loss = 0.24 + seed / 100_000 + (0.001 if no_synth else 0.0)
            rows.append(
                {
                    "phase": "deep",
                    "epoch": epoch,
                    "train_total_loss": 0.8 / epoch + 0.2,
                    "train_L_MSE": 0.3 / epoch,
                    "train_L_synth": synth_loss,
                    "train_L_mag": 0.4 / epoch,
                    "train_L_shape": 0.001 / epoch,
                    "validation_total_loss": 0.9 / epoch + 0.3,
                    "validation_L_MSE": 0.35 / epoch,
                    "validation_L_synth": synth_loss,
                    "validation_L_mag": 0.45 / epoch,
                    "validation_L_shape": 0.0012 / epoch,
                    "validation_online_mae": 0.42 / epoch + seed / 1000,
                    "learning_rate": 0.0005 / epoch,
                }
            )
        _write_csv(root / f"seed_{seed}" / "training_log.csv", rows)


def _horizon_rows(*, mode: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stable = {"full": 89, "no_synth": 91, "phase17": 166}[mode]
    start = 0 if mode == "phase17" else 1
    for horizon in range(start, 201):
        if horizon < stable:
            mae = 0.15 + 0.45 * (stable - horizon) / stable
        else:
            mae = 0.13 - 0.01 * (horizon - stable) / (200 - stable + 1)
        rows.append(
            {
                "horizon_sec": float(horizon),
                "event_count": 6 if horizon < 38 else 8,
                "event_mae": mae,
                "event_rmse": mae * 1.2,
                "event_bias": mae * 0.2,
            }
        )
    return rows


def _main_online_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    references = [7.7, 7.9, 6.6, 7.7, 7.3, 7.0, 7.3, 7.1]
    final_errors = [0.02, -0.04, 0.31, 0.13, -0.12, 0.09, 0.06, 0.28]
    online_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    for horizon in range(1, 201):
        gate = 1.0 - horizon / 200.0
        for index, (event, reference, final_error) in enumerate(
            zip(EVENTS, references, final_errors)
        ):
            residual = gate * (0.12 if index % 2 == 0 else -0.10)
            anchor = reference + final_error + 0.30 * gate
            prediction = anchor + residual
            error = prediction - reference
            used = 3 if "Greece" in event else 5
            stations = "|".join(
                f"S{index}_{horizon % 3}_{station}" for station in range(used)
            )
            row = {
                "event": event,
                "horizon_sec": float(horizon),
                "mw_pred": prediction,
                "mw_reference": reference,
                "error": error,
                "abs_error": abs(error),
                "anchor_mw": anchor,
                "neural_residual_mw": residual,
                "active_station_count": used + index,
                "used_station_count": used,
                "used_stations": stations,
            }
            online_rows.append(row)
            if horizon == 200:
                final_rows.append(row)
    return online_rows, final_rows


def _summary(
    *,
    no_synth: bool,
    final_rows: list[dict[str, object]],
    snapshot_count: int,
) -> dict[str, object]:
    candidates = {
        "17": {"validation_online_mae": 0.38, "validation_final_mae": 0.22},
        "42": {"validation_online_mae": 0.41, "validation_final_mae": 0.27},
        "73": {
            "validation_online_mae": 0.336 if not no_synth else 0.337,
            "validation_final_mae": 0.217,
        },
    }
    seed_summaries = {
        seed: {
            "best_anchor_epoch": 3,
            "best_deep_epoch": 2,
            "validation_online_mae": values["validation_online_mae"],
            "validation_final_mae": values["validation_final_mae"],
        }
        for seed, values in candidates.items()
    }
    errors = [float(row["error"]) for row in final_rows]
    selection = {
        "candidates": candidates,
        "ensemble_used": False,
        "selected_seed": 73,
        "selection_metric": "validation_online_mae",
        "tie_break_metric": "validation_final_mae",
    }
    return {
        "method": "causal_forward_guided_event_neural_v2",
        "ablation": "no_forward_loss" if no_synth else "none",
        "deep_learning": True,
        "input_components": ["R"],
        "uses_ensemble": False,
        "uses_future_waveform": False,
        "uses_final_peak_for_station_selection": False,
        "uses_original_four_term_loss": not no_synth,
        "loss": {
            "lambda_MSE": 1.0,
            "lambda_synth": 0.0 if no_synth else 0.5,
            "lambda_mag": 1.0,
            "lambda_shape": 0.1,
        },
        "selection": selection,
        "seed_summaries": seed_summaries,
        "external": {
            "final_metrics": {
                "event_count": 8,
                "event_mae": sum(abs(value) for value in errors) / 8,
                "event_rmse": (sum(value * value for value in errors) / 8) ** 0.5,
                "event_bias": sum(errors) / 8,
            },
            "online_metrics": {
                "snapshot_count": snapshot_count,
                "event_equal_online_mae": 0.215 if not no_synth else 0.216,
            },
        },
    }


def _fixture_runs(root: Path) -> tuple[Path, Path, Path]:
    full = root / "full"
    no_synth = root / "no_synth"
    phase17 = root / "phase17"
    online_rows, final_rows = _main_online_rows()
    for run, ablation in ((full, False), (no_synth, True)):
        run.mkdir(parents=True)
        summary = _summary(
            no_synth=ablation,
            final_rows=final_rows,
            snapshot_count=len(online_rows),
        )
        (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run / "selection.json").write_text(
            json.dumps(summary["selection"]), encoding="utf-8"
        )
        (run / "config.yaml").write_text(
            f"lambda_synth: {0.0 if ablation else 0.5}\n", encoding="utf-8"
        )
        _training_logs(run, no_synth=ablation)
        _write_csv(run / "external_horizon_metrics.csv", _horizon_rows(mode="no_synth" if ablation else "full"))
    _write_csv(full / "external_online_predictions.csv", online_rows)
    _write_csv(full / "external_final_event_predictions.csv", final_rows)

    phase17.mkdir()
    phase17_summary = {
        "method": "causal_radial_event_neural_v1",
        "uses_ensemble": False,
        "uses_future_waveform": False,
        "uses_final_peak_for_station_selection": False,
    }
    (phase17 / "summary.json").write_text(
        json.dumps(phase17_summary), encoding="utf-8"
    )
    _write_csv(
        phase17 / "external_horizon_metrics.csv",
        _horizon_rows(mode="phase17"),
    )
    return full, no_synth, phase17


def test_training_logs_and_ablation_rows_are_finite(tmp_path: Path) -> None:
    full, no_synth, _ = _fixture_runs(tmp_path)
    full_logs = load_training_logs(full)
    no_synth_logs = load_training_logs(no_synth)
    full_summary = json.loads((full / "summary.json").read_text())
    no_summary = json.loads((no_synth / "summary.json").read_text())

    rows = build_ablation_rows(
        full_summary=full_summary,
        no_synth_summary=no_summary,
        full_logs=full_logs,
        no_synth_logs=no_synth_logs,
    )

    assert len(rows) == 3
    assert {row["seed"] for row in rows} == set(SEEDS)
    assert all(row["L_synth_delta_no_synth_minus_full"] > 0 for row in rows)


def test_generate_bundle_creates_github_gallery(tmp_path: Path) -> None:
    full, no_synth, phase17 = _fixture_runs(tmp_path)
    output = tmp_path / "bundle"

    artifacts = generate_bundle(
        full_run=full,
        no_synth_run=no_synth,
        phase17_run=phase17,
        output_dir=output,
    )

    figure_names = (
        "01_training_dynamics",
        "02_seed_selection",
        "03_online_convergence",
        "04_forward_loss_ablation",
        "05_event_trajectories",
        "06_dynamic_station_selection",
    )
    expected = {
        *(f"{name}.{suffix}" for name in figure_names for suffix in ("png", "pdf")),
        "seed_selection.csv",
        "forward_loss_ablation.csv",
        "external_final_event_predictions.csv",
        "external_horizon_metrics.csv",
        "no_synth_horizon_metrics.csv",
        "phase17_horizon_metrics.csv",
        "external_online_predictions.csv",
        "dynamic_station_summary.csv",
        "README.md",
        "publication_manifest.json",
    }
    assert set(artifacts) == expected
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts.values())
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "not a PINN" in readme
    assert "figures/04_forward_loss_ablation.png" in readme
    assert "there is no averaging" in readme
    manifest = json.loads((output / "publication_manifest.json").read_text())
    assert len(manifest["inputs"]) == 18
    assert len(manifest["outputs"]) == 21
    for name in expected:
        if name.endswith(".csv"):
            assert b"\r\n" not in (output / name).read_bytes()


def test_generate_bundle_refuses_to_overwrite(tmp_path: Path) -> None:
    full, no_synth, phase17 = _fixture_runs(tmp_path)
    output = tmp_path / "bundle"
    output.mkdir()

    with pytest.raises(FileExistsError):
        generate_bundle(
            full_run=full,
            no_synth_run=no_synth,
            phase17_run=phase17,
            output_dir=output,
        )
