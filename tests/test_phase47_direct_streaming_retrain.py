from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.experiments.run_phase47_direct_streaming_retrain import (
    EXPECTED_PARAMETER_COUNT,
    JOINT_EPOCHS,
    JOINT_LEARNING_RATE,
    JOINT_SCHEDULER_T0,
    MAX_PREFIX_HORIZON,
    MIN_PREFIX_HORIZON,
    SEEDS,
    horizon_for_step,
    late_consistency_weight,
    mw_step_consistency,
    training_schedule,
    validation_gate,
)


def test_horizon_cycle_covers_all_streaming_horizons() -> None:
    span = MAX_PREFIX_HORIZON - MIN_PREFIX_HORIZON + 1
    for seed in SEEDS:
        values = [horizon_for_step(step, seed) for step in range(span)]
        assert len(set(values)) == span
        assert min(values) == MIN_PREFIX_HORIZON
        assert max(values) == MAX_PREFIX_HORIZON
        assert horizon_for_step(span, seed) == horizon_for_step(0, seed)


def test_late_consistency_weight_allows_early_revisions() -> None:
    assert late_consistency_weight(20) == 0.0
    assert late_consistency_weight(60) == 0.0
    assert late_consistency_weight(100) == pytest.approx((40.0 / 140.0) ** 2)
    assert late_consistency_weight(180) > late_consistency_weight(100)
    assert late_consistency_weight(200) == 1.0


def test_mw_step_consistency_penalizes_both_directions() -> None:
    previous = torch.tensor([7.0, 8.0])
    current = torch.tensor([7.2, 7.8], requires_grad=True)

    loss, metrics = mw_step_consistency(previous, current)
    loss.backward()

    assert loss.item() > 0.0
    assert metrics["mean_abs_mw_step"] == pytest.approx(0.2)
    assert metrics["downward_fraction"] == pytest.approx(0.5)
    assert current.grad is not None
    assert bool(torch.all(current.grad != 0.0))


def test_mw_step_consistency_honors_event_weights() -> None:
    previous = torch.tensor([7.0, 7.0])
    current = torch.tensor([7.1, 7.4])

    unweighted, _ = mw_step_consistency(previous, current)
    weighted, _ = mw_step_consistency(
        previous,
        current,
        sample_weights=torch.tensor([2.0, 0.5]),
    )

    assert weighted.item() < unweighted.item()


def test_validation_gate_requires_accuracy_endpoint_and_stability() -> None:
    baseline = {
        "endpoint_event_mae": 0.10,
        "endpoint_station_mae": 0.12,
        "streaming_event_mae_mean": 0.30,
        "late_event_abs_step_p95_mw": 0.05,
        "late_station_abs_step_p95_mw": 0.08,
        "late_confirmed_cumulative_log10_l1_p95": 0.10,
    }
    passing = {
        "endpoint_event_mae": 0.102,
        "endpoint_station_mae": 0.122,
        "streaming_event_mae_mean": 0.28,
        "late_event_abs_step_p95_mw": 0.039,
        "late_station_abs_step_p95_mw": 0.079,
        "late_confirmed_cumulative_log10_l1_p95": 0.079,
    }

    assert validation_gate(passing, baseline)["passed"] is True
    changed = dict(passing)
    changed["streaming_event_mae_mean"] = 0.30
    assert validation_gate(changed, baseline)["passed"] is False
    changed = dict(passing)
    changed["endpoint_event_mae"] = 0.106
    assert validation_gate(changed, baseline)["passed"] is False


def test_phase47_keeps_phase39_parameter_count() -> None:
    assert EXPECTED_PARAMETER_COUNT == 1_010_850


def test_joint_from_scratch_restores_original_phase39_schedule() -> None:
    schedule = training_schedule(joint_from_scratch=True)

    assert schedule["phase"] == "Phase48"
    assert schedule["initialization"] == "deterministic_random"
    assert schedule["epochs"] == JOINT_EPOCHS == 200
    assert schedule["learning_rate"] == JOINT_LEARNING_RATE == 1.0e-4
    assert schedule["scheduler"] == "CosineAnnealingWarmRestarts"
    assert schedule["scheduler_T0"] == JOINT_SCHEDULER_T0 == 15


def test_default_schedule_still_reproduces_phase47() -> None:
    schedule = training_schedule(joint_from_scratch=False)

    assert schedule["phase"] == "Phase47"
    assert schedule["initialization"] == "phase39_seed_checkpoint"
    assert schedule["epochs"] == 20
    assert schedule["learning_rate"] == 1.0e-5
    assert schedule["scheduler"] == "constant"


def test_script_help_states_no_adapter_contract() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/run_phase47_direct_streaming_retrain.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--output-root" in result.stdout
    assert "--smoke" in result.stdout
    assert "--joint-from-scratch" in result.stdout
    assert "no adapter" in result.stdout.lower()


def test_validation_report_plotter_help_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plotting/plot_phase47_phase48_direct_streaming_validation_zh.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--phase47-run" in result.stdout
    assert "--phase48-run" in result.stdout
    assert "--output-dir" in result.stdout
