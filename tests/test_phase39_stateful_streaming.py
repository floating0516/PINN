from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.evaluation.evaluate_phase39_stateful_streaming import (
    delayed_confirmed_steps,
    project_stateful_rate,
)


def test_confirmation_delay_defers_locked_source_bins() -> None:
    steps = delayed_confirmed_steps(
        torch.tensor([0.0, 8.0]),
        previous_horizon_sec=10,
        beta_m_per_s=4.0,
        confirmation_delay_sec=6.0,
        source_steps=20,
    )

    assert steps.tolist() == [4, 2]


def test_stateful_projection_retains_only_delayed_confirmed_history() -> None:
    previous = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    current = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]])

    projected, steps = project_stateful_rate(
        previous,
        current,
        source_distance_m=torch.tensor([0.0]),
        current_horizon_sec=10,
        beta_m_per_s=4.0,
        confirmation_delay_sec=6.0,
    )

    assert steps.tolist() == [3]
    assert projected.tolist() == [[1.0, 2.0, 3.0, 40.0, 50.0]]


def test_stateful_projection_remains_nonnegative_and_does_not_mutate_inputs() -> None:
    previous = torch.tensor([[1.0, 2.0, 3.0]])
    current = torch.tensor([[4.0, 5.0, 6.0]])
    previous_copy = previous.clone()
    current_copy = current.clone()

    projected, _ = project_stateful_rate(
        previous,
        current,
        source_distance_m=torch.tensor([0.0]),
        current_horizon_sec=8,
        beta_m_per_s=4.0,
        confirmation_delay_sec=6.0,
    )

    assert bool(torch.all(projected >= 0.0))
    assert torch.equal(previous, previous_copy)
    assert torch.equal(current, current_copy)


def test_stateful_projection_rejects_negative_rates() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        project_stateful_rate(
            torch.tensor([[1.0, -1.0]]),
            torch.tensor([[1.0, 2.0]]),
            source_distance_m=torch.tensor([0.0]),
            current_horizon_sec=8,
            beta_m_per_s=4.0,
        )


def test_script_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/evaluate_phase39_stateful_streaming.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--output-root" in result.stdout
    assert "--device" in result.stdout
