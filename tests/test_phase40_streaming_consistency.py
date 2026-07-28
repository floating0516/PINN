from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.experiments.run_phase40_streaming_consistency import (
    EXPECTED_SCOPE_PARAMETER_COUNTS,
    EXPECTED_TRAINABLE_PARAMETER_COUNT,
    GATES,
    TRAINABLE_SCOPE_PREFIXES,
    cumulative_log_consistency_from_rate,
    freeze_factorized_heads,
    freeze_trainable_scope,
    horizon_for_step,
    validation_gate,
)


class _DummyFactorizedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3)
        self.shape_head = torch.nn.Linear(3, 2)
        self.log10_moment_head = torch.nn.Linear(3, 1)


class _DummyLastTransformerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3)
        self.transformer = torch.nn.Module()
        self.transformer.layers = torch.nn.ModuleList(
            [torch.nn.Linear(3, 3) for _ in range(3)]
        )
        self.post_transformer_norm = torch.nn.LayerNorm(3)
        self.shape_head = torch.nn.Linear(3, 2)
        self.log10_moment_head = torch.nn.Linear(3, 1)


def test_horizon_cycle_covers_every_allowed_horizon_before_repeating() -> None:
    values = [horizon_for_step(step) for step in range(180)]

    assert len(set(values)) == 180
    assert min(values) == 20
    assert max(values) == 199
    assert horizon_for_step(180) == horizon_for_step(0)


def test_consistency_ignores_the_unsupported_future_tail() -> None:
    previous = torch.full((1, 6), 2.0, requires_grad=True)
    current = previous.detach().clone().requires_grad_(True)
    distance = torch.tensor([16.0])
    dt = torch.tensor([1.0])

    baseline, metrics = cumulative_log_consistency_from_rate(
        previous,
        current,
        source_distance_m=distance,
        source_dt_sec=dt,
        previous_horizon_sec=5,
        beta_m_per_s=4.0,
    )
    changed_tail = current.detach().clone()
    changed_tail[:, 1:] = 1000.0
    changed, changed_metrics = cumulative_log_consistency_from_rate(
        previous,
        changed_tail,
        source_distance_m=distance,
        source_dt_sec=dt,
        previous_horizon_sec=5,
        beta_m_per_s=4.0,
    )

    assert metrics["comparable_count"] == 1
    assert baseline.item() == pytest.approx(0.0)
    assert changed.item() == pytest.approx(0.0)
    assert changed_metrics["mean_abs_log10_revision"] == pytest.approx(0.0)


def test_consistency_penalizes_and_backpropagates_confirmed_revision() -> None:
    previous = torch.tensor([[2.0e18, 2.0e18, 2.0e18]], requires_grad=True)
    current = torch.tensor([[1.0e18, 2.0e18, 2.0e18]], requires_grad=True)

    loss, metrics = cumulative_log_consistency_from_rate(
        previous,
        current,
        source_distance_m=torch.tensor([0.0]),
        source_dt_sec=torch.tensor([1.0]),
        previous_horizon_sec=2,
        beta_m_per_s=4.0,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert metrics["downward_fraction"] == 1.0
    assert previous.grad is not None and bool(torch.any(previous.grad != 0.0))
    assert current.grad is not None and bool(torch.any(current.grad != 0.0))


def test_consistency_is_zero_when_no_s_supported_bins_exist() -> None:
    previous = torch.ones((2, 4), requires_grad=True)
    current = torch.ones((2, 4), requires_grad=True)

    loss, metrics = cumulative_log_consistency_from_rate(
        previous,
        current,
        source_distance_m=torch.tensor([100.0, 200.0]),
        source_dt_sec=torch.ones(2),
        previous_horizon_sec=1,
        beta_m_per_s=1.0,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert metrics["comparable_count"] == 0


def test_head_freeze_enables_only_factorized_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _DummyFactorizedModel()
    expected_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(("shape_head.", "log10_moment_head."))
    )
    monkeypatch.setitem(
        EXPECTED_SCOPE_PARAMETER_COUNTS,
        "heads",
        expected_count,
    )

    trainable, count = freeze_factorized_heads(model)  # type: ignore[arg-type]

    assert count == expected_count
    assert sum(parameter.numel() for parameter in trainable) == expected_count
    assert all(
        parameter.requires_grad
        == name.startswith(("shape_head.", "log10_moment_head."))
        for name, parameter in model.named_parameters()
    )
    assert EXPECTED_TRAINABLE_PARAMETER_COUNT == 8_322


def test_last_transformer_scope_enables_only_declared_final_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _DummyLastTransformerModel()
    prefixes = TRAINABLE_SCOPE_PREFIXES["last_transformer"]
    expected_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    )
    monkeypatch.setitem(
        EXPECTED_SCOPE_PARAMETER_COUNTS,
        "last_transformer",
        expected_count,
    )

    trainable, count = freeze_trainable_scope(
        model,  # type: ignore[arg-type]
        "last_transformer",
    )

    assert count == expected_count
    assert sum(parameter.numel() for parameter in trainable) == expected_count
    assert all(
        parameter.requires_grad == name.startswith(prefixes)
        for name, parameter in model.named_parameters()
    )


def test_frozen_scope_parameter_counts_are_locked() -> None:
    assert EXPECTED_SCOPE_PARAMETER_COUNTS == {
        "heads": 8_322,
        "last_transformer": 141_058,
    }


def test_validation_gate_requires_endpoint_and_all_stability_metrics() -> None:
    passing = {
        "endpoint_event_mae": GATES["endpoint_event_mae_max"] - 1.0e-4,
        "endpoint_station_mae": GATES["endpoint_station_mae_max"] - 1.0e-4,
        "late_event_abs_step_p95_mw": (
            GATES["late_event_abs_step_p95_mw_max"] * 0.9
        ),
        "late_station_abs_step_p95_mw": (
            GATES["late_station_abs_step_p95_mw_max"] * 0.9
        ),
        "late_confirmed_cumulative_log10_l1_p95": (
            GATES["late_confirmed_cumulative_log10_l1_p95_max"] * 0.9
        ),
    }

    gate = validation_gate(passing)

    assert gate["passed"] is True
    changed = dict(passing)
    changed["endpoint_event_mae"] = GATES["endpoint_event_mae_max"] + 1.0e-4
    assert validation_gate(changed)["passed"] is False
    changed = dict(passing)
    changed["late_confirmed_cumulative_log10_l1_p95"] = (
        GATES["late_confirmed_cumulative_log10_l1_p95_max"] * 1.01
    )
    assert validation_gate(changed)["passed"] is False


def test_script_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/run_phase40_streaming_consistency.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--output-root" in result.stdout
    assert "--trainable-scope" in result.stdout
    assert "--smoke" in result.stdout
