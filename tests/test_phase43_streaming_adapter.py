from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts.experiments.run_phase43_streaming_adapter import (
    ADAPTER_SEEDS,
    LOSS_WEIGHTS,
    _late_metrics_from_rates,
    encode_rate,
    normalized_adapter_loss,
)


def test_encode_rate_matches_phase39_log1p_coordinate() -> None:
    rate = torch.tensor([[0.0, 1.0e18, 9.0e18]])

    encoded = encode_rate(rate, stf_m_ref=1.0e18)

    assert encoded.tolist()[0] == pytest.approx([0.0, np.log10(2.0), 1.0])


def test_normalized_objective_applies_every_frozen_weight() -> None:
    components = {
        name: torch.tensor(float(index + 1), requires_grad=True)
        for index, name in enumerate(LOSS_WEIGHTS)
    }
    normalizers = {name: float(index + 1) for index, name in enumerate(LOSS_WEIGHTS)}

    total = normalized_adapter_loss(components, normalizers)

    assert total.item() == pytest.approx(sum(LOSS_WEIGHTS.values()))


def test_normalized_objective_rejects_missing_component() -> None:
    components = {name: torch.tensor(1.0) for name in LOSS_WEIGHTS}
    components.pop("late_step")
    normalizers = {name: 1.0 for name in LOSS_WEIGHTS}

    with pytest.raises(ValueError, match="late_step"):
        normalized_adapter_loss(components, normalizers)


def test_late_metric_summary_is_zero_for_constant_stf_sequence() -> None:
    rates = np.full((2, 22, 200), 1.0e18, dtype=np.float32)

    metrics = _late_metrics_from_rates(
        rates,
        events=["event_a", "event_b"],
        catalogs=np.asarray([8.0, 8.0], dtype=np.float32),
        source_distance_m=np.asarray([0.0, 0.0], dtype=np.float32),
        source_dt_sec=np.asarray([1.0, 1.0], dtype=np.float32),
        beta_m_per_s=4_533.0,
    )

    assert metrics["late_event_abs_step_p95_mw"] == pytest.approx(0.0)
    assert metrics["late_station_abs_step_p95_mw"] == pytest.approx(0.0)
    assert metrics["late_confirmed_cumulative_log10_l1_p95"] == pytest.approx(0.0)


def test_phase43_seed_contract_is_fixed() -> None:
    assert ADAPTER_SEEDS == (17, 42, 73)


def test_script_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/run_phase43_streaming_adapter.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--stage" in result.stdout
    assert "--cache-root" in result.stdout
    assert "--normalizers" in result.stdout
