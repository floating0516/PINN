from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts.evaluation.evaluate_phase66_stateful_streaming import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_EPOCH,
    EXPECTED_EXTERNAL_RAW_SHA256,
    EXPECTED_INTERNAL_RAW_SHA256,
    EXPECTED_SEED,
    HORIZONS,
    PROCESSING_DELAY_SEC,
    _streaming_metrics,
    apply_phase66_to_rates,
)
from scripts.plotting.plot_phase66_stateful_report_zh import (
    _stable_horizon,
)


class _CumulativeFakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def stream_sequence_from_rates(
        self,
        raw_rates: torch.Tensor,
        *,
        horizons_sec: tuple[int, ...],
        source_distance_m: torch.Tensor,
        source_dt_sec: torch.Tensor,
        beta_m_per_s: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del source_distance_m, source_dt_sec, beta_m_per_s
        assert len(horizons_sec) == raw_rates.shape[1]
        states = torch.cumsum(raw_rates, dim=1)
        magnitudes = torch.zeros(
            raw_rates.shape[:2],
            device=raw_rates.device,
            dtype=raw_rates.dtype,
        )
        gates = torch.ones_like(raw_rates)
        return states, states, magnitudes, gates


def _constant_rate_for_mw(mw: float) -> float:
    moment = 10.0 ** (1.5 * float(mw) + 9.1)
    return moment / 200.0


def test_phase66_reporting_contract_is_frozen() -> None:
    assert EXPECTED_SEED == 17
    assert EXPECTED_EPOCH == 26
    assert PROCESSING_DELAY_SEC == pytest.approx(6.0)
    assert HORIZONS == tuple(range(20, 201))
    assert len(EXPECTED_CHECKPOINT_SHA256) == 64
    assert len(EXPECTED_INTERNAL_RAW_SHA256) == 64
    assert len(EXPECTED_EXTERNAL_RAW_SHA256) == 64


def test_phase66_application_restarts_after_availability_gap(tmp_path: Path) -> None:
    model = _CumulativeFakeModel()
    raw = np.full((len(HORIZONS), 1, 200), np.nan, dtype=np.float32)
    mask = np.zeros((len(HORIZONS), 1), dtype=bool)
    mask[0:2, 0] = True
    mask[5:7, 0] = True
    raw[0, 0] = 1.0
    raw[1, 0] = 2.0
    raw[5, 0] = 5.0
    raw[6, 0] = 6.0

    stateful, diagnostics = apply_phase66_to_rates(
        model=model,  # type: ignore[arg-type]
        raw_rates=raw,
        available_mask=mask,
        source_distance_m=np.asarray([0.0], dtype=np.float32),
        source_dt_sec=np.asarray([1.0], dtype=np.float32),
        beta_m_per_s=4_533.0,
        output_path=tmp_path / "phase66.npy",
        batch_size=1,
    )

    assert np.array_equal(np.all(np.isfinite(stateful), axis=2), mask)
    assert np.array_equal(stateful[0, 0], raw[0, 0])
    assert np.array_equal(stateful[1, 0], raw[0, 0] + raw[1, 0])
    assert np.array_equal(stateful[5, 0], raw[5, 0])
    assert np.array_equal(stateful[6, 0], raw[5, 0] + raw[6, 0])
    assert diagnostics["mean_retention_gate"] == pytest.approx(1.0)


def test_streaming_metrics_capture_peak_to_final_and_downward_step() -> None:
    trajectory = np.full(len(HORIZONS), 7.0, dtype=np.float64)
    index_60 = HORIZONS.index(60)
    index_100 = HORIZONS.index(100)
    trajectory[index_60:index_100] = np.linspace(
        7.0,
        7.5,
        index_100 - index_60,
        endpoint=False,
    )
    trajectory[index_100:] = np.linspace(
        7.5,
        7.2,
        len(HORIZONS) - index_100,
    )
    rates = np.empty((len(HORIZONS), 1, 200), dtype=np.float32)
    for horizon_index, mw in enumerate(trajectory):
        rates[horizon_index, 0] = _constant_rate_for_mw(float(mw))

    metrics = _streaming_metrics(
        rates,
        available_mask=np.ones(rates.shape[:2], dtype=bool),
        events=["event_a"],
        catalogs=np.asarray([7.2], dtype=np.float32),
        source_distance_m=np.asarray([0.0], dtype=np.float32),
        source_dt_sec=np.asarray([1.0], dtype=np.float32),
        beta_m_per_s=4_533.0,
    )

    expected_step = 0.3 / (len(HORIZONS) - index_100 - 1)
    assert metrics["event_downward_step_max_mw"] == pytest.approx(
        expected_step,
        abs=2.0e-5,
    )
    assert metrics["event_peak_to_final_p95_mw"] == pytest.approx(0.3, abs=2.0e-5)
    assert metrics["endpoint_event_mae"] == pytest.approx(0.0, abs=2.0e-5)


def test_phase66_evaluator_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/evaluate_phase66_stateful_streaming.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--stage" in result.stdout
    assert "internal" in result.stdout
    assert "external" in result.stdout
    assert "all" in result.stdout


def test_phase66_report_stable_horizon_requires_a_stable_suffix() -> None:
    rows = [
        {"observation_horizon_sec": "20", "phase66_event_mae": "0.14"},
        {"observation_horizon_sec": "21", "phase66_event_mae": "0.18"},
        {"observation_horizon_sec": "22", "phase66_event_mae": "0.15"},
        {"observation_horizon_sec": "23", "phase66_event_mae": "0.12"},
    ]

    assert _stable_horizon(rows, "phase66_event_mae") == 22


def test_phase66_plotter_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plotting/plot_phase66_stateful_report_zh.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--run-root" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--full-regression" in result.stdout
