from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts.evaluation.evaluate_phase39_streaming_replay import (
    RawStreamingStation,
    conservative_visible_steps,
    decode_stf_rate,
    decompose_stf_revision,
    evaluate_streaming_replay,
    preprocess_streaming_prefix,
)
from src.data.waveform import WaveformConfig, preprocess_waveform


def _waveform_config() -> WaveformConfig:
    return WaveformConfig(
        sample_rate_hz=1.0,
        start_sec=0.0,
        duration_sec=200.0,
        min_valid_fraction=0.0,
        max_interpolation_gap_sec=0.0,
        baseline_method="median",
        pre_event_start_sec=-20.0,
        pre_event_end_sec=0.0,
        baseline_fallback="pre_p",
        baseline_fallback_max_sec=30.0,
        baseline_min_samples=10,
        filter_type="lowpass",
        cutoff_hz=0.2,
        num_taps=7,
        filter_window="hamming",
    )


def _record() -> RawStreamingStation:
    time = np.arange(-20.0, 211.0, dtype=np.float64)
    radial = np.where(time < 0.0, 0.002, 0.002 + 1.0e-4 * time)
    config = _waveform_config()
    endpoint = preprocess_waveform(
        time[time < 200.0],
        radial[time < 200.0],
        units="m",
        p_arrival_sec=12.0,
        config=config,
    )
    return RawStreamingStation(
        event="Synthetic M7.0",
        event_dir="synthetic",
        station="TEST",
        magnitude_catalog=7.0,
        magnitude_source="test",
        usgs_event_id="test",
        raw_time_sec=time,
        raw_radial_m=radial,
        waveform_config=config,
        source_distance_m=100_000.0,
        epicentral_distance_m=95_000.0,
        theta_deg=30.0,
        azimuth_deg=120.0,
        p_arrival_sec=12.0,
        s_arrival_sec=20.0,
        endpoint_radial_m=endpoint.values_m,
        endpoint_valid_mask=endpoint.valid_mask,
        endpoint_baseline_source=endpoint.baseline_source,
        waveform_start_sec=0.0,
        waveform_phase_adjusted=False,
    )


def test_streaming_prefix_has_true_variable_length_and_no_future_tail() -> None:
    record = _record()

    prefix = preprocess_streaming_prefix(record, observation_horizon_sec=20)

    assert prefix.values_m.shape == (20,)
    assert prefix.valid_mask.shape == (20,)
    assert prefix.issue_time_sec == 25.0
    assert prefix.raw_sample_count == int(np.count_nonzero(record.raw_time_sec <= 25.0))


def test_streaming_prefix_is_invariant_to_samples_after_issue_time() -> None:
    record = _record()
    changed = record.raw_radial_m.copy()
    changed[record.raw_time_sec > 25.0] += 1000.0

    original_prefix = preprocess_streaming_prefix(
        record, observation_horizon_sec=20
    )
    changed_prefix = preprocess_streaming_prefix(
        replace(record, raw_radial_m=changed), observation_horizon_sec=20
    )

    np.testing.assert_array_equal(original_prefix.values_m, changed_prefix.values_m)
    np.testing.assert_array_equal(original_prefix.valid_mask, changed_prefix.valid_mask)


def test_samples_after_200_do_not_change_the_fixed_200_point_endpoint() -> None:
    record = _record()
    changed = record.raw_radial_m.copy()
    changed[record.raw_time_sec > 200.0] -= 500.0

    original = preprocess_streaming_prefix(record, observation_horizon_sec=200)
    modified = preprocess_streaming_prefix(
        replace(record, raw_radial_m=changed), observation_horizon_sec=200
    )

    np.testing.assert_array_equal(original.values_m, record.endpoint_radial_m)
    np.testing.assert_array_equal(original.values_m, modified.values_m)


def test_conservative_visible_steps_requires_complete_source_bins() -> None:
    assert conservative_visible_steps(20, 20.0) == 0
    assert conservative_visible_steps(21, 20.0) == 1
    assert conservative_visible_steps(25, 20.4) == 4
    assert conservative_visible_steps(250, 20.0) == 200


def test_stf_revision_separates_history_new_confirmation_and_future_tail() -> None:
    previous = np.asarray([1.0, 2.0, 3.0, 4.0])
    current = np.asarray([0.5, 3.0, 3.0, 2.0])

    result = decompose_stf_revision(
        current,
        previous,
        previous_confirmed_steps=2,
        current_confirmed_steps=3,
    )

    assert result["previous_confirmed_moment_nm"] == pytest.approx(3.0)
    assert result["confirmed_history_revision_nm"] == pytest.approx(0.5)
    assert result["confirmed_history_l1_revision_nm"] == pytest.approx(1.5)
    assert result["newly_confirmed_moment_nm"] == pytest.approx(3.0)
    assert result["future_tail_revision_nm"] == pytest.approx(-2.0)
    assert result["full_stf_l1_revision_nm"] == pytest.approx(3.5)
    assert result["confirmed_history_moment_decreased"] is False


def test_decoded_stf_rate_is_nonnegative() -> None:
    encoded = torch.tensor([[0.0, 0.1] + [0.0] * 198], dtype=torch.float32)

    rate = decode_stf_rate(encoded, stf_m_ref=1.0e18)

    assert tuple(rate.shape) == (1, 200)
    assert bool(torch.all(rate >= 0.0))


class _RecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.input_lengths: list[int] = []

    def forward(
        self, values: torch.Tensor, meta: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.input_lengths.append(int(values.shape[-1]))
        return torch.full(
            (values.shape[0], 200),
            0.01,
            dtype=values.dtype,
            device=values.device,
        ) + self.anchor * 0.0


def test_replay_passes_each_horizon_without_padding_to_fake_model() -> None:
    model = _RecordingModel()
    config = {
        "dataset": {"stf": {"m_ref": 1.0e18}},
        "geometry": {"network_distance": "hypocentral"},
    }

    result = evaluate_streaming_replay(
        model,  # type: ignore[arg-type]
        config,
        [_record()],
        horizons=(2, 4, 200),
        batch_size=1,
    )

    assert model.input_lengths == [2, 4, 200]
    assert result["rate_nm_per_s"].shape == (3, 1, 200)
    assert bool(np.all(result["available_mask"]))
    assert result["input_endpoint_gate"]["max_abs_radial_input_diff_m"] == 0.0


def test_script_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/evaluate_phase39_streaming_replay.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--smoke" in result.stdout
    assert "variable-length" in result.stdout
