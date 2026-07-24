from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data.waveform import ProcessedWaveform, WaveformConfig
from src.evaluation.delayed_prefix import (
    fir_lookahead_samples,
    preprocess_and_release_delayed_prefix,
    release_delayed_prefix,
)
from src.models.model import PINNModel


def _config(**overrides: object) -> WaveformConfig:
    values = {
        "sample_rate_hz": 1.0,
        "start_sec": 0.0,
        "duration_sec": 12.0,
        "min_valid_fraction": 0.8,
        "max_interpolation_gap_sec": 0.0,
        "baseline_method": "median",
        "pre_event_start_sec": -3.0,
        "pre_event_end_sec": 0.0,
        "baseline_fallback": "pre_p",
        "baseline_fallback_max_sec": 3.0,
        "baseline_min_samples": 3,
        "filter_type": "lowpass",
        "cutoff_hz": 0.1,
        "num_taps": 7,
        "filter_window": "hamming",
    }
    values.update(overrides)
    return WaveformConfig(**values)


def _processed() -> ProcessedWaveform:
    values = np.arange(12, dtype=np.float32)
    valid_mask = np.ones(12, dtype=bool)
    valid_mask[1] = False
    return ProcessedWaveform(
        time_sec=np.arange(12, dtype=np.float32),
        values_m=values,
        valid_mask=valid_mask,
        dt_sec=1.0,
        raw_dt_sec=1.0,
        baseline_m=0.0,
        baseline_source="pre_event",
        valid_fraction=float(valid_mask.mean()),
    )


def test_five_second_release_masks_samples_outside_half_open_prefix() -> None:
    config = _config()
    released = release_delayed_prefix(
        _processed(),
        prefix_steps=4,
        config=config,
    )

    assert fir_lookahead_samples(config) == 3
    assert released.fir_lookahead_samples == 3
    assert released.prefix_steps == 4
    assert released.prefix_sec == 4.0
    assert released.processing_delay_sec == 5.0
    assert released.issue_time_sec == 9.0
    np.testing.assert_array_equal(
        released.released_valid_mask,
        [
            True,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
    )
    np.testing.assert_array_equal(
        released.masked_waveform_m,
        [
            0.0,
            0.0,
            2.0,
            3.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )


def test_delay_shorter_than_fir_lookahead_is_rejected() -> None:
    with pytest.raises(ValueError, match="shorter than the centered FIR lookahead"):
        release_delayed_prefix(
            _processed(),
            prefix_steps=4,
            config=_config(),
            processing_delay_sec=2.0,
        )

    with pytest.raises(ValueError, match="max_interpolation_gap_sec=0"):
        release_delayed_prefix(
            _processed(),
            prefix_steps=4,
            config=replace(_config(), max_interpolation_gap_sec=2.5),
        )

    with pytest.raises(ValueError, match="exactly 7 FIR taps"):
        release_delayed_prefix(
            _processed(),
            prefix_steps=4,
            config=replace(_config(), num_taps=9),
        )

    with pytest.raises(ValueError, match="requires a 5 s delay"):
        release_delayed_prefix(
            _processed(),
            prefix_steps=4,
            config=_config(),
            processing_delay_sec=4.0,
        )


def test_raw_samples_after_issue_time_do_not_change_released_prefix() -> None:
    config = _config()
    time_sec = np.arange(-3.0, 12.0)
    values = np.where(time_sec < 0.0, 0.0, 0.01 * time_sec)
    changed = values.copy()
    changed[time_sec > 9.0] = 1.0e6

    original_release = preprocess_and_release_delayed_prefix(
        time_sec,
        values,
        units="m",
        p_arrival_sec=5.0,
        prefix_steps=4,
        config=config,
    )
    changed_release = preprocess_and_release_delayed_prefix(
        time_sec,
        changed,
        units="m",
        p_arrival_sec=5.0,
        prefix_steps=4,
        config=config,
    )

    assert original_release.issue_time_sec == 9.0
    np.testing.assert_array_equal(
        original_release.released_valid_mask,
        changed_release.released_valid_mask,
    )
    np.testing.assert_array_equal(
        original_release.masked_waveform_m,
        changed_release.masked_waveform_m,
    )


def test_pre_p_baseline_does_not_read_samples_after_issue_time() -> None:
    config = _config(
        pre_event_start_sec=-3.0,
        pre_event_end_sec=0.0,
        baseline_min_samples=3,
    )
    time_sec = np.arange(-1.0, 13.0)
    values = np.where(time_sec < 0.0, 2.0, 2.0 + 0.01 * time_sec)
    changed = values.copy()
    changed[time_sec > 9.0] = 1.0e6

    original_release = preprocess_and_release_delayed_prefix(
        time_sec,
        values,
        units="m",
        p_arrival_sec=12.0,
        prefix_steps=4,
        config=config,
    )
    changed_release = preprocess_and_release_delayed_prefix(
        time_sec,
        changed,
        units="m",
        p_arrival_sec=12.0,
        prefix_steps=4,
        config=config,
    )

    assert original_release.issue_time_sec == 9.0
    np.testing.assert_array_equal(
        original_release.masked_waveform_m,
        changed_release.masked_waveform_m,
    )


def test_unavailable_raw_future_cannot_change_model_stf_or_magnitude() -> None:
    waveform_config = _config()
    time_sec = np.arange(-3.0, 12.0)
    values = np.where(time_sec < 0.0, 0.0, 0.01 * time_sec)
    changed = values.copy()
    changed[time_sec > 9.0] = -1.0e6
    releases = [
        preprocess_and_release_delayed_prefix(
            time_sec,
            item,
            units="m",
            p_arrival_sec=5.0,
            prefix_steps=4,
            config=waveform_config,
        )
        for item in (values, changed)
    ]

    model_config = yaml.safe_load(
        Path("configs/experiments_v2/V2-BASE.yaml").read_text(
            encoding="utf-8"
        )
    )
    model_config["dataset"]["waveform"]["duration_sec"] = 12.0
    model_config["dataset"]["stf"]["duration_sec"] = 12.0
    model_config["model"].update(
        {
            "hidden_dim": 32,
            "num_layers": 1,
            "num_tcn_blocks": 1,
            "transformer_num_layers": 1,
            "dropout": 0.0,
            "stf_output_parameterization": "moment_shape_factorized",
        }
    )
    model = PINNModel(model_config).eval()
    assert model.log10_moment_head is not None
    with torch.no_grad():
        model.log10_moment_head[-1].weight.fill_(0.01)
    metadata = torch.zeros(1, 5)
    with torch.no_grad():
        predictions = [
            model.predict_heads(
                torch.from_numpy(release.masked_waveform_m)
                .float()
                .reshape(1, 1, -1),
                meta=metadata,
            )
            for release in releases
        ]

    torch.testing.assert_close(
        predictions[0].stf_encoded,
        predictions[1].stf_encoded,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        predictions[0].catalog_mw,
        predictions[1].catalog_mw,
        rtol=0.0,
        atol=0.0,
    )


def test_missing_grid_sample_is_not_filled_from_a_future_sample() -> None:
    config = _config(duration_sec=8.0)
    complete_time = np.arange(-3.0, 8.0)
    keep = complete_time != 2.0
    time_sec = complete_time[keep]
    values = np.where(time_sec < 0.0, 0.0, time_sec)
    values[time_sec == 3.0] = 1000.0

    released = preprocess_and_release_delayed_prefix(
        time_sec,
        values,
        units="m",
        p_arrival_sec=5.0,
        prefix_steps=4,
        config=config,
    )

    assert released.released_valid_mask[0]
    assert released.released_valid_mask[1]
    assert not released.released_valid_mask[2]
    assert released.released_valid_mask[3]
    assert released.masked_waveform_m[2] == 0.0
    assert np.all(released.masked_waveform_m[4:] == 0.0)
    assert not np.any(released.released_valid_mask[4:])
