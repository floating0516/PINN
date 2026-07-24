from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
import yaml

from src.data.metadata import build_metadata_tensor, metadata_distance_from_config
from src.evaluation.evaluate_delayed_prefix import evaluate_delayed_prefix
from src.models.model import PINNModel
from src.training.loss_stf_rate_v2 import moment_magnitude_from_rate


def _config() -> dict:
    config = yaml.safe_load(
        Path("configs/experiments_v2/V2-BASE.yaml").read_text(encoding="utf-8")
    )
    config["dataset"]["filter"]["cutoff_hz"] = 0.2
    config["dataset"]["waveform"]["max_interpolation_gap_sec"] = 0.0
    config["model"].update(
        {
            "hidden_dim": 16,
            "num_layers": 1,
            "num_tcn_blocks": 1,
            "transformer_num_layers": 1,
            "dropout": 0.0,
            "input_components": ["radial"],
            "predict_catalog_mw": False,
            "stf_output_parameterization": "direct",
        }
    )
    return config


def _batch(
    radial: torch.Tensor,
    *,
    events: list[str] | None = None,
    stations: list[str] | None = None,
    magnitudes: list[float] | None = None,
    baseline_sources: list[str] | None = None,
    source_distances_m: list[float] | None = None,
) -> dict:
    batch_size, _, time_steps = radial.shape
    return {
        "event": events or [f"E{index}" for index in range(batch_size)],
        "station": stations or [f"S{index}" for index in range(batch_size)],
        "radial": radial.float(),
        "waveform_valid_mask": torch.ones(batch_size, time_steps, dtype=torch.bool),
        "waveform_dt_sec": torch.ones(batch_size),
        "raw_dt_sec": torch.ones(batch_size),
        "stf_dt_sec": torch.ones(batch_size),
        "source_distance_m": torch.tensor(
            source_distances_m or [100_000.0] * batch_size
        ),
        "epicentral_distance_m": torch.full((batch_size,), 95_000.0),
        "theta_deg": torch.full((batch_size,), 45.0),
        "azimuth_deg": torch.full((batch_size,), 30.0),
        "magnitude_catalog": torch.tensor(
            magnitudes or [7.0] * batch_size,
            dtype=torch.float32,
        ),
        "baseline_source": baseline_sources or ["pre_event"] * batch_size,
    }


class _FirstSampleMagnitudeModel(PINNModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._m_ref = float(config["dataset"]["stf"]["m_ref"])
        self._steps = int(self.output_time_steps or 0)

    def forward(
        self,
        x: torch.Tensor,
        meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del meta
        magnitude = x[:, 0, 0]
        moment = torch.pow(10.0, 1.5 * magnitude + 9.1)
        rate = moment / float(self._steps)
        encoded = torch.log10(1.0 + rate / self._m_ref)
        return encoded.unsqueeze(1).expand(-1, self._steps)


def test_future_tail_does_not_change_a_released_horizon() -> None:
    config = _config()
    torch.manual_seed(11)
    model = PINNModel(config).eval()
    prefix = torch.randn(1, 1, 20)
    first = torch.cat((prefix, torch.zeros(1, 1, 180)), dim=-1)
    changed = torch.cat((prefix, torch.full((1, 1, 180), 1.0e4)), dim=-1)

    original = evaluate_delayed_prefix(model, config, [_batch(first)], horizons=(20,))
    modified = evaluate_delayed_prefix(model, config, [_batch(changed)], horizons=(20,))

    assert original["station_rows"][0]["mw_pred"] == pytest.approx(
        modified["station_rows"][0]["mw_pred"], rel=0.0, abs=0.0
    )
    assert original["station_rows"][0]["release_time_sec"] == 25.0


def test_event_medians_are_event_equal_in_horizon_metrics() -> None:
    config = _config()
    model = _FirstSampleMagnitudeModel(config).eval()
    radial = torch.zeros(3, 1, 200)
    radial[:, 0, 0] = torch.tensor([6.0, 8.0, 9.0])
    batch = _batch(
        radial,
        events=["A", "A", "B"],
        stations=["A1", "A2", "B1"],
        magnitudes=[7.5, 7.5, 8.0],
    )

    result = evaluate_delayed_prefix(model, config, [batch], horizons=(20,))
    events = {row["event"]: row for row in result["event_rows"]}
    metrics = result["horizon_metrics"][0]

    assert events["A"]["mw_pred_median"] == pytest.approx(7.0, abs=2.0e-5)
    assert events["A"]["error"] == pytest.approx(-0.5, abs=2.0e-5)
    assert events["B"]["error"] == pytest.approx(1.0, abs=2.0e-5)
    assert metrics["event_equal_mae"] == pytest.approx(0.75, abs=2.0e-5)
    assert metrics["event_equal_rmse"] == pytest.approx(
        math.sqrt(0.625), abs=2.0e-5
    )
    assert metrics["event_equal_bias"] == pytest.approx(0.25, abs=2.0e-5)
    assert metrics["coverage"] == 2
    assert metrics["coverage_fraction"] == 1.0


def test_full_window_matches_standard_stf_inference() -> None:
    config = _config()
    torch.manual_seed(29)
    model = PINNModel(config).eval()
    radial = torch.randn(2, 1, 200)
    batch = _batch(radial, magnitudes=[7.0, 8.0])

    result = evaluate_delayed_prefix(model, config, [batch], horizons=(200,))

    source_distance = batch["source_distance_m"]
    epicentral_distance = batch["epicentral_distance_m"]
    metadata_distance = metadata_distance_from_config(
        config,
        source_distance_m=source_distance,
        epicentral_distance_m=epicentral_distance,
    )
    metadata = build_metadata_tensor(
        metadata_distance,
        batch["theta_deg"],
        batch["azimuth_deg"],
    )
    with torch.no_grad():
        encoded = model(radial, meta=metadata)
        m_ref = float(config["dataset"]["stf"]["m_ref"])
        rate = torch.clamp(m_ref * (torch.pow(10.0, encoded) - 1.0), min=0.0)
        expected = moment_magnitude_from_rate(rate, batch["stf_dt_sec"])

    actual = torch.tensor([row["mw_pred"] for row in result["station_rows"]])
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert {row["release_time_sec"] for row in result["station_rows"]} == {205.0}
    assert result["cohort"] == {
        "cohort": "processed radial peak over the full 200 s record >= 2 cm",
        "radial_peak_min_cm": 2.0,
        "waveform_prefix_causal": True,
        "station_selection_causal": False,
        "end_to_end_causal": False,
    }


def test_pre_p_baseline_waits_until_its_source_window_is_available() -> None:
    config = _config()
    model = _FirstSampleMagnitudeModel(config).eval()
    radial = torch.zeros(2, 1, 200)
    radial[:, 0, 0] = torch.tensor([7.0, 8.0])
    batch = _batch(
        radial,
        events=["PRE_P", "PRE_EVENT"],
        stations=["P1", "E1"],
        magnitudes=[7.0, 8.0],
        baseline_sources=["pre_p", "pre_event"],
        source_distances_m=[40.0 * 7900.0, 100_000.0],
    )

    result = evaluate_delayed_prefix(model, config, [batch], horizons=(20, 40))
    rows_by_horizon = {
        horizon: [
            row
            for row in result["station_rows"]
            if row["observation_horizon_sec"] == horizon
        ]
        for horizon in (20.0, 40.0)
    }
    metrics = {
        row["observation_horizon_sec"]: row
        for row in result["horizon_metrics"]
    }

    assert [row["event"] for row in rows_by_horizon[20.0]] == ["PRE_EVENT"]
    assert {row["event"] for row in rows_by_horizon[40.0]} == {
        "PRE_EVENT",
        "PRE_P",
    }
    assert result["unavailable_rows"] == [
        {
            "event": "PRE_P",
            "station": "P1",
            "observation_horizon_sec": 20.0,
            "release_time_sec": 25.0,
            "reason": "pre_p_baseline_not_available",
            "baseline_ready_time_sec": 30.0,
        }
    ]
    assert metrics[20.0]["unavailable_station_count"] == 1
    assert metrics[20.0]["unavailable_reason_counts"] == {
        "pre_p_baseline_not_available": 1
    }
    assert metrics[40.0]["unavailable_station_count"] == 0


def test_rejects_workflow_head_non_radial_and_invalid_horizons() -> None:
    base = _config()
    model = PINNModel(base)
    loader = [_batch(torch.zeros(1, 1, 200))]

    workflow = copy.deepcopy(base)
    workflow["workflow"] = "station_random_shifted_stf"
    with pytest.raises(ValueError, match="workflow"):
        evaluate_delayed_prefix(model, workflow, loader, horizons=(20,))

    with_head = copy.deepcopy(base)
    with_head["model"]["predict_catalog_mw"] = True
    head_model = PINNModel(with_head)
    with pytest.raises(ValueError, match="catalog Mw head"):
        evaluate_delayed_prefix(head_model, with_head, loader, horizons=(20,))

    radial_tangential = copy.deepcopy(base)
    radial_tangential["model"]["input_components"] = ["radial", "tangential"]
    rt_model = PINNModel(radial_tangential)
    with pytest.raises(ValueError, match="R-only"):
        evaluate_delayed_prefix(rt_model, radial_tangential, loader, horizons=(20,))

    for horizons in ((0,), (201,), (40, 20), (20, 20), (True,)):
        with pytest.raises(ValueError, match="horizon"):
            evaluate_delayed_prefix(model, base, loader, horizons=horizons)
