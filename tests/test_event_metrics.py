import pytest
import torch

from src.evaluation.bootstrap import bootstrap_event_metrics
from src.evaluation.evaluate import (
    _evaluation_time_steps,
    _magnitude_from_rate,
    _predict_outputs,
)
from src.evaluation.metrics import (
    aggregate_event_predictions,
    summarize_predictions,
)
from src.models.model import PINNPrediction


def test_dense_event_does_not_dominate_event_mae() -> None:
    rows = [
        {"event": "A", "mw_pred": 8.0, "mw_catalog": 7.0},
        {"event": "A", "mw_pred": 8.0, "mw_catalog": 7.0},
        {"event": "A", "mw_pred": 8.0, "mw_catalog": 7.0},
        {"event": "B", "mw_pred": 7.0, "mw_catalog": 7.0},
    ]

    events = aggregate_event_predictions(rows, reference_key="mw_catalog")
    metrics = summarize_predictions(
        rows,
        events,
        reference_key="mw_catalog",
    )

    assert metrics["event_mae"] == pytest.approx(0.5)
    assert metrics["station_mae"] == pytest.approx(0.75)


def test_event_prediction_is_station_median() -> None:
    rows = [
        {"event": "A", "mw_pred": 6.0, "mw_catalog": 7.0},
        {"event": "A", "mw_pred": 7.0, "mw_catalog": 7.0},
        {"event": "A", "mw_pred": 10.0, "mw_catalog": 7.0},
    ]

    events = aggregate_event_predictions(rows, reference_key="mw_catalog")

    assert events[0]["mw_pred_median"] == pytest.approx(7.0)
    assert events[0]["error_vs_catalog"] == pytest.approx(0.0)


def test_event_rows_keep_catalog_and_stf_references_separate() -> None:
    rows = [
        {
            "event": "A",
            "mw_pred": 7.2,
            "mw_catalog": 7.0,
            "mw_stf_native": 6.8,
        }
    ]

    event = aggregate_event_predictions(rows, reference_key="mw_catalog")[0]

    assert event["mw_catalog"] == pytest.approx(7.0)
    assert event["mw_stf_native"] == pytest.approx(6.8)
    assert event["error_vs_catalog"] == pytest.approx(0.2)
    assert event["error_vs_stf_native"] == pytest.approx(0.4)
    assert "mw_true" not in event


def test_bootstrap_rejects_duplicate_event_rows() -> None:
    rows = [
        {"event": "A", "mw_pred_median": 7.0, "mw_catalog": 7.0},
        {"event": "A", "mw_pred_median": 7.1, "mw_catalog": 7.0},
    ]

    with pytest.raises(ValueError, match="one row per event"):
        bootstrap_event_metrics(rows, n_bootstrap=10)


def test_v2_magnitude_conversion_needs_no_legacy_physics_config() -> None:
    rate = torch.tensor([[1.0e18, 2.0e18]])

    magnitude = _magnitude_from_rate(
        rate,
        torch.tensor([1.0]),
        pipeline_version=2,
        legacy_criterion=None,
    )

    expected = (2.0 / 3.0) * (torch.log10(torch.tensor(3.0e18)) - 9.1)
    assert magnitude[0] == pytest.approx(float(expected))


def test_v2_evaluation_keeps_waveform_and_stf_lengths_separate() -> None:
    config = {
        "pipeline_version": 2,
        "dataset": {
            "sample_rate_hz": 2.0,
            "waveform": {"duration_sec": 3.0},
            "stf": {"duration_sec": 5.0},
        },
        "training": {},
    }

    assert _evaluation_time_steps(config) == (6, 10)


def test_active_evaluation_uses_three_hundred_step_station_window() -> None:
    config = {
        "pipeline_version": 2,
        "workflow": "station_random_shifted_stf",
        "dataset": {
            "sample_rate_hz": 1.0,
            "waveform": {"duration_sec": 200.0},
            "stf": {
                "duration_sec": 200.0,
                "station_window_duration_sec": 300.0,
            },
        },
        "training": {},
    }

    assert _evaluation_time_steps(config) == (200, 300)


def test_active_evaluation_uses_scalar_mw_and_keeps_window_diagnostic() -> None:
    class ScalarHeadModel:
        def predict_heads(self, waveform, meta=None):
            del meta
            return PINNPrediction(
                stf_encoded=torch.zeros(waveform.shape[0], 300),
                catalog_mw=torch.full((waveform.shape[0],), 7.25),
            )

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("active evaluation must use predict_heads")

    output = _predict_outputs(
        ScalarHeadModel(),
        torch.zeros(2, 1, 200),
        meta=torch.zeros(2, 5),
        stf_m_ref=1.0e18,
        source_dt_sec=torch.ones(2),
        pipeline_version=2,
        active_catalog_head=True,
        legacy_criterion=None,
    )

    assert torch.equal(output.mw_pred, torch.tensor([7.25, 7.25]))
    torch.testing.assert_close(
        output.mw_window_pred,
        torch.tensor([0.6, 0.6]),
    )
    assert output.rate_nm_per_s.shape == (2, 300)
