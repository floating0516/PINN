import pytest

from src.evaluation.bootstrap import bootstrap_event_metrics
from src.evaluation.metrics import (
    aggregate_event_predictions,
    summarize_predictions,
)


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
