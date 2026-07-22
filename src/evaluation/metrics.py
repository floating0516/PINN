from __future__ import annotations

import math
from typing import Any

import numpy as np


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _event_reference(rows: list[dict[str, Any]], key: str) -> float:
    values = _finite_values(rows, key)
    return float(np.median(values)) if values.size else float("nan")


def aggregate_event_predictions(
    station_rows: list[dict[str, Any]],
    *,
    reference_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in station_rows:
        grouped.setdefault(str(row["event"]), []).append(row)

    event_rows: list[dict[str, Any]] = []
    for event in sorted(grouped):
        rows = grouped[event]
        predictions = _finite_values(rows, "mw_pred")
        if not predictions.size:
            continue
        predicted_median = float(np.median(predictions))
        mw_catalog = _event_reference(rows, "mw_catalog")
        mw_stf_native = _event_reference(rows, "mw_stf_native")
        event_rows.append(
            {
                "event": event,
                "mw_pred_median": predicted_median,
                "mw_catalog": mw_catalog,
                "mw_stf_native": mw_stf_native,
                "error_vs_catalog": predicted_median - mw_catalog,
                "error_vs_stf_native": predicted_median - mw_stf_native,
                "n_stations": int(predictions.size),
                "pred_std": (
                    float(np.std(predictions))
                    if predictions.size > 1
                    else 0.0
                ),
                "pred_iqr": (
                    float(
                        np.percentile(predictions, 75)
                        - np.percentile(predictions, 25)
                    )
                    if predictions.size > 1
                    else 0.0
                ),
            }
        )
    return event_rows


def _error_metrics(
    rows: list[dict[str, Any]],
    *,
    prediction_key: str,
    reference_key: str,
) -> tuple[int, float, float, float]:
    errors: list[float] = []
    for row in rows:
        try:
            prediction = float(row[prediction_key])
            reference = float(row[reference_key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(prediction) and math.isfinite(reference):
            errors.append(prediction - reference)
    if not errors:
        return 0, float("nan"), float("nan"), float("nan")
    error_array = np.asarray(errors, dtype=np.float64)
    return (
        len(errors),
        float(np.mean(np.abs(error_array))),
        float(np.sqrt(np.mean(error_array**2))),
        float(np.mean(error_array)),
    )


def summarize_predictions(
    station_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    *,
    reference_key: str,
) -> dict[str, Any]:
    station_count, station_mae, station_rmse, station_bias = _error_metrics(
        station_rows,
        prediction_key="mw_pred",
        reference_key=reference_key,
    )
    event_count, event_mae, event_rmse, event_bias = _error_metrics(
        event_rows,
        prediction_key="mw_pred_median",
        reference_key=reference_key,
    )
    reference = {
        "mw_catalog": "catalog",
        "mw_stf_native": "stf_native",
    }.get(reference_key, reference_key)
    return {
        "reference": reference,
        "event_count": event_count,
        "station_count": station_count,
        "event_mae": event_mae,
        "event_rmse": event_rmse,
        "event_bias": event_bias,
        "station_mae": station_mae,
        "station_rmse": station_rmse,
        "station_bias": station_bias,
    }
