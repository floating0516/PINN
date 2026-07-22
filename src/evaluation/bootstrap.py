"""Event-level bootstrap uncertainty for corrected evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


def _optional_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def load_event_predictions(csv_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                {
                    "event": str(row["event"]),
                    "mw_pred_median": float(row["mw_pred_median"]),
                    "mw_catalog": _optional_float(row.get("mw_catalog")),
                    "mw_stf_native": _optional_float(
                        row.get("mw_stf_native")
                    ),
                    "mechanism": str(row.get("mechanism", "")),
                }
            )
    return rows


def load_station_predictions(csv_path: str | Path) -> list[dict[str, Any]]:
    """Compatibility alias; corrected inputs are event-summary CSV files."""
    return load_event_predictions(csv_path)


def _validated_event_rows(
    rows: list[dict[str, Any]],
    *,
    reference_key: str,
) -> list[dict[str, Any]]:
    events = [str(row.get("event", "")) for row in rows]
    if len(events) != len(set(events)):
        raise ValueError("bootstrap requires exactly one row per event")

    validated: list[dict[str, Any]] = []
    for row, event in zip(rows, events):
        try:
            prediction = float(row["mw_pred_median"])
            reference = float(row[reference_key])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(prediction) or not math.isfinite(reference):
            continue
        validated.append(
            {
                "event": event,
                "prediction": prediction,
                "reference": reference,
                "mechanism": str(row.get("mechanism", "")),
            }
        )
    return validated


def _ci(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "ci_lower": float(np.percentile(values, 2.5)),
        "ci_upper": float(np.percentile(values, 97.5)),
    }


def bootstrap_event_metrics(
    rows: list[dict[str, Any]],
    n_bootstrap: int = 1000,
    seed: int = 42,
    agg: str = "event_rows",
    reference_key: str = "mw_catalog",
) -> dict[str, Any]:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    event_rows = _validated_event_rows(rows, reference_key=reference_key)
    if not event_rows:
        return {"error": "no valid events"}

    errors = np.asarray(
        [row["prediction"] - row["reference"] for row in event_rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        len(errors),
        size=(n_bootstrap, len(errors)),
    )
    sampled_errors = errors[sample_indices]
    mae_samples = np.mean(np.abs(sampled_errors), axis=1)
    rmse_samples = np.sqrt(np.mean(sampled_errors**2, axis=1))
    bias_samples = np.mean(sampled_errors, axis=1)

    result: dict[str, Any] = {
        "n_events": len(event_rows),
        "n_bootstrap": n_bootstrap,
        "aggregation": "event_rows",
        "reference": reference_key,
        "mae": _ci(mae_samples),
        "rmse": _ci(rmse_samples),
        "bias": _ci(bias_samples),
        "event_ci": [
            {
                "event": row["event"],
                "reference": row["reference"],
                "mean": row["prediction"],
                "std": 0.0,
                "ci_lower": row["prediction"],
                "ci_upper": row["prediction"],
                "mechanism": row["mechanism"],
            }
            for row in event_rows
        ],
    }

    by_mechanism: dict[str, np.ndarray] = {}
    for mechanism in sorted({row["mechanism"] for row in event_rows}):
        mechanism_errors = np.asarray(
            [
                row["prediction"] - row["reference"]
                for row in event_rows
                if row["mechanism"] == mechanism
            ],
            dtype=np.float64,
        )
        mechanism_indices = rng.integers(
            0,
            len(mechanism_errors),
            size=(n_bootstrap, len(mechanism_errors)),
        )
        by_mechanism[mechanism] = np.mean(
            np.abs(mechanism_errors[mechanism_indices]),
            axis=1,
        )
    result["by_mechanism"] = {
        mechanism: {
            "n_events": int(
                sum(row["mechanism"] == mechanism for row in event_rows)
            ),
            "mae": _ci(samples),
        }
        for mechanism, samples in by_mechanism.items()
    }
    return result


def compare_two_experiments(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    n_bootstrap: int = 1000,
    seed: int = 42,
    agg: str = "event_rows",
    reference_key: str = "mw_catalog",
) -> dict[str, Any]:
    events_a = {
        row["event"]: row
        for row in _validated_event_rows(rows_a, reference_key=reference_key)
    }
    events_b = {
        row["event"]: row
        for row in _validated_event_rows(rows_b, reference_key=reference_key)
    }
    common = sorted(set(events_a) & set(events_b))
    if not common:
        return {"error": "experiments have no common events"}

    errors_a = np.asarray(
        [events_a[event]["prediction"] - events_a[event]["reference"] for event in common]
    )
    errors_b = np.asarray(
        [events_b[event]["prediction"] - events_b[event]["reference"] for event in common]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(common), size=(n_bootstrap, len(common)))
    delta_mae = np.mean(np.abs(errors_a[indices]), axis=1) - np.mean(
        np.abs(errors_b[indices]),
        axis=1,
    )
    return {
        "n_common_events": len(common),
        "delta_mae_mean": float(np.mean(delta_mae)),
        "delta_mae_ci_lower": float(np.percentile(delta_mae, 2.5)),
        "delta_mae_ci_upper": float(np.percentile(delta_mae, 97.5)),
        "p_value_a_worse": float(np.mean(delta_mae > 0.0)),
    }


def format_results(result: dict[str, Any]) -> str:
    lines = [
        f"Bootstrap (n={result['n_bootstrap']}, unit=event)",
        f"Events: {result['n_events']}",
        "",
    ]
    for metric in ("mae", "rmse", "bias"):
        values = result[metric]
        lines.append(
            f"{metric.upper():>5s}: {values['mean']:.4f} "
            f"95% CI [{values['ci_lower']:.4f}, {values['ci_upper']:.4f}]"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Event-level bootstrap")
    parser.add_argument("--csv", required=True, help="event_summary.csv path")
    parser.add_argument("--csv-b", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference",
        choices=["mw_catalog", "mw_stf_native"],
        default="mw_catalog",
    )
    args = parser.parse_args()

    rows = load_event_predictions(args.csv)
    result = bootstrap_event_metrics(
        rows,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        reference_key=args.reference,
    )
    print(format_results(result))
    if args.csv_b:
        comparison = compare_two_experiments(
            rows,
            load_event_predictions(args.csv_b),
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            reference_key=args.reference,
        )
        print(comparison)


if __name__ == "__main__":
    main()
