from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.data.records_v2 import NormalizedStationRecord
from src.data.stf import ProcessedSTF


MANIFEST_FIELDS = [
    "event_index",
    "event",
    "station",
    "accepted",
    "rejection_reason",
    "magnitude_catalog",
    "mw_stf_native",
    "has_stf",
    "epicentral_distance_km",
    "source_distance_km",
    "theta_deg",
    "azimuth_deg",
    "raw_dt_sec",
    "waveform_dt_sec",
    "valid_fraction",
    "baseline_source",
    "radial_peak_cm",
    "stf_retained_moment_fraction",
]

REJECTION_REASONS = {
    "blacklisted_event",
    "missing_station_coordinates",
    "invalid_waveform",
    "insufficient_baseline",
    "insufficient_valid_fraction",
    "below_radial_peak_threshold",
    "missing_stf",
    "stf_window_too_short",
    "invalid_geometry",
}


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sample_value(
    sample: dict[str, Any] | None,
    key: str,
) -> Any:
    return None if sample is None else sample.get(key)


def make_manifest_row(
    record: NormalizedStationRecord,
    *,
    accepted: bool,
    rejection_reason: str = "",
    sample: dict[str, Any] | None = None,
    processed_stf: ProcessedSTF | None = None,
    has_stf: bool | None = None,
) -> dict[str, Any]:
    if accepted:
        if rejection_reason:
            raise ValueError("accepted manifest rows cannot have a rejection")
    elif rejection_reason not in REJECTION_REASONS:
        raise ValueError(f"unknown rejection reason: {rejection_reason}")

    epicentral_distance_m = _finite_or_none(
        _sample_value(sample, "epicentral_distance_m")
    )
    source_distance_m = _finite_or_none(
        _sample_value(sample, "source_distance_m")
    )
    mw_stf_native = _finite_or_none(
        _sample_value(sample, "mw_stf_native")
    )
    retained_fraction = _finite_or_none(
        _sample_value(sample, "stf_retained_moment_fraction")
    )
    if processed_stf is not None:
        mw_stf_native = processed_stf.mw_native
        retained_fraction = processed_stf.retained_moment_fraction
    if has_stf is None and processed_stf is not None:
        has_stf = True

    return {
        "event_index": record.event_index,
        "event": record.event,
        "station": record.station,
        "accepted": accepted,
        "rejection_reason": rejection_reason,
        "magnitude_catalog": _finite_or_none(record.magnitude_catalog),
        "mw_stf_native": mw_stf_native,
        "has_stf": has_stf,
        "epicentral_distance_km": (
            None
            if epicentral_distance_m is None
            else epicentral_distance_m / 1000.0
        ),
        "source_distance_km": (
            None
            if source_distance_m is None
            else source_distance_m / 1000.0
        ),
        "theta_deg": _finite_or_none(_sample_value(sample, "theta_deg")),
        "azimuth_deg": _finite_or_none(
            _sample_value(sample, "azimuth_deg")
        ),
        "raw_dt_sec": _finite_or_none(_sample_value(sample, "raw_dt_sec")),
        "waveform_dt_sec": _finite_or_none(
            _sample_value(sample, "waveform_dt_sec")
        ),
        "valid_fraction": _finite_or_none(
            _sample_value(sample, "valid_fraction")
        ),
        "baseline_source": _sample_value(sample, "baseline_source") or "",
        "radial_peak_cm": _finite_or_none(
            _sample_value(sample, "radial_peak_cm")
        ),
        "stf_retained_moment_fraction": retained_fraction,
    }


def validate_manifest_rows(rows: list[dict[str, Any]]) -> None:
    expected_fields = set(MANIFEST_FIELDS)
    for index, row in enumerate(rows):
        if set(row) != expected_fields:
            missing = sorted(expected_fields - set(row))
            extra = sorted(set(row) - expected_fields)
            raise ValueError(
                f"manifest row {index} fields differ: "
                f"missing={missing}, extra={extra}"
            )
        if not isinstance(row["accepted"], bool):
            raise ValueError(f"manifest row {index} accepted must be boolean")
        reason = row["rejection_reason"]
        if row["accepted"] and reason:
            raise ValueError(
                f"accepted manifest row {index} has rejection reason"
            )
        if not row["accepted"] and reason not in REJECTION_REASONS:
            raise ValueError(f"unknown rejection reason: {reason}")


def _event_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_count = sum(bool(row["accepted"]) for row in rows)
    rejection_counts = Counter(
        str(row["rejection_reason"])
        for row in rows
        if not row["accepted"]
    )
    return {
        "event_index": int(rows[0]["event_index"]),
        "candidate_station_count": len(rows),
        "accepted_station_count": accepted_count,
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }


def _one_stf_per_event(samples: list[dict[str, Any]]) -> bool:
    if not samples or not all(bool(sample["has_stf"]) for sample in samples):
        return False
    by_event: dict[str, list[np.ndarray]] = {}
    for sample in samples:
        by_event.setdefault(str(sample["event"]), []).append(sample["stf"])
    return all(
        all(np.array_equal(values[0], value) for value in values[1:])
        for values in by_event.values()
    )


def _one_stf_mw_per_event(samples: list[dict[str, Any]]) -> bool:
    if not samples or not all(bool(sample["has_stf"]) for sample in samples):
        return False
    by_event: dict[str, list[float]] = {}
    for sample in samples:
        by_event.setdefault(str(sample["event"]), []).append(
            float(sample["mw_stf_native"])
        )
    return all(
        all(
            math.isfinite(value)
            and math.isclose(value, values[0], rel_tol=0.0, abs_tol=0.0)
            for value in values
        )
        for values in by_event.values()
    )


def build_dataset_summary(dataset: Any) -> dict[str, Any]:
    rows = list(dataset.manifest_rows)
    validate_manifest_rows(rows)
    accepted_rows = [row for row in rows if row["accepted"]]
    rows_by_event: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_event.setdefault(str(row["event"]), []).append(row)

    samples = list(dataset.samples)
    retained_fractions = [
        float(sample["stf_retained_moment_fraction"])
        for sample in samples
        if bool(sample["has_stf"])
        and math.isfinite(float(sample["stf_retained_moment_fraction"]))
    ]
    all_waveform_dt_equal_1s = bool(samples) and all(
        math.isclose(
            float(sample["waveform_dt_sec"]),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for sample in samples
    )
    rejection_counts = Counter(
        str(row["rejection_reason"])
        for row in rows
        if not row["accepted"]
    )
    return {
        "candidate_event_count": len(rows_by_event),
        "accepted_event_count": len(
            {str(row["event"]) for row in accepted_rows}
        ),
        "candidate_station_count": len(rows),
        "accepted_station_count": len(accepted_rows),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "events": {
            event: _event_summary(event_rows)
            for event, event_rows in rows_by_event.items()
        },
        "invariants": {
            "all_waveform_dt_equal_1s": all_waveform_dt_equal_1s,
            "one_stf_per_event": _one_stf_per_event(samples),
            "one_stf_mw_per_event": _one_stf_mw_per_event(samples),
            "min_stf_retained_fraction": (
                min(retained_fractions) if retained_fractions else 0.0
            ),
        },
    }


def audit_passes(
    summary: dict[str, Any],
    *,
    minimum_stf_retained_fraction: float,
) -> bool:
    invariants = summary["invariants"]
    return (
        invariants["all_waveform_dt_equal_1s"] is True
        and invariants["one_stf_per_event"] is True
        and invariants["one_stf_mw_per_event"] is True
        and math.isfinite(float(invariants["min_stf_retained_fraction"]))
        and float(invariants["min_stf_retained_fraction"])
        >= minimum_stf_retained_fraction
    )


def write_dataset_audit(
    dataset: Any,
    *,
    manifest_path: str | Path,
    summary_path: str | Path,
    minimum_stf_retained_fraction: float,
) -> dict[str, Any]:
    if (
        not math.isfinite(minimum_stf_retained_fraction)
        or not 0.0 < minimum_stf_retained_fraction <= 1.0
    ):
        raise ValueError(
            "minimum_stf_retained_fraction must be finite and in (0, 1]"
        )
    rows = list(dataset.manifest_rows)
    validate_manifest_rows(rows)
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = build_dataset_summary(dataset)
    destination = Path(summary_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(
            summary,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    return summary
