from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline.event_amplitude import (  # noqa: E402
    RadialEventRidgeSpec,
    RadialStationObservation,
    fit_radial_event_ridge,
    leave_one_event_out_predictions,
    predict_grouped_events,
)
from src.data.dataset_v2 import CorrectedEarthquakeDataset  # noqa: E402
from src.data.loaders_v2 import _runtime_config  # noqa: E402
from src.utils.provenance import (  # noqa: E402
    current_git_commit,
    git_is_dirty,
    sha256_file,
    utc_now_iso,
)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite result artifact: {path}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize an empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _training_observations(config_path: Path) -> tuple[
    list[RadialStationObservation],
    dict[str, Any],
]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must contain a mapping")
    dataset = CorrectedEarthquakeDataset(_runtime_config(config))
    observations = [
        RadialStationObservation(
            event=str(sample["event"]),
            station=str(sample["station"]),
            radial_peak_cm=float(sample["radial_peak_cm"]),
            source_distance_km=float(sample["source_distance_m"]) / 1000.0,
            magnitude=float(sample["magnitude_catalog"]),
        )
        for sample in dataset.samples
    ]
    return observations, config


def _external_observations(path: Path) -> list[RadialStationObservation]:
    observations: list[RadialStationObservation] = []
    seen: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["event"], row["station"])
            if key in seen:
                raise ValueError(f"duplicate external event/station row: {key}")
            seen.add(key)
            observations.append(
                RadialStationObservation(
                    event=row["event"],
                    station=row["station"],
                    radial_peak_cm=float(row["max_radial_cm"]),
                    source_distance_km=float(row["source_distance_km"]),
                    magnitude=float(row["mw_catalog"]),
                )
            )
    return observations


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = np.asarray([float(row["error"]) for row in rows], dtype=np.float64)
    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("prediction rows must contain finite errors")
    return {
        "event_count": int(errors.size),
        "event_mae": float(np.mean(np.abs(errors))),
        "event_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "event_bias": float(np.mean(errors)),
    }


def run(
    *,
    training_config_path: Path,
    external_station_csv: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    training_observations, training_config = _training_observations(
        training_config_path
    )
    spec = RadialEventRidgeSpec()
    model = fit_radial_event_ridge(training_observations, spec)
    loeo_rows = leave_one_event_out_predictions(training_observations, spec)
    external_observations = _external_observations(external_station_csv)
    external_rows = predict_grouped_events(model, external_observations)
    loeo_metrics = _metric_summary(loeo_rows)
    external_metrics = _metric_summary(external_rows)
    if external_metrics["event_count"] != 8:
        raise ValueError(
            "formal external evaluation requires exactly eight events"
        )

    data_path = Path(training_config["paths"]["data_path"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    model_payload = {
        **model.to_dict(),
        "created_at_utc": utc_now_iso(),
        "training_config_path": str(training_config_path.resolve()),
        "training_config_sha256": sha256_file(training_config_path),
        "training_data_path": str(data_path.resolve()),
        "training_data_sha256": sha256_file(data_path),
    }
    summary = {
        "status": "complete",
        "created_at_utc": utc_now_iso(),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "git_dirty": git_is_dirty(PROJECT_ROOT),
        "method": "radial_event_ridge_v1",
        "training_event_count": len(model.training_events),
        "training_station_count": model.training_observation_count,
        "internal_loeo": loeo_metrics,
        "external": external_metrics,
        "target_event_mae": 0.15,
        "target_passed": (
            external_metrics["event_count"] >= 5
            and external_metrics["event_mae"] <= 0.15
        ),
        "external_station_csv": str(external_station_csv.resolve()),
        "external_station_csv_sha256": sha256_file(external_station_csv),
        "artifact_sha256": {},
    }

    output_root.mkdir(parents=True)
    model_path = output_root / "model.json"
    loeo_path = output_root / "internal_loeo_predictions.csv"
    external_path = output_root / "external_event_predictions.csv"
    _atomic_write(model_path, _json_bytes(model_payload))
    _atomic_write(loeo_path, _csv_bytes(loeo_rows))
    _atomic_write(external_path, _csv_bytes(external_rows))
    summary["artifact_sha256"] = {
        "model.json": sha256_file(model_path),
        "internal_loeo_predictions.csv": sha256_file(loeo_path),
        "external_event_predictions.csv": sha256_file(external_path),
    }
    _atomic_write(output_root / "summary.json", _json_bytes(summary))
    _atomic_write(output_root / "COMPLETE", b"\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate the R-only event amplitude ridge head."
    )
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--external-station-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        training_config_path=args.training_config.resolve(),
        external_station_csv=args.external_station_csv.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
