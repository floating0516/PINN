#!/usr/bin/env python3
"""Run the frozen Phase73 checkpoint on the locked eight-event cohort.

This evaluator is intentionally external-only. It uses the frozen Phase39
proposal-rate cube, rebuilds Phase73's causal E/N/U Crowell hints, and never
loads the internal or grouped-test cohorts.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.evaluate_phase39_second_by_second import (  # noqa: E402
    EXTERNAL_EVENT_ROOT,
    EXPECTED_EVENT_COUNT,
    EXPECTED_STATION_COUNT,
    LABELS_PATH,
    _git_commit,
    _resolve_device,
    hash_external_inputs,
    load_endpoint_reference,
    load_label_contract,
    load_phase39_config,
    validate_frozen_artifacts,
)
from scripts.evaluation.evaluate_phase39_streaming_replay import (  # noqa: E402
    RawStreamingStation,
    build_raw_streaming_records,
)
from scripts.evaluation.evaluate_phase45_posthoc_streaming import (  # noqa: E402
    ENDPOINT_PREDICTION_TOLERANCE_MW,
    EXPECTED_PHASE39_SHA256,
    _endpoint_external_gate,
    build_comparison_tables,
)
from scripts.evaluation.evaluate_phase66_stateful_streaming import (  # noqa: E402
    DEFAULT_EXTERNAL_RAW_RATES,
    EXPECTED_EXTERNAL_RAW_SHA256,
    _load_frozen_rates,
)
from scripts.experiments.run_phase43_streaming_adapter import (  # noqa: E402
    HORIZONS,
    PROCESSING_DELAY_SEC,
)
from scripts.plotting import (  # noqa: E402
    plot_phase73_stateful_validation_zh as phase73_validation,
)
from src.baseline.causal_pgd import RawPGDRecord, causal_pgd_3d  # noqa: E402
from src.baseline.scaling_laws import AVAILABLE_SCALING_LAWS  # noqa: E402
from src.evaluation.evaluate_unseen import load_event_bundle  # noqa: E402
from src.training.loss_stf_rate_v2 import moment_magnitude_from_rate  # noqa: E402
from src.utils.provenance import git_is_dirty, sha256_file  # noqa: E402


PROJECT_HOME = PROJECT_ROOT.parent.parent
DEFAULT_RUN_ROOT = phase73_validation.DEFAULT_RUN_ROOT
DEFAULT_CACHE_ROOT = phase73_validation.DEFAULT_CACHE_ROOT
DEFAULT_HINT_CACHE_ROOT = phase73_validation.DEFAULT_HINT_CACHE_ROOT
EXTERNAL_SPLIT = "external_development_validation"
EXPECTED_PROCESSING_DELAY_SEC = 6.0
DEFAULT_BATCH_SIZE = 32


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    context = (
        gzip.open(path, mode="wt", encoding="utf-8", newline="")
        if path.suffix == ".gz"
        else path.open(mode="w", encoding="utf-8", newline="")
    )
    with context as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if row.get(key) is None else row.get(key)
                    for key in fieldnames
                }
            )


def _validate_new_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output directory must be new or empty: {path}")


def build_external_pgd_records(
    records: Sequence[RawStreamingStation],
    *,
    event_root: Path = EXTERNAL_EVENT_ROOT,
) -> list[RawPGDRecord]:
    """Bind the locked radial replay records to their original E/N/U samples."""
    if not records:
        raise ValueError("external PGD cohort is empty")
    by_event_dir: dict[str, list[RawStreamingStation]] = {}
    for record in records:
        by_event_dir.setdefault(str(record.event_dir), []).append(record)

    raw_by_key: dict[tuple[str, str], RawPGDRecord] = {}
    for event_dir, event_records in sorted(by_event_dir.items()):
        bundle = load_event_bundle(event_root / event_dir)
        stations = {str(station.station): station for station in bundle.stations}
        if len(stations) != len(bundle.stations):
            raise ValueError(f"external bundle has duplicate stations: {event_dir}")
        for record in event_records:
            key = (str(record.event), str(record.station))
            station = stations.get(str(record.station))
            if station is None:
                raise ValueError(f"external PGD station is missing from bundle: {key}")
            arrays = (
                np.asarray(station.t, dtype=np.float64),
                np.asarray(station.e_m, dtype=np.float64),
                np.asarray(station.n_m, dtype=np.float64),
                np.asarray(station.u_m, dtype=np.float64),
            )
            if any(array.ndim != 1 for array in arrays) or len(
                {array.size for array in arrays}
            ) != 1:
                raise ValueError(f"invalid external E/N/U arrays: {key}")
            if key in raw_by_key:
                raise ValueError(f"duplicate external PGD record: {key}")
            raw_by_key[key] = RawPGDRecord(
                event=key[0],
                station=key[1],
                time_sec=arrays[0],
                east_m=arrays[1],
                north_m=arrays[2],
                up_m=arrays[3],
                source_distance_km=float(record.source_distance_m) / 1_000.0,
                p_arrival_sec=float(record.p_arrival_sec),
                magnitude_catalog=float(record.magnitude_catalog),
            )

    expected = {(str(record.event), str(record.station)) for record in records}
    if set(raw_by_key) != expected:
        raise ValueError("external PGD key set differs from radial replay cohort")
    return [raw_by_key[(str(record.event), str(record.station))] for record in records]


def build_causal_hint_cube(
    records: Sequence[RawPGDRecord],
    *,
    horizons: Sequence[int] = HORIZONS,
    processing_delay_sec: float = PROCESSING_DELAY_SEC,
) -> dict[str, np.ndarray]:
    """Build exactly the Phase67 causal Crowell and P-arrival hint tensors."""
    horizon_values = tuple(int(value) for value in horizons)
    if not records or not horizon_values:
        raise ValueError("causal hint cube needs records and horizons")
    if not np.isclose(processing_delay_sec, EXPECTED_PROCESSING_DELAY_SEC):
        raise ValueError("Phase73 external hint delay must remain six seconds")

    pgd_3d_m = np.empty((len(records), len(horizon_values)), dtype=np.float32)
    p_arrived = np.empty((len(records), len(horizon_values)), dtype=np.bool_)
    for station_index, record in enumerate(records):
        for horizon_index, horizon in enumerate(horizon_values):
            pgd_m, _, _ = causal_pgd_3d(
                record,
                observation_horizon_sec=horizon,
                processing_delay_sec=processing_delay_sec,
            )
            pgd_3d_m[station_index, horizon_index] = pgd_m
            p_arrived[station_index, horizon_index] = (
                float(horizon) >= float(record.p_arrival_sec)
            )
    if not bool(np.all(np.isfinite(pgd_3d_m))) or bool(np.any(pgd_3d_m <= 0.0)):
        raise ValueError("external causal PGD hint is invalid")
    source_distance_m = np.asarray(
        [record.source_distance_km * 1_000.0 for record in records],
        dtype=np.float64,
    )
    crowell_mw = phase73_validation._scaling_law_cube(
        pgd_3d_m,
        source_distance_m,
        AVAILABLE_SCALING_LAWS["crowell"],
    )
    if not bool(np.all(np.isfinite(crowell_mw))):
        raise ValueError("external Crowell hint is invalid")
    return {
        "pgd_3d_m": pgd_3d_m,
        "crowell_mw": crowell_mw,
        "p_arrived": p_arrived,
    }


def _validate_external_inputs(
    *,
    raw_rates_path: Path,
) -> tuple[
    list[RawStreamingStation],
    np.ndarray,
    str,
    dict[str, Any],
    dict[str, str],
]:
    external_artifact_hashes = validate_frozen_artifacts()
    external_input_hashes = hash_external_inputs(EXTERNAL_EVENT_ROOT)
    config = load_phase39_config()
    labels = load_label_contract(LABELS_PATH)
    endpoint_reference = load_endpoint_reference()
    records = build_raw_streaming_records(
        config=config,
        event_root=EXTERNAL_EVENT_ROOT,
        labels_by_dir=labels,
        expected_station_keys=set(endpoint_reference["station_predictions"]),
    )
    if len(records) != EXPECTED_STATION_COUNT:
        raise ValueError("Phase73 external station count changed")
    if len({record.event for record in records}) != EXPECTED_EVENT_COUNT:
        raise ValueError("Phase73 external event count changed")
    raw_rates, raw_rates_sha256 = _load_frozen_rates(
        raw_rates_path,
        expected_sha256=EXPECTED_EXTERNAL_RAW_SHA256,
        expected_station_count=EXPECTED_STATION_COUNT,
    )
    available_mask = np.all(np.isfinite(raw_rates), axis=2)
    if not bool(np.all(available_mask)):
        raise ValueError("external Phase39 proposal cube must be complete")
    return (
        records,
        raw_rates,
        raw_rates_sha256,
        endpoint_reference,
        {
            "phase39_frozen_artifacts": external_artifact_hashes,
            "external_inputs": external_input_hashes,
        },
    )


def _validate_phase39_endpoint(
    *,
    raw_rates: np.ndarray,
    records: Sequence[RawStreamingStation],
    endpoint_reference: Mapping[str, Any],
) -> dict[str, float]:
    available_mask = np.all(np.isfinite(raw_rates), axis=2)
    source_dt = np.ones(len(records), dtype=np.float32)
    tables = build_comparison_tables(
        raw_rates=raw_rates,
        adapted_rates=raw_rates,
        available_mask=available_mask,
        events=[record.event for record in records],
        stations=[record.station for record in records],
        catalogs=np.asarray(
            [record.magnitude_catalog for record in records], dtype=np.float32
        ),
        source_dt_sec=source_dt,
        include_station_trajectories=False,
    )
    endpoint_gate = _endpoint_external_gate(
        tables["event_rows"],
        tables["endpoint_station_rows"],
        endpoint_reference,
    )
    if (
        endpoint_gate["max_station_prediction_abs_diff_mw"]
        > ENDPOINT_PREDICTION_TOLERANCE_MW
    ):
        raise ValueError("Phase39 external endpoint reproduction failed")
    return endpoint_gate


def replay_phase73_external(
    *,
    model: torch.nn.Module,
    raw_rates: np.ndarray,
    pgd_mw: np.ndarray,
    pgd_valid: np.ndarray,
    source_distance_m: np.ndarray,
    source_dt_sec: np.ndarray,
    beta_m_per_s: float,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Apply the model-internal Phase73 recurrence to one fixed external cube."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cube = np.asarray(raw_rates)
    station_count = cube.shape[1]
    expected_shape = (station_count, len(HORIZONS))
    if cube.shape[0] != len(HORIZONS) or cube.ndim != 3:
        raise ValueError("Phase73 external proposal cube horizon shape changed")
    if np.asarray(pgd_mw).shape != expected_shape or np.asarray(pgd_valid).shape != expected_shape:
        raise ValueError("Phase73 external hint shape changed")
    if np.asarray(source_distance_m).shape != (station_count,) or np.asarray(
        source_dt_sec
    ).shape != (station_count,):
        raise ValueError("Phase73 external source metadata shape changed")

    parameter = next(model.parameters())
    device = parameter.device
    phase73_mw = np.empty(expected_shape, dtype=np.float32)
    phase39_mw = np.empty(expected_shape, dtype=np.float32)
    plateau_confidence = np.empty(expected_shape, dtype=np.float32)
    revision_mw = np.empty(expected_shape, dtype=np.float32)
    proposal_assimilation_mw = np.empty(expected_shape, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, station_count, batch_size):
            stop = min(start + batch_size, station_count)
            raw_batch = np.asarray(cube[:, start:stop, :]).transpose(1, 0, 2)
            raw_tensor = torch.as_tensor(
                raw_batch.copy(),
                device=device,
                dtype=parameter.dtype,
            )
            source_dt = torch.as_tensor(
                np.asarray(source_dt_sec[start:stop]).copy(),
                device=device,
                dtype=parameter.dtype,
            )
            _, _, state_mw, _, diagnostics = model.stream_sequence_from_rates(
                raw_tensor,
                horizons_sec=HORIZONS,
                source_distance_m=torch.as_tensor(
                    np.asarray(source_distance_m[start:stop]).copy(),
                    device=device,
                    dtype=parameter.dtype,
                ),
                source_dt_sec=source_dt,
                beta_m_per_s=beta_m_per_s,
                pgd_mw_hint=torch.as_tensor(
                    np.asarray(pgd_mw[start:stop]).copy(),
                    device=device,
                    dtype=parameter.dtype,
                ),
                pgd_valid_hint=torch.as_tensor(
                    np.asarray(pgd_valid[start:stop]).copy(),
                    device=device,
                    dtype=torch.bool,
                ),
                return_diagnostics=True,
            )
            raw_mw = moment_magnitude_from_rate(
                raw_tensor.reshape(-1, raw_tensor.shape[-1]),
                source_dt.reshape(-1, 1)
                .expand(-1, raw_tensor.shape[1])
                .reshape(-1),
            ).reshape(raw_tensor.shape[0], raw_tensor.shape[1])
            phase73_mw[start:stop] = state_mw.cpu().numpy().astype(
                np.float32, copy=False
            )
            phase39_mw[start:stop] = raw_mw.cpu().numpy().astype(
                np.float32, copy=False
            )
            for name, output in (
                ("plateau_confidence", plateau_confidence),
                ("revision_mw", revision_mw),
                ("proposal_assimilation_mw", proposal_assimilation_mw),
            ):
                if name not in diagnostics:
                    raise ValueError(f"Phase73 diagnostic is missing: {name}")
                output[start:stop] = diagnostics[name].cpu().numpy().astype(
                    np.float32, copy=False
                )
    outputs = {
        "phase73_mw": phase73_mw,
        "phase39_mw": phase39_mw,
        "plateau_confidence": plateau_confidence,
        "revision_mw": revision_mw,
        "proposal_assimilation_mw": proposal_assimilation_mw,
    }
    if not all(bool(np.all(np.isfinite(value))) for value in outputs.values()):
        raise ValueError("Phase73 external replay produced a nonfinite value")
    return outputs


def build_external_tables(
    *,
    records: Sequence[RawStreamingStation],
    phase73_mw: np.ndarray,
    phase39_mw: np.ndarray,
    pgd_3d_m: np.ndarray,
    crowell_mw: np.ndarray,
    p_arrived: np.ndarray,
    plateau_confidence: np.ndarray,
    revision_mw: np.ndarray,
    proposal_assimilation_mw: np.ndarray,
) -> dict[str, Any]:
    """Aggregate station estimates into the shared Phase73 report schema."""
    station_count = len(records)
    expected_shape = (station_count, len(HORIZONS))
    arrays = (
        phase73_mw,
        phase39_mw,
        pgd_3d_m,
        crowell_mw,
        p_arrived,
        plateau_confidence,
        revision_mw,
        proposal_assimilation_mw,
    )
    if any(np.asarray(value).shape != expected_shape for value in arrays):
        raise ValueError("external output table arrays are not aligned")
    source_distance_m = np.asarray(
        [record.source_distance_m for record in records], dtype=np.float64
    )
    method_cubes = {
        "phase73": np.asarray(phase73_mw, dtype=np.float32),
        "phase39": np.asarray(phase39_mw, dtype=np.float32),
        "crowell": np.asarray(crowell_mw, dtype=np.float32),
        "melgar": phase73_validation._scaling_law_cube(
            pgd_3d_m,
            source_distance_m,
            AVAILABLE_SCALING_LAWS["melgar"],
        ),
        "ruhl": phase73_validation._scaling_law_cube(
            pgd_3d_m,
            source_distance_m,
            AVAILABLE_SCALING_LAWS["ruhl"],
        ),
    }
    events = [str(record.event) for record in records]
    stations = [str(record.station) for record in records]
    catalogs = np.asarray(
        [record.magnitude_catalog for record in records], dtype=np.float64
    )
    event_names, event_catalogs, station_counts, event_cubes = (
        phase73_validation._event_cubes(
            method_cubes,
            events=events,
            catalogs=catalogs,
        )
    )
    if len(event_names) != EXPECTED_EVENT_COUNT:
        raise ValueError("external event aggregation count changed")
    event_rows = phase73_validation.build_event_rows(
        split=EXTERNAL_SPLIT,
        event_names=event_names,
        event_catalogs=event_catalogs,
        station_counts=station_counts,
        event_cubes=event_cubes,
    )
    horizon_rows = phase73_validation._horizon_metric_rows(
        split=EXTERNAL_SPLIT,
        catalogs=catalogs,
        method_cubes=method_cubes,
        event_catalogs=event_catalogs,
        event_cubes=event_cubes,
    )
    diagnostics = phase73_validation.trajectory_diagnostics(
        event_rows,
        split=EXTERNAL_SPLIT,
    )
    station_rows: list[dict[str, Any]] = []
    for station_index, (event, station) in enumerate(zip(events, stations, strict=True)):
        catalog = float(catalogs[station_index])
        for horizon_index, horizon in enumerate(HORIZONS):
            row: dict[str, Any] = {
                "split": EXTERNAL_SPLIT,
                "event": event,
                "station": station,
                "observation_horizon_sec": int(horizon),
                "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                "mw_catalog": catalog,
                "pgd_3d_m": float(pgd_3d_m[station_index, horizon_index]),
                "p_arrived": bool(p_arrived[station_index, horizon_index]),
                "plateau_confidence": float(
                    plateau_confidence[station_index, horizon_index]
                ),
                "revision_mw": float(revision_mw[station_index, horizon_index]),
                "proposal_assimilation_mw": float(
                    proposal_assimilation_mw[station_index, horizon_index]
                ),
            }
            for method in phase73_validation.METHOD_ORDER:
                prediction = float(method_cubes[method][station_index, horizon_index])
                row[f"{method}_mw_pred"] = prediction
                row[f"{method}_error_mw"] = prediction - catalog
                row[f"{method}_abs_error_mw"] = abs(prediction - catalog)
            station_rows.append(row)
    endpoint_rows = [
        row for row in station_rows if int(row["observation_horizon_sec"]) == 200
    ]
    return {
        "method_cubes": method_cubes,
        "event_cubes": event_cubes,
        "event_names": event_names,
        "event_catalogs": event_catalogs,
        "station_counts": station_counts,
        "event_rows": event_rows,
        "horizon_rows": horizon_rows,
        "station_rows": station_rows,
        "endpoint_station_rows": endpoint_rows,
        "trajectory_diagnostics": diagnostics,
    }


def summarize_trajectory_metrics(
    event_cubes: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Summarize model-internal stability in the same late horizons as Phase73."""
    horizons = np.asarray(HORIZONS)
    post120 = horizons >= 120
    post160 = horizons >= 160
    output: dict[str, dict[str, float]] = {}
    for method, cube in event_cubes.items():
        values = np.asarray(cube, dtype=np.float64)
        later = values[:, post120]
        steps = np.diff(later, axis=1)
        peak_to_final = np.max(later, axis=1) - later[:, -1]
        bands = np.max(values[:, post160], axis=1) - np.min(values[:, post160], axis=1)
        output[method] = {
            "post120_abs_step_p95_mw": float(np.percentile(np.abs(steps), 95)),
            "post120_abs_step_max_mw": float(np.max(np.abs(steps))),
            "peak_to_final_p95_mw": float(np.percentile(peak_to_final, 95)),
            "post160_band_width_p95_mw": float(np.percentile(bands, 95)),
            "start_to_end_increase_fraction": float(
                np.mean(values[:, -1] >= values[:, 0])
            ),
        }
    return output


def _metric_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {
        (int(row["observation_horizon_sec"]), str(row["method"])): row
        for row in rows
    }


def evaluate_external(
    *,
    output_dir: Path,
    run_root: Path,
    cache_root: Path,
    hint_cache_root: Path,
    raw_rates_path: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Run the one frozen external Phase73 evaluation and save its artifacts."""
    _validate_new_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not np.isclose(PROCESSING_DELAY_SEC, EXPECTED_PROCESSING_DELAY_SEC):
        raise ValueError("Phase43 processing delay changed from Phase73 contract")

    _, _, config, model, source = phase73_validation._load_selected_source(
        run_root=run_root,
        cache_root=cache_root,
        hint_cache_root=hint_cache_root,
        device=device,
    )
    historical_campaign = _read_json(run_root / "campaign_summary.json")
    if historical_campaign.get("passed") is not False:
        raise ValueError("Phase73 historical validation gate unexpectedly passed")
    (
        records,
        raw_rates,
        raw_rates_sha256,
        endpoint_reference,
        external_provenance,
    ) = _validate_external_inputs(raw_rates_path=raw_rates_path)
    endpoint_gate = _validate_phase39_endpoint(
        raw_rates=raw_rates,
        records=records,
        endpoint_reference=endpoint_reference,
    )
    pgd_records = build_external_pgd_records(records)
    hints = build_causal_hint_cube(pgd_records)
    source_distance_m = np.asarray(
        [record.source_distance_m for record in records], dtype=np.float32
    )
    source_dt_sec = np.ones(len(records), dtype=np.float32)
    replay = replay_phase73_external(
        model=model,
        raw_rates=raw_rates,
        pgd_mw=hints["crowell_mw"],
        pgd_valid=hints["p_arrived"],
        source_distance_m=source_distance_m,
        source_dt_sec=source_dt_sec,
        beta_m_per_s=float(config["physics"]["beta"]),
        batch_size=batch_size,
    )
    tables = build_external_tables(
        records=records,
        phase73_mw=replay["phase73_mw"],
        phase39_mw=replay["phase39_mw"],
        pgd_3d_m=hints["pgd_3d_m"],
        crowell_mw=hints["crowell_mw"],
        p_arrived=hints["p_arrived"],
        plateau_confidence=replay["plateau_confidence"],
        revision_mw=replay["revision_mw"],
        proposal_assimilation_mw=replay["proposal_assimilation_mw"],
    )
    phase39_endpoint = np.asarray(replay["phase39_mw"], dtype=np.float64)[:, -1]
    endpoint_reference_values = np.asarray(
        [
            float(endpoint_reference["station_predictions"][(record.event, record.station)])
            for record in records
        ],
        dtype=np.float64,
    )
    phase39_replay_difference = float(
        np.max(np.abs(phase39_endpoint - endpoint_reference_values))
    )
    if phase39_replay_difference > ENDPOINT_PREDICTION_TOLERANCE_MW:
        raise ValueError("Phase73 raw proposal Mw does not reproduce Phase39")

    paths = {
        "external_pgd_3d_m.npy": hints["pgd_3d_m"],
        "external_crowell_mw.npy": hints["crowell_mw"],
        "external_p_arrived.npy": hints["p_arrived"],
        "external_phase73_mw.npy": replay["phase73_mw"],
        "external_phase39_mw.npy": replay["phase39_mw"],
        "external_plateau_confidence.npy": replay["plateau_confidence"],
        "external_revision_mw.npy": replay["revision_mw"],
        "external_proposal_assimilation_mw.npy": replay[
            "proposal_assimilation_mw"
        ],
    }
    for filename, array in paths.items():
        np.save(output_dir / filename, np.asarray(array))
    _write_csv(
        output_dir / "external_station_predictions.csv.gz",
        tables["station_rows"],
        fieldnames=tuple(tables["station_rows"][0]),
    )
    _write_csv(
        output_dir / "external_endpoint_station_predictions.csv",
        tables["endpoint_station_rows"],
        fieldnames=tuple(tables["endpoint_station_rows"][0]),
    )
    _write_csv(
        output_dir / "external_event_predictions.csv",
        tables["event_rows"],
        fieldnames=tuple(tables["event_rows"][0]),
    )
    _write_csv(
        output_dir / "external_horizon_metrics.csv",
        tables["horizon_rows"],
        fieldnames=tuple(tables["horizon_rows"][0]),
    )
    _write_csv(
        output_dir / "external_trajectory_diagnostics.csv",
        tables["trajectory_diagnostics"],
        fieldnames=tuple(tables["trajectory_diagnostics"][0]),
    )

    metric_lookup = _metric_lookup(tables["horizon_rows"])
    endpoint_metrics = {
        method: {
            "event_mae_mw": float(
                metric_lookup[(200, method)]["event_mae_mw"]
            ),
            "station_mae_mw": float(
                metric_lookup[(200, method)]["station_mae_mw"]
            ),
        }
        for method in phase73_validation.METHOD_ORDER
    }
    endpoint_rows = [
        row
        for row in tables["event_rows"]
        if int(row["observation_horizon_sec"]) == 200
    ]
    improved_events = sum(
        float(row["phase73_abs_error_mw"]) < float(row["phase39_abs_error_mw"])
        for row in endpoint_rows
    )
    generated_files = sorted(
        path for path in output_dir.iterdir() if path.is_file()
    )
    summary = {
        "status": "complete",
        "candidate": "Phase73 seed17 epoch27 frozen closest checkpoint",
        "evaluation_role": "user_authorized_one_time_external_development_validation",
        "cohort_role": "development_validation_not_blind_unseen_event_test",
        "event_count": len(tables["event_names"]),
        "station_count": len(records),
        "processing_delay_sec": PROCESSING_DELAY_SEC,
        "horizons_sec": list(HORIZONS),
        "endpoint_metrics": endpoint_metrics,
        "endpoint_phase39_reproduction_gate": endpoint_gate,
        "phase73_phase39_endpoint_max_abs_diff_mw": phase39_replay_difference,
        "improved_event_count_vs_phase39": improved_events,
        "trajectory_metrics": summarize_trajectory_metrics(tables["event_cubes"]),
        "formal_validation_gate_passed": False,
        "historical_training_campaign": {
            "phase": historical_campaign["phase"],
            "passed": historical_campaign["passed"],
            "internal_test_iterated": historical_campaign["internal_test_iterated"],
            "external_data_loaded": historical_campaign["external_data_loaded"],
            "grouped_test_loaded": historical_campaign["grouped_test_loaded"],
        },
        "internal_test_iterated": False,
        "external_data_loaded": True,
        "grouped_test_loaded": False,
        "source": source,
        "artifact_sha256": {
            "phase39_checkpoint": EXPECTED_PHASE39_SHA256,
            "phase39_external_raw_rates": raw_rates_sha256,
            **external_provenance,
            "runtime_outputs": {
                path.name: sha256_file(path) for path in generated_files
            },
        },
        "provenance": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "git_dirty": git_is_dirty(PROJECT_ROOT),
            "device": str(device),
            "batch_size": batch_size,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Phase73 seed17 epoch27 once on the locked "
            "eight-event development-validation cohort."
        )
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--hint-cache-root", type=Path, default=DEFAULT_HINT_CACHE_ROOT
    )
    parser.add_argument(
        "--external-raw-rates", type=Path, default=DEFAULT_EXTERNAL_RAW_RATES
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    summary = evaluate_external(
        output_dir=args.output_dir.resolve(),
        run_root=args.run_root.resolve(),
        cache_root=args.cache_root.resolve(),
        hint_cache_root=args.hint_cache_root.resolve(),
        raw_rates_path=args.external_raw_rates.resolve(),
        device=_resolve_device(args.device),
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(args.output_dir.resolve()),
                "phase73_event_mae_200s": summary["endpoint_metrics"]["phase73"][
                    "event_mae_mw"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
