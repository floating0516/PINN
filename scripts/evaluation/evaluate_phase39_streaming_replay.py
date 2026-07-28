from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.evaluate_phase39_second_by_second import (  # noqa: E402
    EXPECTED_EVENT_COUNT,
    EXPECTED_STATION_COUNT,
    EXTERNAL_EVENT_NAMES,
    EXTERNAL_EVENT_ROOT,
    FROZEN_PATHS,
    FULL_HORIZONS,
    LABELS_PATH,
    PROCESSING_DELAY_SEC,
    SMOKE_HORIZONS,
    TARGET_ERROR_MW,
    _configure_chinese_plot_style,
    _git_commit,
    _resolve_device,
    _save_figure,
    _style_axis,
    _write_csv,
    _write_json,
    analyze_convergence,
    hash_external_inputs,
    load_endpoint_reference,
    load_label_contract,
    load_model,
    load_phase39_config,
    plot_event_trajectories,
    validate_endpoint_reproduction,
    validate_frozen_artifacts,
    validate_output_dir,
)
from src.data.geometry import compute_source_station_geometry  # noqa: E402
from src.data.metadata import (  # noqa: E402
    build_metadata_tensor,
    metadata_distance_from_config,
)
from src.data.sample_builder import rotate_horizontal_to_rt  # noqa: E402
from src.data.waveform import (  # noqa: E402
    ProcessedWaveform,
    WaveformConfig,
    preprocess_waveform,
    waveform_config_from_v2,
)
from src.evaluation.evaluate_unseen import (  # noqa: E402
    EventBundle,
    StationWaveform,
    _format_event_display_name,
    _phase_preserving_external_waveform_config,
    _station_sample_from_bundle,
    load_event_bundle,
)
from src.models.model import PINNModel  # noqa: E402


SOURCE_TIME_STEPS = 200
SOURCE_DT_SEC = 1.0
ENDPOINT_INPUT_TOLERANCE_M = 1.0e-12
REPORT_PREFIX_HORIZONS = (20, 60, 120, 200)


@dataclass(frozen=True)
class RawStreamingStation:
    event: str
    event_dir: str
    station: str
    magnitude_catalog: float
    magnitude_source: str
    usgs_event_id: str
    raw_time_sec: np.ndarray
    raw_radial_m: np.ndarray
    waveform_config: WaveformConfig
    source_distance_m: float
    epicentral_distance_m: float
    theta_deg: float
    azimuth_deg: float
    p_arrival_sec: float
    s_arrival_sec: float
    endpoint_radial_m: np.ndarray
    endpoint_valid_mask: np.ndarray
    endpoint_baseline_source: str
    waveform_start_sec: float
    waveform_phase_adjusted: bool


@dataclass(frozen=True)
class StreamingPrefix:
    values_m: np.ndarray
    valid_mask: np.ndarray
    observation_horizon_sec: int
    issue_time_sec: float
    raw_sample_count: int
    baseline_m: float
    baseline_source: str
    valid_fraction_full_grid: float


def _crop_station(
    station: StationWaveform,
    *,
    cutoff_sec: float,
    inclusive: bool,
) -> StationWaveform:
    if inclusive:
        keep = np.asarray(station.t, dtype=np.float64) <= float(cutoff_sec)
    else:
        keep = np.asarray(station.t, dtype=np.float64) < float(cutoff_sec)
    return replace(
        station,
        t=np.asarray(station.t)[keep],
        e_m=np.asarray(station.e_m)[keep],
        n_m=np.asarray(station.n_m)[keep],
        u_m=np.asarray(station.u_m)[keep],
    )


def _classify_preprocessing_error(exc: ValueError) -> str:
    detail = str(exc).lower()
    if "baseline" in detail:
        return "insufficient_baseline"
    if "fewer than two" in detail:
        return "insufficient_raw_samples"
    return "invalid_waveform"


def build_raw_streaming_records(
    *,
    config: dict[str, Any],
    event_root: Path,
    labels_by_dir: Mapping[str, Mapping[str, Any]],
    expected_station_keys: set[tuple[str, str]],
    event_names: Sequence[str] = EXTERNAL_EVENT_NAMES,
) -> list[RawStreamingStation]:
    base_waveform_config = waveform_config_from_v2(config)
    alpha_m_per_s = float(config["physics"]["alpha"])
    beta_m_per_s = float(config["physics"]["beta"])
    records: list[RawStreamingStation] = []

    for event_dir_name in event_names:
        bundle = load_event_bundle(event_root / event_dir_name)
        event = _format_event_display_name(
            event_name=bundle.event_name,
            event_dir_name=bundle.event_dir_name,
            magnitude=bundle.magnitude,
        )
        label = labels_by_dir[event_dir_name]
        if event != str(label["event"]):
            raise ValueError(
                f"external display label changed for {event_dir_name}: "
                f"{event!r} != {label['event']!r}"
            )

        for station in bundle.stations:
            key = (event, station.station)
            if key not in expected_station_keys:
                continue

            earliest_issue_station = _crop_station(
                station,
                cutoff_sec=1.0 + PROCESSING_DELAY_SEC,
                inclusive=True,
            )
            early_phase_config = _phase_preserving_external_waveform_config(
                earliest_issue_station,
                base_waveform_config,
            )
            full_phase_config = _phase_preserving_external_waveform_config(
                station,
                base_waveform_config,
            )
            if not math.isclose(
                early_phase_config.start_sec,
                full_phase_config.start_sec,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"station grid phase is not causally known by 6 s: {key}"
                )
            replay_waveform_config = replace(
                early_phase_config,
                min_valid_fraction=0.0,
            )

            endpoint_station = _crop_station(
                station,
                cutoff_sec=200.0,
                inclusive=False,
            )
            endpoint_sample = _station_sample_from_bundle(
                bundle,
                endpoint_station,
                config,
                waveform_config=base_waveform_config,
                radial_peak_min_cm_override=0.0,
            )
            if endpoint_sample is None:
                raise ValueError(f"locked endpoint station no longer reconstructs: {key}")

            geometry = compute_source_station_geometry(
                bundle.latitude,
                bundle.longitude,
                bundle.depth_km,
                station.latitude,
                station.longitude,
            )
            raw_radial_m, _ = rotate_horizontal_to_rt(
                np.asarray(station.e_m, dtype=np.float64),
                np.asarray(station.n_m, dtype=np.float64),
                geometry.azimuth_deg,
            )
            endpoint_radial = np.asarray(endpoint_sample["radial"], dtype=np.float32)
            endpoint_mask = np.asarray(
                endpoint_sample["waveform_valid_mask"], dtype=bool
            )
            if endpoint_radial.shape != (SOURCE_TIME_STEPS,):
                raise ValueError(f"unexpected endpoint waveform shape for {key}")

            records.append(
                RawStreamingStation(
                    event=event,
                    event_dir=event_dir_name,
                    station=station.station,
                    magnitude_catalog=float(label["mw_selected"]),
                    magnitude_source=str(label["mw_source"]),
                    usgs_event_id=str(label["usgs_event_id"]),
                    raw_time_sec=np.asarray(station.t, dtype=np.float64).copy(),
                    raw_radial_m=np.asarray(raw_radial_m, dtype=np.float64),
                    waveform_config=replay_waveform_config,
                    source_distance_m=float(geometry.source_distance_m),
                    epicentral_distance_m=float(geometry.epicentral_distance_m),
                    theta_deg=float(geometry.takeoff_angle_deg),
                    azimuth_deg=float(geometry.azimuth_deg),
                    p_arrival_sec=float(geometry.source_distance_m / alpha_m_per_s),
                    s_arrival_sec=float(geometry.source_distance_m / beta_m_per_s),
                    endpoint_radial_m=endpoint_radial,
                    endpoint_valid_mask=endpoint_mask,
                    endpoint_baseline_source=str(endpoint_sample["baseline_source"]),
                    waveform_start_sec=float(replay_waveform_config.start_sec),
                    waveform_phase_adjusted=not math.isclose(
                        replay_waveform_config.start_sec,
                        base_waveform_config.start_sec,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    ),
                )
            )

    records.sort(key=lambda item: (item.event, item.station))
    actual_keys = {(item.event, item.station) for item in records}
    if actual_keys != expected_station_keys or len(records) != len(actual_keys):
        missing = sorted(expected_station_keys - actual_keys)
        extra = sorted(actual_keys - expected_station_keys)
        raise ValueError(
            "raw streaming cohort differs from locked endpoint; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return records


def preprocess_streaming_prefix(
    record: RawStreamingStation,
    *,
    observation_horizon_sec: int,
) -> StreamingPrefix:
    if isinstance(observation_horizon_sec, bool) or not isinstance(
        observation_horizon_sec, int
    ):
        raise ValueError("observation_horizon_sec must be an integer")
    if observation_horizon_sec < 1 or observation_horizon_sec > SOURCE_TIME_STEPS:
        raise ValueError("observation_horizon_sec must be between 1 and 200")

    issue_time_sec = float(observation_horizon_sec + PROCESSING_DELAY_SEC)
    raw_time = np.asarray(record.raw_time_sec, dtype=np.float64)
    raw_values = np.asarray(record.raw_radial_m, dtype=np.float64)
    available = (
        np.isfinite(raw_time)
        & np.isfinite(raw_values)
        & (raw_time <= issue_time_sec)
    )
    if int(np.count_nonzero(available)) < 2:
        raise ValueError("waveform has fewer than two finite samples at issue time")

    processed = preprocess_waveform(
        raw_time[available],
        raw_values[available],
        units="m",
        p_arrival_sec=record.p_arrival_sec,
        config=record.waveform_config,
    )
    values = np.asarray(
        processed.values_m[:observation_horizon_sec], dtype=np.float32
    )
    valid_mask = np.asarray(
        processed.valid_mask[:observation_horizon_sec], dtype=bool
    )
    if values.shape != (observation_horizon_sec,) or valid_mask.shape != values.shape:
        raise ValueError("streaming prefix shape differs from the observation horizon")
    if np.any(values[~valid_mask] != 0.0):
        raise ValueError("invalid streaming prefix slots must remain zero")
    return StreamingPrefix(
        values_m=values,
        valid_mask=valid_mask,
        observation_horizon_sec=observation_horizon_sec,
        issue_time_sec=issue_time_sec,
        raw_sample_count=int(np.count_nonzero(available)),
        baseline_m=float(processed.baseline_m),
        baseline_source=str(processed.baseline_source),
        valid_fraction_full_grid=float(processed.valid_fraction),
    )


def decode_stf_rate(
    encoded_stf: torch.Tensor,
    *,
    stf_m_ref: float,
) -> torch.Tensor:
    if encoded_stf.ndim != 2 or encoded_stf.shape[1] != SOURCE_TIME_STEPS:
        raise ValueError("encoded STF must have shape (batch, 200)")
    if not bool(torch.isfinite(encoded_stf).all()):
        raise FloatingPointError("model returned a non-finite encoded STF")
    return torch.clamp(
        float(stf_m_ref) * (torch.pow(10.0, encoded_stf) - 1.0),
        min=0.0,
    )


def moment_to_mw(moment_nm: float) -> float:
    moment = max(float(moment_nm), 1.0e10)
    return float((2.0 / 3.0) * (math.log10(moment) - 9.1))


def conservative_visible_steps(
    observation_horizon_sec: int,
    arrival_sec: float,
    *,
    source_steps: int = SOURCE_TIME_STEPS,
) -> int:
    visible_end = max(0.0, float(observation_horizon_sec) - float(arrival_sec))
    return min(source_steps, max(0, int(math.floor(visible_end + 1.0e-12))))


def decompose_stf_revision(
    current_rate: np.ndarray,
    previous_rate: np.ndarray,
    *,
    current_confirmed_steps: int,
    previous_confirmed_steps: int,
    dt_sec: float = SOURCE_DT_SEC,
) -> dict[str, float | bool]:
    current = np.asarray(current_rate, dtype=np.float64)
    previous = np.asarray(previous_rate, dtype=np.float64)
    if current.shape != previous.shape or current.ndim != 1:
        raise ValueError("current and previous STF rates must share one-dimensional shape")
    if not 0 <= previous_confirmed_steps <= current_confirmed_steps <= current.size:
        raise ValueError("confirmed source steps are inconsistent")
    if not math.isfinite(dt_sec) or dt_sec <= 0.0:
        raise ValueError("dt_sec must be positive and finite")

    history_slice = slice(0, previous_confirmed_steps)
    previous_history_moment = float(previous[history_slice].sum() * dt_sec)
    signed_history_revision = float(
        (current[history_slice] - previous[history_slice]).sum() * dt_sec
    )
    history_l1_revision = float(
        np.abs(current[history_slice] - previous[history_slice]).sum() * dt_sec
    )
    newly_confirmed_moment = float(
        current[previous_confirmed_steps:current_confirmed_steps].sum() * dt_sec
    )
    tail_slice = slice(current_confirmed_steps, current.size)
    signed_future_revision = float(
        (current[tail_slice] - previous[tail_slice]).sum() * dt_sec
    )
    future_l1_revision = float(
        np.abs(current[tail_slice] - previous[tail_slice]).sum() * dt_sec
    )
    previous_total = float(previous.sum() * dt_sec)
    full_l1_revision = float(np.abs(current - previous).sum() * dt_sec)
    return {
        "previous_confirmed_moment_nm": previous_history_moment,
        "confirmed_history_revision_nm": signed_history_revision,
        "confirmed_history_revision_fraction": (
            signed_history_revision / previous_history_moment
            if previous_history_moment > 0.0
            else float("nan")
        ),
        "confirmed_history_l1_revision_nm": history_l1_revision,
        "confirmed_history_l1_revision_fraction": (
            history_l1_revision / previous_history_moment
            if previous_history_moment > 0.0
            else float("nan")
        ),
        "confirmed_history_moment_decreased": signed_history_revision < 0.0,
        "newly_confirmed_moment_nm": newly_confirmed_moment,
        "future_tail_revision_nm": signed_future_revision,
        "future_tail_l1_revision_nm": future_l1_revision,
        "full_stf_l1_revision_nm": full_l1_revision,
        "full_stf_l1_revision_fraction": (
            full_l1_revision / previous_total
            if previous_total > 0.0
            else float("nan")
        ),
    }


def _metadata_for_records(
    records: Sequence[RawStreamingStation],
    config: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    source_distance = torch.tensor(
        [record.source_distance_m for record in records],
        device=device,
        dtype=dtype,
    )
    epicentral_distance = torch.tensor(
        [record.epicentral_distance_m for record in records],
        device=device,
        dtype=dtype,
    )
    theta = torch.tensor(
        [record.theta_deg for record in records], device=device, dtype=dtype
    )
    azimuth = torch.tensor(
        [record.azimuth_deg for record in records], device=device, dtype=dtype
    )
    distance = metadata_distance_from_config(
        config,
        source_distance_m=source_distance,
        epicentral_distance_m=epicentral_distance,
    )
    return build_metadata_tensor(distance, theta, azimuth)


def _optional_ratio(values: Sequence[Any]) -> float | None:
    finite = np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64
    )
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else None


def _build_event_outputs(
    *,
    station_rows: Sequence[Mapping[str, Any]],
    unavailable_rows: Sequence[Mapping[str, Any]],
    records: Sequence[RawStreamingStation],
    horizons: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_horizon: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    unavailable_by_horizon: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in station_rows:
        by_horizon[int(row["observation_horizon_sec"])].append(row)
    for row in unavailable_rows:
        unavailable_by_horizon[int(row["observation_horizon_sec"])].append(row)

    event_rows: list[dict[str, Any]] = []
    horizon_metrics: list[dict[str, Any]] = []
    event_names = sorted({record.event for record in records})
    previous_event_predictions: dict[str, float] = {}

    for horizon in horizons:
        current_station_rows = by_horizon[int(horizon)]
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in current_station_rows:
            grouped[str(row["event"])].append(row)

        current_events: list[dict[str, Any]] = []
        for event in sorted(grouped):
            rows = grouped[event]
            predictions = np.asarray([float(row["mw_pred"]) for row in rows])
            catalogs = np.asarray([float(row["mw_catalog"]) for row in rows])
            if not np.allclose(catalogs, catalogs[0], rtol=0.0, atol=1.0e-9):
                raise ValueError(f"catalog magnitude differs within event {event}")
            prediction = float(np.median(predictions))
            previous_prediction = previous_event_predictions.get(event)
            comparable_history = [
                row
                for row in rows
                if row.get("confirmed_history_revision_nm") is not None
                and int(row.get("previous_confirmed_source_steps", 0)) > 0
            ]
            delta_values = [
                float(row["delta_mw_from_previous"])
                for row in rows
                if row.get("delta_mw_from_previous") is not None
            ]
            current_events.append(
                {
                    "event": event,
                    "observation_horizon_sec": float(horizon),
                    "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                    "mw_pred_median": prediction,
                    "mw_catalog": float(catalogs[0]),
                    "error": prediction - float(catalogs[0]),
                    "station_count": len(rows),
                    "station_mw_iqr": float(
                        np.percentile(predictions, 75)
                        - np.percentile(predictions, 25)
                    ),
                    "station_mw_range": float(predictions.max() - predictions.min()),
                    "event_median_delta_mw": (
                        None
                        if previous_prediction is None
                        else prediction - previous_prediction
                    ),
                    "median_station_delta_mw": (
                        float(np.median(delta_values)) if delta_values else None
                    ),
                    "max_abs_station_delta_mw": (
                        float(np.max(np.abs(delta_values))) if delta_values else None
                    ),
                    "station_total_mw_decrease_count": sum(
                        value < 0.0 for value in delta_values
                    ),
                    "confirmed_history_comparison_count": len(comparable_history),
                    "confirmed_history_decrease_count": sum(
                        bool(row["confirmed_history_moment_decreased"])
                        for row in comparable_history
                    ),
                    "confirmed_history_decrease_fraction": (
                        sum(
                            bool(row["confirmed_history_moment_decreased"])
                            for row in comparable_history
                        )
                        / len(comparable_history)
                        if comparable_history
                        else None
                    ),
                    "median_confirmed_history_revision_fraction": _optional_ratio(
                        [
                            row.get("confirmed_history_revision_fraction")
                            for row in comparable_history
                        ]
                    ),
                    "median_confirmed_history_l1_revision_fraction": _optional_ratio(
                        [
                            row.get("confirmed_history_l1_revision_fraction")
                            for row in comparable_history
                        ]
                    ),
                    "median_confirmed_moment_fraction": float(
                        np.median(
                            [float(row["confirmed_moment_fraction"]) for row in rows]
                        )
                    ),
                }
            )
            previous_event_predictions[event] = prediction

        event_rows.extend(current_events)
        errors = np.asarray([float(row["error"]) for row in current_events])
        event_deltas = np.asarray(
            [
                float(row["event_median_delta_mw"])
                for row in current_events
                if row["event_median_delta_mw"] is not None
            ],
            dtype=np.float64,
        )
        station_deltas = np.asarray(
            [
                float(row["delta_mw_from_previous"])
                for row in current_station_rows
                if row.get("delta_mw_from_previous") is not None
            ],
            dtype=np.float64,
        )
        comparable_history = [
            row
            for row in current_station_rows
            if row.get("confirmed_history_revision_nm") is not None
            and int(row.get("previous_confirmed_source_steps", 0)) > 0
        ]
        unavailable = unavailable_by_horizon[int(horizon)]
        reason_counts = Counter(str(row["reason"]) for row in unavailable)
        horizon_metrics.append(
            {
                "observation_horizon_sec": float(horizon),
                "release_time_sec": float(horizon + PROCESSING_DELAY_SEC),
                "event_count": int(errors.size),
                "total_event_count": len(event_names),
                "coverage": int(errors.size),
                "coverage_fraction": (
                    float(errors.size / len(event_names)) if event_names else 0.0
                ),
                "available_station_count": len(current_station_rows),
                "unavailable_station_count": len(unavailable),
                "unavailable_reason_counts": dict(sorted(reason_counts.items())),
                "event_equal_mae": (
                    float(np.mean(np.abs(errors))) if errors.size else float("nan")
                ),
                "event_equal_rmse": (
                    float(np.sqrt(np.mean(errors**2)))
                    if errors.size
                    else float("nan")
                ),
                "event_equal_bias": (
                    float(np.mean(errors)) if errors.size else float("nan")
                ),
                "median_abs_event_median_step_mw": (
                    float(np.median(np.abs(event_deltas)))
                    if event_deltas.size
                    else None
                ),
                "max_abs_event_median_step_mw": (
                    float(np.max(np.abs(event_deltas)))
                    if event_deltas.size
                    else None
                ),
                "event_median_decrease_count": int(np.sum(event_deltas < 0.0)),
                "median_abs_station_step_mw": (
                    float(np.median(np.abs(station_deltas)))
                    if station_deltas.size
                    else None
                ),
                "max_abs_station_step_mw": (
                    float(np.max(np.abs(station_deltas)))
                    if station_deltas.size
                    else None
                ),
                "station_mw_decrease_count": int(np.sum(station_deltas < 0.0)),
                "confirmed_history_comparison_count": len(comparable_history),
                "confirmed_history_decrease_count": sum(
                    bool(row["confirmed_history_moment_decreased"])
                    for row in comparable_history
                ),
                "confirmed_history_decrease_fraction": (
                    sum(
                        bool(row["confirmed_history_moment_decreased"])
                        for row in comparable_history
                    )
                    / len(comparable_history)
                    if comparable_history
                    else None
                ),
                "median_confirmed_history_revision_fraction": _optional_ratio(
                    [
                        row.get("confirmed_history_revision_fraction")
                        for row in comparable_history
                    ]
                ),
                "median_confirmed_history_l1_revision_fraction": _optional_ratio(
                    [
                        row.get("confirmed_history_l1_revision_fraction")
                        for row in comparable_history
                    ]
                ),
            }
        )
    return event_rows, horizon_metrics


def evaluate_streaming_replay(
    model: PINNModel,
    config: dict[str, Any],
    records: Sequence[RawStreamingStation],
    *,
    horizons: Sequence[int],
    batch_size: int,
) -> dict[str, Any]:
    normalized_horizons = tuple(int(value) for value in horizons)
    if normalized_horizons != tuple(sorted(set(normalized_horizons))):
        raise ValueError("horizons must be strictly increasing and unique")
    if not normalized_horizons or normalized_horizons[-1] > SOURCE_TIME_STEPS:
        raise ValueError("streaming horizons must be within 1..200")
    if isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    metadata = _metadata_for_records(
        records, config, device=device, dtype=dtype
    )
    stf_m_ref = float(config["dataset"]["stf"]["m_ref"])
    horizon_count = len(normalized_horizons)
    station_count = len(records)
    rate_cube = np.full(
        (horizon_count, station_count, SOURCE_TIME_STEPS),
        np.nan,
        dtype=np.float32,
    )
    available_mask = np.zeros((horizon_count, station_count), dtype=bool)
    confirmed_steps = np.zeros((horizon_count, station_count), dtype=np.int16)
    mw_values = np.full((horizon_count, station_count), np.nan, dtype=np.float64)
    m0_values = np.full((horizon_count, station_count), np.nan, dtype=np.float64)
    station_rows: list[dict[str, Any]] = []
    unavailable_rows: list[dict[str, Any]] = []
    endpoint_diffs: list[float] = []
    endpoint_mask_mismatch_count = 0
    endpoint_baseline_mismatch_count = 0

    original_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for horizon_index, horizon in enumerate(normalized_horizons):
                prefixes: dict[int, StreamingPrefix] = {}
                for station_index, record in enumerate(records):
                    try:
                        prefixes[station_index] = preprocess_streaming_prefix(
                            record,
                            observation_horizon_sec=horizon,
                        )
                    except ValueError as exc:
                        unavailable_rows.append(
                            {
                                "event": record.event,
                                "station": record.station,
                                "observation_horizon_sec": float(horizon),
                                "release_time_sec": float(
                                    horizon + PROCESSING_DELAY_SEC
                                ),
                                "reason": _classify_preprocessing_error(exc),
                                "detail": str(exc),
                            }
                        )

                available_indices = sorted(prefixes)
                for start in range(0, len(available_indices), batch_size):
                    batch_indices = available_indices[start : start + batch_size]
                    model_input = torch.as_tensor(
                        np.stack(
                            [prefixes[index].values_m for index in batch_indices]
                        ),
                        device=device,
                        dtype=dtype,
                    ).unsqueeze(1)
                    if model_input.shape[-1] != horizon:
                        raise RuntimeError("model input was padded beyond the true prefix")
                    selected = torch.as_tensor(
                        batch_indices, device=device, dtype=torch.long
                    )
                    encoded = model(
                        model_input,
                        meta=metadata.index_select(0, selected),
                    )
                    rates = decode_stf_rate(encoded, stf_m_ref=stf_m_ref)
                    rate_values = rates.detach().cpu().numpy().astype(
                        np.float32, copy=False
                    )

                    for local_index, station_index in enumerate(batch_indices):
                        record = records[station_index]
                        prefix = prefixes[station_index]
                        rate = np.asarray(rate_values[local_index], dtype=np.float32)
                        m0_total = float(
                            np.asarray(rate, dtype=np.float64).sum() * SOURCE_DT_SEC
                        )
                        mw_pred = moment_to_mw(m0_total)
                        p_steps = conservative_visible_steps(
                            horizon, record.p_arrival_sec
                        )
                        s_steps = conservative_visible_steps(
                            horizon, record.s_arrival_sec
                        )
                        m0_p_visible = float(
                            np.asarray(rate[:p_steps], dtype=np.float64).sum()
                            * SOURCE_DT_SEC
                        )
                        m0_confirmed = float(
                            np.asarray(rate[:s_steps], dtype=np.float64).sum()
                            * SOURCE_DT_SEC
                        )
                        m0_remaining = max(0.0, m0_total - m0_confirmed)

                        rate_cube[horizon_index, station_index] = rate
                        available_mask[horizon_index, station_index] = True
                        confirmed_steps[horizon_index, station_index] = s_steps
                        mw_values[horizon_index, station_index] = mw_pred
                        m0_values[horizon_index, station_index] = m0_total

                        row: dict[str, Any] = {
                            "event": record.event,
                            "station": record.station,
                            "observation_horizon_sec": float(horizon),
                            "release_time_sec": float(prefix.issue_time_sec),
                            "model_input_steps": int(model_input.shape[-1]),
                            "mw_pred": mw_pred,
                            "mw_catalog": record.magnitude_catalog,
                            "error": mw_pred - record.magnitude_catalog,
                            "m0_total_nm": m0_total,
                            "p_visible_source_end_sec": max(
                                0.0, float(horizon) - record.p_arrival_sec
                            ),
                            "p_visible_source_steps": p_steps,
                            "s_confirmed_source_end_sec": max(
                                0.0, float(horizon) - record.s_arrival_sec
                            ),
                            "s_confirmed_source_steps": s_steps,
                            "m0_p_visible_nm": m0_p_visible,
                            "m0_confirmed_nm": m0_confirmed,
                            "mw_confirmed": (
                                moment_to_mw(m0_confirmed)
                                if m0_confirmed > 0.0
                                else None
                            ),
                            "m0_remaining_nm": m0_remaining,
                            "confirmed_moment_fraction": (
                                m0_confirmed / m0_total if m0_total > 0.0 else 0.0
                            ),
                            "raw_sample_count_at_issue": prefix.raw_sample_count,
                            "prefix_valid_sample_count": int(
                                np.count_nonzero(prefix.valid_mask)
                            ),
                            "prefix_valid_fraction": float(
                                np.mean(prefix.valid_mask)
                            ),
                            "baseline_m": prefix.baseline_m,
                            "baseline_source": prefix.baseline_source,
                            "waveform_start_sec": record.waveform_start_sec,
                            "waveform_phase_adjusted": record.waveform_phase_adjusted,
                            "source_distance_m": record.source_distance_m,
                            "p_arrival_sec": record.p_arrival_sec,
                            "s_arrival_sec": record.s_arrival_sec,
                            "previous_observation_horizon_sec": None,
                            "delta_mw_from_previous": None,
                            "delta_m0_total_nm": None,
                            "previous_confirmed_source_steps": None,
                            "previous_confirmed_moment_nm": None,
                            "confirmed_history_revision_nm": None,
                            "confirmed_history_revision_fraction": None,
                            "confirmed_history_l1_revision_nm": None,
                            "confirmed_history_l1_revision_fraction": None,
                            "confirmed_history_moment_decreased": None,
                            "newly_confirmed_moment_nm": None,
                            "future_tail_revision_nm": None,
                            "future_tail_l1_revision_nm": None,
                            "full_stf_l1_revision_nm": None,
                            "full_stf_l1_revision_fraction": None,
                        }

                        if (
                            horizon_index > 0
                            and available_mask[horizon_index - 1, station_index]
                        ):
                            previous_rate = rate_cube[
                                horizon_index - 1, station_index
                            ]
                            previous_steps = int(
                                confirmed_steps[horizon_index - 1, station_index]
                            )
                            revision = decompose_stf_revision(
                                rate,
                                previous_rate,
                                current_confirmed_steps=s_steps,
                                previous_confirmed_steps=previous_steps,
                            )
                            row.update(revision)
                            row.update(
                                {
                                    "previous_observation_horizon_sec": float(
                                        normalized_horizons[horizon_index - 1]
                                    ),
                                    "delta_mw_from_previous": mw_pred
                                    - float(
                                        mw_values[horizon_index - 1, station_index]
                                    ),
                                    "delta_m0_total_nm": m0_total
                                    - float(
                                        m0_values[horizon_index - 1, station_index]
                                    ),
                                    "previous_confirmed_source_steps": previous_steps,
                                }
                            )
                        station_rows.append(row)

                        if horizon == SOURCE_TIME_STEPS:
                            endpoint_diffs.append(
                                float(
                                    np.max(
                                        np.abs(
                                            prefix.values_m
                                            - record.endpoint_radial_m
                                        )
                                    )
                                )
                            )
                            endpoint_mask_mismatch_count += int(
                                not np.array_equal(
                                    prefix.valid_mask,
                                    record.endpoint_valid_mask,
                                )
                            )
                            endpoint_baseline_mismatch_count += int(
                                prefix.baseline_source
                                != record.endpoint_baseline_source
                            )
    finally:
        model.train(original_training)

    if len(endpoint_diffs) != len(records):
        raise ValueError("h=200 did not produce every locked station input")
    max_input_diff = max(endpoint_diffs, default=0.0)
    if max_input_diff > ENDPOINT_INPUT_TOLERANCE_M:
        raise ValueError(
            f"h=200 streaming input mismatch: {max_input_diff:.9g} m"
        )
    if endpoint_mask_mismatch_count or endpoint_baseline_mismatch_count:
        raise ValueError(
            "h=200 streaming preprocessing metadata differs from the locked endpoint"
        )

    station_rows.sort(
        key=lambda row: (
            float(row["observation_horizon_sec"]),
            str(row["event"]),
            str(row["station"]),
        )
    )
    unavailable_rows.sort(
        key=lambda row: (
            float(row["observation_horizon_sec"]),
            str(row["event"]),
            str(row["station"]),
        )
    )
    event_rows, horizon_metrics = _build_event_outputs(
        station_rows=station_rows,
        unavailable_rows=unavailable_rows,
        records=records,
        horizons=normalized_horizons,
    )
    return {
        "station_rows": station_rows,
        "unavailable_rows": unavailable_rows,
        "event_rows": event_rows,
        "horizon_metrics": horizon_metrics,
        "rate_nm_per_s": rate_cube,
        "available_mask": available_mask,
        "confirmed_source_steps": confirmed_steps,
        "input_endpoint_gate": {
            "tolerance_m": ENDPOINT_INPUT_TOLERANCE_M,
            "station_count": len(endpoint_diffs),
            "max_abs_radial_input_diff_m": max_input_diff,
            "valid_mask_mismatch_count": endpoint_mask_mismatch_count,
            "baseline_source_mismatch_count": endpoint_baseline_mismatch_count,
        },
    }


def summarize_streaming_stability(
    result: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    first_full = analysis["overall"][
        "first_full_station_availability_observation_sec"
    ]
    if first_full is None:
        first_full = int(analysis["overall"]["final_observation_sec"])

    station_steps = [
        row
        for row in result["station_rows"]
        if row.get("delta_mw_from_previous") is not None
        and int(row["observation_horizon_sec"]) >= int(first_full)
    ]
    event_steps = [
        row
        for row in result["event_rows"]
        if row.get("event_median_delta_mw") is not None
        and int(row["observation_horizon_sec"]) >= int(first_full)
    ]
    history = [
        row
        for row in station_steps
        if int(row.get("previous_confirmed_source_steps") or 0) > 0
        and row.get("confirmed_history_revision_nm") is not None
    ]
    late_history = [
        row for row in history if int(row["observation_horizon_sec"]) >= 180
    ]
    late_station_steps = [
        row
        for row in station_steps
        if int(row["observation_horizon_sec"]) >= 180
    ]
    late_event_steps = [
        row
        for row in event_steps
        if int(row["observation_horizon_sec"]) >= 180
    ]
    station_abs_steps = np.asarray(
        [abs(float(row["delta_mw_from_previous"])) for row in station_steps]
    )
    event_abs_steps = np.asarray(
        [abs(float(row["event_median_delta_mw"])) for row in event_steps]
    )
    late_station_abs_steps = np.asarray(
        [abs(float(row["delta_mw_from_previous"])) for row in late_station_steps]
    )
    late_event_abs_steps = np.asarray(
        [abs(float(row["event_median_delta_mw"])) for row in late_event_steps]
    )
    worst_station = max(
        station_steps,
        key=lambda row: abs(float(row["delta_mw_from_previous"])),
        default=None,
    )
    worst_event = max(
        event_steps,
        key=lambda row: abs(float(row["event_median_delta_mw"])),
        default=None,
    )
    largest_station_decrease = min(
        station_steps,
        key=lambda row: float(row["delta_mw_from_previous"]),
        default=None,
    )
    largest_event_decrease = min(
        event_steps,
        key=lambda row: float(row["event_median_delta_mw"]),
        default=None,
    )

    def downward_fraction(rows: Sequence[Mapping[str, Any]]) -> float | None:
        if not rows:
            return None
        return float(
            np.mean(
                [bool(row["confirmed_history_moment_decreased"]) for row in rows]
            )
        )

    return {
        "first_full_station_availability_observation_sec": int(first_full),
        "station_step_comparison_count_after_full_availability": len(station_steps),
        "median_abs_station_step_mw_after_full_availability": (
            float(np.median(station_abs_steps)) if station_abs_steps.size else None
        ),
        "p95_abs_station_step_mw_after_full_availability": (
            float(np.percentile(station_abs_steps, 95))
            if station_abs_steps.size
            else None
        ),
        "max_abs_station_step_mw_after_full_availability": (
            float(np.max(station_abs_steps)) if station_abs_steps.size else None
        ),
        "station_mw_decrease_fraction_after_full_availability": (
            float(
                np.mean(
                    [float(row["delta_mw_from_previous"]) < 0.0 for row in station_steps]
                )
            )
            if station_steps
            else None
        ),
        "event_step_comparison_count_after_full_availability": len(event_steps),
        "median_abs_event_median_step_mw_after_full_availability": (
            float(np.median(event_abs_steps)) if event_abs_steps.size else None
        ),
        "max_abs_event_median_step_mw_after_full_availability": (
            float(np.max(event_abs_steps)) if event_abs_steps.size else None
        ),
        "event_median_decrease_fraction_after_full_availability": (
            float(
                np.mean(
                    [float(row["event_median_delta_mw"]) < 0.0 for row in event_steps]
                )
            )
            if event_steps
            else None
        ),
        "late_180_200_median_abs_station_step_mw": (
            float(np.median(late_station_abs_steps))
            if late_station_abs_steps.size
            else None
        ),
        "late_180_200_p95_abs_station_step_mw": (
            float(np.percentile(late_station_abs_steps, 95))
            if late_station_abs_steps.size
            else None
        ),
        "late_180_200_max_abs_station_step_mw": (
            float(np.max(late_station_abs_steps))
            if late_station_abs_steps.size
            else None
        ),
        "late_180_200_median_abs_event_median_step_mw": (
            float(np.median(late_event_abs_steps))
            if late_event_abs_steps.size
            else None
        ),
        "late_180_200_p95_abs_event_median_step_mw": (
            float(np.percentile(late_event_abs_steps, 95))
            if late_event_abs_steps.size
            else None
        ),
        "late_180_200_max_abs_event_median_step_mw": (
            float(np.max(late_event_abs_steps))
            if late_event_abs_steps.size
            else None
        ),
        "confirmed_history_comparison_count": len(history),
        "confirmed_history_downward_revision_fraction": downward_fraction(history),
        "late_180_200_confirmed_history_comparison_count": len(late_history),
        "late_180_200_confirmed_history_downward_revision_fraction": downward_fraction(
            late_history
        ),
        "median_confirmed_history_l1_revision_fraction": _optional_ratio(
            [row.get("confirmed_history_l1_revision_fraction") for row in history]
        ),
        "late_180_200_median_confirmed_history_l1_revision_fraction": _optional_ratio(
            [
                row.get("confirmed_history_l1_revision_fraction")
                for row in late_history
            ]
        ),
        "worst_station_step": (
            None
            if worst_station is None
            else {
                "event": str(worst_station["event"]),
                "station": str(worst_station["station"]),
                "observation_horizon_sec": int(
                    worst_station["observation_horizon_sec"]
                ),
                "delta_mw": float(worst_station["delta_mw_from_previous"]),
            }
        ),
        "worst_event_median_step": (
            None
            if worst_event is None
            else {
                "event": str(worst_event["event"]),
                "observation_horizon_sec": int(
                    worst_event["observation_horizon_sec"]
                ),
                "delta_mw": float(worst_event["event_median_delta_mw"]),
            }
        ),
        "largest_station_mw_decrease": (
            None
            if largest_station_decrease is None
            else {
                "event": str(largest_station_decrease["event"]),
                "station": str(largest_station_decrease["station"]),
                "observation_horizon_sec": int(
                    largest_station_decrease["observation_horizon_sec"]
                ),
                "delta_mw": float(
                    largest_station_decrease["delta_mw_from_previous"]
                ),
            }
        ),
        "largest_event_median_decrease": (
            None
            if largest_event_decrease is None
            else {
                "event": str(largest_event_decrease["event"]),
                "observation_horizon_sec": int(
                    largest_event_decrease["observation_horizon_sec"]
                ),
                "delta_mw": float(
                    largest_event_decrease["event_median_delta_mw"]
                ),
            }
        ),
    }


def _finite_series(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray(
        [
            np.nan if row.get(key) is None else float(row[key])
            for row in rows
        ],
        dtype=np.float64,
    )


def plot_streaming_stability(
    result: Mapping[str, Any],
    *,
    output_stem: Path,
) -> list[Path]:
    _configure_chinese_plot_style()
    rows = result["horizon_metrics"]
    horizon = np.asarray([float(row["observation_horizon_sec"]) for row in rows])
    event_step = _finite_series(rows, "max_abs_event_median_step_mw")
    station_step = _finite_series(rows, "max_abs_station_step_mw")
    history_down = _finite_series(rows, "confirmed_history_decrease_fraction")
    history_l1 = _finite_series(
        rows, "median_confirmed_history_l1_revision_fraction"
    )
    stations = np.asarray([int(row["available_station_count"]) for row in rows])
    comparisons = np.asarray(
        [int(row["confirmed_history_comparison_count"]) for row in rows]
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 7.5), sharex=True)
    for axis in axes:
        _style_axis(axis)

    axes[0].plot(horizon, event_step, color="#0072B2", label="事件中位数最大单秒跳变")
    axes[0].plot(horizon, station_step, color="#D55E00", alpha=0.85, label="台站最大单秒跳变")
    axes[0].set_ylabel("|ΔMw|")
    axes[0].set_title("A  每次新输入触发的独立重估幅度", loc="left")
    axes[0].legend(loc="upper right")

    axes[1].plot(horizon, history_down, color="#CC79A7", label="已确认历史被向下重写的台站比例")
    axes[1].plot(horizon, history_l1, color="#009E73", label="已确认历史 L1 改写 / 上一秒确认矩")
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_ylabel("比例")
    axes[1].set_title("B  仅审计上一秒已获 S 波支持的震源历史", loc="left")
    axes[1].legend(loc="upper right")

    axes[2].step(horizon, stations, where="mid", color="#0072B2", label="可用台站")
    axes[2].step(horizon, comparisons, where="mid", color="#6B6B6B", linestyle="--", label="可比较已确认历史的台站")
    axes[2].set_ylabel("台站数")
    axes[2].set_xlabel("观测时长 (s)")
    axes[2].set_title("C  可用性与传播延迟约束", loc="left")
    axes[2].legend(loc="lower right")
    axes[2].set_xlim(float(horizon.min()), float(horizon.max()))

    fig.suptitle("Phase39 真实流式回放的稳定性审计", y=0.995, fontsize=13)
    fig.text(
        0.5,
        0.012,
        "每秒重新运行固定离线模型；STF 各点非负，但相邻两次完整 STF 之间没有单调或状态约束。",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.08, top=0.94, hspace=0.36)
    return _save_figure(fig, output_stem)


def _station_index(
    records: Sequence[RawStreamingStation], *, event: str, station: str
) -> int:
    for index, record in enumerate(records):
        if record.event == event and record.station == station:
            return index
    raise KeyError((event, station))


def plot_worst_station_stf(
    result: Mapping[str, Any],
    records: Sequence[RawStreamingStation],
    stability: Mapping[str, Any],
    *,
    output_stem: Path,
) -> list[Path]:
    worst = stability["largest_station_mw_decrease"]
    if worst is None:
        raise ValueError("no station transition is available for the STF figure")
    drop_horizon = float(worst["observation_horizon_sec"])
    station_index = _station_index(
        records,
        event=str(worst["event"]),
        station=str(worst["station"]),
    )
    horizons = np.asarray(
        [float(row["observation_horizon_sec"]) for row in result["horizon_metrics"]]
    )
    rates = np.asarray(result["rate_nm_per_s"][:, station_index], dtype=np.float64)
    available = np.asarray(result["available_mask"][:, station_index], dtype=bool)
    totals = np.nansum(rates, axis=1) * SOURCE_DT_SEC
    shapes = rates / np.maximum(totals[:, None], 1.0)
    log_shapes = np.log10(np.maximum(shapes, 1.0e-9))
    log_shapes[~available] = np.nan
    rows = [
        row
        for row in result["station_rows"]
        if str(row["event"]) == str(worst["event"])
        and str(row["station"]) == str(worst["station"])
    ]
    rows.sort(key=lambda row: float(row["observation_horizon_sec"]))
    mw = _finite_series(rows, "mw_pred")
    available_horizons = horizons[available]
    drop_indices = np.flatnonzero(np.isclose(available_horizons, drop_horizon))
    if drop_indices.size != 1:
        raise ValueError("largest station decrease horizon is missing from its trajectory")
    drop_mw = float(mw[int(drop_indices[0])])
    confirmed_fraction = _finite_series(rows, "confirmed_moment_fraction")
    history_revision = _finite_series(
        rows, "confirmed_history_revision_fraction"
    )

    _configure_chinese_plot_style()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.0, 8.0),
        sharex=False,
        gridspec_kw={"height_ratios": (2.1, 1.0, 1.0)},
    )
    heat = axes[0].imshow(
        log_shapes,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(0, SOURCE_TIME_STEPS, horizons[0], horizons[-1]),
        cmap="viridis",
        vmin=-7.0,
        vmax=-1.0,
    )
    axes[0].set_ylabel("观测时长 (s)")
    axes[0].set_xlabel("预测 STF 震源时间 (s)")
    axes[0].set_title("A  每秒重新预测的完整 STF 形状 (log10 归一矩率)", loc="left")
    axes[0].axhline(drop_horizon, color="#D55E00", linestyle="--", linewidth=1.0)
    colorbar = fig.colorbar(heat, ax=axes[0], pad=0.015)
    colorbar.set_label("log10(每秒矩比例)")

    _style_axis(axes[1])
    axes[1].plot(available_horizons, mw, color="#0072B2", label="总 STF 积分 Mw")
    axes[1].axhline(
        records[station_index].magnitude_catalog,
        color="#333333",
        linestyle="--",
        label="USGS Mw",
    )
    axes[1].set_ylabel("Mw")
    axes[1].set_title("B  非负 STF 的总积分仍可在相邻预测间下降", loc="left")
    axes[1].axvline(drop_horizon, color="#D55E00", linestyle="--", linewidth=1.0)
    axes[1].annotate(
        f"单秒下降 {float(worst['delta_mw']):.3f} Mw",
        xy=(drop_horizon, drop_mw),
        xytext=(8, 18),
        textcoords="offset points",
        color="#A83D00",
        arrowprops={"arrowstyle": "->", "color": "#A83D00", "lw": 0.8},
    )
    axes[1].legend(loc="best")

    _style_axis(axes[2])
    axes[2].plot(available_horizons, confirmed_fraction, color="#009E73", label="S 支持的确认矩 / 总预测矩")
    axes[2].plot(available_horizons, history_revision, color="#CC79A7", label="已确认历史的有符号改写比例")
    axes[2].axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[2].axvline(drop_horizon, color="#D55E00", linestyle="--", linewidth=1.0)
    axes[2].set_xlabel("观测时长 (s)")
    axes[2].set_ylabel("比例")
    axes[2].set_title("C  传播确认与历史重写", loc="left")
    axes[2].legend(loc="best")

    fig.suptitle(
        f"最大台站单秒下降示例：{worst['event']} / {worst['station']}",
        y=0.995,
        fontsize=13,
    )
    fig.subplots_adjust(left=0.10, right=0.92, bottom=0.07, top=0.94, hspace=0.42)
    return _save_figure(fig, output_stem)


def plot_streaming_input_example(
    record: RawStreamingStation,
    *,
    output_stem: Path,
) -> list[Path]:
    prefixes = [
        preprocess_streaming_prefix(record, observation_horizon_sec=horizon)
        for horizon in REPORT_PREFIX_HORIZONS
    ]
    _configure_chinese_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6), sharey=True)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for axis, prefix, color in zip(axes.flat, prefixes, colors):
        _style_axis(axis)
        sample_time = (
            record.waveform_start_sec
            + np.arange(prefix.values_m.size, dtype=np.float64)
        )
        axis.plot(sample_time, prefix.values_m * 100.0, color=color)
        axis.scatter(
            sample_time[~prefix.valid_mask],
            np.zeros(int(np.count_nonzero(~prefix.valid_mask))),
            facecolors="none",
            edgecolors="#555555",
            s=14,
            linewidth=0.7,
            label="缺测槽置零",
        )
        axis.set_xlim(
            record.waveform_start_sec,
            record.waveform_start_sec + prefix.values_m.size,
        )
        axis.set_title(
            f"输入 B×1×{prefix.values_m.size}；发布 {prefix.issue_time_sec:g} s",
            fontsize=9,
        )
        axis.set_xlabel("观测时间 (s)")
    axes[0, 0].set_ylabel("处理后 R 位移 (cm)")
    axes[1, 0].set_ylabel("处理后 R 位移 (cm)")
    fig.suptitle(
        f"真实流式模型输入示例：{record.event} / {record.station}",
        y=0.995,
        fontsize=13,
    )
    fig.text(
        0.5,
        0.015,
        "每幅图只包含当时可发布的真实前缀；横轴在 h 处结束，不向 200 s 补未来零尾。",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.91, hspace=0.43, wspace=0.20)
    return _save_figure(fig, output_stem)


def build_chinese_report(
    *,
    analysis: Mapping[str, Any],
    endpoint_gate: Mapping[str, Any],
    input_gate: Mapping[str, Any],
    stability: Mapping[str, Any],
    cohort: Mapping[str, Any],
) -> str:
    overall = analysis["overall"]
    worst_station = stability["worst_station_step"]
    worst_event = stability["worst_event_median_step"]
    largest_station_decrease = stability["largest_station_mw_decrease"]
    down = stability["confirmed_history_downward_revision_fraction"]
    late_down = stability[
        "late_180_200_confirmed_history_downward_revision_fraction"
    ]
    lines = [
        "# Phase39 真实流式回放与 STF 一致性审计",
        "",
        "## 结论",
        "",
        "这次回放已经把旧评估中的未来零尾移除。每个观测时刻 `h` 只使用发布时刻 "
        "`h+5 s` 前实际到达的原始 E/N，重新估计基线、旋转到 R、执行 7 点 "
        "Hamming FIR，然后把真实的 `B×1×h` 前缀送入固定 Phase39 seed42。",
        "",
        "模型输出的每条 STF 内部确实逐点非负；但相邻两秒是两次独立的完整 STF "
        "反演，网络没有状态继承、累计矩单调约束或历史冻结约束。因此后一秒可以把"
        "前一秒整条 STF 的幅值和形状一起调低，总积分矩与 Mw 都可以下降。",
        "",
        "本结果是固定离线模型的 streaming mismatch audit，不表示 Phase39 已经变成"
        "严格在线或严格因果模型。",
        "",
        "## 关键数值",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| h=200 Event MAE | {float(endpoint_gate['event_mae']):.6f} Mw |",
        f"| h=200 输入相对锁定端点最大差异 | {float(input_gate['max_abs_radial_input_diff_m']):.3g} m |",
        f"| 全部 {EXPECTED_STATION_COUNT} 台站可用起点 | {int(stability['first_full_station_availability_observation_sec'])} s 观测 / {int(stability['first_full_station_availability_observation_sec'] + PROCESSING_DELAY_SEC)} s 发布 |",
        f"| 全可用后事件中位数最大单秒跳变 | {float(stability['max_abs_event_median_step_mw_after_full_availability']):.6f} Mw |",
        f"| 全可用后台站最大单秒跳变 | {float(stability['max_abs_station_step_mw_after_full_availability']):.6f} Mw |",
        f"| 180–200 s 事件中位数最大单秒跳变 | {float(stability['late_180_200_max_abs_event_median_step_mw']):.6f} Mw |",
        f"| 180–200 s 台站最大单秒跳变 | {float(stability['late_180_200_max_abs_station_step_mw']):.6f} Mw |",
        f"| 已获 S 支持的历史被向下重写比例 | {float(down):.1%} |" if down is not None else "| 已获 S 支持的历史被向下重写比例 | 无可比较样本 |",
        f"| 180–200 s 已确认历史向下重写比例 | {float(late_down):.1%} |" if late_down is not None else "| 180–200 s 已确认历史向下重写比例 | 无可比较样本 |",
        "",
        f"整体 Event MAE 从 `{int(overall['stable_event_mae_within_target_observation_sec'])} s` "
        f"起持续不超过 `{TARGET_ERROR_MW:.2f} Mw`，但这只是误差带判据。"
        f"在 180–200 s 内，事件中位数仍可单秒变化 "
        f"`{float(stability['late_180_200_max_abs_event_median_step_mw']):.3f} Mw`，"
        "因此不能据此称为数值输出已经稳定。",
        "",
    ]
    if worst_event is not None:
        lines.extend(
            [
                f"最大事件中位数单秒跳变出现在 `{worst_event['event']}` 的 "
                f"`{worst_event['observation_horizon_sec']} s`："
                f"`{float(worst_event['delta_mw']):+.6f} Mw`。",
                "",
            ]
        )
    if worst_station is not None:
        lines.extend(
            [
                f"最大台站单秒跳变出现在 `{worst_station['event']} / "
                f"{worst_station['station']}` 的 "
                f"`{worst_station['observation_horizon_sec']} s`："
                f"`{float(worst_station['delta_mw']):+.6f} Mw`。",
                "",
            ]
        )
    if largest_station_decrease is not None:
        lines.extend(
            [
                f"最大台站单秒下降出现在 `{largest_station_decrease['event']} / "
                f"{largest_station_decrease['station']}` 的 "
                f"`{largest_station_decrease['observation_horizon_sec']} s`："
                f"`{float(largest_station_decrease['delta_mw']):+.6f} Mw`。"
                "第三、四图使用该台站。",
                "",
            ]
        )
    lines.extend(
        [
            "## 图件",
            "",
            "![八事件流式震级轨迹](figures/01_event_trajectories.png)",
            "",
            "![流式稳定性审计](figures/02_streaming_stability.png)",
            "",
            "![最大下降台站的 STF 重估](figures/03_worst_station_stf.png)",
            "",
            "![真实可变长度输入示例](figures/04_streaming_input_example.png)",
            "",
            "## 如何理解“非负 STF 但震级下降”",
            "",
            "对单次预测，`rate(t) ≥ 0` 保证该条 STF 的累计矩随震源时间不下降。"
            "它没有保证不同发布时间的两条曲线满足 "
            "`rate_(h+1)(t) ≥ rate_h(t)`，也没有保证两条曲线的积分单调。",
            "",
            "本报告把变化拆成三部分：上一秒已经通过 S 波传播得到支持的历史段改写、"
            "本秒新增确认段、以及仍属于预测的未来尾部。只有第一部分直接对应用户指出的"
            "“在原来基础上反悔”问题。",
            "",
            "## 下一步修复边界",
            "",
            "若要让输出真正沿上一秒累积，需要改模型或训练合同，而不是修改回放脚本。"
            "可行方向是因果编码器加状态、直接预测非负矩增量并累计，或对已获传播支持的 "
            "STF 历史加入跨时刻一致性约束。任何一种都属于新的科学变量，需要重新训练并"
            "按冻结 validation 协议比较，不能把本次诊断结果直接称为修复后的在线模型。",
            "",
            "## 数据与解释限制",
            "",
            f"- 固定 cohort：{cohort['event_count']} 个事件、{cohort['station_count']} 个台站。",
            "- 外部八事件仍是 `development_validation`，不是新的无偏最终测试集。",
            "- 台站身份来自完整 200 s cm0 cohort，因此台站选择本身仍是离线的。",
            "- 对称 TCN padding 和无 causal mask Transformer 均未改变。",
            "- S 波确认区间采用保守的完整 1 s bin 定义，不冻结尚未传播到台站的 STF 尾部。",
            "",
            "## 结果文件",
            "",
            "- `station_streaming_predictions.csv`：逐站逐秒总矩、确认矩、剩余矩和历史改写。",
            "- `event_streaming_predictions.csv`：逐事件台站中位数及跳变统计。",
            "- `horizon_streaming_metrics.csv`：逐秒总体稳定性指标。",
            "- `stf_replay.npz`：完整非负 STF 数组、可用掩码与传播确认步数。",
            "- `summary.json`、`cohort_contract.json`、`provenance.json`：冻结合同与输入哈希。",
            "",
            f"旧锁定端点复现容差为 `{float(endpoint_gate['tolerance_mw']):.3g} Mw`；"
            f"逐站最大差异为 `{float(endpoint_gate['max_station_prediction_abs_diff_mw']):.3g} Mw`。",
            "",
        ]
    )
    return "\n".join(lines)


def build_cohort_contract(
    records: Sequence[RawStreamingStation], *, horizons: Sequence[int]
) -> dict[str, Any]:
    event_counts = Counter(record.event for record in records)
    return {
        "evaluation_role": "development_validation",
        "event_count": len(event_counts),
        "station_count": len(records),
        "event_station_counts": dict(sorted(event_counts.items())),
        "phase_adjusted_station_count": sum(
            record.waveform_phase_adjusted for record in records
        ),
        "observation_horizons_sec": [int(value) for value in horizons],
        "release_time_rule": "issue_time_sec = observation_horizon_sec + 5",
        "raw_arrival_rule": "use finite E/N rows with timestamp <= issue_time_sec",
        "network_input_rule": "processed radial prefix only, shape B x 1 x h",
        "future_padding": "none",
        "station_selection": (
            "locked cm0 identities selected from complete 200 s records; offline cohort"
        ),
        "source_confirmation": (
            "floor(max(0, h - hypocentral_distance / beta)) complete 1 s bins"
        ),
        "interpretation": (
            "raw-streaming mismatch audit of a fixed noncausal offline model; "
            "not a repaired online model"
        ),
    }


def write_outputs(
    *,
    output_dir: Path,
    result: Mapping[str, Any],
    analysis: Mapping[str, Any],
    endpoint_gate: Mapping[str, Any],
    stability: Mapping[str, Any],
    records: Sequence[RawStreamingStation],
    cohort: Mapping[str, Any],
    provenance: Mapping[str, Any],
    create_report: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    station_fields = (
        "event", "station", "observation_horizon_sec", "release_time_sec",
        "model_input_steps", "mw_pred", "mw_catalog", "error", "m0_total_nm",
        "p_visible_source_end_sec", "p_visible_source_steps",
        "s_confirmed_source_end_sec", "s_confirmed_source_steps",
        "m0_p_visible_nm", "m0_confirmed_nm", "mw_confirmed", "m0_remaining_nm",
        "confirmed_moment_fraction", "raw_sample_count_at_issue",
        "prefix_valid_sample_count", "prefix_valid_fraction", "baseline_m",
        "baseline_source", "waveform_start_sec", "waveform_phase_adjusted",
        "source_distance_m", "p_arrival_sec", "s_arrival_sec",
        "previous_observation_horizon_sec", "delta_mw_from_previous",
        "delta_m0_total_nm", "previous_confirmed_source_steps",
        "previous_confirmed_moment_nm", "confirmed_history_revision_nm",
        "confirmed_history_revision_fraction", "confirmed_history_l1_revision_nm",
        "confirmed_history_l1_revision_fraction",
        "confirmed_history_moment_decreased", "newly_confirmed_moment_nm",
        "future_tail_revision_nm", "future_tail_l1_revision_nm",
        "full_stf_l1_revision_nm", "full_stf_l1_revision_fraction",
    )
    _write_csv(
        output_dir / "station_streaming_predictions.csv",
        result["station_rows"],
        fieldnames=station_fields,
    )
    _write_csv(
        output_dir / "unavailable_streaming_stations.csv",
        result["unavailable_rows"],
        fieldnames=(
            "event", "station", "observation_horizon_sec", "release_time_sec",
            "reason", "detail",
        ),
    )
    _write_csv(
        output_dir / "event_streaming_predictions.csv",
        analysis["event_rows"],
        fieldnames=(
            "event", "observation_horizon_sec", "release_time_sec",
            "mw_pred_median", "mw_catalog", "error", "abs_error", "within_target",
            "station_count", "station_mw_iqr", "station_mw_range",
            "event_median_delta_mw", "median_station_delta_mw",
            "max_abs_station_delta_mw", "station_total_mw_decrease_count",
            "confirmed_history_comparison_count", "confirmed_history_decrease_count",
            "confirmed_history_decrease_fraction",
            "median_confirmed_history_revision_fraction",
            "median_confirmed_history_l1_revision_fraction",
            "median_confirmed_moment_fraction",
        ),
    )
    _write_csv(
        output_dir / "horizon_streaming_metrics.csv",
        analysis["horizon_metrics"],
        fieldnames=(
            "observation_horizon_sec", "release_time_sec", "event_count",
            "total_event_count", "coverage", "coverage_fraction",
            "available_station_count", "unavailable_station_count",
            "unavailable_reason_counts", "event_equal_mae", "event_equal_rmse",
            "event_equal_bias", "within_target_event_count",
            "within_target_event_fraction", "median_abs_event_median_step_mw",
            "max_abs_event_median_step_mw", "event_median_decrease_count",
            "median_abs_station_step_mw", "max_abs_station_step_mw",
            "station_mw_decrease_count", "confirmed_history_comparison_count",
            "confirmed_history_decrease_count", "confirmed_history_decrease_fraction",
            "median_confirmed_history_revision_fraction",
            "median_confirmed_history_l1_revision_fraction",
        ),
    )
    _write_csv(
        output_dir / "event_convergence.csv",
        analysis["event_convergence"],
        fieldnames=(
            "event", "mw_catalog", "final_mw_pred_median", "final_error",
            "final_abs_error", "final_station_count",
            "first_available_observation_sec", "first_within_target_observation_sec",
            "first_within_target_release_sec", "stable_within_target_observation_sec",
            "stable_within_target_release_sec", "right_censored_at_200_sec",
        ),
    )
    np.savez_compressed(
        output_dir / "stf_replay.npz",
        rate_nm_per_s=np.asarray(result["rate_nm_per_s"], dtype=np.float32),
        available_mask=np.asarray(result["available_mask"], dtype=bool),
        confirmed_source_steps=np.asarray(
            result["confirmed_source_steps"], dtype=np.int16
        ),
        observation_horizons_sec=np.asarray(
            cohort["observation_horizons_sec"], dtype=np.int16
        ),
        source_time_sec=np.arange(SOURCE_TIME_STEPS, dtype=np.float32),
        event=np.asarray([record.event for record in records]),
        station=np.asarray([record.station for record in records]),
        magnitude_catalog=np.asarray(
            [record.magnitude_catalog for record in records], dtype=np.float32
        ),
        source_distance_m=np.asarray(
            [record.source_distance_m for record in records], dtype=np.float64
        ),
        p_arrival_sec=np.asarray(
            [record.p_arrival_sec for record in records], dtype=np.float64
        ),
        s_arrival_sec=np.asarray(
            [record.s_arrival_sec for record in records], dtype=np.float64
        ),
    )
    _write_json(output_dir / "cohort_contract.json", cohort)
    _write_json(output_dir / "provenance.json", provenance)
    summary = {
        "status": "complete" if analysis["complete_second_grid"] else "smoke_complete",
        "model": "Phase39 Glehman scalar + global invariant, seed42",
        "evaluation_role": "development_validation",
        "evaluation_type": "raw_streaming_mismatch_audit",
        "network_input": "variable length B x 1 x h without future zero padding",
        "endpoint_input_reproduction": dict(result["input_endpoint_gate"]),
        "endpoint_prediction_reproduction": dict(endpoint_gate),
        "overall": analysis["overall"],
        "streaming_stability": dict(stability),
        "event_convergence": analysis["event_convergence"],
    }
    _write_json(output_dir / "summary.json", summary)

    if create_report:
        figures = output_dir / "figures"
        plot_event_trajectories(
            analysis, output_stem=figures / "01_event_trajectories"
        )
        plot_streaming_stability(
            result, output_stem=figures / "02_streaming_stability"
        )
        plot_worst_station_stf(
            result,
            records,
            stability,
            output_stem=figures / "03_worst_station_stf",
        )
        worst = stability["largest_station_mw_decrease"]
        if worst is None:
            input_record = records[0]
        else:
            input_record = records[
                _station_index(
                    records,
                    event=str(worst["event"]),
                    station=str(worst["station"]),
                )
            ]
        plot_streaming_input_example(
            input_record,
            output_stem=figures / "04_streaming_input_example",
        )
        (output_dir / "REPORT_ZH.md").write_text(
            build_chinese_report(
                analysis=analysis,
                endpoint_gate=endpoint_gate,
                input_gate=result["input_endpoint_gate"],
                stability=stability,
                cohort=cohort,
            ),
            encoding="utf-8",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay raw external E/N streams through the frozen Phase39 seed42 "
            "checkpoint using variable-length radial prefixes and audit STF revisions."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New or empty directory for streaming replay artifacts.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Inference device (default: cpu for locked endpoint reproduction).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EXPECTED_STATION_COUNT,
        help="Station batch size at each horizon (default: all 158 stations).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Replay only 20, 100, and 200 s and skip report figures.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    validate_output_dir(output_dir)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    horizons = SMOKE_HORIZONS if args.smoke else FULL_HORIZONS

    print("[1/8] validating frozen Phase39 artifacts", flush=True)
    artifact_hashes = validate_frozen_artifacts()
    config = load_phase39_config()
    labels = load_label_contract(LABELS_PATH)
    endpoint_reference = load_endpoint_reference()

    print("[2/8] hashing the eight external raw event bundles", flush=True)
    external_hashes = hash_external_inputs(EXTERNAL_EVENT_ROOT)

    print("[3/8] loading raw E/N once and locking the 158 station identities", flush=True)
    records = build_raw_streaming_records(
        config=config,
        event_root=EXTERNAL_EVENT_ROOT,
        labels_by_dir=labels,
        expected_station_keys=set(endpoint_reference["station_predictions"]),
    )
    if len(records) != EXPECTED_STATION_COUNT:
        raise ValueError("streaming station count changed")

    eval_config = copy.deepcopy(config)
    eval_config["dataset"]["radial_peak_min_cm"] = 0.0
    device = _resolve_device(args.device)
    print(
        f"[4/8] loading fixed seed42 on {device} and replaying {len(horizons)} horizons",
        flush=True,
    )
    model = load_model(eval_config, device=device)
    result = evaluate_streaming_replay(
        model,
        eval_config,
        records,
        horizons=horizons,
        batch_size=args.batch_size,
    )

    print("[5/8] enforcing h=200 input and prediction endpoint gates", flush=True)
    endpoint_gate = validate_endpoint_reproduction(result, endpoint_reference)
    analysis = analyze_convergence(result, horizons=horizons)
    stability = summarize_streaming_stability(result, analysis)
    cohort = build_cohort_contract(records, horizons=horizons)
    if cohort["event_count"] != EXPECTED_EVENT_COUNT:
        raise ValueError("streaming event count changed")

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "horizons_sec": list(horizons),
        "processing_delay_sec": PROCESSING_DELAY_SEC,
        "raw_crop_boundary": "timestamp <= issue_time_sec",
        "artifact_paths": {name: str(path) for name, path in FROZEN_PATHS.items()},
        "artifact_sha256": artifact_hashes,
        "external_event_root": str(EXTERNAL_EVENT_ROOT),
        "external_input_hashes": external_hashes,
        "checkpoint_selection": (
            "fixed Phase39 seed42 selected previously by formal validation; "
            "no training, seed selection, threshold tuning, or ensemble here"
        ),
        "scientific_boundary": (
            "streaming mismatch audit of a symmetric-TCN/unmasked-Transformer "
            "offline checkpoint, not a repaired causal model"
        ),
    }

    print("[6/8] writing replay tables, STF arrays, and provenance", flush=True)
    write_outputs(
        output_dir=output_dir,
        result=result,
        analysis=analysis,
        endpoint_gate=endpoint_gate,
        stability=stability,
        records=records,
        cohort=cohort,
        provenance=provenance,
        create_report=not args.smoke,
    )
    print("[7/8] validating persisted STF nonnegativity", flush=True)
    finite_rates = np.asarray(result["rate_nm_per_s"])[
        np.asarray(result["available_mask"])
    ]
    if finite_rates.size == 0 or np.any(~np.isfinite(finite_rates)):
        raise ValueError("persisted available STF rates are non-finite")
    if np.any(finite_rates < 0.0):
        raise ValueError("persisted STF rates must be nonnegative")

    print("[8/8] complete", flush=True)
    print(f"output_dir={output_dir}")
    print(f"endpoint_event_mae={float(endpoint_gate['event_mae']):.12f}")
    print(
        "endpoint_input_max_abs_diff_m="
        f"{float(result['input_endpoint_gate']['max_abs_radial_input_diff_m']):.12g}"
    )
    if not args.smoke:
        print(
            "max_abs_event_median_step_mw="
            f"{float(stability['max_abs_event_median_step_mw_after_full_availability']):.12f}"
        )
        print(
            "max_abs_station_step_mw="
            f"{float(stability['max_abs_station_step_mw_after_full_availability']):.12f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
