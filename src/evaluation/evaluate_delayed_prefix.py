"""In-memory delayed-prefix evaluation for STF-only v2 station models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from numbers import Integral
from typing import Any

import numpy as np
import torch

from src.data.metadata import build_metadata_tensor, metadata_distance_from_config
from src.data.waveform import ProcessedWaveform, waveform_config_from_v2
from src.evaluation.delayed_prefix import (
    MANUSCRIPT_PROCESSING_DELAY_SEC,
    fir_lookahead_samples,
    release_delayed_prefix,
)
from src.models.model import PINNModel
from src.training.loss_stf_rate_v2 import moment_magnitude_from_rate
from src.utils.config_v2 import (
    validate_config_v2,
    waveform_input_components_from_config,
)


DEFAULT_HORIZONS_SEC = tuple(range(20, 201, 20))


def _model_device_and_dtype(model: PINNModel) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(model.parameters())
    except StopIteration as exc:  # pragma: no cover - PINNModel always has parameters
        raise ValueError("model has no parameters") from exc
    return parameter.device, parameter.dtype


def _validate_contract(model: PINNModel, config: dict[str, Any]) -> None:
    if not isinstance(model, PINNModel):
        raise TypeError("delayed-prefix evaluation requires PINNModel")
    if config.get("workflow") is not None:
        raise ValueError("delayed-prefix evaluation forbids workflow models")
    validate_config_v2(config)
    if waveform_input_components_from_config(config) != ("radial",):
        raise ValueError("delayed-prefix evaluation requires R-only input")
    if tuple(getattr(model, "input_components", ())) != ("radial",):
        raise ValueError("model is not R-only")
    if bool(config.get("model", {}).get("predict_catalog_mw", False)):
        raise ValueError("delayed-prefix evaluation forbids a catalog Mw head")
    if getattr(model, "magnitude_head", None) is not None:
        raise ValueError("model has an enabled catalog Mw head")

    waveform_config = waveform_config_from_v2(config)
    fir_lookahead_samples(waveform_config)
    if not math.isclose(
        waveform_config.max_interpolation_gap_sec,
        0.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "delayed-prefix evaluation requires max_interpolation_gap_sec=0"
        )


def _validate_horizons(
    horizons: Sequence[int],
    config: dict[str, Any],
) -> tuple[tuple[int, int], ...]:
    values = tuple(horizons)
    if not values:
        raise ValueError("horizons must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in values
    ):
        raise ValueError("horizons must contain integer seconds")
    normalized = tuple(int(value) for value in values)
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError("horizons must be strictly increasing and unique")

    dataset = config["dataset"]
    sample_rate_hz = float(dataset["sample_rate_hz"])
    duration_sec = float(dataset["waveform"]["duration_sec"])
    maximum_steps = int(round(duration_sec * sample_rate_hz))
    result: list[tuple[int, int]] = []
    for horizon_sec in normalized:
        prefix_steps_float = horizon_sec * sample_rate_hz
        prefix_steps = int(round(prefix_steps_float))
        if horizon_sec <= 0 or not math.isclose(
            prefix_steps_float,
            float(prefix_steps),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("horizon is not aligned to the waveform sample grid")
        if prefix_steps > maximum_steps:
            raise ValueError("horizon exceeds the configured waveform duration")
        result.append((horizon_sec, prefix_steps))
    return tuple(result)


def _batch_strings(value: Any, batch_size: int, name: str) -> list[str]:
    if isinstance(value, str):
        if batch_size != 1:
            raise ValueError(f"{name} must contain one value per station")
        return [value]
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain one value per station") from exc
    if len(values) != batch_size:
        raise ValueError(f"{name} must contain one value per station")
    return [str(item) for item in values]


def _batch_vector(
    batch: Mapping[str, Any],
    name: str,
    batch_size: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if name not in batch:
        raise ValueError(f"batch is missing {name}")
    value = torch.as_tensor(batch[name], dtype=dtype).reshape(-1)
    if value.shape != (batch_size,):
        raise ValueError(f"{name} must have shape (batch,)")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _processed_waveforms(
    batch: Mapping[str, Any],
    *,
    expected_steps: int,
) -> tuple[list[ProcessedWaveform], torch.Tensor]:
    if "radial" not in batch or "waveform_valid_mask" not in batch:
        raise ValueError("batch requires radial and waveform_valid_mask")
    radial = torch.as_tensor(batch["radial"]).detach().cpu()
    if radial.ndim == 2:
        radial = radial.unsqueeze(1)
    if radial.ndim != 3 or radial.shape[1] != 1:
        raise ValueError("radial must have shape (batch,1,time)")
    if radial.shape[-1] != expected_steps:
        raise ValueError("radial length differs from the configured waveform")
    if not radial.is_floating_point() or not bool(torch.isfinite(radial).all()):
        raise ValueError("radial must be a finite floating-point tensor")

    valid_mask = torch.as_tensor(batch["waveform_valid_mask"]).detach().cpu().bool()
    if valid_mask.ndim == 3 and valid_mask.shape[1] == 1:
        valid_mask = valid_mask[:, 0]
    if valid_mask.shape != (radial.shape[0], expected_steps):
        raise ValueError("waveform_valid_mask must have shape (batch,time)")
    invalid_values = radial[:, 0][~valid_mask]
    if invalid_values.numel() and bool(torch.count_nonzero(invalid_values)):
        raise ValueError("processed radial values must be zero where the mask is false")

    batch_size = radial.shape[0]
    dt = _batch_vector(batch, "waveform_dt_sec", batch_size, dtype=torch.float64)
    raw_dt = (
        _batch_vector(batch, "raw_dt_sec", batch_size, dtype=torch.float64)
        if "raw_dt_sec" in batch
        else dt
    )
    if "baseline_source" not in batch:
        raise ValueError("batch is missing baseline_source")
    baseline_sources = _batch_strings(
        batch["baseline_source"], batch_size, "baseline_source"
    )
    if any(source not in {"pre_event", "pre_p"} for source in baseline_sources):
        raise ValueError("baseline_source must be pre_event or pre_p")

    processed: list[ProcessedWaveform] = []
    for index in range(batch_size):
        dt_sec = float(dt[index])
        if dt_sec <= 0.0:
            raise ValueError("waveform_dt_sec must be positive")
        mask = valid_mask[index].numpy()
        processed.append(
            ProcessedWaveform(
                time_sec=(
                    np.arange(expected_steps, dtype=np.float64) * dt_sec
                ).astype(np.float32),
                values_m=radial[index, 0].numpy(),
                valid_mask=mask,
                dt_sec=dt_sec,
                raw_dt_sec=float(raw_dt[index]),
                baseline_m=0.0,
                baseline_source=baseline_sources[index],
                valid_fraction=float(np.mean(mask)),
            )
        )
    return processed, radial


def _event_outputs(
    station_rows: list[dict[str, Any]],
    unavailable_rows: list[dict[str, Any]],
    horizons: tuple[tuple[int, int], ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total_events = len(
        {
            str(row["event"])
            for row in [*station_rows, *unavailable_rows]
        }
    )
    event_rows: list[dict[str, Any]] = []
    horizon_metrics: list[dict[str, Any]] = []
    for horizon_sec, _ in horizons:
        selected = [
            row
            for row in station_rows
            if int(row["observation_horizon_sec"]) == horizon_sec
        ]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in selected:
            grouped.setdefault(str(row["event"]), []).append(row)

        current_events: list[dict[str, Any]] = []
        for event in sorted(grouped):
            rows = grouped[event]
            predictions = np.asarray(
                [float(row["mw_pred"]) for row in rows], dtype=np.float64
            )
            catalogs = np.asarray(
                [float(row["mw_catalog"]) for row in rows], dtype=np.float64
            )
            if not np.allclose(catalogs, catalogs[0], rtol=0.0, atol=1.0e-6):
                raise ValueError(f"catalog magnitude differs within event {event}")
            predicted_median = float(np.median(predictions))
            catalog_mw = float(np.median(catalogs))
            current_events.append(
                {
                    "event": event,
                    "observation_horizon_sec": float(horizon_sec),
                    "release_time_sec": float(
                        horizon_sec + MANUSCRIPT_PROCESSING_DELAY_SEC
                    ),
                    "mw_pred_median": predicted_median,
                    "mw_catalog": catalog_mw,
                    "error": predicted_median - catalog_mw,
                    "station_count": len(rows),
                }
            )
        event_rows.extend(current_events)

        unavailable = [
            row
            for row in unavailable_rows
            if int(row["observation_horizon_sec"]) == horizon_sec
        ]
        reason_counts: dict[str, int] = {}
        for row in unavailable:
            reason = str(row["reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        errors = np.asarray(
            [float(row["error"]) for row in current_events], dtype=np.float64
        )
        event_count = int(errors.size)
        horizon_metrics.append(
            {
                "observation_horizon_sec": float(horizon_sec),
                "release_time_sec": float(
                    horizon_sec + MANUSCRIPT_PROCESSING_DELAY_SEC
                ),
                "event_count": event_count,
                "total_event_count": total_events,
                "coverage": event_count,
                "coverage_fraction": (
                    event_count / total_events if total_events else 0.0
                ),
                "available_station_count": len(selected),
                "unavailable_station_count": len(unavailable),
                "unavailable_reason_counts": reason_counts,
                "event_equal_mae": (
                    float(np.mean(np.abs(errors)))
                    if errors.size
                    else float("nan")
                ),
                "event_equal_rmse": (
                    float(np.sqrt(np.mean(errors**2)))
                    if errors.size
                    else float("nan")
                ),
                "event_equal_bias": (
                    float(np.mean(errors)) if errors.size else float("nan")
                ),
            }
        )
    return event_rows, horizon_metrics


def _cohort_contract(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    duration_sec = float(dataset["waveform"]["duration_sec"])
    threshold_cm = float(dataset["radial_peak_min_cm"])
    return {
        "cohort": (
            f"processed radial peak over the full {duration_sec:g} s record "
            f">= {threshold_cm:g} cm"
        ),
        "radial_peak_min_cm": threshold_cm,
        "waveform_prefix_causal": True,
        "station_selection_causal": False,
        "end_to_end_causal": False,
    }


def evaluate_delayed_prefix(
    model: PINNModel,
    config: dict[str, Any],
    loader: Iterable[Mapping[str, Any]],
    horizons: Sequence[int] = DEFAULT_HORIZONS_SEC,
) -> dict[str, Any]:
    """Evaluate independently released R prefixes without writing artifacts."""

    _validate_contract(model, config)
    validated_horizons = _validate_horizons(horizons, config)
    waveform_config = waveform_config_from_v2(config)
    device, dtype = _model_device_and_dtype(model)
    dataset = config["dataset"]
    expected_steps = int(
        round(
            float(dataset["waveform"]["duration_sec"])
            * float(dataset["sample_rate_hz"])
        )
    )
    expected_stf_steps = int(
        round(
            float(dataset["stf"]["duration_sec"])
            * float(dataset["sample_rate_hz"])
        )
    )
    stf_m_ref = float(dataset["stf"]["m_ref"])
    alpha_m_per_s = float(config["physics"]["alpha"])
    baseline_fallback_max_sec = float(
        dataset["baseline"]["fallback_max_sec"]
    )

    station_rows: list[dict[str, Any]] = []
    unavailable_rows: list[dict[str, Any]] = []
    seen_samples = 0
    original_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                processed, radial = _processed_waveforms(
                    batch,
                    expected_steps=expected_steps,
                )
                batch_size = radial.shape[0]
                seen_samples += batch_size
                events = _batch_strings(batch.get("event"), batch_size, "event")
                stations = _batch_strings(
                    batch.get("station"), batch_size, "station"
                )
                catalog_mw = _batch_vector(
                    batch,
                    "magnitude_catalog",
                    batch_size,
                    dtype=torch.float64,
                )
                source_dt_sec = _batch_vector(
                    batch,
                    "stf_dt_sec",
                    batch_size,
                    dtype=dtype,
                ).to(device)
                source_distance_values = _batch_vector(
                    batch,
                    "source_distance_m",
                    batch_size,
                    dtype=torch.float64,
                )
                source_distance_m = source_distance_values.to(
                    device=device,
                    dtype=dtype,
                )
                epicentral_distance_m = _batch_vector(
                    batch,
                    "epicentral_distance_m",
                    batch_size,
                    dtype=dtype,
                ).to(device)
                theta_deg = _batch_vector(
                    batch, "theta_deg", batch_size, dtype=dtype
                ).to(device)
                azimuth_deg = _batch_vector(
                    batch, "azimuth_deg", batch_size, dtype=dtype
                ).to(device)
                metadata_distance_m = metadata_distance_from_config(
                    config,
                    source_distance_m=source_distance_m,
                    epicentral_distance_m=epicentral_distance_m,
                )
                metadata = build_metadata_tensor(
                    metadata_distance_m,
                    theta_deg,
                    azimuth_deg,
                )

                for horizon_sec, prefix_steps in validated_horizons:
                    releases = [
                        release_delayed_prefix(
                            item,
                            prefix_steps=prefix_steps,
                            config=waveform_config,
                        )
                        for item in processed
                    ]
                    available_indices: list[int] = []
                    for index, (item, release) in enumerate(
                        zip(processed, releases)
                    ):
                        if item.baseline_source == "pre_p":
                            baseline_ready_sec = min(
                                float(source_distance_values[index])
                                / alpha_m_per_s,
                                baseline_fallback_max_sec,
                            )
                            if release.issue_time_sec + 1.0e-9 < baseline_ready_sec:
                                unavailable_rows.append(
                                    {
                                        "event": events[index],
                                        "station": stations[index],
                                        "observation_horizon_sec": float(
                                            horizon_sec
                                        ),
                                        "release_time_sec": float(
                                            release.issue_time_sec
                                        ),
                                        "reason": "pre_p_baseline_not_available",
                                        "baseline_ready_time_sec": float(
                                            baseline_ready_sec
                                        ),
                                    }
                                )
                                continue
                        available_indices.append(index)
                    if not available_indices:
                        continue

                    masked = torch.as_tensor(
                        np.stack(
                            [
                                releases[index].masked_waveform_m
                                for index in available_indices
                            ]
                        ),
                        device=device,
                        dtype=dtype,
                    ).unsqueeze(1)
                    selected_indices = torch.as_tensor(
                        available_indices,
                        device=device,
                        dtype=torch.long,
                    )
                    encoded_stf = model(
                        masked,
                        meta=metadata.index_select(0, selected_indices),
                    )
                    if (
                        encoded_stf.ndim != 2
                        or encoded_stf.shape[0] != len(available_indices)
                        or encoded_stf.shape[1] != expected_stf_steps
                    ):
                        raise ValueError(
                            "model STF shape differs from the configured source window"
                        )
                    if not bool(torch.isfinite(encoded_stf).all()):
                        raise FloatingPointError("model returned a non-finite STF")
                    rate_nm_per_s = torch.clamp(
                        stf_m_ref * (torch.pow(10.0, encoded_stf) - 1.0),
                        min=0.0,
                    )
                    mw_pred = moment_magnitude_from_rate(
                        rate_nm_per_s,
                        source_dt_sec.index_select(0, selected_indices),
                    )
                    if not bool(torch.isfinite(mw_pred).all()):
                        raise FloatingPointError("STF-integrated Mw is non-finite")

                    for prediction_index, index in enumerate(available_indices):
                        release = releases[index]
                        expected_release = (
                            horizon_sec + MANUSCRIPT_PROCESSING_DELAY_SEC
                        )
                        if not math.isclose(
                            release.issue_time_sec,
                            expected_release,
                            rel_tol=0.0,
                            abs_tol=1.0e-9,
                        ):
                            raise RuntimeError("delayed-prefix release time mismatch")
                        prediction = float(
                            mw_pred[prediction_index].detach().cpu()
                        )
                        catalog = float(catalog_mw[index])
                        station_rows.append(
                            {
                                "event": events[index],
                                "station": stations[index],
                                "observation_horizon_sec": float(horizon_sec),
                                "release_time_sec": float(release.issue_time_sec),
                                "mw_pred": prediction,
                                "mw_catalog": catalog,
                                "error": prediction - catalog,
                            }
                        )
    finally:
        model.train(original_training)

    if seen_samples == 0:
        raise ValueError("loader produced no station samples")
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
    keys = {
        (
            str(row["event"]),
            str(row["station"]),
            float(row["observation_horizon_sec"]),
        )
        for row in station_rows
    }
    if len(keys) != len(station_rows):
        raise ValueError("loader contains duplicate event/station samples")

    event_rows, horizon_metrics = _event_outputs(
        station_rows,
        unavailable_rows,
        validated_horizons,
    )
    return {
        "cohort": _cohort_contract(config),
        "station_rows": station_rows,
        "unavailable_rows": unavailable_rows,
        "event_rows": event_rows,
        "horizon_metrics": horizon_metrics,
    }


__all__ = ["DEFAULT_HORIZONS_SEC", "evaluate_delayed_prefix"]
