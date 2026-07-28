from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.waveform import waveform_config_from_v2  # noqa: E402
from src.evaluation.evaluate_delayed_prefix import (  # noqa: E402
    evaluate_delayed_prefix,
)
from src.evaluation.evaluate_unseen import (  # noqa: E402
    _format_event_display_name,
    _station_sample_from_bundle,
    load_event_bundle,
)
from src.models.model import PINNModel  # noqa: E402
from src.utils.config_v2 import validate_config_on_startup  # noqa: E402


MODEL_DIR = Path(
    "/home/lihe/PINN_Mag/runs/"
    "phase39-manuscript-stf-glehman-scalar-20260726T111256Z-7fbb7a7/"
    "train/candidate/seed_42/models/20260726_192953"
)
CHECKPOINT_PATH = MODEL_DIR / "best_model.pth"
CONFIG_PATH = MODEL_DIR / "config.yaml"
SNAPSHOT_PATH = Path(
    "/home/lihe/PINN_Mag/data/magnitude-label-snapshots/"
    "usgs-priority-20260723T044422Z-40d808a/"
    "gnss_events_matched.usgs_priority.npz"
)
LABELS_PATH = SNAPSHOT_PATH.parent / "external_magnitude_labels.csv"
EXTERNAL_EVENT_ROOT = Path(
    "/home/lihe/PINN_Mag/incoming/legacy-task17/GNSS_EQDATA"
)
LOCKED_EXTERNAL_EVAL_PATH = Path(
    "/home/lihe/PINN_Mag/runs/fable-eval-20260726T125507Z/"
    "external_eval.json"
)
LOCKED_ENDPOINT_DIR = Path(
    "/home/lihe/PINN_Mag/runs/fable-eval-20260726T125507Z/"
    "glehman_gi_seed42/cm0/raw"
)
LOCKED_STATION_PATH = LOCKED_ENDPOINT_DIR / "station_predictions.csv"
LOCKED_EVENT_PATH = LOCKED_ENDPOINT_DIR / "event_summary.csv"

EXTERNAL_EVENT_NAMES = (
    "iquique-aftershock-2014-chile",
    "nepal-aftershock-2015",
    "kodiak-2018-alaska",
    "samos-2020-greece",
    "luding-2022-china",
    "xizang-2025-southern-tibetan-plateau",
    "myanmar-2025-mandalay",
    "sand-point-2025-alaska",
)

EXPECTED_SHA256 = {
    "checkpoint": "73500f365a58b248204d02333716f31674435927e9fc1c7d55a1453786b406f7",
    "config": "a05181166c7f40cae755ffbfbd0f4adfdb6a83703a89299677fa2be5f8ff1966",
    "snapshot": "2e1fa4c12fc1eb03ffc8bf9235491f0886d4b0d360ebcb3486baca4d948cfd6a",
    "labels": "f5c7db67ee362bd7898a2270d43774ff1f21f4f28318442d2bb3828319a7a435",
    "external_eval": "c903a75e85503d864bb44ad243ac6c476a9e4d9069c211f3dee3ae36b4204abf",
    "endpoint_stations": "fbb21c5f95e40505d655c84e99c9526839853de559e65b8380901e6b99a35ebf",
    "endpoint_events": "197d89b1620d4f939553ae3adf2965e4e5e12bcfe0e2dcc1f8dcea12fe43fff6",
}
FROZEN_PATHS = {
    "checkpoint": CHECKPOINT_PATH,
    "config": CONFIG_PATH,
    "snapshot": SNAPSHOT_PATH,
    "labels": LABELS_PATH,
    "external_eval": LOCKED_EXTERNAL_EVAL_PATH,
    "endpoint_stations": LOCKED_STATION_PATH,
    "endpoint_events": LOCKED_EVENT_PATH,
}

EXPECTED_EVENT_COUNT = 8
EXPECTED_STATION_COUNT = 158
EXPECTED_ENDPOINT_EVENT_MAE = 0.14773725867271437
TARGET_ERROR_MW = 0.15
PROCESSING_DELAY_SEC = 5.0
FULL_HORIZONS = tuple(range(1, 201))
SMOKE_HORIZONS = (20, 100, 200)
ENDPOINT_TOLERANCE_MW = 2.0e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def validate_frozen_artifacts() -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, path in FROZEN_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen {name} artifact: {path}")
        actual[name] = sha256_file(path)
        if actual[name] != EXPECTED_SHA256[name]:
            raise ValueError(
                f"frozen {name} SHA-256 changed: {actual[name]} != "
                f"{EXPECTED_SHA256[name]}"
            )
    return actual


def load_phase39_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config_on_startup(config)

    dataset = config["dataset"]
    model = config["model"]
    training = config["training"]
    loss = training["stf_rate_loss"]
    expected = {
        "sample_rate_hz": (float(dataset["sample_rate_hz"]), 1.0),
        "waveform_duration_sec": (
            float(dataset["waveform"]["duration_sec"]),
            200.0,
        ),
        "max_interpolation_gap_sec": (
            float(dataset["waveform"]["max_interpolation_gap_sec"]),
            0.0,
        ),
        "filter_cutoff_hz": (float(dataset["filter"]["cutoff_hz"]), 0.2),
        "filter_num_taps": (int(dataset["filter"]["num_taps"]), 7),
        "seed": (int(training["random_seed"]), 42),
        "lambda_MSE": (float(loss["lambda_MSE"]), 1.0),
        "lambda_synth": (float(loss["lambda_synth"]), 0.5),
        "lambda_mag": (float(loss["lambda_mag"]), 1.0),
        "lambda_shape": (float(loss["lambda_shape"]), 0.0),
    }
    changed = [name for name, (actual, wanted) in expected.items() if actual != wanted]
    if changed:
        raise ValueError("Phase39 numerical contract changed: " + ", ".join(changed))
    if tuple(model["input_components"]) != ("radial",):
        raise ValueError("Phase39 evaluation requires R-only input")
    if model["stf_output_parameterization"] != "moment_shape_factorized":
        raise ValueError("Phase39 STF output parameterization changed")
    if loss["synth_polarity_mode"] != "global_invariant":
        raise ValueError("Phase39 global-invariant synth contract changed")
    if loss["radiation_coefficient_contract"] != "glehman_scalar":
        raise ValueError("Phase39 radiation coefficient contract changed")
    if config["evaluation"]["aggregation"] != "event_median":
        raise ValueError("Phase39 event aggregation changed")
    if config["evaluation"]["external_role"] != "development_validation":
        raise ValueError("Phase39 external evaluation role changed")
    if Path(config["paths"]["data_path"]).resolve() != SNAPSHOT_PATH.resolve():
        raise ValueError("Phase39 config no longer points to the frozen snapshot")
    return config


def load_label_contract(
    path: Path,
    *,
    expected_event_dirs: Sequence[str] = EXTERNAL_EVENT_NAMES,
) -> dict[str, dict[str, Any]]:
    rows = _read_csv(path)
    labels: dict[str, dict[str, Any]] = {}
    seen_events: set[str] = set()
    for row in rows:
        event_dir = str(row["event_dir"])
        event = str(row["event"])
        if event_dir in labels:
            raise ValueError(f"duplicate external label event_dir: {event_dir}")
        if event in seen_events:
            raise ValueError(f"duplicate external label event: {event}")
        seen_events.add(event)
        labels[event_dir] = {
            "event": event,
            "event_dir": event_dir,
            "mw_selected": _finite_float(row["mw_selected"], field="mw_selected"),
            "mw_source": str(row["mw_source"]),
            "usgs_event_id": str(row["usgs_event_id"]),
        }
    expected = set(expected_event_dirs)
    if set(labels) != expected:
        missing = sorted(expected - set(labels))
        extra = sorted(set(labels) - expected)
        raise ValueError(f"external label event set changed; missing={missing}, extra={extra}")
    return labels


def load_endpoint_reference(
    *,
    external_eval_path: Path = LOCKED_EXTERNAL_EVAL_PATH,
    station_path: Path = LOCKED_STATION_PATH,
    event_path: Path = LOCKED_EVENT_PATH,
) -> dict[str, Any]:
    summary = json.loads(external_eval_path.read_text(encoding="utf-8"))
    phase39 = summary["checkpoints"]["glehman_gi_seed42"]["thresholds"]["cm0"]
    event_mae = _finite_float(phase39["event_mae"], field="event_mae")
    if not math.isclose(
        event_mae,
        EXPECTED_ENDPOINT_EVENT_MAE,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("locked Phase39 endpoint Event MAE changed")
    if Path(phase39["raw_output_dir"]).resolve() != LOCKED_ENDPOINT_DIR.resolve():
        raise ValueError("locked Phase39 raw endpoint directory changed")

    stations: dict[tuple[str, str], float] = {}
    for row in _read_csv(station_path):
        key = (str(row["event"]), str(row["station"]))
        if key in stations:
            raise ValueError(f"duplicate locked endpoint station: {key}")
        stations[key] = _finite_float(row["mw_pred"], field="mw_pred")
    events: dict[str, float] = {}
    for row in _read_csv(event_path):
        event = str(row["event"])
        if event in events:
            raise ValueError(f"duplicate locked endpoint event: {event}")
        events[event] = _finite_float(row["mw_pred_median"], field="mw_pred_median")
    if len(stations) != EXPECTED_STATION_COUNT or len(events) != EXPECTED_EVENT_COUNT:
        raise ValueError("locked Phase39 endpoint cohort size changed")
    return {
        "station_predictions": stations,
        "event_predictions": events,
        "event_mae": event_mae,
    }


def validate_fixed_cohort(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_station_keys: set[tuple[str, str]],
) -> None:
    keys = [(str(row["event"]), str(row["station"])) for row in records]
    if len(set(keys)) != len(keys):
        raise ValueError("reconstructed external cohort contains duplicate stations")
    actual = set(keys)
    if actual != expected_station_keys:
        missing = sorted(expected_station_keys - actual)
        extra = sorted(actual - expected_station_keys)
        raise ValueError(
            "reconstructed external cohort differs from locked endpoint; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )


def build_external_station_records(
    *,
    config: dict[str, Any],
    event_root: Path,
    labels_by_dir: Mapping[str, Mapping[str, Any]],
    expected_station_keys: set[tuple[str, str]],
    event_names: Sequence[str] = EXTERNAL_EVENT_NAMES,
) -> list[dict[str, Any]]:
    waveform_config = waveform_config_from_v2(config)
    expected_steps = int(
        round(
            float(config["dataset"]["sample_rate_hz"])
            * float(config["dataset"]["waveform"]["duration_sec"])
        )
    )
    source_dt_sec = 1.0 / float(config["dataset"]["sample_rate_hz"])
    records: list[dict[str, Any]] = []
    for event_dir_name in event_names:
        event_dir = event_root / event_dir_name
        bundle = load_event_bundle(
            event_dir,
            waveform_available_before_sec=200.0,
        )
        label = labels_by_dir[event_dir_name]
        event = _format_event_display_name(
            event_name=bundle.event_name,
            event_dir_name=bundle.event_dir_name,
            magnitude=bundle.magnitude,
        )
        if event != str(label["event"]):
            raise ValueError(
                f"external display label changed for {event_dir_name}: "
                f"{event!r} != {label['event']!r}"
            )
        for station in bundle.stations:
            sample = _station_sample_from_bundle(
                bundle,
                station,
                config,
                waveform_config=waveform_config,
                radial_peak_min_cm_override=0.0,
            )
            if sample is None:
                continue
            radial = np.asarray(sample["radial"], dtype=np.float32)
            valid_mask = np.asarray(sample["waveform_valid_mask"], dtype=bool)
            if radial.shape != (expected_steps,) or valid_mask.shape != radial.shape:
                raise ValueError(f"unexpected processed waveform shape for {event}/{station.station}")
            if np.any(radial[~valid_mask] != 0.0):
                raise ValueError(f"invalid waveform slots are nonzero for {event}/{station.station}")
            records.append(
                {
                    "event": event,
                    "event_dir": event_dir_name,
                    "station": station.station,
                    "radial": radial,
                    "waveform_valid_mask": valid_mask,
                    "waveform_dt_sec": float(sample["waveform_dt_sec"]),
                    "raw_dt_sec": float(sample["raw_dt_sec"]),
                    "stf_dt_sec": source_dt_sec,
                    "source_distance_m": float(sample["source_distance_m"]),
                    "epicentral_distance_m": float(sample["epicentral_distance_m"]),
                    "theta_deg": float(sample["theta_deg"]),
                    "azimuth_deg": float(sample["azimuth_deg"]),
                    "magnitude_catalog": float(label["mw_selected"]),
                    "magnitude_source": str(label["mw_source"]),
                    "usgs_event_id": str(label["usgs_event_id"]),
                    "baseline_source": str(sample["baseline_source"]),
                    "waveform_start_sec": float(sample["waveform_start_sec"]),
                    "waveform_phase_adjusted": bool(sample["waveform_phase_adjusted"]),
                    "radial_peak_cm": float(sample["radial_peak_cm"]),
                }
            )
    records.sort(key=lambda row: (str(row["event"]), str(row["station"])))
    validate_fixed_cohort(records, expected_station_keys=expected_station_keys)
    return records


def records_to_batches(
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> Iterable[dict[str, Any]]:
    if isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    for start in range(0, len(records), batch_size):
        rows = records[start : start + batch_size]
        yield {
            "event": [str(row["event"]) for row in rows],
            "station": [str(row["station"]) for row in rows],
            "radial": torch.from_numpy(
                np.stack([np.asarray(row["radial"], dtype=np.float32) for row in rows])
            ).unsqueeze(1),
            "waveform_valid_mask": torch.from_numpy(
                np.stack(
                    [np.asarray(row["waveform_valid_mask"], dtype=bool) for row in rows]
                )
            ),
            "waveform_dt_sec": [float(row["waveform_dt_sec"]) for row in rows],
            "raw_dt_sec": [float(row["raw_dt_sec"]) for row in rows],
            "stf_dt_sec": [float(row["stf_dt_sec"]) for row in rows],
            "source_distance_m": [float(row["source_distance_m"]) for row in rows],
            "epicentral_distance_m": [
                float(row["epicentral_distance_m"]) for row in rows
            ],
            "theta_deg": [float(row["theta_deg"]) for row in rows],
            "azimuth_deg": [float(row["azimuth_deg"]) for row in rows],
            "magnitude_catalog": [float(row["magnitude_catalog"]) for row in rows],
            "baseline_source": [str(row["baseline_source"]) for row in rows],
        }


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def load_model(config: dict[str, Any], *, device: torch.device) -> PINNModel:
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model = PINNModel(config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def validate_endpoint_reproduction(
    result: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    tolerance_mw: float = ENDPOINT_TOLERANCE_MW,
) -> dict[str, float]:
    actual_stations = {
        (str(row["event"]), str(row["station"])): float(row["mw_pred"])
        for row in result["station_rows"]
        if int(row["observation_horizon_sec"]) == 200
    }
    expected_stations = dict(reference["station_predictions"])
    if set(actual_stations) != set(expected_stations):
        raise ValueError("h=200 station keys differ from the locked endpoint")
    station_diffs = {
        key: abs(actual_stations[key] - float(expected_stations[key]))
        for key in actual_stations
    }
    max_station_diff = max(station_diffs.values(), default=0.0)
    if max_station_diff > tolerance_mw:
        worst = max(station_diffs, key=station_diffs.get)
        raise ValueError(
            f"h=200 station prediction mismatch at {worst}: "
            f"{max_station_diff:.9g} Mw > {tolerance_mw:.9g} Mw"
        )

    actual_events = {
        str(row["event"]): float(row["mw_pred_median"])
        for row in result["event_rows"]
        if int(row["observation_horizon_sec"]) == 200
    }
    expected_events = dict(reference["event_predictions"])
    if set(actual_events) != set(expected_events):
        raise ValueError("h=200 event keys differ from the locked endpoint")
    event_diffs = {
        event: abs(actual_events[event] - float(expected_events[event]))
        for event in actual_events
    }
    max_event_diff = max(event_diffs.values(), default=0.0)
    if max_event_diff > tolerance_mw:
        worst = max(event_diffs, key=event_diffs.get)
        raise ValueError(
            f"h=200 event median mismatch at {worst}: "
            f"{max_event_diff:.9g} Mw > {tolerance_mw:.9g} Mw"
        )

    final_rows = [
        row
        for row in result["event_rows"]
        if int(row["observation_horizon_sec"]) == 200
    ]
    event_mae = float(np.mean([abs(float(row["error"])) for row in final_rows]))
    event_mae_diff = abs(event_mae - float(reference["event_mae"]))
    if event_mae_diff > tolerance_mw:
        raise ValueError(
            f"h=200 Event MAE mismatch: {event_mae_diff:.9g} Mw > "
            f"{tolerance_mw:.9g} Mw"
        )
    return {
        "tolerance_mw": tolerance_mw,
        "max_station_prediction_abs_diff_mw": max_station_diff,
        "max_event_median_abs_diff_mw": max_event_diff,
        "event_mae": event_mae,
        "event_mae_abs_diff_mw": event_mae_diff,
    }


def _suffix_stable_horizon(
    values_by_horizon: Mapping[int, bool],
    horizons: Sequence[int],
) -> int | None:
    for index, horizon in enumerate(horizons):
        if all(values_by_horizon.get(later, False) for later in horizons[index:]):
            return int(horizon)
    return None


def analyze_convergence(
    result: Mapping[str, Any],
    *,
    horizons: Sequence[int],
    target_error_mw: float = TARGET_ERROR_MW,
) -> dict[str, Any]:
    normalized_horizons = tuple(int(value) for value in horizons)
    event_rows: list[dict[str, Any]] = []
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_horizon: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source in result["event_rows"]:
        row = dict(source)
        horizon = int(row["observation_horizon_sec"])
        row["abs_error"] = abs(float(row["error"]))
        row["within_target"] = bool(row["abs_error"] <= target_error_mw)
        event_rows.append(row)
        by_event[str(row["event"])].append(row)
        by_horizon[horizon].append(row)
    event_names = sorted(by_event)

    convergence_rows: list[dict[str, Any]] = []
    for event in event_names:
        sequence = sorted(
            by_event[event], key=lambda row: int(row["observation_horizon_sec"])
        )
        lookup = {int(row["observation_horizon_sec"]): row for row in sequence}
        within = {
            horizon: bool(lookup[horizon]["within_target"])
            for horizon in lookup
        }
        first_within = next(
            (horizon for horizon in normalized_horizons if within.get(horizon, False)),
            None,
        )
        stable = _suffix_stable_horizon(within, normalized_horizons)
        final = lookup[normalized_horizons[-1]]
        convergence_rows.append(
            {
                "event": event,
                "mw_catalog": float(final["mw_catalog"]),
                "final_mw_pred_median": float(final["mw_pred_median"]),
                "final_error": float(final["error"]),
                "final_abs_error": float(final["abs_error"]),
                "final_station_count": int(final["station_count"]),
                "first_available_observation_sec": int(
                    min(int(row["observation_horizon_sec"]) for row in sequence)
                ),
                "first_within_target_observation_sec": first_within,
                "first_within_target_release_sec": (
                    None if first_within is None else first_within + PROCESSING_DELAY_SEC
                ),
                "stable_within_target_observation_sec": stable,
                "stable_within_target_release_sec": (
                    None if stable is None else stable + PROCESSING_DELAY_SEC
                ),
                "right_censored_at_200_sec": stable is None,
            }
        )

    source_metrics = {
        int(row["observation_horizon_sec"]): dict(row)
        for row in result["horizon_metrics"]
    }
    horizon_metrics: list[dict[str, Any]] = []
    for horizon in normalized_horizons:
        row = source_metrics[horizon]
        current_events = by_horizon[horizon]
        row["within_target_event_count"] = sum(
            bool(event["within_target"]) for event in current_events
        )
        row["within_target_event_fraction"] = (
            row["within_target_event_count"] / len(current_events)
            if current_events
            else 0.0
        )
        horizon_metrics.append(row)

    event_count = len(event_names)
    overall_within = {
        int(row["observation_horizon_sec"]): (
            int(row["event_count"]) == event_count
            and float(row["event_equal_mae"]) <= target_error_mw
        )
        for row in horizon_metrics
    }
    overall_first = next(
        (horizon for horizon in normalized_horizons if overall_within[horizon]),
        None,
    )
    overall_stable = _suffix_stable_horizon(overall_within, normalized_horizons)

    final_metric = horizon_metrics[-1]
    final_station_count = int(final_metric["available_station_count"])
    full_station = {
        int(row["observation_horizon_sec"]): (
            int(row["available_station_count"]) == final_station_count
        )
        for row in horizon_metrics
    }
    first_full_station = _suffix_stable_horizon(full_station, normalized_horizons)
    return {
        "complete_second_grid": normalized_horizons == FULL_HORIZONS,
        "target_error_mw": target_error_mw,
        "event_rows": event_rows,
        "horizon_metrics": horizon_metrics,
        "event_convergence": sorted(
            convergence_rows,
            key=lambda row: (float(row["mw_catalog"]), str(row["event"])),
        ),
        "overall": {
            "event_count": event_count,
            "final_observation_sec": normalized_horizons[-1],
            "final_release_sec": normalized_horizons[-1] + PROCESSING_DELAY_SEC,
            "final_event_mae": float(final_metric["event_equal_mae"]),
            "final_event_rmse": float(final_metric["event_equal_rmse"]),
            "final_event_bias": float(final_metric["event_equal_bias"]),
            "final_within_target_event_count": int(
                final_metric["within_target_event_count"]
            ),
            "final_station_count": final_station_count,
            "first_event_mae_within_target_observation_sec": overall_first,
            "first_event_mae_within_target_release_sec": (
                None if overall_first is None else overall_first + PROCESSING_DELAY_SEC
            ),
            "stable_event_mae_within_target_observation_sec": overall_stable,
            "stable_event_mae_within_target_release_sec": (
                None if overall_stable is None else overall_stable + PROCESSING_DELAY_SEC
            ),
            "stable_individual_event_count": sum(
                row["stable_within_target_observation_sec"] is not None
                for row in convergence_rows
            ),
            "first_full_station_availability_observation_sec": first_full_station,
            "first_full_station_availability_release_sec": (
                None
                if first_full_station is None
                else first_full_station + PROCESSING_DELAY_SEC
            ),
        },
    }


def hash_external_inputs(
    event_root: Path,
    *,
    event_names: Sequence[str] = EXTERNAL_EVENT_NAMES,
) -> dict[str, Any]:
    file_hashes: dict[str, str] = {}
    for event_name in event_names:
        for filename in ("event.json", "stations.csv", "waveforms.csv.gz"):
            path = event_root / event_name / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing external input: {path}")
            relative = str(path.relative_to(event_root))
            file_hashes[relative] = sha256_file(path)
    aggregate = hashlib.sha256()
    for relative, digest in sorted(file_hashes.items()):
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "aggregate_sha256": aggregate.hexdigest(),
        "files": file_hashes,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for source in rows:
            row = {
                key: "" if value is None else value
                for key, value in source.items()
            }
            if isinstance(row.get("unavailable_reason_counts"), dict):
                row["unavailable_reason_counts"] = json.dumps(
                    row["unavailable_reason_counts"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            writer.writerow(row)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def build_cohort_contract(
    records: Sequence[Mapping[str, Any]],
    result_cohort: Mapping[str, Any],
    *,
    horizons: Sequence[int],
) -> dict[str, Any]:
    event_counts = Counter(str(row["event"]) for row in records)
    baseline_counts = Counter(str(row["baseline_source"]) for row in records)
    phase_adjusted = sum(bool(row["waveform_phase_adjusted"]) for row in records)
    return {
        **dict(result_cohort),
        "external_role": "development_validation",
        "event_count": len(event_counts),
        "station_count": len(records),
        "event_station_counts": dict(sorted(event_counts.items())),
        "baseline_source_counts": dict(sorted(baseline_counts.items())),
        "phase_adjusted_station_count": phase_adjusted,
        "zero_phase_station_count": len(records) - phase_adjusted,
        "observation_horizons_sec": [int(value) for value in horizons],
        "release_time_rule": "observation_horizon_sec + 5 s",
        "station_selection": (
            "fixed cm0 cohort reconstructed from the complete 200 s external records"
        ),
        "interpretation": (
            "processed-waveform prefix sensitivity diagnostic for a fixed noncausal "
            "architecture; not end-to-end causal inference"
        ),
    }


def _configure_chinese_plot_style() -> None:
    font_candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/arphic/ukai.ttc"),
        Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
    )
    family = "DejaVu Sans"
    for path in font_candidates:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            break
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _style_axis(axis: Any) -> None:
    axis.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#C7C7C7")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save_figure(fig: Any, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, pdf]


def plot_overall_metrics(
    analysis: Mapping[str, Any],
    *,
    output_stem: Path,
) -> list[Path]:
    _configure_chinese_plot_style()
    rows = analysis["horizon_metrics"]
    horizon = np.asarray([float(row["observation_horizon_sec"]) for row in rows])
    mae = np.asarray([float(row["event_equal_mae"]) for row in rows])
    rmse = np.asarray([float(row["event_equal_rmse"]) for row in rows])
    bias = np.asarray([float(row["event_equal_bias"]) for row in rows])
    within = np.asarray([int(row["within_target_event_count"]) for row in rows])
    stations = np.asarray([int(row["available_station_count"]) for row in rows])
    stable = analysis["overall"]["stable_event_mae_within_target_observation_sec"]

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.2), sharex=True)
    ax_error, ax_bias, ax_coverage = axes
    for axis in axes:
        _style_axis(axis)
        if stable is not None:
            axis.axvline(float(stable), color="#009E73", linestyle=":", linewidth=1.1)

    ax_error.plot(horizon, mae, color="#0072B2", label="事件等权 MAE")
    ax_error.plot(horizon, rmse, color="#D55E00", label="事件等权 RMSE")
    ax_error.axhline(
        TARGET_ERROR_MW,
        color="#009E73",
        linestyle="--",
        linewidth=1.0,
        label="0.15 Mw 目标",
    )
    ax_error.set_ylabel("误差 (Mw)")
    ax_error.text(
        0.01,
        0.94,
        "A  逐秒事件误差",
        transform=ax_error.transAxes,
        va="top",
        fontsize=10,
    )
    ax_error.legend(loc="upper right", ncol=3)
    if stable is not None:
        ax_error.annotate(
            f"持续达标: {int(stable)} s\n发布: {int(stable + PROCESSING_DELAY_SEC)} s",
            xy=(float(stable), TARGET_ERROR_MW),
            xytext=(8, 18),
            textcoords="offset points",
            color="#006B4F",
            arrowprops={"arrowstyle": "->", "color": "#006B4F", "lw": 0.8},
        )

    ax_bias.plot(horizon, bias, color="#CC79A7", label="事件等权偏差")
    ax_bias.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
    ax_bias.set_ylabel("偏差 (Mw)")
    ax_bias.text(
        0.01,
        0.94,
        "B  有符号偏差",
        transform=ax_bias.transAxes,
        va="top",
        fontsize=10,
    )
    ax_bias.legend(loc="best")

    ax_coverage.step(
        horizon,
        within,
        where="mid",
        color="#009E73",
        label="|误差| ≤ 0.15 的事件数",
    )
    ax_coverage.set_ylabel("事件数")
    ax_coverage.set_ylim(-0.25, EXPECTED_EVENT_COUNT + 0.5)
    ax_coverage.set_yticks(range(0, EXPECTED_EVENT_COUNT + 1, 2))
    ax_coverage.text(
        0.01,
        0.94,
        "C  事件达标数与可用台站数",
        transform=ax_coverage.transAxes,
        va="top",
        fontsize=10,
    )
    station_axis = ax_coverage.twinx()
    station_axis.plot(
        horizon,
        stations,
        color="#6B6B6B",
        linestyle="--",
        label="可用台站数",
    )
    station_axis.set_ylabel("台站数")
    station_axis.set_ylim(0, EXPECTED_STATION_COUNT + 10)
    station_axis.spines["top"].set_visible(False)
    handles_a, labels_a = ax_coverage.get_legend_handles_labels()
    handles_b, labels_b = station_axis.get_legend_handles_labels()
    ax_coverage.legend(handles_a + handles_b, labels_a + labels_b, loc="lower right")
    ax_coverage.set_xlabel("震源时刻后的观测时长 (s)")
    ax_coverage.set_xlim(1, 200)
    ax_coverage.set_xticks((1, 25, 50, 75, 100, 125, 150, 175, 200))

    release_axis = ax_error.secondary_xaxis(
        "top",
        functions=(
            lambda value: value + PROCESSING_DELAY_SEC,
            lambda value: value - PROCESSING_DELAY_SEC,
        ),
    )
    release_axis.set_xlabel("可发布时间 (s)")
    release_axis.set_xticks((6, 30, 55, 80, 105, 130, 155, 180, 205))
    fig.suptitle("Phase39 固定模型逐秒前缀收敛", y=0.995, fontsize=13)
    fig.text(
        0.5,
        0.012,
        "8 个训练未见但已反复使用的开发事件；固定 cm0 cohort；事件内取台站预测中位数。",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.subplots_adjust(left=0.11, right=0.88, bottom=0.13, top=0.88, hspace=0.34)
    return _save_figure(fig, output_stem)


def plot_event_trajectories(
    analysis: Mapping[str, Any],
    *,
    output_stem: Path,
) -> list[Path]:
    _configure_chinese_plot_style()
    by_event: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in analysis["event_rows"]:
        by_event[str(row["event"])].append(row)
    convergence = {
        str(row["event"]): row for row in analysis["event_convergence"]
    }
    ordered = sorted(by_event, key=lambda event: (float(by_event[event][0]["mw_catalog"]), event))

    fig, axes = plt.subplots(4, 2, figsize=(8.0, 9.6), sharex=True)
    for axis, event in zip(axes.flat, ordered):
        _style_axis(axis)
        sequence = sorted(
            by_event[event], key=lambda row: float(row["observation_horizon_sec"])
        )
        horizon = np.asarray([float(row["observation_horizon_sec"]) for row in sequence])
        prediction = np.asarray([float(row["mw_pred_median"]) for row in sequence])
        catalog = float(sequence[0]["mw_catalog"])
        axis.axhspan(
            catalog - TARGET_ERROR_MW,
            catalog + TARGET_ERROR_MW,
            color="#009E73",
            alpha=0.12,
            linewidth=0,
        )
        axis.axhline(catalog, color="#333333", linestyle="--", linewidth=0.9)
        axis.plot(horizon, prediction, color="#0072B2", linewidth=1.35)
        stable = convergence[event]["stable_within_target_observation_sec"]
        if stable is not None:
            stable_row = next(
                row for row in sequence if int(row["observation_horizon_sec"]) == int(stable)
            )
            axis.scatter(
                [float(stable)],
                [float(stable_row["mw_pred_median"])],
                color="#D55E00",
                marker="D",
                s=24,
                zorder=4,
            )
            status = f"持续达标 {int(stable)} s / 发布 {int(stable + 5)} s"
        else:
            status = "截至 200 s 未持续进入 ±0.15"
        axis.text(
            0.02,
            0.05,
            status,
            transform=axis.transAxes,
            fontsize=7.5,
            va="bottom",
            color="#444444",
        )
        low = min(float(prediction.min()), catalog - TARGET_ERROR_MW)
        high = max(float(prediction.max()), catalog + TARGET_ERROR_MW)
        margin = max(0.12, 0.06 * (high - low))
        axis.set_ylim(low - margin, high + margin)
        axis.set_title(
            f"{event}  |  参考 {catalog:.1f}  |  200 s 预测 {prediction[-1]:.3f}",
            fontsize=9,
        )
        axis.set_xlim(1, 200)
        axis.set_xticks((1, 50, 100, 150, 200))

    fig.supxlabel("震源时刻后的观测时长 (s)", y=0.035)
    fig.supylabel("事件台站中位数 Mw", x=0.025)
    fig.suptitle("Phase39 八个外部开发事件的逐秒预测轨迹", y=0.985, fontsize=13)
    fig.legend(
        handles=[
            Line2D([0], [0], color="#0072B2", label="Phase39 事件中位数预测"),
            Line2D([0], [0], color="#333333", linestyle="--", label="USGS 参考 Mw"),
            Line2D(
                [0],
                [0],
                color="#D55E00",
                marker="D",
                linestyle="none",
                label="从该秒起一直保持在误差带内",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.075, top=0.90, hspace=0.42, wspace=0.20)
    return _save_figure(fig, output_stem)


def plot_convergence_times(
    analysis: Mapping[str, Any],
    *,
    output_stem: Path,
) -> list[Path]:
    _configure_chinese_plot_style()
    rows = sorted(
        analysis["event_convergence"],
        key=lambda row: (-float(row["mw_catalog"]), str(row["event"])),
    )
    y = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    _style_axis(axis)
    axis.grid(True, axis="x", linestyle=":", linewidth=0.5, color="#C7C7C7")
    axis.grid(False, axis="y")
    for index, row in enumerate(rows):
        first = row["first_within_target_observation_sec"]
        stable = row["stable_within_target_observation_sec"]
        if first is not None:
            axis.scatter(first, index, color="#6B6B6B", marker="o", s=28, zorder=3)
        if stable is not None:
            if first is not None:
                axis.plot([first, stable], [index, index], color="#A7A7A7", linewidth=1.0)
            axis.scatter(stable, index, color="#0072B2", marker="D", s=34, zorder=4)
            axis.text(stable + 3, index, f"{int(stable)} s", va="center", fontsize=8)
        else:
            axis.scatter(200, index, facecolors="none", edgecolors="#D55E00", marker="o", s=42, linewidth=1.3)
            axis.text(196, index, ">200 s", ha="right", va="center", fontsize=8, color="#A83D00")
    axis.set_yticks(
        y,
        [f"{row['event']}  (Mw {float(row['mw_catalog']):.1f})" for row in rows],
    )
    axis.invert_yaxis()
    axis.set_xlim(0, 218)
    axis.set_xticks((0, 25, 50, 75, 100, 125, 150, 175, 200))
    axis.set_xlabel("观测时长 (s)")
    axis.set_title("首次进入与持续收敛时间")
    release_axis = axis.secondary_xaxis(
        "top",
        functions=(
            lambda value: value + PROCESSING_DELAY_SEC,
            lambda value: value - PROCESSING_DELAY_SEC,
        ),
    )
    release_axis.set_xlabel("可发布时间 (s)")
    release_axis.set_xticks((5, 30, 55, 80, 105, 130, 155, 180, 205))
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#6B6B6B", linestyle="none", label="首次进入 ±0.15"),
            Line2D([0], [0], marker="D", color="#0072B2", linestyle="none", label="此后持续保持到 200 s"),
            Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="#D55E00", color="none", label="截至 200 s 未持续收敛"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=3,
    )
    fig.text(
        0.5,
        0.015,
        "判据固定为 |事件中位数预测 − USGS Mw| ≤ 0.15；发布时间 = 观测时长 + 5 s。",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.subplots_adjust(left=0.29, right=0.96, bottom=0.29, top=0.80)
    return _save_figure(fig, output_stem)


def _format_optional_seconds(value: Any, *, release: bool = False) -> str:
    if value is None:
        return ">200 s" if not release else ">205 s"
    return f"{int(float(value))} s"


def build_chinese_report(
    *,
    analysis: Mapping[str, Any],
    endpoint_gate: Mapping[str, Any],
    cohort: Mapping[str, Any],
) -> str:
    overall = analysis["overall"]
    stable = overall["stable_event_mae_within_target_observation_sec"]
    stable_release = overall["stable_event_mae_within_target_release_sec"]
    if stable is None:
        headline = "总体事件等权 MAE 截至 200 秒尚未持续低于 0.15 Mw。"
    else:
        headline = (
            f"总体事件等权 MAE 从 **{int(stable)} 秒观测时长**起持续不高于 "
            f"0.15 Mw，对应最早可发布时间 **{int(stable_release)} 秒**。"
        )
    lines = [
        "# Phase39 逐秒波形前缀收敛报告",
        "",
        "> 固定 Phase39 Glehman scalar + global invariant、seed42 checkpoint；不训练、不调网络、不换 seed、不调整 0.15 Mw 判据。",
        "",
        "## 结论",
        "",
        headline,
        "",
        f"在完整 200 秒时，事件等权 MAE 为 **{float(overall['final_event_mae']):.6f} Mw**，"
        f"与既有 Phase39 外部结果一致；八个事件中有 **{int(overall['final_within_target_event_count'])}/8** "
        "落在 ±0.15 Mw 误差带内。按“进入后每秒一直保持到 200 秒”的严格定义，"
        f"共有 **{int(overall['stable_individual_event_count'])}/8** 个事件在窗口内完成个体收敛。",
        "",
        f"200 秒端点复现门槛通过：逐站最大差异为 "
        f"`{float(endpoint_gate['max_station_prediction_abs_diff_mw']):.3g} Mw`，"
        f"事件中位数最大差异为 `{float(endpoint_gate['max_event_median_abs_diff_mw']):.3g} Mw`。",
        "",
        "![总体逐秒指标](figures/01_overall_metrics.png)",
        "",
        "[PDF 图件](figures/01_overall_metrics.pdf)",
        "",
        "## 八个事件轨迹",
        "",
        "![事件逐秒轨迹](figures/02_event_trajectories.png)",
        "",
        "[PDF 图件](figures/02_event_trajectories.pdf)",
        "",
        "![事件收敛时间](figures/03_convergence_times.png)",
        "",
        "[PDF 图件](figures/03_convergence_times.pdf)",
        "",
        "## 事件级结果",
        "",
        "| 事件 | 参考 Mw | 200 s 预测 | 200 s 绝对误差 | 首次进入 | 持续收敛观测时长 | 可发布时间 | 台站数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["event_convergence"]:
        lines.append(
            f"| {row['event']} | {float(row['mw_catalog']):.1f} | "
            f"{float(row['final_mw_pred_median']):.3f} | "
            f"{float(row['final_abs_error']):.3f} | "
            f"{_format_optional_seconds(row['first_within_target_observation_sec'])} | "
            f"{_format_optional_seconds(row['stable_within_target_observation_sec'])} | "
            f"{_format_optional_seconds(row['stable_within_target_release_sec'], release=True)} | "
            f"{int(row['final_station_count'])} |"
        )
    lines.extend(
        [
            "",
            "## 评估口径",
            "",
            "- 输入仍是单台站 R 分量。每个整数秒 `h=1..200` 只保留完整预处理张量的前 `h` 个 1 Hz 槽位，后续值和有效掩码全部清零。",
            "- 7 点居中 Hamming FIR 需要最多 3 秒未来支撑，因此结果同时报告 `h` 秒观测时长和 `h+5` 秒发布时间。",
            "- 每个台站独立输出 Mw；同一事件对可用台站预测取中位数。参考值来自冻结 USGS 快照的 `mw_selected`。",
            "- 收敛定义为首次满足 `|Event error| <= 0.15 Mw`，并且此后每个整数秒都保持到 200 秒。",
            f"- 固定 cohort 为 {int(cohort['event_count'])} 个事件、{int(cohort['station_count'])} 个台站；完整台站可用性从 "
            f"{_format_optional_seconds(overall['first_full_station_availability_observation_sec'])} 起持续保持。",
            "",
            "## 必须保留的边界",
            "",
            "这是一项固定 checkpoint 的波形前缀敏感性诊断，不是严格因果网络验证。Phase39 的 TCN 使用对称 padding，Transformer 没有 causal mask；固定台站 cohort 也来自完整 200 秒记录的离线筛选。五秒延迟只覆盖预处理 FIR 的未来支撑，不会改变网络内部结构。",
            "",
            "这八个事件确实没有进入模型训练，但它们已被多轮方案比较反复使用，因此研究角色仍是 `development_validation`。本报告可以回答“固定 Phase39 在这批开发事件上需要多少秒达到稳定误差水平”，不能单独把结果升级为无偏的最终未见事件泛化证据。",
            "",
            "## 工件",
            "",
            "- `summary.json`：总体结论、端点复现和事件收敛时间。",
            "- `event_predictions.csv`：8 个事件 × 200 个观测秒的预测轨迹。",
            "- `station_predictions.csv`：逐站逐秒预测。",
            "- `horizon_metrics.csv`：每秒 Event MAE、RMSE、bias、覆盖率和达标事件数。",
            "- `cohort_contract.json` 与 `provenance.json`：固定 cohort 和输入哈希。",
            "",
        ]
    )
    return "\n".join(lines)


def validate_output_dir(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output directory must be new or empty: {path}")


def write_outputs(
    *,
    output_dir: Path,
    result: Mapping[str, Any],
    analysis: Mapping[str, Any],
    endpoint_gate: Mapping[str, Any],
    cohort: Mapping[str, Any],
    provenance: Mapping[str, Any],
    create_report: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "station_predictions.csv",
        result["station_rows"],
        fieldnames=(
            "event",
            "station",
            "observation_horizon_sec",
            "release_time_sec",
            "mw_pred",
            "mw_catalog",
            "error",
        ),
    )
    _write_csv(
        output_dir / "unavailable_stations.csv",
        result["unavailable_rows"],
        fieldnames=(
            "event",
            "station",
            "observation_horizon_sec",
            "release_time_sec",
            "reason",
            "baseline_ready_time_sec",
        ),
    )
    _write_csv(
        output_dir / "event_predictions.csv",
        analysis["event_rows"],
        fieldnames=(
            "event",
            "observation_horizon_sec",
            "release_time_sec",
            "mw_pred_median",
            "mw_catalog",
            "error",
            "abs_error",
            "within_target",
            "station_count",
        ),
    )
    _write_csv(
        output_dir / "horizon_metrics.csv",
        analysis["horizon_metrics"],
        fieldnames=(
            "observation_horizon_sec",
            "release_time_sec",
            "event_count",
            "total_event_count",
            "coverage",
            "coverage_fraction",
            "available_station_count",
            "unavailable_station_count",
            "unavailable_reason_counts",
            "event_equal_mae",
            "event_equal_rmse",
            "event_equal_bias",
            "within_target_event_count",
            "within_target_event_fraction",
        ),
    )
    _write_csv(
        output_dir / "event_convergence.csv",
        analysis["event_convergence"],
        fieldnames=(
            "event",
            "mw_catalog",
            "final_mw_pred_median",
            "final_error",
            "final_abs_error",
            "final_station_count",
            "first_available_observation_sec",
            "first_within_target_observation_sec",
            "first_within_target_release_sec",
            "stable_within_target_observation_sec",
            "stable_within_target_release_sec",
            "right_censored_at_200_sec",
        ),
    )
    _write_json(output_dir / "horizon_metrics.json", analysis["horizon_metrics"])
    _write_json(output_dir / "cohort_contract.json", cohort)
    _write_json(output_dir / "provenance.json", provenance)
    summary = {
        "status": "complete" if analysis["complete_second_grid"] else "smoke_complete",
        "model": "Phase39 Glehman scalar + global invariant, seed42",
        "evaluation_role": "development_validation",
        "target_error_mw": TARGET_ERROR_MW,
        "convergence_rule": (
            "earliest integer observation horizon inside the target band whose "
            "entire suffix through 200 s remains inside"
        ),
        "endpoint_reproduction": dict(endpoint_gate),
        "overall": analysis["overall"],
        "event_convergence": analysis["event_convergence"],
    }
    _write_json(output_dir / "summary.json", summary)

    if create_report:
        figures = output_dir / "figures"
        plot_overall_metrics(analysis, output_stem=figures / "01_overall_metrics")
        plot_event_trajectories(
            analysis,
            output_stem=figures / "02_event_trajectories",
        )
        plot_convergence_times(
            analysis,
            output_stem=figures / "03_convergence_times",
        )
        (output_dir / "REPORT_ZH.md").write_text(
            build_chinese_report(
                analysis=analysis,
                endpoint_gate=endpoint_gate,
                cohort=cohort,
            ),
            encoding="utf-8",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen Phase39 seed42 checkpoint at every one-second "
            "processed-waveform prefix and report suffix-stable convergence."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New or empty directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help=(
            "Inference device (default: cpu, matching the locked Phase39 "
            "external endpoint)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EXPECTED_STATION_COUNT,
        help="Station inference batch size (default: all 158 stations).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Evaluate only 20, 100, and 200 s; skip figures and report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    validate_output_dir(output_dir)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    horizons = SMOKE_HORIZONS if args.smoke else FULL_HORIZONS

    print("[1/7] validating frozen artifacts and Phase39 config", flush=True)
    artifact_hashes = validate_frozen_artifacts()
    config = load_phase39_config()
    labels = load_label_contract(LABELS_PATH)
    endpoint_reference = load_endpoint_reference()

    print("[2/7] hashing external event inputs", flush=True)
    external_hashes = hash_external_inputs(EXTERNAL_EVENT_ROOT)

    print("[3/7] reconstructing the locked 8-event / 158-station cohort", flush=True)
    records = build_external_station_records(
        config=config,
        event_root=EXTERNAL_EVENT_ROOT,
        labels_by_dir=labels,
        expected_station_keys=set(endpoint_reference["station_predictions"]),
    )
    eval_config = copy.deepcopy(config)
    eval_config["dataset"]["radial_peak_min_cm"] = 0.0
    batches = list(records_to_batches(records, batch_size=args.batch_size))

    device = _resolve_device(args.device)
    print(
        f"[4/7] loading Phase39 on {device} and evaluating {len(horizons)} horizons",
        flush=True,
    )
    model = load_model(eval_config, device=device)
    result = evaluate_delayed_prefix(
        model,
        eval_config,
        batches,
        horizons=horizons,
    )

    print("[5/7] enforcing the h=200 station/event endpoint gate", flush=True)
    endpoint_gate = validate_endpoint_reproduction(result, endpoint_reference)
    analysis = analyze_convergence(result, horizons=horizons)

    cohort = build_cohort_contract(
        records,
        result["cohort"],
        horizons=horizons,
    )
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "horizons_sec": list(horizons),
        "processing_delay_sec": PROCESSING_DELAY_SEC,
        "artifact_paths": {name: str(path) for name, path in FROZEN_PATHS.items()},
        "artifact_sha256": artifact_hashes,
        "external_event_root": str(EXTERNAL_EVENT_ROOT),
        "external_input_hashes": external_hashes,
        "checkpoint_selection": (
            "seed42 selected by the existing Phase39 validation stage; no selection "
            "or tuning was performed by this evaluation"
        ),
    }

    print("[6/7] writing tables and provenance", flush=True)
    write_outputs(
        output_dir=output_dir,
        result=result,
        analysis=analysis,
        endpoint_gate=endpoint_gate,
        cohort=cohort,
        provenance=provenance,
        create_report=not args.smoke,
    )
    print("[7/7] complete", flush=True)
    overall = analysis["overall"]
    print(f"output_dir={output_dir}")
    print(f"endpoint_event_mae={float(endpoint_gate['event_mae']):.12f}")
    if not args.smoke:
        stable = overall["stable_event_mae_within_target_observation_sec"]
        release = overall["stable_event_mae_within_target_release_sec"]
        print(f"stable_event_mae_observation_sec={stable}")
        print(f"stable_event_mae_release_sec={release}")
        print(
            "stable_individual_events="
            f"{overall['stable_individual_event_count']}/{overall['event_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
