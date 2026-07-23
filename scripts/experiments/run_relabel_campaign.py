from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.experiments.run_corrected_matrix as corrected_matrix  # noqa: E402
from scripts.evaluation.recompute_relabel_metrics import (  # noqa: E402
    pair_prediction_rows,
    summarize_paired_rows,
)
from src.data.dataset_v2 import CorrectedEarthquakeDataset  # noqa: E402
from src.data.loaders_v2 import (  # noqa: E402
    _runtime_config as resolve_data_paths,
    get_data_loaders_v2,
)
from src.data.manifest import audit_passes, write_dataset_audit  # noqa: E402
from src.evaluation.evaluate_unseen import evaluate_unseen_events  # noqa: E402
from src.utils.config_v2 import validate_config_v2  # noqa: E402
from src.utils.provenance import (  # noqa: E402
    current_git_commit,
    git_is_dirty,
    sha256_file,
    utc_now_iso,
)


SEEDS = (17, 42, 73)
EXPECTED_ACCEPTED_EVENTS = 31
EXPECTED_ACCEPTED_STATIONS = 2483


@dataclass(frozen=True)
class PilotGate:
    passed: bool
    frozen_selected_event_mae: float
    pilot_selected_event_mae: float
    degradation: float
    max_degradation: float
    stop_threshold: float


def prepare_relabel_config(
    frozen_config: dict[str, Any],
    candidate_npz: str | Path,
) -> dict[str, Any]:
    result = copy.deepcopy(frozen_config)
    result["paths"]["data_path"] = str(Path(candidate_npz).resolve())
    return result


def evaluate_pilot_gate(
    *,
    frozen_selected_event_mae: float,
    pilot_selected_event_mae: float,
    max_degradation: float = 0.05,
) -> PilotGate:
    baseline = float(frozen_selected_event_mae)
    pilot = float(pilot_selected_event_mae)
    tolerance = float(max_degradation)
    if not all(math.isfinite(value) for value in (baseline, pilot, tolerance)):
        raise ValueError("pilot gate values must be finite")
    if tolerance < 0.0:
        raise ValueError("max_degradation must be nonnegative")
    degradation = pilot - baseline
    threshold = baseline + tolerance
    return PilotGate(
        passed=not (pilot > threshold),
        frozen_selected_event_mae=baseline,
        pilot_selected_event_mae=pilot,
        degradation=degradation,
        max_degradation=tolerance,
        stop_threshold=threshold,
    )


def assert_split_assignment_matches(
    candidate_split: str | Path,
    frozen_split: str | Path,
) -> None:
    candidate = json.loads(Path(candidate_split).read_text(encoding="utf-8"))
    frozen = json.loads(Path(frozen_split).read_text(encoding="utf-8"))
    for key in (
        "assignment_sha256",
        "sample_keys",
        "per_event_station_counts",
        "protocol",
    ):
        if candidate.get(key) != frozen.get(key):
            raise ValueError(f"split assignment differs for {key}")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return payload


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _atomic_write(path: Path, payload: bytes, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite campaign artifact: {path}")
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


def _config_bytes(config: dict[str, Any]) -> bytes:
    return yaml.safe_dump(
        config,
        allow_unicode=True,
        sort_keys=False,
    ).encode("utf-8")


def _frozen_run_root(frozen_config_path: Path) -> Path:
    if frozen_config_path.parent.name != "preflight":
        raise ValueError("frozen config must be inside a preflight directory")
    return frozen_config_path.parent.parent


def _frozen_split_path(frozen_run: Path, seed: int) -> Path:
    matches = sorted(
        (frozen_run / "campaign").glob(
            f"*/seed_{seed}/models/*/split.json"
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected one frozen split for seed {seed}, found {len(matches)}"
        )
    return matches[0]


def _audit_candidate(
    config: dict[str, Any],
    *,
    preflight_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    dataset = CorrectedEarthquakeDataset(resolve_data_paths(config))
    accepted_events = {str(sample["event"]) for sample in dataset.samples}
    if len(dataset.samples) != EXPECTED_ACCEPTED_STATIONS:
        raise ValueError(
            f"candidate accepted station count is {len(dataset.samples)}, "
            f"expected {EXPECTED_ACCEPTED_STATIONS}"
        )
    if len(accepted_events) != EXPECTED_ACCEPTED_EVENTS:
        raise ValueError(
            f"candidate accepted event count is {len(accepted_events)}, "
            f"expected {EXPECTED_ACCEPTED_EVENTS}"
        )
    manifest_path = preflight_dir / "dataset_manifest.csv"
    summary_path = preflight_dir / "dataset_summary.json"
    summary = write_dataset_audit(
        dataset,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )
    if not audit_passes(summary):
        raise ValueError("candidate dataset audit failed")
    return manifest_path, summary


def _run_device_smokes(
    config: dict[str, Any],
    *,
    output_path: Path,
) -> dict[str, Any]:
    loaders = get_data_loaders_v2(config)
    devices = [torch.device("cpu")]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved training campaign")
    devices.append(torch.device("cuda"))
    original_get_device = corrected_matrix.get_preferred_device
    results: dict[str, Any] = {}
    try:
        for device in devices:
            corrected_matrix.get_preferred_device = lambda device=device: device
            corrected_matrix._assert_backpropagates(config, loaders[0])
            results[device.type] = {
                "passed": True,
                "device": str(device),
            }
    finally:
        corrected_matrix.get_preferred_device = original_get_device
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _atomic_write(output_path, _json_bytes(results))
    return results


def _comparison_baseline(path: Path) -> float:
    payload = _load_json(path)
    value = float(payload["formal"]["seed_42_selected_event_mae"])
    if not math.isfinite(value):
        raise ValueError("frozen comparison seed-42 baseline is non-finite")
    return value


def _validate_config_change(
    frozen: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    differences = corrected_matrix.config_diff_paths(frozen, candidate)
    if differences != {"paths.data_path"}:
        raise ValueError(
            "relabel config changes fields other than scalar dataset path: "
            f"{sorted(differences)}"
        )


def _validate_training_rows(
    rows: list[dict[str, Any]],
    *,
    required_seeds: tuple[int, ...],
    frozen_run: Path,
) -> None:
    successful = [row for row in rows if row.get("status") == "ok"]
    actual_seeds = tuple(sorted(int(row["seed"]) for row in successful))
    if actual_seeds != tuple(sorted(required_seeds)):
        raise ValueError(
            f"training seed set differs: expected={required_seeds}, "
            f"actual={actual_seeds}"
        )
    for row in successful:
        seed = int(row["seed"])
        assert_split_assignment_matches(
            row["split_manifest_path"],
            _frozen_split_path(frozen_run, seed),
        )
        for key in (
            "within_event_station_event_mae_catalog",
            "within_event_station_station_mae_catalog",
        ):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"training produced non-finite {key}")


def _external_event_dirs(event_root: Path) -> list[Path]:
    paths = [event_root / name for name in corrected_matrix.EXTERNAL_EVENT_NAMES]
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"external events are missing: {missing}")
    return paths


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _external_ensemble(
    event_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in event_rows:
        grouped.setdefault(str(row["event"]), []).append(row)
    ensemble: list[dict[str, Any]] = []
    for event in sorted(grouped):
        rows = grouped[event]
        if {int(row["seed"]) for row in rows} != set(SEEDS):
            raise ValueError(f"external ensemble seed set differs for {event}")
        selected = {float(row["mw_selected"]) for row in rows}
        old = {float(row["mw_old"]) for row in rows}
        if len(selected) != 1 or len(old) != 1:
            raise ValueError(f"external labels differ by seed for {event}")
        ensemble.append(
            {
                "event": event,
                "mw_pred_ensemble": float(
                    np.mean([float(row["mw_pred_median"]) for row in rows])
                ),
                "mw_selected": selected.pop(),
                "mw_old": old.pop(),
                "mw_source": rows[0]["mw_source"],
                "mw_source_rank": int(rows[0]["mw_source_rank"]),
            }
        )
    summary = summarize_paired_rows(
        ensemble,
        prediction_key="mw_pred_ensemble",
    )
    return ensemble, summary


def _evaluate_external(
    *,
    formal_rows: list[dict[str, Any]],
    event_root: Path,
    label_manifest: Path,
    output_root: Path,
) -> dict[str, Any]:
    labels = _read_csv(label_manifest)
    event_dirs = _external_event_dirs(event_root)
    all_stations: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    seed_summaries: dict[str, Any] = {}
    for training_row in sorted(formal_rows, key=lambda row: int(row["seed"])):
        seed = int(training_row["seed"])
        model_dir = Path(training_row["checkpoint_path"]).parent
        result = evaluate_unseen_events(
            event_dirs=event_dirs,
            model_dir=model_dir,
            output_dir=output_root / f"seed_{seed}",
            radial_peak_min_cm_override=0.0,
            save_plots=False,
        )
        stations = pair_prediction_rows(
            result["station_rows"],
            labels,
            prediction_key="mw_pred",
            old_reference_key="mw_catalog",
        )
        events = pair_prediction_rows(
            result["event_rows"],
            labels,
            prediction_key="mw_pred_median",
            old_reference_key="mw_catalog",
        )
        stations = [{"seed": seed, **row} for row in stations]
        events = [{"seed": seed, **row} for row in events]
        all_stations.extend(stations)
        all_events.extend(events)
        seed_summaries[str(seed)] = {
            "station": summarize_paired_rows(
                stations,
                prediction_key="mw_pred",
            ),
            "event": summarize_paired_rows(
                events,
                prediction_key="mw_pred_median",
            ),
        }
    ensemble_rows, ensemble_summary = _external_ensemble(all_events)
    _atomic_write(output_root / "station_predictions_all_seeds.csv", _csv_bytes(all_stations))
    _atomic_write(output_root / "event_predictions_all_seeds.csv", _csv_bytes(all_events))
    _atomic_write(output_root / "ensemble_event_predictions.csv", _csv_bytes(ensemble_rows))
    summary = {
        "seeds": seed_summaries,
        "ensemble_event": ensemble_summary,
    }
    _atomic_write(output_root / "summary.json", _json_bytes(summary))
    return summary


def run_campaign(
    *,
    snapshot_root: str | Path,
    frozen_config: str | Path,
    frozen_comparison: str | Path,
    output_root: str | Path,
    event_root: str | Path,
    epochs: int,
    resume: bool,
) -> dict[str, Any]:
    if int(epochs) != 200:
        raise ValueError("approved campaign requires exactly 200 epochs")
    snapshot = Path(snapshot_root).resolve()
    frozen_config_path = Path(frozen_config).resolve()
    comparison_path = Path(frozen_comparison).resolve()
    output = Path(output_root).resolve()
    external_root = Path(event_root).resolve()
    candidate_npz = snapshot / "gnss_events_matched.usgs_priority.npz"
    external_label_manifest = snapshot / "external_magnitude_labels.csv"
    for required in (
        snapshot / "COMPLETE",
        candidate_npz,
        snapshot / "magnitude_labels.csv",
        external_label_manifest,
        frozen_config_path,
        comparison_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if git_is_dirty(PROJECT_ROOT):
        raise ValueError("training campaign requires a clean worktree")
    complete_path = output / "COMPLETE"
    summary_path = output / "campaign_summary.json"
    if resume and complete_path.is_file() and summary_path.is_file():
        return _load_json(summary_path)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"campaign output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    preflight = output / "preflight"
    preflight.mkdir(exist_ok=True)

    frozen = _load_yaml(frozen_config_path)
    config = prepare_relabel_config(frozen, candidate_npz)
    _validate_config_change(frozen, config)
    validate_config_v2(config)
    config_path = preflight / "config_v2.usgs_priority.yaml"
    if not config_path.exists():
        _atomic_write(config_path, _config_bytes(config))
    elif _load_yaml(config_path) != config:
        raise ValueError("resume config differs from existing preflight config")

    manifest_path = preflight / "dataset_manifest.csv"
    dataset_summary_path = preflight / "dataset_summary.json"
    if resume and manifest_path.is_file() and dataset_summary_path.is_file():
        dataset_summary = _load_json(dataset_summary_path)
    else:
        manifest_path, dataset_summary = _audit_candidate(
            config,
            preflight_dir=preflight,
        )
    smoke_path = preflight / "finite_smoke.json"
    if not (resume and smoke_path.is_file()):
        _run_device_smokes(config, output_path=smoke_path)

    frozen_run = _frozen_run_root(frozen_config_path)
    baseline = _comparison_baseline(comparison_path)
    pilot_rows = corrected_matrix.run_matrix(
        configs={"usgs_priority": config},
        mode="within-event-station",
        output_root=output / "pilot" / "campaign",
        seeds=(42,),
        epochs=epochs,
        max_events=None,
        event_root=None,
        resume=resume,
        dataset_manifest_path=manifest_path,
    )
    _validate_training_rows(
        pilot_rows,
        required_seeds=(42,),
        frozen_run=frozen_run,
    )
    pilot_mae = float(
        pilot_rows[0]["within_event_station_event_mae_catalog"]
    )
    gate = evaluate_pilot_gate(
        frozen_selected_event_mae=baseline,
        pilot_selected_event_mae=pilot_mae,
    )
    gate_payload = {
        "passed": gate.passed,
        "frozen_selected_event_mae": gate.frozen_selected_event_mae,
        "pilot_selected_event_mae": gate.pilot_selected_event_mae,
        "degradation": gate.degradation,
        "max_degradation": gate.max_degradation,
        "stop_threshold": gate.stop_threshold,
    }
    gate_path = output / "pilot" / "gate.json"
    _atomic_write(gate_path, _json_bytes(gate_payload), overwrite=resume)
    if not gate.passed:
        _atomic_write(output / "PILOT_GATE_FAILED", b"\n", overwrite=resume)
        summary = {
            "status": "pilot_gate_failed",
            "pilot_gate": gate_payload,
            "pilot_rows": pilot_rows,
        }
        _atomic_write(summary_path, _json_bytes(summary), overwrite=resume)
        return summary

    formal_rows = corrected_matrix.run_matrix(
        configs={"usgs_priority": config},
        mode="within-event-station",
        output_root=output / "formal" / "campaign",
        seeds=SEEDS,
        epochs=epochs,
        max_events=None,
        event_root=None,
        resume=resume,
        dataset_manifest_path=manifest_path,
    )
    _validate_training_rows(
        formal_rows,
        required_seeds=SEEDS,
        frozen_run=frozen_run,
    )
    external_summary = _evaluate_external(
        formal_rows=formal_rows,
        event_root=external_root,
        label_manifest=external_label_manifest,
        output_root=output / "external",
    )
    summary = {
        "status": "complete",
        "created_at_utc": utc_now_iso(),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "git_dirty": git_is_dirty(PROJECT_ROOT),
        "candidate_npz": str(candidate_npz),
        "candidate_sha256": sha256_file(candidate_npz),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_summary": dataset_summary,
        "pilot_gate": gate_payload,
        "pilot_rows": pilot_rows,
        "formal_rows": formal_rows,
        "external": external_summary,
    }
    _atomic_write(summary_path, _json_bytes(summary), overwrite=resume)
    _atomic_write(complete_path, b"\n", overwrite=resume)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the gated USGS-priority relabel training campaign",
    )
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--frozen-comparison", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--event-root", required=True, type=Path)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_campaign(
        snapshot_root=args.snapshot_root,
        frozen_config=args.frozen_config,
        frozen_comparison=args.frozen_comparison,
        output_root=args.output_root,
        event_root=args.event_root,
        epochs=args.epochs,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
