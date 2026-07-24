from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    Subset,
    WeightedRandomSampler,
)
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
from src.data.manifest import (  # noqa: E402
    MANIFEST_FIELDS,
    audit_passes,
    build_dataset_summary,
)
from src.evaluation.delayed_prefix import (  # noqa: E402
    MANUSCRIPT_PROCESSING_DELAY_SEC,
)
from src.evaluation.evaluate_delayed_prefix import (  # noqa: E402
    DEFAULT_HORIZONS_SEC,
    evaluate_delayed_prefix,
)
from src.models.model import PINNModel  # noqa: E402
from src.training.loss_stf_rate_v2 import STFRateWaveformLossV2  # noqa: E402
from src.training.train import _prepare_v2_batch  # noqa: E402
from src.utils.config_v2 import validate_config_v2  # noqa: E402
from src.utils.device import get_preferred_device  # noqa: E402
from src.utils.provenance import (  # noqa: E402
    current_git_commit,
    git_is_dirty,
    sha256_file,
    utc_now_iso,
)


SEEDS = (17, 42, 73)
STF_OUTPUT_PARAMETERIZATION_AXIS = "stf_output_parameterization"
SCHEDULER_T0_AXIS = "scheduler_T0"
RADIAL_DYNAMIC_RANGE_STEM_AXIS = "radial_dynamic_range_stem"
EVENT_BALANCED_SAMPLING_AXIS = "event_balanced_sampling"
VARIANT_AXES = {
    STF_OUTPUT_PARAMETERIZATION_AXIS: {
        "baseline": "direct",
        "candidate": "moment_shape_factorized",
    },
    SCHEDULER_T0_AXIS: {
        "baseline": 15,
        "candidate": 195,
    },
    RADIAL_DYNAMIC_RANGE_STEM_AXIS: {
        "baseline": "none",
        "candidate": "asinh_residual",
    },
    EVENT_BALANCED_SAMPLING_AXIS: {
        "baseline": False,
        "candidate": True,
    },
}
VARIANT_AXIS_PATHS = {
    STF_OUTPUT_PARAMETERIZATION_AXIS: ("model", "stf_output_parameterization"),
    SCHEDULER_T0_AXIS: ("training", "scheduler_T0"),
    RADIAL_DYNAMIC_RANGE_STEM_AXIS: ("model", "radial_dynamic_range_stem"),
    EVENT_BALANCED_SAMPLING_AXIS: ("training", "event_balanced_sampling"),
}
# Backward-compatible alias for the Phase23 campaign and its persisted tests.
VARIANTS = VARIANT_AXES[STF_OUTPUT_PARAMETERIZATION_AXIS]
VALIDATION_METRIC = "validation_event_mae_catalog"
EXPECTED_SOURCE_SHA256 = (
    "2e1fa4c12fc1eb03ffc8bf9235491f0886d4b0d360ebcb3486baca4d948cfd6a"
)
EXPECTED_EVENT_COUNT = 31
EXPECTED_STATION_COUNT = 2558
EXPECTED_SPLIT_COUNTS = (1788, 385, 385)
EXPECTED_SPLIT_SHA256 = {
    17: "fa5c5d1cd3bdb3e8a775140a9bea4adce885b151eff717627d5e8ab82fd4e9a8",
    42: "5ac2e07ed186dce737a3592694632775b7bbf603bf922a4a74fa6b86a3d5c240",
    73: "786d029482fc8c6c8000b939380f7d4f6fdab2cc0dfe39a09fa982d1d9049548",
}
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
INTERNAL_EVENT_MAE_MAXIMUM = 0.15
FULL_OBSERVATION_HORIZON_SEC = 200.0
FULL_RELEASE_TIME_SEC = (
    FULL_OBSERVATION_HORIZON_SEC + MANUSCRIPT_PROCESSING_DELAY_SEC
)


class LockedTestLoader:
    """Expose test-set size to train(), but fail if training iterates test data."""

    def __init__(self, loader: Any) -> None:
        self.dataset = loader.dataset
        self._batch_count = len(loader)

    def __len__(self) -> int:
        return self._batch_count

    def __iter__(self) -> Iterable[Any]:
        raise RuntimeError("the train stage is forbidden from reading locked test data")


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


def _yaml_bytes(payload: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(payload),
        allow_unicode=True,
        sort_keys=False,
    ).encode("utf-8")


def _csv_bytes(
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> bytes:
    if not rows and fieldnames is None:
        raise ValueError("cannot serialize an empty CSV")
    fields = list(fieldnames or rows[0].keys())
    if not fields:
        raise ValueError("CSV fieldnames must not be empty")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _atomic_write(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    _atomic_write(path, _json_bytes(payload), overwrite=overwrite)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return payload


def _artifact(path: str | Path) -> dict[str, str]:
    candidate = Path(path).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return {"path": str(candidate), "sha256": sha256_file(candidate)}


def _validate_artifact(reference: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(reference.get("path", "")))
    expected = str(reference.get("sha256", ""))
    if not path.is_file() or not expected:
        raise FileNotFoundError(f"{label} artifact is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} artifact hash changed: {path}")
    return path


def _stage_dir(output_root: Path, stage: str) -> Path:
    return output_root / stage


def _prepare_stage(path: Path, *, resume: bool) -> dict[str, Any] | None:
    summary_path = path / "summary.json"
    complete_path = path / "COMPLETE"
    if complete_path.is_file() and summary_path.is_file():
        summary = _load_json(summary_path)
        if resume:
            return summary
        raise FileExistsError(f"stage is already complete: {path}")
    if path.exists():
        unexpected = [item for item in path.iterdir() if item.name != "console.log"]
        if unexpected and not resume:
            raise FileExistsError(f"stage output is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return None


def _finish_stage(path: Path, summary: Mapping[str, Any], *, resume: bool) -> None:
    _atomic_json(path / "summary.json", dict(summary), overwrite=resume)
    _atomic_write(path / "COMPLETE", b"\n", overwrite=resume)


def _require_stage(output_root: Path, stage: str) -> dict[str, Any]:
    path = _stage_dir(output_root, stage)
    summary_path = path / "summary.json"
    if not (path / "COMPLETE").is_file() or not summary_path.is_file():
        raise RuntimeError(f"required stage is incomplete: {stage}")
    payload = _load_json(summary_path)
    if payload.get("stage") != stage:
        raise ValueError(f"stage summary identity mismatch: {stage}")
    return payload


def _config_diff_paths(
    base: Any,
    candidate: Any,
    *,
    prefix: str = "",
) -> set[str]:
    if isinstance(base, Mapping) and isinstance(candidate, Mapping):
        differences: set[str] = set()
        keys = set(base) | set(candidate)
        for key in keys:
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in base or key not in candidate:
                differences.add(name)
            else:
                differences.update(
                    _config_diff_paths(base[key], candidate[key], prefix=name)
                )
        return differences
    if isinstance(base, (list, tuple)) and isinstance(candidate, (list, tuple)):
        return set() if list(base) == list(candidate) else {prefix}
    return set() if base == candidate else {prefix}


def variant_axis_from_config(base_config: Mapping[str, Any]) -> str:
    model = base_config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("formal config is missing model")
    training = base_config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("formal config is missing training")
    parameterization = model.get("stf_output_parameterization")
    scheduler_t0 = training.get("scheduler_T0")
    stem_is_explicit = "radial_dynamic_range_stem" in model
    campaign = base_config.get("campaign", {})
    if not isinstance(campaign, Mapping):
        raise ValueError("formal config campaign marker must be a mapping")
    explicit_axis = campaign.get("variant_axis")
    if explicit_axis is not None:
        if explicit_axis != EVENT_BALANCED_SAMPLING_AXIS:
            raise ValueError(f"unsupported formal campaign axis: {explicit_axis!r}")
        if (
            parameterization != "moment_shape_factorized"
            or scheduler_t0 != 15
            or stem_is_explicit
            or training.get("event_balanced_sampling") is not False
        ):
            raise ValueError(
                "Phase26 event-balanced baseline requires factorized STF, "
                "scheduler_T0=15, the original radial stem, and "
                "event_balanced_sampling=false"
            )
        return EVENT_BALANCED_SAMPLING_AXIS
    if (
        parameterization == "direct"
        and scheduler_t0 == 15
        and not stem_is_explicit
    ):
        return STF_OUTPUT_PARAMETERIZATION_AXIS
    if parameterization == "moment_shape_factorized" and scheduler_t0 == 15:
        if stem_is_explicit:
            if model.get("radial_dynamic_range_stem") == "none":
                return RADIAL_DYNAMIC_RANGE_STEM_AXIS
            raise ValueError(
                "Phase25 formal baseline requires explicit "
                "model.radial_dynamic_range_stem='none'"
            )
        return SCHEDULER_T0_AXIS
    raise ValueError(
        "formal config must describe the Phase23 direct baseline or the "
        "Phase24 factorized/T0=15 baseline or the Phase25 factorized/T0=15 "
        "baseline with explicit radial dynamic-range stem or the explicitly "
        "marked Phase26 event-balanced baseline"
    )


def build_variant_configs(
    base_config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    axis = variant_axis_from_config(base_config)
    variants: dict[str, dict[str, Any]] = {}
    for name, value in VARIANT_AXES[axis].items():
        config = copy.deepcopy(dict(base_config))
        section, key = VARIANT_AXIS_PATHS[axis]
        config[section][key] = value
        variants[name] = config
    differences = _config_diff_paths(variants["baseline"], variants["candidate"])
    expected = {".".join(VARIANT_AXIS_PATHS[axis])}
    if differences != expected:
        raise ValueError(
            "baseline/candidate scientific diff changed: "
            f"expected={sorted(expected)}, actual={sorted(differences)}"
        )
    return variants


def _require_exact(config: Mapping[str, Any], path: Sequence[str], expected: Any) -> None:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"formal config is missing {'.'.join(path)}")
        value = value[key]
    if value != expected or type(value) is not type(expected):
        raise ValueError(
            f"formal config requires {'.'.join(path)}={expected!r}; got {value!r}"
        )


def validate_formal_config(config: dict[str, Any]) -> None:
    validate_config_v2(config)
    if "workflow" in config:
        raise ValueError("manuscript STF config must use a source-aligned STF")
    required = {
        ("pipeline_version",): 2,
        ("dataset", "sample_rate_hz"): 1.0,
        ("dataset", "radial_peak_min_cm"): 2.0,
        ("dataset", "waveform", "duration_sec"): 200.0,
        ("dataset", "waveform", "max_interpolation_gap_sec"): 0.0,
        ("dataset", "filter", "type"): "lowpass",
        ("dataset", "filter", "cutoff_hz"): 0.2,
        ("dataset", "filter", "num_taps"): 7,
        ("dataset", "filter", "window"): "hamming",
        ("dataset", "stf", "duration_sec"): 200.0,
        ("dataset", "stf", "magnitude_target"): "catalog",
        ("dataset", "stf", "preserve_integral"): True,
        ("physics", "distance_mode"): "hypocentral",
        ("physics", "delay_mode"): "absolute",
        ("model", "hidden_dim"): 128,
        ("model", "num_tcn_blocks"): 6,
        ("model", "transformer_num_layers"): 3,
        ("model", "dropout"): 0.2,
        ("model", "input_components"): ["radial"],
        ("model", "predict_catalog_mw"): False,
        ("training", "split_protocol"): "within_event_station",
        ("training", "validation_event_fraction"): 0.15,
        ("training", "test_event_fraction"): 0.15,
        ("training", "event_balanced_sampling"): False,
        ("training", "early_stop_metric"): "event_mae_catalog",
        ("training", "checkpoint_metric"): "event_mae_catalog",
        ("training", "early_stop_min_delta"): 0.0,
        ("training", "epochs"): 200,
        ("training", "warmup_epochs"): 5,
        ("training", "scheduler_T_mult"): 2,
        ("training", "stf_rate_loss", "lambda_MSE"): 1.0,
        ("training", "stf_rate_loss", "lambda_synth"): 0.5,
        ("training", "stf_rate_loss", "lambda_mag"): 1.0,
        ("training", "stf_rate_loss", "lambda_shape"): 0.1,
        ("training", "stf_rate_loss", "include_intermediate_field"): False,
        ("training", "stf_rate_loss", "include_far_field_P"): True,
        ("training", "stf_rate_loss", "include_far_field_S"): True,
        ("training", "stf_rate_loss", "radiation_pattern_mode"): "full",
        ("evaluation", "primary_reference"): "catalog",
    }
    for path, expected in required.items():
        _require_exact(config, path, expected)
    build_variant_configs(config)


def _runtime_config(
    config: Mapping[str, Any],
    *,
    run_root: Path,
    seed: int,
    dataset_manifest: Path,
) -> dict[str, Any]:
    runtime = copy.deepcopy(dict(config))
    runtime["training"]["random_seed"] = int(seed)
    runtime["paths"].update(
        {
            "output_dir": str(run_root),
            "models_dir": str(run_root / "models"),
            "logs_dir": str(run_root / "logs"),
            "results_dir": str(run_root / "results"),
            "dataset_manifest_path": str(dataset_manifest.resolve()),
        }
    )
    validate_config_v2(runtime)
    return runtime


def _training_sampling_manifest(
    train_loader: DataLoader[Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = train_loader.dataset
    if not isinstance(dataset, Subset):
        raise TypeError("formal training loader must wrap a Subset")
    samples = getattr(dataset.dataset, "samples", None)
    if not isinstance(samples, list):
        raise TypeError("formal training dataset must expose sample metadata")

    sample_rows = [samples[int(index)] for index in dataset.indices]
    events = [str(row["event"]) for row in sample_rows]
    stations = [str(row["station"]) for row in sample_rows]
    counts = Counter(events)
    if (
        len(events) != EXPECTED_SPLIT_COUNTS[0]
        or len(counts) != EXPECTED_EVENT_COUNT
    ):
        raise ValueError("formal training sampling cohort changed")

    enabled = bool(config["training"]["event_balanced_sampling"])
    sampler = train_loader.sampler
    if enabled:
        if not isinstance(sampler, WeightedRandomSampler):
            raise TypeError("event-balanced training requires WeightedRandomSampler")
        if not sampler.replacement:
            raise ValueError("event-balanced training must sample with replacement")
        weights = [float(value) for value in sampler.weights.tolist()]
        expected_weights = [1.0 / counts[event] for event in events]
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, expected in zip(weights, expected_weights, strict=True)
        ):
            raise ValueError("event-balanced sampler weights changed")
        mode = "event_equal_with_replacement"
    else:
        if not isinstance(sampler, RandomSampler) or sampler.replacement:
            raise TypeError(
                "baseline training requires shuffled sampling without replacement"
            )
        weights = [1.0] * len(events)
        mode = "station_uniform_without_replacement"

    draw_count = int(sampler.num_samples)
    if draw_count != len(events):
        raise ValueError("formal sampler draw count changed")
    if sampler.generator is not train_loader.generator:
        raise ValueError("sampler and DataLoader must share their seeded generator")

    total_weight = sum(weights)
    event_weights: dict[str, float] = {event: 0.0 for event in counts}
    for event, weight in zip(events, weights, strict=True):
        event_weights[event] += weight
    event_draws = [
        draw_count * value / total_weight for value in event_weights.values()
    ]
    if enabled:
        record_probabilities = [weight / total_weight for weight in weights]
        expected_unique = sum(
            1.0 - (1.0 - probability) ** draw_count
            for probability in record_probabilities
        )
    else:
        expected_unique = float(draw_count)
    weight_rows = [
        {"event": event, "station": station, "weight": weight}
        for event, station, weight in zip(events, stations, weights, strict=True)
    ]
    return {
        "schema_version": 1,
        "mode": mode,
        "event_balanced_sampling": enabled,
        "sampler_class": type(sampler).__name__,
        "replacement": bool(sampler.replacement),
        "draw_count": draw_count,
        "record_count": len(events),
        "event_count": len(counts),
        "event_record_count_minimum": min(counts.values()),
        "event_record_count_maximum": max(counts.values()),
        "expected_event_draws_minimum": min(event_draws),
        "expected_event_draws_maximum": max(event_draws),
        "expected_unique_record_count": expected_unique,
        "sample_weight_sha256": hashlib.sha256(_json_bytes(weight_rows)).hexdigest(),
    }


def _assert_split_manifest(manifest: Mapping[str, Any], *, seed: int) -> None:
    counts = tuple(
        int(manifest[f"{name}_record_count"])
        for name in ("train", "validation", "test")
    )
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"seed {seed} split counts changed: "
            f"expected={EXPECTED_SPLIT_COUNTS}, actual={counts}"
        )
    if int(manifest.get("seed", -1)) != seed:
        raise ValueError(f"split seed mismatch: expected {seed}")
    if str(manifest.get("protocol")) != "within_event_station":
        raise ValueError("formal split protocol changed")
    assignment = str(manifest.get("assignment_sha256", ""))
    if assignment != EXPECTED_SPLIT_SHA256[seed]:
        raise ValueError(f"seed {seed} split assignment changed: {assignment}")


def _assert_same_split(candidate: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
    for key in (
        "protocol",
        "seed",
        "assignment_sha256",
        "sample_keys",
        "per_event_station_counts",
    ):
        if candidate.get(key) != frozen.get(key):
            raise ValueError(f"training split differs from preflight: {key}")


def run_preflight(
    *,
    config_path: Path,
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    stage_dir = _stage_dir(output_root, "preflight")
    completed = _prepare_stage(stage_dir, resume=resume)
    if completed is not None:
        return completed
    if git_is_dirty(PROJECT_ROOT):
        raise ValueError("formal preflight requires a clean Git worktree")

    source_config = resolve_data_paths(_load_yaml(config_path.resolve()))
    validate_formal_config(source_config)
    source_data = Path(source_config["paths"]["data_path"])
    source_hash = sha256_file(source_data)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"source NPZ hash changed: {source_hash}")

    dataset = CorrectedEarthquakeDataset(source_config)
    accepted_events = {str(sample["event"]) for sample in dataset.samples}
    if len(dataset.samples) != EXPECTED_STATION_COUNT:
        raise ValueError(
            f"accepted station count changed: {len(dataset.samples)}"
        )
    if len(accepted_events) != EXPECTED_EVENT_COUNT:
        raise ValueError(f"accepted event count changed: {len(accepted_events)}")
    dataset_summary = build_dataset_summary(dataset)
    if not audit_passes(dataset_summary):
        raise ValueError("dataset audit invariants failed")

    frozen_config_path = stage_dir / "config.yaml"
    manifest_path = stage_dir / "dataset_manifest.csv"
    dataset_summary_path = stage_dir / "dataset_summary.json"
    _atomic_write(
        frozen_config_path,
        _yaml_bytes(source_config),
        overwrite=resume,
    )
    _atomic_write(
        manifest_path,
        _csv_bytes(dataset.manifest_rows, fieldnames=MANIFEST_FIELDS),
        overwrite=resume,
    )
    _atomic_json(dataset_summary_path, dataset_summary, overwrite=resume)

    split_references: dict[str, Any] = {}
    for seed in SEEDS:
        split_config = copy.deepcopy(source_config)
        split_config["training"]["random_seed"] = seed
        _, _, _, split_manifest = get_data_loaders_v2(split_config)
        _assert_split_manifest(split_manifest, seed=seed)
        split_path = stage_dir / f"split_seed_{seed}.json"
        _atomic_json(split_path, split_manifest, overwrite=resume)
        split_references[str(seed)] = {
            "assignment_sha256": split_manifest["assignment_sha256"],
            "train_record_count": split_manifest["train_record_count"],
            "validation_record_count": split_manifest["validation_record_count"],
            "test_record_count": split_manifest["test_record_count"],
            "manifest": _artifact(split_path),
        }

    summary = {
        "stage": "preflight",
        "status": "complete",
        "variant_axis": variant_axis_from_config(source_config),
        "created_at_utc": utc_now_iso(),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "git_dirty": False,
        "source_data": _artifact(source_data),
        "source_config": _artifact(config_path),
        "frozen_config": _artifact(frozen_config_path),
        "dataset_manifest": _artifact(manifest_path),
        "dataset_summary": _artifact(dataset_summary_path),
        "accepted_event_count": len(accepted_events),
        "accepted_station_count": len(dataset.samples),
        "filter_cutoff_hz": 0.2,
        "radial_peak_min_cm": 2.0,
        "seeds": list(SEEDS),
        "splits": split_references,
    }
    _finish_stage(stage_dir, summary, resume=resume)
    return summary


def _preflight_config(preflight: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    config_path = _validate_artifact(preflight["frozen_config"], label="preflight config")
    manifest_path = _validate_artifact(
        preflight["dataset_manifest"],
        label="dataset manifest",
    )
    config = _load_yaml(config_path)
    validate_formal_config(config)
    if str(preflight["source_data"]["sha256"]) != EXPECTED_SOURCE_SHA256:
        raise ValueError("preflight source hash is not the frozen USGS snapshot")
    return config, manifest_path


def _smoke_one_device(
    config: dict[str, Any],
    *,
    dataset: CorrectedEarthquakeDataset,
    device: torch.device,
) -> dict[str, Any]:
    batch_size = min(2, len(dataset))
    loader = DataLoader(
        Subset(dataset, list(range(batch_size))),
        batch_size=batch_size,
        shuffle=False,
    )
    batch = next(iter(loader))
    prepared = _prepare_v2_batch(batch, config, device)
    model = PINNModel(config).to(device).train()
    criterion = STFRateWaveformLossV2(config).to(device)
    prediction = model(prepared.model_input, meta=prepared.metadata)
    loss, metrics = criterion(
        prediction,
        pred_catalog_mw=None,
        radial_obs=prepared.radial,
        source_distance_m=prepared.source_distance_m,
        theta_deg=prepared.theta_deg,
        phi_slip_deg=prepared.phi_slip_deg,
        source_dt_sec=prepared.source_dt_sec,
        observation_dt_sec=prepared.observation_dt_sec,
        waveform_valid_mask=prepared.waveform_valid_mask,
        stf_true=prepared.stf_true,
        has_stf=prepared.has_stf,
        true_mag=prepared.true_mag,
    )
    values = [float(loss.detach().cpu()), *map(float, metrics.values())]
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError(f"non-finite smoke metrics on {device}: {metrics}")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not gradients or any(gradient is None for gradient in gradients):
        raise FloatingPointError(f"smoke backward missed gradients on {device}")
    if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise FloatingPointError(f"smoke backward is non-finite on {device}")
    result = {
        "device": str(device),
        "batch_size": batch_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "loss": float(loss.detach().cpu()),
        "metrics": metrics,
        "passed": True,
    }
    del batch, prepared, prediction, criterion, model, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_smoke(
    *,
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    stage_dir = _stage_dir(output_root, "smoke")
    completed = _prepare_stage(stage_dir, resume=resume)
    if completed is not None:
        return completed
    preflight = _require_stage(output_root, "preflight")
    config, _ = _preflight_config(preflight)
    variants = build_variant_configs(config)
    dataset = CorrectedEarthquakeDataset(config)
    if not torch.cuda.is_available():
        raise RuntimeError("formal smoke requires both CPU and CUDA")
    results: dict[str, Any] = {}
    for variant, variant_config in variants.items():
        results[variant] = {}
        for device in (torch.device("cpu"), torch.device("cuda")):
            results[variant][device.type] = _smoke_one_device(
                variant_config,
                dataset=dataset,
                device=device,
            )
    summary = {
        "stage": "smoke",
        "status": "complete",
        "created_at_utc": utc_now_iso(),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "preflight_summary": _artifact(
            _stage_dir(output_root, "preflight") / "summary.json"
        ),
        "test_evaluated": False,
        "external_evaluated": False,
        "results": results,
    }
    _finish_stage(stage_dir, summary, resume=resume)
    return summary


def select_checkpoint_from_validation_log(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_delta: float,
) -> dict[str, Any]:
    if minimum_delta < 0.0 or not math.isfinite(minimum_delta):
        raise ValueError("minimum_delta must be finite and nonnegative")
    best_value = float("inf")
    selected: dict[str, Any] | None = None
    for row in rows:
        try:
            epoch = int(row["Epoch"])
            value = float(row[VALIDATION_METRIC])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("validation log lacks the frozen selection metric") from error
        if not math.isfinite(value):
            raise ValueError("validation selection metric is non-finite")
        if value < best_value - minimum_delta:
            best_value = value
            selected = {"epoch": epoch, VALIDATION_METRIC: value}
    if selected is None:
        raise ValueError("validation log contains no selectable epoch")
    return selected


def select_seed_by_validation(seed_rows: Mapping[int, Mapping[str, Any]]) -> int:
    if set(seed_rows) != set(SEEDS):
        raise ValueError(f"seed selection requires exactly {SEEDS}")
    candidates: list[tuple[float, int]] = []
    for seed in SEEDS:
        value = float(seed_rows[seed][VALIDATION_METRIC])
        if not math.isfinite(value):
            raise ValueError(f"seed {seed} validation metric is non-finite")
        candidates.append((value, seed))
    return min(candidates)[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _seed_summary_is_valid(
    summary: Mapping[str, Any],
    *,
    require_sampling: bool = False,
) -> bool:
    try:
        for name in ("checkpoint", "config", "split", "training_log", "run_manifest"):
            _validate_artifact(summary[name], label=name)
        if require_sampling or "sampling" in summary:
            _validate_artifact(summary["sampling"], label="sampling")
        return int(summary["seed"]) in SEEDS and math.isfinite(
            float(summary[VALIDATION_METRIC])
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _train_one_seed(
    *,
    variant: str,
    config: dict[str, Any],
    seed: int,
    seed_root: Path,
    dataset_manifest: Path,
    frozen_split_path: Path,
    resume: bool,
) -> dict[str, Any]:
    seed_summary_path = seed_root / "seed_summary.json"
    if seed_summary_path.is_file():
        summary = _load_json(seed_summary_path)
        campaign = config.get("campaign", {})
        require_sampling = (
            isinstance(campaign, Mapping)
            and campaign.get("variant_axis") == EVENT_BALANCED_SAMPLING_AXIS
        )
        if resume and _seed_summary_is_valid(
            summary,
            require_sampling=require_sampling,
        ):
            return summary
        if not resume:
            raise FileExistsError(f"seed output already exists: {seed_root}")

    runtime = _runtime_config(
        config,
        run_root=seed_root,
        seed=seed,
        dataset_manifest=dataset_manifest,
    )
    train_loader, validation_loader, test_loader, split_manifest = (
        get_data_loaders_v2(runtime)
    )
    _assert_split_manifest(split_manifest, seed=seed)
    frozen_split = _load_json(frozen_split_path)
    _assert_same_split(split_manifest, frozen_split)
    sampling_manifest_path = seed_root / "sampling_manifest.json"
    _atomic_json(
        sampling_manifest_path,
        _training_sampling_manifest(train_loader, runtime),
        overwrite=resume,
    )
    locked_loaders = (
        train_loader,
        validation_loader,
        LockedTestLoader(test_loader),
        split_manifest,
    )
    result = corrected_matrix._train_or_resume(
        runtime,
        locked_loaders,
        resume=resume,
    )
    verified = corrected_matrix._verify_training_result(runtime, result)
    log_path = Path(verified["training_log_path"])
    checkpoint_selection = select_checkpoint_from_validation_log(
        _read_csv(log_path),
        minimum_delta=float(runtime["training"]["early_stop_min_delta"]),
    )
    if not math.isclose(
        float(result["best_mw_mae"]),
        float(checkpoint_selection[VALIDATION_METRIC]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("training result and validation-log checkpoint disagree")
    summary = {
        "variant": variant,
        "parameterization": config["model"]["stf_output_parameterization"],
        "scheduler_T0": int(config["training"]["scheduler_T0"]),
        "radial_dynamic_range_stem": str(
            config["model"].get("radial_dynamic_range_stem", "none")
        ),
        "event_balanced_sampling": bool(
            config["training"]["event_balanced_sampling"]
        ),
        "seed": seed,
        **checkpoint_selection,
        "checkpoint": _artifact(verified["checkpoint_path"]),
        "config": _artifact(result["config_snapshot_path"]),
        "split": _artifact(result["split_manifest_path"]),
        "training_log": _artifact(log_path),
        "run_manifest": _artifact(result["run_manifest_path"]),
        "sampling": _artifact(sampling_manifest_path),
        "last_checkpoint": _artifact(result["last_checkpoint_path"]),
        "split_assignment_sha256": split_manifest["assignment_sha256"],
        "parameter_count": int(verified["parameter_count"]),
        "device": str(result["device"]),
    }
    _atomic_json(seed_summary_path, summary, overwrite=resume)
    return summary


def run_train(
    *,
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    stage_dir = _stage_dir(output_root, "train")
    completed = _prepare_stage(stage_dir, resume=resume)
    if completed is not None:
        return completed
    preflight = _require_stage(output_root, "preflight")
    smoke = _require_stage(output_root, "smoke")
    if smoke.get("status") != "complete":
        raise RuntimeError("smoke stage did not pass")
    config, dataset_manifest = _preflight_config(preflight)
    variant_axis = variant_axis_from_config(config)
    variants = build_variant_configs(config)
    variant_summaries: dict[str, Any] = {}
    for variant, variant_config in variants.items():
        seed_summaries: dict[int, dict[str, Any]] = {}
        for seed in SEEDS:
            frozen_split_path = _validate_artifact(
                preflight["splits"][str(seed)]["manifest"],
                label=f"preflight split seed {seed}",
            )
            seed_summaries[seed] = _train_one_seed(
                variant=variant,
                config=variant_config,
                seed=seed,
                seed_root=stage_dir / variant / f"seed_{seed}",
                dataset_manifest=dataset_manifest,
                frozen_split_path=frozen_split_path,
                resume=resume,
            )
        selected_seed = select_seed_by_validation(seed_summaries)
        selection = {
            "selected_seed": selected_seed,
            "selection_metric": VALIDATION_METRIC,
            "ensemble_used": False,
            "candidates": {
                str(seed): float(seed_summaries[seed][VALIDATION_METRIC])
                for seed in SEEDS
            },
        }
        selection_path = stage_dir / variant / "selection.json"
        _atomic_json(selection_path, selection, overwrite=resume)
        variant_summaries[variant] = {
            "parameterization": variant_config["model"][
                "stf_output_parameterization"
            ],
            "scheduler_T0": int(variant_config["training"]["scheduler_T0"]),
            "radial_dynamic_range_stem": str(
                variant_config["model"].get("radial_dynamic_range_stem", "none")
            ),
            "event_balanced_sampling": bool(
                variant_config["training"]["event_balanced_sampling"]
            ),
            "scientific_diff_from_baseline": sorted(
                _config_diff_paths(variants["baseline"], variant_config)
            ),
            "seeds": {str(seed): seed_summaries[seed] for seed in SEEDS},
            "selection": selection,
            "selection_artifact": _artifact(selection_path),
        }
    summary = {
        "stage": "train",
        "status": "complete",
        "created_at_utc": utc_now_iso(),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "preflight_summary": _artifact(
            _stage_dir(output_root, "preflight") / "summary.json"
        ),
        "smoke_summary": _artifact(
            _stage_dir(output_root, "smoke") / "summary.json"
        ),
        "variant_axis": variant_axis,
        "selection_metric": VALIDATION_METRIC,
        "test_evaluated": False,
        "external_evaluated": False,
        "variants": variant_summaries,
    }
    _finish_stage(stage_dir, summary, resume=resume)
    return summary


def candidate_validation_improves(train_summary: Mapping[str, Any]) -> bool:
    variants = train_summary["variants"]
    baseline_selection = variants["baseline"]["selection"]
    candidate_selection = variants["candidate"]["selection"]
    baseline_seed = str(baseline_selection["selected_seed"])
    candidate_seed = str(candidate_selection["selected_seed"])
    baseline = float(variants["baseline"]["seeds"][baseline_seed][VALIDATION_METRIC])
    candidate = float(variants["candidate"]["seeds"][candidate_seed][VALIDATION_METRIC])
    if not math.isfinite(baseline) or not math.isfinite(candidate):
        raise ValueError("selected validation metrics must be finite")
    return candidate < baseline


def _selected_seed_summary(
    train_summary: Mapping[str, Any],
    variant: str,
) -> Mapping[str, Any]:
    variant_summary = train_summary["variants"][variant]
    selected_seed = int(variant_summary["selection"]["selected_seed"])
    if selected_seed not in SEEDS:
        raise ValueError(f"invalid selected seed for {variant}: {selected_seed}")
    return variant_summary["seeds"][str(selected_seed)]


def _evaluation_artifacts(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metrics": _artifact(result["metrics_json_path"]),
        "station_predictions": _artifact(result["station_csv_path"]),
        "event_predictions": _artifact(result["event_csv_path"]),
        "result_registry": _artifact(result["result_registry_path"]),
    }


def _persist_delayed_prefix_result(
    result: Mapping[str, Any],
    *,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    station_rows = list(result["station_rows"])
    event_rows = list(result["event_rows"])
    unavailable_rows = list(result["unavailable_rows"])
    horizon_metrics = list(result["horizon_metrics"])
    cohort = dict(result["cohort"])
    if not station_rows or not event_rows or not horizon_metrics:
        raise ValueError("delayed-prefix evaluation produced incomplete outputs")
    expected_cohort = {
        "radial_peak_min_cm": 2.0,
        "waveform_prefix_causal": True,
        "station_selection_causal": False,
        "end_to_end_causal": False,
    }
    if any(cohort.get(name) != value for name, value in expected_cohort.items()):
        raise ValueError("delayed-prefix cohort contract changed")
    observed_horizons = tuple(
        int(row["observation_horizon_sec"]) for row in horizon_metrics
    )
    if observed_horizons != DEFAULT_HORIZONS_SEC:
        raise ValueError("delayed-prefix horizons changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    station_path = output_dir / "station_predictions.csv"
    event_path = output_dir / "event_predictions.csv"
    unavailable_path = output_dir / "unavailable_stations.csv"
    horizon_path = output_dir / "horizon_metrics.json"
    cohort_path = output_dir / "cohort_contract.json"
    _atomic_write(station_path, _csv_bytes(station_rows), overwrite=resume)
    _atomic_write(event_path, _csv_bytes(event_rows), overwrite=resume)
    _atomic_write(
        unavailable_path,
        _csv_bytes(
            unavailable_rows,
            fieldnames=(
                "event",
                "station",
                "observation_horizon_sec",
                "release_time_sec",
                "reason",
                "baseline_ready_time_sec",
            ),
        ),
        overwrite=resume,
    )
    _atomic_json(horizon_path, horizon_metrics, overwrite=resume)
    _atomic_json(cohort_path, cohort, overwrite=resume)
    return {
        "horizons_sec": list(DEFAULT_HORIZONS_SEC),
        "cohort": cohort,
        "station_prediction_count": len(station_rows),
        "event_prediction_count": len(event_rows),
        "unavailable_station_count": len(unavailable_rows),
        "horizon_metrics": horizon_metrics,
        "artifacts": {
            "station_predictions": _artifact(station_path),
            "event_predictions": _artifact(event_path),
            "unavailable_stations": _artifact(unavailable_path),
            "horizon_metrics": _artifact(horizon_path),
            "cohort_contract": _artifact(cohort_path),
        },
    }


def _evaluate_delayed_locked_test(
    *,
    checkpoint_path: Path,
    config: dict[str, Any],
    test_loader: Any,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    device = get_preferred_device()
    model = PINNModel(config).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    result = evaluate_delayed_prefix(model.eval(), config, test_loader)
    return _persist_delayed_prefix_result(
        result,
        output_dir=output_dir,
        resume=resume,
    )


def _evaluate_locked_test(
    *,
    seed_summary: Mapping[str, Any],
    output_dir: Path,
    include_delayed_prefix: bool,
    resume: bool,
) -> dict[str, Any]:
    from src.evaluation.evaluate import evaluate

    checkpoint_path = _validate_artifact(seed_summary["checkpoint"], label="checkpoint")
    config_path = _validate_artifact(seed_summary["config"], label="config")
    split_path = _validate_artifact(seed_summary["split"], label="split")
    config = _load_yaml(config_path)
    _, _, test_loader, split_manifest = get_data_loaders_v2(config)
    _assert_same_split(split_manifest, _load_json(split_path))
    evaluation_config = copy.deepcopy(config)
    evaluation_config["paths"]["models_dir"] = str(checkpoint_path.parent)
    evaluation_config["paths"]["results_dir"] = str(output_dir)
    result = evaluate(
        model_path=checkpoint_path,
        results_run_id=f"selected_seed_{int(seed_summary['seed'])}",
        config=evaluation_config,
        save_plots=False,
        show_plots=False,
        save_metrics=True,
        test_loader=test_loader,
    )
    metrics = dict(result["metrics"])
    required = (
        "station_mae",
        "station_rmse",
        "station_bias",
        "event_mae",
        "event_rmse",
        "event_bias",
    )
    if not all(math.isfinite(float(metrics[name])) for name in required):
        raise ValueError("locked-test metrics are incomplete")
    summary = {
        "selected_seed": int(seed_summary["seed"]),
        "metrics": metrics,
        "artifacts": _evaluation_artifacts(result),
    }
    if include_delayed_prefix:
        summary["delayed_prefix"] = _evaluate_delayed_locked_test(
            checkpoint_path=checkpoint_path,
            config=config,
            test_loader=test_loader,
            output_dir=output_dir / "delayed_prefix",
            resume=resume,
        )
    return summary


def _load_completed_evaluation(
    path: Path,
    *,
    require_delayed_prefix: bool,
) -> dict[str, Any] | None:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        return None
    summary = _load_json(summary_path)
    try:
        for reference in summary["artifacts"].values():
            _validate_artifact(reference, label="evaluation")
        if require_delayed_prefix:
            for reference in summary["delayed_prefix"]["artifacts"].values():
                _validate_artifact(reference, label="delayed-prefix evaluation")
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return summary


def run_internal(
    *,
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    stage_dir = _stage_dir(output_root, "internal")
    completed = _prepare_stage(stage_dir, resume=resume)
    if completed is not None:
        return completed
    train_summary = _require_stage(output_root, "train")
    if train_summary.get("status") != "complete":
        raise RuntimeError("train stage did not complete")
    validation_passed = candidate_validation_improves(train_summary)
    baseline_seed = _selected_seed_summary(train_summary, "baseline")
    candidate_seed = _selected_seed_summary(train_summary, "candidate")
    validation_gate = {
        "passed": validation_passed,
        "baseline": float(baseline_seed[VALIDATION_METRIC]),
        "candidate": float(candidate_seed[VALIDATION_METRIC]),
        "metric": VALIDATION_METRIC,
        "rule": "candidate < baseline",
    }
    if not validation_passed:
        summary = {
            "stage": "internal",
            "status": "candidate_validation_gate_failed",
            "created_at_utc": utc_now_iso(),
            "train_summary": _artifact(
                _stage_dir(output_root, "train") / "summary.json"
            ),
            "validation_gate": validation_gate,
            "test_evaluated": False,
            "external_evaluated": False,
        }
        _finish_stage(stage_dir, summary, resume=resume)
        return summary

    evaluations: dict[str, Any] = {}
    for variant, seed_summary in (
        ("baseline", baseline_seed),
        ("candidate", candidate_seed),
    ):
        variant_dir = stage_dir / variant
        include_delayed_prefix = variant == "candidate"
        result = (
            _load_completed_evaluation(
                variant_dir,
                require_delayed_prefix=include_delayed_prefix,
            )
            if resume
            else None
        )
        if result is None:
            result = _evaluate_locked_test(
                seed_summary=seed_summary,
                output_dir=variant_dir / "results",
                include_delayed_prefix=include_delayed_prefix,
                resume=resume,
            )
            _atomic_json(variant_dir / "summary.json", result, overwrite=resume)
        evaluations[variant] = result
    baseline_event_mae = float(evaluations["baseline"]["metrics"]["event_mae"])
    candidate_event_mae = float(evaluations["candidate"]["metrics"]["event_mae"])
    candidate_gate = {
        "passed": candidate_event_mae < INTERNAL_EVENT_MAE_MAXIMUM,
        "event_mae": candidate_event_mae,
        "maximum_exclusive": INTERNAL_EVENT_MAE_MAXIMUM,
    }
    frozen_test_diagnostic = {
        "baseline_event_mae": baseline_event_mae,
        "candidate_event_mae": candidate_event_mae,
        "candidate_minus_baseline": candidate_event_mae - baseline_event_mae,
        "candidate_improved": candidate_event_mae < baseline_event_mae,
        "used_for_selection_or_gate": False,
    }
    status = "complete" if candidate_gate["passed"] else "candidate_internal_gate_failed"
    summary = {
        "stage": "internal",
        "status": status,
        "created_at_utc": utc_now_iso(),
        "train_summary": _artifact(
            _stage_dir(output_root, "train") / "summary.json"
        ),
        "validation_gate": validation_gate,
        "candidate_gate": candidate_gate,
        "frozen_test_diagnostic": frozen_test_diagnostic,
        "test_evaluated": True,
        "external_evaluated": False,
        "variants": evaluations,
    }
    _finish_stage(stage_dir, summary, resume=resume)
    return summary


def _external_event_dirs(event_root: Path) -> list[Path]:
    paths = [event_root / name for name in EXTERNAL_EVENT_NAMES]
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"fixed external events are missing: {missing}")
    return paths


def _external_input_hashes(
    event_dirs: Sequence[Path],
    label_manifest: Path,
) -> dict[str, str]:
    inputs = [label_manifest]
    for event_dir in event_dirs:
        inputs.extend(
            event_dir / name
            for name in ("event.json", "stations.csv", "waveforms.csv.gz")
        )
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"external input is missing: {missing}")
    return {str(path.resolve()): sha256_file(path) for path in inputs}


def _evaluate_external_threshold(
    *,
    model_dir: Path,
    event_dirs: Sequence[Path],
    label_rows: Sequence[Mapping[str, Any]],
    threshold_cm: float,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    from src.evaluation.evaluate_unseen import evaluate_unseen_events

    result = evaluate_unseen_events(
        event_dirs=[str(path) for path in event_dirs],
        model_dir=model_dir,
        output_dir=output_dir / "raw",
        radial_peak_min_cm_override=threshold_cm,
        save_plots=False,
    )
    station_rows = [
        {
            **row,
            "observation_horizon_sec": FULL_OBSERVATION_HORIZON_SEC,
            "release_time_sec": FULL_RELEASE_TIME_SEC,
        }
        for row in pair_prediction_rows(
            result["station_rows"],
            label_rows,
            prediction_key="mw_pred",
            old_reference_key="mw_catalog",
        )
    ]
    event_rows = [
        {
            **row,
            "observation_horizon_sec": FULL_OBSERVATION_HORIZON_SEC,
            "release_time_sec": FULL_RELEASE_TIME_SEC,
        }
        for row in pair_prediction_rows(
            result["event_rows"],
            label_rows,
            prediction_key="mw_pred_median",
            old_reference_key="mw_catalog",
        )
    ]
    if not station_rows or not event_rows:
        raise ValueError(f"external threshold {threshold_cm:g} produced no predictions")
    station_path = output_dir / "station_predictions_usgs.csv"
    event_path = output_dir / "event_predictions_usgs.csv"
    _atomic_write(station_path, _csv_bytes(station_rows), overwrite=resume)
    _atomic_write(event_path, _csv_bytes(event_rows), overwrite=resume)
    station_metrics = summarize_paired_rows(
        station_rows,
        prediction_key="mw_pred",
    )["selected"]
    event_metrics = summarize_paired_rows(
        event_rows,
        prediction_key="mw_pred_median",
    )["selected"]
    summary = {
        "threshold_cm": threshold_cm,
        "observation_horizon_sec": FULL_OBSERVATION_HORIZON_SEC,
        "release_time_sec": FULL_RELEASE_TIME_SEC,
        "station_metrics": station_metrics,
        "event_metrics": event_metrics,
        "station_predictions": _artifact(station_path),
        "event_predictions": _artifact(event_path),
        "raw_station_predictions": _artifact(result["station_csv"]),
        "raw_event_predictions": _artifact(result["event_csv"]),
    }
    _atomic_json(output_dir / "summary.json", summary, overwrite=resume)
    return summary


def _load_completed_threshold(path: Path) -> dict[str, Any] | None:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        return None
    summary = _load_json(summary_path)
    try:
        for name in (
            "station_predictions",
            "event_predictions",
            "raw_station_predictions",
            "raw_event_predictions",
        ):
            _validate_artifact(summary[name], label=name)
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return summary


def run_external(
    *,
    output_root: Path,
    event_root: Path,
    resume: bool,
) -> dict[str, Any]:
    stage_dir = _stage_dir(output_root, "external")
    completed = _prepare_stage(stage_dir, resume=resume)
    if completed is not None:
        return completed
    internal = _require_stage(output_root, "internal")
    if internal.get("status") != "complete" or not bool(
        internal.get("candidate_gate", {}).get("passed")
    ):
        raise RuntimeError("external stage requires a passed internal candidate gate")
    train_summary = _require_stage(output_root, "train")
    candidate_seed = _selected_seed_summary(train_summary, "candidate")
    if int(candidate_seed["seed"]) != int(
        internal["variants"]["candidate"]["selected_seed"]
    ):
        raise ValueError("internal and train selected candidate seeds differ")
    checkpoint = _validate_artifact(candidate_seed["checkpoint"], label="candidate checkpoint")
    _validate_artifact(candidate_seed["config"], label="candidate config")
    model_dir = checkpoint.parent

    preflight = _require_stage(output_root, "preflight")
    source_path = _validate_artifact(preflight["source_data"], label="source NPZ")
    label_manifest = source_path.parent / "external_magnitude_labels.csv"
    label_rows = _read_csv(label_manifest)
    if len(label_rows) != len(EXTERNAL_EVENT_NAMES):
        raise ValueError("external label manifest must contain exactly eight events")
    event_dirs = _external_event_dirs(event_root.resolve())
    input_hashes = _external_input_hashes(event_dirs, label_manifest)

    threshold_summaries: dict[str, Any] = {}
    for threshold in (0.0, 1.0, 2.0):
        label = f"cm{int(threshold)}"
        threshold_dir = stage_dir / label
        result = _load_completed_threshold(threshold_dir) if resume else None
        if result is None:
            result = _evaluate_external_threshold(
                model_dir=model_dir,
                event_dirs=event_dirs,
                label_rows=label_rows,
                threshold_cm=threshold,
                output_dir=threshold_dir,
                resume=resume,
            )
        threshold_summaries[label] = result
    cm0_event_count = int(threshold_summaries["cm0"]["event_metrics"]["count"])
    coverage_passed = cm0_event_count == len(EXTERNAL_EVENT_NAMES)
    summary = {
        "stage": "external",
        "status": "complete" if coverage_passed else "cm0_coverage_gate_failed",
        "created_at_utc": utc_now_iso(),
        "internal_summary": _artifact(
            _stage_dir(output_root, "internal") / "summary.json"
        ),
        "selected_variant": "candidate",
        "selected_seed": int(candidate_seed["seed"]),
        "ensemble_used": False,
        "observation_horizon_sec": FULL_OBSERVATION_HORIZON_SEC,
        "release_time_sec": FULL_RELEASE_TIME_SEC,
        "input_sha256": input_hashes,
        "label_manifest": _artifact(label_manifest),
        "cm0_coverage_gate": {
            "passed": coverage_passed,
            "event_count": cm0_event_count,
            "required_event_count": len(EXTERNAL_EVENT_NAMES),
        },
        "thresholds": threshold_summaries,
    }
    _finish_stage(stage_dir, summary, resume=resume)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the staged manuscript-aligned single-station STF campaign"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("preflight", "smoke", "train", "internal", "external"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "experiments"
        / "manuscript_station_stf_usgs.yaml",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--event-root",
        type=Path,
        default=Path(
            os.environ.get(
                "PINN_EXTERNAL_EVENT_ROOT",
                "/home/lihe/PINN_Mag/incoming/legacy-task17/GNSS_EQDATA",
            )
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root.resolve()
    if args.stage == "preflight":
        summary = run_preflight(
            config_path=args.config,
            output_root=output_root,
            resume=args.resume,
        )
    elif args.stage == "smoke":
        summary = run_smoke(output_root=output_root, resume=args.resume)
    elif args.stage == "train":
        summary = run_train(output_root=output_root, resume=args.resume)
    elif args.stage == "internal":
        summary = run_internal(output_root=output_root, resume=args.resume)
    else:
        summary = run_external(
            output_root=output_root,
            event_root=args.event_root,
            resume=args.resume,
        )
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "status": summary["status"],
                "output_root": str(output_root),
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
