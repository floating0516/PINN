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
from src.data.splits import (  # noqa: E402
    INVERSE_COUNT_FULL_DATA_ESTIMATOR,
    REPLACEMENT_SAMPLING_ESTIMATOR,
    make_event_inverse_count_weights,
    resolve_event_balance_exponent,
    resolve_event_balance_estimator,
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
from src.utils.config_v2 import (  # noqa: E402
    magnitude_penalty_from_config,
    moment_head_dropout_from_config,
    moment_linear_skip_from_config,
    validate_config_v2,
)
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
EVENT_BALANCE_ESTIMATOR_AXIS = "event_balance_estimator"
MAGNITUDE_PENALTY_AXIS = "magnitude_penalty"
EVENT_BALANCE_EXPONENT_AXIS = "event_balance_exponent"
MOMENT_LINEAR_SKIP_AXIS = "moment_linear_skip"
MOMENT_HEAD_DROPOUT_AXIS = "moment_head_dropout"
LOSS_WEIGHT_PROFILE_AXIS = "loss_weight_profile"
STATION_UNIFORM_ESTIMATOR = "station_uniform"
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
    EVENT_BALANCE_ESTIMATOR_AXIS: {
        "baseline": REPLACEMENT_SAMPLING_ESTIMATOR,
        "candidate": INVERSE_COUNT_FULL_DATA_ESTIMATOR,
    },
    MAGNITUDE_PENALTY_AXIS: {
        "baseline": "squared",
        "candidate": "absolute",
    },
    EVENT_BALANCE_EXPONENT_AXIS: {
        "baseline": 1.0,
        "candidate": 0.5,
    },
    MOMENT_LINEAR_SKIP_AXIS: {
        "baseline": False,
        "candidate": True,
    },
    MOMENT_HEAD_DROPOUT_AXIS: {
        "baseline": True,
        "candidate": False,
    },
}
VARIANT_AXIS_PATHS = {
    STF_OUTPUT_PARAMETERIZATION_AXIS: ("model", "stf_output_parameterization"),
    SCHEDULER_T0_AXIS: ("training", "scheduler_T0"),
    RADIAL_DYNAMIC_RANGE_STEM_AXIS: ("model", "radial_dynamic_range_stem"),
    EVENT_BALANCED_SAMPLING_AXIS: ("training", "event_balanced_sampling"),
    EVENT_BALANCE_ESTIMATOR_AXIS: ("training", "event_balance_estimator"),
    MAGNITUDE_PENALTY_AXIS: (
        "training",
        "stf_rate_loss",
        "magnitude_penalty",
    ),
    EVENT_BALANCE_EXPONENT_AXIS: ("training", "event_balance_exponent"),
    MOMENT_LINEAR_SKIP_AXIS: ("model", "moment_linear_skip"),
    MOMENT_HEAD_DROPOUT_AXIS: ("model", "moment_head_dropout"),
}
LOSS_WEIGHT_KEYS = (
    "lambda_MSE",
    "lambda_synth",
    "lambda_mag",
    "lambda_shape",
)
LOSS_WEIGHT_PATHS = {
    name: ("training", "stf_rate_loss", name) for name in LOSS_WEIGHT_KEYS
}
LOSS_WEIGHT_PROFILES = {
    "baseline": {
        "lambda_MSE": 1.0,
        "lambda_synth": 0.5,
        "lambda_mag": 1.0,
        "lambda_shape": 0.1,
    },
    "w01": {
        "lambda_MSE": 0.822781998,
        "lambda_synth": 2.937882947,
        "lambda_mag": 1.075052702,
        "lambda_shape": 44.807934578,
    },
    "w10": {
        "lambda_MSE": 1.068322822,
        "lambda_synth": 5.018296600,
        "lambda_mag": 0.866201862,
        "lambda_shape": 52.112544857,
    },
}
LOSS_WEIGHT_GENERATION_SEED = 20260725
LOSS_WEIGHT_PROPOSAL_POOL_SIZE = 12
LOSS_WEIGHT_PROPOSAL_POOL_SHA256 = (
    "7d35115e92b76540572fe2e5f8ea1e3caec288a979d66a6f7937f3b00ca68dcc"
)
LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256 = (
    "2235d225dd480f675ab3bdd4b044fae420f4420a9a0f40c26fc70bfd9399df64"
)
# Backward-compatible alias for the Phase23 campaign and its persisted tests.
VARIANTS = VARIANT_AXES[STF_OUTPUT_PARAMETERIZATION_AXIS]
VALIDATION_METRIC = "validation_event_mae_catalog"
EXPECTED_SOURCE_SHA256 = (
    "2e1fa4c12fc1eb03ffc8bf9235491f0886d4b0d360ebcb3486baca4d948cfd6a"
)
EXPECTED_EVENT_COUNT = 31
EXPECTED_STATION_COUNT = 2558
EXPECTED_SPLIT_COUNTS = (1788, 385, 385)
EXPECTED_TRAIN_EVENT_STATION_COUNT_MINIMUM = 1
EXPECTED_TRAIN_EVENT_STATION_COUNT_MAXIMUM = 482
EXPECTED_SPLIT_SHA256 = {
    17: "fa5c5d1cd3bdb3e8a775140a9bea4adce885b151eff717627d5e8ab82fd4e9a8",
    42: "5ac2e07ed186dce737a3592694632775b7bbf603bf922a4a74fa6b86a3d5c240",
    73: "786d029482fc8c6c8000b939380f7d4f6fdab2cc0dfe39a09fa982d1d9049548",
}
PHASE27_INCUMBENT_VALIDATION_BY_SEED = {
    17: 0.11135358810424804,
    42: 0.1320822874704997,
    73: 0.20192739168802898,
}
PHASE27_INCUMBENT_CHECKPOINT_SHA256_BY_SEED = {
    17: "c7d50f3d5ecfa9418f33743209a8e390431545047ca97539c6155c829ab94805",
    42: "1f596c161c9497961d5e3af3f903945b66a33097479edf6257e92b71980c8a91",
    73: "6e0587a8392660fe06d0c6b36343a1cb6fb274ac88b3fbe5b73a0c286399480f",
}
PHASE27_INCUMBENT_OBJECTIVE_WEIGHT_SHA256_BY_SEED = {
    17: "3b365d2ffaa4b31da6802109d2948e8238aa9944e2321a86eea8d57f0e935f2d",
    42: "63fc5b066b74950a04c92aeb802504dcd35c80f45f453f34ea554ebdafc9ff80",
    73: "e74873eae50a4445ba8738ce017213fb25da7fd86713e8ab1e20c7216aa2aeec",
}
PHASE27_INCUMBENT_SAMPLE_WEIGHT_SHA256_BY_SEED = {
    17: "c556a533e2a6c96888f5cb3927aab4c48af5575746a1d5c14ca8123f38ec8f9d",
    42: "23c634370084be979caee0d28329b6c4be38c97dc2d1c049fc3fdc582bafc9e5",
    73: "eab66f0502dc9632c8ccf9cad01e2f2f8fdbc30194442b691f8a341e116d6acd",
}
PHASE27_INCUMBENT_SELECTED_SEED = 17
PHASE27_INCUMBENT_VALIDATION = PHASE27_INCUMBENT_VALIDATION_BY_SEED[
    PHASE27_INCUMBENT_SELECTED_SEED
]
FROZEN_PHASE27_INCUMBENT_AXES = frozenset(
    {
        EVENT_BALANCE_EXPONENT_AXIS,
        MOMENT_LINEAR_SKIP_AXIS,
        MOMENT_HEAD_DROPOUT_AXIS,
        LOSS_WEIGHT_PROFILE_AXIS,
    }
)
PHASE30_PARAMETER_COUNT_BY_VARIANT = {
    "baseline": 1_010_850,
    "candidate": 1_010_978,
}
PHASE31_PARAMETER_COUNT_BY_VARIANT = {
    "baseline": 1_010_850,
    "candidate": 1_010_850,
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
PHASE27_INCUMBENT_TEST_EVENT_MAE = 0.1372873624165853
FULL_OBSERVATION_HORIZON_SEC = 200.0
FULL_RELEASE_TIME_SEC = (
    FULL_OBSERVATION_HORIZON_SEC + MANUSCRIPT_PROCESSING_DELAY_SEC
)
FORMAL_FIR_LOOKAHEAD_SEC = 3.0
FORMAL_MIN_VALID_FRACTION = 0.99
EXTERNAL_WAVEFORM_GRID_SCHEMA_VERSION = 2


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


def _compact_canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _candidate_profile_sha256(profiles: Mapping[str, Any]) -> str:
    candidates = {
        name: profiles[name]
        for name in sorted(set(profiles) - {"baseline"})
    }
    return hashlib.sha256(_compact_canonical_json_bytes(candidates)).hexdigest()


def loss_weights_from_config(config: Mapping[str, Any]) -> dict[str, float]:
    try:
        loss_config = config["training"]["stf_rate_loss"]
    except (KeyError, TypeError) as error:
        raise ValueError("formal config is missing training.stf_rate_loss") from error
    if not isinstance(loss_config, Mapping):
        raise ValueError("formal config is missing training.stf_rate_loss")
    weights: dict[str, float] = {}
    for name in LOSS_WEIGHT_KEYS:
        value = loss_config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"loss weight {name} must be a finite positive number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"loss weight {name} must be a finite positive number")
        weights[name] = numeric
    return weights


def loss_weight_profiles_from_config(
    config: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    campaign = config.get("campaign")
    if not isinstance(campaign, Mapping):
        raise ValueError("loss-weight campaign marker must be a mapping")
    search = campaign.get("loss_weight_search")
    if not isinstance(search, Mapping):
        raise ValueError("loss-weight campaign is missing loss_weight_search")
    expected_keys = {
        "generation_method",
        "generation_seed",
        "proposal_pool_size",
        "selection_rule",
        "proposal_pool_sha256",
        "candidate_profile_sha256",
        "profiles",
    }
    if set(search) != expected_keys:
        raise ValueError("loss-weight search metadata schema changed")
    expected_metadata = {
        "generation_method": "fixed 12-point train-only log-Latin-hypercube pool",
        "generation_seed": LOSS_WEIGHT_GENERATION_SEED,
        "proposal_pool_size": LOSS_WEIGHT_PROPOSAL_POOL_SIZE,
        "selection_rule": (
            "w01 is the best train-gradient proxy; w10 is the first "
            "higher-ranked profile passing the preregistered diversity check"
        ),
        "proposal_pool_sha256": LOSS_WEIGHT_PROPOSAL_POOL_SHA256,
        "candidate_profile_sha256": LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256,
    }
    for name, expected in expected_metadata.items():
        value = search.get(name)
        if value != expected or type(value) is not type(expected):
            raise ValueError(
                f"loss-weight search {name} changed: expected={expected!r}, "
                f"actual={value!r}"
            )
    raw_profiles = search.get("profiles")
    if not isinstance(raw_profiles, Mapping) or set(raw_profiles) != set(
        LOSS_WEIGHT_PROFILES
    ):
        raise ValueError("loss-weight profile identities changed")
    profiles: dict[str, dict[str, float]] = {}
    for profile_id in LOSS_WEIGHT_PROFILES:
        raw_weights = raw_profiles.get(profile_id)
        if not isinstance(raw_weights, Mapping) or set(raw_weights) != set(
            LOSS_WEIGHT_KEYS
        ):
            raise ValueError(f"loss-weight profile {profile_id} schema changed")
        weights: dict[str, float] = {}
        for name in LOSS_WEIGHT_KEYS:
            value = raw_weights.get(name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"loss-weight profile {profile_id}.{name} must be finite and positive"
                )
            expected = LOSS_WEIGHT_PROFILES[profile_id][name]
            if value != expected:
                raise ValueError(
                    f"loss-weight profile {profile_id}.{name} changed: "
                    f"expected={expected!r}, actual={value!r}"
                )
            weights[name] = value
        profiles[profile_id] = weights
    actual_hash = _candidate_profile_sha256(profiles)
    if actual_hash != LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256:
        raise ValueError(
            "loss-weight candidate profile hash changed: "
            f"expected={LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256}, actual={actual_hash}"
        )
    return profiles


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
    moment_skip_is_explicit = "moment_linear_skip" in model
    moment_dropout_is_explicit = "moment_head_dropout" in model
    estimator_is_explicit = "event_balance_estimator" in training
    exponent_is_explicit = "event_balance_exponent" in training
    loss_config = training.get("stf_rate_loss")
    if not isinstance(loss_config, Mapping):
        raise ValueError("formal config is missing training.stf_rate_loss")
    magnitude_penalty_is_explicit = "magnitude_penalty" in loss_config
    campaign = base_config.get("campaign", {})
    if not isinstance(campaign, Mapping):
        raise ValueError("formal config campaign marker must be a mapping")
    explicit_axis = campaign.get("variant_axis")
    if explicit_axis is not None:
        if explicit_axis == EVENT_BALANCED_SAMPLING_AXIS:
            if (
                parameterization != "moment_shape_factorized"
                or scheduler_t0 != 15
                or stem_is_explicit
                or moment_skip_is_explicit
                or moment_dropout_is_explicit
                or estimator_is_explicit
                or exponent_is_explicit
                or magnitude_penalty_is_explicit
                or training.get("event_balanced_sampling") is not False
            ):
                raise ValueError(
                    "Phase26 event-balanced baseline requires factorized STF, "
                    "scheduler_T0=15, the original radial stem, no explicit "
                    "event-balance estimator, and event_balanced_sampling=False"
                )
            return EVENT_BALANCED_SAMPLING_AXIS
        if explicit_axis == EVENT_BALANCE_ESTIMATOR_AXIS:
            if (
                parameterization != "moment_shape_factorized"
                or scheduler_t0 != 15
                or stem_is_explicit
                or moment_skip_is_explicit
                or moment_dropout_is_explicit
                or exponent_is_explicit
                or magnitude_penalty_is_explicit
                or training.get("event_balanced_sampling") is not True
                or training.get("event_balance_estimator")
                != REPLACEMENT_SAMPLING_ESTIMATOR
            ):
                raise ValueError(
                    "Phase27 full-data objective baseline requires factorized "
                    "STF, scheduler_T0=15, the original radial stem, "
                    "event_balanced_sampling=true, and "
                    "event_balance_estimator='replacement_sampling'"
                )
            return EVENT_BALANCE_ESTIMATOR_AXIS
        if explicit_axis == MAGNITUDE_PENALTY_AXIS:
            if (
                parameterization != "moment_shape_factorized"
                or scheduler_t0 != 15
                or stem_is_explicit
                or moment_skip_is_explicit
                or moment_dropout_is_explicit
                or exponent_is_explicit
                or training.get("event_balanced_sampling") is not True
                or training.get("event_balance_estimator")
                != INVERSE_COUNT_FULL_DATA_ESTIMATOR
                or loss_config.get("magnitude_penalty") != "squared"
            ):
                raise ValueError(
                    "Phase28 magnitude-penalty baseline requires factorized "
                    "STF, scheduler_T0=15, the original radial stem, "
                    "event_balanced_sampling=true, "
                    "event_balance_estimator='inverse_count_full_data', and "
                    "magnitude_penalty='squared'"
                )
            return MAGNITUDE_PENALTY_AXIS
        if explicit_axis == EVENT_BALANCE_EXPONENT_AXIS:
            if (
                parameterization != "moment_shape_factorized"
                or scheduler_t0 != 15
                or stem_is_explicit
                or moment_skip_is_explicit
                or moment_dropout_is_explicit
                or training.get("event_balanced_sampling") is not True
                or training.get("event_balance_estimator")
                != INVERSE_COUNT_FULL_DATA_ESTIMATOR
                or not exponent_is_explicit
                or training.get("event_balance_exponent") != 1.0
                or magnitude_penalty_is_explicit
                or loss_config.get("magnitude_penalty", "squared") != "squared"
            ):
                raise ValueError(
                    "Phase29 tempered inverse-count baseline requires factorized "
                    "STF, scheduler_T0=15, the original radial stem, "
                    "event_balanced_sampling=true, "
                    "event_balance_estimator='inverse_count_full_data', "
                    "event_balance_exponent=1.0, and the default squared "
                    "magnitude penalty"
                )
            return EVENT_BALANCE_EXPONENT_AXIS
        if explicit_axis == MOMENT_LINEAR_SKIP_AXIS:
            if (
                parameterization != "moment_shape_factorized"
                or scheduler_t0 != 15
                or stem_is_explicit
                or not moment_skip_is_explicit
                or model.get("moment_linear_skip") is not False
                or moment_dropout_is_explicit
                or training.get("event_balanced_sampling") is not True
                or training.get("event_balance_estimator")
                != INVERSE_COUNT_FULL_DATA_ESTIMATOR
                or exponent_is_explicit
                or magnitude_penalty_is_explicit
                or loss_config.get("magnitude_penalty", "squared") != "squared"
            ):
                raise ValueError(
                    "Phase30 moment-linear-skip baseline requires factorized "
                    "STF, scheduler_T0=15, the original radial stem, "
                    "event_balanced_sampling=true, "
                    "event_balance_estimator='inverse_count_full_data', the "
                    "default p=1 and squared magnitude penalty, and "
                    "model.moment_linear_skip=false"
                )
            return MOMENT_LINEAR_SKIP_AXIS
        if explicit_axis == MOMENT_HEAD_DROPOUT_AXIS:
            if (
                parameterization != "moment_shape_factorized"
                or scheduler_t0 != 15
                or stem_is_explicit
                or moment_skip_is_explicit
                or not moment_dropout_is_explicit
                or model.get("moment_head_dropout") is not True
                or training.get("event_balanced_sampling") is not True
                or training.get("event_balance_estimator")
                != INVERSE_COUNT_FULL_DATA_ESTIMATOR
                or exponent_is_explicit
                or magnitude_penalty_is_explicit
                or loss_config.get("magnitude_penalty", "squared") != "squared"
            ):
                raise ValueError(
                    "Phase31 moment-head-dropout baseline requires factorized "
                    "STF, scheduler_T0=15, the original radial stem, "
                    "event_balanced_sampling=true, "
                    "event_balance_estimator='inverse_count_full_data', the "
                    "default p=1 and squared magnitude penalty, no explicit "
                    "moment-linear skip, and model.moment_head_dropout=true"
                )
            return MOMENT_HEAD_DROPOUT_AXIS
        if explicit_axis == LOSS_WEIGHT_PROFILE_AXIS:
            if (
                parameterization != "moment_shape_factorized"
                or scheduler_t0 != 15
                or stem_is_explicit
                or moment_skip_is_explicit
                or moment_dropout_is_explicit
                or training.get("event_balanced_sampling") is not True
                or training.get("event_balance_estimator")
                != INVERSE_COUNT_FULL_DATA_ESTIMATOR
                or exponent_is_explicit
                or magnitude_penalty_is_explicit
                or loss_config.get("magnitude_penalty", "squared") != "squared"
            ):
                raise ValueError(
                    "Phase32 loss-weight baseline requires the frozen Phase27 "
                    "factorized STF candidate with scheduler_T0=15, the original "
                    "radial stem, inverse-count full-data event balancing, p=1, "
                    "and squared magnitude penalty"
                )
            profiles = loss_weight_profiles_from_config(base_config)
            if loss_weights_from_config(base_config) not in profiles.values():
                raise ValueError(
                    "Phase32 active loss weights do not match a frozen profile"
                )
            return LOSS_WEIGHT_PROFILE_AXIS
        else:
            raise ValueError(f"unsupported formal campaign axis: {explicit_axis!r}")
    if (
        parameterization == "direct"
        and scheduler_t0 == 15
        and not stem_is_explicit
        and not moment_skip_is_explicit
        and not moment_dropout_is_explicit
        and not estimator_is_explicit
        and not exponent_is_explicit
        and not magnitude_penalty_is_explicit
    ):
        return STF_OUTPUT_PARAMETERIZATION_AXIS
    if (
        parameterization == "moment_shape_factorized"
        and scheduler_t0 == 15
        and not moment_skip_is_explicit
        and not moment_dropout_is_explicit
        and not estimator_is_explicit
        and not exponent_is_explicit
        and not magnitude_penalty_is_explicit
    ):
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
        "marked Phase26/Phase27 event-balance baselines or Phase28 magnitude "
        "penalty baseline or Phase29 event-balance exponent baseline or "
        "Phase30 moment-linear-skip baseline or Phase31 moment-head-dropout "
        "baseline or Phase32 loss-weight-profile baseline"
    )


def build_variant_configs(
    base_config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    axis = variant_axis_from_config(base_config)
    if axis == LOSS_WEIGHT_PROFILE_AXIS:
        profiles = loss_weight_profiles_from_config(base_config)
        variants: dict[str, dict[str, Any]] = {}
        for profile_id, weights in profiles.items():
            config = copy.deepcopy(dict(base_config))
            loss_config = config["training"]["stf_rate_loss"]
            for name, value in weights.items():
                loss_config[name] = value
            variants[profile_id] = config
        for profile_id, config in variants.items():
            differences = _config_diff_paths(variants["baseline"], config)
            expected = (
                set()
                if profile_id == "baseline"
                else {".".join(LOSS_WEIGHT_PATHS[name]) for name in LOSS_WEIGHT_KEYS}
            )
            if differences != expected:
                raise ValueError(
                    f"loss-weight profile {profile_id} scientific diff changed: "
                    f"expected={sorted(expected)}, actual={sorted(differences)}"
                )
        return variants
    variants: dict[str, dict[str, Any]] = {}
    for name, value in VARIANT_AXES[axis].items():
        config = copy.deepcopy(dict(base_config))
        path = VARIANT_AXIS_PATHS[axis]
        node: Any = config
        for key in path[:-1]:
            if not isinstance(node, dict) or not isinstance(node.get(key), dict):
                raise ValueError(
                    f"formal config is missing {'.'.join(path[:-1])}"
                )
            node = node[key]
        node[path[-1]] = value
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
    variant_axis = variant_axis_from_config(config)
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
        ("training", "early_stop_metric"): "event_mae_catalog",
        ("training", "checkpoint_metric"): "event_mae_catalog",
        ("training", "early_stop_min_delta"): 0.0,
        ("training", "grad_clip_norm"): 1.0,
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
    _require_exact(
        config,
        ("training", "event_balanced_sampling"),
        variant_axis
        in {
            EVENT_BALANCE_ESTIMATOR_AXIS,
            MAGNITUDE_PENALTY_AXIS,
            EVENT_BALANCE_EXPONENT_AXIS,
            MOMENT_LINEAR_SKIP_AXIS,
            MOMENT_HEAD_DROPOUT_AXIS,
            LOSS_WEIGHT_PROFILE_AXIS,
        },
    )
    variants = build_variant_configs(config)
    if variant_axis == EVENT_BALANCE_EXPONENT_AXIS:
        phase27_config = _load_yaml(
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / "manuscript_station_stf_usgs_event_loss_weighted.yaml"
        )
        phase27_incumbent = build_variant_configs(phase27_config)["candidate"]
        phase29_baseline = copy.deepcopy(variants["baseline"])
        phase27_incumbent.pop("campaign")
        phase29_baseline.pop("campaign")
        phase29_baseline["training"].pop("event_balance_exponent")
        if phase29_baseline != phase27_incumbent:
            differences = _config_diff_paths(
                phase27_incumbent,
                phase29_baseline,
            )
            raise ValueError(
                "Phase29 p=1 baseline differs from the frozen Phase27 "
                f"candidate: {sorted(differences)}"
            )
    if variant_axis == MOMENT_LINEAR_SKIP_AXIS:
        phase27_config = _load_yaml(
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / "manuscript_station_stf_usgs_event_loss_weighted.yaml"
        )
        phase27_incumbent = build_variant_configs(phase27_config)["candidate"]
        phase30_baseline = copy.deepcopy(variants["baseline"])
        phase27_incumbent.pop("campaign")
        phase30_baseline.pop("campaign")
        phase30_baseline["model"].pop("moment_linear_skip")
        if phase30_baseline != phase27_incumbent:
            differences = _config_diff_paths(
                phase27_incumbent,
                phase30_baseline,
            )
            raise ValueError(
                "Phase30 baseline differs from the frozen Phase27 candidate: "
                f"{sorted(differences)}"
            )
    if variant_axis == MOMENT_HEAD_DROPOUT_AXIS:
        phase27_config = _load_yaml(
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / "manuscript_station_stf_usgs_event_loss_weighted.yaml"
        )
        phase27_incumbent = build_variant_configs(phase27_config)["candidate"]
        phase31_baseline = copy.deepcopy(variants["baseline"])
        phase27_incumbent.pop("campaign")
        phase31_baseline.pop("campaign")
        phase31_baseline["model"].pop("moment_head_dropout")
        if phase31_baseline != phase27_incumbent:
            differences = _config_diff_paths(
                phase27_incumbent,
                phase31_baseline,
            )
            raise ValueError(
                "Phase31 baseline differs from the frozen Phase27 candidate: "
                f"{sorted(differences)}"
            )
    if variant_axis == LOSS_WEIGHT_PROFILE_AXIS:
        phase27_config = _load_yaml(
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / "manuscript_station_stf_usgs_event_loss_weighted.yaml"
        )
        phase27_incumbent = build_variant_configs(phase27_config)["candidate"]
        phase32_baseline = copy.deepcopy(variants["baseline"])
        phase27_incumbent.pop("campaign")
        phase32_baseline.pop("campaign")
        if phase32_baseline != phase27_incumbent:
            differences = _config_diff_paths(
                phase27_incumbent,
                phase32_baseline,
            )
            raise ValueError(
                "Phase32 baseline differs from the frozen Phase27 candidate: "
                f"{sorted(differences)}"
            )


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


def event_balance_estimator_from_config(config: Mapping[str, Any]) -> str:
    training = config["training"]
    if not bool(training["event_balanced_sampling"]):
        return STATION_UNIFORM_ESTIMATOR
    return resolve_event_balance_estimator(training)


def event_balance_exponent_from_config(config: Mapping[str, Any]) -> float:
    return resolve_event_balance_exponent(config["training"])


def magnitude_penalty_from_formal_config(config: Mapping[str, Any]) -> str:
    return magnitude_penalty_from_config(dict(config))


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
        or min(counts.values()) != EXPECTED_TRAIN_EVENT_STATION_COUNT_MINIMUM
        or max(counts.values()) != EXPECTED_TRAIN_EVENT_STATION_COUNT_MAXIMUM
    ):
        raise ValueError("formal training sampling cohort changed")

    enabled = bool(config["training"]["event_balanced_sampling"])
    estimator = event_balance_estimator_from_config(config)
    exponent = event_balance_exponent_from_config(config)
    sampler = train_loader.sampler
    objective_normalization_constant = 1.0
    objective_weight_formula = "1"
    objective_event_mass_formula = "n_event"
    equal_event_objective_mass = False
    if estimator == REPLACEMENT_SAMPLING_ESTIMATOR:
        if not isinstance(sampler, WeightedRandomSampler):
            raise TypeError("event-balanced training requires WeightedRandomSampler")
        if not sampler.replacement:
            raise ValueError("event-balanced training must sample with replacement")
        sampling_weights = [float(value) for value in sampler.weights.tolist()]
        expected_weights = [1.0 / counts[event] for event in events]
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, expected in zip(
                sampling_weights,
                expected_weights,
                strict=True,
            )
        ):
            raise ValueError("event-balanced sampler weights changed")
        objective_weights = [1.0] * len(events)
        mode = "event_equal_with_replacement"
    elif estimator == INVERSE_COUNT_FULL_DATA_ESTIMATOR:
        if not isinstance(sampler, RandomSampler) or sampler.replacement:
            raise TypeError(
                "full-data event objective requires shuffled sampling without "
                "replacement"
            )
        sampling_weights = [1.0] * len(events)
        objective_weights = make_event_inverse_count_weights(
            events,
            exponent=exponent,
        )
        if exponent == 1.0:
            objective_normalization_constant = len(events) / len(counts)
            expected_loader_weights = {
                event: objective_normalization_constant / count
                for event, count in counts.items()
            }
            mode = "event_equal_inverse_count_full_data"
            objective_weight_formula = "N/(E*n_event)"
            objective_event_mass_formula = "N/E"
            equal_event_objective_mass = True
        else:
            objective_normalization_constant = len(events) / sum(
                count ** (1.0 - exponent) for count in counts.values()
            )
            expected_loader_weights = {
                event: objective_normalization_constant * count ** (-exponent)
                for event, count in counts.items()
            }
            mode = "tempered_inverse_count_full_data"
            objective_weight_formula = (
                "C*n_event^(-p), C=N/sum_event(n_event^(1-p))"
            )
            objective_event_mass_formula = "C*n_event^(1-p)"
        loader_weights = getattr(
            train_loader,
            "event_balance_weights_by_event",
            None,
        )
        if not isinstance(loader_weights, Mapping) or set(loader_weights) != set(
            expected_loader_weights
        ):
            raise ValueError("full-data loader event-weight mapping changed")
        if any(
            not math.isclose(
                float(loader_weights[event]),
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for event, expected in expected_loader_weights.items()
        ):
            raise ValueError("full-data loader event weights changed")
        loader_exponent = getattr(train_loader, "event_balance_exponent", None)
        if (
            isinstance(loader_exponent, bool)
            or not isinstance(loader_exponent, (int, float))
            or not math.isclose(
                float(loader_exponent),
                exponent,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("full-data loader event-balance exponent changed")
    elif estimator == STATION_UNIFORM_ESTIMATOR:
        if not isinstance(sampler, RandomSampler) or sampler.replacement:
            raise TypeError(
                "baseline training requires shuffled sampling without replacement"
            )
        sampling_weights = [1.0] * len(events)
        objective_weights = [1.0] * len(events)
        mode = "station_uniform_without_replacement"
    else:
        raise ValueError(f"unknown event-balance estimator: {estimator}")

    draw_count = int(sampler.num_samples)
    if draw_count != len(events):
        raise ValueError("formal sampler draw count changed")
    if sampler.generator is not train_loader.generator:
        raise ValueError("sampler and DataLoader must share their seeded generator")

    total_weight = sum(sampling_weights)
    event_weights: dict[str, float] = {event: 0.0 for event in counts}
    event_objective_weights: dict[str, float] = {
        event: 0.0 for event in counts
    }
    for event, sampling_weight, objective_weight in zip(
        events,
        sampling_weights,
        objective_weights,
        strict=True,
    ):
        event_weights[event] += sampling_weight
        event_objective_weights[event] += objective_weight
    event_draws = [
        draw_count * value / total_weight for value in event_weights.values()
    ]
    if not math.isclose(
        sum(objective_weights),
        float(len(events)),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("formal objective weights must have global mean one")
    if estimator == INVERSE_COUNT_FULL_DATA_ESTIMATOR:
        expected_event_masses = {
            event: expected_loader_weights[event] * count
            for event, count in counts.items()
        }
        if any(
            not math.isclose(
                event_objective_weights[event],
                expected_mass,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            for event, expected_mass in expected_event_masses.items()
        ):
            raise ValueError("full-data objective event masses changed")
        if equal_event_objective_mass and any(
            not math.isclose(
                mass,
                len(events) / len(counts),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            for mass in event_objective_weights.values()
        ):
            raise ValueError("full-data objective event masses are not equal")
    if estimator == REPLACEMENT_SAMPLING_ESTIMATOR:
        record_probabilities = [
            weight / total_weight for weight in sampling_weights
        ]
        expected_unique = sum(
            1.0 - (1.0 - probability) ** draw_count
            for probability in record_probabilities
        )
    else:
        expected_unique = float(draw_count)
    sampling_weight_rows = [
        {"event": event, "station": station, "weight": weight}
        for event, station, weight in zip(
            events,
            stations,
            sampling_weights,
            strict=True,
        )
    ]
    objective_weight_rows = [
        {"event": event, "station": station, "weight": weight}
        for event, station, weight in zip(
            events,
            stations,
            objective_weights,
            strict=True,
        )
    ]
    sampling_weight_sha256 = hashlib.sha256(
        _json_bytes(sampling_weight_rows)
    ).hexdigest()
    objective_weight_sha256 = hashlib.sha256(
        _json_bytes(objective_weight_rows)
    ).hexdigest()
    event_objective_masses = list(event_objective_weights.values())
    event_objective_mass_ess = sum(event_objective_masses) ** 2 / sum(
        mass**2 for mass in event_objective_masses
    )
    manifest = {
        "schema_version": (
            3 if "event_balance_exponent" in config["training"] else 2
        ),
        "mode": mode,
        "event_balanced_sampling": enabled,
        "event_balance_estimator": estimator,
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
        "expected_unique_record_fraction": expected_unique / len(events),
        "optimizer_step_count": math.ceil(draw_count / train_loader.batch_size),
        "loss_weights_applied": (
            estimator == INVERSE_COUNT_FULL_DATA_ESTIMATOR
        ),
        "objective_weight_formula": objective_weight_formula,
        "objective_reduction": (
            "mean(sample_weight * per_sample_loss)"
            if estimator == INVERSE_COUNT_FULL_DATA_ESTIMATOR
            else "existing_unweighted_reduction"
        ),
        "objective_weight_minimum": min(objective_weights),
        "objective_weight_maximum": max(objective_weights),
        "event_objective_mass_minimum": min(event_objective_weights.values()),
        "event_objective_mass_maximum": max(event_objective_weights.values()),
        "sample_weight_sha256": sampling_weight_sha256,
        "sampling_weight_sha256": sampling_weight_sha256,
        "objective_weight_sha256": objective_weight_sha256,
    }
    if "event_balance_exponent" in config["training"]:
        event_mass_minimum = min(event_objective_masses)
        event_mass_maximum = max(event_objective_masses)
        manifest.update(
            {
                "event_balance_exponent": exponent,
                "objective_normalization_constant": (
                    objective_normalization_constant
                ),
                "objective_event_mass_formula": objective_event_mass_formula,
                "equal_event_objective_mass": equal_event_objective_mass,
                "event_objective_mass_ratio": (
                    event_mass_maximum / event_mass_minimum
                ),
                "event_objective_mass_ess": event_objective_mass_ess,
            }
        )
    return manifest


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


def _formal_sample_weight_probe(
    split_manifests: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    if (
        event_balance_estimator_from_config(config)
        != INVERSE_COUNT_FULL_DATA_ESTIMATOR
    ):
        return None
    if set(split_manifests) != set(SEEDS):
        raise ValueError(f"formal weight probe requires exactly seeds {SEEDS}")
    exponent = event_balance_exponent_from_config(config)
    by_seed: dict[str, Any] = {}
    all_weights: list[float] = []
    for seed in SEEDS:
        manifest = split_manifests[seed]
        rows = manifest.get("per_event_station_counts")
        if not isinstance(rows, Mapping):
            raise ValueError(f"seed {seed} split lacks per-event station counts")
        counts = {
            str(event): int(split_counts["train"])
            for event, split_counts in rows.items()
            if isinstance(split_counts, Mapping) and int(split_counts["train"]) > 0
        }
        if (
            len(counts) != EXPECTED_EVENT_COUNT
            or sum(counts.values()) != EXPECTED_SPLIT_COUNTS[0]
            or min(counts.values()) != EXPECTED_TRAIN_EVENT_STATION_COUNT_MINIMUM
            or max(counts.values()) != EXPECTED_TRAIN_EVENT_STATION_COUNT_MAXIMUM
        ):
            raise ValueError(f"seed {seed} formal weight-probe cohort changed")
        events = [
            event
            for event, count in counts.items()
            for _ in range(count)
        ]
        weights = make_event_inverse_count_weights(events, exponent=exponent)
        minimum = min(weights)
        maximum = max(weights)
        all_weights.extend((minimum, maximum))
        by_seed[str(seed)] = {
            "minimum": minimum,
            "maximum": maximum,
        }
    return {
        "source": "frozen_train_split_event_counts",
        "event_balance_exponent": exponent,
        "minimum": min(all_weights),
        "maximum": max(all_weights),
        "by_seed": by_seed,
    }


def _smoke_one_device(
    config: dict[str, Any],
    *,
    dataset: CorrectedEarthquakeDataset,
    device: torch.device,
    sample_weight_probe: Mapping[str, Any] | None,
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
    sample_weights = None
    if (
        event_balance_estimator_from_config(config)
        == INVERSE_COUNT_FULL_DATA_ESTIMATOR
    ):
        if sample_weight_probe is None:
            raise ValueError("full-data smoke requires a formal sample-weight probe")
        sample_weights = torch.tensor(
            [
                float(sample_weight_probe["minimum"]),
                float(sample_weight_probe["maximum"]),
            ],
            device=device,
        )[:batch_size]
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
        sample_weights=sample_weights,
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
        "event_balance_estimator": event_balance_estimator_from_config(config),
        "event_balance_exponent": event_balance_exponent_from_config(config),
        "magnitude_penalty": magnitude_penalty_from_formal_config(
            config
        ),
        "moment_linear_skip": moment_linear_skip_from_config(config),
        "moment_head_dropout": moment_head_dropout_from_config(config),
        "loss_weights": loss_weights_from_config(config),
        "sample_weights_exercised": sample_weights is not None,
        "sample_weight_probe": (
            dict(sample_weight_probe) if sample_weight_probe is not None else None
        ),
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
    variant_axis = variant_axis_from_config(config)
    variants = build_variant_configs(config)
    dataset = CorrectedEarthquakeDataset(config)
    split_manifests = {
        seed: _load_json(
            _validate_artifact(
                preflight["splits"][str(seed)]["manifest"],
                label=f"preflight split seed {seed}",
            )
        )
        for seed in SEEDS
    }
    if not torch.cuda.is_available():
        raise RuntimeError("formal smoke requires both CPU and CUDA")
    results: dict[str, Any] = {}
    for variant, variant_config in variants.items():
        results[variant] = {}
        sample_weight_probe = _formal_sample_weight_probe(
            split_manifests,
            variant_config,
        )
        for device in (torch.device("cpu"), torch.device("cuda")):
            result = _smoke_one_device(
                variant_config,
                dataset=dataset,
                device=device,
                sample_weight_probe=sample_weight_probe,
            )
            if variant_axis == LOSS_WEIGHT_PROFILE_AXIS:
                result.update(
                    {
                        "loss_weight_profile": variant,
                        "loss_weight_profile_sha256": (
                            LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256
                        ),
                    }
                )
            if (
                variant_axis == MOMENT_LINEAR_SKIP_AXIS
                and int(result["parameter_count"])
                != PHASE30_PARAMETER_COUNT_BY_VARIANT[variant]
            ):
                raise ValueError(
                    "Phase30 parameter count changed for "
                    f"{variant}: expected="
                    f"{PHASE30_PARAMETER_COUNT_BY_VARIANT[variant]}, "
                    f"actual={result['parameter_count']}"
                )
            if (
                variant_axis == MOMENT_HEAD_DROPOUT_AXIS
                and int(result["parameter_count"])
                != PHASE31_PARAMETER_COUNT_BY_VARIANT[variant]
            ):
                raise ValueError(
                    "Phase31 parameter count changed for "
                    f"{variant}: expected="
                    f"{PHASE31_PARAMETER_COUNT_BY_VARIANT[variant]}, "
                    f"actual={result['parameter_count']}"
                )
            results[variant][device.type] = result
    summary = {
        "stage": "smoke",
        "status": "complete",
        "created_at_utc": utc_now_iso(),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "variant_axis": variant_axis,
        "preflight_summary": _artifact(
            _stage_dir(output_root, "preflight") / "summary.json"
        ),
        "test_evaluated": False,
        "external_evaluated": False,
        "results": results,
    }
    if variant_axis == LOSS_WEIGHT_PROFILE_AXIS:
        summary.update(
            {
                "loss_weight_profiles": loss_weight_profiles_from_config(config),
                "loss_weight_profile_sha256": (
                    LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256
                ),
            }
        )
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


def phase29_incumbent_reproduction(
    seed_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(seed_rows) != set(SEEDS):
        raise ValueError(f"Phase29 incumbent reproduction requires exactly {SEEDS}")
    rows: dict[str, Any] = {}
    passed = True
    for seed in SEEDS:
        actual_metric = float(seed_rows[seed][VALIDATION_METRIC])
        actual_checkpoint_sha256 = str(
            seed_rows[seed]["checkpoint"]["sha256"]
        )
        expected_metric = PHASE27_INCUMBENT_VALIDATION_BY_SEED[seed]
        expected_checkpoint_sha256 = (
            PHASE27_INCUMBENT_CHECKPOINT_SHA256_BY_SEED[seed]
        )
        metric_matches = math.isclose(
            actual_metric,
            expected_metric,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        checkpoint_matches = actual_checkpoint_sha256 == expected_checkpoint_sha256
        seed_passed = metric_matches and checkpoint_matches
        passed = passed and seed_passed
        rows[str(seed)] = {
            "passed": seed_passed,
            "actual_validation_event_mae_catalog": actual_metric,
            "expected_validation_event_mae_catalog": expected_metric,
            "validation_metric_exact_match": metric_matches,
            "actual_checkpoint_sha256": actual_checkpoint_sha256,
            "expected_checkpoint_sha256": expected_checkpoint_sha256,
            "checkpoint_sha256_exact_match": checkpoint_matches,
        }
    selected_seed = select_seed_by_validation(seed_rows)
    selected_seed_matches = selected_seed == PHASE27_INCUMBENT_SELECTED_SEED
    passed = passed and selected_seed_matches
    return {
        "passed": passed,
        "rule": (
            "all p=1 baseline validation metrics and checkpoint SHA-256 values "
            "must exactly reproduce the frozen Phase27 candidate"
        ),
        "selected_seed": selected_seed,
        "expected_selected_seed": PHASE27_INCUMBENT_SELECTED_SEED,
        "selected_seed_matches": selected_seed_matches,
        "seeds": rows,
    }


def select_loss_weight_profile(
    variants: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(variants) != set(LOSS_WEIGHT_PROFILES):
        raise ValueError(
            "loss-weight selection requires exactly baseline, w01, and w10"
        )
    baseline_rows = variants["baseline"].get("seeds")
    expected_seed_keys = {str(seed) for seed in SEEDS}
    if not isinstance(baseline_rows, Mapping) or set(baseline_rows) != (
        expected_seed_keys
    ):
        raise ValueError("loss-weight baseline seed summaries are missing")
    profile_rows: dict[str, Any] = {}
    ranking: list[tuple[float, str]] = []
    for profile_id in sorted(set(LOSS_WEIGHT_PROFILES) - {"baseline"}):
        candidate_rows = variants[profile_id].get("seeds")
        if not isinstance(candidate_rows, Mapping) or set(candidate_rows) != (
            expected_seed_keys
        ):
            raise ValueError(f"loss-weight {profile_id} seed summaries are missing")
        deltas: dict[str, float] = {}
        for seed in SEEDS:
            try:
                baseline_value = float(
                    baseline_rows[str(seed)][VALIDATION_METRIC]
                )
                candidate_value = float(
                    candidate_rows[str(seed)][VALIDATION_METRIC]
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"loss-weight {profile_id} validation evidence is incomplete"
                ) from error
            if not math.isfinite(baseline_value) or not math.isfinite(
                candidate_value
            ):
                raise ValueError("loss-weight validation metrics must be finite")
            deltas[str(seed)] = candidate_value - baseline_value
        ordered_deltas = sorted(deltas.values())
        median_delta = ordered_deltas[len(ordered_deltas) // 2]
        mean_delta = sum(ordered_deltas) / len(ordered_deltas)
        improved_seed_count = sum(delta < 0.0 for delta in ordered_deltas)
        selected_seed = select_seed_by_validation(
            {seed: candidate_rows[str(seed)] for seed in SEEDS}
        )
        profile_rows[profile_id] = {
            "paired_deltas": deltas,
            "paired_mean_delta": mean_delta,
            "paired_median_delta": median_delta,
            "improved_seed_count": improved_seed_count,
            "improved_seed_majority": improved_seed_count >= 2,
            "selected_seed": selected_seed,
            "selected_validation_event_mae_catalog": float(
                candidate_rows[str(selected_seed)][VALIDATION_METRIC]
            ),
        }
        ranking.append((median_delta, profile_id))
    winner_median_delta, winner_profile = min(ranking)
    winner = profile_rows[winner_profile]
    return {
        "selection_metric": VALIDATION_METRIC,
        "ranking_rule": (
            "minimum paired three-seed median delta; ties by profile id"
        ),
        "winner_profile": winner_profile,
        "winner_paired_mean_delta": winner["paired_mean_delta"],
        "winner_paired_median_delta": winner_median_delta,
        "winner_improved_seed_count": winner["improved_seed_count"],
        "winner_improved_seed_majority": winner["improved_seed_majority"],
        "profiles": profile_rows,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _phase29_sampling_manifest_is_valid(
    manifest: Mapping[str, Any],
    *,
    expected_exponent: float,
) -> bool:
    try:
        exponent = float(manifest["event_balance_exponent"])
        normalization = float(manifest["objective_normalization_constant"])
        record_count = int(manifest["record_count"])
        event_count = int(manifest["event_count"])
        count_minimum = int(manifest["event_record_count_minimum"])
        count_maximum = int(manifest["event_record_count_maximum"])
        weight_minimum = float(manifest["objective_weight_minimum"])
        weight_maximum = float(manifest["objective_weight_maximum"])
        mass_minimum = float(manifest["event_objective_mass_minimum"])
        mass_maximum = float(manifest["event_objective_mass_maximum"])
        mass_ratio = float(manifest["event_objective_mass_ratio"])
        mass_ess = float(manifest["event_objective_mass_ess"])
    except (KeyError, TypeError, ValueError):
        return False
    finite_values = (
        exponent,
        normalization,
        weight_minimum,
        weight_maximum,
        mass_minimum,
        mass_maximum,
        mass_ratio,
        mass_ess,
    )
    if not all(math.isfinite(value) for value in finite_values):
        return False
    if (
        int(manifest.get("schema_version", -1)) != 3
        or manifest.get("event_balance_estimator")
        != INVERSE_COUNT_FULL_DATA_ESTIMATOR
        or not math.isclose(
            exponent,
            expected_exponent,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or record_count != EXPECTED_SPLIT_COUNTS[0]
        or event_count != EXPECTED_EVENT_COUNT
        or count_minimum != EXPECTED_TRAIN_EVENT_STATION_COUNT_MINIMUM
        or count_maximum != EXPECTED_TRAIN_EVENT_STATION_COUNT_MAXIMUM
        or normalization <= 0.0
    ):
        return False

    event_equal = expected_exponent == 1.0
    expected_mode = (
        "event_equal_inverse_count_full_data"
        if event_equal
        else "tempered_inverse_count_full_data"
    )
    expected_weight_formula = (
        "N/(E*n_event)"
        if event_equal
        else "C*n_event^(-p), C=N/sum_event(n_event^(1-p))"
    )
    expected_mass_formula = "N/E" if event_equal else "C*n_event^(1-p)"
    if (
        manifest.get("mode") != expected_mode
        or manifest.get("objective_weight_formula") != expected_weight_formula
        or manifest.get("objective_event_mass_formula") != expected_mass_formula
        or manifest.get("equal_event_objective_mass") is not event_equal
    ):
        return False

    expected_weight_minimum = normalization * count_maximum ** (-exponent)
    expected_weight_maximum = normalization * count_minimum ** (-exponent)
    expected_mass_minimum = normalization * count_minimum ** (1.0 - exponent)
    expected_mass_maximum = normalization * count_maximum ** (1.0 - exponent)
    expected_mass_ratio = (count_maximum / count_minimum) ** (1.0 - exponent)
    analytical_values = (
        (weight_minimum, expected_weight_minimum),
        (weight_maximum, expected_weight_maximum),
        (mass_minimum, expected_mass_minimum),
        (mass_maximum, expected_mass_maximum),
        (mass_ratio, expected_mass_ratio),
    )
    if any(
        not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12)
        for actual, expected in analytical_values
    ):
        return False
    if not 0.0 < mass_ess <= event_count:
        return False
    if event_equal and not math.isclose(
        mass_ess,
        float(event_count),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        return False
    if not event_equal and not mass_ess < event_count:
        return False
    return True


def _phase30_sampling_manifest_is_valid(
    manifest: Mapping[str, Any],
    *,
    seed: int,
) -> bool:
    if seed not in SEEDS:
        return False
    normalization = EXPECTED_SPLIT_COUNTS[0] / EXPECTED_EVENT_COUNT
    expected_weight_minimum = (
        normalization / EXPECTED_TRAIN_EVENT_STATION_COUNT_MAXIMUM
    )
    try:
        numerical_values = (
            (float(manifest["expected_unique_record_count"]), 1788.0),
            (float(manifest["expected_unique_record_fraction"]), 1.0),
            (float(manifest["objective_weight_minimum"]), expected_weight_minimum),
            (float(manifest["objective_weight_maximum"]), normalization),
            (float(manifest["event_objective_mass_minimum"]), normalization),
            (float(manifest["event_objective_mass_maximum"]), normalization),
        )
    except (KeyError, TypeError, ValueError):
        return False
    if any(
        not math.isfinite(actual)
        or not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12)
        for actual, expected in numerical_values
    ):
        return False
    return (
        int(manifest.get("schema_version", -1)) == 2
        and "event_balance_exponent" not in manifest
        and manifest.get("mode") == "event_equal_inverse_count_full_data"
        and manifest.get("event_balanced_sampling") is True
        and manifest.get("event_balance_estimator")
        == INVERSE_COUNT_FULL_DATA_ESTIMATOR
        and manifest.get("sampler_class") == "RandomSampler"
        and manifest.get("replacement") is False
        and int(manifest.get("draw_count", -1)) == EXPECTED_SPLIT_COUNTS[0]
        and int(manifest.get("record_count", -1)) == EXPECTED_SPLIT_COUNTS[0]
        and int(manifest.get("event_count", -1)) == EXPECTED_EVENT_COUNT
        and int(manifest.get("event_record_count_minimum", -1))
        == EXPECTED_TRAIN_EVENT_STATION_COUNT_MINIMUM
        and int(manifest.get("event_record_count_maximum", -1))
        == EXPECTED_TRAIN_EVENT_STATION_COUNT_MAXIMUM
        and int(manifest.get("optimizer_step_count", -1)) == 28
        and manifest.get("loss_weights_applied") is True
        and manifest.get("objective_weight_formula") == "N/(E*n_event)"
        and manifest.get("objective_reduction")
        == "mean(sample_weight * per_sample_loss)"
        and manifest.get("objective_weight_sha256")
        == PHASE27_INCUMBENT_OBJECTIVE_WEIGHT_SHA256_BY_SEED[seed]
        and manifest.get("sample_weight_sha256")
        == PHASE27_INCUMBENT_SAMPLE_WEIGHT_SHA256_BY_SEED[seed]
        and manifest.get("sampling_weight_sha256")
        == PHASE27_INCUMBENT_SAMPLE_WEIGHT_SHA256_BY_SEED[seed]
    )


def _seed_summary_is_valid(
    summary: Mapping[str, Any],
    *,
    require_sampling: bool = False,
    expected_magnitude_penalty: str | None = None,
    expected_event_balance_exponent: float | None = None,
    expected_moment_linear_skip: bool | None = None,
    expected_moment_head_dropout: bool | None = None,
    expected_loss_weight_profile: str | None = None,
    expected_loss_weights: Mapping[str, float] | None = None,
    expected_loss_weight_profile_sha256: str | None = None,
    expected_variant: str | None = None,
    expected_seed: int | None = None,
    expected_split_assignment_sha256: str | None = None,
    expected_config: Mapping[str, Any] | None = None,
    expected_git_commit: str | None = None,
) -> bool:
    try:
        for name in ("checkpoint", "config", "split", "training_log", "run_manifest"):
            _validate_artifact(summary[name], label=name)
        if require_sampling or "sampling" in summary:
            _validate_artifact(summary["sampling"], label="sampling")
        if (
            expected_magnitude_penalty is not None
            and summary.get("magnitude_penalty")
            != expected_magnitude_penalty
        ):
            return False
        if (
            expected_moment_linear_skip is not None
            and summary.get("moment_linear_skip")
            is not expected_moment_linear_skip
        ):
            return False
        if (
            expected_moment_head_dropout is not None
            and summary.get("moment_head_dropout")
            is not expected_moment_head_dropout
        ):
            return False
        if (
            expected_loss_weight_profile is not None
            and summary.get("loss_weight_profile") != expected_loss_weight_profile
        ):
            return False
        if expected_loss_weights is not None:
            summary_weights = summary.get("loss_weights")
            if not isinstance(summary_weights, Mapping) or dict(
                summary_weights
            ) != dict(expected_loss_weights):
                return False
        if (
            expected_loss_weight_profile_sha256 is not None
            and summary.get("loss_weight_profile_sha256")
            != expected_loss_weight_profile_sha256
        ):
            return False
        if expected_event_balance_exponent is not None:
            summary_exponent = summary.get("event_balance_exponent")
            if (
                isinstance(summary_exponent, bool)
                or not isinstance(summary_exponent, (int, float))
                or not math.isclose(
                    float(summary_exponent),
                    expected_event_balance_exponent,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                return False
            sampling_manifest = _load_json(
                Path(str(summary["sampling"]["path"]))
            )
            if not isinstance(sampling_manifest, Mapping) or not (
                _phase29_sampling_manifest_is_valid(
                    sampling_manifest,
                    expected_exponent=expected_event_balance_exponent,
                )
            ):
                return False
        if (
            expected_moment_linear_skip is not None
            or expected_moment_head_dropout is not None
            or expected_loss_weight_profile is not None
        ):
            if expected_seed is None:
                return False
            summary_exponent = summary.get("event_balance_exponent")
            if (
                isinstance(summary_exponent, bool)
                or not isinstance(summary_exponent, (int, float))
                or float(summary_exponent) != 1.0
            ):
                return False
            sampling_manifest = _load_json(
                Path(str(summary["sampling"]["path"]))
            )
            if not isinstance(sampling_manifest, Mapping) or not (
                _phase30_sampling_manifest_is_valid(
                    sampling_manifest,
                    seed=expected_seed,
                )
            ):
                return False
        if (
            expected_variant is not None
            and summary.get("variant") != expected_variant
        ):
            return False
        if (
            expected_seed is not None
            and int(summary.get("seed")) != expected_seed
        ):
            return False
        if (
            expected_split_assignment_sha256 is not None
            and summary.get("split_assignment_sha256")
            != expected_split_assignment_sha256
        ):
            return False
        if expected_split_assignment_sha256 is not None:
            if expected_seed is None:
                return False
            persisted_split = _load_json(Path(str(summary["split"]["path"])))
            _assert_split_manifest(persisted_split, seed=expected_seed)
        if expected_config is not None:
            config_path = Path(str(summary["config"]["path"]))
            with config_path.open(encoding="utf-8") as stream:
                persisted_config = yaml.safe_load(stream)
            if persisted_config != dict(expected_config):
                return False
        if expected_git_commit is not None:
            if summary.get("git_commit") != expected_git_commit:
                return False
            run_manifest = _load_json(Path(str(summary["run_manifest"]["path"])))
            if (
                run_manifest.get("git_commit") != expected_git_commit
                or run_manifest.get("git_dirty") is not False
            ):
                return False
        return int(summary["seed"]) in SEEDS and math.isfinite(
            float(summary[VALIDATION_METRIC])
        )
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError):
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
    campaign = config.get("campaign", {})
    campaign_axis = (
        campaign.get("variant_axis") if isinstance(campaign, Mapping) else None
    )
    if seed_summary_path.is_file():
        summary = _load_json(seed_summary_path)
        require_sampling = (
            campaign_axis
            in {
                EVENT_BALANCED_SAMPLING_AXIS,
                EVENT_BALANCE_ESTIMATOR_AXIS,
                MAGNITUDE_PENALTY_AXIS,
                EVENT_BALANCE_EXPONENT_AXIS,
                MOMENT_LINEAR_SKIP_AXIS,
                MOMENT_HEAD_DROPOUT_AXIS,
                LOSS_WEIGHT_PROFILE_AXIS,
            }
        )
        expected_magnitude_penalty = (
            magnitude_penalty_from_formal_config(config)
            if campaign_axis == MAGNITUDE_PENALTY_AXIS
            else None
        )
        expected_event_balance_exponent = (
            event_balance_exponent_from_config(config)
            if campaign_axis == EVENT_BALANCE_EXPONENT_AXIS
            else None
        )
        expected_moment_linear_skip = (
            moment_linear_skip_from_config(config)
            if campaign_axis == MOMENT_LINEAR_SKIP_AXIS
            else None
        )
        expected_moment_head_dropout = (
            moment_head_dropout_from_config(config)
            if campaign_axis == MOMENT_HEAD_DROPOUT_AXIS
            else None
        )
        expected_loss_weight_profile = (
            variant if campaign_axis == LOSS_WEIGHT_PROFILE_AXIS else None
        )
        expected_loss_weights = (
            loss_weights_from_config(config)
            if campaign_axis == LOSS_WEIGHT_PROFILE_AXIS
            else None
        )
        expected_loss_weight_profile_sha256 = (
            LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256
            if campaign_axis == LOSS_WEIGHT_PROFILE_AXIS
            else None
        )
        expected_git_commit = (
            current_git_commit(PROJECT_ROOT)
            if expected_event_balance_exponent is not None
            or expected_moment_linear_skip is not None
            or expected_moment_head_dropout is not None
            or expected_loss_weight_profile is not None
            else None
        )
        strict_resume = (
            expected_magnitude_penalty is not None
            or expected_event_balance_exponent is not None
            or expected_moment_linear_skip is not None
            or expected_moment_head_dropout is not None
            or expected_loss_weight_profile is not None
        )
        expected_runtime_config = (
            _runtime_config(
                config,
                run_root=seed_root,
                seed=seed,
                dataset_manifest=dataset_manifest,
            )
            if strict_resume
            else None
        )
        if resume and _seed_summary_is_valid(
            summary,
            require_sampling=require_sampling,
            expected_magnitude_penalty=expected_magnitude_penalty,
            expected_event_balance_exponent=expected_event_balance_exponent,
            expected_moment_linear_skip=expected_moment_linear_skip,
            expected_moment_head_dropout=expected_moment_head_dropout,
            expected_loss_weight_profile=expected_loss_weight_profile,
            expected_loss_weights=expected_loss_weights,
            expected_loss_weight_profile_sha256=(
                expected_loss_weight_profile_sha256
            ),
            expected_variant=variant if strict_resume else None,
            expected_seed=seed if strict_resume else None,
            expected_split_assignment_sha256=(
                EXPECTED_SPLIT_SHA256[seed] if strict_resume else None
            ),
            expected_config=expected_runtime_config,
            expected_git_commit=expected_git_commit,
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
        "event_balance_estimator": event_balance_estimator_from_config(config),
        "event_balance_exponent": event_balance_exponent_from_config(config),
        "magnitude_penalty": magnitude_penalty_from_formal_config(
            config
        ),
        "moment_linear_skip": moment_linear_skip_from_config(config),
        "moment_head_dropout": moment_head_dropout_from_config(config),
        "loss_weights": loss_weights_from_config(config),
        "seed": seed,
        "git_commit": current_git_commit(PROJECT_ROOT),
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
    if campaign_axis == LOSS_WEIGHT_PROFILE_AXIS:
        summary.update(
            {
                "loss_weight_profile": variant,
                "loss_weight_profile_sha256": (
                    LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256
                ),
            }
        )
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
    incumbent_reproduction: dict[str, Any] | None = None
    incumbent_reproduction_artifact: dict[str, str] | None = None
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
        if (
            variant_axis in FROZEN_PHASE27_INCUMBENT_AXES
            and variant == "baseline"
        ):
            incumbent_reproduction = phase29_incumbent_reproduction(seed_summaries)
            reproduction_path = stage_dir / variant / "incumbent_reproduction.json"
            _atomic_json(
                reproduction_path,
                incumbent_reproduction,
                overwrite=resume,
            )
            incumbent_reproduction_artifact = _artifact(reproduction_path)
            if not incumbent_reproduction["passed"]:
                campaign_name = (
                    "Phase29 p=1"
                    if variant_axis == EVENT_BALANCE_EXPONENT_AXIS
                    else (
                        "Phase30 moment-linear-skip"
                        if variant_axis == MOMENT_LINEAR_SKIP_AXIS
                        else (
                            "Phase31 moment-head-dropout"
                            if variant_axis == MOMENT_HEAD_DROPOUT_AXIS
                            else "Phase32 loss-weight-profile"
                        )
                    )
                )
                raise ValueError(
                    f"{campaign_name} baseline did not exactly reproduce the frozen "
                    "Phase27 incumbent; refusing to train or compare the candidate"
                )
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
            "event_balance_estimator": event_balance_estimator_from_config(
                variant_config
            ),
            "event_balance_exponent": event_balance_exponent_from_config(
                variant_config
            ),
            "magnitude_penalty": magnitude_penalty_from_formal_config(
                variant_config
            ),
            "moment_linear_skip": moment_linear_skip_from_config(
                variant_config
            ),
            "moment_head_dropout": moment_head_dropout_from_config(
                variant_config
            ),
            "loss_weights": loss_weights_from_config(variant_config),
            "scientific_diff_from_baseline": sorted(
                _config_diff_paths(variants["baseline"], variant_config)
            ),
            "seeds": {str(seed): seed_summaries[seed] for seed in SEEDS},
            "selection": selection,
            "selection_artifact": _artifact(selection_path),
        }
        if variant_axis == LOSS_WEIGHT_PROFILE_AXIS:
            variant_summaries[variant].update(
                {
                    "loss_weight_profile": variant,
                    "loss_weight_profile_sha256": (
                        LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256
                    ),
                }
            )
    profile_selection: dict[str, Any] | None = None
    profile_selection_artifact: dict[str, str] | None = None
    if variant_axis == LOSS_WEIGHT_PROFILE_AXIS:
        profile_selection = select_loss_weight_profile(variant_summaries)
        profile_selection_path = stage_dir / "loss_weight_profile_selection.json"
        _atomic_json(
            profile_selection_path,
            profile_selection,
            overwrite=resume,
        )
        profile_selection_artifact = _artifact(profile_selection_path)
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
    if variant_axis in FROZEN_PHASE27_INCUMBENT_AXES:
        if incumbent_reproduction is None or incumbent_reproduction_artifact is None:
            raise RuntimeError("incumbent reproduction gate was not evaluated")
        summary["incumbent_reproduction"] = {
            **incumbent_reproduction,
            "artifact": incumbent_reproduction_artifact,
        }
    if variant_axis == LOSS_WEIGHT_PROFILE_AXIS:
        if profile_selection is None or profile_selection_artifact is None:
            raise RuntimeError("loss-weight profile selection was not evaluated")
        summary.update(
            {
                "loss_weight_profiles": loss_weight_profiles_from_config(config),
                "loss_weight_profile_sha256": (
                    LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256
                ),
                "loss_weight_profile_selection": {
                    **profile_selection,
                    "artifact": profile_selection_artifact,
                },
                "selected_candidate_variant": profile_selection[
                    "winner_profile"
                ],
            }
        )
    _finish_stage(stage_dir, summary, resume=resume)
    return summary


def _validated_phase27_incumbent_reproduction(
    train_summary: Mapping[str, Any],
    *,
    campaign_name: str,
) -> Mapping[str, Any]:
    variants = train_summary["variants"]
    reproduction = train_summary.get("incumbent_reproduction")
    if not isinstance(reproduction, Mapping) or reproduction.get("passed") is not True:
        raise ValueError(
            f"{campaign_name} incumbent reproduction gate is missing or failed"
        )
    reproduction_path = _validate_artifact(
        reproduction["artifact"],
        label=f"{campaign_name} incumbent reproduction",
    )
    persisted_reproduction = _load_json(reproduction_path)
    recorded_reproduction = {
        key: value for key, value in reproduction.items() if key != "artifact"
    }
    if persisted_reproduction != recorded_reproduction:
        raise ValueError(f"{campaign_name} incumbent reproduction artifact changed")
    baseline_rows = {
        seed: variants["baseline"]["seeds"][str(seed)] for seed in SEEDS
    }
    expected_reproduction = phase29_incumbent_reproduction(baseline_rows)
    if expected_reproduction != persisted_reproduction:
        raise ValueError(
            f"{campaign_name} incumbent reproduction evidence is inconsistent"
        )
    return reproduction


def _validated_loss_weight_profile_selection(
    train_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if train_summary.get("loss_weight_profile_sha256") != (
        LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256
    ):
        raise ValueError("loss-weight profile hash is missing or changed")
    if train_summary.get("loss_weight_profiles") != LOSS_WEIGHT_PROFILES:
        raise ValueError("loss-weight profile table is missing or changed")
    variants = train_summary.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != set(
        LOSS_WEIGHT_PROFILES
    ):
        raise ValueError("loss-weight train variants changed")
    for profile_id, expected_weights in LOSS_WEIGHT_PROFILES.items():
        profile = variants[profile_id]
        if (
            profile.get("loss_weight_profile") != profile_id
            or profile.get("loss_weight_profile_sha256")
            != LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256
            or profile.get("loss_weights") != expected_weights
        ):
            raise ValueError(f"loss-weight train variant {profile_id} changed")
        seeds = profile.get("seeds")
        if not isinstance(seeds, Mapping) or set(seeds) != {
            str(seed) for seed in SEEDS
        }:
            raise ValueError(f"loss-weight seed set changed for {profile_id}")
        for seed in SEEDS:
            row = seeds[str(seed)]
            if (
                row.get("loss_weight_profile") != profile_id
                or row.get("loss_weight_profile_sha256")
                != LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256
                or row.get("loss_weights") != expected_weights
            ):
                raise ValueError(
                    f"loss-weight seed summary changed for {profile_id}/{seed}"
                )
    recorded = train_summary.get("loss_weight_profile_selection")
    if not isinstance(recorded, Mapping):
        raise ValueError("loss-weight profile selection is missing")
    selection_path = _validate_artifact(
        recorded["artifact"],
        label="loss-weight profile selection",
    )
    persisted = _load_json(selection_path)
    recorded_without_artifact = {
        key: value for key, value in recorded.items() if key != "artifact"
    }
    if persisted != recorded_without_artifact:
        raise ValueError("loss-weight profile selection artifact changed")
    expected = select_loss_weight_profile(variants)
    if persisted != expected:
        raise ValueError("loss-weight profile selection evidence is inconsistent")
    for profile_id in LOSS_WEIGHT_PROFILES:
        profile = variants[profile_id]
        selection = profile.get("selection")
        if not isinstance(selection, Mapping):
            raise ValueError(f"loss-weight seed selection missing for {profile_id}")
        expected_seed = (
            PHASE27_INCUMBENT_SELECTED_SEED
            if profile_id == "baseline"
            else expected["profiles"][profile_id]["selected_seed"]
        )
        if int(selection.get("selected_seed", -1)) != expected_seed:
            raise ValueError(f"loss-weight selected seed changed for {profile_id}")
        selection_path = _validate_artifact(
            profile["selection_artifact"],
            label=f"loss-weight seed selection {profile_id}",
        )
        if _load_json(selection_path) != dict(selection):
            raise ValueError(
                f"loss-weight seed selection artifact changed for {profile_id}"
            )
    if train_summary.get("selected_candidate_variant") != expected["winner_profile"]:
        raise ValueError("selected loss-weight candidate variant changed")
    return expected


def candidate_validation_improves(train_summary: Mapping[str, Any]) -> bool:
    variants = train_summary["variants"]
    variant_axis = train_summary.get("variant_axis")
    if variant_axis == LOSS_WEIGHT_PROFILE_AXIS:
        _validated_phase27_incumbent_reproduction(
            train_summary,
            campaign_name="Phase32",
        )
        selection = _validated_loss_weight_profile_selection(train_summary)
        winner_profile = str(selection["winner_profile"])
        winner_seed = str(
            variants[winner_profile]["selection"]["selected_seed"]
        )
        winner_validation = float(
            variants[winner_profile]["seeds"][winner_seed][VALIDATION_METRIC]
        )
        if not math.isfinite(winner_validation):
            raise ValueError("selected validation metrics must be finite")
        return (
            bool(selection["winner_improved_seed_majority"])
            and float(selection["winner_paired_mean_delta"]) < 0.0
            and float(selection["winner_paired_median_delta"]) < 0.0
            and winner_validation < PHASE27_INCUMBENT_VALIDATION
        )
    baseline_selection = variants["baseline"]["selection"]
    candidate_selection = variants["candidate"]["selection"]
    baseline_seed = str(baseline_selection["selected_seed"])
    candidate_seed = str(candidate_selection["selected_seed"])
    baseline = float(variants["baseline"]["seeds"][baseline_seed][VALIDATION_METRIC])
    candidate = float(variants["candidate"]["seeds"][candidate_seed][VALIDATION_METRIC])
    if not math.isfinite(baseline) or not math.isfinite(candidate):
        raise ValueError("selected validation metrics must be finite")
    if variant_axis in FROZEN_PHASE27_INCUMBENT_AXES:
        campaign_name = (
            "Phase29"
            if variant_axis == EVENT_BALANCE_EXPONENT_AXIS
            else (
                "Phase30"
                if variant_axis == MOMENT_LINEAR_SKIP_AXIS
                else "Phase31"
            )
        )
        _validated_phase27_incumbent_reproduction(
            train_summary,
            campaign_name=campaign_name,
        )
        return candidate < PHASE27_INCUMBENT_VALIDATION
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
    variant_axis = train_summary.get("variant_axis")
    candidate_variant = (
        str(train_summary["selected_candidate_variant"])
        if variant_axis == LOSS_WEIGHT_PROFILE_AXIS
        else "candidate"
    )
    baseline_seed = _selected_seed_summary(train_summary, "baseline")
    candidate_seed = _selected_seed_summary(train_summary, candidate_variant)
    if variant_axis == LOSS_WEIGHT_PROFILE_AXIS:
        profile_selection = _validated_loss_weight_profile_selection(train_summary)
        validation_gate = {
            "passed": validation_passed,
            "baseline_reproduction": float(baseline_seed[VALIDATION_METRIC]),
            "frozen_incumbent": PHASE27_INCUMBENT_VALIDATION,
            "frozen_incumbent_seed": PHASE27_INCUMBENT_SELECTED_SEED,
            "selected_candidate_variant": candidate_variant,
            "candidate": float(candidate_seed[VALIDATION_METRIC]),
            "winner_paired_median_delta": profile_selection[
                "winner_paired_median_delta"
            ],
            "winner_paired_mean_delta": profile_selection[
                "winner_paired_mean_delta"
            ],
            "winner_improved_seed_count": profile_selection[
                "winner_improved_seed_count"
            ],
            "winner_improved_seed_majority": profile_selection[
                "winner_improved_seed_majority"
            ],
            "metric": VALIDATION_METRIC,
            "rule": (
                "winner has at least two negative paired seed deltas, negative "
                "paired mean and median deltas, and its selected seed is below "
                "the frozen Phase27 incumbent"
            ),
            "incumbent_reproduction_verified": True,
        }
    elif variant_axis in FROZEN_PHASE27_INCUMBENT_AXES:
        validation_gate = {
            "passed": validation_passed,
            "baseline_reproduction": float(baseline_seed[VALIDATION_METRIC]),
            "frozen_incumbent": PHASE27_INCUMBENT_VALIDATION,
            "frozen_incumbent_seed": PHASE27_INCUMBENT_SELECTED_SEED,
            "candidate": float(candidate_seed[VALIDATION_METRIC]),
            "metric": VALIDATION_METRIC,
            "rule": "candidate < frozen Phase27 incumbent",
            "incumbent_reproduction_verified": True,
        }
    else:
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
    for variant, source_variant, seed_summary in (
        ("baseline", "baseline", baseline_seed),
        ("candidate", candidate_variant, candidate_seed),
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
            if variant_axis == LOSS_WEIGHT_PROFILE_AXIS:
                result.update(
                    {
                        "source_variant": source_variant,
                        "loss_weight_profile": source_variant,
                        "loss_weights": LOSS_WEIGHT_PROFILES[source_variant],
                        "loss_weight_profile_sha256": (
                            LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256
                        ),
                    }
                )
            _atomic_json(variant_dir / "summary.json", result, overwrite=resume)
        elif variant_axis == LOSS_WEIGHT_PROFILE_AXIS and (
            int(result.get("selected_seed", -1)) != int(seed_summary["seed"])
            or result.get("source_variant") != source_variant
            or result.get("loss_weight_profile") != source_variant
            or result.get("loss_weights") != LOSS_WEIGHT_PROFILES[source_variant]
            or result.get("loss_weight_profile_sha256")
            != LOSS_WEIGHT_CANDIDATE_PROFILE_SHA256
        ):
            raise ValueError("resumed loss-weight internal evaluation changed")
        evaluations[variant] = result
    baseline_event_mae = float(evaluations["baseline"]["metrics"]["event_mae"])
    candidate_event_mae = float(evaluations["candidate"]["metrics"]["event_mae"])
    if variant_axis == LOSS_WEIGHT_PROFILE_AXIS:
        candidate_gate = {
            "passed": (
                candidate_event_mae < PHASE27_INCUMBENT_TEST_EVENT_MAE
                and candidate_event_mae < INTERNAL_EVENT_MAE_MAXIMUM
            ),
            "event_mae": candidate_event_mae,
            "phase27_incumbent_maximum_exclusive": (
                PHASE27_INCUMBENT_TEST_EVENT_MAE
            ),
            "absolute_maximum_exclusive": INTERNAL_EVENT_MAE_MAXIMUM,
            "rule": (
                "candidate Event MAE < frozen Phase27 Event MAE and < 0.15"
            ),
        }
    else:
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
    if variant_axis == LOSS_WEIGHT_PROFILE_AXIS:
        summary["selected_candidate_variant"] = candidate_variant
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
        waveform_available_before_sec=FULL_OBSERVATION_HORIZON_SEC,
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
    waveform_starts = [float(row["waveform_start_sec"]) for row in station_rows]
    interpolation_gaps = {
        float(row["waveform_max_interpolation_gap_sec"])
        for row in station_rows
    }
    if interpolation_gaps != {0.0}:
        raise ValueError("formal external evaluation must not interpolate waveforms")
    available_before_values = {
        float(row["waveform_available_before_sec"]) for row in station_rows
    }
    if available_before_values != {FULL_OBSERVATION_HORIZON_SEC}:
        raise ValueError("formal external waveform availability cutoff changed")
    phase_adjusted_count = sum(
        bool(row["waveform_phase_adjusted"]) for row in station_rows
    )
    slot_counts = {int(row["waveform_slot_count"]) for row in station_rows}
    if slot_counts != {int(FULL_OBSERVATION_HORIZON_SEC)}:
        raise ValueError("formal external waveform slot count changed")
    valid_counts = [int(row["waveform_valid_sample_count"]) for row in station_rows]
    masked_counts = [int(row["waveform_masked_sample_count"]) for row in station_rows]
    valid_fractions = [float(row["waveform_valid_fraction"]) for row in station_rows]
    if any(
        valid + masked != int(FULL_OBSERVATION_HORIZON_SEC)
        for valid, masked in zip(valid_counts, masked_counts, strict=True)
    ):
        raise ValueError("external valid and masked sample counts are inconsistent")
    for valid, masked, fraction in zip(
        valid_counts,
        masked_counts,
        valid_fractions,
        strict=True,
    ):
        expected_fraction = valid / int(FULL_OBSERVATION_HORIZON_SEC)
        if not math.isclose(fraction, expected_fraction, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("external valid fraction disagrees with sample counts")
        if fraction < FORMAL_MIN_VALID_FRACTION:
            raise ValueError("external waveform violates the formal valid-fraction gate")
    raw_dts = [float(row["waveform_raw_dt_sec"]) for row in station_rows]
    if any(
        not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1.0e-8)
        for value in raw_dts
    ):
        raise ValueError("formal external waveforms must retain a 1 Hz raw grid")
    baseline_source_counts: dict[str, int] = {}
    for row in station_rows:
        source = str(row["waveform_baseline_source"])
        baseline_source_counts[source] = baseline_source_counts.get(source, 0) + 1
    maximum_last_sample_sec = (
        max(waveform_starts) + FULL_OBSERVATION_HORIZON_SEC - 1.0
    )
    maximum_filter_support_sec = (
        maximum_last_sample_sec + FORMAL_FIR_LOOKAHEAD_SEC
    )
    if maximum_last_sample_sec >= FULL_OBSERVATION_HORIZON_SEC:
        raise ValueError("external phase grid reads at or beyond the 200 s horizon")
    if maximum_filter_support_sec >= FULL_RELEASE_TIME_SEC:
        raise ValueError("external FIR support exceeds the five-second release budget")
    event_names = sorted(str(row["event"]) for row in event_rows)
    if len(set(event_names)) != len(event_names):
        raise ValueError("external threshold produced duplicate event summaries")
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
        "waveform_grid": {
            "schema_version": EXTERNAL_WAVEFORM_GRID_SCHEMA_VERSION,
            "mode": "raw_sample_phase_no_interpolation",
            "configured_start_sec": 0.0,
            "raw_input_available_before_sec": FULL_OBSERVATION_HORIZON_SEC,
            "minimum_start_sec": min(waveform_starts),
            "maximum_start_sec": max(waveform_starts),
            "slot_count": int(FULL_OBSERVATION_HORIZON_SEC),
            "minimum_valid_sample_count": min(valid_counts),
            "maximum_masked_sample_count": max(masked_counts),
            "minimum_valid_fraction": min(valid_fractions),
            "raw_dt_sec_minimum": min(raw_dts),
            "raw_dt_sec_maximum": max(raw_dts),
            "baseline_source_counts": baseline_source_counts,
            "maximum_last_sample_sec": maximum_last_sample_sec,
            "fir_boundary_mode": "zero_padded_same",
            "fir_nominal_lookahead_sec": FORMAL_FIR_LOOKAHEAD_SEC,
            "nominal_maximum_filter_support_sec": maximum_filter_support_sec,
            "release_margin_sec": (
                FULL_RELEASE_TIME_SEC - maximum_filter_support_sec
            ),
            "max_interpolation_gap_sec": 0.0,
            "phase_adjusted_station_count": phase_adjusted_count,
            "station_count": len(station_rows),
        },
        "event_names": event_names,
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
        if (
            int(summary["waveform_grid"]["schema_version"])
            != EXTERNAL_WAVEFORM_GRID_SCHEMA_VERSION
        ):
            return None
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


def _external_coverage_gate(
    threshold_summary: Mapping[str, Any],
    label_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required_event_names = sorted(str(row["event"]) for row in label_rows)
    observed_event_names = sorted(
        str(event) for event in threshold_summary.get("event_names", [])
    )
    event_count = int(threshold_summary["event_metrics"]["count"])
    missing_event_names = sorted(set(required_event_names) - set(observed_event_names))
    unexpected_event_names = sorted(set(observed_event_names) - set(required_event_names))
    return {
        "passed": (
            event_count == len(required_event_names)
            and observed_event_names == required_event_names
        ),
        "event_count": event_count,
        "required_event_count": len(required_event_names),
        "observed_event_names": observed_event_names,
        "required_event_names": required_event_names,
        "missing_event_names": missing_event_names,
        "unexpected_event_names": unexpected_event_names,
    }


def run_external(
    *,
    output_root: Path,
    event_root: Path,
    resume: bool,
) -> dict[str, Any]:
    stage_dir = _stage_dir(output_root, "external")
    completed = _prepare_stage(stage_dir, resume=resume)
    if completed is not None:
        if (
            int(completed.get("external_waveform_grid_schema_version", -1))
            != EXTERNAL_WAVEFORM_GRID_SCHEMA_VERSION
        ):
            raise ValueError(
                "completed external stage uses an obsolete waveform-grid schema; "
                "preserve it and use a new output namespace"
            )
        return completed
    internal = _require_stage(output_root, "internal")
    if internal.get("status") != "complete" or not bool(
        internal.get("candidate_gate", {}).get("passed")
    ):
        raise RuntimeError("external stage requires a passed internal candidate gate")
    train_summary = _require_stage(output_root, "train")
    if train_summary.get("variant_axis") == LOSS_WEIGHT_PROFILE_AXIS:
        _validated_phase27_incumbent_reproduction(
            train_summary,
            campaign_name="Phase32",
        )
        profile_selection = _validated_loss_weight_profile_selection(train_summary)
        candidate_variant = str(profile_selection["winner_profile"])
    else:
        candidate_variant = "candidate"
    candidate_seed = _selected_seed_summary(train_summary, candidate_variant)
    if int(candidate_seed["seed"]) != int(
        internal["variants"]["candidate"]["selected_seed"]
    ):
        raise ValueError("internal and train selected candidate seeds differ")
    if train_summary.get("variant_axis") == LOSS_WEIGHT_PROFILE_AXIS and (
        internal.get("selected_candidate_variant") != candidate_variant
        or internal["variants"]["candidate"].get("source_variant")
        != candidate_variant
    ):
        raise ValueError("internal and train selected loss-weight profiles differ")
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
    coverage_gate = _external_coverage_gate(
        threshold_summaries["cm0"],
        label_rows,
    )
    coverage_passed = bool(coverage_gate["passed"])
    summary = {
        "stage": "external",
        "status": "complete" if coverage_passed else "cm0_coverage_gate_failed",
        "created_at_utc": utc_now_iso(),
        "evaluation_git_commit": current_git_commit(PROJECT_ROOT),
        "evaluation_git_dirty": git_is_dirty(PROJECT_ROOT),
        "external_waveform_grid_schema_version": (
            EXTERNAL_WAVEFORM_GRID_SCHEMA_VERSION
        ),
        "internal_summary": _artifact(
            _stage_dir(output_root, "internal") / "summary.json"
        ),
        "selected_variant": candidate_variant,
        "selected_seed": int(candidate_seed["seed"]),
        "ensemble_used": False,
        "observation_horizon_sec": FULL_OBSERVATION_HORIZON_SEC,
        "release_time_sec": FULL_RELEASE_TIME_SEC,
        "input_sha256": input_hashes,
        "label_manifest": _artifact(label_manifest),
        "cm0_coverage_gate": coverage_gate,
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
