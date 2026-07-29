from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_phase40_streaming_consistency import (  # noqa: E402
    GATES as PHASE40_GATES,
    PHASE39_CHECKPOINT,
    load_frozen_config,
    validate_source_artifacts,
)
from scripts.experiments.run_phase43_streaming_adapter import (  # noqa: E402
    CacheBundle,
    HORIZONS,
    LATE_HORIZONS,
    _late_metrics_from_rates,
    _tensor_batch,
    encode_rate,
    load_cache,
)
from src.models.model import PINNModel  # noqa: E402
from src.training.checkpointing import atomic_torch_save  # noqa: E402
from src.training.loss_stf_rate_v2 import (  # noqa: E402
    moment_magnitude_from_rate,
)
from src.training.train import _build_stf_rate_criterion  # noqa: E402
from src.utils.config_v2 import stf_m_ref_from_config  # noqa: E402
from src.utils.device import configure_runtime  # noqa: E402
from src.utils.provenance import (  # noqa: E402
    current_git_commit,
    git_is_dirty,
    sha256_file,
)


DEFAULT_CACHE_ROOT = Path(
    "/home/lihe/PINN_Mag/runs/"
    "phase43-streaming-prefix-cache-20260728T115851Z-7bc63eb"
)
SEEDS = (17, 42, 73)
EXPECTED_BASE_PARAMETER_COUNT = 1_010_850
EXPECTED_TRANSITION_PARAMETER_COUNT = 1_029
EXPECTED_TOTAL_PARAMETER_COUNT = (
    EXPECTED_BASE_PARAMETER_COUNT + EXPECTED_TRANSITION_PARAMETER_COUNT
)
EXPECTED_TRAIN_COUNT = 1_788
EXPECTED_VALIDATION_COUNT = 385

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP_NORM = 1.0
SUPPORT_RAMP_SEC = 6.0
MAX_PROPOSAL_CORRECTION_LOG10 = 1.0
MAX_MOMENT_DOWN_FRACTION_PER_STEP = 0.01
EARLY_MOMENT_DOWN_FRACTION_PER_STEP = 0.1
MOMENT_STABILITY_START_SEC = 60
MOMENT_REBASE_START_SEC = 40
MAX_MOMENT_PROPOSAL_CORRECTION_LOG10 = 3.0
FULL_STF_ALIGNMENT_START_SEC = 180
FULL_STF_ALIGNMENT_DOWN_FRACTION_PER_STEP = 0.03
STEP_HUBER_BETA_MW = 0.01
HISTORY_HUBER_BETA_LOG10 = 0.05
MULTISCALE_OFFSETS = (5, 20, 60)
NORMALIZER_BATCHES = 8
NORMALIZER_SEED = 42
LOSS_WEIGHTS = {
    "endpoint_science": 2.0,
    "released_sequence": 1.0,
    "endpoint_teacher": 1.0,
    "downward_step": 2.0,
    "multiscale_downward": 2.0,
    "confirmed_history": 1.0,
}
NORMALIZER_FLOORS = {
    "endpoint_science": 1.0e-8,
    "released_sequence": 1.0e-8,
    "endpoint_teacher": 1.0e-4,
    "downward_step": 1.0e-8,
    "multiscale_downward": 1.0e-8,
    "confirmed_history": 1.0e-8,
}
VALIDATION_GATES = {
    **PHASE40_GATES,
    "event_downward_step_p95_mw_max": 0.010,
    "event_downward_step_max_mw_max": 0.050,
    "event_peak_to_final_p95_mw_max": 0.150,
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
            writer.writerow(
                {
                    key: "" if source.get(key) is None else source.get(key)
                    for key in fieldnames
                }
            )


def _validate_new_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output directory must be new or empty: {path}")


def phase50_config() -> dict[str, Any]:
    config = copy.deepcopy(load_frozen_config())
    config["model"]["stateful_streaming"] = {
        "mode": "released_stf_gru",
        "local_channels": 4,
        "hidden_size": 8,
        "support_ramp_sec": SUPPORT_RAMP_SEC,
        "initial_gate_logit": -4.0,
        "max_proposal_correction_log10": MAX_PROPOSAL_CORRECTION_LOG10,
        "max_moment_down_fraction_per_step": (
            MAX_MOMENT_DOWN_FRACTION_PER_STEP
        ),
        "early_moment_down_fraction_per_step": (
            EARLY_MOMENT_DOWN_FRACTION_PER_STEP
        ),
        "moment_stability_start_sec": MOMENT_STABILITY_START_SEC,
        "max_moment_proposal_correction_log10": (
            MAX_MOMENT_PROPOSAL_CORRECTION_LOG10
        ),
        "use_moment_rebase_window": True,
        "moment_rebase_start_sec": MOMENT_REBASE_START_SEC,
        "use_full_stf_alignment": True,
        "full_stf_alignment_start_sec": FULL_STF_ALIGNMENT_START_SEC,
        "full_stf_alignment_down_fraction_per_step": (
            FULL_STF_ALIGNMENT_DOWN_FRACTION_PER_STEP
        ),
    }
    return config


def _transition_parameters(model: PINNModel) -> list[torch.nn.Parameter]:
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("released_stf_transition.")
    ]
    count = sum(parameter.numel() for parameter in parameters)
    if count != EXPECTED_TRANSITION_PARAMETER_COUNT:
        raise ValueError(
            f"Phase50 transition parameter count changed: {count} != "
            f"{EXPECTED_TRANSITION_PARAMETER_COUNT}"
        )
    return parameters


def freeze_transition_scope(model: PINNModel) -> list[torch.nn.Parameter]:
    parameters = _transition_parameters(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("released_stf_transition."))
    return parameters


def load_phase50_model(
    config: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[PINNModel, dict[str, torch.Tensor]]:
    source_state = torch.load(
        PHASE39_CHECKPOINT,
        map_location=device,
        weights_only=True,
    )
    model = PINNModel(config).to(device)
    incompatible = model.load_state_dict(source_state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "Phase39 source produced unexpected Phase50 keys: "
            + ", ".join(incompatible.unexpected_keys)
        )
    if not incompatible.missing_keys or any(
        not key.startswith("released_stf_transition.")
        for key in incompatible.missing_keys
    ):
        raise ValueError("Phase50 source load missing-key contract changed")
    total = sum(parameter.numel() for parameter in model.parameters())
    if total != EXPECTED_TOTAL_PARAMETER_COUNT:
        raise ValueError(
            f"Phase50 parameter count changed: {total} != "
            f"{EXPECTED_TOTAL_PARAMETER_COUNT}"
        )
    freeze_transition_scope(model)
    frozen_source = {
        name: value.detach().cpu().clone()
        for name, value in source_state.items()
    }
    return model, frozen_source


def _assert_backbone_unchanged(
    model: PINNModel,
    frozen_source: Mapping[str, torch.Tensor],
) -> None:
    current = model.state_dict()
    if set(frozen_source) - set(current):
        raise ValueError("Phase50 checkpoint lost Phase39 backbone tensors")
    changed = [
        name
        for name, source in frozen_source.items()
        if not torch.equal(current[name].detach().cpu(), source)
    ]
    if changed:
        raise RuntimeError(
            "Phase50 modified frozen Phase39 tensors: "
            + ", ".join(changed[:10])
        )


def causal_support_fraction(
    batch: Mapping[str, torch.Tensor],
    *,
    beta_m_per_s: float,
    source_steps: int,
) -> torch.Tensor:
    reference = batch["raw_rate"]
    horizons = torch.as_tensor(
        HORIZONS,
        device=reference.device,
        dtype=reference.dtype,
    ).reshape(1, -1, 1)
    distance = batch["source_distance_m"].reshape(-1, 1, 1)
    source_index = torch.arange(
        source_steps,
        device=reference.device,
        dtype=reference.dtype,
    ).reshape(1, 1, -1)
    age = horizons - distance / float(beta_m_per_s) - source_index
    return torch.clamp(age / SUPPORT_RAMP_SEC, min=0.0, max=1.0)


def target_support_fraction(causal_support: torch.Tensor) -> torch.Tensor:
    if causal_support.ndim != 3 or causal_support.shape[1] != len(HORIZONS):
        raise ValueError("causal support shape changed")
    horizons = causal_support.new_tensor(HORIZONS).reshape(1, -1, 1)
    alignment_weight = torch.clamp(
        (
            horizons
            - float(FULL_STF_ALIGNMENT_START_SEC)
            + 1.0
        )
        / float(HORIZONS[-1] - FULL_STF_ALIGNMENT_START_SEC + 1),
        min=0.0,
        max=1.0,
    )
    return causal_support + alignment_weight * (1.0 - causal_support)


def _weighted_mean(
    per_sample: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    if per_sample.ndim != 1 or per_sample.shape != sample_weights.shape:
        raise ValueError("weighted values must have shape (batch,)")
    return torch.mean(per_sample * sample_weights)


def normalizer_training_indices(train_indices: np.ndarray) -> np.ndarray:
    sample_count = NORMALIZER_BATCHES * BATCH_SIZE
    if train_indices.ndim != 1 or len(train_indices) < sample_count:
        raise ValueError("Phase50 has insufficient training records for audit")
    generator = np.random.default_rng(NORMALIZER_SEED)
    return generator.choice(
        train_indices,
        size=sample_count,
        replace=False,
    )


def stateful_loss_components(
    model: PINNModel,
    batch: Mapping[str, torch.Tensor],
    *,
    criterion: Any,
    config: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    beta = float(config["physics"]["beta"])
    source_dt = batch["source_dt_sec"]
    sample_weights = batch["sample_weight"]
    states, encoded_states, state_mw, gates = model.stream_sequence_from_rates(
        batch["raw_rate"],
        horizons_sec=HORIZONS,
        source_distance_m=batch["source_distance_m"],
        source_dt_sec=source_dt,
        beta_m_per_s=beta,
    )
    causal_support = causal_support_fraction(
        batch,
        beta_m_per_s=beta,
        source_steps=states.shape[2],
    )
    target_rate = batch["stf"].unsqueeze(1) * target_support_fraction(
        causal_support
    )
    target_encoded = encode_rate(
        target_rate,
        stf_m_ref=stf_m_ref_from_config(dict(config)),
    )
    target_mw = moment_magnitude_from_rate(
        target_rate.reshape(-1, target_rate.shape[2]),
        source_dt.reshape(-1, 1).expand(-1, target_rate.shape[1]).reshape(-1),
    ).reshape(target_rate.shape[0], target_rate.shape[1])

    endpoint_science, endpoint_parts = criterion(
        encoded_states[:, -1],
        pred_catalog_mw=None,
        radial_obs=batch["radial"],
        source_distance_m=batch["source_distance_m"],
        theta_deg=batch["theta_deg"],
        phi_slip_deg=batch["phi_slip_deg"],
        source_dt_sec=source_dt,
        observation_dt_sec=batch["observation_dt_sec"],
        waveform_valid_mask=batch["waveform_valid_mask"],
        stf_true=batch["stf"],
        has_stf=torch.ones_like(batch["magnitude_catalog"], dtype=torch.bool),
        true_mag=batch["magnitude_catalog"],
        sample_weights=sample_weights,
    )

    encoded_error = (encoded_states - target_encoded).pow(2).mean(dim=2)
    magnitude_error = torch.abs(state_mw - target_mw)
    target_moment = torch.sum(
        target_rate * source_dt.reshape(-1, 1, 1),
        dim=2,
    )
    full_moment = torch.sum(
        batch["stf"] * source_dt.reshape(-1, 1),
        dim=1,
    ).clamp_min(1.0e10)
    information_weight = 0.05 + 0.95 * torch.clamp(
        target_moment / full_moment.unsqueeze(1),
        min=0.0,
        max=1.0,
    )
    sequence_per_sample = torch.sum(
        information_weight * (encoded_error + magnitude_error),
        dim=1,
    ) / information_weight.sum(dim=1).clamp_min(1.0e-12)
    released_sequence = _weighted_mean(sequence_per_sample, sample_weights)

    raw_endpoint_encoded = encode_rate(
        batch["raw_rate"][:, -1],
        stf_m_ref=stf_m_ref_from_config(dict(config)),
    )
    raw_endpoint_mw = moment_magnitude_from_rate(
        batch["raw_rate"][:, -1],
        source_dt,
    )
    endpoint_teacher_mw_error = torch.abs(
        state_mw[:, -1] - raw_endpoint_mw
    )
    endpoint_teacher = _weighted_mean(
        (
            (encoded_states[:, -1] - raw_endpoint_encoded)
            .pow(2)
            .mean(dim=1)
            + endpoint_teacher_mw_error
        ),
        sample_weights,
    )

    adjacent_down = torch.relu(state_mw[:, :-1] - state_mw[:, 1:])
    adjacent_loss = F.smooth_l1_loss(
        adjacent_down,
        torch.zeros_like(adjacent_down),
        reduction="none",
        beta=STEP_HUBER_BETA_MW,
    ).mean(dim=1)
    downward_step = _weighted_mean(adjacent_loss, sample_weights)

    multiscale_per_sample = state_mw.new_zeros(state_mw.shape[0])
    for offset in MULTISCALE_OFFSETS:
        down = torch.relu(state_mw[:, :-offset] - state_mw[:, offset:])
        term = F.smooth_l1_loss(
            down,
            torch.zeros_like(down),
            reduction="none",
            beta=STEP_HUBER_BETA_MW,
        ).mean(dim=1)
        multiscale_per_sample = multiscale_per_sample + term
    multiscale_per_sample = multiscale_per_sample / len(MULTISCALE_OFFSETS)
    multiscale_downward = _weighted_mean(
        multiscale_per_sample,
        sample_weights,
    )

    dt = source_dt.reshape(-1, 1, 1)
    cumulative_log = torch.log10(
        torch.cumsum(states * dt, dim=2).clamp_min(1.0e10)
    )
    confirmed_delta = cumulative_log[:, 1:] - cumulative_log[:, :-1]
    previous_support = causal_support[:, :-1] >= 1.0
    confirmed_loss = F.smooth_l1_loss(
        confirmed_delta,
        torch.zeros_like(confirmed_delta),
        reduction="none",
        beta=HISTORY_HUBER_BETA_LOG10,
    )
    confirmed_count = previous_support.sum(dim=(1, 2)).clamp_min(1)
    confirmed_per_sample = (
        (confirmed_loss * previous_support).sum(dim=(1, 2)) / confirmed_count
    )
    confirmed_history = _weighted_mean(
        confirmed_per_sample,
        sample_weights,
    )

    components = {
        "endpoint_science": endpoint_science,
        "released_sequence": released_sequence,
        "endpoint_teacher": endpoint_teacher,
        "downward_step": downward_step,
        "multiscale_downward": multiscale_downward,
        "confirmed_history": confirmed_history,
    }
    diagnostics = {
        "mean_gate": float(gates.detach().mean().cpu()),
        "late_mean_gate": float(
            gates[:, HORIZONS.index(LATE_HORIZONS[0]) :]
            .detach()
            .mean()
            .cpu()
        ),
        "mean_downward_step_mw": float(adjacent_down.detach().mean().cpu()),
        "target_mw_mae": float(magnitude_error.detach().mean().cpu()),
        "endpoint_teacher_mw_mae": float(
            endpoint_teacher_mw_error.detach().mean().cpu()
        ),
        "endpoint_L_MSE": float(endpoint_parts["L_MSE"]),
        "endpoint_L_synth": float(endpoint_parts["L_synth"]),
        "endpoint_L_mag": float(endpoint_parts["L_mag"]),
    }
    return components, diagnostics


def normalized_loss(
    components: Mapping[str, torch.Tensor],
    normalizers: Mapping[str, float],
) -> torch.Tensor:
    if set(components) != set(LOSS_WEIGHTS):
        raise ValueError("Phase50 loss component set changed")
    if set(normalizers) != set(LOSS_WEIGHTS):
        raise ValueError("Phase50 normalizer set changed")
    total: torch.Tensor | None = None
    for name, weight in LOSS_WEIGHTS.items():
        normalizer = float(normalizers[name])
        if not math.isfinite(normalizer) or normalizer <= 0.0:
            raise ValueError(f"invalid normalizer for {name}")
        term = float(weight) * components[name] / normalizer
        total = term if total is None else total + term
    if total is None:  # pragma: no cover
        raise AssertionError("Phase50 objective is empty")
    return total


def audit_loss_scales(
    *,
    cache_root: Path,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    _validate_new_directory(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_root)
    config = phase50_config()
    validate_source_artifacts()
    configure_runtime(42, device)
    torch.manual_seed(42)
    model, frozen_source = load_phase50_model(config, device=device)
    criterion = _build_stf_rate_criterion(config, device)
    train_indices = np.flatnonzero(cache.arrays["split_code"] == 0)
    if len(train_indices) != EXPECTED_TRAIN_COUNT:
        raise ValueError("Phase50 training count changed")
    audit_indices = normalizer_training_indices(train_indices)
    component_sums = {name: 0.0 for name in LOSS_WEIGHTS}
    sample_count = 0
    first_batch: dict[str, torch.Tensor] | None = None
    for start in range(0, len(audit_indices), BATCH_SIZE):
        indices = audit_indices[start : start + BATCH_SIZE]
        batch = _tensor_batch(cache, indices, device=device)
        if first_batch is None:
            first_batch = batch
        with torch.no_grad():
            components, _ = stateful_loss_components(
                model,
                batch,
                criterion=criterion,
                config=config,
            )
        batch_size = len(indices)
        sample_count += batch_size
        for name, value in components.items():
            component_sums[name] += float(value.detach().cpu()) * batch_size
    if sample_count == 0 or first_batch is None:
        raise ValueError("Phase50 audit produced no samples")
    normalizers = {
        name: max(component_sums[name] / sample_count, NORMALIZER_FLOORS[name])
        for name in LOSS_WEIGHTS
    }
    components, diagnostics = stateful_loss_components(
        model,
        first_batch,
        criterion=criterion,
        config=config,
    )
    parameters = tuple(_transition_parameters(model))
    gradient_norms: dict[str, float] = {}
    for index, name in enumerate(LOSS_WEIGHTS):
        gradients = torch.autograd.grad(
            components[name] / normalizers[name],
            parameters,
            retain_graph=index < len(LOSS_WEIGHTS) - 1,
            allow_unused=True,
        )
        squared = sum(
            float(torch.sum(gradient.detach().pow(2)).cpu())
            for gradient in gradients
            if gradient is not None
        )
        gradient_norms[name] = math.sqrt(squared)
    _assert_backbone_unchanged(model, frozen_source)
    payload = {
        "status": "loss_scale_audit_complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(cache_root),
        "cache_raw_rates_sha256": cache.manifest["raw_rates_sha256"],
        "cache_arrays_sha256": cache.manifest["arrays_sha256"],
        "normalizer_batches": NORMALIZER_BATCHES,
        "normalizer_sample_count": sample_count,
        "normalizer_sampling": {
            "method": "seeded_without_replacement",
            "seed": NORMALIZER_SEED,
            "event_count": len(
                {
                    str(cache.records[int(index)]["event"])
                    for index in audit_indices
                }
            ),
        },
        "normalizers": normalizers,
        "loss_weights": dict(LOSS_WEIGHTS),
        "endpoint_teacher": "encoded_stf_mse_plus_raw_phase39_mw_l1",
        "normalized_gradient_norms_first_batch": gradient_norms,
        "initial_diagnostics_first_batch": diagnostics,
        "provenance": {
            "git_commit": current_git_commit(PROJECT_ROOT),
            "git_dirty": git_is_dirty(PROJECT_ROOT),
            "source_artifact_sha256": validate_source_artifacts(),
            "internal_test_iterated": False,
            "external_data_loaded": False,
            "grouped_test_loaded": False,
        },
    }
    _write_json(output_root / "normalizers.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def load_normalizers(
    path: Path,
    *,
    cache: CacheBundle,
) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "loss_scale_audit_complete":
        raise ValueError("Phase50 normalizer audit is incomplete")
    if payload["cache_raw_rates_sha256"] != cache.manifest["raw_rates_sha256"]:
        raise ValueError("Phase50 normalizers use a different raw cache")
    if payload["cache_arrays_sha256"] != cache.manifest["arrays_sha256"]:
        raise ValueError("Phase50 normalizers use different cache arrays")
    if payload["loss_weights"] != LOSS_WEIGHTS:
        raise ValueError("Phase50 normalizer loss weights changed")
    sampling = payload.get("normalizer_sampling", {})
    if not isinstance(sampling, dict) or sampling.get("method") != (
        "seeded_without_replacement"
    ) or sampling.get("seed") != NORMALIZER_SEED:
        raise ValueError("Phase50 normalizers use the biased audit sampling")
    normalizers = {
        name: float(payload["normalizers"][name]) for name in LOSS_WEIGHTS
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in normalizers.values()):
        raise ValueError("Phase50 normalizers must be positive and finite")
    return normalizers


def _event_series(
    station_mw: np.ndarray,
    events: Sequence[str],
) -> np.ndarray:
    values = []
    event_array = np.asarray([str(event) for event in events])
    for event in sorted(set(event_array)):
        values.append(np.median(station_mw[event_array == event], axis=0))
    return np.stack(values)


def evaluate_model(
    model: PINNModel,
    cache: CacheBundle,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    validation_indices = np.flatnonzero(cache.arrays["split_code"] == 1)
    if len(validation_indices) != EXPECTED_VALIDATION_COUNT:
        raise ValueError("Phase50 validation count changed")
    selected_indices: list[int] = []
    state_cubes: list[np.ndarray] = []
    target_mw_errors: list[np.ndarray] = []
    gate_sum = 0.0
    gate_count = 0
    model.eval()
    with torch.no_grad():
        for batch_index, start in enumerate(
            range(0, len(validation_indices), BATCH_SIZE)
        ):
            if max_batches is not None and batch_index >= max_batches:
                break
            indices = validation_indices[start : start + BATCH_SIZE]
            batch = _tensor_batch(cache, indices, device=device)
            states, _, state_mw, gates = model.stream_sequence_from_rates(
                batch["raw_rate"],
                horizons_sec=HORIZONS,
                source_distance_m=batch["source_distance_m"],
                source_dt_sec=batch["source_dt_sec"],
                beta_m_per_s=float(config["physics"]["beta"]),
            )
            causal_support = causal_support_fraction(
                batch,
                beta_m_per_s=float(config["physics"]["beta"]),
                source_steps=states.shape[2],
            )
            target_rate = batch["stf"].unsqueeze(1) * target_support_fraction(
                causal_support
            )
            target_mw = moment_magnitude_from_rate(
                target_rate.reshape(-1, target_rate.shape[2]),
                batch["source_dt_sec"]
                .reshape(-1, 1)
                .expand(-1, target_rate.shape[1])
                .reshape(-1),
            ).reshape(target_rate.shape[0], target_rate.shape[1])
            target_mw_errors.append(
                torch.abs(state_mw - target_mw).cpu().numpy()
            )
            state_cubes.append(states.cpu().numpy().astype(np.float32, copy=False))
            selected_indices.extend(int(value) for value in indices)
            gate_sum += float(gates.sum().cpu())
            gate_count += int(gates.numel())
    if not state_cubes:
        raise ValueError("Phase50 validation produced no batches")

    states = np.concatenate(state_cubes, axis=0)
    target_errors = np.concatenate(target_mw_errors, axis=0)
    selected = np.asarray(selected_indices, dtype=np.int64)
    events = [str(cache.records[index]["event"]) for index in selected]
    catalogs = np.asarray(cache.arrays["magnitude_catalog"][selected])
    source_distance = np.asarray(cache.arrays["source_distance_m"][selected])
    source_dt = np.asarray(cache.arrays["source_dt_sec"][selected])
    late_start = HORIZONS.index(LATE_HORIZONS[0])
    metrics = _late_metrics_from_rates(
        states[:, late_start:],
        events=events,
        catalogs=catalogs,
        source_distance_m=source_distance,
        source_dt_sec=source_dt,
        beta_m_per_s=float(config["physics"]["beta"]),
    )
    moments = np.sum(
        np.maximum(states, 0.0) * source_dt.reshape(-1, 1, 1),
        axis=2,
    )
    station_mw = (2.0 / 3.0) * (
        np.log10(np.maximum(moments, 1.0e10)) - 9.1
    )
    event_mw = _event_series(station_mw, events)
    event_steps = np.diff(event_mw, axis=1)
    event_downward = np.maximum(-event_steps, 0.0)
    after_60 = event_mw[:, np.asarray(HORIZONS) >= 60]
    peak_to_final = np.max(after_60, axis=1) - after_60[:, -1]
    station_after_60 = station_mw[:, np.asarray(HORIZONS) >= 60]
    station_peak_to_final = (
        np.max(station_after_60, axis=1) - station_after_60[:, -1]
    )
    metrics.update(
        {
            "event_downward_step_p95_mw": float(
                np.percentile(event_downward, 95)
            ),
            "event_downward_step_max_mw": float(np.max(event_downward)),
            "event_downward_fraction": float(np.mean(event_steps < 0.0)),
            "event_peak_to_final_p95_mw": float(
                np.percentile(peak_to_final, 95)
            ),
            "event_peak_to_final_mean_mw": float(np.mean(peak_to_final)),
            "station_peak_to_final_p95_mw": float(
                np.percentile(station_peak_to_final, 95)
            ),
            "event_start_to_end_increase_fraction": float(
                np.mean(event_mw[:, -1] >= event_mw[:, 0])
            ),
            "released_target_station_mw_mae": float(np.mean(target_errors)),
        }
    )
    return metrics, {"validation_mean_gate": gate_sum / max(gate_count, 1)}


def validation_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    endpoint_event_pass = (
        float(metrics["endpoint_event_mae"])
        <= VALIDATION_GATES["endpoint_event_mae_max"]
    )
    endpoint_station_pass = (
        float(metrics["endpoint_station_mae"])
        <= VALIDATION_GATES["endpoint_station_mae_max"]
    )
    ratios = {
        "late_event_p95_ratio": (
            float(metrics["late_event_abs_step_p95_mw"])
            / VALIDATION_GATES["late_event_abs_step_p95_mw_max"]
        ),
        "late_station_p95_ratio": (
            float(metrics["late_station_abs_step_p95_mw"])
            / VALIDATION_GATES["late_station_abs_step_p95_mw_max"]
        ),
        "late_confirmed_history_ratio": (
            float(metrics["late_confirmed_cumulative_log10_l1_p95"])
            / VALIDATION_GATES[
                "late_confirmed_cumulative_log10_l1_p95_max"
            ]
        ),
        "event_downward_step_ratio": (
            float(metrics["event_downward_step_p95_mw"])
            / VALIDATION_GATES["event_downward_step_p95_mw_max"]
        ),
        "event_downward_step_max_ratio": (
            float(metrics["event_downward_step_max_mw"])
            / VALIDATION_GATES["event_downward_step_max_mw_max"]
        ),
        "event_peak_to_final_ratio": (
            float(metrics["event_peak_to_final_p95_mw"])
            / VALIDATION_GATES["event_peak_to_final_p95_mw_max"]
        ),
    }
    score = max(ratios.values())
    endpoint_preserved = endpoint_event_pass and endpoint_station_pass
    streaming_passed = score <= 1.0
    return {
        "endpoint_event_passed": endpoint_event_pass,
        "endpoint_station_passed": endpoint_station_pass,
        "endpoint_preserved": endpoint_preserved,
        **ratios,
        "selection_score": score,
        "streaming_passed": streaming_passed,
        "passed": endpoint_preserved and streaming_passed,
    }


def _protocol(
    *,
    cache: CacheBundle,
    normalizer_path: Path,
    normalizers: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "phase": "Phase50",
        "model_class": "PINNModel",
        "stateful_mode": "released_stf_gru",
        "external_adapter": False,
        "source_model": "Phase39 Glehman scalar + global invariant, seed42",
        "source_checkpoint": str(PHASE39_CHECKPOINT),
        "source_checkpoint_sha256": cache.manifest["source_artifact_sha256"][
            "checkpoint"
        ],
        "base_parameter_count": EXPECTED_BASE_PARAMETER_COUNT,
        "transition_parameter_count": EXPECTED_TRANSITION_PARAMETER_COUNT,
        "total_parameter_count": EXPECTED_TOTAL_PARAMETER_COUNT,
        "trainable_parameters": "released_stf_transition.* only",
        "horizons": list(HORIZONS),
        "support_ramp_sec": SUPPORT_RAMP_SEC,
        "proposal_correction": {
            "coordinate": "per_source_bin_log10",
            "max_absolute_log10": MAX_PROPOSAL_CORRECTION_LOG10,
            "initialization": "identity",
        },
        "stateful_total_moment": {
            "coordinate": "asymmetric_linear_moment_evidence",
            "hidden_size": 8,
            "max_proposal_correction_log10": (
                MAX_MOMENT_PROPOSAL_CORRECTION_LOG10
            ),
            "initial_upward_fraction": 1.0 / (1.0 + math.exp(-4.0)),
            "initial_early_downward_fraction": (
                0.5 * EARLY_MOMENT_DOWN_FRACTION_PER_STEP
            ),
            "initial_late_downward_fraction": (
                0.5 * MAX_MOMENT_DOWN_FRACTION_PER_STEP
            ),
            "early_max_downward_fraction_per_step": (
                EARLY_MOMENT_DOWN_FRACTION_PER_STEP
            ),
            "max_downward_fraction_per_step": (
                MAX_MOMENT_DOWN_FRACTION_PER_STEP
            ),
            "stability_start_sec": MOMENT_STABILITY_START_SEC,
            "candidate_rebase_window_sec": [
                MOMENT_REBASE_START_SEC,
                MOMENT_STABILITY_START_SEC,
            ],
            "complete_stf_alignment_window_sec": [
                FULL_STF_ALIGNMENT_START_SEC,
                HORIZONS[-1],
            ],
            "moment_evidence_features": [
                "causal_candidate_log10_moment",
                "complete_proposal_log10_moment",
                "complete_proposal_minus_previous_state_log10",
            ],
            "alignment_max_downward_fraction_per_step": (
                FULL_STF_ALIGNMENT_DOWN_FRACTION_PER_STEP
            ),
            "theoretical_60_to_179_decline_bound_mw": (
                (2.0 / 3.0)
                * -float(
                    FULL_STF_ALIGNMENT_START_SEC
                    - 1
                    - MOMENT_STABILITY_START_SEC
                )
                * math.log10(1.0 - MAX_MOMENT_DOWN_FRACTION_PER_STEP)
            ),
            "theoretical_alignment_step_bound_mw": (
                (2.0 / 3.0)
                * -math.log10(
                    1.0 - FULL_STF_ALIGNMENT_DOWN_FRACTION_PER_STEP
                )
            ),
        },
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "multiscale_offsets": list(MULTISCALE_OFFSETS),
        "loss_weights": dict(LOSS_WEIGHTS),
        "loss_normalizers": dict(normalizers),
        "normalizer_path": str(normalizer_path),
        "normalizer_sha256": sha256_file(normalizer_path),
        "cache_root": str(cache.root),
        "cache_raw_rates_sha256": cache.manifest["raw_rates_sha256"],
        "cache_arrays_sha256": cache.manifest["arrays_sha256"],
        "validation_gates": dict(VALIDATION_GATES),
        "selection": (
            "within each seed choose the endpoint-preserving epoch with the "
            "lowest worst normalized streaming ratio; then choose one seed by "
            "the same score; never ensemble"
        ),
        "hidden_data": (
            "internal test, external development events, and grouped test are closed"
        ),
    }


def train_seed(
    *,
    seed: int,
    seed_root: Path,
    cache: CacheBundle,
    normalizers: Mapping[str, float],
    normalizer_path: Path,
    device: torch.device,
    smoke: bool,
) -> dict[str, Any]:
    seed_root.mkdir(parents=True, exist_ok=False)
    config = phase50_config()
    configure_runtime(seed, device)
    torch.manual_seed(seed)
    model, frozen_source = load_phase50_model(config, device=device)
    transition_parameters = _transition_parameters(model)
    initial_transition = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith("released_stf_transition.")
    }
    optimizer = torch.optim.AdamW(
        transition_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = _build_stf_rate_criterion(config, device)
    train_indices = np.flatnonzero(cache.arrays["split_code"] == 0)
    if len(train_indices) != EXPECTED_TRAIN_COUNT:
        raise ValueError("Phase50 training count changed")

    protocol = _protocol(
        cache=cache,
        normalizer_path=normalizer_path,
        normalizers=normalizers,
    )
    _write_json(seed_root / "protocol.json", protocol)
    with (seed_root / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    baseline_metrics, baseline_gate_metrics = evaluate_model(
        model,
        cache,
        config=config,
        device=device,
        max_batches=2 if smoke else None,
    )
    _write_json(seed_root / "baseline_validation_metrics.json", baseline_metrics)
    best_score = float("inf")
    best_epoch = 0
    best_metrics = dict(baseline_metrics)
    best_gate: dict[str, Any] | None = None
    atomic_torch_save(dict(model.state_dict()), seed_root / "best_model.pth")

    rows: list[dict[str, Any]] = []
    epoch_count = 1 if smoke else EPOCHS
    max_train_batches = 2 if smoke else None
    for epoch in range(1, epoch_count + 1):
        model.train()
        generator = np.random.default_rng(seed * 10_000 + epoch)
        shuffled = generator.permutation(train_indices)
        seen = 0
        total_sum = 0.0
        component_sums = {name: 0.0 for name in LOSS_WEIGHTS}
        diagnostic_sums = {
            "mean_gate": 0.0,
            "late_mean_gate": 0.0,
            "mean_downward_step_mw": 0.0,
            "target_mw_mae": 0.0,
            "endpoint_teacher_mw_mae": 0.0,
            "endpoint_L_MSE": 0.0,
            "endpoint_L_synth": 0.0,
            "endpoint_L_mag": 0.0,
        }
        for batch_index, start in enumerate(range(0, len(shuffled), BATCH_SIZE)):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            indices = shuffled[start : start + BATCH_SIZE]
            batch = _tensor_batch(cache, indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            components, diagnostics = stateful_loss_components(
                model,
                batch,
                criterion=criterion,
                config=config,
            )
            total = normalized_loss(components, normalizers)
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("Phase50 objective became non-finite")
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                transition_parameters,
                max_norm=GRAD_CLIP_NORM,
            )
            optimizer.step()
            batch_size = len(indices)
            seen += batch_size
            total_sum += float(total.detach().cpu()) * batch_size
            for name, value in components.items():
                component_sums[name] += float(value.detach().cpu()) * batch_size
            for name, value in diagnostics.items():
                diagnostic_sums[name] += float(value) * batch_size
        if seen == 0:
            raise ValueError("Phase50 training produced no samples")
        _assert_backbone_unchanged(model, frozen_source)

        validation_metrics, validation_gate_metrics = evaluate_model(
            model,
            cache,
            config=config,
            device=device,
            max_batches=2 if smoke else None,
        )
        gate = None if smoke else validation_gate(validation_metrics)
        row = {
            "epoch": epoch,
            "train_total_normalized_loss": total_sum / seen,
            **{
                f"train_{name}": component_sums[name] / seen
                for name in LOSS_WEIGHTS
            },
            **{
                f"train_{name}": diagnostic_sums[name] / seen
                for name in diagnostic_sums
            },
            **validation_metrics,
            **validation_gate_metrics,
            "endpoint_preserved": (
                None if gate is None else bool(gate["endpoint_preserved"])
            ),
            "selection_score": (
                None if gate is None else float(gate["selection_score"])
            ),
            "validation_gate_passed": (
                None if gate is None else bool(gate["passed"])
            ),
        }
        rows.append(row)
        if gate is not None and bool(gate["endpoint_preserved"]):
            score = float(gate["selection_score"])
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_metrics = dict(validation_metrics)
                best_gate = dict(gate)
                atomic_torch_save(
                    dict(model.state_dict()),
                    seed_root / "best_model.pth",
                )
        atomic_torch_save(dict(model.state_dict()), seed_root / "last_model.pth")
        _write_json(seed_root / "epoch_metrics.json", rows)
        _write_csv(
            seed_root / "epoch_metrics.csv",
            rows,
            fieldnames=tuple(rows[0]),
        )
        print(
            f"seed={seed} epoch={epoch}/{epoch_count} "
            f"train={row['train_total_normalized_loss']:.6f} "
            f"event={row['endpoint_event_mae']:.6f} "
            f"late={row['late_event_abs_step_p95_mw']:.6f} "
            f"down={row['event_downward_step_p95_mw']:.6f} "
            f"downmax={row['event_downward_step_max_mw']:.6f} "
            f"drift={row['event_peak_to_final_p95_mw']:.6f} "
            f"score={row['selection_score']}",
            flush=True,
        )

    if smoke:
        best_epoch = 1
        best_metrics = dict(validation_metrics)
        best_gate = None
        atomic_torch_save(dict(model.state_dict()), seed_root / "best_model.pth")
        status = "smoke_complete"
        passed = True
    else:
        passed = bool(best_gate and best_gate["passed"] and best_epoch > 0)
        status = "validation_gate_passed" if passed else "validation_gate_failed"

    _assert_backbone_unchanged(model, frozen_source)
    changed_transition = [
        name
        for name, value in model.state_dict().items()
        if name.startswith("released_stf_transition.")
        and not torch.equal(value.detach().cpu(), initial_transition[name])
    ]
    if not changed_transition:
        raise RuntimeError("Phase50 training changed no transition tensors")
    summary = {
        "phase": "Phase50",
        "seed": seed,
        "status": status,
        "passed": passed,
        "smoke": smoke,
        "selected_epoch": best_epoch,
        "selected_metrics": best_metrics,
        "selected_gate": best_gate,
        "baseline_metrics": baseline_metrics,
        "baseline_gate_metrics": baseline_gate_metrics,
        "changed_transition_tensor_count": len(changed_transition),
        "changed_transition_tensors": changed_transition,
        "backbone_frozen_and_unchanged": True,
        "best_model": {
            "path": str(seed_root / "best_model.pth"),
            "sha256": sha256_file(seed_root / "best_model.pth"),
        },
        "protocol": protocol,
        "provenance": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": current_git_commit(PROJECT_ROOT),
            "git_dirty": git_is_dirty(PROJECT_ROOT),
            "device": str(device),
            "source_artifact_sha256": validate_source_artifacts(),
            "internal_test_iterated": False,
            "external_data_loaded": False,
            "grouped_test_loaded": False,
        },
    }
    _write_json(seed_root / "summary.json", summary)
    return summary


def run_campaign(
    *,
    cache_root: Path,
    output_root: Path,
    normalizer_path: Path,
    device: torch.device,
    smoke: bool,
) -> dict[str, Any]:
    _validate_new_directory(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_root)
    normalizers = load_normalizers(normalizer_path, cache=cache)
    seeds = (42,) if smoke else SEEDS
    summaries = [
        train_seed(
            seed=seed,
            seed_root=output_root / f"seed_{seed}",
            cache=cache,
            normalizers=normalizers,
            normalizer_path=normalizer_path,
            device=device,
            smoke=smoke,
        )
        for seed in seeds
    ]
    if smoke:
        selected = summaries[0]
        status = "smoke_complete"
    else:
        passing = [summary for summary in summaries if summary["passed"]]
        selected = (
            min(
                passing,
                key=lambda summary: float(summary["selected_gate"]["selection_score"]),
            )
            if passing
            else None
        )
        status = "validation_gate_passed" if selected is not None else "validation_gate_failed"
    campaign = {
        "phase": "Phase50",
        "status": status,
        "passed": bool(smoke or selected is not None),
        "selected_seed": None if selected is None else int(selected["seed"]),
        "selected_epoch": (
            None if selected is None else int(selected["selected_epoch"])
        ),
        "seed_summaries": summaries,
        "internal_test_iterated": False,
        "external_data_loaded": False,
        "grouped_test_loaded": False,
    }
    _write_json(output_root / "campaign_summary.json", campaign)
    return campaign


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the model-internal Phase50 released-STF recurrent state; "
            "no external adapter or projection is used."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("audit", "smoke", "train"),
        required=True,
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--normalizer-path", type=Path)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    if args.stage == "audit":
        audit_loss_scales(
            cache_root=cache_root,
            output_root=output_root,
            device=device,
        )
        return 0
    if args.normalizer_path is None:
        raise SystemExit("--normalizer-path is required for smoke/train")
    campaign = run_campaign(
        cache_root=cache_root,
        output_root=output_root,
        normalizer_path=args.normalizer_path.resolve(),
        device=device,
        smoke=args.stage == "smoke",
    )
    print(json.dumps(campaign, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
