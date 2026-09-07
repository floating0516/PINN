"""Direction 1: causal Phase 39 trained on released-moment prefix labels.

What is kept from the validation-frozen causal Phase 39 protocol
(``run_phase39_causal_expanded_fixed_split.py``):

* the unchanged Phase 39 network (1,010,850 parameters, factorized STF head),
  trained from scratch on the fixed 24 / 6 / 9 split, seed 73;
* one random prefix ``h in [5, 199]`` per step plus the 200 s endpoint;
* ``L_synth`` with ``lambda_synth = 0.5`` on the observed prefix only;
* physically consistent counterfactual moment scaling;
* endpoint losses (full STF MSE + final magnitude) untouched.

What changes (and only this):

* prefix input is zero-padded to 200 s instead of truncated-and-stretched, so
  the STF time axis stays absolute;
* prefix STF / magnitude labels are causal: only source time
  ``< h - tau_P(station)`` is supervised, and the prefix magnitude target is
  the released moment ``B_ref(h)`` rather than the final catalog magnitude;
* the old error-descent term (final-magnitude error from a prefix vs. the
  endpoint) is switched off because that quantity is no longer supervised.

Test loader is never iterated here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments import run_phase39_causal_direct as direct  # noqa: E402
from scripts.experiments import run_phase39_expanded_fixed_split as fixed  # noqa: E402
from scripts.experiments.run_phase39_causal_expanded_fixed_split import (  # noqa: E402
    _load_fixed_source,
)
from src.evaluation.metrics import (  # noqa: E402
    aggregate_event_predictions,
    summarize_predictions,
)
from src.models.model import PINNModel  # noqa: E402
from src.training.checkpointing import atomic_torch_save  # noqa: E402
from src.training.loss_stf_rate_v2 import moment_magnitude_from_rate  # noqa: E402
from src.training.released_moment import (  # noqa: E402
    constrained_source_mask,
    released_magnitude,
    released_monotone_loss,
    released_prefix_losses,
    sample_distance_stratified_delta_mw,
    zero_pad_prefix,
)
from src.training.train import (  # noqa: E402
    _batch_event_sample_weights,
    _build_stf_rate_criterion,
    _prepare_v2_batch,
    _training_event_balance_weights,
)
from src.utils.device import configure_runtime  # noqa: E402
from src.utils.provenance import (  # noqa: E402
    current_git_commit,
    git_is_dirty,
    sha256_file,
)


DEFAULT_EXPERIMENT_CONFIG = (
    PROJECT_ROOT / "configs" / "experiments" / "phase39_causal_released_moment.yaml"
)
SOURCE_STEPS = direct.SOURCE_STEPS
PREFIX_PRESENTATIONS = ("zero_pad", "truncate")
RELEASED_METHOD = "released"          # B: integral over source time < h - tau_P
FINAL_METHOD = "final"                # A: full 200 s integral (old report quantity)
RELEASED_REFERENCE_METHOD = "released_ref"  # B applied to the SCARDEC reference STF
CONSTRAINED_SUFFIX = "_constrained"  # B*: event median over P-arrived stations only
MIN_REPORT_WINDOW_SEC = 1.0


def p_arrival_seconds(criterion: Any, source_distance_m: torch.Tensor) -> torch.Tensor:
    return criterion.travel_time.delays(source_distance_m).p_sec.reshape(-1)


def present_prefix(
    model_input: torch.Tensor,
    horizon_sec: int,
    *,
    presentation: str,
) -> torch.Tensor:
    if presentation == "zero_pad":
        return zero_pad_prefix(model_input, int(horizon_sec))
    if presentation == "truncate":
        return model_input[:, :, : int(horizon_sec)]
    raise ValueError(f"unsupported prefix presentation: {presentation!r}")


def reference_encoded(criterion: Any, stf_true: torch.Tensor) -> torch.Tensor:
    return torch.log10(1.0 + torch.clamp(stf_true, min=0.0) / criterion.stf_m_ref)


def released_prefix_loss(
    model: PINNModel,
    criterion: Any,
    prepared: Any,
    *,
    horizon_sec: int,
    sample_weights: torch.Tensor | None,
    presentation: str,
    floor_nm: float,
    min_constrained_window_sec: float,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    """Prefix loss = lambda_synth L_synth^h + lambda_MSE L_MSE^h + lambda_mag L_mag^h."""
    encoded = model(
        present_prefix(prepared.model_input, horizon_sec, presentation=presentation),
        meta=prepared.metadata,
    )
    synth_loss, synth_parts = criterion(
        encoded,
        pred_catalog_mw=None,
        radial_obs=prepared.radial[:, :, :horizon_sec],
        source_distance_m=prepared.source_distance_m,
        theta_deg=prepared.theta_deg,
        phi_slip_deg=prepared.phi_slip_deg,
        source_dt_sec=prepared.source_dt_sec,
        observation_dt_sec=prepared.observation_dt_sec,
        waveform_valid_mask=prepared.waveform_valid_mask[:, :horizon_sec],
        stf_true=None,
        has_stf=None,
        true_mag=None,
        sample_weights=sample_weights,
    )
    rate_hat = criterion._decode_rate(encoded)
    mask = constrained_source_mask(
        float(horizon_sec),
        p_arrival_seconds(criterion, prepared.source_distance_m),
        source_steps=SOURCE_STEPS,
        source_dt_sec=prepared.source_dt_sec,
    ).to(device=rate_hat.device, dtype=rate_hat.dtype)
    l_mse, l_mag, released_parts = released_prefix_losses(
        pred_encoded=encoded,
        rate_hat=rate_hat,
        ref_encoded=reference_encoded(criterion, prepared.stf_true),
        stf_true=prepared.stf_true,
        mask=mask,
        source_dt_sec=prepared.source_dt_sec,
        sample_weights=sample_weights,
        min_constrained_window_sec=min_constrained_window_sec,
        floor_nm=floor_nm,
    )
    loss = (
        synth_loss
        + float(criterion.lambda_MSE) * l_mse
        + float(criterion.lambda_mag) * l_mag
    )
    b_pred = released_magnitude(rate_hat, mask, prepared.source_dt_sec, floor_nm=floor_nm)
    parts = {
        "L_synth": float(synth_parts["L_synth"]),
        "L_MSE_causal": float(l_mse.detach().cpu()),
        "L_mag_released": float(l_mag.detach().cpu()),
        **released_parts,
    }
    return loss, parts, b_pred


def _station_rows(
    batch: Mapping[str, Any],
    predictions: torch.Tensor,
    *,
    method: str,
    horizon_sec: int,
    constrained_window_sec: torch.Tensor,
) -> list[dict[str, Any]]:
    predicted = predictions.reshape(-1).detach().cpu().numpy()
    window = constrained_window_sec.reshape(-1).detach().cpu().numpy()
    catalogs = batch["magnitude_catalog"].reshape(-1).detach().cpu().numpy()
    native = batch["mw_stf_native"].reshape(-1).detach().cpu().numpy()
    return [
        {
            "method": method,
            "observation_horizon_sec": int(horizon_sec),
            "event": str(batch["event"][index]),
            "station": str(batch["station"][index]),
            "mw_pred": float(predicted[index]),
            "mw_catalog": float(catalogs[index]),
            "mw_stf_native": float(native[index]),
            "constrained_window_sec": float(window[index]),
        }
        for index in range(len(predicted))
    ]


def evaluate_released_horizons(
    model: PINNModel,
    criterion: Any,
    config: dict[str, Any],
    loader: Any,
    *,
    horizons: Sequence[int],
    presentation: str,
    floor_nm: float,
    station_horizons: set[int] | None = None,
    max_batches: int | None = None,
    stf_sink: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate B (released), A (final) and B_ref at every horizon.

    ``stf_sink`` (optional) collects the predicted STF for every station and
    horizon as float16 so that the replay can be inspected offline.
    """
    device = next(model.parameters()).device
    methods = (RELEASED_METHOD, FINAL_METHOD, RELEASED_REFERENCE_METHOD)
    rows_by_method: dict[str, dict[int, list[dict[str, Any]]]] = {
        method: {int(h): [] for h in horizons} for method in methods
    }
    stf_chunks: list[np.ndarray] = []
    stf_keys: list[tuple[str, str]] = []
    original_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                prepared = _prepare_v2_batch(batch, config, device)
                tau = p_arrival_seconds(criterion, prepared.source_distance_m)
                batch_stf: list[np.ndarray] = []
                for horizon in horizons:
                    encoded = model(
                        present_prefix(
                            prepared.model_input,
                            int(horizon),
                            presentation=presentation,
                        ),
                        meta=prepared.metadata,
                    )
                    rate = criterion._decode_rate(encoded)
                    mask = constrained_source_mask(
                        float(horizon),
                        tau,
                        source_steps=SOURCE_STEPS,
                        source_dt_sec=prepared.source_dt_sec,
                    ).to(device=rate.device, dtype=rate.dtype)
                    window = mask.sum(dim=1) * prepared.source_dt_sec.reshape(-1)
                    values = {
                        RELEASED_METHOD: released_magnitude(
                            rate, mask, prepared.source_dt_sec, floor_nm=floor_nm
                        ),
                        FINAL_METHOD: moment_magnitude_from_rate(
                            rate, prepared.source_dt_sec
                        ),
                        RELEASED_REFERENCE_METHOD: released_magnitude(
                            prepared.stf_true,
                            mask,
                            prepared.source_dt_sec,
                            floor_nm=floor_nm,
                        ),
                    }
                    for method, mw in values.items():
                        rows_by_method[method][int(horizon)].extend(
                            _station_rows(
                                batch,
                                mw,
                                method=method,
                                horizon_sec=int(horizon),
                                constrained_window_sec=window,
                            )
                        )
                    if stf_sink is not None:
                        # Stored in units of m_ref (1e18 N m/s) as float32; raw N m/s
                        # overflows float16 and wastes float32 exponent range.
                        batch_stf.append(
                            (rate.detach() / float(criterion.stf_m_ref))
                            .to(torch.float32)
                            .cpu()
                            .numpy()
                        )
                if stf_sink is not None:
                    # (stations, horizons, source_steps)
                    stf_chunks.append(np.stack(batch_stf, axis=1))
                    stf_keys.extend(
                        (str(batch["event"][i]), str(batch["station"][i]))
                        for i in range(len(batch["event"]))
                    )
    finally:
        model.train(original_training)

    horizon_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []

    def _aggregate(method_name: str, horizon: int, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        events = aggregate_event_predictions(rows, reference_key="mw_catalog")
        event_rows.extend(
            {"method": method_name, "observation_horizon_sec": int(horizon), **row}
            for row in events
        )
        horizon_rows.append(
            {
                "method": method_name,
                "observation_horizon_sec": int(horizon),
                "reference": "catalog",
                **summarize_predictions(rows, events, reference_key="mw_catalog"),
            }
        )

    for method in methods:
        for horizon in horizons:
            rows = rows_by_method[method][int(horizon)]
            _aggregate(method, int(horizon), rows)
            if method in (RELEASED_METHOD, RELEASED_REFERENCE_METHOD):
                # B*: event median over stations whose constrained window is at
                # least 1 s (P has arrived), mirroring PGD's "wherever defined".
                _aggregate(
                    f"{method}{CONSTRAINED_SUFFIX}",
                    int(horizon),
                    [
                        {**row, "method": f"{method}{CONSTRAINED_SUFFIX}"}
                        for row in rows
                        if float(row["constrained_window_sec"]) >= MIN_REPORT_WINDOW_SEC
                    ],
                )
            if station_horizons is not None and int(horizon) in station_horizons:
                station_rows.extend(rows)
    if stf_sink is not None:
        stf_sink["stf_over_m_ref"] = (
            np.concatenate(stf_chunks, axis=0) if stf_chunks else np.zeros((0, 0, 0), dtype=np.float32)
        )
        stf_sink["m_ref_nm"] = float(criterion.stf_m_ref)
        stf_sink["keys"] = stf_keys
        stf_sink["horizons"] = [int(h) for h in horizons]
    return {
        "horizon_rows": horizon_rows,
        "event_rows": event_rows,
        "station_rows": station_rows,
    }


def _rows_for(rows: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row["method"] == method]


def _anchor_lookup(rows: Sequence[Mapping[str, Any]], method: str) -> dict[int, Mapping[str, Any]]:
    return direct._metric_lookup(_rows_for(rows, method))


def run_experiment(
    *,
    experiment_config_path: Path,
    output_root: Path,
    smoke: bool,
    device: torch.device,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    experiment = direct._read_yaml(experiment_config_path)
    training = experiment["training"]
    evaluation = experiment["evaluation"]
    source_seed = int(experiment["seed"])  # endpoint candidate whose config/split we reuse
    seed = int(experiment.get("training_seed", source_seed))  # RNG for init, prefix sampling, augmentation
    configure_runtime(seed, device)

    synth_weight = float(training["lambda_synth"])
    if synth_weight <= 0.0:
        raise ValueError("direction 1 keeps the synthesized-waveform loss; lambda_synth must be > 0")
    presentation = str(training.get("prefix_presentation", "zero_pad"))
    if presentation not in PREFIX_PRESENTATIONS:
        raise ValueError(f"prefix_presentation must be one of {PREFIX_PRESENTATIONS}")
    floor_nm = float(training.get("released_moment_floor_nm", 1.0e15))
    min_window = float(training.get("min_constrained_window_sec", 1.0))
    if not math.isfinite(floor_nm) or floor_nm <= 0.0 or not math.isfinite(min_window) or min_window < 0.0:
        raise ValueError("invalid released-moment floor / minimum window")
    if float(training.get("error_descent_weight", 0.0)) != 0.0:
        raise ValueError(
            "error_descent_weight must be 0 here: the prefix no longer predicts the "
            "final magnitude, so the old descent term has no target"
        )
    monotone_weight = float(training.get("released_monotone_weight", 0.0))
    monotone_slack = float(training.get("released_monotone_slack_mw", 0.03))
    prefix_pair_gap_sec = int(training.get("prefix_pair_gap_sec", 0))
    if monotone_weight < 0.0 or not math.isfinite(monotone_weight):
        raise ValueError("released_monotone_weight must be finite and nonnegative")
    if monotone_weight > 0.0 and prefix_pair_gap_sec < 1:
        raise ValueError("released_monotone_weight requires prefix_pair_gap_sec >= 1")
    moment_scale_weight = float(training.get("moment_scale_augmentation_weight", 0.0))
    moment_scale_min_mw = float(training.get("moment_scale_delta_min_mw", -0.75))
    moment_scale_max_mw = float(training.get("moment_scale_delta_max_mw", 0.5))
    moment_scale_stratified = bool(training.get("moment_scale_distance_stratified", False))
    moment_scale_near_km = float(training.get("moment_scale_near_km", 100.0))
    moment_scale_far_km = float(training.get("moment_scale_far_km", 300.0))
    moment_scale_far_min_mw = float(training.get("moment_scale_far_delta_min_mw", -1.5))
    moment_scale_far_max_mw = float(training.get("moment_scale_far_delta_max_mw", 0.0))
    if moment_scale_weight < 0.0 or (
        moment_scale_weight > 0.0 and moment_scale_min_mw >= moment_scale_max_mw
    ):
        raise ValueError("invalid moment-scale augmentation settings")
    if moment_scale_stratified and (
        moment_scale_far_min_mw >= moment_scale_far_max_mw or moment_scale_far_km <= moment_scale_near_km
    ):
        raise ValueError("invalid distance-stratified moment-scale settings")
    curriculum_start_epoch = int(training.get("prefix_curriculum_start_epoch", 0))
    curriculum_full_epoch = int(training.get("prefix_curriculum_full_epoch", 1))
    direct.causal_curriculum_scale(
        1, start_epoch=curriculum_start_epoch, full_epoch=curriculum_full_epoch
    )

    source = _load_fixed_source(experiment, device=device)
    config = source["config"]
    config["training"]["stf_rate_loss"]["lambda_synth"] = synth_weight
    criterion = _build_stf_rate_criterion(config, device)
    if not criterion.origin_aligned:
        raise ValueError("released-moment labels require origin-aligned (absolute) delays")

    model = PINNModel(config).to(device)
    initialization = str(training.get("initialization", "scratch"))
    direct.initialize_direct_model(model, source["state"], initialization=initialization)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != direct.EXPECTED_PHASE39_PARAMETER_COUNT:
        raise ValueError(
            f"parameter count differs from Phase39: {parameter_count} != "
            f"{direct.EXPECTED_PHASE39_PARAMETER_COUNT}"
        )
    trainable_scope = str(training.get("trainable_scope", "all"))
    trainable_parameters = direct.configure_trainable_scope(model, scope=trainable_scope)
    trainable_parameter_count = sum(p.numel() for p in trainable_parameters)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(training.get("scheduler_T0", 15)),
        T_mult=int(training.get("scheduler_T_mult", 2)),
        eta_min=float(training.get("scheduler_eta_min", 1.0e-6)),
    )
    warmup_epochs = int(training.get("warmup_epochs", 5))
    event_weights = _training_event_balance_weights(config, source["train_loader"])
    anchor_horizons = tuple(int(v) for v in evaluation["anchor_horizons_sec"])
    max_eval_batches = 2 if smoke else None

    protocol = {
        "method": "causal-prefix Phase39 with released-moment labels (direction 1)",
        "architecture_changed": False,
        "adapter_used": False,
        "teacher_used": False,
        "recurrent_state_used": False,
        "handcrafted_features_used": False,
        "total_parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "initialization": initialization,
        "trainable_scope": trainable_scope,
        "source_checkpoint": str(source["checkpoint_path"]),
        "source_checkpoint_sha256": sha256_file(source["checkpoint_path"]),
        "source_config": str(source["config_path"]),
        "source_config_sha256": sha256_file(source["config_path"]),
        "split_assignment_sha256": source["split_manifest"]["assignment_sha256"],
        "split_protocol": "expanded_fixed_24_train_6_validation_9_test",
        "development_role": "fixed_validation_only",
        "train_event_count": len(fixed.TRAIN_EVENTS),
        "validation_event_count": len(fixed.VALIDATION_EVENTS),
        "held_out_test_event_count": len(fixed.TEST_EVENTS),
        "held_out_test_loader_iterated": False,
        "seed": seed,
        "source_candidate_seed": source_seed,
        "lambda_synth": synth_weight,
        "lambda_MSE": float(criterion.lambda_MSE),
        "lambda_mag": float(criterion.lambda_mag),
        "moment_scale_augmentation_weight": moment_scale_weight,
        "moment_scale_delta_min_mw": moment_scale_min_mw,
        "moment_scale_delta_max_mw": moment_scale_max_mw,
        "moment_scale_distance_stratified": moment_scale_stratified,
        "moment_scale_near_km": moment_scale_near_km,
        "moment_scale_far_km": moment_scale_far_km,
        "moment_scale_far_delta_min_mw": moment_scale_far_min_mw,
        "moment_scale_far_delta_max_mw": moment_scale_far_max_mw,
        "prefix_presentation": presentation,
        "prefix_pair_gap_sec": prefix_pair_gap_sec,
        "error_descent_weight": 0.0,
        "released_monotone_weight": monotone_weight,
        "released_monotone_slack_mw": monotone_slack,
        "released_moment_floor_nm": floor_nm,
        "min_constrained_window_sec": min_window,
        "event_balanced_sampling": bool(config["training"].get("event_balanced_sampling", False)),
        "event_balance_estimator": str(
            config["training"].get("event_balance_estimator", "replacement_sampling")
        ),
        "released_moment_definition": (
            "B(h) = Mw(sum_k clamp(STF_k,0) * m_k * dt) with m_k = clip((h - tau_P)/dt - k, 0, 1); "
            "tau_P = hypocentral_distance / alpha (origin-aligned axis). Label uses the SCARDEC "
            "reference STF, prediction uses the model STF from the same prefix. "
            "A(h) = full 200 s integral (reported for reference only, unsupervised on prefixes)."
        ),
        "prefix_supervision": (
            "L_prefix = lambda_synth L_synth[0,h) + lambda_MSE L_MSE(source < h - tau_P) "
            "+ lambda_mag (B_pred - B_ref)^2; endpoint loss unchanged"
        ),
        "synth_polarity_mode": criterion.synth_polarity_mode,
        "radiation_coefficient_contract": criterion.radiation_coefficient_contract,
        "include_intermediate_field": bool(criterion.include_intermediate),
        "test_split_iterated": False,
        "external_events_loaded": False,
        "git_commit": current_git_commit(PROJECT_ROOT),
        "git_dirty": git_is_dirty(PROJECT_ROOT),
        "device": str(device),
        "smoke": bool(smoke),
        "experiment_config": experiment,
    }
    direct._write_json(output_root / "protocol.json", protocol)
    direct._write_json(output_root / "split.json", source["split_manifest"])
    (output_root / "experiment_config.yaml").write_text(
        yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    best_path = output_root / "best_model.pth"
    best_epoch = -1
    best_endpoint = math.inf
    patience = 0
    global_step = 0
    epoch_rows: list[dict[str, Any]] = []
    epochs = 1 if smoke else int(training["epochs"])
    max_train_batches = 2 if smoke else None

    for epoch in range(1, epochs + 1):
        curriculum_scale = direct.causal_curriculum_scale(
            epoch, start_epoch=curriculum_start_epoch, full_epoch=curriculum_full_epoch
        )
        effective_prefix_weight = float(training["prefix_science_weight"]) * curriculum_scale
        effective_monotone_weight = monotone_weight * curriculum_scale
        model.train()
        totals = {
            "total": 0.0,
            "prefix": 0.0,
            "endpoint": 0.0,
            "synth": 0.0,
            "mse_causal": 0.0,
            "mag_released": 0.0,
            "monotone": 0.0,
            "included": 0.0,
            "window": 0.0,
            "released_abs_error": 0.0,
        }
        seen = 0
        for batch_index, batch in enumerate(source["train_loader"]):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            prepared = _prepare_v2_batch(batch, config, device)
            weights = _batch_event_sample_weights(batch, event_weights, reference=prepared.radial)
            if moment_scale_weight > 0.0:
                if moment_scale_stratified:
                    delta_mw = sample_distance_stratified_delta_mw(
                        prepared.source_distance_m,
                        near_km=moment_scale_near_km,
                        far_km=moment_scale_far_km,
                        near_min_mw=moment_scale_min_mw,
                        near_max_mw=moment_scale_max_mw,
                        far_min_mw=moment_scale_far_min_mw,
                        far_max_mw=moment_scale_far_max_mw,
                    ).to(device=prepared.true_mag.device, dtype=prepared.true_mag.dtype)
                else:
                    delta_mw = torch.empty_like(prepared.true_mag).uniform_(
                        moment_scale_min_mw, moment_scale_max_mw
                    )
                prepared, weights = direct.augment_prepared_with_moment_scaling(
                    prepared, weights, delta_mw=delta_mw, augmentation_weight=moment_scale_weight
                )
            optimizer.zero_grad(set_to_none=True)

            monotone_loss = prepared.radial.new_zeros(())
            if curriculum_scale <= 0.0:
                prefix_loss = prepared.radial.new_zeros(())
                prefix_parts = {
                    "L_synth": 0.0,
                    "L_MSE_causal": 0.0,
                    "L_mag_released": 0.0,
                    "included_fraction": 0.0,
                    "mean_constrained_window_sec": 0.0,
                    "released_abs_error_mw": 0.0,
                }
            else:
                if prefix_pair_gap_sec > 0:
                    early_h, late_h = direct.ordered_prefix_horizons_for_step(
                        global_step,
                        seed=seed,
                        minimum=int(training["minimum_prefix_horizon_sec"]),
                        maximum=int(training["maximum_prefix_horizon_sec"]),
                        multiplier=int(training["horizon_cycle_multiplier"]),
                        gap_sec=prefix_pair_gap_sec,
                    )
                    horizons_this_step = (early_h, late_h)
                else:
                    horizons_this_step = (
                        direct.prefix_horizon_for_step(
                            global_step,
                            seed=seed,
                            minimum=int(training["minimum_prefix_horizon_sec"]),
                            maximum=int(training["maximum_prefix_horizon_sec"]),
                            multiplier=int(training["horizon_cycle_multiplier"]),
                        ),
                    )
                losses, parts_list, b_values = [], [], []
                for h in horizons_this_step:
                    loss_h, parts_h, b_h = released_prefix_loss(
                        model,
                        criterion,
                        prepared,
                        horizon_sec=int(h),
                        sample_weights=weights,
                        presentation=presentation,
                        floor_nm=floor_nm,
                        min_constrained_window_sec=min_window,
                    )
                    losses.append(loss_h)
                    parts_list.append(parts_h)
                    b_values.append(b_h)
                prefix_loss = torch.stack(losses).mean()
                prefix_parts = {
                    key: float(np.mean([p[key] for p in parts_list])) for key in parts_list[0]
                }
                if effective_monotone_weight > 0.0 and len(b_values) == 2:
                    monotone_loss = released_monotone_loss(
                        b_values[0], b_values[1], slack_mw=monotone_slack, sample_weights=weights
                    )

            endpoint_loss, endpoint_parts, _ = direct._science_loss(
                model, criterion, prepared, horizon_sec=SOURCE_STEPS, sample_weights=weights
            )
            total_loss = (
                effective_prefix_weight * prefix_loss
                + float(training["endpoint_science_weight"]) * endpoint_loss
                + effective_monotone_weight * monotone_loss
            )
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("released-moment causal loss became non-finite")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(training["grad_clip_norm"]))
            optimizer.step()

            batch_size = int(prepared.radial.shape[0])
            seen += batch_size
            totals["total"] += float(total_loss.detach().cpu()) * batch_size
            totals["prefix"] += float(prefix_loss.detach().cpu()) * batch_size
            totals["endpoint"] += float(endpoint_loss.detach().cpu()) * batch_size
            totals["synth"] += 0.5 * (prefix_parts["L_synth"] + float(endpoint_parts["L_synth"])) * batch_size
            totals["mse_causal"] += prefix_parts["L_MSE_causal"] * batch_size
            totals["mag_released"] += prefix_parts["L_mag_released"] * batch_size
            totals["monotone"] += float(monotone_loss.detach().cpu()) * batch_size
            totals["included"] += prefix_parts["included_fraction"] * batch_size
            totals["window"] += prefix_parts["mean_constrained_window_sec"] * batch_size
            totals["released_abs_error"] += prefix_parts["released_abs_error_mw"] * batch_size
            global_step += 1
        if seen == 0:
            raise ValueError("training loader produced no samples")

        validation = evaluate_released_horizons(
            model,
            criterion,
            config,
            source["validation_loader"],
            horizons=anchor_horizons,
            presentation=presentation,
            floor_nm=floor_nm,
            max_batches=max_eval_batches,
        )
        released_lookup = _anchor_lookup(validation["horizon_rows"], RELEASED_METHOD)
        final_lookup = _anchor_lookup(validation["horizon_rows"], FINAL_METHOD)
        endpoint_mae = float(released_lookup[SOURCE_STEPS]["event_mae"])
        improved = endpoint_mae < best_endpoint - 1.0e-6
        if improved:
            best_endpoint = endpoint_mae
            best_epoch = epoch
            patience = 0
            atomic_torch_save(dict(model.state_dict()), best_path)
        else:
            patience += 1
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_total_loss": totals["total"] / seen,
            "train_prefix_loss": totals["prefix"] / seen,
            "train_endpoint_loss": totals["endpoint"] / seen,
            "train_mean_synth_loss": totals["synth"] / seen,
            "train_prefix_causal_stf_mse": totals["mse_causal"] / seen,
            "train_prefix_released_mag_loss": totals["mag_released"] / seen,
            "train_released_monotone_loss": totals["monotone"] / seen,
            "train_prefix_included_fraction": totals["included"] / seen,
            "train_prefix_mean_constrained_window_sec": totals["window"] / seen,
            "train_prefix_released_abs_error_mw": totals["released_abs_error"] / seen,
            "causal_curriculum_scale": curriculum_scale,
            "effective_prefix_weight": effective_prefix_weight,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_endpoint_event_mae": endpoint_mae,
            "best_validation_endpoint_event_mae": best_endpoint,
            "best_epoch": best_epoch,
            "improved": improved,
            **{
                f"validation_released_event_mae_{h:03d}s": float(released_lookup[h]["event_mae"])
                for h in anchor_horizons
            },
            **{
                f"validation_final_event_mae_{h:03d}s": float(final_lookup[h]["event_mae"])
                for h in anchor_horizons
            },
        }
        epoch_rows.append(row)
        direct._write_csv(output_root / "training_epoch_metrics.csv", epoch_rows)
        print(
            f"epoch={epoch:03d} loss={row['train_total_loss']:.6f} "
            f"endpoint={endpoint_mae:.6f} best={best_endpoint:.6f} "
            f"B30={row['validation_released_event_mae_030s']:.3f} "
            f"B60={row['validation_released_event_mae_060s']:.3f} "
            f"B120={row['validation_released_event_mae_120s']:.3f} "
            f"incl={row['train_prefix_included_fraction']:.2f}",
            flush=True,
        )
        if epoch >= warmup_epochs:
            scheduler.step()
        if not smoke and patience >= int(training["early_stop_patience"]):
            break

    if best_epoch < 0:
        raise RuntimeError("no epoch improved the validation endpoint; nothing saved")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True), strict=True)
    full_horizons = tuple(
        range(int(evaluation["full_horizon_start_sec"]), int(evaluation["full_horizon_end_sec"]) + 1)
    )
    if smoke:
        full_horizons = anchor_horizons
    stf_sink: dict[str, Any] = {}
    final = evaluate_released_horizons(
        model,
        criterion,
        config,
        source["validation_loader"],
        horizons=full_horizons,
        presentation=presentation,
        floor_nm=floor_nm,
        station_horizons=set(anchor_horizons),
        max_batches=max_eval_batches,
        stf_sink=stf_sink,
    )
    direct._write_csv(output_root / "validation_horizon_metrics.csv", final["horizon_rows"])
    direct._write_csv(output_root / "validation_event_predictions.csv", final["event_rows"])
    direct._write_csv(
        output_root / "validation_anchor_station_predictions.csv", final["station_rows"]
    )
    np.savez_compressed(
        output_root / "validation_prefix_stf.npz",
        stf_over_m_ref=stf_sink["stf_over_m_ref"],
        m_ref_nm=np.asarray(stf_sink["m_ref_nm"]),
        events=np.asarray([k[0] for k in stf_sink["keys"]]),
        stations=np.asarray([k[1] for k in stf_sink["keys"]]),
        horizons_sec=np.asarray(stf_sink["horizons"], dtype=np.int16),
    )

    trajectories = {
        method: direct.summarize_trajectory(
            _rows_for(final["horizon_rows"], method),
            _rows_for(final["event_rows"], method),
            endpoint_band_tolerance_mw=float(evaluation["endpoint_band_tolerance_mw"]),
        )
        for method in (RELEASED_METHOD, FINAL_METHOD, RELEASED_REFERENCE_METHOD)
    }
    target = float(evaluation["endpoint_target_mw"])
    stretch = float(evaluation["endpoint_stretch_target_mw"])
    summary = {
        "status": "complete",
        "best_epoch": best_epoch,
        "selection_metric": "validation released (=final at 200 s) event MAE at 200 s",
        "released_endpoint_event_mae_mw": trajectories[RELEASED_METHOD]["endpoint_event_mae_mw"],
        "endpoint_target_mw": target,
        "endpoint_target_passed": trajectories[RELEASED_METHOD]["endpoint_event_mae_mw"] < target,
        "endpoint_stretch_target_mw": stretch,
        "endpoint_stretch_target_passed": (
            trajectories[RELEASED_METHOD]["endpoint_event_mae_mw"] < stretch
        ),
        "lambda_synth": synth_weight,
        "prefix_presentation": presentation,
        "released_moment_floor_nm": floor_nm,
        "min_constrained_window_sec": min_window,
        "moment_scale_augmentation_weight": moment_scale_weight,
        "moment_scale_delta_min_mw": moment_scale_min_mw,
        "moment_scale_delta_max_mw": moment_scale_max_mw,
        "moment_scale_distance_stratified": moment_scale_stratified,
        "error_descent_weight": 0.0,
        "released_monotone_weight": monotone_weight,
        "initialization": initialization,
        "trainable_scope": trainable_scope,
        "trainable_parameter_count": trainable_parameter_count,
        "architecture_changed": False,
        "test_split_iterated": False,
        "held_out_test_loader_iterated": False,
        "external_events_loaded": False,
        "split_protocol": "expanded_fixed_24_train_6_validation_9_test",
        "development_role": "fixed_validation_only",
        "trajectory": trajectories,
        "checkpoint_path": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
        "validation_prefix_stf_path": str(output_root / "validation_prefix_stf.npz"),
    }
    direct._write_json(output_root / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train causal-prefix Phase 39 with released-moment prefix labels on the "
            "frozen expanded fixed split (validation only; test never iterated)."
        )
    )
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root or (
        Path("/home/lihe/PINN_Mag/runs") / f"phase39-causal-released-moment-{timestamp}"
    )
    summary = run_experiment(
        experiment_config_path=args.experiment_config.resolve(),
        output_root=output_root.resolve(),
        smoke=bool(args.smoke),
        device=device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
