from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

from src.data.splits import (
    resolve_event_balance_exponent,
    resolve_event_balance_estimator,
)


_FORBIDDEN_PATHS = (
    ("dataset", "p_velocity_mps"),
    ("physics", "attenuation"),
    ("physics", "geometrical_spreading_factor"),
    ("physics", "free_surface_factor"),
    ("training", "stf_rate_loss", "skip_travel_delays"),
    ("training", "stf_rate_loss", "lambda_nonneg"),
)


def _lookup(config: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False, None
        node = node[key]
    return True, node


def _required(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    exists, value = _lookup(config, path)
    if not exists:
        raise ValueError(f"第二版配置缺少必需键: {'.'.join(path)}")
    return value


def _as_float(config: dict[str, Any], path: tuple[str, ...]) -> float:
    value = _required(config, path)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"第二版配置要求有限数值: {'.'.join(path)}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"第二版配置要求有限数值: {'.'.join(path)}")
    return number


def _as_int(config: dict[str, Any], path: tuple[str, ...]) -> int:
    value = _required(config, path)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"第二版配置要求整数: {'.'.join(path)}")
    return int(value)


def _require_value(
    config: dict[str, Any],
    path: tuple[str, ...],
    expected: Any,
) -> None:
    value = _required(config, path)
    if value != expected or type(value) is not type(expected):
        raise ValueError(
            f"第二版配置要求 {'.'.join(path)}={expected!r}"
        )


def _validate_station_random_workflow(config: dict[str, Any]) -> None:
    station_duration = _as_float(
        config,
        ("dataset", "stf", "station_window_duration_sec"),
    )
    if station_duration != 300.0:
        raise ValueError(
            "dataset.stf.station_window_duration_sec 必须严格等于 300.0"
        )
    _require_value(
        config,
        ("dataset", "stf", "station_alignment"),
        "p_arrival",
    )
    _require_value(
        config,
        ("dataset", "stf", "station_preserve_integral"),
        False,
    )
    _require_value(
        config,
        ("dataset", "stf", "magnitude_target"),
        "stf_native",
    )
    _require_value(
        config,
        ("physics", "travel_time_model"),
        "constant_velocity",
    )
    _require_value(
        config,
        ("physics", "delay_mode"),
        "p_aligned_relative",
    )
    _require_value(
        config,
        ("model", "predict_catalog_mw"),
        True,
    )
    _as_float(config, ("model", "catalog_mw_initial_bias"))
    _require_value(
        config,
        ("training", "split_protocol"),
        "within_event_station",
    )
    event_balanced_sampling = _required(
        config,
        ("training", "event_balanced_sampling"),
    )
    if not isinstance(event_balanced_sampling, bool):
        raise ValueError("training.event_balanced_sampling must be boolean")
    resolve_event_balance_estimator(config["training"])
    resolve_event_balance_exponent(config["training"])
    if _as_int(config, ("training", "early_stop_patience")) != 0:
        raise ValueError("training.early_stop_patience 必须严格等于 0")
    checkpoint_metric = _required(
        config,
        ("training", "checkpoint_metric"),
    )
    if not isinstance(checkpoint_metric, str) or checkpoint_metric not in {
        "station_mae_catalog",
        "event_mae_catalog",
    }:
        raise ValueError(
            "training.checkpoint_metric must be station_mae_catalog "
            "or event_mae_catalog"
        )
    _require_value(
        config,
        ("evaluation", "external_role"),
        "sanity",
    )


def validate_config_on_startup(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("配置必须是映射")
    if "pipeline_version" not in config:
        return

    pipeline_version = config["pipeline_version"]
    if isinstance(pipeline_version, bool) or not isinstance(
        pipeline_version, Integral
    ):
        raise ValueError("pipeline_version 必须是整数 1 或 2")
    if pipeline_version == 1:
        return
    if pipeline_version != 2:
        raise ValueError("pipeline_version 必须是整数 1 或 2")

    validate_config_v2(config)


def waveform_input_components_from_config(
    config: dict[str, Any],
) -> tuple[str, ...]:
    value = (config.get("model", {}) or {}).get(
        "input_components",
        ["radial"],
    )
    if not isinstance(value, (list, tuple)):
        raise ValueError("model.input_components must be a sequence")
    components = tuple(value)
    if components not in {
        ("radial",),
        ("radial", "tangential"),
    }:
        raise ValueError(
            "model.input_components must be ['radial'] or "
            "['radial', 'tangential']"
        )
    return components


def radial_dynamic_range_stem_from_config(config: dict[str, Any]) -> str:
    model = config.get("model", {}) or {}
    value = model.get("radial_dynamic_range_stem", "none")
    if not isinstance(value, str) or value not in {
        "none",
        "asinh_residual",
    }:
        raise ValueError(
            "model.radial_dynamic_range_stem must be none or asinh_residual"
        )
    if (
        value == "asinh_residual"
        and waveform_input_components_from_config(config) != ("radial",)
    ):
        raise ValueError(
            "model.radial_dynamic_range_stem=asinh_residual requires R-only input"
        )
    return value


def moment_linear_skip_from_config(config: dict[str, Any]) -> bool:
    model = config.get("model", {}) or {}
    value = model.get("moment_linear_skip", False)
    if not isinstance(value, bool):
        raise ValueError("model.moment_linear_skip must be boolean")
    if (
        value
        and model.get("stf_output_parameterization", "direct")
        != "moment_shape_factorized"
    ):
        raise ValueError(
            "model.moment_linear_skip=true requires "
            "model.stf_output_parameterization=moment_shape_factorized"
        )
    return value


def moment_head_dropout_from_config(config: dict[str, Any]) -> bool:
    model = config.get("model", {}) or {}
    value = model.get("moment_head_dropout", True)
    if not isinstance(value, bool):
        raise ValueError("model.moment_head_dropout must be boolean")
    if (
        not value
        and model.get("stf_output_parameterization", "direct")
        != "moment_shape_factorized"
    ):
        raise ValueError(
            "model.moment_head_dropout=false requires "
            "model.stf_output_parameterization=moment_shape_factorized"
        )
    return value


def stateful_streaming_config_from_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    model = config.get("model", {}) or {}
    raw = model.get("stateful_streaming", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("model.stateful_streaming must be a mapping")
    mode = raw.get("mode", "none")
    if not isinstance(mode, str) or mode not in {
        "none",
        "released_stf_gru",
    }:
        raise ValueError(
            "model.stateful_streaming.mode must be none or released_stf_gru"
        )
    if mode == "none":
        extra = set(raw) - {"mode"}
        if extra:
            raise ValueError(
                "model.stateful_streaming parameters require "
                "mode=released_stf_gru"
            )
        return {"mode": "none"}
    if model.get("stf_output_parameterization", "direct") != (
        "moment_shape_factorized"
    ):
        raise ValueError(
            "released_stf_gru requires "
            "model.stf_output_parameterization=moment_shape_factorized"
        )
    if waveform_input_components_from_config(config) != ("radial",):
        raise ValueError("released_stf_gru requires R-only input")

    allowed = {
        "mode",
        "local_channels",
        "hidden_size",
        "support_ramp_sec",
        "initial_gate_logit",
        "max_proposal_correction_log10",
        "max_moment_down_fraction_per_step",
        "early_moment_down_fraction_per_step",
        "moment_stability_start_sec",
        "max_moment_proposal_correction_log10",
        "use_moment_rebase_window",
        "moment_rebase_start_sec",
        "use_full_stf_alignment",
        "full_stf_alignment_start_sec",
        "full_stf_alignment_down_fraction_per_step",
    }
    extra = set(raw) - allowed
    if extra:
        raise ValueError(
            "unknown model.stateful_streaming keys: "
            + ", ".join(sorted(extra))
        )

    def positive_int(name: str, default: int) -> int:
        value = raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"model.stateful_streaming.{name} must be a positive integer"
            )
        result = int(value)
        if result < 1:
            raise ValueError(
                f"model.stateful_streaming.{name} must be a positive integer"
            )
        return result

    def finite_float(name: str, default: float) -> float:
        value = raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"model.stateful_streaming.{name} must be finite"
            )
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(
                f"model.stateful_streaming.{name} must be finite"
            )
        return result

    support_ramp_sec = finite_float("support_ramp_sec", 6.0)
    if support_ramp_sec <= 0.0:
        raise ValueError(
            "model.stateful_streaming.support_ramp_sec must be positive"
        )
    max_proposal_correction_log10 = finite_float(
        "max_proposal_correction_log10",
        1.0,
    )
    if max_proposal_correction_log10 <= 0.0:
        raise ValueError(
            "model.stateful_streaming.max_proposal_correction_log10 "
            "must be positive"
        )
    max_moment_down_fraction_per_step = finite_float(
        "max_moment_down_fraction_per_step",
        0.003,
    )
    if not 0.0 < max_moment_down_fraction_per_step < 1.0:
        raise ValueError(
            "model.stateful_streaming.max_moment_down_fraction_per_step "
            "must be between zero and one"
        )
    early_moment_down_fraction_per_step = finite_float(
        "early_moment_down_fraction_per_step",
        0.1,
    )
    if not 0.0 < early_moment_down_fraction_per_step < 1.0:
        raise ValueError(
            "model.stateful_streaming.early_moment_down_fraction_per_step "
            "must be between zero and one"
        )
    if (
        early_moment_down_fraction_per_step
        < max_moment_down_fraction_per_step
    ):
        raise ValueError(
            "model.stateful_streaming.early_moment_down_fraction_per_step "
            "must be at least max_moment_down_fraction_per_step"
        )
    max_moment_proposal_correction_log10 = finite_float(
        "max_moment_proposal_correction_log10",
        3.0,
    )
    if max_moment_proposal_correction_log10 <= 0.0:
        raise ValueError(
            "model.stateful_streaming.max_moment_proposal_correction_log10 "
            "must be positive"
        )
    use_moment_rebase_window = raw.get(
        "use_moment_rebase_window",
        False,
    )
    if not isinstance(use_moment_rebase_window, bool):
        raise ValueError(
            "model.stateful_streaming.use_moment_rebase_window "
            "must be boolean"
        )
    moment_stability_start_sec = positive_int(
        "moment_stability_start_sec",
        60,
    )
    moment_rebase_start_sec = positive_int(
        "moment_rebase_start_sec",
        40,
    )
    if moment_rebase_start_sec > moment_stability_start_sec:
        raise ValueError(
            "model.stateful_streaming.moment_rebase_start_sec must not "
            "exceed moment_stability_start_sec"
        )
    use_full_stf_alignment = raw.get("use_full_stf_alignment", False)
    if not isinstance(use_full_stf_alignment, bool):
        raise ValueError(
            "model.stateful_streaming.use_full_stf_alignment must be boolean"
        )
    full_stf_alignment_start_sec = positive_int(
        "full_stf_alignment_start_sec",
        180,
    )
    full_stf_alignment_down_fraction_per_step = finite_float(
        "full_stf_alignment_down_fraction_per_step",
        0.03,
    )
    if not 0.0 < full_stf_alignment_down_fraction_per_step < 1.0:
        raise ValueError(
            "model.stateful_streaming."
            "full_stf_alignment_down_fraction_per_step must be between "
            "zero and one"
        )
    if (
        use_full_stf_alignment
        and full_stf_alignment_start_sec <= moment_stability_start_sec
    ):
        raise ValueError(
            "model.stateful_streaming.full_stf_alignment_start_sec must "
            "be after moment_stability_start_sec"
        )
    return {
        "mode": mode,
        "local_channels": positive_int("local_channels", 4),
        "hidden_size": positive_int("hidden_size", 8),
        "support_ramp_sec": support_ramp_sec,
        "initial_gate_logit": finite_float("initial_gate_logit", -4.0),
        "max_proposal_correction_log10": max_proposal_correction_log10,
        "max_moment_down_fraction_per_step": (
            max_moment_down_fraction_per_step
        ),
        "early_moment_down_fraction_per_step": (
            early_moment_down_fraction_per_step
        ),
        "moment_stability_start_sec": moment_stability_start_sec,
        "max_moment_proposal_correction_log10": (
            max_moment_proposal_correction_log10
        ),
        "use_moment_rebase_window": use_moment_rebase_window,
        "moment_rebase_start_sec": moment_rebase_start_sec,
        "use_full_stf_alignment": use_full_stf_alignment,
        "full_stf_alignment_start_sec": full_stf_alignment_start_sec,
        "full_stf_alignment_down_fraction_per_step": (
            full_stf_alignment_down_fraction_per_step
        ),
    }


def magnitude_penalty_from_config(config: dict[str, Any]) -> str:
    training = config.get("training", {}) or {}
    loss_config = training.get("stf_rate_loss", {}) or {}
    if not isinstance(loss_config, dict):
        raise ValueError("training.stf_rate_loss must be a mapping")
    value = loss_config.get("magnitude_penalty", "squared")
    if not isinstance(value, str) or value not in {"squared", "absolute"}:
        raise ValueError(
            "training.stf_rate_loss.magnitude_penalty must be "
            "squared or absolute"
        )
    return value


def synth_polarity_mode_from_config(config: dict[str, Any]) -> str:
    training = config.get("training", {}) or {}
    loss_config = training.get("stf_rate_loss", {}) or {}
    if not isinstance(loss_config, dict):
        raise ValueError("training.stf_rate_loss must be a mapping")
    value = loss_config.get("synth_polarity_mode", "signed")
    if not isinstance(value, str) or value not in {
        "signed",
        "global_invariant",
    }:
        raise ValueError(
            "training.stf_rate_loss.synth_polarity_mode must be "
            "signed or global_invariant"
        )
    return value


def radiation_coefficient_contract_from_config(config: dict[str, Any]) -> str:
    training = config.get("training", {}) or {}
    loss_config = training.get("stf_rate_loss", {}) or {}
    if not isinstance(loss_config, dict):
        raise ValueError("training.stf_rate_loss must be a mapping")
    value = loss_config.get(
        "radiation_coefficient_contract",
        "horizontal_projected",
    )
    if not isinstance(value, str) or value not in {
        "horizontal_projected",
        "glehman_scalar",
    }:
        raise ValueError(
            "training.stf_rate_loss.radiation_coefficient_contract must be "
            "horizontal_projected or glehman_scalar"
        )
    return value


def validate_config_v2(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("第二版配置必须是映射")

    pipeline_version = _as_int(config, ("pipeline_version",))
    if pipeline_version != 2:
        raise ValueError("pipeline_version 必须为 2")

    waveform_input_components_from_config(config)
    radial_dynamic_range_stem_from_config(config)
    moment_linear_skip_from_config(config)
    moment_head_dropout_from_config(config)
    stateful_streaming_config_from_config(config)
    magnitude_penalty_from_config(config)
    synth_polarity_mode_from_config(config)
    radiation_coefficient_contract_from_config(config)

    for path in _FORBIDDEN_PATHS:
        exists, _ = _lookup(config, path)
        if exists:
            raise ValueError(f"第二版配置禁止旧语义键: {'.'.join(path)}")

    sample_rate_hz = _as_float(config, ("dataset", "sample_rate_hz"))
    if sample_rate_hz != 1.0:
        raise ValueError("第二版主管线要求 dataset.sample_rate_hz == 1.0")

    filter_type = _required(config, ("dataset", "filter", "type"))
    if filter_type != "lowpass":
        raise ValueError("第二版主管线要求 dataset.filter.type=lowpass")

    cutoff_hz = _as_float(config, ("dataset", "filter", "cutoff_hz"))
    if cutoff_hz not in {0.1, 0.2}:
        raise ValueError(
            "dataset.filter.cutoff_hz must be 0.1 for the corrected legacy "
            "contract or 0.2 for the manuscript contract"
        )

    num_taps_path = ("dataset", "filter", "num_taps")
    num_taps = _as_int(config, num_taps_path)
    if num_taps != 7:
        raise ValueError("dataset.filter.num_taps 必须严格等于 7")

    stf_start_sec = _as_float(config, ("dataset", "stf", "start_sec"))
    if stf_start_sec != 0.0:
        raise ValueError("第二版主管线要求 dataset.stf.start_sec == 0.0")

    stf_duration_sec = _as_float(config, ("dataset", "stf", "duration_sec"))
    if stf_duration_sec <= 0.0:
        raise ValueError("dataset.stf.duration_sec 必须为正")

    preserve_integral = _required(
        config, ("dataset", "stf", "preserve_integral")
    )
    if preserve_integral is not True:
        raise ValueError("第二版主管线要求 dataset.stf.preserve_integral=true")

    retained_fraction = _as_float(
        config, ("dataset", "stf", "min_retained_moment_fraction")
    )
    if retained_fraction < 0.995:
        raise ValueError("dataset.stf.min_retained_moment_fraction 不得低于 0.995")

    if _as_float(config, ("dataset", "stf", "m_ref")) <= 0.0:
        raise ValueError("dataset.stf.m_ref 必须为正")

    rate_representation = str(
        _required(config, ("training", "rate_representation"))
    ).lower()
    if rate_representation != "log1p":
        raise ValueError("第二版主管线要求 training.rate_representation=log1p")

    sampling_exists, event_balanced_sampling = _lookup(
        config,
        ("training", "event_balanced_sampling"),
    )
    if sampling_exists and not isinstance(event_balanced_sampling, bool):
        raise ValueError("training.event_balanced_sampling must be boolean")
    resolve_event_balance_estimator(config.get("training", {}))
    resolve_event_balance_exponent(config.get("training", {}))

    distance_mode = _required(config, ("physics", "distance_mode"))
    if distance_mode != "hypocentral":
        raise ValueError("第二版主配置要求 physics.distance_mode=hypocentral")

    workflow = config.get("workflow")
    if workflow is None:
        delay_mode = _required(config, ("physics", "delay_mode"))
        if delay_mode != "absolute":
            raise ValueError("第二版主配置要求 physics.delay_mode=absolute")
    elif workflow == "station_random_shifted_stf":
        _validate_station_random_workflow(config)
    else:
        raise ValueError(f"不支持的第二版 workflow: {workflow}")


def stf_m_ref_from_config(config: dict[str, Any]) -> float:
    validate_config_v2(config)
    return float(config["dataset"]["stf"]["m_ref"])


def stf_output_steps_from_config(config: dict[str, Any]) -> int:
    validate_config_v2(config)
    dataset = config["dataset"]
    duration_sec = (
        dataset["stf"]["station_window_duration_sec"]
        if config.get("workflow") == "station_random_shifted_stf"
        else dataset["stf"]["duration_sec"]
    )
    return int(
        round(
            float(duration_sec)
            * float(dataset["sample_rate_hz"])
        )
    )
