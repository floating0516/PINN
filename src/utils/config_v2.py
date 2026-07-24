from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any


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


def validate_config_v2(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("第二版配置必须是映射")

    pipeline_version = _as_int(config, ("pipeline_version",))
    if pipeline_version != 2:
        raise ValueError("pipeline_version 必须为 2")

    waveform_input_components_from_config(config)

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
