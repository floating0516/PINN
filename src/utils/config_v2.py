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


def validate_config_v2(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("第二版配置必须是映射")

    pipeline_version = _as_int(config, ("pipeline_version",))
    if pipeline_version != 2:
        raise ValueError("pipeline_version 必须为 2")

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
    if cutoff_hz != 0.1:
        raise ValueError("第二版主管线要求 dataset.filter.cutoff_hz == 0.1")

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

    delay_mode = _required(config, ("physics", "delay_mode"))
    if delay_mode != "absolute":
        raise ValueError("第二版主配置要求 physics.delay_mode=absolute")


def stf_m_ref_from_config(config: dict[str, Any]) -> float:
    validate_config_v2(config)
    return float(config["dataset"]["stf"]["m_ref"])


def stf_output_steps_from_config(config: dict[str, Any]) -> int:
    validate_config_v2(config)
    dataset = config["dataset"]
    return int(
        round(
            float(dataset["stf"]["duration_sec"])
            * float(dataset["sample_rate_hz"])
        )
    )
