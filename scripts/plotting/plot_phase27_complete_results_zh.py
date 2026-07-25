#!/usr/bin/env python3
"""Generate the complete Chinese Phase27 methodology and result package."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence
import warnings

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _style import apply_pub_style, style_axes  # noqa: E402
from plot_phase17_causal_online_results import (  # noqa: E402
    _read_csv,
    _read_json,
    _save_figure,
    _sha256,
    _write_csv,
)
from plot_phase27_magnitude_convergence import (  # noqa: E402
    CENSOR_PLOT_SEC,
    EXPECTED_EVENT_COUNT,
    EXPECTED_GIT_COMMIT,
    EXPECTED_HORIZONS,
    EXPECTED_SELECTED_SEED,
    GROUP_BY_KEY,
    GROUP_SPECS,
    PROCESSING_DELAY_SEC,
    TARGET_ERROR_MW,
    _csv_ready,
    _jittered_catalog_positions,
    _kaplan_meier_median_observation,
    _median_optional,
    _station_availability,
    build_event_convergence_rows,
    load_phase27_inputs,
)


ZH_FONT_CANDIDATES = (
    "Noto Sans CJK SC",
    "PingFang SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
)
EXPECTED_PARAMETER_COUNT = 1_010_850
EXPECTED_DATASET_EVENTS = 31
EXPECTED_DATASET_STATIONS = 2558
EXPECTED_SPLIT_COUNTS = {"train": 1788, "validation": 385, "test": 385}
EXPECTED_EXTERNAL_EVENTS = (
    "Iquique 2014 M7.7",
    "Nepal 2015 M7.3",
    "Kodiak 2018 M7.9",
    "Samos 2020 M7.0",
    "Xizang 2025 M7.1",
    "Mandalay 2025 M7.7",
    "Sand 2025 M7.3",
)
EXPECTED_MISSING_CM2_EVENT = "Luding 2022 M6.6"
EXPECTED_EXTERNAL_METHOD_MAE = {
    "phase27": 0.15627034732273642,
    "crowell": 0.1696399617158069,
    "melgar": 0.24219000754955466,
    "ruhl": 0.4219492630720252,
}
EXPECTED_INTERNAL_METRICS = {
    "event_mae": 0.1372873624165853,
    "event_bias": -0.01514135996500651,
    "station_mae": 0.10736585344587053,
    "station_count": 385,
}

COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#D55E00",
    "yellow": "#E69F00",
    "pink": "#CC79A7",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "ink": "#1F2933",
    "red": "#B42318",
}

GROUP_LABEL_ZH = {
    "mw_lt_7": "Mw < 7.0",
    "mw_7_to_lt_8": "7.0 ≤ Mw < 8.0",
    "mw_ge_8": "Mw ≥ 8.0",
}

CONVERGENCE_SCOPE_NOTE_ZH = (
    "锁定内部测试；同事件未见台站划分；≥2 cm 队列按完整 200 s 径向峰值回顾性确定。\n"
    "相同坐标使用确定性横向偏移；后缀只检查至 200 s，空心点表示超过 200 s 的右删失。"
)

METHOD_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "phase27",
        "label_zh": "本文模型（Phase27）",
        "prediction": "mw_pred_median",
        "error": "error_vs_catalog",
        "stations": "n_stations",
        "color": COLORS["blue"],
        "marker": "o",
    },
    {
        "key": "crowell",
        "label_zh": "Crowell PGD",
        "prediction": "pgd_crowell_mw_pred_median",
        "error": "pgd_crowell_error",
        "stations": "pgd_crowell_n_stations",
        "color": COLORS["orange"],
        "marker": "s",
    },
    {
        "key": "melgar",
        "label_zh": "Melgar PGD",
        "prediction": "pgd_melgar_mw_pred_median",
        "error": "pgd_melgar_error",
        "stations": "pgd_melgar_n_stations",
        "color": COLORS["pink"],
        "marker": "^",
    },
    {
        "key": "ruhl",
        "label_zh": "Ruhl PGD",
        "prediction": "pgd_ruhl_mw_pred_median",
        "error": "pgd_ruhl_error",
        "stations": "pgd_ruhl_n_stations",
        "color": COLORS["green"],
        "marker": "D",
    },
)

EXTERNAL_EVENT_LABEL_ZH = {
    "Iquique 2014 M7.7": "伊基克 2014",
    "Nepal 2015 M7.3": "尼泊尔 2015",
    "Kodiak 2018 M7.9": "科迪亚克 2018",
    "Samos 2020 M7.0": "萨摩斯 2020",
    "Xizang 2025 M7.1": "西藏 2025",
    "Mandalay 2025 M7.7": "曼德勒 2025",
    "Sand 2025 M7.3": "桑德角 2025",
}


def _find_zh_font() -> tuple[str, Path]:
    for family in ZH_FONT_CANDIDATES:
        try:
            path = Path(
                font_manager.findfont(
                    font_manager.FontProperties(family=family),
                    fallback_to_default=False,
                )
            ).resolve()
        except ValueError:
            continue
        if path.is_file():
            return family, path
    raise RuntimeError("未找到可用于发布图件的简体中文字体")


def _apply_zh_style() -> tuple[str, Path]:
    family, path = _find_zh_font()
    apply_pub_style()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )
    return family, path


def _save_zh_figure(fig: Any, output_stem: Path) -> tuple[Path, Path]:
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        paths = _save_figure(fig, output_stem)
    missing = [
        str(item.message)
        for item in captured
        if "Glyph" in str(item.message) and "missing from font" in str(item.message)
    ]
    if missing:
        raise RuntimeError("中文图件存在缺失字形: " + "; ".join(missing))
    return paths


def _panel_label(axis: Any, label: str, *, x: float = -0.14, y: float = 1.08) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def _require_close(actual: Any, expected: float, *, field: str, atol: float = 1e-12) -> None:
    value = float(actual)
    if not math.isfinite(value) or not math.isclose(value, expected, abs_tol=atol):
        raise ValueError(f"{field} changed: {value!r} != {expected!r}")


def _require_mapping_value(
    payload: Mapping[str, Any], path: Sequence[str], expected: Any
) -> None:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError("missing config field: " + ".".join(path))
        value = value[key]
    if value != expected:
        raise ValueError(
            f"config field {'.'.join(path)} changed: {value!r} != {expected!r}"
        )


def _load_yaml_object(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain an object: {path}")
    return payload


def _assert_path_within(path: Path, root: Path, *, field: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} points outside the formal run") from exc
    return resolved


def _validate_selected_model(
    *,
    run_dir: Path,
    seed_summary: Mapping[str, Any],
    config: Mapping[str, Any],
    sampling: Mapping[str, Any],
    dataset_summary: Mapping[str, Any],
    split: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> None:
    if int(seed_summary.get("seed", -1)) != EXPECTED_SELECTED_SEED:
        raise ValueError("selected seed summary is not seed 17")
    if seed_summary.get("variant") != "candidate":
        raise ValueError("selected seed summary is not the Phase27 candidate")
    if int(seed_summary.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("selected model parameter count changed")
    if seed_summary.get("parameterization") != "moment_shape_factorized":
        raise ValueError("selected STF parameterization changed")
    if seed_summary.get("event_balance_estimator") != "inverse_count_full_data":
        raise ValueError("selected event-balance estimator changed")
    if not bool(seed_summary.get("event_balanced_sampling")):
        raise ValueError("selected run no longer uses the event-balanced objective")
    if seed_summary.get("checkpoint", {}).get("sha256") != registry.get(
        "checkpoint", {}
    ).get("sha256"):
        raise ValueError("selected checkpoint differs between training and test")

    exact_config = {
        ("dataset", "sample_rate_hz"): 1.0,
        ("dataset", "radial_peak_min_cm"): 2.0,
        ("dataset", "waveform", "duration_sec"): 200.0,
        ("dataset", "waveform", "max_interpolation_gap_sec"): 0.0,
        ("dataset", "filter", "type"): "lowpass",
        ("dataset", "filter", "cutoff_hz"): 0.2,
        ("dataset", "filter", "num_taps"): 7,
        ("dataset", "filter", "window"): "hamming",
        ("dataset", "stf", "magnitude_target"): "catalog",
        ("physics", "delay_mode"): "absolute",
        ("model", "hidden_dim"): 128,
        ("model", "num_tcn_blocks"): 6,
        ("model", "transformer_num_layers"): 3,
        ("model", "dropout"): 0.2,
        ("model", "use_meta"): True,
        ("model", "input_components"): ["radial"],
        ("model", "predict_catalog_mw"): False,
        ("model", "stf_output_parameterization"): "moment_shape_factorized",
        ("training", "random_seed"): 17,
        ("training", "split_protocol"): "within_event_station",
        ("training", "event_balanced_sampling"): True,
        ("training", "event_balance_estimator"): "inverse_count_full_data",
        ("training", "stf_rate_loss", "lambda_MSE"): 1.0,
        ("training", "stf_rate_loss", "lambda_synth"): 0.5,
        ("training", "stf_rate_loss", "lambda_mag"): 1.0,
        ("training", "stf_rate_loss", "lambda_shape"): 0.1,
        ("training", "stf_rate_loss", "include_intermediate_field"): False,
        ("training", "stf_rate_loss", "include_far_field_P"): True,
        ("training", "stf_rate_loss", "include_far_field_S"): True,
        ("training", "stf_rate_loss", "radiation_pattern_mode"): "full",
        ("evaluation", "aggregation"): "event_median",
        ("evaluation", "external_role"): "development_validation",
    }
    for path, expected in exact_config.items():
        _require_mapping_value(config, path, expected)

    if int(dataset_summary.get("accepted_event_count", -1)) != EXPECTED_DATASET_EVENTS:
        raise ValueError("formal dataset event count changed")
    if int(dataset_summary.get("accepted_station_count", -1)) != EXPECTED_DATASET_STATIONS:
        raise ValueError("formal dataset station count changed")
    if split.get("protocol") != "within_event_station":
        raise ValueError("formal split protocol changed")
    split_counts = split.get("catalog_mw_summary", {})
    for key, expected in EXPECTED_SPLIT_COUNTS.items():
        if int(split_counts.get(key, {}).get("count", -1)) != expected:
            raise ValueError(f"formal {key} split count changed")
    if split.get("assignment_sha256") != seed_summary.get("split_assignment_sha256"):
        raise ValueError("selected split hash changed")

    expected_sampling = {
        "schema_version": 2,
        "mode": "event_equal_inverse_count_full_data",
        "event_balance_estimator": "inverse_count_full_data",
        "event_balanced_sampling": True,
        "replacement": False,
        "loss_weights_applied": True,
        "objective_reduction": "mean(sample_weight * per_sample_loss)",
        "objective_weight_formula": "N/(E*n_event)",
        "record_count": 1788,
        "draw_count": 1788,
        "event_count": 31,
    }
    for key, expected in expected_sampling.items():
        if sampling.get(key) != expected:
            raise ValueError(f"selected sampling field changed: {key}")
    _require_close(
        sampling["event_objective_mass_maximum"],
        sampling["event_objective_mass_minimum"],
        field="event objective mass equality",
        atol=1e-9,
    )


def build_overall_horizon_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    horizon_metrics: Sequence[Mapping[str, Any]],
    convergence_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    persisted = {
        float(row["observation_horizon_sec"]): row for row in horizon_metrics
    }
    stable_by_event = {
        str(row["event"]): row["stable_within_0p15_observation_sec"]
        for row in convergence_rows
    }
    result: list[dict[str, Any]] = []
    for horizon in EXPECTED_HORIZONS:
        rows = [
            row
            for row in prediction_rows
            if float(row["observation_horizon_sec"]) == horizon
        ]
        if len(rows) != EXPECTED_EVENT_COUNT:
            raise ValueError("overall horizon row count changed")
        errors = np.asarray([float(row["error"]) for row in rows])
        source = persisted[horizon]
        current = int(np.count_nonzero(np.abs(errors) <= TARGET_ERROR_MW))
        suffix = sum(
            stable is not None and float(stable) <= horizon
            for stable in stable_by_event.values()
        )
        result.append(
            {
                "observation_horizon_sec": horizon,
                "release_time_sec": horizon + PROCESSING_DELAY_SEC,
                "event_count": EXPECTED_EVENT_COUNT,
                "available_station_count": int(source["available_station_count"]),
                "event_equal_mae": float(np.mean(np.abs(errors))),
                "event_equal_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "event_equal_bias": float(np.mean(errors)),
                "within_0p15_count": current,
                "within_0p15_fraction": current / EXPECTED_EVENT_COUNT,
                "suffix_stable_count": int(suffix),
                "suffix_stable_fraction": suffix / EXPECTED_EVENT_COUNT,
            }
        )
    return result


def build_final_event_rows(
    convergence_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        {
            "event": str(row["event"]),
            "mw_catalog": float(row["mw_catalog"]),
            "magnitude_group": str(row["magnitude_group"]),
            "n_stations_200s": int(row["final_station_count"]),
            "mw_pred_median_200s": float(row["final_mw_pred_median"]),
            "error_200s": float(row["final_error"]),
            "abs_error_200s": float(row["final_abs_error"]),
            "within_0p15_200s": float(row["final_abs_error"])
            <= TARGET_ERROR_MW,
        }
        for row in convergence_rows
    ]
    return sorted(rows, key=lambda row: (-float(row["abs_error_200s"]), row["event"]))


def build_external_comparison_rows(
    source_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(source_rows) != len(EXPECTED_EXTERNAL_EVENTS):
        raise ValueError("external cm2 table does not contain seven events")
    by_event = {str(row["event"]): row for row in source_rows}
    if set(by_event) != set(EXPECTED_EXTERNAL_EVENTS):
        raise ValueError("external cm2 event identity changed")
    long_rows: list[dict[str, Any]] = []
    for event in EXPECTED_EXTERNAL_EVENTS:
        source = by_event[event]
        catalog = float(source["mw_catalog"])
        if not math.isfinite(catalog):
            raise ValueError("external catalog Mw is not finite")
        horizon = float(source["observation_horizon_sec"])
        release = float(source["release_time_sec"])
        if horizon != 200.0 or release != 205.0:
            raise ValueError("external cm2 timing contract changed")
        for spec in METHOD_SPECS:
            prediction = float(source[str(spec["prediction"])])
            error = float(source[str(spec["error"])])
            stations = int(source[str(spec["stations"])])
            if not all(math.isfinite(value) for value in (prediction, error)):
                raise ValueError("external method output is not finite")
            if not math.isclose(prediction - catalog, error, abs_tol=2e-7):
                raise ValueError("external method error is inconsistent")
            if stations < 1:
                raise ValueError("external method has no stations")
            long_rows.append(
                {
                    "event": event,
                    "event_label_zh": EXTERNAL_EVENT_LABEL_ZH[event],
                    "mw_catalog": catalog,
                    "n_stations": stations,
                    "method": str(spec["key"]),
                    "method_label_zh": str(spec["label_zh"]),
                    "mw_pred_median": prediction,
                    "error_vs_catalog": error,
                    "abs_error_vs_catalog": abs(error),
                    "observation_horizon_sec": horizon,
                    "release_time_sec": release,
                }
            )
    summaries: list[dict[str, Any]] = []
    for spec in METHOD_SPECS:
        rows = [row for row in long_rows if row["method"] == spec["key"]]
        errors = np.asarray([float(row["error_vs_catalog"]) for row in rows])
        summary = {
            "method": str(spec["key"]),
            "method_label_zh": str(spec["label_zh"]),
            "event_count": len(rows),
            "event_mae": float(np.mean(np.abs(errors))),
            "event_bias": float(np.mean(errors)),
            "event_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        }
        _require_close(
            summary["event_mae"],
            EXPECTED_EXTERNAL_METHOD_MAE[str(spec["key"])],
            field=f"external {spec['key']} MAE",
            atol=2e-12,
        )
        summaries.append(summary)
    return long_rows, summaries


def load_complete_inputs(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    data = load_phase27_inputs(run_dir)
    paths = dict(data["paths"])
    paths["base_generator_script"] = paths.pop("generator_script")
    fixed_paths = {
        "zh_generator_script": Path(__file__).resolve(),
        "preflight_summary": run_dir / "preflight" / "summary.json",
        "dataset_summary": run_dir / "preflight" / "dataset_summary.json",
        "split_seed_17": run_dir / "preflight" / "split_seed_17.json",
        "selected_seed_summary": run_dir
        / "train"
        / "candidate"
        / "seed_17"
        / "seed_summary.json",
        "selected_sampling_manifest": run_dir
        / "train"
        / "candidate"
        / "seed_17"
        / "sampling_manifest.json",
        "external_summary": run_dir / "external" / "summary.json",
        "external_cm2_summary": run_dir / "external" / "cm2" / "summary.json",
        "external_cm2_predictions": run_dir
        / "external"
        / "cm2"
        / "event_predictions_usgs.csv",
    }
    paths.update(fixed_paths)
    missing = [str(path) for path in paths.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("missing complete Phase27 inputs: " + ", ".join(missing))

    seed_summary = _read_json(paths["selected_seed_summary"])
    config_path = _assert_path_within(
        Path(str(seed_summary.get("config", {}).get("path", ""))),
        run_dir,
        field="selected config",
    )
    paths["selected_config"] = config_path
    expected_config_sha = str(seed_summary.get("config", {}).get("sha256", ""))
    if _sha256(config_path) != expected_config_sha:
        raise ValueError("selected config SHA-256 changed")
    sampling_path = paths["selected_sampling_manifest"]
    expected_sampling_sha = str(seed_summary.get("sampling", {}).get("sha256", ""))
    if _sha256(sampling_path) != expected_sampling_sha:
        raise ValueError("selected sampling SHA-256 changed")

    config = _load_yaml_object(config_path)
    sampling = _read_json(sampling_path)
    preflight_summary = _read_json(paths["preflight_summary"])
    dataset_summary = _read_json(paths["dataset_summary"])
    split = _read_json(paths["split_seed_17"])
    _validate_selected_model(
        run_dir=run_dir,
        seed_summary=seed_summary,
        config=config,
        sampling=sampling,
        dataset_summary=dataset_summary,
        split=split,
        registry=data["registry"],
    )
    if preflight_summary.get("status") != "complete":
        raise ValueError("Phase27 preflight is not complete")
    if preflight_summary.get("git_commit") != EXPECTED_GIT_COMMIT:
        raise ValueError("Phase27 preflight commit changed")
    if bool(preflight_summary.get("git_dirty")):
        raise ValueError("Phase27 preflight worktree was dirty")
    if preflight_summary.get("seeds") != [17, 42, 73]:
        raise ValueError("Phase27 formal seed list changed")
    for seed in ("17", "42", "73"):
        seed_split = preflight_summary.get("splits", {}).get(seed, {})
        for split_name, expected in EXPECTED_SPLIT_COUNTS.items():
            key = {
                "train": "train_record_count",
                "validation": "validation_record_count",
                "test": "test_record_count",
            }[split_name]
            if int(seed_split.get(key, -1)) != expected:
                raise ValueError(f"seed {seed} {split_name} count changed")
    if preflight_summary.get("source_data", {}).get("sha256") != (
        "2e1fa4c12fc1eb03ffc8bf9235491f0886d4b0d360ebcb3486baca4d948cfd6a"
    ):
        raise ValueError("Phase27 source snapshot SHA-256 changed")

    selection = data["selection"]
    expected_selection = {
        "17": 0.11135358810424804,
        "42": 0.1320822874704997,
        "73": 0.20192739168802898,
    }
    if set(selection.get("candidates", {})) != set(expected_selection):
        raise ValueError("Phase27 validation seed set changed")
    for seed, expected in expected_selection.items():
        _require_close(
            selection["candidates"][seed],
            expected,
            field=f"seed {seed} validation MAE",
        )

    external_summary = _read_json(paths["external_summary"])
    cm2_summary = _read_json(paths["external_cm2_summary"])
    if external_summary.get("status") != "complete":
        raise ValueError("external stage is not complete")
    if external_summary.get("evaluation_git_commit") != EXPECTED_GIT_COMMIT:
        raise ValueError("external evaluator commit changed")
    if bool(external_summary.get("evaluation_git_dirty")):
        raise ValueError("external evaluator was dirty")
    if int(external_summary.get("selected_seed", -1)) != EXPECTED_SELECTED_SEED:
        raise ValueError("external stage does not use seed 17")
    if bool(external_summary.get("ensemble_used")):
        raise ValueError("external stage unexpectedly uses an ensemble")
    required_events = set(
        external_summary.get("cm0_coverage_gate", {}).get(
            "required_event_names", []
        )
    )
    cm2_events = set(cm2_summary.get("event_names", []))
    if required_events - cm2_events != {EXPECTED_MISSING_CM2_EVENT}:
        raise ValueError("external cm2 missing-event identity changed")
    if cm2_events != set(EXPECTED_EXTERNAL_EVENTS):
        raise ValueError("external cm2 event set changed")
    if float(cm2_summary.get("threshold_cm", -1)) != 2.0:
        raise ValueError("external threshold changed")
    if int(cm2_summary.get("event_metrics", {}).get("count", -1)) != 7:
        raise ValueError("external cm2 event count changed")
    if int(cm2_summary.get("station_metrics", {}).get("count", -1)) != 76:
        raise ValueError("external cm2 station count changed")
    if float(cm2_summary.get("observation_horizon_sec", -1)) != 200.0:
        raise ValueError("external observation horizon changed")
    if float(cm2_summary.get("release_time_sec", -1)) != 205.0:
        raise ValueError("external release time changed")
    if float(cm2_summary.get("waveform_grid", {}).get("max_interpolation_gap_sec", -1)) != 0.0:
        raise ValueError("external evaluation unexpectedly allows interpolation")
    expected_external_sha = str(
        cm2_summary.get("event_predictions", {}).get("sha256", "")
    )
    if _sha256(paths["external_cm2_predictions"]) != expected_external_sha:
        raise ValueError("external cm2 prediction SHA-256 changed")

    external_rows, external_method_rows = build_external_comparison_rows(
        _read_csv(paths["external_cm2_predictions"])
    )
    _require_close(
        cm2_summary["event_metrics"]["mae"],
        EXPECTED_EXTERNAL_METHOD_MAE["phase27"],
        field="external cm2 summary MAE",
        atol=2e-12,
    )
    top_cm2 = external_summary.get("thresholds", {}).get("cm2", {})
    _require_close(
        top_cm2.get("event_metrics", {}).get("mae"),
        EXPECTED_EXTERNAL_METHOD_MAE["phase27"],
        field="external top-level cm2 MAE",
        atol=2e-12,
    )
    if int(top_cm2.get("event_metrics", {}).get("count", -1)) != 7:
        raise ValueError("external top-level cm2 count changed")

    overall_rows = build_overall_horizon_rows(
        data["prediction_rows"], data["horizon_metrics"], data["convergence_rows"]
    )
    for field, expected in EXPECTED_INTERNAL_METRICS.items():
        if field == "station_count":
            if int(data["metrics"].get(field, -1)) != expected:
                raise ValueError("locked internal station count changed")
        else:
            _require_close(
                data["metrics"].get(field),
                float(expected),
                field=f"locked internal {field}",
                atol=2e-12,
            )
    expected_overall = {
        20.0: (0.8121666510899862, -0.7462419589360555, 4),
        200.0: (0.1372873624165853, -0.01514135996500651, 19),
    }
    for horizon, (mae, bias, within) in expected_overall.items():
        row = next(
            item
            for item in overall_rows
            if float(item["observation_horizon_sec"]) == horizon
        )
        _require_close(row["event_equal_mae"], mae, field=f"{horizon:g}s Event MAE", atol=2e-12)
        _require_close(row["event_equal_bias"], bias, field=f"{horizon:g}s Event bias", atol=2e-12)
        if int(row["within_0p15_count"]) != within:
            raise ValueError(f"{horizon:g}s within-band count changed")
    if int(overall_rows[-1]["suffix_stable_count"]) != 19:
        raise ValueError("200 s suffix-stable event count changed")
    expected_group_counts = {"mw_lt_7": 6, "mw_7_to_lt_8": 17, "mw_ge_8": 7}
    for group, expected in expected_group_counts.items():
        final_group = _group_row(data["group_rows"], 200.0, group)
        if int(final_group["event_count"]) != expected:
            raise ValueError(f"magnitude-group event count changed: {group}")
    data.update(
        {
            "paths": paths,
            "seed_summary": seed_summary,
            "config": config,
            "sampling": sampling,
            "preflight_summary": preflight_summary,
            "dataset_summary": dataset_summary,
            "split": split,
            "external_summary": external_summary,
            "external_cm2_summary": cm2_summary,
            "external_rows": external_rows,
            "external_method_rows": external_method_rows,
            "overall_rows": overall_rows,
            "final_event_rows": build_final_event_rows(data["convergence_rows"]),
        }
    )
    return data


def _diagram_box(
    axis: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 8.4,
    weight: str = "normal",
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.25,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=COLORS["ink"],
        linespacing=1.25,
    )


def _diagram_arrow(
    axis: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["ink"],
    dashed: bool = False,
    mutation_scale: float = 10.0,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=1.2,
            linestyle="--" if dashed else "-",
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def plot_method_and_timing(*, output_stem: Path) -> tuple[Path, Path]:
    _apply_zh_style()
    fig = plt.figure(figsize=(10.5, 6.2))
    ax_model = fig.add_axes((0.035, 0.53, 0.93, 0.39))
    ax_loss = fig.add_axes((0.035, 0.08, 0.53, 0.35))
    ax_time = fig.add_axes((0.605, 0.08, 0.36, 0.35))
    for axis in (ax_model, ax_loss, ax_time):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")

    ax_model.text(-0.01, 1.03, "A", fontsize=12, fontweight="bold", va="top")
    ax_model.text(
        0.025,
        1.03,
        "单台站 R-only、STF 积分定震级的神经网络",
        fontsize=11.5,
        fontweight="bold",
        va="top",
    )
    blocks = (
        (0.01, 0.33, 0.125, 0.47, "单台站输入\nR(t)，1 Hz，200 s\n保留绝对振幅\n+ 5维几何", "#EAF3FA", COLORS["blue"]),
        (0.16, 0.38, 0.105, 0.37, "Conv1D\n核宽 7\nGELU + GN", "#F4F7FA", COLORS["gray"]),
        (0.29, 0.33, 0.135, 0.47, "残差膨胀 TCN ×6\n膨胀率\n1, 2, 4, 8, 16, 32", "#F4F7FA", COLORS["gray"]),
        (0.45, 0.33, 0.13, 0.47, "SE 通道注意力\n正弦位置编码\n几何嵌入逐时相加", "#ECF8F3", COLORS["green"]),
        (0.605, 0.33, 0.13, 0.47, "Transformer ×3\nhidden = 128\n4 heads\nLayerNorm", "#EEF3FA", COLORS["blue"]),
        (0.765, 0.25, 0.105, 0.63, "矩-形状分解头\n\n尺度分支\nlog10 M0\n\n形状分支\np(t) ≥ 0\n∫p(t)dt = 1", "#FFF5E8", COLORS["yellow"]),
        (0.892, 0.31, 0.085, 0.51, "唯一物理输出\nMdot0(t)=M0p(t)\n\n积分得到 Mw\n无独立 Mw 头", "#FFF0EA", COLORS["orange"]),
    )
    for x, y, width, height, label, face, edge in blocks:
        _diagram_box(
            ax_model,
            x,
            y,
            width,
            height,
            label,
            facecolor=face,
            edgecolor=edge,
        )
    for start_x, end_x in ((0.135, 0.16), (0.265, 0.29), (0.425, 0.45), (0.58, 0.605), (0.735, 0.765), (0.87, 0.892)):
        _diagram_arrow(ax_model, (start_x, 0.565), (end_x, 0.565))
    ax_model.text(
        0.99,
        0.11,
        "推理样本始终是一条台站记录；事件值仅在评估时取各台站 Mw 中位数",
        ha="right",
        va="center",
        fontsize=8.2,
        color=COLORS["gray"],
    )
    ax_model.text(
        0.515,
        0.19,
        "几何输入：[ln r，sin θ，cos θ，sin φ，cos φ]；不含震源机制",
        ha="center",
        va="center",
        fontsize=8.1,
        color=COLORS["green"],
    )

    ax_loss.text(-0.02, 1.02, "B", fontsize=12, fontweight="bold", va="top")
    ax_loss.text(
        0.025,
        1.02,
        "SCARDEC 监督、可微正演与事件等权训练目标",
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    _diagram_box(
        ax_loss,
        0.37,
        0.51,
        0.24,
        0.22,
        "预测非负 STF  Mdot0(t)",
        facecolor="#FFF5E8",
        edgecolor=COLORS["yellow"],
        weight="bold",
    )
    _diagram_box(
        ax_loss,
        0.02,
        0.56,
        0.25,
        0.25,
        "SCARDEC 时间形状\n总矩缩放至 USGS 目录 Mw",
        facecolor="#F4F7FA",
        edgecolor=COLORS["gray"],
    )
    _diagram_box(
        ax_loss,
        0.72,
        0.56,
        0.25,
        0.25,
        "积分 Mdot0(t) → M0 → Mw\n与 USGS 目录 Mw 比较",
        facecolor="#F4F7FA",
        edgecolor=COLORS["gray"],
    )
    _diagram_box(
        ax_loss,
        0.12,
        0.22,
        0.27,
        0.20,
        "远场 P+S 正演算子\n绝对 tP/tS + 完整辐射系数",
        facecolor="#FFF0EA",
        edgecolor=COLORS["orange"],
    )
    _diagram_box(
        ax_loss,
        0.61,
        0.22,
        0.27,
        0.20,
        "观测径向位移 R(t)\n机制仅用于离线正演损失",
        facecolor="#EAF3FA",
        edgecolor=COLORS["blue"],
    )
    _diagram_arrow(ax_loss, (0.27, 0.68), (0.37, 0.64), dashed=True, color=COLORS["gray"])
    _diagram_arrow(ax_loss, (0.61, 0.64), (0.72, 0.68), dashed=True, color=COLORS["gray"])
    _diagram_arrow(ax_loss, (0.44, 0.51), (0.30, 0.42), color=COLORS["orange"])
    _diagram_arrow(ax_loss, (0.39, 0.32), (0.61, 0.32), dashed=True, color=COLORS["blue"])
    ax_loss.text(0.31, 0.76, "L_MSE + L_shape", fontsize=7.7, color=COLORS["gray"])
    ax_loss.text(0.65, 0.76, "L_mag", fontsize=7.7, color=COLORS["gray"])
    ax_loss.text(0.48, 0.36, "L_synth", fontsize=7.7, color=COLORS["blue"], ha="center")
    ax_loss.text(
        0.5,
        0.01,
        "L = 1.0 L_MSE + 0.5 L_synth + 1.0 L_mag + 0.1 L_shape\n"
        "Phase27：每轮遍历全部 N=1788 条训练记录，四项逐样本损失均乘 "
        "w_i=N/(E·n_event)",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color=COLORS["ink"],
        linespacing=1.4,
    )

    ax_time.text(-0.03, 1.02, "C", fontsize=12, fontweight="bold", va="top")
    ax_time.text(
        0.03,
        1.02,
        "观测时刻 h 与发布时间 h+5 s",
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    x0, xh, x3, x5 = 0.07, 0.56, 0.75, 0.93
    axis_y = 0.58
    ax_time.annotate(
        "",
        xy=(0.97, axis_y),
        xytext=(0.04, axis_y),
        arrowprops={"arrowstyle": "->", "color": COLORS["ink"], "lw": 1.2},
    )
    for x, label in ((x0, "0"), (xh, "h"), (x3, "h+3"), (x5, "h+5")):
        ax_time.plot([x, x], [axis_y - 0.04, axis_y + 0.04], color=COLORS["ink"], lw=1)
        ax_time.text(x, axis_y - 0.08, label, ha="center", va="top", fontsize=8.5)
    ax_time.plot([x0, xh], [0.72, 0.72], color=COLORS["blue"], lw=8, solid_capstyle="butt")
    ax_time.plot([xh, x3], [0.72, 0.72], color=COLORS["orange"], lw=8, solid_capstyle="butt")
    ax_time.plot([x3, x5], [0.72, 0.72], color=COLORS["gray"], lw=8, solid_capstyle="butt")
    ax_time.text((x0 + xh) / 2, 0.78, "保留的处理后 R 前缀 [0,h)", ha="center", fontsize=8.1)
    ax_time.text((xh + x3) / 2, 0.78, "FIR 未来支撑", ha="center", fontsize=7.5)
    ax_time.text((x3 + x5) / 2, 0.78, "计算余量", ha="center", fontsize=7.5)
    ax_time.scatter([x5], [axis_y], marker="D", s=36, color=COLORS["green"], zorder=4)
    ax_time.text(x5, 0.64, "发布", ha="center", fontsize=8.5, color=COLORS["green"], fontweight="bold")
    ax_time.add_patch(
        FancyBboxPatch(
            (0.05, 0.11),
            0.90,
            0.20,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#FFF1F0",
            edgecolor=COLORS["red"],
            linewidth=1.2,
        )
    )
    ax_time.text(
        0.5,
        0.21,
        "队列成员资格使用完整 0–200 s 处理后径向峰值 ≥2 cm",
        ha="center",
        va="center",
        fontsize=8.4,
        color=COLORS["red"],
        fontweight="bold",
    )
    ax_time.text(
        0.5,
        0.075,
        "所以波形前缀相对于 h+5 s 发布是因果的，但台站筛选是回顾性的，\n"
        "不能称为端到端实时因果系统。",
        ha="center",
        va="top",
        fontsize=8.0,
        color=COLORS["ink"],
        linespacing=1.35,
    )
    fig.suptitle("Phase27 完整方法、物理监督与五秒发布语义", y=0.985, fontsize=13)
    return _save_zh_figure(fig, output_stem)


def plot_overall_internal_convergence(
    *, rows: Sequence[Mapping[str, Any]], output_stem: Path
) -> tuple[Path, Path]:
    _apply_zh_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.35), sharex=True)
    horizon = np.asarray([float(row["observation_horizon_sec"]) for row in rows])
    mae = np.asarray([float(row["event_equal_mae"]) for row in rows])
    bias = np.asarray([float(row["event_equal_bias"]) for row in rows])
    fraction = np.asarray([float(row["within_0p15_fraction"]) for row in rows])
    for axis in axes:
        style_axes(axis)
        axis.set_xlim(16, 204)
        axis.set_xticks((20, 60, 100, 140, 180, 200))
        axis.set_xlabel("震后观测时长（s）")
    axes[0].plot(horizon, mae, color=COLORS["blue"], marker="o", markersize=4.8)
    axes[0].axhline(TARGET_ERROR_MW, color=COLORS["gray"], linestyle="--", lw=1)
    axes[0].set_ylim(0, 0.88)
    axes[0].set_ylabel("事件等权 MAE（Mw）")
    axes[0].set_title("总体绝对误差")
    axes[0].annotate(
        f"{mae[-1]:.3f}",
        (horizon[-1], mae[-1]),
        xytext=(-23, 11),
        textcoords="offset points",
        fontsize=8,
        color=COLORS["blue"],
    )
    axes[1].plot(horizon, bias, color=COLORS["orange"], marker="s", markersize=4.6)
    axes[1].axhline(0, color=COLORS["gray"], linestyle="--", lw=1)
    axes[1].set_ylim(-0.82, 0.18)
    axes[1].set_ylabel("事件等权偏差（Mw）")
    axes[1].set_title("相对目录 Mw 的有符号偏差")
    axes[1].annotate(
        f"{bias[-1]:+.3f}",
        (horizon[-1], bias[-1]),
        xytext=(-28, -16),
        textcoords="offset points",
        fontsize=8,
        color=COLORS["orange"],
    )
    axes[2].plot(horizon, fraction, color=COLORS["green"], marker="D", markersize=4.5)
    axes[2].set_ylim(0, 0.72)
    axes[2].set_yticks((0, 0.2, 0.4, 0.6), ("0%", "20%", "40%", "60%"))
    axes[2].set_ylabel("当前时点达标事件比例")
    axes[2].set_title("|误差| ≤ 0.15 Mw")
    for index in (0, len(rows) - 1):
        axes[2].annotate(
            f"{int(rows[index]['within_0p15_count'])}/30",
            (horizon[index], fraction[index]),
            xytext=(5 if index == 0 else -30, 9),
            textcoords="offset points",
            fontsize=8,
            color=COLORS["green"],
        )
    release_axis = axes[0].secondary_xaxis(
        "top",
        functions=(
            lambda value: value + PROCESSING_DELAY_SEC,
            lambda value: value - PROCESSING_DELAY_SEC,
        ),
    )
    release_axis.set_xlabel("发布时间（s）")
    release_axis.set_xticks((25, 65, 105, 145, 185, 205))
    for index, axis in enumerate(axes):
        _panel_label(axis, chr(ord("A") + index), x=-0.23)
    fig.suptitle("Phase27 锁定内部测试集：总体逐时精度", y=0.995)
    fig.text(
        0.5,
        0.018,
        "事件覆盖始终为 30/30；20 s 时可用台站 357/385，40 s 起为 385/385。"
        "达标比例为各时点即时值，并非累计成功率；队列为完整记录 ≥2 cm 的回顾性队列。",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color="#444444",
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.23, top=0.75, wspace=0.36)
    return _save_zh_figure(fig, output_stem)


def plot_magnitude_group_convergence_zh(
    *,
    rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> tuple[Path, Path]:
    _apply_zh_style()
    fig, (ax_mae, ax_bias) = plt.subplots(1, 2, figsize=(7.6, 3.65), sharex=True)
    for axis in (ax_mae, ax_bias):
        style_axes(axis)
        axis.set_xticks((20, 60, 100, 140, 180, 200))
        axis.set_xlim(16, 204)
        axis.set_xlabel("震后观测时长（s）")
    for spec in GROUP_SPECS:
        group = [row for row in rows if row["magnitude_group"] == spec["key"]]
        horizon = np.asarray([float(row["observation_horizon_sec"]) for row in group])
        mae = np.asarray([float(row["event_mae"]) for row in group])
        bias = np.asarray([float(row["event_bias"]) for row in group])
        count = int(group[0]["event_count"])
        label = f"{GROUP_LABEL_ZH[str(spec['key'])]}（n={count}）"
        for axis, values in ((ax_mae, mae), (ax_bias, bias)):
            axis.plot(
                horizon,
                values,
                color=spec["color"],
                marker=spec["marker"],
                markersize=5.0,
                linewidth=1.6,
                label=label,
            )
    ax_mae.axhline(TARGET_ERROR_MW, color=COLORS["gray"], linestyle="--", lw=1)
    ax_mae.set_ylim(0, 1.62)
    ax_mae.set_ylabel("事件等权 MAE（Mw）")
    ax_mae.set_title("按目录震级分组的绝对误差")
    ax_bias.axhline(0, color=COLORS["gray"], linestyle="--", lw=1)
    ax_bias.set_ylim(-1.62, 0.62)
    ax_bias.set_ylabel("事件等权偏差（Mw）")
    ax_bias.set_title("相对目录 Mw 的有符号偏差")
    handles, labels = ax_mae.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.52, 0.93))
    release_axis = ax_mae.secondary_xaxis(
        "top",
        functions=(
            lambda value: value + PROCESSING_DELAY_SEC,
            lambda value: value - PROCESSING_DELAY_SEC,
        ),
    )
    release_axis.set_xlabel("发布时间（s）")
    release_axis.set_xticks((25, 65, 105, 145, 185, 205))
    _panel_label(ax_mae, "A", x=-0.20)
    _panel_label(ax_bias, "B", x=-0.20)
    totals, first_full, _ = _station_availability(prediction_rows)
    fig.suptitle("Phase27 内部测试：分震级逐时收敛", y=0.99)
    fig.text(
        0.5,
        0.018,
        f"20 s 时可用台站 {totals[20.0]}/385；{first_full:.0f} s 起为 385/385；"
        "事件覆盖 30/30。分组描述相关模式，不表示震级本身造成收敛差异。",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.67, wspace=0.34)
    return _save_zh_figure(fig, output_stem)


def plot_high_magnitude_trajectories_zh(
    *,
    prediction_rows: Sequence[Mapping[str, Any]],
    convergence_rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> tuple[Path, Path]:
    _apply_zh_style()
    convergence = {str(row["event"]): row for row in convergence_rows}
    by_event: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        if float(row["mw_catalog"]) >= 8.0:
            by_event[str(row["event"])].append(row)
    ordered = sorted(
        by_event,
        key=lambda event: (-float(by_event[event][0]["mw_catalog"]), event),
    )
    if len(ordered) != 7:
        raise ValueError("expected seven Mw >= 8 events")

    fig, axes = plt.subplots(4, 2, figsize=(7.6, 8.5), sharex=True)
    flat = list(axes.flat)
    for index, event in enumerate(ordered):
        axis = flat[index]
        style_axes(axis)
        sequence = sorted(
            by_event[event], key=lambda row: float(row["observation_horizon_sec"])
        )
        horizon = np.asarray([float(row["observation_horizon_sec"]) for row in sequence])
        prediction = np.asarray([float(row["mw_pred_median"]) for row in sequence])
        catalog = float(sequence[0]["mw_catalog"])
        axis.axhspan(
            catalog - TARGET_ERROR_MW,
            catalog + TARGET_ERROR_MW,
            color=COLORS["green"],
            alpha=0.11,
            linewidth=0,
            zorder=0,
        )
        axis.axhline(catalog, color="#333333", linestyle="--", linewidth=1.0)
        axis.plot(
            horizon,
            prediction,
            color=COLORS["blue"],
            marker="o",
            markersize=4.1,
            linewidth=1.5,
            zorder=3,
        )
        stable = convergence[event]["stable_within_0p15_observation_sec"]
        if stable is not None:
            stable_value = next(
                float(row["mw_pred_median"])
                for row in sequence
                if float(row["observation_horizon_sec"]) == float(stable)
            )
            axis.scatter(
                [float(stable)],
                [stable_value],
                color=COLORS["green"],
                marker="D",
                s=28,
                zorder=4,
            )
        margin = max(
            0.20,
            0.08
            * (max(float(prediction.max()), catalog) - min(float(prediction.min()), catalog)),
        )
        axis.set_ylim(
            min(float(prediction.min()), catalog - TARGET_ERROR_MW) - margin,
            max(float(prediction.max()), catalog + TARGET_ERROR_MW) + margin,
        )
        station_counts = [int(row["station_count"]) for row in sequence]
        station_title = (
            f"n20/n200={station_counts[0]}/{station_counts[-1]}"
            if station_counts[0] != station_counts[-1]
            else f"n200={station_counts[-1]}"
        )
        axis.set_title(f"{event} | 目录 Mw {catalog:.2f} | {station_title}", fontsize=9.0)
        axis.set_xticks((20, 80, 140, 200))
        _panel_label(axis, chr(ord("A") + index), x=-0.12)

    legend_axis = flat[-1]
    legend_axis.axis("off")
    legend_axis.legend(
        handles=[
            Line2D([0], [0], color=COLORS["blue"], marker="o", label="事件中位数预测"),
            Line2D([0], [0], color="#333333", linestyle="--", label="目录 Mw"),
            Line2D(
                [0],
                [0],
                color=COLORS["green"],
                marker="D",
                linestyle="none",
                label="从该采样点至 200 s 均在误差带内",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.82),
    )
    legend_axis.text(
        0.5,
        0.17,
        "绿色阴影：目录 Mw ± 0.15\n"
        "每个观测时长在 5 s 后发布\n"
        "n200：200 s 可用台站数\n"
        "波形前缀因果；≥2 cm 队列为回顾性筛选",
        ha="center",
        va="center",
        fontsize=7.5,
        transform=legend_axis.transAxes,
        linespacing=1.35,
    )
    fig.supxlabel("震后观测时长（s）", y=0.027)
    fig.supylabel("事件中位数预测 Mw", x=0.025)
    fig.suptitle("Phase27 内部测试：7 个 Mw ≥ 8 事件的逐时轨迹", y=0.985)
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.075,
        top=0.94,
        hspace=0.48,
        wspace=0.22,
    )
    return _save_zh_figure(fig, output_stem)


def plot_convergence_time_by_magnitude_zh(
    *, rows: Sequence[Mapping[str, Any]], output_stem: Path
) -> tuple[Path, Path]:
    _apply_zh_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.85), sharex=True, sharey=True)
    panels = (
        (axes[0], "first_within_0p15_observation_sec", "首次进入 ±0.15 Mw 误差带"),
        (
            axes[1],
            "stable_within_0p15_observation_sec",
            "从该点至 200 s\n持续位于 ±0.15 Mw 内",
        ),
    )
    for panel_index, (axis, field, title) in enumerate(panels):
        style_axes(axis)
        x_positions = _jittered_catalog_positions(rows, field=field)
        for spec in GROUP_SPECS:
            group = [row for row in rows if row["magnitude_group"] == spec["key"]]
            reached = [row for row in group if row[field] is not None]
            censored = [row for row in group if row[field] is None]
            if reached:
                axis.scatter(
                    [x_positions[str(row["event"])] for row in reached],
                    [float(row[field]) for row in reached],
                    color=spec["color"],
                    marker=spec["marker"],
                    s=34,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
            if censored:
                axis.scatter(
                    [x_positions[str(row["event"])] for row in censored],
                    [CENSOR_PLOT_SEC for _ in censored],
                    facecolors="none",
                    edgecolors=spec["color"],
                    marker=spec["marker"],
                    s=46,
                    linewidth=1.3,
                    zorder=4,
                )
        axis.axhline(200, color="#999999", linestyle=":", linewidth=0.9)
        axis.set_xlim(5.75, 9.32)
        axis.set_ylim(12, 230)
        axis.set_xticks((6.0, 7.0, 8.0, 9.0))
        primary_ticks = (*EXPECTED_HORIZONS, CENSOR_PLOT_SEC)
        primary_labels = (*[f"{value:.0f}" for value in EXPECTED_HORIZONS], ">200")
        axis.set_yticks(primary_ticks, primary_labels)
        axis.set_xlabel("目录 Mw")
        axis.set_title(title, fontsize=9.5)
        _panel_label(axis, chr(ord("A") + panel_index), x=-0.16)
    axes[0].set_ylabel("观测时长（s）")
    release_axis = axes[1].secondary_yaxis(
        "right",
        functions=(
            lambda value: value + PROCESSING_DELAY_SEC,
            lambda value: value - PROCESSING_DELAY_SEC,
        ),
    )
    release_axis.set_ylabel("发布时间（s）")
    release_ticks = tuple(value + PROCESSING_DELAY_SEC for value in EXPECTED_HORIZONS)
    release_axis.set_yticks(
        (*release_ticks, CENSOR_PLOT_SEC + PROCESSING_DELAY_SEC),
        (*[f"{value:.0f}" for value in release_ticks], ">205"),
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=spec["color"],
            marker=spec["marker"],
            linestyle="none",
            label=GROUP_LABEL_ZH[str(spec["key"])],
        )
        for spec in GROUP_SPECS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#666666",
            marker="o",
            markerfacecolor="none",
            linestyle="none",
            label="超过 200 s，右删失",
        )
    )
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.52, 0.89))
    fig.suptitle("Phase27 锁定内部测试：目录震级与事件估计达到误差带的时间", y=0.995)
    fig.text(
        0.5,
        0.018,
        CONVERGENCE_SCOPE_NOTE_ZH,
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#444444",
        linespacing=1.3,
    )
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.25, top=0.68, wspace=0.22)
    return _save_zh_figure(fig, output_stem)


def plot_final_event_errors_zh(
    *, rows: Sequence[Mapping[str, Any]], output_stem: Path
) -> tuple[Path, Path]:
    _apply_zh_style()
    ordered = sorted(rows, key=lambda row: (-float(row["abs_error_200s"]), str(row["event"])))
    if len(ordered) != EXPECTED_EVENT_COUNT:
        raise ValueError("final event plot does not contain 30 events")
    y = np.arange(len(ordered))
    fig, (ax_error, ax_count) = plt.subplots(
        1,
        2,
        figsize=(8.4, 9.0),
        sharey=True,
        gridspec_kw={"width_ratios": (2.35, 1.0), "wspace": 0.05},
    )
    style_axes(ax_error)
    style_axes(ax_count)
    ax_error.axvspan(-TARGET_ERROR_MW, TARGET_ERROR_MW, color=COLORS["green"], alpha=0.09)
    ax_error.axvline(0, color="#333333", linewidth=0.9)
    ax_error.axvline(-TARGET_ERROR_MW, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    ax_error.axvline(TARGET_ERROR_MW, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    for index, row in enumerate(ordered):
        spec = GROUP_BY_KEY[str(row["magnitude_group"])]
        error = float(row["error_200s"])
        ax_error.hlines(index, min(0, error), max(0, error), color=spec["color"], linewidth=1.35)
        ax_error.scatter(
            [error],
            [index],
            color=spec["color"],
            marker=spec["marker"],
            s=30,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    ax_error.set_yticks(y, [str(row["event"]) for row in ordered])
    ax_error.invert_yaxis()
    ax_error.set_xlim(-0.63, 0.36)
    ax_error.set_xlabel("200 s 事件误差：预测 Mw − 目录 Mw")
    ax_error.set_title("有符号终值误差（按绝对误差降序）")
    counts = np.asarray([int(row["n_stations_200s"]) for row in ordered])
    ax_count.barh(y, counts, color="#A8B0BA", height=0.58, edgecolor="none")
    ax_count.set_xscale("log")
    ax_count.set_xlim(0.8, 170)
    ax_count.set_xticks((1, 3, 10, 30, 100), ("1", "3", "10", "30", "100"))
    ax_count.set_xlabel("200 s 可用台站数（对数）")
    ax_count.set_title("台站背景")
    ax_count.tick_params(axis="y", labelleft=False)
    for index, count in enumerate(counts):
        ax_count.text(
            min(float(count) * 1.12, 145),
            index,
            str(count),
            va="center",
            ha="left",
            fontsize=6.8,
            color=COLORS["ink"],
        )
    handles = [
        Line2D(
            [0],
            [0],
            color=spec["color"],
            marker=spec["marker"],
            linestyle="none",
            label=GROUP_LABEL_ZH[str(spec["key"])],
        )
        for spec in GROUP_SPECS
    ]
    ax_error.legend(handles=handles, loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.015))
    _panel_label(ax_error, "A", x=-0.28, y=1.04)
    _panel_label(ax_count, "B", x=-0.12, y=1.04)
    within_015 = sum(float(row["abs_error_200s"]) <= 0.15 for row in ordered)
    within_020 = sum(float(row["abs_error_200s"]) <= 0.20 for row in ordered)
    fig.suptitle("Phase27 锁定内部测试：30 个事件的 200 s 终值误差", y=0.985)
    fig.text(
        0.5,
        0.018,
        f"同事件未见台站划分；{within_015}/30 个事件在 ±0.15 Mw 内，"
        f"{within_020}/30 个在 ±0.20 Mw 内。台站数仅作背景，不表示误差因果关系。",
        ha="center",
        va="bottom",
        fontsize=7.3,
        color="#444444",
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.075, top=0.92)
    return _save_zh_figure(fig, output_stem)


def plot_external_cm2_comparison_zh(
    *,
    event_rows: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> tuple[Path, Path]:
    _apply_zh_style()
    fig = plt.figure(figsize=(8.4, 5.9))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.45, 1.0), hspace=0.42)
    ax_event = fig.add_subplot(grid[0, 0])
    ax_summary = fig.add_subplot(grid[1, 0])
    style_axes(ax_event)
    style_axes(ax_summary)
    x = np.arange(len(EXPECTED_EXTERNAL_EVENTS), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(METHOD_SPECS))
    for offset, spec in zip(offsets, METHOD_SPECS):
        rows = [row for row in event_rows if row["method"] == spec["key"]]
        by_event = {str(row["event"]): row for row in rows}
        values = [float(by_event[event]["abs_error_vs_catalog"]) for event in EXPECTED_EXTERNAL_EVENTS]
        ax_event.plot(
            x + offset,
            values,
            color=spec["color"],
            marker=spec["marker"],
            markersize=5.2,
            linewidth=1.0,
            linestyle="none",
            label=str(spec["label_zh"]),
            zorder=3,
        )
    ax_event.axhline(0.15, color=COLORS["gray"], linestyle="--", linewidth=1.0)
    ax_event.set_ylim(0, 0.69)
    ax_event.set_ylabel("绝对误差（Mw）")
    ax_event.set_xticks(
        x,
        [EXTERNAL_EVENT_LABEL_ZH[event] for event in EXPECTED_EXTERNAL_EVENTS],
        rotation=16,
        ha="right",
    )
    ax_event.set_title("同一 7 个外部事件的逐事件误差（处理后径向峰值 ≥2 cm）")
    ax_event.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        columnspacing=1.5,
        handletextpad=0.55,
    )
    _panel_label(ax_event, "A", x=-0.09)

    summary_lookup = {str(row["method"]): row for row in method_rows}
    method_order = [str(spec["key"]) for spec in METHOD_SPECS]
    y = np.arange(len(method_order))
    ax_summary.axvline(0, color="#333333", linewidth=0.9)
    ax_summary.axvline(0.15, color=COLORS["gray"], linestyle="--", linewidth=0.9)
    for index, spec in enumerate(METHOD_SPECS):
        row = summary_lookup[str(spec["key"])]
        mae = float(row["event_mae"])
        bias = float(row["event_bias"])
        ax_summary.scatter(
            [mae],
            [index - 0.11],
            color=spec["color"],
            marker="o",
            s=42,
            zorder=3,
        )
        ax_summary.scatter(
            [bias],
            [index + 0.11],
            facecolor="white",
            edgecolor=spec["color"],
            marker="D",
            s=38,
            linewidth=1.2,
            zorder=3,
        )
        ax_summary.text(mae + 0.012, index - 0.11, f"{mae:.3f}", va="center", fontsize=7.5)
        ax_summary.text(bias - 0.012, index + 0.11, f"{bias:+.3f}", va="center", ha="right", fontsize=7.5)
    ax_summary.set_yticks(y, [str(spec["label_zh"]) for spec in METHOD_SPECS])
    ax_summary.invert_yaxis()
    ax_summary.set_xlim(-0.49, 0.50)
    ax_summary.set_xlabel("七事件聚合指标（Mw）")
    ax_summary.set_title("聚合 MAE（实心圆）与有符号偏差（空心菱形）")
    _panel_label(ax_summary, "B", x=-0.09)
    fig.suptitle("外部开发比较：≥2 cm 阈值下的 7/8 事件结果", y=0.985)
    fig.text(
        0.5,
        0.018,
        "共 76 条台站记录；Luding 2022 M6.6 没有台站达到 2 cm，因此该图不是八事件结果。"
        "外部集合未用于选 seed 或继续调参，也不是无偏论文最终测试集。",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.105, top=0.82)
    return _save_zh_figure(fig, output_stem)


def _format_horizon(value: Any, *, censored: bool = False) -> str:
    if value is None:
        return ">200 s（右删失）" if censored else "未达到"
    return f"{float(value):.0f} s"


def _group_row(
    rows: Sequence[Mapping[str, Any]], horizon: float, group: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if float(row["observation_horizon_sec"]) == horizon
        and str(row["magnitude_group"]) == group
    ]
    if len(matches) != 1:
        raise ValueError("magnitude-group lookup is not unique")
    return matches[0]


def _readme_text_zh(data: Mapping[str, Any]) -> str:
    metrics = data["metrics"]
    selection = data["selection"]
    internal = data["internal_summary"]
    convergence_rows = data["convergence_rows"]
    final_rows = data["final_event_rows"]
    group_rows = data["group_rows"]
    external_summary = data["external_summary"]
    method_rows = data["external_method_rows"]
    external_rows = data["external_rows"]
    registry = data["registry"]
    sampling = data["sampling"]
    preflight = data["preflight_summary"]
    stable_count = sum(
        row["stable_within_0p15_observation_sec"] is not None
        for row in convergence_rows
    )
    high_rows = sorted(
        [row for row in convergence_rows if float(row["mw_catalog"]) >= 8.0],
        key=lambda row: (-float(row["mw_catalog"]), str(row["event"])),
    )
    method_lookup = {str(row["method"]): row for row in method_rows}
    external_by_method_event = {
        (str(row["method"]), str(row["event"])): row for row in external_rows
    }
    current_external = [row for row in external_rows if row["method"] == "phase27"]
    current_within = sum(float(row["abs_error_vs_catalog"]) <= 0.15 for row in current_external)
    cm0 = external_summary["thresholds"]["cm0"]
    station_counts = np.asarray([int(row["n_stations_200s"]) for row in final_rows])
    within_020 = sum(float(row["abs_error_200s"]) <= 0.20 for row in final_rows)

    lines = [
        "# Phase27 中文完整方法与结果图集",
        "",
        "> 本页对应验证集选择的单一 seed 17。内部测试是同一批事件中的未见台站，"
        "不是未见事件测试；外部 8 事件是开发验证集合，不是无偏论文最终测试集。",
        "",
        "## 核心结果",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 内部 validation Event MAE | {float(internal['validation_gate']['candidate']):.6f} |",
        f"| 锁定内部 test Event MAE（200 s 观测 / 205 s 发布） | {float(metrics['event_mae']):.6f} |",
        f"| 锁定内部 test Station MAE | {float(metrics['station_mae']):.6f} |",
        f"| 锁定内部 test Event bias | {float(metrics['event_bias']):+.6f} |",
        f"| 内部事件 / 台站 | {int(metrics['event_count'])} / {int(metrics['station_count'])} |",
        f"| 从某个采样点至 200 s 始终满足绝对误差≤0.15 Mw | {stable_count}/30 事件 |",
        f"| 外部 ≥2 cm Event MAE | {EXPECTED_EXTERNAL_METHOD_MAE['phase27']:.6f}（7/8 事件，76 台站） |",
        f"| 外部 cm0 Event MAE | {float(cm0['event_metrics']['mae']):.6f}（8/8 事件） |",
        "",
        f"外部 ≥2 cm 的第 8 个事件 Luding 2022 M6.6 没有任何台站达到阈值，因此 "
        f"0.156270 只能称为 7/8 事件结果。当前模型在这 7 个事件中有 {current_within}/7 "
        "满足 |误差|≤0.15；它不能替代完整八事件指标。",
        "",
        "## 方法总览",
        "",
        "### 1. 数据、预处理与划分",
        "",
        f"- 活动快照包含 {int(preflight['accepted_event_count'])} 个可用事件、"
        f"{int(preflight['accepted_station_count'])} 条台站记录。数据 SHA-256："
        f"`{preflight['source_data']['sha256']}`。",
        "- 每个样本只输入一条台站径向位移 R；不使用 T/Z、动态 top-5、事件共享 STF、"
        "经验幅值-距离锚点或独立 Mw 预测头。",
        "- 波形保留物理绝对振幅，采样率 1 Hz，窗口 0–200 s；正式流程不插值。"
        "使用 7 taps（order 6）Hamming FIR 低通，截止频率 0.2 Hz。",
        "- 训练队列按完整 200 s 处理后径向峰值 ≥2 cm 决定成员资格。这个筛选是回顾性的。",
        "- seeds 17/42/73 都使用 within-event station split，每个 seed 为 "
        "1788/385/385 条 train/validation/test。Puebla 只有一条记录，只留在训练集，"
        "所以 validation/test 各覆盖 30 个事件。",
        "",
        "### 2. 网络与 STF 唯一震级路径",
        "",
        "- 主干依次为 Conv1D stem、6 个残差膨胀 TCN block、SE 通道注意力、"
        "正弦位置编码与 5 维机制无关几何嵌入、3 层 Transformer。hidden size 为 128，"
        "Transformer 为 4 heads，dropout 为 0.2；总参数量 1,010,850。",
        "- 几何输入为 `[ln(r), sin(theta), cos(theta), sin(phi), cos(phi)]`。"
        "震源机制不进入推理输入；它只用于离线训练期完整辐射系数和正演波形损失。",
        "- 输出头把 STF 分成总矩尺度和归一化时间形状：形状分支产生 `p(t)≥0` 且 "
        "`integral p(t)dt=1`，尺度分支产生 `log10(M0)`，最终 "
        "`Mdot0(t)=10^log10(M0) * p(t)`。因此 STF 非负且积分严格等于 M0。",
        "- `Mw=(2/3)(log10(M0)-9.1)` 只由同一 STF 的积分得到，没有第二条标量震级路径。",
        "",
        "### 3. 物理正演与四项损失",
        "",
        "SCARDEC 提供 STF 时间形状，总矩缩放到 USGS 目录 Mw。可微正演算子使用震源距、"
        "`alpha=7900 m/s`、`beta=4533 m/s`、`rho=3400 kg/m^3`、绝对 P/S 延迟、"
        "远场 P+S 和完整辐射系数；中场项关闭。正演形式参考 "
        "[DOI 10.1029/2025JB033222](https://doi.org/10.1029/2025JB033222)。",
        "",
        "```text",
        "L = 1.0 L_MSE + 0.5 L_synth + 1.0 L_mag + 0.1 L_shape",
        "```",
        "",
        "- `L_MSE`：`log10(1 + Mdot0 / M_ref)` 编码空间中的预测 STF 与参考 STF "
        "均方误差，其中 `M_ref = 1e18 N·m/s`。",
        "- `L_synth`：预测 STF 经可微正演后与观测径向位移的一致性。",
        "- `L_mag`：STF 积分得到的 Mw 与 USGS 目录 Mw 的平方误差。",
        "- `L_shape`：积分归一化后 STF 时间形状的均方误差。",
        "",
        "非负性由输出参数化直接保证，不存在第五个 `L_nonneg`。",
        "",
        "### 4. Phase27 的单一改动",
        "",
        "Phase27 不改网络、输入、四项损失或划分，只改变训练目标的估计方式。每个 epoch "
        "遍历全部 N=1788 条训练记录，并对四项逐样本损失统一施加：",
        "",
        "```text",
        "w_i = N / (E * n_event),    E = 31",
        "batch loss = mean(w_i * per_sample_loss_i)",
        "```",
        "",
        f"因此每个训练事件的总目标质量相同；权重范围为 "
        f"{float(sampling['objective_weight_minimum']):.6f}–"
        f"{float(sampling['objective_weight_maximum']):.6f}。这不是新增损失项，也不是事件级模型。",
        "",
        "### 5. seed 选择与数据边界",
        "",
        "| Seed | validation Event MAE | 是否选择 |",
        "|---:|---:|:---:|",
    ]
    for seed in ("17", "42", "73"):
        lines.append(
            f"| {seed} | {float(selection['candidates'][seed]):.6f} | "
            f"{'是' if int(seed) == EXPECTED_SELECTED_SEED else '否'} |"
        )
    lines.extend(
        [
            "",
            "只按内部 validation 选择 seed 17，不做 seed 平均。候选冻结后才一次性读取锁定内部 test；"
            "只有内部 Event MAE 小于 0.15 后，才报告外部开发集合。外部结果没有用于选择 seed 或 Phase27 变量。",
            "",
            "## 1. 方法、物理监督与五秒发布语义",
            "",
            "![方法、物理监督与五秒发布语义](figures/01_method_and_causal_timing.png)",
            "",
            "[查看可编辑文字 PDF](figures/01_method_and_causal_timing.pdf)",
            "",
            "对观测时长 h，模型输入只保留 `[0,h)` 的处理后波形槽，后续槽置零；居中 FIR "
            "最多需要到 h+3 s 的原始波形支撑，结果在 h+5 s 发布。因此它相对于发布时间是波形前缀因果的。"
            "TCN 卷积和 Transformer 本身并不是严格 causal/masked 结构，不应这样命名。更重要的是，"
            "≥2 cm 队列依赖完整 200 s 峰值，所以系统不是端到端实时因果选站。",
            "",
            "## 2. 内部总体逐时精度",
            "",
            "![内部总体逐时精度](figures/02_overall_internal_convergence.png)",
            "",
            "[查看 PDF](figures/02_overall_internal_convergence.pdf)",
            "",
            "总体 Event MAE 从 20 s 观测（25 s 发布）的 0.8122 降至 200 s 观测（205 s 发布）的 "
            "0.1373；bias 从 -0.7462 收敛到 -0.0151。当前时点满足 |误差|≤0.15 的事件由 "
            "4/30 增至 19/30。该比例逐时独立计算，可能非单调，不是累计成功率。",
            "",
            "## 3. 分震级逐时收敛",
            "",
            "![分震级逐时收敛](figures/03_magnitude_group_convergence.png)",
            "",
            "[查看 PDF](figures/03_magnitude_group_convergence.pdf)",
            "",
            "| 观测 / 发布 | Mw < 7 MAE | 7≤Mw<8 MAE | Mw≥8 MAE | Mw≥8 bias |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon in (20.0, 100.0, 180.0, 200.0):
        low = _group_row(group_rows, horizon, "mw_lt_7")
        middle = _group_row(group_rows, horizon, "mw_7_to_lt_8")
        high = _group_row(group_rows, horizon, "mw_ge_8")
        lines.append(
            f"| {horizon:.0f} / {horizon + 5:.0f} s | {float(low['event_mae']):.4f} | "
            f"{float(middle['event_mae']):.4f} | {float(high['event_mae']):.4f} | "
            f"{float(high['event_bias']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "Mw≥8 组在 20 s 时的 MAE 与 bias 都为约 1.5068，表现为一致的早期低估；"
            "到 200 s 三组 MAE 为 0.1789/0.1298/0.1198。分组图是描述性结果，不能解释为震级本身造成差异。",
            "",
            "## 4. 七个 Mw≥8 事件轨迹",
            "",
            "![七个 Mw≥8 事件轨迹](figures/04_high_magnitude_event_trajectories.png)",
            "",
            "[查看 PDF](figures/04_high_magnitude_event_trajectories.pdf)",
            "",
            "| 事件 | 目录 Mw | 首次进入 ±0.15 | 从该点至 200 s 均达标 | 200 s 绝对误差 | n200 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in high_rows:
        stable = _format_horizon(
            row["stable_within_0p15_observation_sec"], censored=True
        )
        lines.append(
            f"| {row['event']} | {float(row['mw_catalog']):.2f} | "
            f"{_format_horizon(row['first_within_0p15_observation_sec'])} | {stable} | "
            f"{float(row['final_abs_error']):.4f} | {int(row['final_station_count'])} |"
        )
    lines.extend(
        [
            "",
            "Tokachi2003 在 20 s 只有 6 个可用台站，到 200 s 为 34 个；其余高震级事件的"
            "逐时台站数不变。图中绿色菱形只表示从该采样点到 200 s 终点持续达标。",
            "",
            "## 5. 目录震级与进入误差带的时间",
            "",
            "![目录震级与收敛时间](figures/05_convergence_time_by_magnitude.png)",
            "",
            "[查看 PDF](figures/05_convergence_time_by_magnitude.pdf)",
            "",
            "首次进入误差带与后缀达标不是一回事：事件可能先进入、随后又离开。后缀达标要求"
            "之后所有已采样时点直到 200 s 都保持在带内。没有满足的 11 个事件按超过 200 s "
            "右删失，不能强行赋值为 200 s。",
            "",
            "| 震级组 | 事件数 | 已进入者的首次进入中位数 | 删失感知的后缀中位数 | 200 s 前后缀达标 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for spec in GROUP_SPECS:
        group = [row for row in convergence_rows if row["magnitude_group"] == spec["key"]]
        first_median = _median_optional(
            [row["first_within_0p15_observation_sec"] for row in group]
        )
        stable_median = _kaplan_meier_median_observation(group)
        group_stable = sum(row["stable_within_0p15_observation_sec"] is not None for row in group)
        stable_display = (
            "未达到中位数"
            if stable_median is None
            else f"{stable_median:.0f} s 观测 / {stable_median + 5:.0f} s 发布"
        )
        lines.append(
            f"| {GROUP_LABEL_ZH[str(spec['key'])]} | {len(group)} | "
            f"{_format_horizon(first_median)} | {stable_display} | {group_stable}/{len(group)} |"
        )
    lines.extend(
        [
            "",
            "## 6. 30 个内部事件的终值误差与台站背景",
            "",
            "![内部事件终值误差与台站背景](figures/06_final_event_errors_and_station_counts.png)",
            "",
            "[查看 PDF](figures/06_final_event_errors_and_station_counts.pdf)",
            "",
            f"200 s 时 {stable_count}/30 个事件在 ±0.15 Mw 内，{within_020}/30 个在 ±0.20 Mw 内。"
            f"每事件 test 台站数范围 {int(station_counts.min())}–{int(station_counts.max())}，"
            f"中位数 {float(np.median(station_counts)):.0f}。下表列出绝对误差最大的事件；完整 30 行见 CSV。",
            "",
            "| 事件 | 目录 Mw | 预测 Mw | 有符号误差 | 绝对误差 | 台站数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final_rows[:8]:
        lines.append(
            f"| {row['event']} | {float(row['mw_catalog']):.2f} | "
            f"{float(row['mw_pred_median_200s']):.3f} | {float(row['error_200s']):+.3f} | "
            f"{float(row['abs_error_200s']):.3f} | {int(row['n_stations_200s'])} |"
        )
    lines.extend(
        [
            "",
            "少台站事件中存在大误差，但多台站并不保证误差小；右栏台站数只是上下文，不能据此建立因果解释。",
            "",
            "## 7. 外部 ≥2 cm 的同事件方法比较",
            "",
            "![外部同事件方法比较](figures/07_external_cm2_method_comparison.png)",
            "",
            "[查看 PDF](figures/07_external_cm2_method_comparison.pdf)",
            "",
            "| 方法 | 七事件 MAE | bias | RMSE | 绝对误差≤0.15 Mw |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for spec in METHOD_SPECS:
        row = method_lookup[str(spec["key"])]
        rows_for_method = [item for item in external_rows if item["method"] == spec["key"]]
        within = sum(float(item["abs_error_vs_catalog"]) <= 0.15 for item in rows_for_method)
        lines.append(
            f"| {spec['label_zh']} | {float(row['event_mae']):.6f} | "
            f"{float(row['event_bias']):+.6f} | {float(row['event_rmse']):.6f} | {within}/7 |"
        )
    lines.extend(
        [
            "",
            "| 事件 | 目录 Mw | Phase27 绝对误差 | Crowell | Melgar | Ruhl | Phase27 台站数 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for event in EXPECTED_EXTERNAL_EVENTS:
        current = external_by_method_event[("phase27", event)]
        lines.append(
            f"| {EXTERNAL_EVENT_LABEL_ZH[event]} | {float(current['mw_catalog']):.1f} | "
            f"{float(current['abs_error_vs_catalog']):.3f} | "
            f"{float(external_by_method_event[('crowell', event)]['abs_error_vs_catalog']):.3f} | "
            f"{float(external_by_method_event[('melgar', event)]['abs_error_vs_catalog']):.3f} | "
            f"{float(external_by_method_event[('ruhl', event)]['abs_error_vs_catalog']):.3f} | "
            f"{int(current['n_stations'])} |"
        )
    lines.extend(
        [
            "",
            "本文模型在相同 7 事件上优于三种 PGD 汇总 MAE，但 0.156270 仍高于 0.15，"
            "且缺少 Luding。不能据此声称已经完成 8/8 外部目标，也没有进行显著性检验。",
            "",
            "## 结论边界",
            "",
            "- 推荐名称：**单台站 R-only、STF 积分定震级的物理正演约束神经网络**。",
            "- 逐时实验是波形前缀诊断；网络不是严格因果卷积或 masked Transformer。",
            "- 内部 test 是同事件未见台站，不证明对全新事件的无偏泛化。",
            "- ≥2 cm 台站成员来自完整记录，当前评估不是端到端实时系统。",
            "- 外部 8 事件已用于开发验证，不能继续据此选择结构或作为论文最终盲测。",
            "- 后缀达标只定义到 200 s 观测终点，不推断 200 s 之后的稳定性。",
            "",
            "## 数据表与可复现来源",
            "",
            "- [总体逐时指标](overall_horizon_metrics.csv)",
            "- [分震级逐时指标](magnitude_group_horizon_metrics.csv)",
            "- [全部事件逐时预测](event_predictions_by_horizon.csv)",
            "- [事件首次/后缀达标与删失表](event_convergence_summary.csv)",
            "- [内部 30 事件终值误差](final_event_errors.csv)",
            "- [外部 ≥2 cm 逐事件四方法对比](external_cm2_event_comparison.csv)",
            "- [外部 ≥2 cm 四方法汇总](external_cm2_method_summary.csv)",
            "- [发布清单与 SHA-256](publication_manifest.json)",
            "- [可复现中文生成器](../../../scripts/plotting/plot_phase27_complete_results_zh.py)",
            "- [共享英文科学派生实现](../../../scripts/plotting/plot_phase27_magnitude_convergence.py)",
            "",
            f"模型/评估 commit：`{EXPECTED_GIT_COMMIT}`",
            f"选择 checkpoint SHA-256：`{registry['checkpoint']['sha256']}`",
            "正式运行：`phase27-manuscript-stf-event-loss-weighted-20260724T221820Z-e02aeca`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_publication_manifest(
    *,
    path: Path,
    data: Mapping[str, Any],
    output_files: Sequence[Path],
    temporary_root: Path,
    final_root: Path,
    font_family: str,
    font_path: Path,
) -> None:
    inputs = {
        key: {"path": str(Path(source).resolve()), "sha256": _sha256(Path(source))}
        for key, source in sorted(data["paths"].items())
    }
    inputs["zh_font_file"] = {"path": str(font_path), "sha256": _sha256(font_path)}
    outputs: dict[str, dict[str, str]] = {}
    for source in sorted(output_files):
        relative = source.relative_to(temporary_root)
        outputs[str(relative)] = {
            "path": str((final_root / relative).resolve()),
            "sha256": _sha256(source),
        }
    manifest = {
        "schema_version": 1,
        "locale": "zh-CN",
        "font": {"family": font_family, "path": str(font_path)},
        "analysis_contract": {
            "selected_seed": 17,
            "ensemble_used": False,
            "split_protocol": "within_event_station",
            "internal_test_role": "same_event_unseen_station",
            "external_role": "development_validation",
            "input_components": ["radial"],
            "station_sample_model": True,
            "stf_parameterization": "moment_shape_factorized",
            "independent_mw_head": False,
            "event_balance_estimator": "inverse_count_full_data",
            "observation_horizons_sec": list(EXPECTED_HORIZONS),
            "processing_delay_sec": PROCESSING_DELAY_SEC,
            "target_absolute_error_mw": TARGET_ERROR_MW,
            "right_censor_horizon_sec": 200.0,
            "waveform_prefix_causal_at_release": True,
            "station_selection_causal": False,
            "end_to_end_causal": False,
            "external_cm2_coverage": "7/8",
            "external_cm2_missing_event": EXPECTED_MISSING_CM2_EVENT,
        },
        "inputs": inputs,
        "outputs": outputs,
    }
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_bundle(*, run_dir: Path, output_dir: Path) -> dict[str, Path]:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    data = load_complete_inputs(run_dir)
    font_family, font_path = _apply_zh_style()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output directory already exists: {temporary}")
    temporary.mkdir()
    output_files: list[Path] = []
    try:
        figures_dir = temporary / "figures"
        figures_dir.mkdir()
        figure_specs = (
            (
                "01_method_and_causal_timing",
                lambda stem: plot_method_and_timing(output_stem=stem),
            ),
            (
                "02_overall_internal_convergence",
                lambda stem: plot_overall_internal_convergence(
                    rows=data["overall_rows"], output_stem=stem
                ),
            ),
            (
                "03_magnitude_group_convergence",
                lambda stem: plot_magnitude_group_convergence_zh(
                    rows=data["group_rows"],
                    prediction_rows=data["prediction_rows"],
                    output_stem=stem,
                ),
            ),
            (
                "04_high_magnitude_event_trajectories",
                lambda stem: plot_high_magnitude_trajectories_zh(
                    prediction_rows=data["prediction_rows"],
                    convergence_rows=data["convergence_rows"],
                    output_stem=stem,
                ),
            ),
            (
                "05_convergence_time_by_magnitude",
                lambda stem: plot_convergence_time_by_magnitude_zh(
                    rows=data["convergence_rows"], output_stem=stem
                ),
            ),
            (
                "06_final_event_errors_and_station_counts",
                lambda stem: plot_final_event_errors_zh(
                    rows=data["final_event_rows"], output_stem=stem
                ),
            ),
            (
                "07_external_cm2_method_comparison",
                lambda stem: plot_external_cm2_comparison_zh(
                    event_rows=data["external_rows"],
                    method_rows=data["external_method_rows"],
                    output_stem=stem,
                ),
            ),
        )
        for name, plotter in figure_specs:
            png, pdf = plotter(figures_dir / name)
            output_files.extend((png, pdf))

        csv_outputs = {
            "overall_horizon_metrics.csv": data["overall_rows"],
            "magnitude_group_horizon_metrics.csv": data["group_rows"],
            "event_predictions_by_horizon.csv": data["prediction_rows"],
            "event_convergence_summary.csv": data["convergence_rows"],
            "final_event_errors.csv": data["final_event_rows"],
            "external_cm2_event_comparison.csv": data["external_rows"],
            "external_cm2_method_summary.csv": data["external_method_rows"],
        }
        for name, rows in csv_outputs.items():
            path = temporary / name
            _write_csv(path, _csv_ready(rows))
            output_files.append(path)

        readme = temporary / "README.md"
        with readme.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(_readme_text_zh(data))
        output_files.append(readme)

        manifest = temporary / "publication_manifest.json"
        _write_publication_manifest(
            path=manifest,
            data=data,
            output_files=output_files,
            temporary_root=temporary,
            final_root=output_dir,
            font_family=font_family,
            font_path=font_path,
        )
        output_files.append(manifest)
        relative_paths = [path.relative_to(temporary) for path in output_files]
        temporary.rename(output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        str(relative): output_dir / relative
        for relative in sorted(relative_paths, key=str)
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成 Phase27 中文完整方法与结果图集"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = generate_bundle(
        run_dir=args.run_dir.resolve(), output_dir=args.output_dir.resolve()
    )
    print(f"generated {len(artifacts)} Phase27 Chinese publication artifacts")


if __name__ == "__main__":
    main()
