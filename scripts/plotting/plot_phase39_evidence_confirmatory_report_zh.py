#!/usr/bin/env python3
"""Generate the Chinese Phase39 evidence and confirmatory-protocol report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import fmean
import textwrap
from typing import Any, Iterable, Mapping, Sequence
import warnings

import matplotlib as mpl

mpl.use("Agg")

from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PROJECT_HOME = REPO_ROOT.parent.parent
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "docs" / "reports" / "phase39-evidence-confirmatory-plan-zh"
)
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
REPORT_COMMIT = "7fbb7a7bbe3d747cc031449eb0df54a85840541b"
DATASET_SHA256 = "2e1fa4c12fc1eb03ffc8bf9235491f0886d4b0d360ebcb3486baca4d948cfd6a"

SOURCE_PATHS = {
    "phase22_seed_metrics": (
        REPO_ROOT
        / "docs"
        / "results"
        / "phase22-causal-forward-guided-station-subset"
        / "seed_metrics.csv"
    ),
    "phase22_external_events": (
        REPO_ROOT
        / "docs"
        / "results"
        / "phase22-causal-forward-guided-station-subset"
        / "external_final_event_comparison.csv"
    ),
    "phase27_selection": (
        PROJECT_HOME
        / "runs"
        / "phase27-manuscript-stf-event-loss-weighted-20260724T221820Z-e02aeca"
        / "train"
        / "candidate"
        / "selection.json"
    ),
    "phase27_internal": (
        PROJECT_HOME
        / "runs"
        / "phase27-manuscript-stf-event-loss-weighted-20260724T221820Z-e02aeca"
        / "internal"
        / "summary.json"
    ),
    "phase33_selection": (
        PROJECT_HOME
        / "runs"
        / "phase33-manuscript-stf-no-shape-20260726T013031Z-31b4fb8"
        / "train"
        / "candidate"
        / "selection.json"
    ),
    "phase33_internal": (
        PROJECT_HOME
        / "runs"
        / "phase33-manuscript-stf-no-shape-20260726T013031Z-31b4fb8"
        / "internal"
        / "candidate"
        / "summary.json"
    ),
    "phase38_selection": (
        PROJECT_HOME
        / "runs"
        / "phase39-manuscript-stf-glehman-scalar-20260726T111256Z-7fbb7a7"
        / "train"
        / "baseline"
        / "selection.json"
    ),
    "phase39_selection": (
        PROJECT_HOME
        / "runs"
        / "phase39-manuscript-stf-glehman-scalar-20260726T111256Z-7fbb7a7"
        / "train"
        / "candidate"
        / "selection.json"
    ),
    "fable_external_eval": (
        PROJECT_HOME
        / "runs"
        / "fable-eval-20260726T125507Z"
        / "external_eval.json"
    ),
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
    "lighter_gray": "#F3F4F6",
    "ink": "#1F2933",
    "red": "#B42318",
    "white": "#FFFFFF",
    "blue_fill": "#E8F1F8",
    "orange_fill": "#FCEFEA",
    "green_fill": "#E7F5EF",
    "yellow_fill": "#FFF5D6",
}

EXPECTED = {
    "phase22_internal_event_mae": 0.22277125081708354,
    "phase22_external_event_mae": 0.12386398870806837,
    "phase27_validation_event_mae": 0.11135358810424804,
    "phase27_internal_event_mae": 0.1372873624165853,
    "phase33_validation_event_mae": 0.10100007057189941,
    "phase33_internal_event_mae": 0.13653383255004883,
    "phase38_validation_event_mae": 0.12674380938212076,
    "phase39_validation_event_mae": 0.11433351834615071,
    "phase33_development_event_mae": 0.24948920607566838,
    "phase38_development_event_mae": 0.1935169875621796,
    "no_synth_development_event_mae": 0.20611478686332707,
    "phase39_development_event_mae": 0.14773725867271437,
    "phase39_development_station_mae": 0.26082386366928684,
    "no_synth_development_station_mae": 0.2607091982153398,
    "paired_delta": -0.05837752819061268,
    "paired_bootstrap_low": -0.103548,
    "paired_bootstrap_high": -0.015489,
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_close(name: str, actual: float, expected: float, tol: float = 1e-7) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(
            "{0}: expected {1:.12f}, got {2:.12f}".format(name, expected, actual)
        )


def _compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for head in range(total + 1):
        for tail in _compositions(total - head, parts - 1):
            yield (head,) + tail


def _exact_bootstrap_percentile_ci(
    values: Sequence[float],
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> tuple[float, float]:
    n = len(values)
    factorial_n = math.factorial(n)
    weighted: list[tuple[float, int]] = []
    for counts in _compositions(n, n):
        weight = factorial_n
        total = 0.0
        for count, value in zip(counts, values):
            weight //= math.factorial(count)
            total += count * value
        weighted.append((total / n, weight))
    weighted.sort(key=lambda item: item[0])
    total_weight = n**n

    result: list[float] = []
    for quantile in quantiles:
        threshold = math.ceil(quantile * total_weight)
        cumulative = 0
        selected = weighted[-1][0]
        for value, weight in weighted:
            cumulative += weight
            if cumulative >= threshold:
                selected = value
                break
        result.append(selected)
    return result[0], result[1]


def _exact_sign_flip_p(values: Sequence[float]) -> float:
    observed = abs(fmean(values))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(fmean(sign * value for sign, value in zip(signs, values)))
        extreme += statistic >= observed - 1e-15
        total += 1
    return extreme / total


def _load_evidence() -> dict[str, Any]:
    missing = [str(path) for path in SOURCE_PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing report sources:\n" + "\n".join(missing))

    phase22_rows = _read_csv(SOURCE_PATHS["phase22_seed_metrics"])
    phase22_selected = next(row for row in phase22_rows if row["selected"] == "True")
    phase22_internal = float(phase22_selected["candidate_test_final_mae"])
    phase22_external_rows = _read_csv(SOURCE_PATHS["phase22_external_events"])
    phase22_external = fmean(
        float(row["candidate_abs_error"]) for row in phase22_external_rows
    )

    phase27_selection = _read_json(SOURCE_PATHS["phase27_selection"])
    phase27_internal_payload = _read_json(SOURCE_PATHS["phase27_internal"])
    phase27_validation = float(phase27_selection["candidates"]["17"])
    phase27_internal = float(
        phase27_internal_payload["variants"]["candidate"]["metrics"]["event_mae"]
    )

    phase33_selection = _read_json(SOURCE_PATHS["phase33_selection"])
    phase33_internal_payload = _read_json(SOURCE_PATHS["phase33_internal"])
    phase33_validation = float(phase33_selection["candidates"]["17"])
    phase33_internal = float(phase33_internal_payload["metrics"]["event_mae"])

    phase38_selection = _read_json(SOURCE_PATHS["phase38_selection"])
    phase39_selection = _read_json(SOURCE_PATHS["phase39_selection"])
    phase38_validation = float(phase38_selection["candidates"]["42"])
    phase39_validation = float(phase39_selection["candidates"]["42"])

    fable = _read_json(SOURCE_PATHS["fable_external_eval"])
    checkpoints = fable["checkpoints"]

    def cm0(key: str) -> dict[str, Any]:
        return checkpoints[key]["thresholds"]["cm0"]

    signed = cm0("signed_seed17_champion")
    global_invariant = cm0("global_invariant_seed42")
    no_synth = cm0("no_synth_seed42")
    glehman_gi = cm0("glehman_gi_seed42")

    no_synth_events = {
        row["event"]: float(row["abs_error"]) for row in no_synth["events"]
    }
    glehman_events = {
        row["event"]: float(row["abs_error"]) for row in glehman_gi["events"]
    }
    if no_synth_events.keys() != glehman_events.keys():
        raise AssertionError("Phase39 and no-synth event sets differ")
    paired_deltas = [
        glehman_events[event] - no_synth_events[event]
        for event in sorted(no_synth_events)
    ]
    paired_delta = fmean(paired_deltas)
    improved_count = sum(delta < 0.0 for delta in paired_deltas)
    bootstrap_ci = _exact_bootstrap_percentile_ci(paired_deltas)
    sign_flip_p = _exact_sign_flip_p(paired_deltas)

    checks = {
        "phase22_internal_event_mae": phase22_internal,
        "phase22_external_event_mae": phase22_external,
        "phase27_validation_event_mae": phase27_validation,
        "phase27_internal_event_mae": phase27_internal,
        "phase33_validation_event_mae": phase33_validation,
        "phase33_internal_event_mae": phase33_internal,
        "phase38_validation_event_mae": phase38_validation,
        "phase39_validation_event_mae": phase39_validation,
        "phase33_development_event_mae": float(signed["event_mae"]),
        "phase38_development_event_mae": float(global_invariant["event_mae"]),
        "no_synth_development_event_mae": float(no_synth["event_mae"]),
        "phase39_development_event_mae": float(glehman_gi["event_mae"]),
        "phase39_development_station_mae": float(glehman_gi["station_mae"]),
        "no_synth_development_station_mae": float(no_synth["station_mae"]),
        "paired_delta": paired_delta,
    }
    for name, actual in checks.items():
        _assert_close(name, actual, EXPECTED[name])

    if improved_count != 6:
        raise AssertionError("Expected Phase39 to improve 6/8 event errors")
    if phase27_selection["selected_seed"] != 17:
        raise AssertionError("Unexpected Phase27 selected seed")
    if phase33_selection["selected_seed"] != 17:
        raise AssertionError("Unexpected Phase33 selected seed")
    if phase39_selection["selected_seed"] != 42:
        raise AssertionError("Unexpected Phase39 selected seed")
    if int(glehman_gi["event_count"]) != 8 or int(glehman_gi["station_count"]) != 158:
        raise AssertionError("Unexpected Fable development cohort coverage")

    return {
        **checks,
        "phase22_selected_seed": 73,
        "phase27_selected_seed": 17,
        "phase33_selected_seed": 17,
        "phase39_selected_seed": 42,
        "phase39_validation_by_seed": {
            int(seed): float(value)
            for seed, value in phase39_selection["candidates"].items()
        },
        "paired_deltas": paired_deltas,
        "paired_improved_count": improved_count,
        "paired_event_count": len(paired_deltas),
        "paired_bootstrap_low": EXPECTED["paired_bootstrap_low"],
        "paired_bootstrap_high": EXPECTED["paired_bootstrap_high"],
        "paired_exact_bootstrap_low": bootstrap_ci[0],
        "paired_exact_bootstrap_high": bootstrap_ci[1],
        "paired_sign_flip_p": sign_flip_p,
        "development_event_count": int(glehman_gi["event_count"]),
        "development_station_count": int(glehman_gi["station_count"]),
        "phase33_development_station_mae": float(signed["station_mae"]),
        "phase38_development_station_mae": float(global_invariant["station_mae"]),
    }


def _configure_matplotlib() -> str:
    if not FONT_PATH.is_file():
        raise FileNotFoundError("Missing Chinese font: " + str(FONT_PATH))
    font_manager.fontManager.addfont(str(FONT_PATH))
    family = "Noto Sans CJK SC"
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
            "figure.facecolor": COLORS["white"],
            "savefig.facecolor": COLORS["white"],
        }
    )
    return family


def _save_figure(fig: Any, output_stem: Path) -> tuple[Path, Path]:
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fig.savefig(
            png_path,
            dpi=220,
            bbox_inches="tight",
            facecolor=COLORS["white"],
            metadata={"Software": "PINN_Mag report generator"},
        )
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor=COLORS["white"],
            metadata={
                "Creator": "PINN_Mag report generator",
                "CreationDate": None,
                "ModDate": None,
            },
        )
    missing_glyphs = [
        str(item.message)
        for item in captured
        if "Glyph" in str(item.message) and "missing" in str(item.message)
    ]
    if missing_glyphs:
        raise RuntimeError("Missing glyphs: " + "; ".join(missing_glyphs))
    plt.close(fig)
    return png_path, pdf_path


def _box(
    axis: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 10.0,
    weight: str = "normal",
    align: str = "center",
    linewidth: float = 1.2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.add_patch(patch)
    text_x = x + width / 2 if align == "center" else x + 0.025
    axis.text(
        text_x,
        y + height / 2,
        text,
        transform=axis.transAxes,
        ha=align,
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        weight=weight,
        linespacing=1.35,
    )


def _arrow(
    axis: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["gray"],
    connectionstyle: str = "arc3",
    linewidth: float = 1.5,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        color=color,
        connectionstyle=connectionstyle,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.add_patch(arrow)


def _plot_method_family_boundary(output_stem: Path) -> tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(13.0, 7.2))
    axis.set_axis_off()
    fig.suptitle(
        "两套方法共享 R/STF/正演方向，但推理语义不能混称",
        fontsize=16,
        weight="bold",
        y=0.98,
        color=COLORS["ink"],
    )
    axis.text(
        0.5,
        0.94,
        "Phase22 是因果事件模型；Phase39 是完整 200 s 的单台站离线模型",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["gray"],
    )

    _box(
        axis,
        0.025,
        0.12,
        0.445,
        0.76,
        "",
        facecolor=COLORS["blue_fill"],
        edgecolor=COLORS["blue"],
        linewidth=1.5,
    )
    _box(
        axis,
        0.53,
        0.12,
        0.445,
        0.76,
        "",
        facecolor=COLORS["orange_fill"],
        edgecolor=COLORS["orange"],
        linewidth=1.5,
    )
    axis.text(
        0.247,
        0.83,
        "Phase22  因果事件模型",
        transform=axis.transAxes,
        ha="center",
        fontsize=13,
        weight="bold",
        color=COLORS["blue"],
    )
    axis.text(
        0.752,
        0.83,
        "Phase39  单台站 STF 模型",
        transform=axis.transAxes,
        ha="center",
        fontsize=13,
        weight="bold",
        color=COLORS["orange"],
    )

    left_boxes = [
        (0.075, 0.64, "事件级输入\n严格波形前缀 · R-only\n动态 top-5 台站"),
        (0.075, 0.43, "因果 TCN + masked Transformer\n共享事件 STF"),
        (0.075, 0.22, "每秒更新 · 6 s 处理延迟\n输出事件 Mw"),
    ]
    for index, (x, y, label) in enumerate(left_boxes):
        _box(
            axis,
            x,
            y,
            0.345,
            0.135,
            label,
            facecolor=COLORS["white"],
            edgecolor=COLORS["blue"],
            fontsize=10.2,
        )
        if index < len(left_boxes) - 1:
            _arrow(axis, (0.247, y - 0.012), (0.247, y - 0.055), color=COLORS["blue"])

    right_boxes = [
        (0.58, 0.64, "单台站 200 点 R\n完整 0–199 s\n>2 cm 队列由完整记录确定"),
        (0.58, 0.43, "对称 TCN + 无 mask Transformer\n每台站独立 STF / M0"),
        (0.58, 0.22, "各台站独立 Mw\n事件取台站中位数\n输出离线事件 Mw"),
    ]
    for index, (x, y, label) in enumerate(right_boxes):
        _box(
            axis,
            x,
            y,
            0.345,
            0.135,
            label,
            facecolor=COLORS["white"],
            edgecolor=COLORS["orange"],
            fontsize=10.0,
        )
        if index < len(right_boxes) - 1:
            _arrow(
                axis,
                (0.752, y - 0.012),
                (0.752, y - 0.055),
                color=COLORS["orange"],
            )

    axis.text(
        0.5,
        0.505,
        "不可直接\n互换指标",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        color=COLORS["red"],
        linespacing=1.3,
    )
    axis.text(
        0.5,
        0.075,
        "共同点：R-only · STF/M0 震级路径 · 正演约束方向    |    结论边界：Phase39 不是严格逐秒模型，也不应称为真正 PINN",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["ink"],
    )
    return _save_figure(fig, output_stem)


def _plot_evidence_landscape(
    evidence: Mapping[str, Any],
    output_stem: Path,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.4, 6.8),
        gridspec_kw={"width_ratios": [1.03, 1.0]},
    )
    fig.suptitle(
        "当前证据回答两个不同问题：同事件台站插值 vs. 未见事件开发诊断",
        fontsize=15,
        weight="bold",
        color=COLORS["ink"],
        y=0.99,
    )

    left = axes[0]
    names = ["Phase27\nseed17", "Phase33\nseed17", "Phase39\nseed42"]
    y = np.arange(len(names))[::-1]
    validation = np.array(
        [
            evidence["phase27_validation_event_mae"],
            evidence["phase33_validation_event_mae"],
            evidence["phase39_validation_event_mae"],
        ]
    )
    internal = np.array(
        [
            evidence["phase27_internal_event_mae"],
            evidence["phase33_internal_event_mae"],
            np.nan,
        ]
    )
    for y_value, val_value, test_value in zip(y, validation, internal):
        if np.isfinite(test_value):
            left.plot(
                [val_value, test_value],
                [y_value, y_value],
                color=COLORS["light_gray"],
                linewidth=3.0,
                zorder=1,
            )
    left.scatter(
        validation,
        y,
        s=85,
        marker="o",
        color=COLORS["green"],
        edgecolor=COLORS["white"],
        linewidth=0.8,
        label="validation Event MAE",
        zorder=3,
    )
    finite = np.isfinite(internal)
    left.scatter(
        internal[finite],
        y[finite],
        s=85,
        marker="s",
        color=COLORS["blue"],
        edgecolor=COLORS["white"],
        linewidth=0.8,
        label="锁定 internal test Event MAE",
        zorder=3,
    )
    for x_value, y_value in zip(validation, y):
        left.text(
            x_value - 0.002,
            y_value + 0.16,
            "{0:.3f}".format(x_value),
            ha="right",
            va="bottom",
            fontsize=9,
            color=COLORS["green"],
        )
    for x_value, y_value in zip(internal[finite], y[finite]):
        left.text(
            x_value + 0.002,
            y_value - 0.17,
            "{0:.3f}".format(x_value),
            ha="left",
            va="top",
            fontsize=9,
            color=COLORS["blue"],
        )
    left.text(
        0.151,
        y[-1],
        "正式 internal/test 未开启\n原 within-event gate 未通过",
        ha="left",
        va="center",
        fontsize=9,
        color=COLORS["red"],
    )
    left.axvline(
        0.15,
        linestyle="--",
        linewidth=1.0,
        color=COLORS["gray"],
        alpha=0.8,
    )
    left.text(
        0.151,
        2.45,
        "原 0.15 门槛",
        ha="left",
        va="center",
        fontsize=8.5,
        color=COLORS["gray"],
    )
    left.set_yticks(y, names)
    left.set_xlim(0.085, 0.188)
    left.set_ylim(-0.65, 2.65)
    left.set_xlabel("Event MAE (Mw，越低越好)")
    left.set_title(
        "A  正式 campaign：within_event_station",
        loc="left",
        weight="bold",
        pad=14,
    )
    left.text(
        0.0,
        1.01,
        "同一事件的不同台站分散在 train / validation / test",
        transform=left.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=COLORS["gray"],
    )
    left.grid(axis="x", linestyle=":", linewidth=0.7, color=COLORS["light_gray"])
    left.spines["top"].set_visible(False)
    left.spines["right"].set_visible(False)
    left.legend(loc="lower right", frameon=False)

    right = axes[1]
    dev_names = [
        "Phase39\nGlehman+GI",
        "Phase38\nGI",
        "matched\nno-synth",
        "Phase33\nsigned",
    ]
    dev_values = np.array(
        [
            evidence["phase39_development_event_mae"],
            evidence["phase38_development_event_mae"],
            evidence["no_synth_development_event_mae"],
            evidence["phase33_development_event_mae"],
        ]
    )
    dev_y = np.arange(len(dev_names))[::-1]
    dev_colors = [
        COLORS["blue"],
        COLORS["green"],
        COLORS["orange"],
        COLORS["gray"],
    ]
    right.hlines(
        dev_y,
        xmin=0.12,
        xmax=dev_values,
        color=COLORS["light_gray"],
        linewidth=3.0,
        zorder=1,
    )
    right.scatter(
        dev_values,
        dev_y,
        s=105,
        color=dev_colors,
        edgecolor=COLORS["white"],
        linewidth=0.8,
        zorder=3,
    )
    for x_value, y_value, color in zip(dev_values, dev_y, dev_colors):
        right.text(
            x_value + 0.004,
            y_value,
            "{0:.3f}".format(x_value),
            ha="left",
            va="center",
            fontsize=9.5,
            color=color,
            weight="bold" if color == COLORS["blue"] else "normal",
        )
    right.axvline(
        0.15,
        linestyle="--",
        linewidth=1.0,
        color=COLORS["gray"],
        alpha=0.8,
    )
    right.set_yticks(dev_y, dev_names)
    right.set_xlim(0.12, 0.275)
    right.set_ylim(-0.85, 3.65)
    right.set_xlabel("Event MAE (Mw，越低越好)")
    right.set_title(
        "B  后验诊断：8 事件 development_validation",
        loc="left",
        weight="bold",
        pad=14,
    )
    right.text(
        0.0,
        1.01,
        "同一 8 事件 / 158 台站；不得继续用于模型或 seed 选择",
        transform=right.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=COLORS["gray"],
    )
    right.grid(axis="x", linestyle=":", linewidth=0.7, color=COLORS["light_gray"])
    right.spines["top"].set_visible(False)
    right.spines["right"].set_visible(False)

    right.text(
        0.5,
        -0.19,
        (
            "同 seed42：Phase39 − no-synth = {0:+.3f} Mw；6/8 事件改善；"
            "事件级 bootstrap 95% CI [{1:+.3f}, {2:+.3f}]\n"
            "但 Station MAE 为 {3:.6f} vs {4:.6f}：优势主要来自事件中位数聚合，"
            "不是台站级普遍改善"
        ).format(
            evidence["paired_delta"],
            evidence["paired_bootstrap_low"],
            evidence["paired_bootstrap_high"],
            evidence["phase39_development_station_mae"],
            evidence["no_synth_development_station_mae"],
        ),
        transform=right.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=COLORS["ink"],
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLORS["yellow_fill"],
            "edgecolor": COLORS["yellow"],
            "linewidth": 1.0,
        },
    )
    fig.text(
        0.5,
        0.02,
        "左右两栏的 split、因果语义与证据角色不同，不能合并为一个无偏排行榜。",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=COLORS["red"],
    )
    fig.subplots_adjust(left=0.1, right=0.98, top=0.83, bottom=0.27, wspace=0.34)
    return _save_figure(fig, output_stem)


def _plot_confirmatory_protocol(output_stem: Path) -> tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(14.0, 8.0))
    axis.set_axis_off()
    fig.suptitle(
        "推荐的确认性协议：固定事件折、同 seed 配对、外层 OOF、事件级推断",
        fontsize=15.5,
        weight="bold",
        color=COLORS["ink"],
        y=0.98,
    )
    axis.text(
        0.5,
        0.93,
        "建议先实现并哈希协议，再开始任何正式训练或打开 held-out event",
        transform=axis.transAxes,
        ha="center",
        fontsize=10.5,
        color=COLORS["gray"],
    )

    axis.text(
        0.02,
        0.78,
        "协议冻结",
        transform=axis.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color=COLORS["blue"],
    )
    _box(
        axis,
        0.065,
        0.72,
        0.19,
        0.14,
        "31 个事件\n按 Mw 与台站数分层",
        facecolor=COLORS["blue_fill"],
        edgecolor=COLORS["blue"],
        fontsize=10.5,
        weight="bold",
    )
    _box(
        axis,
        0.32,
        0.72,
        0.23,
        0.14,
        "固定 5 个 outer folds\n每 fold 约 6–7 个事件",
        facecolor=COLORS["blue_fill"],
        edgecolor=COLORS["blue"],
        fontsize=10.5,
    )
    _box(
        axis,
        0.62,
        0.72,
        0.28,
        0.14,
        "所有 arms / seeds 共用同一 folds\n写入 manifest + SHA-256",
        facecolor=COLORS["blue_fill"],
        edgecolor=COLORS["blue"],
        fontsize=10.2,
    )
    _arrow(axis, (0.255, 0.79), (0.32, 0.79), color=COLORS["blue"])
    _arrow(axis, (0.55, 0.79), (0.62, 0.79), color=COLORS["blue"])

    axis.text(
        0.02,
        0.49,
        "配对训练",
        transform=axis.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color=COLORS["orange"],
    )
    _box(
        axis,
        0.065,
        0.43,
        0.16,
        0.14,
        "每个 fold × seed\n17 / 42 / 73",
        facecolor=COLORS["orange_fill"],
        edgecolor=COLORS["orange"],
        fontsize=10.5,
        weight="bold",
    )
    _box(
        axis,
        0.285,
        0.52,
        0.22,
        0.105,
        "Phase39\nGlehman + GI · λsynth=0.5",
        facecolor=COLORS["green_fill"],
        edgecolor=COLORS["green"],
        fontsize=10.0,
    )
    _box(
        axis,
        0.285,
        0.365,
        0.22,
        0.105,
        "matched no-synth\n唯一科学差异：λsynth=0",
        facecolor=COLORS["orange_fill"],
        edgecolor=COLORS["orange"],
        fontsize=10.0,
    )
    _box(
        axis,
        0.58,
        0.43,
        0.25,
        0.14,
        "inner validation 只选 checkpoint\nouter held-out event 始终不可见",
        facecolor=COLORS["lighter_gray"],
        edgecolor=COLORS["gray"],
        fontsize=10.1,
    )
    _arrow(axis, (0.225, 0.5), (0.285, 0.575), color=COLORS["orange"])
    _arrow(axis, (0.225, 0.5), (0.285, 0.417), color=COLORS["orange"])
    _arrow(axis, (0.505, 0.575), (0.58, 0.52), color=COLORS["green"])
    _arrow(axis, (0.505, 0.417), (0.58, 0.48), color=COLORS["orange"])
    _arrow(axis, (0.76, 0.72), (0.76, 0.585), color=COLORS["blue"])

    axis.text(
        0.02,
        0.20,
        "外层推断",
        transform=axis.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color=COLORS["green"],
    )
    bottom_boxes = [
        (0.065, 0.14, 0.18, "一次性生成\nheld-out station 预测"),
        (0.29, 0.14, 0.18, "拼接所有 folds\n形成 OOF 事件集"),
        (0.515, 0.14, 0.18, "事件内取台站中位数\n每事件一个误差"),
        (0.74, 0.14, 0.19, "配对 ΔMAE\nbootstrap + permutation"),
    ]
    for index, (x, y, width, label) in enumerate(bottom_boxes):
        _box(
            axis,
            x,
            y,
            width,
            0.13,
            label,
            facecolor=COLORS["green_fill"],
            edgecolor=COLORS["green"],
            fontsize=10.0,
            weight="bold" if index == 3 else "normal",
        )
        if index < len(bottom_boxes) - 1:
            next_x = bottom_boxes[index + 1][0]
            _arrow(
                axis,
                (x + width, y + 0.065),
                (next_x, y + 0.065),
                color=COLORS["green"],
            )
    axis.plot(
        [0.705, 0.705, 0.155],
        [0.43, 0.305, 0.305],
        transform=axis.transAxes,
        color=COLORS["gray"],
        linewidth=1.5,
        clip_on=False,
    )
    _arrow(axis, (0.155, 0.305), (0.155, 0.274), color=COLORS["gray"])

    _box(
        axis,
        0.065,
        0.005,
        0.865,
        0.09,
        (
            "护栏：三 seed 分别报告，不做 prediction ensemble；within_event_station 仅作次要插值指标\n"
            "既有 8 事件保持 development_validation；现有 grouped 6 test 在协议冻结前继续封存"
        ),
        facecolor=COLORS["yellow_fill"],
        edgecolor=COLORS["yellow"],
        fontsize=9.0,
    )
    return _save_figure(fig, output_stem)


def _evidence_rows(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "model": "Phase22 causal event model",
            "phase": 22,
            "evaluation_role": "locked_internal_test",
            "split_protocol": "event_model_station_random",
            "seed": 73,
            "event_mae": evidence["phase22_internal_event_mae"],
            "station_mae": "",
            "event_count": 31,
            "station_count": "",
            "formal_status": "locked",
            "source_key": "phase22_seed_metrics",
        },
        {
            "model": "Phase22 causal event model",
            "phase": 22,
            "evaluation_role": "development_validation",
            "split_protocol": "external_8_event_dynamic_top5",
            "seed": 73,
            "event_mae": evidence["phase22_external_event_mae"],
            "station_mae": "",
            "event_count": 8,
            "station_count": 159,
            "formal_status": "development_only",
            "source_key": "phase22_external_events",
        },
        {
            "model": "Phase27 station-wise STF",
            "phase": 27,
            "evaluation_role": "validation",
            "split_protocol": "within_event_station",
            "seed": 17,
            "event_mae": evidence["phase27_validation_event_mae"],
            "station_mae": "",
            "event_count": 30,
            "station_count": 385,
            "formal_status": "formal",
            "source_key": "phase27_selection",
        },
        {
            "model": "Phase27 station-wise STF",
            "phase": 27,
            "evaluation_role": "locked_internal_test",
            "split_protocol": "within_event_station",
            "seed": 17,
            "event_mae": evidence["phase27_internal_event_mae"],
            "station_mae": 0.10736585344587053,
            "event_count": 30,
            "station_count": 385,
            "formal_status": "locked",
            "source_key": "phase27_internal",
        },
        {
            "model": "Phase33 no-shape",
            "phase": 33,
            "evaluation_role": "validation",
            "split_protocol": "within_event_station",
            "seed": 17,
            "event_mae": evidence["phase33_validation_event_mae"],
            "station_mae": "",
            "event_count": 30,
            "station_count": 385,
            "formal_status": "formal",
            "source_key": "phase33_selection",
        },
        {
            "model": "Phase33 no-shape",
            "phase": 33,
            "evaluation_role": "locked_internal_test",
            "split_protocol": "within_event_station",
            "seed": 17,
            "event_mae": evidence["phase33_internal_event_mae"],
            "station_mae": 0.1026051546072031,
            "event_count": 30,
            "station_count": 385,
            "formal_status": "locked",
            "source_key": "phase33_internal",
        },
        {
            "model": "Phase33 signed checkpoint",
            "phase": 33,
            "evaluation_role": "development_validation",
            "split_protocol": "external_8_event_cm0",
            "seed": 17,
            "event_mae": evidence["phase33_development_event_mae"],
            "station_mae": evidence["phase33_development_station_mae"],
            "event_count": 8,
            "station_count": 158,
            "formal_status": "post_hoc_exploratory",
            "source_key": "fable_external_eval",
        },
        {
            "model": "Matched no-synth",
            "phase": 34,
            "evaluation_role": "development_validation",
            "split_protocol": "external_8_event_cm0",
            "seed": 42,
            "event_mae": evidence["no_synth_development_event_mae"],
            "station_mae": evidence["no_synth_development_station_mae"],
            "event_count": 8,
            "station_count": 158,
            "formal_status": "post_hoc_exploratory",
            "source_key": "fable_external_eval",
        },
        {
            "model": "Phase38 global invariant",
            "phase": 38,
            "evaluation_role": "development_validation",
            "split_protocol": "external_8_event_cm0",
            "seed": 42,
            "event_mae": evidence["phase38_development_event_mae"],
            "station_mae": evidence["phase38_development_station_mae"],
            "event_count": 8,
            "station_count": 158,
            "formal_status": "post_hoc_exploratory",
            "source_key": "fable_external_eval",
        },
        {
            "model": "Phase39 Glehman+GI",
            "phase": 39,
            "evaluation_role": "validation",
            "split_protocol": "within_event_station",
            "seed": 42,
            "event_mae": evidence["phase39_validation_event_mae"],
            "station_mae": "",
            "event_count": 30,
            "station_count": 385,
            "formal_status": "formal_gate_failed",
            "source_key": "phase39_selection",
        },
        {
            "model": "Phase39 Glehman+GI",
            "phase": 39,
            "evaluation_role": "development_validation",
            "split_protocol": "external_8_event_cm0",
            "seed": 42,
            "event_mae": evidence["phase39_development_event_mae"],
            "station_mae": evidence["phase39_development_station_mae"],
            "event_count": 8,
            "station_count": 158,
            "formal_status": "post_hoc_exploratory",
            "source_key": "fable_external_eval",
        },
    ]


def _write_protocol(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "status": "draft_for_review_not_preregistered",
        "objective": "paired unseen-event comparison of Phase39 Glehman+GI versus matched no-synth",
        "outer_protocol": {
            "type": "five_fold_grouped_event_cv",
            "event_count": 31,
            "fold_count": 5,
            "stratification": ["catalog_magnitude_band", "usable_station_count"],
            "folds_shared_across_arms_and_seeds": True,
            "held_out_prediction": "one_time_out_of_fold",
        },
        "arms": {
            "phase39": {
                "radiation_coefficient": "glehman_scalar",
                "synthetic_polarity_mode": "global_invariant",
                "lambda_synth": 0.5,
            },
            "matched_no_synth": {
                "radiation_coefficient": "glehman_scalar",
                "synthetic_polarity_mode": "global_invariant",
                "lambda_synth": 0.0,
            },
            "only_scientific_difference": "lambda_synth",
        },
        "seeds": [17, 42, 73],
        "seed_policy": {
            "prediction_ensemble": False,
            "paired_by_same_seed": True,
            "report_seed_specific_oof_results": True,
            "final_single_model_seed": "select_by_frozen_validation_only after model-family decision",
        },
        "inner_selection": {
            "uses_outer_events": False,
            "checkpoint_metric": "inner_validation_event_mae_catalog",
        },
        "primary_estimand": "mean_event_absolute_error_phase39_minus_no_synth",
        "primary_inference": [
            "event_level_paired_bootstrap",
            "exact_or_monte_carlo_paired_permutation",
        ],
        "secondary_metrics": [
            "event_bias",
            "event_rmse",
            "station_mae",
            "within_event_station_mae",
        ],
        "frozen_data_boundaries": {
            "external_8_events": "development_validation_only",
            "existing_grouped_6_test_events": "do_not_open_before_protocol_freeze",
            "final_claim": "requires new external cohort or untouched preregistered outer test",
        },
    }
    _write_json(path, payload)


def _write_readme(path: Path, evidence: Mapping[str, Any]) -> None:
    tick = chr(96)
    report = textwrap.dedent(
        """
        # PINN_Mag 当前证据与确认性事件级评估计划

        > 状态：2026-07-28 中文预览报告。本文档只整理既有证据并提出下一步协议；没有启动新训练、没有打开现有 grouped 6 个 test 事件，也没有把反复使用的外部 8 事件重新用于模型选择。

        ## 执行摘要

        当前最稳妥的结论不是“Phase39 已经获胜”，而是：

        - **Phase33** 仍是 §within_event_station§ 指标下证据最完整的锁定模型，内部 Event MAE 为 **{phase33_internal:.6f}**。该划分衡量同一事件内的未见台站插值，不能证明对新地震事件的泛化。
        - **Phase39** 的正式 validation 在 seed42 为 **{phase39_validation:.6f}**，但没有通过原三重 gate，因此正式 internal/external stage 没有开启。
        - 单独的后验八事件开发评估中，Phase39 Event MAE 为 **{phase39_development:.6f}**，优于同 seed42 的 matched no-synth **{no_synth_development:.6f}**。这使 Phase39 成为当前最值得确认的候选，但八事件已经反复使用，只能标为 §development_validation§。
        - 下一步应先冻结 **事件级配对 grouped CV**，不再直接调网络。

        | 当前角色 | 模型 | Event MAE | 能回答的问题 |
        |---|---|---:|---|
        | 锁定同事件 incumbent | Phase33 seed17 | {phase33_internal:.6f} | 同事件未见台站插值 |
        | 未见事件开发候选 | Phase39 seed42 | {phase39_development:.6f} | 八事件后验开发诊断 |
        | 下一步对照 | Phase39 vs matched no-synth | Δ={paired_delta:+.6f} | 正演约束是否改善事件级泛化 |
        | 最终论文模型 | 尚未确定 | — | 需要冻结事件级协议和新外部确认 |

        ## 1. 两套方法必须分开叙述

        ![Phase22 与 Phase39 方法族边界](figures/01_method_family_boundary.png)

        [PDF 版本](figures/01_method_family_boundary.pdf)

        Phase22 与 Phase39 都使用 R、STF/M0 震级路径和正演约束方向，但它们不是同一推理系统。

        | 维度 | Phase22 | Phase39 |
        |---|---|---|
        | 推理单位 | 事件 | 单台站，事件后聚合 |
        | 输入语义 | 严格波形前缀 | 完整 0–199 s |
        | 台站策略 | 动态 top-5 | 完整记录 >2 cm 后验入选 |
        | 时序网络 | causal TCN + masked Transformer | 对称 TCN + 无 mask Transformer |
        | STF | 事件共享 | 每台站独立 |
        | 结果 | 逐秒事件 Mw，6 s 处理延迟 | 200 s 离线台站 Mw 中位数 |
        | 准确称呼 | 因果事件模型 | 正演约束 R-only 多任务神经网络 |

        因此，Phase39 的 §0.147737§ 不能写成 Phase22 风格的严格逐秒动态 top-5 结果，也不应把该网络称为真正 PINN。

        ## 2. 当前证据分层

        ![当前证据分层](figures/02_evidence_landscape.png)

        [PDF 版本](figures/02_evidence_landscape.pdf)

        左栏与右栏回答不同问题：

        - 正式 campaign 的 §within_event_station§ validation/test 把同一事件的不同台站分到不同集合，主要测量事件内台站插值。
        - Fable 八事件结果来自同一 §cm0§、8 事件、158 台站的后验开发评估。它揭示旧 gate 可能与未见事件目标错位，但不能追认最终模型。

        八事件同 seed42 配对结果为：

        | 指标 | Phase39 Glehman+GI | matched no-synth | 差异 |
        |---|---:|---:|---:|
        | Event MAE | {phase39_development:.6f} | {no_synth_development:.6f} | {paired_delta:+.6f} |
        | Station MAE | {phase39_station:.6f} | {no_synth_station:.6f} | {station_delta:+.6f} |
        | 改善事件数 | {improved_count}/{event_count} | — | — |
        | Fable 审计 bootstrap 10k 95% CI | [{bootstrap_low:+.6f}, {bootstrap_high:+.6f}] | — | — |
        | 穷举 bootstrap 分布复核 | [{exact_bootstrap_low:+.6f}, {exact_bootstrap_high:+.6f}] | — | — |
        | exact sign-flip p | {sign_flip_p:.6f} | — | — |

        Phase39 的优势主要出现在“台站预测取事件中位数”之后；Station MAE 略差于 no-synth。论文若保留该结果，必须写成**事件聚合层面的开发证据**，不能写成所有台站都更准。

        支持性 grouped-event 探索实验在外部八事件均值上 3/3 seeds 改善，平均 Δ 为 -0.035653，但其预注册 grouped validation 只有 2/3 改善且平均变差，外部置信区间也跨零。它支持继续确认，不构成最终证明。

        ## 3. Phase23–39 的最小修改轨迹

        | Phase | 唯一主要变量 | 结论 |
        |---:|---|---|
        | 23 | 恢复初稿模型并改为总矩/形状分解 STF 头 | 内部失败 |
        | 24–25 | monotonic cosine；asinh 双动态范围 stem | validation 失败 |
        | 26–27 | 事件均衡采样；全数据事件逆频率加权 | Phase27 内部 Event MAE 0.137287 |
        | 28–31 | mag loss、事件权重指数、moment skip、dropout | 均未通过 validation gate |
        | 32 | 四损失相对权重搜索 | W10 内部 0.175297，方向搁置 |
        | 33 | 删除 §L_shape§ | 锁定同事件 incumbent，内部 0.136534 |
        | 34 | 删除 MSE/synth/mag 的消融 | synth 在旧 validation 上不稳定 |
        | 38 | §global_invariant§ 整条波形极性不变性 | 原 gate 失败 |
        | 39 | §horizontal_projected → glehman_scalar§ | 原 gate 失败；后验开发证据最佳 |

        这条轨迹说明：旧 gate 能筛选同事件台站插值，却不能可靠回答“新事件上正演损失是否有用”。

        ## 4. 推荐的确认性协议

        ![确认性 grouped-event 协议](figures/03_confirmatory_protocol.png)

        [PDF 版本](figures/03_confirmatory_protocol.pdf)

        推荐先冻结一个 5 折事件级协议，再执行正式训练：

        1. **固定 outer folds**：31 个事件按目录震级和可用台站数分层，形成 5 个约 6–7 事件的外层 fold。fold 与分层统计写入 manifest 并哈希；所有 arms、seeds 共用同一划分。
        2. **严格 matched arms**：Phase39 为 Glehman scalar + global invariant + §lambda_synth=0.5§；no-synth 除 §lambda_synth=0§ 外完全一致。
        3. **相同 seed 配对**：只使用 17/42/73。每个 seed 分别生成 OOF 结果，不做 prediction ensemble；seed 不再同时改变事件划分。
        4. **内层只选 checkpoint**：inner validation 只使用 outer-train 事件，不能接触当前 fold 的 held-out event。
        5. **一次性 outer inference**：每个事件只在其 held-out fold 中产生一次 OOF 预测；先保存台站预测，再按事件取台站 Mw 中位数。
        6. **事件级配对推断**：主估计量为 §Δ = MAE_Phase39 − MAE_no-synth§；报告每个 seed 的 Δ、事件级 bootstrap CI、paired permutation/sign-flip，并做 leave-one-event-out influence。
        7. **最终单模型**：确认模型族后，再用冻结 validation 在 17/42/73 中选一个 seed；不平均、不 ensemble，最后才进入新的外部事件或预注册的 untouched test。

        当前 grouped 实验中未打开的 6 个 test 事件必须继续封存，直到协议、实现测试、arm diff、fold manifest 和统计脚本全部冻结并留 SHA。

        ## 5. 建议的判定语言

        | 结果 | 建议表述 |
        |---|---|
        | Δ<0 且 95% CI 上界<0，seed 方向稳定 | 正演约束在预注册事件级评估中得到确认 |
        | Δ<0 但 CI 跨 0 | 方向支持，但证据不足，不能宣布确认 |
        | Δ≥0 | 当前 matched 配方不支持未见事件增益 |
        | within-event 改善、outer 不改善 | 仅支持同事件台站插值 |
        | outer 改善、Station MAE 不改善 | 明确写成事件聚合层面的增益 |

        不论结果如何，八事件 §development_validation§ 都不能升级为最终盲测。

        ## 6. 当前可以与不可以支持的主张

        **目前可以支持：**

        - Phase33 是同事件台站插值指标下证据最完整的锁定模型。
        - Phase39 是当前未见事件开发证据下最值得确认的候选。
        - 旧 §within_event_station§ gate 对未见事件目标存在明显错位风险。
        - Phase39 相对同 seed no-synth 的八事件优势主要发生在事件中位数层面。

        **目前不能支持：**

        - Phase39 已被无偏确认为论文最终模型。
        - 正演损失已在新外部数据上得到最终证明。
        - Phase39 是严格实时、严格逐秒或端到端因果模型。
        - Phase39 是真正 PINN。
        - 八事件结果可以继续用于网络、阈值或 seed 选择。

        ## 7. 数据与复现

        - [证据指标表](evidence_metrics.csv)
        - [事件级配对统计](paired_effects.csv)
        - [建议协议机器可读草案](proposed_protocol.json)
        - [发布清单与 SHA-256](publication_manifest.json)
        - [可复现生成器](../../../scripts/plotting/plot_phase39_evidence_confirmatory_report_zh.py)
        - [Phase22 已发布结果](../../results/phase22-causal-forward-guided-station-subset/README.md)
        - [Phase27 中文完整图集](../../results/phase27-manuscript-stf-event-loss-weighted-zh/README.md)

        - 报告 commit：§{commit}§
        - 数据快照 SHA-256：§{dataset_sha}§
        - 报告角色：§development_evidence_and_protocol_draft§
        """
    ).format(
        phase33_internal=evidence["phase33_internal_event_mae"],
        phase39_validation=evidence["phase39_validation_event_mae"],
        phase39_development=evidence["phase39_development_event_mae"],
        no_synth_development=evidence["no_synth_development_event_mae"],
        paired_delta=evidence["paired_delta"],
        phase39_station=evidence["phase39_development_station_mae"],
        no_synth_station=evidence["no_synth_development_station_mae"],
        station_delta=(
            evidence["phase39_development_station_mae"]
            - evidence["no_synth_development_station_mae"]
        ),
        improved_count=evidence["paired_improved_count"],
        event_count=evidence["paired_event_count"],
        bootstrap_low=evidence["paired_bootstrap_low"],
        bootstrap_high=evidence["paired_bootstrap_high"],
        sign_flip_p=evidence["paired_sign_flip_p"],
        commit=REPORT_COMMIT,
        exact_bootstrap_low=evidence["paired_exact_bootstrap_low"],
        exact_bootstrap_high=evidence["paired_exact_bootstrap_high"],
        dataset_sha=DATASET_SHA256,
    )
    path.write_text(report.replace("§", tick).lstrip(), encoding="utf-8")


def _write_manifest(
    output_dir: Path,
    output_paths: Sequence[Path],
    font_family: str,
    evidence: Mapping[str, Any],
) -> Path:
    sources = {
        key: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for key, path in SOURCE_PATHS.items()
    }
    outputs = {
        str(path.relative_to(output_dir)): {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
        for path in output_paths
    }
    manifest = {
        "schema_version": 1,
        "report_status": "development_evidence_and_protocol_draft",
        "generated_for": "GitHub Chinese report preview",
        "analysis_contract": {
            "training_started": False,
            "frozen_grouped_test_opened": False,
            "external_8_event_role": "development_validation",
            "phase39_formal_internal_external_opened": False,
            "recommended_next_action": "freeze paired grouped-event protocol",
            "prediction_ensemble": False,
        },
        "verified_metrics": {
            "phase33_locked_internal_event_mae": evidence[
                "phase33_internal_event_mae"
            ],
            "phase39_formal_validation_event_mae": evidence[
                "phase39_validation_event_mae"
            ],
            "phase39_development_event_mae": evidence[
                "phase39_development_event_mae"
            ],
            "phase39_minus_no_synth_event_mae": evidence["paired_delta"],
            "phase39_vs_no_synth_improved_events": "{0}/{1}".format(
                evidence["paired_improved_count"],
                evidence["paired_event_count"],
            ),
        },
        "font": {
            "family": font_family,
            "path": str(FONT_PATH),
            "sha256": _sha256(FONT_PATH),
        },
        "generator": {
            "path": str(SCRIPT_PATH),
            "sha256": _sha256(SCRIPT_PATH),
        },
        "sources": sources,
        "outputs": outputs,
        "git_commit": REPORT_COMMIT,
        "dataset_sha256": DATASET_SHA256,
    }
    manifest_path = output_dir / "publication_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def generate(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    evidence = _load_evidence()
    font_family = _configure_matplotlib()

    evidence_csv = output_dir / "evidence_metrics.csv"
    _write_csv(
        evidence_csv,
        [
            "model",
            "phase",
            "evaluation_role",
            "split_protocol",
            "seed",
            "event_mae",
            "station_mae",
            "event_count",
            "station_count",
            "formal_status",
            "source_key",
        ],
        _evidence_rows(evidence),
    )

    paired_csv = output_dir / "paired_effects.csv"
    _write_csv(
        paired_csv,
        [
            "comparison",
            "evaluation_role",
            "seed",
            "event_count",
            "improved_event_count",
            "mean_abs_error_delta",
            "audit_bootstrap_10k_95_low",
            "audit_bootstrap_10k_95_high",
            "exact_bootstrap_distribution_95_low",
            "exact_bootstrap_distribution_95_high",
            "exact_sign_flip_p",
            "interpretation",
        ],
        [
            {
                "comparison": "Phase39 Glehman+GI minus matched no-synth",
                "evaluation_role": "development_validation",
                "seed": 42,
                "event_count": evidence["paired_event_count"],
                "improved_event_count": evidence["paired_improved_count"],
                "mean_abs_error_delta": evidence["paired_delta"],
                "audit_bootstrap_10k_95_low": evidence["paired_bootstrap_low"],
                "audit_bootstrap_10k_95_high": evidence["paired_bootstrap_high"],
                "exact_bootstrap_distribution_95_low": evidence["paired_exact_bootstrap_low"],
                "exact_bootstrap_distribution_95_high": evidence["paired_exact_bootstrap_high"],
                "exact_sign_flip_p": evidence["paired_sign_flip_p"],
                "interpretation": "supportive post-hoc evidence; not final confirmation",
            }
        ],
    )

    protocol_json = output_dir / "proposed_protocol.json"
    _write_protocol(protocol_json)

    figure_paths: list[Path] = []
    figure_paths.extend(
        _plot_method_family_boundary(figures_dir / "01_method_family_boundary")
    )
    figure_paths.extend(
        _plot_evidence_landscape(
            evidence,
            figures_dir / "02_evidence_landscape",
        )
    )
    figure_paths.extend(
        _plot_confirmatory_protocol(figures_dir / "03_confirmatory_protocol")
    )

    readme_path = output_dir / "README.md"
    _write_readme(readme_path, evidence)

    output_paths = [
        readme_path,
        evidence_csv,
        paired_csv,
        protocol_json,
        *figure_paths,
    ]
    manifest_path = _write_manifest(
        output_dir,
        output_paths,
        font_family,
        evidence,
    )

    return {
        "output_dir": str(output_dir),
        "readme": str(readme_path),
        "manifest": str(manifest_path),
        "figures": [str(path) for path in figure_paths],
        "paired_bootstrap_audit_ci": [
            evidence["paired_bootstrap_low"],
            evidence["paired_bootstrap_high"],
        ],
        "paired_bootstrap_exact_ci": [
            evidence["paired_exact_bootstrap_low"],
            evidence["paired_exact_bootstrap_high"],
        ],
        "paired_sign_flip_p": evidence["paired_sign_flip_p"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for the GitHub report package",
    )
    args = parser.parse_args()
    result = generate(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
