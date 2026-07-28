from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_phase40_streaming_consistency import (  # noqa: E402
    BASELINE_METRICS,
    EXPECTED_SPLIT_ASSIGNMENT_SHA256,
    GATES,
    PHASE39_CHECKPOINT,
    _absolute_summary,
    _assert_formal_baseline,
    _build_stf_rate_criterion,
    _prepare_v2_batch,
    _write_csv,
    _write_json,
    conservative_s_supported_steps,
    load_frozen_config,
    validate_source_artifacts,
    validation_gate,
)
from src.data.loaders_v2 import get_data_loaders_v2  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    aggregate_event_predictions,
    summarize_predictions,
)
from src.models.model import PINNModel  # noqa: E402
from src.training.loss_stf_rate_v2 import (  # noqa: E402
    moment_magnitude_from_rate,
)
from src.utils.device import configure_runtime  # noqa: E402
from src.utils.provenance import (  # noqa: E402
    current_git_commit,
    git_is_dirty,
    sha256_file,
)


HORIZONS = tuple(range(1, 201))
LATE_TRANSITION_START = 180
CONFIRMATION_DELAY_SEC = 6.0


def delayed_confirmed_steps(
    source_distance_m: torch.Tensor,
    *,
    previous_horizon_sec: int,
    beta_m_per_s: float,
    confirmation_delay_sec: float = CONFIRMATION_DELAY_SEC,
    source_steps: int = 200,
) -> torch.Tensor:
    if previous_horizon_sec < 0:
        raise ValueError("previous_horizon_sec must be nonnegative")
    if not math.isfinite(beta_m_per_s) or beta_m_per_s <= 0.0:
        raise ValueError("beta_m_per_s must be positive and finite")
    if not math.isfinite(confirmation_delay_sec) or confirmation_delay_sec < 0.0:
        raise ValueError("confirmation_delay_sec must be finite and nonnegative")
    delay = (
        source_distance_m.reshape(-1) / float(beta_m_per_s)
        + float(confirmation_delay_sec)
    )
    visible = torch.clamp(float(previous_horizon_sec) - delay, min=0.0)
    return torch.floor(visible + 1.0e-12).to(torch.long).clamp(
        min=0,
        max=int(source_steps),
    )


def project_stateful_rate(
    previous_state_rate: torch.Tensor,
    raw_current_rate: torch.Tensor,
    *,
    source_distance_m: torch.Tensor,
    current_horizon_sec: int,
    beta_m_per_s: float,
    confirmation_delay_sec: float = CONFIRMATION_DELAY_SEC,
) -> tuple[torch.Tensor, torch.Tensor]:
    if previous_state_rate.ndim != 2:
        raise ValueError("previous_state_rate must have shape (batch, source_time)")
    if raw_current_rate.shape != previous_state_rate.shape:
        raise ValueError("raw and previous state rates must have identical shapes")
    if current_horizon_sec <= 1:
        raise ValueError("current_horizon_sec must be greater than one")
    if bool(torch.any(previous_state_rate < 0.0)) or bool(
        torch.any(raw_current_rate < 0.0)
    ):
        raise ValueError("stateful projection requires nonnegative STF rates")
    steps = delayed_confirmed_steps(
        source_distance_m,
        previous_horizon_sec=current_horizon_sec - 1,
        beta_m_per_s=beta_m_per_s,
        confirmation_delay_sec=confirmation_delay_sec,
        source_steps=raw_current_rate.shape[1],
    )
    source_index = torch.arange(
        raw_current_rate.shape[1],
        device=raw_current_rate.device,
    ).unsqueeze(0)
    locked = source_index < steps.unsqueeze(1)
    state = torch.where(locked, previous_state_rate, raw_current_rate)
    return state, steps


def _confirmed_revision_values(
    previous_rate: torch.Tensor,
    current_rate: torch.Tensor,
    *,
    supported_steps: torch.Tensor,
    source_dt_sec: torch.Tensor,
) -> tuple[list[float], list[bool]]:
    source_index = torch.arange(
        previous_rate.shape[1],
        device=previous_rate.device,
    ).unsqueeze(0)
    mask = source_index < supported_steps.unsqueeze(1)
    comparable = supported_steps > 0
    if not bool(comparable.any()):
        return [], []
    dt = source_dt_sec.reshape(-1, 1)
    previous_cumulative = torch.cumsum(previous_rate * dt, dim=1).clamp_min(1.0e10)
    current_cumulative = torch.cumsum(current_rate * dt, dim=1).clamp_min(1.0e10)
    absolute_log = (
        torch.abs(
            torch.log10(current_cumulative)
            - torch.log10(previous_cumulative)
        )
        * mask
    ).sum(dim=1) / supported_steps.clamp_min(1)
    previous_confirmed = (previous_rate * dt * mask).sum(dim=1)
    current_confirmed = (current_rate * dt * mask).sum(dim=1)
    return (
        [float(value) for value in absolute_log[comparable].cpu()],
        [
            bool(value)
            for value in (
                current_confirmed[comparable] < previous_confirmed[comparable]
            ).cpu()
        ],
    )


def _streaming_metrics(
    *,
    endpoint_station_rows: list[dict[str, Any]],
    endpoint_event_rows: list[dict[str, Any]],
    station_steps: Sequence[float],
    event_steps: Sequence[float],
    confirmed_log_revisions: Sequence[float],
    confirmed_downward: Sequence[bool],
) -> dict[str, Any]:
    endpoint = summarize_predictions(
        endpoint_station_rows,
        endpoint_event_rows,
        reference_key="mw_catalog",
    )
    station_summary = _absolute_summary(station_steps)
    event_summary = _absolute_summary(event_steps)
    confirmed_summary = _absolute_summary(confirmed_log_revisions)
    return {
        "validation_station_count": int(endpoint["station_count"]),
        "validation_event_count": int(endpoint["event_count"]),
        "endpoint_event_mae": float(endpoint["event_mae"]),
        "endpoint_station_mae": float(endpoint["station_mae"]),
        "late_event_abs_step_median_mw": event_summary["median"],
        "late_event_abs_step_p95_mw": event_summary["p95"],
        "late_event_abs_step_max_mw": event_summary["maximum"],
        "late_station_abs_step_median_mw": station_summary["median"],
        "late_station_abs_step_p95_mw": station_summary["p95"],
        "late_station_abs_step_max_mw": station_summary["maximum"],
        "late_confirmed_cumulative_log10_l1_median": confirmed_summary["median"],
        "late_confirmed_cumulative_log10_l1_p95": confirmed_summary["p95"],
        "late_confirmed_cumulative_log10_l1_max": confirmed_summary["maximum"],
        "late_confirmed_downward_fraction": (
            float(np.mean(confirmed_downward))
            if confirmed_downward
            else float("nan")
        ),
        "late_confirmed_comparison_count": len(confirmed_downward),
    }


def _station_rows(
    *,
    events: Sequence[str],
    stations: Sequence[str],
    catalogs: Sequence[float],
    predictions: Sequence[float],
) -> list[dict[str, Any]]:
    return [
        {
            "event": events[index],
            "station": stations[index],
            "mw_catalog": catalogs[index],
            "mw_pred": predictions[index],
        }
        for index in range(len(events))
    ]


def _plot_event_mae(
    horizon_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    horizons = [int(row["horizon_sec"]) for row in horizon_rows]
    raw = [float(row["raw_event_mae"]) for row in horizon_rows]
    stateful = [float(row["stateful_event_mae"]) for row in horizon_rows]
    figure, axis = plt.subplots(figsize=(10.0, 5.2))
    axis.plot(horizons, raw, color="#6b7280", linewidth=1.2, label="Raw Phase39")
    axis.plot(
        horizons,
        stateful,
        color="#006d77",
        linewidth=1.8,
        label="Stateful projection",
    )
    axis.axvspan(180, 200, color="#e9c46a", alpha=0.18, label="Late window")
    axis.set_xlabel("Issue time (s)")
    axis.set_ylabel("Validation event MAE (Mw)")
    axis.set_title("Phase39 internal validation streaming trajectory")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_normalized_gates(
    raw_metrics: Mapping[str, Any],
    stateful_metrics: Mapping[str, Any],
    output_path: Path,
) -> None:
    labels = ["Event step p95", "Station step p95", "Confirmed STF p95"]
    keys = [
        "late_event_abs_step_p95_mw",
        "late_station_abs_step_p95_mw",
        "late_confirmed_cumulative_log10_l1_p95",
    ]
    gate_keys = [
        "late_event_abs_step_p95_mw_max",
        "late_station_abs_step_p95_mw_max",
        "late_confirmed_cumulative_log10_l1_p95_max",
    ]
    raw = [float(raw_metrics[key]) / float(GATES[gate]) for key, gate in zip(keys, gate_keys)]
    stateful = [
        float(stateful_metrics[key]) / float(GATES[gate])
        for key, gate in zip(keys, gate_keys)
    ]
    x = np.arange(len(labels))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.6, 5.0))
    axis.bar(x - width / 2, raw, width, color="#6b7280", label="Raw Phase39")
    axis.bar(
        x + width / 2,
        stateful,
        width,
        color="#006d77",
        label="Stateful projection",
    )
    axis.axhline(1.0, color="#b42318", linestyle="--", linewidth=1.2, label="Gate")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Metric / frozen gate (lower is better)")
    axis.set_title("Late streaming stability against frozen gates")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _write_report(
    output_root: Path,
    *,
    raw_metrics: Mapping[str, Any],
    stateful_metrics: Mapping[str, Any],
    gate: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    verdict = "通过" if bool(gate["passed"]) else "未通过"
    report = f"""# Phase42：Phase39 流式 STF 状态投影

## 结论

冻结的 internal validation gate **{verdict}**。Phase39 网络、checkpoint 和
全部 1,010,850 个参数均未修改；本实验只在逐秒推理后增加状态投影。

| 指标 | 原始 Phase39 | 状态投影 | 冻结门限 |
|---|---:|---:|---:|
| 200 秒 Event MAE | {float(raw_metrics['endpoint_event_mae']):.6f} | {float(stateful_metrics['endpoint_event_mae']):.6f} | <= {float(GATES['endpoint_event_mae_max']):.6f} |
| 200 秒 Station MAE | {float(raw_metrics['endpoint_station_mae']):.6f} | {float(stateful_metrics['endpoint_station_mae']):.6f} | <= {float(GATES['endpoint_station_mae_max']):.6f} |
| 180--200 秒 Event 单步 p95 | {float(raw_metrics['late_event_abs_step_p95_mw']):.6f} | {float(stateful_metrics['late_event_abs_step_p95_mw']):.6f} | <= {float(GATES['late_event_abs_step_p95_mw_max']):.6f} |
| 180--200 秒 Station 单步 p95 | {float(raw_metrics['late_station_abs_step_p95_mw']):.6f} | {float(stateful_metrics['late_station_abs_step_p95_mw']):.6f} | <= {float(GATES['late_station_abs_step_p95_mw_max']):.6f} |
| 已支持 STF 累积对数改写 p95 | {float(raw_metrics['late_confirmed_cumulative_log10_l1_p95']):.6f} | {float(stateful_metrics['late_confirmed_cumulative_log10_l1_p95']):.6f} | <= {float(GATES['late_confirmed_cumulative_log10_l1_p95_max']):.6f} |

## 状态规则

每秒仍由原始 Phase39 产生一条新的非负 200 点 STF。若某个源时间点对应的
S 波证据已经到达并经过 6 秒确认窗口，该点沿用上一秒的状态；其余新支持区间
和未来尾部使用当前预测。因此早期总震级仍可升降，后期随着锁定历史增加应逐步稳定。

## 图件

- [逐秒 Event MAE](figure_event_mae.png)
- [晚期稳定性门限对比](figure_late_stability_gates.png)

## 边界

- 本次只使用原 `within_event_station` validation，未打开 internal test、外部
  8 事件或 grouped test。
- 该 validation 回放切分的是现有预处理张量，用于比较状态更新逻辑；它不等同于
  已发布外部评估中的逐秒原始波形重处理，也不能证明未见事件泛化。
- 通过只代表状态层可进入下一次冻结验证；不能把 Phase39 改称严格因果模型。

## 可复现性

- Source checkpoint SHA-256: `{provenance['source_checkpoint_sha256']}`
- Split assignment SHA-256: `{provenance['split_assignment_sha256']}`
- Git commit: `{provenance['git_commit']}`
"""
    (output_root / "REPORT_ZH.md").write_text(report, encoding="utf-8")


def evaluate_stateful_projection(
    *,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise ValueError(f"output root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_hashes = validate_source_artifacts()
    config = load_frozen_config()
    configure_runtime(42, device)
    train_loader, validation_loader, test_loader, split_manifest = get_data_loaders_v2(config)
    del train_loader, test_loader
    if split_manifest["assignment_sha256"] != EXPECTED_SPLIT_ASSIGNMENT_SHA256:
        raise ValueError("Phase39 split assignment changed")
    if len(validation_loader.dataset) != 385:
        raise ValueError("Phase39 validation record count changed")

    model = PINNModel(config).to(device)
    source_state = torch.load(PHASE39_CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(source_state, strict=True)
    model.eval()
    criterion = _build_stf_rate_criterion(config, device)
    beta_m_per_s = float(config["physics"]["beta"])

    events: list[str] = []
    stations: list[str] = []
    catalogs: list[float] = []
    raw_mw_by_horizon = {horizon: [] for horizon in HORIZONS}
    stateful_mw_by_horizon = {horizon: [] for horizon in HORIZONS}
    locked_fraction_sum = {horizon: 0.0 for horizon in HORIZONS}
    locked_fraction_count = {horizon: 0 for horizon in HORIZONS}
    raw_confirmed_log: list[float] = []
    stateful_confirmed_log: list[float] = []
    raw_confirmed_downward: list[bool] = []
    stateful_confirmed_downward: list[bool] = []

    with torch.no_grad():
        for batch in validation_loader:
            prepared = _prepare_v2_batch(batch, config, device)
            batch_size = int(prepared.radial.shape[0])
            events.extend([str(value) for value in batch["event"]])
            stations.extend([str(value) for value in batch["station"]])
            catalogs.extend(
                [float(value) for value in batch["magnitude_catalog"].view(-1)]
            )
            previous_raw_rate: torch.Tensor | None = None
            previous_stateful_rate: torch.Tensor | None = None
            for horizon in HORIZONS:
                encoded = model(
                    prepared.model_input[:, :, :horizon],
                    meta=prepared.metadata,
                )
                raw_rate = criterion._decode_rate(encoded)
                if not bool(torch.isfinite(raw_rate).all()) or bool(
                    torch.any(raw_rate < 0.0)
                ):
                    raise FloatingPointError("Phase39 produced an invalid STF rate")
                if previous_stateful_rate is None:
                    stateful_rate = raw_rate.clone()
                    locked_steps = torch.zeros(
                        batch_size,
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    stateful_rate, locked_steps = project_stateful_rate(
                        previous_stateful_rate,
                        raw_rate,
                        source_distance_m=prepared.source_distance_m,
                        current_horizon_sec=horizon,
                        beta_m_per_s=beta_m_per_s,
                    )
                raw_mw = moment_magnitude_from_rate(
                    raw_rate,
                    prepared.source_dt_sec,
                )
                stateful_mw = moment_magnitude_from_rate(
                    stateful_rate,
                    prepared.source_dt_sec,
                )
                raw_mw_by_horizon[horizon].extend(
                    [float(value) for value in raw_mw.cpu()]
                )
                stateful_mw_by_horizon[horizon].extend(
                    [float(value) for value in stateful_mw.cpu()]
                )
                locked_fraction_sum[horizon] += float(
                    locked_steps.float().sum().cpu()
                ) / float(raw_rate.shape[1])
                locked_fraction_count[horizon] += batch_size

                if (
                    horizon >= LATE_TRANSITION_START
                    and previous_raw_rate is not None
                    and previous_stateful_rate is not None
                ):
                    supported_steps = conservative_s_supported_steps(
                        prepared.source_distance_m,
                        previous_horizon_sec=horizon - 1,
                        beta_m_per_s=beta_m_per_s,
                        source_steps=raw_rate.shape[1],
                    )
                    raw_values, raw_down = _confirmed_revision_values(
                        previous_raw_rate,
                        raw_rate,
                        supported_steps=supported_steps,
                        source_dt_sec=prepared.source_dt_sec,
                    )
                    stateful_values, stateful_down = _confirmed_revision_values(
                        previous_stateful_rate,
                        stateful_rate,
                        supported_steps=supported_steps,
                        source_dt_sec=prepared.source_dt_sec,
                    )
                    raw_confirmed_log.extend(raw_values)
                    raw_confirmed_downward.extend(raw_down)
                    stateful_confirmed_log.extend(stateful_values)
                    stateful_confirmed_downward.extend(stateful_down)
                previous_raw_rate = raw_rate
                previous_stateful_rate = stateful_rate

    if len(events) != 385 or len(set(events)) != 30:
        raise ValueError("validation cohort identity changed")

    horizon_rows: list[dict[str, Any]] = []
    event_prediction_rows: list[dict[str, Any]] = []
    raw_event_predictions: dict[int, dict[str, float]] = {}
    stateful_event_predictions: dict[int, dict[str, float]] = {}
    raw_endpoint_station_rows: list[dict[str, Any]] = []
    stateful_endpoint_station_rows: list[dict[str, Any]] = []
    raw_endpoint_event_rows: list[dict[str, Any]] = []
    stateful_endpoint_event_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        raw_station_rows = _station_rows(
            events=events,
            stations=stations,
            catalogs=catalogs,
            predictions=raw_mw_by_horizon[horizon],
        )
        stateful_station_rows = _station_rows(
            events=events,
            stations=stations,
            catalogs=catalogs,
            predictions=stateful_mw_by_horizon[horizon],
        )
        raw_event_rows = aggregate_event_predictions(
            raw_station_rows,
            reference_key="mw_catalog",
        )
        stateful_event_rows = aggregate_event_predictions(
            stateful_station_rows,
            reference_key="mw_catalog",
        )
        raw_summary = summarize_predictions(
            raw_station_rows,
            raw_event_rows,
            reference_key="mw_catalog",
        )
        stateful_summary = summarize_predictions(
            stateful_station_rows,
            stateful_event_rows,
            reference_key="mw_catalog",
        )
        raw_event_predictions[horizon] = {
            str(row["event"]): float(row["mw_pred_median"])
            for row in raw_event_rows
        }
        stateful_event_predictions[horizon] = {
            str(row["event"]): float(row["mw_pred_median"])
            for row in stateful_event_rows
        }
        raw_station_step = (
            []
            if horizon == 1
            else np.asarray(raw_mw_by_horizon[horizon])
            - np.asarray(raw_mw_by_horizon[horizon - 1])
        )
        stateful_station_step = (
            []
            if horizon == 1
            else np.asarray(stateful_mw_by_horizon[horizon])
            - np.asarray(stateful_mw_by_horizon[horizon - 1])
        )
        raw_event_step = (
            []
            if horizon == 1
            else [
                raw_event_predictions[horizon][event]
                - raw_event_predictions[horizon - 1][event]
                for event in sorted(raw_event_predictions[horizon])
            ]
        )
        stateful_event_step = (
            []
            if horizon == 1
            else [
                stateful_event_predictions[horizon][event]
                - stateful_event_predictions[horizon - 1][event]
                for event in sorted(stateful_event_predictions[horizon])
            ]
        )
        horizon_rows.append(
            {
                "horizon_sec": horizon,
                "raw_event_mae": raw_summary["event_mae"],
                "stateful_event_mae": stateful_summary["event_mae"],
                "raw_station_mae": raw_summary["station_mae"],
                "stateful_station_mae": stateful_summary["station_mae"],
                "raw_event_abs_step_p95_mw": _absolute_summary(raw_event_step)["p95"],
                "stateful_event_abs_step_p95_mw": _absolute_summary(stateful_event_step)["p95"],
                "raw_station_abs_step_p95_mw": _absolute_summary(raw_station_step)["p95"],
                "stateful_station_abs_step_p95_mw": _absolute_summary(stateful_station_step)["p95"],
                "mean_locked_source_fraction": (
                    locked_fraction_sum[horizon] / locked_fraction_count[horizon]
                ),
            }
        )
        raw_by_event = {str(row["event"]): row for row in raw_event_rows}
        stateful_by_event = {str(row["event"]): row for row in stateful_event_rows}
        for event in sorted(raw_by_event):
            event_prediction_rows.append(
                {
                    "horizon_sec": horizon,
                    "event": event,
                    "mw_catalog": raw_by_event[event]["mw_catalog"],
                    "raw_mw_pred_median": raw_by_event[event]["mw_pred_median"],
                    "stateful_mw_pred_median": stateful_by_event[event]["mw_pred_median"],
                    "n_stations": raw_by_event[event]["n_stations"],
                }
            )
        if horizon == 200:
            raw_endpoint_station_rows = raw_station_rows
            stateful_endpoint_station_rows = stateful_station_rows
            raw_endpoint_event_rows = raw_event_rows
            stateful_endpoint_event_rows = stateful_event_rows

    raw_station_steps = [
        float(value)
        for horizon in range(LATE_TRANSITION_START, 201)
        for value in (
            np.asarray(raw_mw_by_horizon[horizon])
            - np.asarray(raw_mw_by_horizon[horizon - 1])
        )
    ]
    stateful_station_steps = [
        float(value)
        for horizon in range(LATE_TRANSITION_START, 201)
        for value in (
            np.asarray(stateful_mw_by_horizon[horizon])
            - np.asarray(stateful_mw_by_horizon[horizon - 1])
        )
    ]
    raw_event_steps = [
        raw_event_predictions[horizon][event]
        - raw_event_predictions[horizon - 1][event]
        for horizon in range(LATE_TRANSITION_START, 201)
        for event in sorted(raw_event_predictions[horizon])
    ]
    stateful_event_steps = [
        stateful_event_predictions[horizon][event]
        - stateful_event_predictions[horizon - 1][event]
        for horizon in range(LATE_TRANSITION_START, 201)
        for event in sorted(stateful_event_predictions[horizon])
    ]
    raw_metrics = _streaming_metrics(
        endpoint_station_rows=raw_endpoint_station_rows,
        endpoint_event_rows=raw_endpoint_event_rows,
        station_steps=raw_station_steps,
        event_steps=raw_event_steps,
        confirmed_log_revisions=raw_confirmed_log,
        confirmed_downward=raw_confirmed_downward,
    )
    stateful_metrics = _streaming_metrics(
        endpoint_station_rows=stateful_endpoint_station_rows,
        endpoint_event_rows=stateful_endpoint_event_rows,
        station_steps=stateful_station_steps,
        event_steps=stateful_event_steps,
        confirmed_log_revisions=stateful_confirmed_log,
        confirmed_downward=stateful_confirmed_downward,
    )
    _assert_formal_baseline(raw_metrics)
    gate = validation_gate(stateful_metrics)

    protocol = {
        "source_model": "Phase39 Glehman scalar + global invariant, seed42",
        "source_checkpoint": str(PHASE39_CHECKPOINT),
        "model_weights_changed": False,
        "horizons_sec": [1, 200],
        "state_rule": "retain previously delayed-S-confirmed bins; use current raw STF elsewhere",
        "confirmation_delay_sec": CONFIRMATION_DELAY_SEC,
        "beta_m_per_s": beta_m_per_s,
        "late_transition_window_sec": [LATE_TRANSITION_START, 200],
        "baseline_metrics": dict(BASELINE_METRICS),
        "gates": dict(GATES),
        "input_interpretation": "prefix slices of the frozen internal processed tensor",
        "hidden_data": "internal test, external development events, and grouped test are not iterated",
    }
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(PROJECT_ROOT),
        "git_dirty": git_is_dirty(PROJECT_ROOT),
        "device": str(device),
        "source_checkpoint_sha256": source_hashes["checkpoint"],
        "source_config_sha256": source_hashes["config"],
        "source_split_sha256": source_hashes["split"],
        "split_assignment_sha256": split_manifest["assignment_sha256"],
        "internal_test_iterated": False,
        "external_data_loaded": False,
        "grouped_test_loaded": False,
    }
    _write_json(output_root / "protocol.json", protocol)
    _write_json(output_root / "provenance.json", provenance)
    _write_csv(
        output_root / "horizon_metrics.csv",
        horizon_rows,
        fieldnames=tuple(horizon_rows[0]),
    )
    _write_csv(
        output_root / "event_predictions.csv",
        event_prediction_rows,
        fieldnames=tuple(event_prediction_rows[0]),
    )
    endpoint_rows = []
    for raw_row, stateful_row in zip(
        raw_endpoint_station_rows,
        stateful_endpoint_station_rows,
        strict=True,
    ):
        endpoint_rows.append(
            {
                "event": raw_row["event"],
                "station": raw_row["station"],
                "mw_catalog": raw_row["mw_catalog"],
                "raw_mw_pred": raw_row["mw_pred"],
                "stateful_mw_pred": stateful_row["mw_pred"],
            }
        )
    _write_csv(
        output_root / "endpoint_station_predictions.csv",
        endpoint_rows,
        fieldnames=tuple(endpoint_rows[0]),
    )
    _plot_event_mae(horizon_rows, output_root / "figure_event_mae.png")
    _plot_normalized_gates(
        raw_metrics,
        stateful_metrics,
        output_root / "figure_late_stability_gates.png",
    )
    summary = {
        "status": "validation_gate_passed" if gate["passed"] else "validation_gate_failed",
        "passed": bool(gate["passed"]),
        "source_checkpoint_sha256": sha256_file(PHASE39_CHECKPOINT),
        "model_weights_changed": False,
        "raw_metrics": raw_metrics,
        "stateful_metrics": stateful_metrics,
        "stateful_gate": gate,
        "protocol": protocol,
        "provenance": provenance,
    }
    _write_json(output_root / "summary.json", summary)
    _write_report(
        output_root,
        raw_metrics=raw_metrics,
        stateful_metrics=stateful_metrics,
        gate=gate,
        provenance=provenance,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a no-training stateful STF projection for Phase39.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="New or empty output directory.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    summary = evaluate_stateful_projection(
        output_root=args.output_root.resolve(),
        device=torch.device(args.device),
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "status": summary["status"],
                "passed": summary["passed"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
