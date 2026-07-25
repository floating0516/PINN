from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.plotting import plot_phase27_complete_results_zh as zh
from scripts.plotting.plot_phase27_magnitude_convergence import (
    _magnitude_group,
    build_event_convergence_rows,
    build_magnitude_group_rows,
)


def _external_source_rows() -> list[dict[str, str]]:
    catalogs = {
        "Iquique 2014 M7.7": 7.7,
        "Nepal 2015 M7.3": 7.3,
        "Kodiak 2018 M7.9": 7.9,
        "Samos 2020 M7.0": 7.0,
        "Xizang 2025 M7.1": 7.1,
        "Mandalay 2025 M7.7": 7.7,
        "Sand 2025 M7.3": 7.3,
    }
    stations = {
        "Iquique 2014 M7.7": 11,
        "Nepal 2015 M7.3": 5,
        "Kodiak 2018 M7.9": 46,
        "Samos 2020 M7.0": 3,
        "Xizang 2025 M7.1": 2,
        "Mandalay 2025 M7.7": 5,
        "Sand 2025 M7.3": 4,
    }
    errors = {
        "phase27": (
            -0.3143350601196291,
            -0.21706943511962873,
            0.13892564773559535,
            -0.10299921035766602,
            0.03872551918029821,
            0.13820343017578107,
            -0.14363412857055646,
        ),
        "crowell": (
            -0.22247982391679333,
            -0.13433403498594387,
            0.20493789033979404,
            -0.1315534065907018,
            -0.025655219721169686,
            0.2855141485223882,
            -0.18300520793385733,
        ),
        "melgar": (
            -0.38678252689834025,
            -0.1741908659457163,
            -0.1617110809058211,
            -0.2856820840507206,
            -0.2245589263085197,
            0.0355525721356269,
            -0.4268519966021378,
        ),
        "ruhl": (
            -0.586501289503536,
            -0.37845659670154763,
            -0.2860375679822802,
            -0.518617657038762,
            -0.4426297317245478,
            -0.09856666404174419,
            -0.6428353345117586,
        ),
    }
    fields = {
        "phase27": ("mw_pred_median", "error_vs_catalog", "n_stations"),
        "crowell": (
            "pgd_crowell_mw_pred_median",
            "pgd_crowell_error",
            "pgd_crowell_n_stations",
        ),
        "melgar": (
            "pgd_melgar_mw_pred_median",
            "pgd_melgar_error",
            "pgd_melgar_n_stations",
        ),
        "ruhl": (
            "pgd_ruhl_mw_pred_median",
            "pgd_ruhl_error",
            "pgd_ruhl_n_stations",
        ),
    }
    rows: list[dict[str, str]] = []
    for index, event in enumerate(zh.EXPECTED_EXTERNAL_EVENTS):
        catalog = catalogs[event]
        row = {
            "event": event,
            "mw_catalog": str(catalog),
            "observation_horizon_sec": "200.0",
            "release_time_sec": "205.0",
        }
        for method, (prediction_field, error_field, station_field) in fields.items():
            error = errors[method][index]
            row[prediction_field] = str(catalog + error)
            row[error_field] = str(error)
            row[station_field] = str(stations[event])
        rows.append(row)
    return rows


def _synthetic_data(tmp_path: Path) -> dict[str, object]:
    input_a = tmp_path / "input-a.json"
    input_b = tmp_path / "input-b.csv"
    input_a.write_text("{}\n", encoding="utf-8")
    input_b.write_text("a\n1\n", encoding="utf-8")

    magnitudes = [
        *[6.0 + 0.16 * index for index in range(6)],
        *[7.0 + 0.05 * index for index in range(17)],
        *[8.1 + 0.10 * index for index in range(7)],
    ]
    prediction_rows: list[dict[str, object]] = []
    for index, mw in enumerate(magnitudes):
        event = f"Event{index:02d}"
        final_error = 0.08 if index % 2 == 0 else -0.08
        for horizon in zh.EXPECTED_HORIZONS:
            early_deficit = (200.0 - horizon) / 180.0 * (0.35 + 0.25 * (mw >= 8.0))
            prediction = mw + final_error - early_deficit
            error = prediction - mw
            prediction_rows.append(
                {
                    "event": event,
                    "observation_horizon_sec": horizon,
                    "release_time_sec": horizon + 5.0,
                    "mw_pred_median": prediction,
                    "mw_catalog": mw,
                    "error": error,
                    "abs_error": abs(error),
                    "station_count": 1 + index % 12,
                    "magnitude_group": _magnitude_group(mw),
                }
            )
    convergence_rows = build_event_convergence_rows(prediction_rows)
    group_rows = build_magnitude_group_rows(prediction_rows)
    horizon_metrics = []
    for horizon in zh.EXPECTED_HORIZONS:
        rows = [
            row
            for row in prediction_rows
            if float(row["observation_horizon_sec"]) == horizon
        ]
        errors = [float(row["error"]) for row in rows]
        horizon_metrics.append(
            {
                "observation_horizon_sec": horizon,
                "release_time_sec": horizon + 5.0,
                "event_count": 30,
                "available_station_count": sum(int(row["station_count"]) for row in rows),
                "event_equal_mae": sum(abs(value) for value in errors) / 30,
                "event_equal_rmse": (sum(value * value for value in errors) / 30) ** 0.5,
                "event_equal_bias": sum(errors) / 30,
            }
        )
    overall_rows = zh.build_overall_horizon_rows(
        prediction_rows, horizon_metrics, convergence_rows
    )
    final_rows = zh.build_final_event_rows(convergence_rows)
    external_rows, method_rows = zh.build_external_comparison_rows(
        _external_source_rows()
    )
    return {
        "paths": {"input_a": input_a, "input_b": input_b},
        "prediction_rows": prediction_rows,
        "convergence_rows": convergence_rows,
        "group_rows": group_rows,
        "horizon_metrics": horizon_metrics,
        "overall_rows": overall_rows,
        "final_event_rows": final_rows,
        "external_rows": external_rows,
        "external_method_rows": method_rows,
        "metrics": {
            "event_mae": 0.1372873624165853,
            "event_bias": -0.01514135996500651,
            "station_mae": 0.10736585344587053,
            "event_count": 30,
            "station_count": 385,
        },
        "selection": {
            "candidates": {
                "17": 0.11135358810424804,
                "42": 0.1320822874704997,
                "73": 0.20192739168802898,
            }
        },
        "internal_summary": {
            "validation_gate": {"candidate": 0.11135358810424804}
        },
        "external_summary": {
            "thresholds": {"cm0": {"event_metrics": {"mae": 0.22668169736862187}}}
        },
        "registry": {"checkpoint": {"sha256": "c" * 64}},
        "sampling": {
            "objective_weight_minimum": 0.11966269575692678,
            "objective_weight_maximum": 57.67741935483871,
        },
        "preflight_summary": {
            "accepted_event_count": 31,
            "accepted_station_count": 2558,
            "source_data": {"sha256": "2" * 64},
        },
    }


def test_external_comparison_reproduces_frozen_same_event_metrics() -> None:
    event_rows, summaries = zh.build_external_comparison_rows(
        _external_source_rows()
    )
    assert len(event_rows) == 28
    assert len(summaries) == 4
    assert {row["method"]: row["event_mae"] for row in summaries} == pytest.approx(
        zh.EXPECTED_EXTERNAL_METHOD_MAE, abs=2e-12
    )

    changed = _external_source_rows()
    changed[0]["pgd_crowell_error"] = str(
        float(changed[0]["pgd_crowell_error"]) + 0.01
    )
    changed[0]["pgd_crowell_mw_pred_median"] = str(
        float(changed[0]["pgd_crowell_mw_pred_median"]) + 0.01
    )
    with pytest.raises(ValueError, match="external crowell MAE changed"):
        zh.build_external_comparison_rows(changed)


def test_generate_bundle_creates_complete_chinese_gallery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _synthetic_data(tmp_path)
    monkeypatch.setattr(zh, "load_complete_inputs", lambda _: data)
    output = tmp_path / "bundle-zh"

    artifacts = zh.generate_bundle(run_dir=tmp_path, output_dir=output)

    figure_names = (
        "01_method_and_causal_timing",
        "02_overall_internal_convergence",
        "03_magnitude_group_convergence",
        "04_high_magnitude_event_trajectories",
        "05_convergence_time_by_magnitude",
        "06_final_event_errors_and_station_counts",
        "07_external_cm2_method_comparison",
    )
    csv_names = (
        "overall_horizon_metrics.csv",
        "magnitude_group_horizon_metrics.csv",
        "event_predictions_by_horizon.csv",
        "event_convergence_summary.csv",
        "final_event_errors.csv",
        "external_cm2_event_comparison.csv",
        "external_cm2_method_summary.csv",
    )
    expected = {
        *(f"figures/{name}.{suffix}" for name in figure_names for suffix in ("png", "pdf")),
        *csv_names,
        "README.md",
        "publication_manifest.json",
    }
    assert set(artifacts) == expected
    assert len(artifacts) == 23
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts.values())
    assert all(artifacts[f"figures/{name}.png"].read_bytes().startswith(b"\x89PNG") for name in figure_names)
    assert all(artifacts[f"figures/{name}.pdf"].read_bytes().startswith(b"%PDF") for name in figure_names)
    readme = artifacts["README.md"].read_text(encoding="utf-8")
    assert "单台站 R-only、STF 积分定震级" in readme
    assert "不是端到端实时因果" in readme
    assert "Luding 2022 M6.6" in readme
    assert "log10(1 + Mdot0 / M_ref)" in readme
    assert "M_ref = 1e18 N·m/s" in readme
    assert readme.count("![") == 7
    assert "| |误差|" not in readme
    assert "绝对误差≤0.15 Mw" in readme
    assert all(line == line.rstrip() for line in readme.splitlines())
    for name in csv_names:
        assert b"\r\n" not in artifacts[name].read_bytes()

    manifest = json.loads(artifacts["publication_manifest.json"].read_text())
    assert manifest["locale"] == "zh-CN"
    assert manifest["font"]["family"] in zh.ZH_FONT_CANDIDATES
    assert manifest["analysis_contract"]["external_cm2_coverage"] == "7/8"
    assert manifest["analysis_contract"]["station_selection_causal"] is False
    assert len(manifest["outputs"]) == 22
    for relative, record in manifest["outputs"].items():
        digest = hashlib.sha256((output / relative).read_bytes()).hexdigest()
        assert digest == record["sha256"]

    with pytest.raises(FileExistsError):
        zh.generate_bundle(run_dir=tmp_path, output_dir=output)


def test_convergence_figure_states_locked_retrospective_scope() -> None:
    note = zh.CONVERGENCE_SCOPE_NOTE_ZH
    assert "锁定内部测试" in note
    assert "同事件未见台站划分" in note
    assert "完整 200 s 径向峰值回顾性确定" in note


def test_generate_bundle_is_atomic_after_plot_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _synthetic_data(tmp_path)
    monkeypatch.setattr(zh, "load_complete_inputs", lambda _: data)

    def fail_plot(**_: object) -> tuple[Path, Path]:
        raise RuntimeError("synthetic plot failure")

    monkeypatch.setattr(zh, "plot_overall_internal_convergence", fail_plot)
    output = tmp_path / "failed-bundle"
    with pytest.raises(RuntimeError, match="synthetic plot failure"):
        zh.generate_bundle(run_dir=tmp_path, output_dir=output)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-bundle.tmp-*"))
