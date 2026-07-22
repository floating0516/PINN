from __future__ import annotations

import copy
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from scripts.experiments.run_corrected_matrix import (
    MATRIX_IDS,
    SEEDS,
    _completed_row_is_valid,
    _prepare_output_root,
    config_diff_paths,
    load_matrix_configs,
    metrics_at_thresholds,
    select_configuration,
)
from src.data import dataset_v2, loaders_v2
from src.data.dataset_v2 import CorrectedEarthquakeDataset
from src.data.metadata import metadata_distance_from_config
from src.data.records_v2 import NormalizedStationRecord
from src.data.stf import ProcessedSTF, moment_to_mw
from src.evaluation.evaluate_unseen import write_unseen_event_outputs
from src.training.train import _prepare_v2_batch, _select_early_stop_value
from src.utils.config_v2 import validate_config_v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments_v2"

EXPECTED_IDS = (
    "V2-BASE",
    "V2-FULL",
    "V2-NOSYNTH",
    "V2-NOSTF",
    "V2-NOMETA",
    "V2-GAIN144",
    "V2-DELTA-META",
    "V2-CATALOG-SCALED-STF",
)

EXPECTED_DIFFS = {
    "V2-BASE": set(),
    "V2-FULL": {"training.stf_rate_loss.include_intermediate_field"},
    "V2-NOSYNTH": {"training.stf_rate_loss.lambda_synth"},
    "V2-NOSTF": {
        "training.stf_rate_loss.lambda_MSE",
        "training.stf_rate_loss.lambda_shape",
    },
    "V2-NOMETA": {"model.use_meta"},
    "V2-GAIN144": {"physics.amplitude_gain"},
    "V2-DELTA-META": {"geometry.network_distance"},
    "V2-CATALOG-SCALED-STF": {"dataset.stf.magnitude_target"},
}


def test_fixed_matrix_contains_only_the_eight_declared_variants() -> None:
    configs = load_matrix_configs(CONFIG_DIR)

    assert MATRIX_IDS == EXPECTED_IDS
    assert SEEDS == (17, 42, 73)
    assert tuple(configs) == EXPECTED_IDS
    base = configs["V2-BASE"]
    for experiment_id, config in configs.items():
        validate_config_v2(config)
        assert config["pipeline_version"] == 2
        assert config_diff_paths(base, config) == EXPECTED_DIFFS[experiment_id]

    assert base["geometry"]["network_distance"] == "hypocentral"
    assert base["dataset"]["stf"]["magnitude_target"] == "stf_native"
    assert configs["V2-FULL"]["training"]["stf_rate_loss"]["include_intermediate_field"] is True
    assert configs["V2-NOSYNTH"]["training"]["stf_rate_loss"]["lambda_synth"] == 0.0
    assert configs["V2-NOSTF"]["training"]["stf_rate_loss"]["lambda_MSE"] == 0.0
    assert configs["V2-NOSTF"]["training"]["stf_rate_loss"]["lambda_shape"] == 0.0
    assert configs["V2-NOMETA"]["model"]["use_meta"] is False
    assert configs["V2-GAIN144"]["physics"]["amplitude_gain"] == 1.44
    assert configs["V2-DELTA-META"]["geometry"]["network_distance"] == "epicentral"
    assert configs["V2-CATALOG-SCALED-STF"]["dataset"]["stf"]["magnitude_target"] == "catalog"


def _prepared_batch() -> dict[str, torch.Tensor]:
    return {
        "radial": torch.ones(1, 1, 4),
        "source_distance_m": torch.tensor([5_000.0]),
        "epicentral_distance_m": torch.tensor([3_000.0]),
        "theta_deg": torch.tensor([30.0]),
        "azimuth_deg": torch.tensor([90.0]),
        "phi_slip_deg": torch.tensor([45.0]),
        "stf": torch.full((1, 4), 1.0e18),
        "stf_dt_sec": torch.tensor([1.0]),
        "waveform_dt_sec": torch.tensor([1.0]),
        "waveform_valid_mask": torch.ones(1, 4, dtype=torch.bool),
        "has_stf": torch.tensor([True]),
        "mw_stf_native": torch.tensor([6.5]),
        "magnitude_catalog": torch.tensor([7.0]),
    }


def test_delta_metadata_changes_only_network_distance() -> None:
    config = {
        "geometry": {"network_distance": "epicentral"},
        "dataset": {"stf": {"magnitude_target": "stf_native"}},
    }
    batch = _prepared_batch()

    selected = metadata_distance_from_config(
        config,
        source_distance_m=batch["source_distance_m"],
        epicentral_distance_m=batch["epicentral_distance_m"],
    )
    prepared = _prepare_v2_batch(batch, config, torch.device("cpu"))

    torch.testing.assert_close(selected, batch["epicentral_distance_m"])
    torch.testing.assert_close(prepared.source_distance_m, batch["source_distance_m"])
    torch.testing.assert_close(
        prepared.metadata[:, 0],
        torch.log(batch["epicentral_distance_m"]),
    )


def test_unknown_network_metadata_distance_is_rejected() -> None:
    with pytest.raises(ValueError, match="network_distance"):
        metadata_distance_from_config(
            {"geometry": {"network_distance": "ambiguous"}},
            source_distance_m=torch.tensor([1.0]),
            epicentral_distance_m=torch.tensor([1.0]),
        )


def _record(station: str) -> NormalizedStationRecord:
    values = np.arange(4.0)
    return NormalizedStationRecord(
        event_index=0,
        event="Event A",
        magnitude_catalog=7.0,
        event_lat=35.0,
        event_lon=140.0,
        depth_km=10.0,
        strike=0.0,
        dip=45.0,
        rake=90.0,
        mechanism=2,
        station=station,
        station_lat=35.1,
        station_lon=140.0,
        time_sec=values,
        east=values,
        north=values,
        vertical=values,
        origin_sec=None,
    )


def test_catalog_scaled_stf_is_event_invariant_and_conserves_catalog_moment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_rate = np.full(4, 1.0e18)
    native_moment = float(np.sum(native_rate))
    processed = ProcessedSTF(
        time_sec=np.arange(4.0),
        rate_nm_per_s=native_rate,
        dt_sec=1.0,
        native_moment_nm=native_moment,
        grid_moment_before_rescale_nm=native_moment,
        retained_moment_fraction=1.0,
        mw_native=moment_to_mw(native_moment),
    )
    dataset = CorrectedEarthquakeDataset.__new__(CorrectedEarthquakeDataset)
    dataset.config = {
        "dataset": {"units": "m", "radial_peak_min_cm": 0.0, "stf": {"magnitude_target": "catalog"}},
        "physics": {"alpha": 7_900.0},
    }
    dataset.stf_m_ref = 1.0e18
    dataset.stf_output_steps = 4
    dataset.waveform_config = object()
    dataset.blacklist = set()
    dataset.allow_missing_stf = False
    dataset.samples = []
    dataset.rejections = []
    dataset.manifest_rows = []
    monkeypatch.setattr(dataset, "_event_stf", lambda event: (processed, "event.stf"))
    monkeypatch.setattr(
        dataset_v2,
        "build_station_sample",
        lambda *args, **kwargs: {"station": args[0].station},
    )
    monkeypatch.setattr(dataset_v2, "make_manifest_row", lambda *args, **kwargs: {})

    dataset._consume_record(_record("S1"))
    dataset._consume_record(_record("S2"))

    expected_moment = 10.0 ** (1.5 * 7.0 + 9.1)
    first, second = dataset.samples
    assert np.array_equal(first["stf"], second["stf"])
    assert float(np.sum(first["stf"]) * first["stf_dt_sec"]) == pytest.approx(
        expected_moment,
        rel=2.0e-7,
    )
    np.testing.assert_allclose(
        first["stf_log"],
        np.log10(1.0 + first["stf"] / dataset.stf_m_ref),
        rtol=1.0e-6,
    )
    assert first["mw_stf_native"] == processed.mw_native


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("val_loss", 3.0),
        ("mw_mae", 2.0),
        ("event_mae_catalog", 1.0),
    ],
)
def test_early_stop_metric_dispatch_is_explicit(metric: str, expected: float) -> None:
    value = _select_early_stop_value(
        metric,
        val_loss=3.0,
        station_mw_mae=2.0,
        event_mae_catalog=1.0,
    )

    assert value == expected


def test_unknown_early_stop_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="early_stop_metric"):
        _select_early_stop_value(
            "silent-fallback",
            val_loss=3.0,
            station_mw_mae=2.0,
            event_mae_catalog=1.0,
        )


def _external_rows() -> list[dict[str, float | str]]:
    return [
        {"event": "A", "station": "A1", "mw_pred": 6.0, "mw_catalog": 7.0, "mw_stf_native": 6.9, "max_radial_cm": 0.5},
        {"event": "A", "station": "A2", "mw_pred": 7.0, "mw_catalog": 7.0, "mw_stf_native": 6.9, "max_radial_cm": 2.0},
        {"event": "A", "station": "A3", "mw_pred": 8.0, "mw_catalog": 7.0, "mw_stf_native": 6.9, "max_radial_cm": 2.5},
        {"event": "B", "station": "B1", "mw_pred": 7.0, "mw_catalog": 7.0, "mw_stf_native": 7.1, "max_radial_cm": 3.0},
    ]


def test_external_threshold_metrics_use_strict_cm_cutoffs_and_event_medians() -> None:
    metrics = metrics_at_thresholds(_external_rows())

    assert tuple(metrics) == ("cm0", "cm1", "cm2")
    assert metrics["cm0"]["station_count"] == 4
    assert metrics["cm0"]["event_mae_catalog"] == pytest.approx(0.0)
    assert metrics["cm1"]["station_count"] == 3
    assert metrics["cm1"]["event_mae_catalog"] == pytest.approx(0.25)
    assert metrics["cm2"]["station_count"] == 2
    assert metrics["cm2"]["event_mae_catalog"] == pytest.approx(0.5)


def _selection_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for experiment_id, mae, bias, parameters in (
        ("V2-BASE", 0.20, 0.03, 100),
        ("V2-FULL", 0.20, -0.02, 120),
        ("V2-NOMETA", 0.20, 0.02, 90),
    ):
        for seed in SEEDS:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "seed": seed,
                    "status": "ok",
                    "cm2_event_mae_catalog": mae,
                    "cm2_event_bias_catalog": bias,
                    "parameter_count": parameters,
                }
            )
    return rows


def test_selection_uses_cm2_mae_then_absolute_bias_then_parameter_count() -> None:
    selection = select_configuration(_selection_rows(), required_seeds=SEEDS)

    assert selection["experiment_id"] == "V2-NOMETA"
    assert selection["cm2_event_mae_catalog_mean"] == pytest.approx(0.20)
    assert selection["cm2_absolute_event_bias_catalog_mean"] == pytest.approx(0.02)
    assert selection["parameter_count"] == 90


def test_selection_rejects_incomplete_seed_sets() -> None:
    rows = _selection_rows()
    rows.pop()

    with pytest.raises(ValueError, match="seed"):
        select_configuration(rows, required_seeds=SEEDS)


def test_resume_skip_requires_intact_hashed_artifacts(tmp_path: Path) -> None:
    from src.utils.provenance import sha256_file

    checkpoint = tmp_path / "best_model.pth"
    training_log = tmp_path / "training.csv"
    run_manifest = tmp_path / "run_manifest.json"
    checkpoint.write_bytes(b"checkpoint")
    training_log.write_text("Epoch,Loss\n1,0.1\n", encoding="utf-8")
    checkpoint_hash = sha256_file(checkpoint)
    run_manifest.write_text(
        "{"
        '"completed_at_utc":"2026-07-22T00:00:00+00:00",'
        f'"checkpoint_sha256":"{checkpoint_hash}"'
        "}\n",
        encoding="utf-8",
    )

    row = {
        "status": "ok",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "training_log_path": str(training_log),
        "training_log_sha256": sha256_file(training_log),
        "run_manifest_path": str(run_manifest),
    }

    assert _completed_row_is_valid(row) is True
    checkpoint.write_bytes(b"corrupted")
    assert _completed_row_is_valid(row) is False


def test_new_campaign_allows_only_a_precreated_console_log(tmp_path: Path) -> None:
    output_root = tmp_path / "campaign"
    output_root.mkdir()
    (output_root / "console.log").write_text("starting\n", encoding="utf-8")

    _prepare_output_root(output_root, resume=False)
    (output_root / "unexpected.txt").write_text("wrong run\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        _prepare_output_root(output_root, resume=False)


class _SmokeDataset(Dataset):
    def __init__(self, config: dict) -> None:
        del config
        self.samples = [
            {"event": event, "value": torch.tensor(index)}
            for index, event in enumerate("AABBCCDDEE")
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.samples[index]


def test_smoke_loader_limits_the_real_pipeline_by_whole_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loaders_v2, "CorrectedEarthquakeDataset", _SmokeDataset)
    config = {
        "pipeline_version": 2,
        "training": {
            "split_protocol": "grouped_event",
            "validation_event_fraction": 0.2,
            "test_event_fraction": 0.0,
            "random_seed": 17,
            "batch_size": 2,
            "event_balanced_sampling": True,
            "num_workers": 0,
        },
    }

    _, _, _, manifest = loaders_v2.get_data_loaders_v2(config, max_events=3)

    retained = set(manifest["train_events"] + manifest["validation_events"])
    assert retained == {"A", "B", "C"}
    assert manifest["train_record_count"] + manifest["validation_record_count"] == 6


def test_metrics_only_external_output_writes_csv_without_figures(tmp_path: Path) -> None:
    station_rows = _external_rows()
    event_rows = [
        {
            "event": "A",
            "mw_catalog": 7.0,
            "mw_pred_median": 7.0,
            "error_vs_catalog": 0.0,
        }
    ]

    paths = write_unseen_event_outputs(
        output_dir=tmp_path,
        station_rows=station_rows,
        event_rows=event_rows,
        save_plots=False,
    )

    assert paths["station_csv"].is_file()
    assert paths["event_csv"].is_file()
    assert not paths["station_scatter"].exists()
    assert not paths["event_summary_figure"].exists()
    assert paths["station_panels"] == {}
    assert paths["event_mw_figures"] == {}


def test_corrected_matrix_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/run_corrected_matrix.py",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "external-validation" in result.stdout
    assert "within-event-station" in result.stdout
    assert "--max-events" in result.stdout
    assert "--dataset-manifest" in result.stdout
