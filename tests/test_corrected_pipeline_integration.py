from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.data import dataset_v2
from src.data.dataset_v2 import (
    CorrectedEarthquakeDataset,
    _load_stf_files,
)
from src.data.loaders_v2 import get_data_loaders_v2
from src.evaluation.evaluate import evaluate
from src.models.model import PINNModel
from src.training.train import train


def make_v2_dataset_config(
    *,
    npz_path: Path,
    stf_dir: Path,
    allow_missing_stf: bool = False,
    radial_peak_min_cm: float = 0.0,
) -> dict[str, Any]:
    return {
        "pipeline_version": 2,
        "dataset": {
            "blacklist_events": [],
            "units": "m",
            "sample_rate_hz": 1.0,
            "radial_peak_min_cm": radial_peak_min_cm,
            "allow_missing_stf": allow_missing_stf,
            "waveform": {
                "start_sec": 0.0,
                "duration_sec": 200.0,
                "min_valid_fraction": 1.0,
                "max_interpolation_gap_sec": 2.5,
            },
            "baseline": {
                "method": "median",
                "pre_event_start_sec": -20.0,
                "pre_event_end_sec": 0.0,
                "fallback": "pre_p",
                "fallback_max_sec": 30.0,
                "min_samples": 10,
            },
            "filter": {
                "type": "lowpass",
                "cutoff_hz": 0.1,
                "num_taps": 7,
                "window": "hamming",
            },
            "stf": {
                "path": str(stf_dir),
                "start_sec": 0.0,
                "duration_sec": 200.0,
                "min_retained_moment_fraction": 0.995,
                "preserve_integral": True,
                "m_ref": 1.0e18,
            },
        },
        "physics": {
            "rho": 3400.0,
            "alpha": 7900.0,
            "beta": 4533.0,
            "distance_mode": "hypocentral",
            "delay_mode": "absolute",
            "amplitude_gain": 1.0,
        },
        "paths": {"data_path": str(npz_path)},
        "training": {"rate_representation": "log1p"},
        "evaluation": {
            "primary_reference": "catalog",
            "aggregation": "event_median",
        },
    }


def make_station_window_config(
    *,
    npz_path: Path,
    stf_dir: Path,
) -> dict[str, Any]:
    config = make_v2_dataset_config(
        npz_path=npz_path,
        stf_dir=stf_dir,
    )
    config["workflow"] = "station_random_shifted_stf"
    config["dataset"]["stf"].update(
        {
            "station_window_duration_sec": 300.0,
            "station_alignment": "p_arrival",
            "station_preserve_integral": False,
            "magnitude_target": "stf_native",
        }
    )
    config["physics"].update(
        {
            "travel_time_model": "constant_velocity",
            "delay_mode": "p_aligned_relative",
        }
    )
    config["model"] = {
        "predict_catalog_mw": True,
        "catalog_mw_initial_bias": 8.0,
    }
    config["training"].update(
        {
            "split_protocol": "within_event_station",
            "event_balanced_sampling": False,
            "early_stop_patience": 0,
            "checkpoint_metric": "station_mae_catalog",
        }
    )
    config["evaluation"]["external_role"] = "sanity"
    return config


def _relative_time() -> np.ndarray:
    return np.arange(-20.0, 210.0, dtype=np.float64)


def _waveform_payload(
    *,
    name: str,
    latitude: float,
    longitude: float,
    absolute_origin_sec: float | None = None,
) -> dict[str, Any]:
    relative_time = _relative_time()
    time_sec = relative_time.copy()
    payload: dict[str, Any] = {
        "name": name,
        "lat": latitude,
        "lon": longitude,
        "t": time_sec,
        "E": np.zeros_like(relative_time),
        "N": 0.1 * np.sin(relative_time / 20.0),
        "U": 0.02 * np.cos(relative_time / 25.0),
    }
    if absolute_origin_sec is not None:
        payload["t"] = time_sec + absolute_origin_sec
        payload["origin"] = absolute_origin_sec
    return payload


def _write_stations_npz(
    path: Path,
    *,
    event_name: str = "Event A",
    station_count: int = 2,
    use_absolute_time: bool = False,
) -> None:
    station_records = []
    for index in range(station_count):
        station_records.append(
            _waveform_payload(
                name=f"ST{index}",
                latitude=35.1 + 1.9 * index,
                longitude=140.0,
                absolute_origin_sec=1000.0 if use_absolute_time else None,
            )
        )
    stations = np.empty(1, dtype=object)
    stations[0] = station_records
    np.savez(
        path,
        events=np.array([event_name], dtype=object),
        magnitude=np.array([7.0]),
        latitude=np.array([35.0]),
        longitude=np.array([140.0]),
        depth_km=np.array([20.0]),
        strike=np.array([30.0]),
        dip=np.array([45.0]),
        rake=np.array([90.0]),
        mechanism=np.array(["Reverse"], dtype=object),
        stations=stations,
    )


def _write_enu_npz(path: Path) -> None:
    payload = _waveform_payload(
        name="ENU1",
        latitude=35.2,
        longitude=140.1,
    )
    station_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"name", "lat", "lon"}
    }
    enu = np.empty(1, dtype=object)
    enu[0] = {"ENU1": station_payload}
    station_info = np.empty(1, dtype=object)
    station_info[0] = {"ENU1": {"lat": 35.2, "lon": 140.1}}
    np.savez(
        path,
        events=np.array(["Legacy Event"], dtype=object),
        magnitude=np.array([6.5]),
        latitude=np.array([35.0]),
        longitude=np.array([140.0]),
        depth_km=np.array([15.0]),
        strike=np.array([10.0]),
        dip=np.array([30.0]),
        rake=np.array([95.0]),
        mechanism=np.array([b"Reverse"], dtype=object),
        enu=enu,
        station_info=station_info,
    )


def _write_stf(
    root: Path,
    event_stem: str,
    *,
    duration_sec: int = 100,
    constant: bool = False,
) -> Path:
    root.mkdir(exist_ok=True)
    path = root / f"{event_stem}.stf"
    time_sec = np.arange(float(duration_sec))
    if constant:
        rate_nm_per_s = np.full_like(time_sec, 1.0e18)
    else:
        rate_nm_per_s = (
            np.maximum(0.0, 1.0 - np.abs(time_sec - 40.0) / 30.0)
            * 1.0e18
        )
    with path.open("w", encoding="utf-8") as stream:
        for time_value, rate_value in zip(time_sec, rate_nm_per_s):
            stream.write(f"{time_value} {rate_value}\n")
    return path


def test_same_event_has_one_stf_and_one_reference_mw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    npz_path = tmp_path / "events.npz"
    _write_stations_npz(npz_path)
    stf_dir = tmp_path / "stf"
    _write_stf(stf_dir, "eventa")
    call_count = 0
    real_resampler = dataset_v2.resample_source_stf

    def counting_resampler(*args: Any, **kwargs: Any):
        nonlocal call_count
        call_count += 1
        return real_resampler(*args, **kwargs)

    monkeypatch.setattr(dataset_v2, "resample_source_stf", counting_resampler)
    dataset = CorrectedEarthquakeDataset(
        make_v2_dataset_config(npz_path=npz_path, stf_dir=stf_dir)
    )

    assert len(dataset) == 2
    assert call_count == 1
    assert set(dataset.event_stf_cache) == {"eventa"}
    first, second = dataset.samples
    assert np.array_equal(first["stf"], second["stf"])
    assert first["stf"].shape == (200,)
    assert first["mw_stf_native"] == second["mw_stf_native"]
    assert first["source_distance_m"] != second["source_distance_m"]
    assert first["azimuth_deg"] == pytest.approx(0.0)
    assert "phi_slip_deg" in first
    assert {
        "distance",
        "dt",
        "magnitude",
        "theta",
        "phi",
        "phi_deg",
    }.isdisjoint(first)

    item = dataset[0]
    assert item["radial"].shape == (1, 200)
    assert item["vertical"].shape == (200,)
    assert item["stf"].shape == (200,)
    assert item["waveform_valid_mask"].dtype == torch.bool
    assert item["has_stf"].item() is True


def test_station_workflow_shifts_same_event_stf_by_station_p_arrival(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "events.npz"
    _write_stations_npz(npz_path)
    stf_dir = tmp_path / "stf"
    _write_stf(stf_dir, "eventa")

    dataset = CorrectedEarthquakeDataset(
        make_station_window_config(npz_path=npz_path, stf_dir=stf_dir)
    )

    first, second = dataset.samples
    assert first["stf"].shape == second["stf"].shape == (300,)
    assert not np.array_equal(first["stf"], second["stf"])
    assert first["p_arrival_sec"] != second["p_arrival_sec"]
    assert first["s_arrival_sec"] != second["s_arrival_sec"]
    assert first["full_event_moment_nm"] == pytest.approx(
        second["full_event_moment_nm"]
    )
    assert first["station_window_moment_nm"] <= first["full_event_moment_nm"]
    assert second["station_window_moment_nm"] <= second["full_event_moment_nm"]
    assert 0.0 <= first["stf_retained_moment_fraction"] <= 1.0
    assert 0.0 <= second["stf_retained_moment_fraction"] <= 1.0

    item = dataset[0]
    assert item["stf"].shape == (300,)
    assert item["p_arrival_sec"].ndim == 0
    assert item["mw_stf_window"].ndim == 0


def test_active_workflow_trains_reloads_and_evaluates_locked_station_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    npz_path = tmp_path / "events.npz"
    _write_stations_npz(npz_path, station_count=20)
    stf_dir = tmp_path / "stf"
    _write_stf(stf_dir, "eventa")
    config = make_station_window_config(
        npz_path=npz_path,
        stf_dir=stf_dir,
    )
    config["model"].update(
        {
            "hidden_dim": 8,
            "num_layers": 1,
            "num_tcn_blocks": 1,
            "transformer_num_layers": 1,
            "dropout": 0.0,
            "use_meta": True,
        }
    )
    config["paths"].update(
        {
            "logs_dir": str(tmp_path / "logs"),
            "models_dir": str(tmp_path / "models"),
            "results_dir": str(tmp_path / "results"),
        }
    )
    config["training"].update(
        {
            "loss_name": "stf_rate",
            "random_seed": 42,
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "validation_event_fraction": 0.15,
            "test_event_fraction": 0.15,
            "early_stop_metric": "mw_mae",
            "early_stop_min_delta": 0.0,
            "warmup_epochs": 0,
            "scheduler_T0": 2,
            "scheduler_T_mult": 1,
            "swa_start": 0,
            "num_workers": 0,
            "stf_rate_loss": {
                "lambda_MSE": 1.0,
                "lambda_synth": 0.5,
                "lambda_mag": 1.0,
                "lambda_shape": 0.1,
                "include_intermediate_field": False,
                "radiation_pattern_mode": "full",
            },
        }
    )
    config["evaluation"]["sample_grid_rows"] = 1
    loaders = get_data_loaders_v2(config)
    train_loader, validation_loader, test_loader, manifest = loaders
    monkeypatch.setattr(
        "src.training.train.get_preferred_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "src.evaluation.evaluate.get_preferred_device",
        lambda: torch.device("cpu"),
    )

    training = train(config=config, data_loaders=loaders)
    checkpoint = torch.load(
        training["best_model_path"],
        map_location="cpu",
    )
    reloaded = PINNModel(config)
    reloaded.load_state_dict(checkpoint, strict=True)
    result = evaluate(
        model_path=training["best_model_path"],
        config=config,
        test_loader=test_loader,
        save_plots=False,
        save_metrics=False,
    )

    assert len(train_loader.dataset) == 14
    assert len(validation_loader.dataset) == 3
    assert len(test_loader.dataset) == 3
    assert manifest["assignment_sha256"]
    assert sum(len(keys) for keys in manifest["sample_keys"].values()) == 20
    assert np.isfinite(result["mae"])
    assert np.isfinite(result["rmse"])
    assert all(np.isfinite(row["mw_pred"]) for row in result["station_rows"])
    assert all(
        np.isfinite(row["mw_window_pred"])
        for row in result["station_rows"]
    )


def test_stations_layout_converts_absolute_timestamps_to_origin_time(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "absolute.npz"
    _write_stations_npz(
        npz_path,
        station_count=1,
        use_absolute_time=True,
    )
    stf_dir = tmp_path / "stf"
    _write_stf(stf_dir, "eventa")

    dataset = CorrectedEarthquakeDataset(
        make_v2_dataset_config(npz_path=npz_path, stf_dir=stf_dir)
    )

    assert len(dataset) == 1
    assert dataset.samples[0]["valid_fraction"] == 1.0
    assert dataset.samples[0]["baseline_source"] == "pre_event"


def test_enu_station_info_layout_is_normalized_once(tmp_path: Path) -> None:
    npz_path = tmp_path / "legacy.npz"
    _write_enu_npz(npz_path)
    stf_dir = tmp_path / "stf"
    _write_stf(stf_dir, "legacy_event")

    dataset = CorrectedEarthquakeDataset(
        make_v2_dataset_config(npz_path=npz_path, stf_dir=stf_dir)
    )

    assert len(dataset) == 1
    sample = dataset.samples[0]
    assert sample["station"] == "ENU1"
    assert sample["mechanism"] == 2
    assert sample["magnitude_catalog"] == pytest.approx(6.5)
    assert sample["source_distance_m"] > sample["epicentral_distance_m"]


def test_failed_stf_processing_is_cached_once_per_normalized_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    npz_path = tmp_path / "events.npz"
    _write_stations_npz(npz_path, event_name="Event-A")
    stf_dir = tmp_path / "stf"
    _write_stf(stf_dir, "event_a", duration_sec=300, constant=True)
    call_count = 0
    real_resampler = dataset_v2.resample_source_stf

    def counting_resampler(*args: Any, **kwargs: Any):
        nonlocal call_count
        call_count += 1
        return real_resampler(*args, **kwargs)

    monkeypatch.setattr(dataset_v2, "resample_source_stf", counting_resampler)
    dataset = CorrectedEarthquakeDataset(
        make_v2_dataset_config(npz_path=npz_path, stf_dir=stf_dir)
    )

    assert len(dataset) == 0
    assert call_count == 1
    assert set(dataset.event_stf_cache) == {"eventa"}
    assert [item["reason"] for item in dataset.rejections] == [
        "stf_window_too_short",
        "stf_window_too_short",
    ]


def test_allow_missing_stf_emits_explicit_empty_target(tmp_path: Path) -> None:
    npz_path = tmp_path / "events.npz"
    _write_stations_npz(npz_path, station_count=1)
    stf_dir = tmp_path / "empty_stf"
    stf_dir.mkdir()

    dataset = CorrectedEarthquakeDataset(
        make_v2_dataset_config(
            npz_path=npz_path,
            stf_dir=stf_dir,
            allow_missing_stf=True,
        )
    )

    assert len(dataset) == 1
    sample = dataset.samples[0]
    assert sample["has_stf"] is False
    assert np.array_equal(sample["stf"], np.zeros(200))
    assert np.isnan(sample["mw_stf_native"])
    assert sample["stf_path"] == ""


def test_duplicate_normalized_stf_names_are_rejected(tmp_path: Path) -> None:
    stf_dir = tmp_path / "stf"
    _write_stf(stf_dir, "Event-A")
    _write_stf(stf_dir, "Event_A")

    with pytest.raises(ValueError, match="duplicate normalized STF key"):
        _load_stf_files(stf_dir)
