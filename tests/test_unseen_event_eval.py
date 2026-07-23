import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.external_records import record_from_external_bundle
from src.data.records_v2 import _iter_normalized_station_records
from src.data.sample_builder import build_station_sample
from src.data.waveform import waveform_config_from_v2
from src.evaluation.evaluate_unseen import (
    EventBundle,
    StationWaveform,
    _format_event_display_name,
    _station_sample_from_bundle,
    evaluate_unseen_events,
    load_event_bundle,
    plot_event_station_waveforms,
    plot_unseen_event_mw_figure,
    plot_unseen_station_panels,
    summarize_event_predictions,
    write_unseen_event_outputs,
)
from src.models.model import PINNPrediction


def _write_event_dir(base_dir: Path, event_name: str = "xizang-test") -> Path:
    event_dir = base_dir / event_name
    event_dir.mkdir()

    (event_dir / "event.json").write_text(
        json.dumps(
            {
                "event": "Test Event",
                "latitude": 28.6,
                "longitude": 87.3,
                "depth_km": 10.0,
                "magnitude": 7.1,
                "mechanism": "normal",
                "strike": 123.0,
                "dip": 45.0,
                "rake": -90.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (event_dir / "stations.csv").write_text(
        "Station,Latitude,Longitude,Sampling_Hz,Waveform_Rows\n"
        "STA1,29.0,87.8,1,4\n",
        encoding="utf-8",
    )

    rows = [
        ["STA1", "2025-01-07T01:05:16Z", -1.0, "E", 0.001, 1, "src.txt"],
        ["STA1", "2025-01-07T01:05:16Z", -1.0, "N", 0.002, 1, "src.txt"],
        ["STA1", "2025-01-07T01:05:16Z", -1.0, "U", 0.003, 1, "src.txt"],
        ["STA1", "2025-01-07T01:05:17Z", 0.0, "E", 0.011, 1, "src.txt"],
        ["STA1", "2025-01-07T01:05:17Z", 0.0, "N", 0.012, 1, "src.txt"],
        ["STA1", "2025-01-07T01:05:17Z", 0.0, "U", 0.013, 1, "src.txt"],
        ["STA1", "2025-01-07T01:05:18Z", 1.0, "E", 0.021, 1, "src.txt"],
        ["STA1", "2025-01-07T01:05:18Z", 1.0, "N", 0.022, 1, "src.txt"],
        ["STA1", "2025-01-07T01:05:18Z", 1.0, "U", 0.023, 1, "src.txt"],
        ["STA1", "2025-01-07T01:05:19Z", 2.0, "E", 0.031, 1, "src.txt"],
        ["STA1", "2025-01-07T01:05:19Z", 2.0, "N", 0.032, 1, "src.txt"],
        ["STA1", "2025-01-07T01:05:19Z", 2.0, "U", 0.033, 1, "src.txt"],
    ]
    with gzip.open(event_dir / "waveforms.csv.gz", "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Station", "Time_UTC", "Time_Offset_s", "Component", "Value_m", "Sampling_Hz", "Source_File"])
        writer.writerows(rows)
    return event_dir


def _external_v2_config(*, radial_peak_min_cm: float = 0.0) -> dict:
    return {
        "dataset": {
            "sample_rate_hz": 1.0,
            "radial_peak_min_cm": radial_peak_min_cm,
            "waveform": {
                "start_sec": 0.0,
                "duration_sec": 3.0,
                "min_valid_fraction": 1.0,
                "max_interpolation_gap_sec": 2.5,
            },
            "baseline": {
                "method": "median",
                "pre_event_start_sec": -1.0,
                "pre_event_end_sec": 0.0,
                "fallback": "pre_p",
                "fallback_max_sec": 30.0,
                "min_samples": 1,
            },
            "filter": {
                "type": "none",
                "cutoff_hz": 0.1,
                "num_taps": 7,
                "window": "hamming",
            },
        },
        "physics": {"alpha": 7900.0},
    }


def test_load_event_bundle_reads_station_waveforms(tmp_path: Path):
    """验证：外部事件目录可被解析为模型评估所需的台站三分量波形"""
    event_dir = _write_event_dir(tmp_path)

    bundle = load_event_bundle(event_dir)

    assert bundle.event_name == "Test Event"
    assert bundle.magnitude == pytest.approx(7.1)
    assert bundle.mechanism == "normal"
    assert len(bundle.stations) == 1
    assert bundle.strike == pytest.approx(123.0)
    assert bundle.dip == pytest.approx(45.0)
    assert bundle.rake == pytest.approx(-90.0)
    station = bundle.stations[0]
    assert station.station == "STA1"
    assert station.dt == pytest.approx(1.0)
    assert station.e_m.tolist() == pytest.approx([0.001, 0.011, 0.021, 0.031])
    assert station.n_m.tolist() == pytest.approx([0.002, 0.012, 0.022, 0.032])
    assert station.u_m.tolist() == pytest.approx([0.003, 0.013, 0.023, 0.033])


def test_load_event_bundle_allows_missing_depth_km(tmp_path: Path):
    """验证：event.json 中 depth_km 为 null 时，未见事件读取会回退到 0.0 而不是报错"""
    event_dir = _write_event_dir(tmp_path, event_name="missing-depth")
    event_meta_path = event_dir / "event.json"
    meta = json.loads(event_meta_path.read_text(encoding="utf-8"))
    meta["depth_km"] = None
    event_meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    bundle = load_event_bundle(event_dir)

    assert bundle.depth_km == pytest.approx(0.0)


def test_load_event_bundle_allows_missing_u_component(tmp_path: Path):
    """验证：只有 E/N 两分量时，未见事件读取仍保留台站，并用零值补齐 U 分量"""
    event_dir = _write_event_dir(tmp_path, event_name="missing-u")
    waveform_path = event_dir / "waveforms.csv.gz"
    rows = []
    with gzip.open(waveform_path, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["Component"] != "U":
                rows.append(row)
    with gzip.open(waveform_path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Station", "Time_UTC", "Time_Offset_s", "Component", "Value_m", "Sampling_Hz", "Source_File"])
        writer.writeheader()
        writer.writerows(rows)

    bundle = load_event_bundle(event_dir)

    assert len(bundle.stations) == 1
    assert bundle.stations[0].u_m.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_load_event_bundle_preserves_iquique_canonical_value_m(tmp_path: Path):
    """Canonical Value_m data must not be rescaled from an event-name guess."""
    event_dir = _write_event_dir(tmp_path, event_name="iquique-aftershock-2014")
    event_meta_path = event_dir / "event.json"
    meta = json.loads(event_meta_path.read_text(encoding="utf-8"))
    meta["event"] = "Iquique Aftershock 2014"
    meta["magnitude"] = 7.7
    event_meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    bundle = load_event_bundle(event_dir)

    assert bundle.stations[0].e_m.tolist() == pytest.approx([0.001, 0.011, 0.021, 0.031])
    assert bundle.stations[0].n_m.tolist() == pytest.approx([0.002, 0.012, 0.022, 0.032])
    assert bundle.stations[0].u_m.tolist() == pytest.approx([0.003, 0.013, 0.023, 0.033])


def test_load_event_bundle_preserves_nepal_canonical_value_m(tmp_path: Path):
    """Canonical Value_m data must remain meters for Nepal aftershock too."""
    event_dir = _write_event_dir(tmp_path, event_name="nepal-aftershock-2015")
    event_meta_path = event_dir / "event.json"
    meta = json.loads(event_meta_path.read_text(encoding="utf-8"))
    meta["event"] = "M 7.3 - Nepal"
    meta["magnitude"] = 7.3
    event_meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    bundle = load_event_bundle(event_dir)

    assert bundle.stations[0].e_m.tolist() == pytest.approx([0.001, 0.011, 0.021, 0.031])
    assert bundle.stations[0].n_m.tolist() == pytest.approx([0.002, 0.012, 0.022, 0.032])
    assert bundle.stations[0].u_m.tolist() == pytest.approx([0.003, 0.013, 0.023, 0.033])


def test_format_event_display_name_uses_place_year_magnitude_style():
    """验证：未见事件显示名按“地点 年份 Mx.x”格式缩短，避免结果表里的标题过长"""
    label = _format_event_display_name(
        event_name="M 7.7 - 53 km SW of Iquique, Chile",
        event_dir_name="iquique-aftershock-2014-chile",
        magnitude=7.7,
    )

    assert label == "Iquique 2014 M7.7"


def test_summarize_event_predictions_returns_median_and_spread():
    """验证：事件级汇总使用中位数，并返回离散度统计量"""
    summary = summarize_event_predictions(
        event_name="Xizang",
        mw_catalog=7.1,
        predictions=[7.0, 7.2, 7.4],
    )

    assert summary["event"] == "Xizang"
    assert summary["mw_catalog"] == pytest.approx(7.1)
    assert summary["mw_pred_median"] == pytest.approx(7.2)
    assert summary["error_vs_catalog"] == pytest.approx(0.1)
    assert summary["n_stations"] == 3
    assert summary["pred_iqr"] == pytest.approx(0.2)


def test_station_sample_exposes_max_radial_cm(tmp_path: Path):
    """验证：单台站评估样本会暴露最大径向位移，供结果表格输出"""
    event_dir = _write_event_dir(tmp_path)
    bundle = load_event_bundle(event_dir)

    sample = _station_sample_from_bundle(
        bundle,
        bundle.stations[0],
        _external_v2_config(),
    )

    assert sample is not None
    assert "radial_peak_cm" in sample
    assert float(sample["radial_peak_cm"]) > 0.0


def test_active_unseen_evaluation_uses_catalog_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = yaml.safe_load(
        Path("configs/config_v2.yaml").read_text(encoding="utf-8")
    )
    config["model"]["input_components"] = ["radial", "tangential"]
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    (model_dir / "best_model.pth").write_bytes(b"checkpoint")

    received_waveforms: list[torch.Tensor] = []

    class ScalarHeadModel:
        def __init__(self, _config):
            pass

        def to(self, _device):
            return self

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return self

        def predict_heads(self, waveform, meta=None):
            del meta
            received_waveforms.append(waveform.detach().cpu())
            return PINNPrediction(
                stf_encoded=torch.zeros(waveform.shape[0], 300),
                catalog_mw=torch.full(
                    (waveform.shape[0],),
                    7.3,
                    device=waveform.device,
                ),
            )

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("active unseen evaluation must use predict_heads")

    bundle = EventBundle(
        event_name="External Event",
        magnitude=7.1,
        latitude=0.0,
        longitude=0.0,
        depth_km=10.0,
        mechanism="normal",
        stations=[
            StationWaveform(
                station="STA",
                latitude=0.1,
                longitude=0.1,
                t=np.arange(200.0),
                e_m=np.linspace(0.001, 0.01, 200),
                n_m=np.zeros(200),
                u_m=np.zeros(200),
                dt=1.0,
            )
        ],
    )
    sample = {
        "radial": np.zeros(200, dtype=np.float32),
        "tangential": np.full(200, 2.0, dtype=np.float32),
        "source_distance_m": 20_000.0,
        "epicentral_distance_m": 17_000.0,
        "theta_deg": 30.0,
        "azimuth_deg": 45.0,
        "waveform_dt_sec": 1.0,
        "radial_peak_cm": 3.0,
    }
    monkeypatch.setattr(
        "src.evaluation.evaluate_unseen.get_preferred_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr("src.evaluation.evaluate_unseen.torch.load", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("src.evaluation.evaluate_unseen.PINNModel", ScalarHeadModel)
    monkeypatch.setattr("src.evaluation.evaluate_unseen.load_event_bundle", lambda _path: bundle)
    monkeypatch.setattr(
        "src.evaluation.evaluate_unseen._station_sample_from_bundle",
        lambda *_args, **_kwargs: sample,
    )

    result = evaluate_unseen_events(
        event_dirs=[tmp_path / "event"],
        model_dir=model_dir,
        output_dir=tmp_path / "output",
        save_plots=False,
    )

    row = result["station_rows"][0]
    assert row["mw_pred"] == pytest.approx(7.3)
    assert row["mw_window_pred"] == pytest.approx(0.6)
    assert received_waveforms[0].shape == (1, 2, 200)
    torch.testing.assert_close(
        received_waveforms[0][0, 0],
        torch.zeros(200),
    )
    torch.testing.assert_close(
        received_waveforms[0][0, 1],
        torch.full((200,), 2.0),
    )


def test_external_adapter_matches_training_npz_preprocessing(tmp_path: Path):
    time_sec = np.arange(-2.0, 6.0)
    east = 0.01 * time_sec**2
    north = 0.02 * time_sec
    vertical = -0.005 * time_sec
    station_payload = {
        "name": "STA",
        "lat": 35.2,
        "lon": 140.1,
        "t": time_sec,
        "E": east,
        "N": north,
        "U": vertical,
    }
    stations = np.empty(1, dtype=object)
    stations[0] = [station_payload]
    npz_path = tmp_path / "training.npz"
    np.savez(
        npz_path,
        events=np.array(["Event A"], dtype=object),
        magnitude=np.array([7.0]),
        latitude=np.array([35.0]),
        longitude=np.array([140.0]),
        depth_km=np.array([20.0]),
        strike=np.array([30.0]),
        dip=np.array([45.0]),
        rake=np.array([90.0]),
        mechanism=np.array(["reverse"], dtype=object),
        stations=stations,
    )
    config = _external_v2_config()
    config["dataset"]["waveform"].update(
        {"duration_sec": 5.0, "min_valid_fraction": 1.0}
    )
    config["dataset"]["baseline"].update(
        {
            "pre_event_start_sec": -2.0,
            "pre_event_end_sec": 0.0,
            "min_samples": 2,
        }
    )
    waveform_config = waveform_config_from_v2(config)

    with np.load(npz_path, allow_pickle=True) as data:
        training_record = next(_iter_normalized_station_records(data))
    training_sample = build_station_sample(
        training_record,
        units="m",
        waveform_config=waveform_config,
        alpha_m_per_s=7900.0,
        radial_peak_min_cm=0.0,
    )
    bundle = EventBundle(
        event_name="Event A",
        magnitude=7.0,
        latitude=35.0,
        longitude=140.0,
        depth_km=20.0,
        mechanism="reverse",
        strike=30.0,
        dip=45.0,
        rake=90.0,
        stations=[],
    )
    station = StationWaveform(
        station="STA",
        latitude=35.2,
        longitude=140.1,
        t=time_sec,
        e_m=east,
        n_m=north,
        u_m=vertical,
        dt=1.0,
    )
    external_record = record_from_external_bundle(bundle, station)
    external_sample = build_station_sample(
        external_record,
        units="m",
        waveform_config=waveform_config,
        alpha_m_per_s=7900.0,
        radial_peak_min_cm=0.0,
    )

    assert np.allclose(training_sample["radial"], external_sample["radial"])
    assert np.allclose(
        training_sample["tangential"],
        external_sample["tangential"],
    )
    assert np.array_equal(
        training_sample["waveform_valid_mask"],
        external_sample["waveform_valid_mask"],
    )
    for key in ("source_distance_m", "theta_deg", "azimuth_deg"):
        assert training_sample[key] == pytest.approx(external_sample[key])


def test_plot_event_station_waveforms_creates_waveform_figure(tmp_path: Path):
    """验证：可为单个事件生成按台站堆叠的波形图"""
    bundle = EventBundle(
        event_name="EventA",
        magnitude=7.7,
        latitude=0.0,
        longitude=0.0,
        depth_km=10.0,
        mechanism="reverse",
        stations=[
            StationWaveform(
                station="STA1",
                latitude=0.0,
                longitude=0.0,
                t=np.array([0.0, 1.0, 2.0], dtype=float),
                e_m=np.array([0.0, 0.1, 0.2], dtype=float),
                n_m=np.array([0.0, 0.2, 0.1], dtype=float),
                u_m=np.array([0.0, 0.05, 0.1], dtype=float),
                dt=1.0,
            ),
            StationWaveform(
                station="STA2",
                latitude=0.0,
                longitude=0.0,
                t=np.array([0.0, 1.0, 2.0], dtype=float),
                e_m=np.array([0.0, -0.1, -0.2], dtype=float),
                n_m=np.array([0.0, -0.2, -0.1], dtype=float),
                u_m=np.array([0.0, -0.05, -0.1], dtype=float),
                dt=1.0,
            ),
        ],
    )

    fig_path = plot_event_station_waveforms(
        bundle=bundle,
        save_path=tmp_path / "event_waveforms.png",
    )

    assert fig_path.exists()


def test_plot_unseen_station_panels_creates_panel_figure(tmp_path: Path):
    """验证：单个事件会生成按最大径向位移排序的三列面板图"""
    panel_rows = [
        {
            "event": "EventA",
            "station": "STA_LOW",
            "mw_catalog": 7.1,
            "mw_pred": 7.0,
            "source_distance_km": 120.0,
            "max_radial_cm": 1.5,
            "radial": np.array([0.0, 0.01, 0.02], dtype=float),
            "pred_rate": np.array([0.0, 1.0e18, 2.0e18], dtype=float),
            "mw_series": np.array([0.0, 6.5, 7.0], dtype=float),
            "time_axis": np.array([0.0, 1.0, 2.0], dtype=float),
        },
        {
            "event": "EventA",
            "station": "STA_HIGH",
            "mw_catalog": 7.1,
            "mw_pred": 7.2,
            "source_distance_km": 80.0,
            "max_radial_cm": 4.5,
            "radial": np.array([0.0, 0.03, 0.01], dtype=float),
            "pred_rate": np.array([0.0, 2.0e18, 1.0e18], dtype=float),
            "mw_series": np.array([0.0, 6.8, 7.2], dtype=float),
            "time_axis": np.array([0.0, 1.0, 2.0], dtype=float),
        },
    ]

    panel_path = plot_unseen_station_panels(
        panel_rows=panel_rows,
        save_path=tmp_path / "event_a_panels.png",
        top_n=1,
        sort_by="max_radial_cm",
    )

    assert panel_path.exists()


def test_plot_unseen_station_panels_uses_independent_time_axes(tmp_path: Path):
    """Waveform and source outputs may have different configured lengths."""
    panel_path = plot_unseen_station_panels(
        panel_rows=[
            {
                "event": "EventA",
                "station": "STA",
                "mw_catalog": 7.1,
                "mw_pred": 7.0,
                "source_distance_km": 120.0,
                "max_radial_cm": 1.5,
                "radial": np.array([0.0, 0.01], dtype=float),
                "pred_rate": np.array([0.0, 1.0e18, 2.0e18], dtype=float),
                "mw_series": np.array([0.0, 6.5, 7.0], dtype=float),
                "waveform_time_axis": np.array([0.0, 1.0], dtype=float),
                "source_time_axis": np.array([0.0, 1.0, 2.0], dtype=float),
                "time_axis": np.array([0.0, 1.0, 2.0], dtype=float),
            }
        ],
        save_path=tmp_path / "independent_axes.png",
        top_n=1,
    )

    assert panel_path.exists()


def test_plot_unseen_event_mw_figure_creates_single_event_summary_figure(tmp_path: Path):
    """验证：单个事件可生成叠加各台站 Mw(t) 的汇总风格图"""
    panel_rows = [
        {
            "event": "EventA",
            "station": "STA1",
            "mw_catalog": 7.1,
            "mw_pred": 7.0,
            "source_distance_km": 120.0,
            "max_radial_cm": 3.2,
            "mw_series": np.array([0.0, 6.0, 6.8, 7.0], dtype=float),
            "time_axis": np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        },
        {
            "event": "EventA",
            "station": "STA2",
            "mw_catalog": 7.1,
            "mw_pred": 7.2,
            "source_distance_km": 90.0,
            "max_radial_cm": 4.1,
            "mw_series": np.array([0.0, 6.2, 7.0, 7.2], dtype=float),
            "time_axis": np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        },
    ]

    fig_path = plot_unseen_event_mw_figure(
        panel_rows=panel_rows,
        event_name="EventA",
        save_path=tmp_path / "event_a_mw_summary.png",
    )

    assert fig_path.exists()


def test_plot_unseen_event_mw_figure_uses_true_magnitude_centered_ylim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """验证：事件 Mw 汇总图纵轴按真实震级动态设置为 [Mw_true-2, Mw_true+0.5]"""
    panel_rows = [
        {
            "event": "EventA",
            "station": "STA1",
            "mw_catalog": 7.1,
            "mw_pred": 7.0,
            "source_distance_km": 120.0,
            "max_radial_cm": 3.2,
            "mw_series": np.array([0.0, 6.0, 6.8, 7.0], dtype=float),
            "time_axis": np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        }
    ]
    captured = {}

    import matplotlib.axes

    original_set_ylim = matplotlib.axes.Axes.set_ylim

    def spy_set_ylim(self, bottom=None, top=None, *args, **kwargs):
        captured["bottom"] = bottom
        captured["top"] = top
        return original_set_ylim(self, bottom=bottom, top=top, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylim", spy_set_ylim)

    fig_path = plot_unseen_event_mw_figure(
        panel_rows=panel_rows,
        event_name="EventA",
        save_path=tmp_path / "event_a_mw_summary_ylim.png",
    )

    assert fig_path.exists()
    assert captured["bottom"] == pytest.approx(5.1)
    assert captured["top"] == pytest.approx(7.6)


def test_plot_unseen_event_mw_figure_draws_true_magnitude_pm_03_guides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """验证：事件 Mw 汇总图会绘制真实震级上下 0.3 的虚线参考线"""
    panel_rows = [
        {
            "event": "EventA",
            "station": "STA1",
            "mw_catalog": 7.1,
            "mw_pred": 7.0,
            "source_distance_km": 120.0,
            "max_radial_cm": 3.2,
            "mw_series": np.array([0.0, 6.0, 6.8, 7.0], dtype=float),
            "time_axis": np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        }
    ]
    captured = []

    import matplotlib.axes

    original_axhline = matplotlib.axes.Axes.axhline

    def spy_axhline(self, y=0, *args, **kwargs):
        captured.append(float(y))
        return original_axhline(self, y=y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "axhline", spy_axhline)

    fig_path = plot_unseen_event_mw_figure(
        panel_rows=panel_rows,
        event_name="EventA",
        save_path=tmp_path / "event_a_mw_summary_guides.png",
    )

    assert fig_path.exists()
    assert captured == pytest.approx([7.1, 7.4, 6.8])


def test_write_unseen_event_outputs_creates_one_panel_per_event(tmp_path: Path):
    """验证：不同事件应各自生成各自的 panel 图文件"""
    station_rows = [
        {
            "event": "EventA",
            "station": "STA1",
            "mw_catalog": 7.1,
            "mw_pred": 7.0,
            "error_vs_catalog": -0.1,
            "source_distance_km": 120.0,
            "mechanism": "normal",
            "dt": 1.0,
            "max_radial_cm": 3.2,
            "used_in_event_summary": True,
        }
    ]
    event_rows = [
        {
            "event": "EventA",
            "mw_catalog": 7.1,
            "mw_pred_median": 7.0,
            "error_vs_catalog": -0.1,
            "n_stations": 1,
            "pred_std": 0.0,
            "pred_iqr": 0.0,
        },
        {
            "event": "EventB",
            "mw_catalog": 7.7,
            "mw_pred_median": 7.8,
            "error_vs_catalog": 0.1,
            "n_stations": 1,
            "pred_std": 0.0,
            "pred_iqr": 0.0,
        },
    ]
    panel_rows = [
        {
            "event": "EventA",
            "station": "STA1",
            "mw_catalog": 7.1,
            "mw_pred": 7.0,
            "source_distance_km": 120.0,
            "max_radial_cm": 3.2,
            "radial": np.array([0.0, 0.01, 0.02], dtype=float),
            "pred_rate": np.array([0.0, 1.0e18, 2.0e18], dtype=float),
            "mw_series": np.array([0.0, 6.5, 7.0], dtype=float),
            "time_axis": np.array([0.0, 1.0, 2.0], dtype=float),
        },
        {
            "event": "EventB",
            "station": "STA2",
            "mw_catalog": 7.7,
            "mw_pred": 7.8,
            "source_distance_km": 80.0,
            "max_radial_cm": 4.5,
            "radial": np.array([0.0, 0.03, 0.01], dtype=float),
            "pred_rate": np.array([0.0, 2.0e18, 1.0e18], dtype=float),
            "mw_series": np.array([0.0, 7.0, 7.8], dtype=float),
            "time_axis": np.array([0.0, 1.0, 2.0], dtype=float),
        },
    ]

    outputs = write_unseen_event_outputs(
        output_dir=tmp_path,
        station_rows=station_rows,
        event_rows=event_rows,
        panel_rows=panel_rows,
    )

    assert len(outputs["station_panels"]["EventA"]) == 1
    assert len(outputs["station_panels"]["EventB"]) == 1
    assert outputs["station_panels"]["EventA"][0].exists()
    assert outputs["station_panels"]["EventB"][0].exists()


def test_write_unseen_event_outputs_splits_station_panels_into_three_pages(tmp_path: Path):
    """验证：单个事件台站过多时，panel 图按每张 4 个台站分页，最多输出 3 张"""
    event_name = "EventA"
    panel_rows = []
    for idx in range(13):
        panel_rows.append(
            {
                "event": event_name,
                "station": f"STA{idx:02d}",
                "mw_catalog": 7.1,
                "mw_pred": 7.0 + idx * 0.01,
                "source_distance_km": 80.0 + idx,
                "max_radial_cm": 20.0 - idx,
                "radial": np.array([0.0, 0.01, 0.02], dtype=float),
                "pred_rate": np.array([0.0, 1.0e18, 2.0e18], dtype=float),
                "mw_series": np.array([0.0, 6.5, 7.0], dtype=float),
                "time_axis": np.array([0.0, 1.0, 2.0], dtype=float),
            }
        )

    outputs = write_unseen_event_outputs(
        output_dir=tmp_path,
        station_rows=[],
        event_rows=[
            {
                "event": event_name,
                "mw_catalog": 7.1,
                "mw_pred_median": 7.0,
                "error_vs_catalog": -0.1,
                "n_stations": 13,
                "pred_std": 0.0,
                "pred_iqr": 0.0,
            }
        ],
        panel_rows=panel_rows,
    )

    pages = outputs["station_panels"][event_name]
    assert len(pages) == 3
    assert pages[0].name.endswith("_1.png")
    assert pages[1].name.endswith("_2.png")
    assert pages[2].name.endswith("_3.png")
    assert all(path.exists() for path in pages)


def test_write_unseen_event_outputs_creates_event_mw_figures(tmp_path: Path):
    """验证：未见地震评估结果会为每个事件各自写出 Mw(t) 汇总风格图"""
    outputs = write_unseen_event_outputs(
        output_dir=tmp_path,
        station_rows=[],
        event_rows=[
            {
                "event": "EventA",
                "mw_catalog": 7.1,
                "mw_pred_median": 7.0,
                "error_vs_catalog": -0.1,
                "n_stations": 2,
                "pred_std": 0.1,
                "pred_iqr": 0.2,
            }
        ],
        panel_rows=[
            {
                "event": "EventA",
                "station": "STA1",
                "mw_catalog": 7.1,
                "mw_pred": 7.0,
                "source_distance_km": 120.0,
                "max_radial_cm": 3.2,
                "radial": np.array([0.0, 0.01, 0.02, 0.03], dtype=float),
                "pred_rate": np.array([0.0, 1.0e18, 2.0e18, 3.0e18], dtype=float),
                "mw_series": np.array([0.0, 6.0, 6.8, 7.0], dtype=float),
                "time_axis": np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
            },
            {
                "event": "EventA",
                "station": "STA2",
                "mw_catalog": 7.1,
                "mw_pred": 7.2,
                "source_distance_km": 90.0,
                "max_radial_cm": 4.1,
                "radial": np.array([0.0, 0.02, 0.03, 0.02], dtype=float),
                "pred_rate": np.array([0.0, 1.2e18, 2.2e18, 2.8e18], dtype=float),
                "mw_series": np.array([0.0, 6.2, 7.0, 7.2], dtype=float),
                "time_axis": np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
            },
        ],
    )

    assert outputs["event_mw_figures"]["EventA"].exists()


def test_write_unseen_event_outputs_creates_csv_and_figures(tmp_path: Path):
    """验证：未见地震评估结果会写出台站级、事件级 CSV 与图件"""
    station_rows = [
        {
            "event": "Xizang",
            "station": "STA1",
            "mw_catalog": 7.1,
            "mw_pred": 7.0,
            "error_vs_catalog": -0.1,
            "source_distance_km": 120.0,
            "mechanism": "normal",
            "dt": 1.0,
            "max_radial_cm": 3.2,
            "station_lat": 29.0,
            "station_lon": 87.8,
            "used_in_event_summary": True,
        },
        {
            "event": "Myanmar",
            "station": "STA2",
            "mw_catalog": 7.7,
            "mw_pred": 7.9,
            "error_vs_catalog": 0.2,
            "source_distance_km": 160.0,
            "mechanism": "strike-slip",
            "dt": 1.0,
            "max_radial_cm": 4.6,
            "station_lat": 21.9,
            "station_lon": 96.1,
            "used_in_event_summary": True,
        },
    ]
    event_rows = [
        {
            "event": "Xizang",
            "mw_catalog": 7.1,
            "mw_pred_median": 7.0,
            "error_vs_catalog": -0.1,
            "n_stations": 1,
            "pred_std": 0.0,
            "pred_iqr": 0.0,
            "event_lat": 28.6,
            "event_lon": 87.3,
            "strike": 123.0,
            "dip": 45.0,
            "rake": -90.0,
        },
        {
            "event": "Myanmar",
            "mw_catalog": 7.7,
            "mw_pred_median": 7.9,
            "error_vs_catalog": 0.2,
            "n_stations": 1,
            "pred_std": 0.0,
            "pred_iqr": 0.0,
            "event_lat": 21.8,
            "event_lon": 95.9,
            "strike": 110.0,
            "dip": 50.0,
            "rake": 180.0,
        },
    ]

    outputs = write_unseen_event_outputs(
        output_dir=tmp_path,
        station_rows=station_rows,
        event_rows=event_rows,
    )

    assert outputs["station_csv"].exists()
    assert outputs["event_csv"].exists()
    assert outputs["station_scatter"].exists()
    assert outputs["event_summary_figure"].exists()

    station_csv_text = outputs["station_csv"].read_text(encoding="utf-8")
    event_csv_text = outputs["event_csv"].read_text(encoding="utf-8")
    assert "max_radial_cm" in station_csv_text
    assert "station_lat" in station_csv_text
    assert "station_lon" in station_csv_text
    assert "event_lat" in event_csv_text
    assert "event_lon" in event_csv_text
    assert "strike" in event_csv_text
    assert "dip" in event_csv_text
    assert "rake" in event_csv_text


def test_write_unseen_event_outputs_preserves_station_pgd_columns(tmp_path: Path):
    """验证：台站级 CSV 会保留 PGD 三法结果列，不因首行字段较少而丢失"""
    outputs = write_unseen_event_outputs(
        output_dir=tmp_path,
        station_rows=[
            {
                "event": "Xizang",
                "station": "STA1",
                "mw_catalog": 7.1,
                "mw_pred": 7.0,
                "error_vs_catalog": -0.1,
                "source_distance_km": 120.0,
                "mechanism": "normal",
                "dt": 1.0,
                "max_radial_cm": 3.2,
                "used_in_event_summary": True,
            },
            {
                "event": "Xizang",
                "station": "STA2",
                "mw_catalog": 7.1,
                "mw_pred": 7.2,
                "error_vs_catalog": 0.1,
                "source_distance_km": 130.0,
                "mechanism": "normal",
                "dt": 1.0,
                "max_radial_cm": 4.1,
                "station_lat": 29.1,
                "station_lon": 87.9,
                "used_in_event_summary": True,
                "pgd_epicentral_distance_km": 130.4,
                "pgd_3d_m": 0.052,
                "pgd_mw_crowell": 7.3,
                "pgd_mw_ruhl": 7.1,
                "pgd_mw_melgar": 7.2,
                "pgd_error_crowell": 0.2,
                "pgd_error_ruhl": 0.0,
                "pgd_error_melgar": 0.1,
            },
        ],
        event_rows=[],
    )

    with outputs["station_csv"].open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[1]["station_lat"] == "29.1"
    assert rows[1]["station_lon"] == "87.9"
    assert rows[1]["pgd_epicentral_distance_km"] == "130.4"
    assert rows[1]["pgd_3d_m"] == "0.052"
    assert rows[1]["pgd_mw_crowell"] == "7.3"
    assert rows[1]["pgd_mw_ruhl"] == "7.1"
    assert rows[1]["pgd_mw_melgar"] == "7.2"
    assert rows[1]["pgd_error_crowell"] == "0.2"
    assert rows[1]["pgd_error_ruhl"] == "0.0"
    assert rows[1]["pgd_error_melgar"] == "0.1"


def test_write_unseen_event_outputs_preserves_event_pgd_columns(tmp_path: Path):
    """验证：事件级 CSV 会保留三种 PGD 汇总列"""
    outputs = write_unseen_event_outputs(
        output_dir=tmp_path,
        station_rows=[],
        event_rows=[
            {
                "event": "Xizang",
                "mw_catalog": 7.1,
                "mw_pred_median": 7.0,
                "error_vs_catalog": -0.1,
                "n_stations": 1,
                "pred_std": 0.0,
                "pred_iqr": 0.0,
            },
            {
                "event": "Myanmar",
                "mw_catalog": 7.7,
                "mw_pred_median": 7.9,
                "error_vs_catalog": 0.2,
                "n_stations": 2,
                "pred_std": 0.1,
                "pred_iqr": 0.2,
                "event_lat": 21.8,
                "event_lon": 95.9,
                "strike": 123.0,
                "dip": 45.0,
                "rake": -90.0,
                "pgd_crowell_mw_pred_median": 7.8,
                "pgd_crowell_error": 0.1,
                "pgd_crowell_n_stations": 2,
                "pgd_crowell_pred_iqr": 0.3,
                "pgd_ruhl_mw_pred_median": 7.6,
                "pgd_ruhl_error": -0.1,
                "pgd_ruhl_n_stations": 2,
                "pgd_ruhl_pred_iqr": 0.2,
                "pgd_melgar_mw_pred_median": 7.7,
                "pgd_melgar_error": 0.0,
                "pgd_melgar_n_stations": 2,
                "pgd_melgar_pred_iqr": 0.1,
            },
        ],
    )

    with outputs["event_csv"].open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[1]["event_lat"] == "21.8"
    assert rows[1]["event_lon"] == "95.9"
    assert rows[1]["strike"] == "123.0"
    assert rows[1]["dip"] == "45.0"
    assert rows[1]["rake"] == "-90.0"
    assert rows[1]["pgd_crowell_mw_pred_median"] == "7.8"
    assert rows[1]["pgd_crowell_error"] == "0.1"
    assert rows[1]["pgd_crowell_n_stations"] == "2"
    assert rows[1]["pgd_crowell_pred_iqr"] == "0.3"
    assert rows[1]["pgd_ruhl_mw_pred_median"] == "7.6"
    assert rows[1]["pgd_ruhl_error"] == "-0.1"
    assert rows[1]["pgd_ruhl_n_stations"] == "2"
    assert rows[1]["pgd_ruhl_pred_iqr"] == "0.2"
    assert rows[1]["pgd_melgar_mw_pred_median"] == "7.7"
    assert rows[1]["pgd_melgar_error"] == "0.0"
    assert rows[1]["pgd_melgar_n_stations"] == "2"
    assert rows[1]["pgd_melgar_pred_iqr"] == "0.1"
