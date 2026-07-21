from pathlib import Path
import csv
import gzip
import json
import sys

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG1_SRC = PROJECT_ROOT / "paper" / "srl" / "figure_sources" / "fig1_event_map"
if not (FIG1_SRC / "plot_figure1_with_panel_b.py").is_file():
    pytest.skip(
        "paper figure source is not included in the lightweight package",
        allow_module_level=True,
    )
if str(FIG1_SRC) not in sys.path:
    sys.path.insert(0, str(FIG1_SRC))

from plot_figure1_with_panel_b import (  # noqa: E402
    StationEventRecord,
    build_component_images,
    collect_station_event_records,
    collect_figure1_station_event_records,
    make_figure1_with_panel_b,
    make_time_grid,
    compute_wiggle_scale,
    distance_tick_rows,
    normalize_component_rows,
    symmetric_displacement_limit,
)


def _write_npz(path: Path, enu: list[dict], station_info: list[dict], events: list[str] | None = None) -> Path:
    event_names = events or ["Synthetic2020"]
    np.savez(
        path,
        events=np.array(event_names),
        latitude=np.zeros(len(event_names), dtype=float),
        longitude=np.zeros(len(event_names), dtype=float),
        enu=np.array(enu, dtype=object),
        station_info=np.array(station_info, dtype=object),
    )
    return path


def _write_unseen_event_dir(path: Path) -> Path:
    path.mkdir()
    (path / "event.json").write_text(
        json.dumps({"event": "Unseen", "magnitude": 7.0, "latitude": 0.0, "longitude": 0.0}),
        encoding="utf-8",
    )
    with (path / "stations.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Station", "Latitude", "Longitude"])
        writer.writeheader()
        writer.writerow({"Station": "USTA", "Latitude": 0.0, "Longitude": 0.1})
    with gzip.open(path / "waveforms.csv.gz", "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Station", "Component", "Time_Offset_s", "Value_m"])
        writer.writeheader()
        for component in ["E", "N", "U"]:
            for t, value in [(0.0, 0.001), (1.0, 0.002), (2.0, 0.003)]:
                writer.writerow({"Station": "USTA", "Component": component, "Time_Offset_s": t, "Value_m": value})
    return path


def _wave(scale: float = 1.0) -> dict[str, np.ndarray]:
    t = np.array([0.0, 1.0, 2.0])
    return {
        "t": t,
        "E": scale * np.array([1.0, 2.0, 3.0]),
        "N": scale * np.array([4.0, 5.0, 6.0]),
        "U": scale * np.array([7.0, 8.0, 9.0]),
    }


def test_collect_station_event_records_sorts_by_epicentral_distance(tmp_path: Path) -> None:
    npz_path = _write_npz(
        tmp_path / "synthetic.npz",
        [{"STA_FAR": _wave(), "STA_NEAR": _wave(), "STA_MID": _wave()}],
        [
            {
                "STA_FAR": {"lat": 0.0, "lon": 2.0},
                "STA_NEAR": {"lat": 0.0, "lon": 0.1},
                "STA_MID": {"lat": 0.0, "lon": 1.0},
            }
        ],
    )

    records = collect_station_event_records(npz_path)

    assert [record.station for record in records] == ["STA_NEAR", "STA_MID", "STA_FAR"]
    assert records[0].distance_km < records[1].distance_km < records[2].distance_km


def test_collect_station_event_records_skips_invalid_station_metadata(tmp_path: Path) -> None:
    npz_path = _write_npz(
        tmp_path / "synthetic.npz",
        [{"GOOD": _wave(), "BAD": _wave()}],
        [{"GOOD": {"lat": 0.0, "lon": 0.1}, "BAD": {"lat": np.nan, "lon": np.nan}}],
    )

    records = collect_station_event_records(npz_path)

    assert [record.station for record in records] == ["GOOD"]


def test_collect_figure1_station_event_records_excludes_blacklist_and_adds_unseen(tmp_path: Path) -> None:
    npz_path = _write_npz(
        tmp_path / "synthetic.npz",
        [{"KEEP": _wave()}, {"DROP": _wave()}],
        [{"KEEP": {"lat": 0.0, "lon": 0.1}}, {"DROP": {"lat": 0.0, "lon": 0.2}}],
        events=["KeepEvent", "DropEvent"],
    )
    unseen_dir = _write_unseen_event_dir(tmp_path / "unseen-event")

    records = collect_figure1_station_event_records(
        npz_path=npz_path,
        blacklist_events={"DropEvent"},
        unseen_event_dirs=[unseen_dir],
    )

    assert {record.event_name for record in records} == {"KeepEvent", "Unseen"}
    unseen = next(record for record in records if record.event_name == "Unseen")
    np.testing.assert_allclose(unseen.e_mm, [1.0, 2.0, 3.0])



def test_build_component_images_keeps_enu_components_separate() -> None:
    records = [
        StationEventRecord(
            event_index=0,
            event_name="E0",
            station="S0",
            distance_km=10.0,
            t=np.array([0.0, 1.0, 2.0]),
            e_mm=np.array([1.0, 2.0, 3.0]),
            n_mm=np.array([4.0, 5.0, 6.0]),
            u_mm=np.array([7.0, 8.0, 9.0]),
        ),
        StationEventRecord(
            event_index=0,
            event_name="E0",
            station="S1",
            distance_km=20.0,
            t=np.array([0.0, 1.0, 2.0]),
            e_mm=np.array([10.0, 20.0, 30.0]),
            n_mm=np.array([40.0, 50.0, 60.0]),
            u_mm=np.array([70.0, 80.0, 90.0]),
        ),
    ]
    time_grid = make_time_grid(0.0, 2.0, 1.0)

    images = build_component_images(records, time_grid)

    assert images["E"].shape == (2, 3)
    assert images["N"].shape == (2, 3)
    assert images["U"].shape == (2, 3)
    np.testing.assert_allclose(images["E"][0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(images["N"][0], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(images["U"][0], [7.0, 8.0, 9.0])


def test_symmetric_displacement_limit_is_positive() -> None:
    images = {
        "E": np.array([[-2.0, 1.0]]),
        "N": np.array([[0.0, 3.0]]),
        "U": np.array([[-4.0, 0.0]]),
    }

    limit = symmetric_displacement_limit(images, percentile=100.0)

    assert limit == 4.0


def test_normalize_component_rows_scales_each_trace_for_dense_overview() -> None:
    image = np.array([[0.0, 2.0, 4.0], [10.0, 20.0, 30.0]])

    normalized = normalize_component_rows(image)

    np.testing.assert_allclose(normalized[0], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(normalized[1], [-1.0, 0.0, 1.0])


def test_normalize_component_rows_handles_empty_window_without_warning() -> None:
    image = np.array([[np.nan, np.nan, np.nan], [0.0, 1.0, 2.0]])

    normalized = normalize_component_rows(image)

    np.testing.assert_allclose(normalized[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(normalized[1], [-1.0, 0.0, 1.0])


def test_compute_wiggle_scale_uses_robust_amplitude_and_distance_range() -> None:
    records = [
        StationEventRecord(0, "E0", "S0", 100.0, np.array([0.0, 1.0]), np.array([0.0, 10.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])),
        StationEventRecord(0, "E0", "S1", 300.0, np.array([0.0, 1.0]), np.array([0.0, 20.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])),
    ]

    scale = compute_wiggle_scale(records, percentile=100.0)

    assert scale.norm_factor == 0.3
    assert scale.scale_mm == 10.0
    assert scale.visual_amplitude_km == 6.0


def test_distance_tick_rows_maps_fixed_distance_ticks_to_sorted_rows() -> None:
    records = [
        StationEventRecord(0, "E0", "S0", 0.0, np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])),
        StationEventRecord(0, "E0", "S1", 210.0, np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])),
        StationEventRecord(0, "E0", "S2", 790.0, np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])),
    ]

    rows = distance_tick_rows(records, [0, 200, 800])

    assert rows == [0, 1, 2]


def test_make_figure1_with_panel_b_writes_png_only(tmp_path: Path) -> None:
    panel_a_path = tmp_path / "panel_a.png"
    output_path = tmp_path / "fig1_event_map.png"
    Image.new("RGB", (300, 160), "white").save(panel_a_path)
    npz_path = _write_npz(
        tmp_path / "synthetic.npz",
        [{"STA0": _wave(), "STA1": _wave(2.0)}],
        [{"STA0": {"lat": 0.0, "lon": 0.1}, "STA1": {"lat": 0.0, "lon": 0.2}}],
    )

    saved = make_figure1_with_panel_b(
        npz_path=npz_path,
        panel_a_path=panel_a_path,
        output_path=output_path,
        time_min_sec=0.0,
        time_max_sec=2.0,
        sample_dt_sec=1.0,
    )

    assert saved == output_path
    assert output_path.exists()
    assert not output_path.with_suffix(".pdf").exists()
