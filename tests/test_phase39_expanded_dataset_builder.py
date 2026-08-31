from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.data.build_phase39_expanded_dataset import (
    deterministic_savez_compressed,
    normalize_station_code,
    parse_scardec_header,
    station_identity,
)
from src.data.records_v2 import NormalizedStationRecord
from src.data.sample_builder import SampleRejected, build_station_sample
from src.data.waveform import WaveformConfig


def test_station_code_normalization_removes_only_terminal_gnss_suffix() -> None:
    assert normalize_station_code(" 0030.gnss ") == "0030"
    assert station_identity("ABCD.GNSS") == "abcd"
    assert normalize_station_code("gnss.ABCD") == "gnss.ABCD"


def test_parse_scardec_header_uses_first_nodal_plane(tmp_path: Path) -> None:
    source = tmp_path / "fctoptsource_event"
    source.write_text(
        "2022 09 19 18 05 08.0 18.455 -102.956\n"
        "15. 3.475E+20 7.627 306 17 102 114 74 86\n"
        "0.0 0.0\n1.0 1.0\n",
        encoding="utf-8",
    )

    header = parse_scardec_header(source)

    assert header.origin_time == "2022-09-19T18:05:08Z"
    assert header.moment_nm == pytest.approx(3.475e20)
    assert (header.strike, header.dip, header.rake) == (306.0, 17.0, 102.0)


def test_deterministic_npz_writer_has_stable_bytes(tmp_path: Path) -> None:
    arrays = {
        "events": np.asarray(["event"], dtype=object),
        "values": np.asarray([1.0, 2.0], dtype=np.float64),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    deterministic_savez_compressed(first, arrays)
    deterministic_savez_compressed(second, arrays)

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()
    with np.load(first, allow_pickle=True) as data:
        assert data["events"].tolist() == ["event"]
        np.testing.assert_array_equal(data["values"], arrays["values"])


def test_station_phase_metadata_preserves_exact_one_hz_samples() -> None:
    phase_start = 0.23
    time_sec = np.arange(-61.0, 201.0, dtype=np.float64) + phase_start
    north = np.zeros_like(time_sec)
    north[time_sec >= 10.0] = 0.03
    config = WaveformConfig(
        sample_rate_hz=1.0,
        start_sec=0.0,
        duration_sec=200.0,
        min_valid_fraction=0.99,
        max_interpolation_gap_sec=0.0,
        baseline_method="median",
        pre_event_start_sec=-60.0,
        pre_event_end_sec=0.0,
        baseline_fallback="pre_p",
        baseline_fallback_max_sec=30.0,
        baseline_min_samples=10,
        filter_type="none",
        cutoff_hz=0.2,
        num_taps=7,
        filter_window="hamming",
    )
    common = dict(
        event_index=0,
        event="phase-event",
        magnitude_catalog=7.0,
        event_lat=0.0,
        event_lon=0.0,
        depth_km=10.0,
        strike=0.0,
        dip=90.0,
        rake=0.0,
        mechanism=1,
        station="ABCD",
        station_lat=1.0,
        station_lon=0.0,
        time_sec=time_sec,
        east=np.zeros_like(time_sec),
        north=north,
        vertical=np.zeros_like(time_sec),
        origin_sec=None,
    )

    with pytest.raises(SampleRejected, match="valid fraction"):
        build_station_sample(
            NormalizedStationRecord(**common),
            units="m",
            waveform_config=config,
            alpha_m_per_s=7900.0,
            radial_peak_min_cm=2.0,
        )

    sample = build_station_sample(
        NormalizedStationRecord(
            **common,
            waveform_start_sec=phase_start,
        ),
        units="m",
        waveform_config=config,
        alpha_m_per_s=7900.0,
        radial_peak_min_cm=2.0,
    )

    assert sample["waveform_start_sec"] == pytest.approx(phase_start)
    assert sample["valid_fraction"] == pytest.approx(1.0)
    assert sample["radial_peak_cm"] == pytest.approx(3.0)
