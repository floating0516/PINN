from __future__ import annotations

from datetime import datetime, timezone
import math

import pytest

from src.data.magnitude_relabel import (
    AmbiguousUSGSMatch,
    LocalEvent,
    MissingMagnitudeSource,
    NoUSGSMatch,
    assess_source_conflicts,
    extract_usgs_magnitude,
    match_usgs_event,
    read_native_stf,
    resolve_magnitude,
)
from src.data.stf import moment_to_mw


def _local_event(
    *,
    event: str = "Example",
    origin_time: str = "2020-01-02T03:04:05",
    latitude: float = 10.0,
    longitude: float = 20.0,
    gcmt_mw: float = 7.1,
) -> LocalEvent:
    return LocalEvent(
        event_index=0,
        event=event,
        origin_time=origin_time,
        latitude=latitude,
        longitude=longitude,
        gcmt_mw=gcmt_mw,
    )


def _feature(
    event_id: str,
    *,
    seconds: float = 0.0,
    latitude: float = 10.0,
    longitude: float = 20.0,
    magnitude: float = 7.1,
    magnitude_type: str = "mww",
) -> dict[str, object]:
    epoch = datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc).timestamp()
    return {
        "id": event_id,
        "properties": {
            "time": int((epoch + seconds) * 1000),
            "mag": magnitude,
            "magType": magnitude_type,
            "detail": f"https://example.test/{event_id}",
        },
        "geometry": {"coordinates": [longitude, latitude, 12.0]},
    }


@pytest.mark.parametrize("magnitude_type", ["Mw", "MWW", "mwc", "Mwr", "mwb"])
def test_usgs_preferred_mw_family_has_first_priority(magnitude_type: str) -> None:
    detail = {
        "id": "us-test",
        "properties": {
            "mag": 7.2,
            "magType": magnitude_type,
            "updated": 1_600_000_000_000,
            "products": {},
        },
    }

    selected = extract_usgs_magnitude(detail)

    assert selected is not None
    assert selected.mw == pytest.approx(7.2)
    assert selected.source == "usgs_preferred"
    assert selected.source_rank == 1
    assert selected.magnitude_type == magnitude_type
    assert selected.usgs_event_id == "us-test"


def test_non_mw_preferred_falls_through_to_highest_weight_usable_moment_tensor() -> None:
    high_weight_moment = 1.0e20
    detail = {
        "id": "us-mt",
        "properties": {
            "mag": 7.0,
            "magType": "mb",
            "products": {
                "moment-tensor": [
                    {
                        "code": "lower",
                        "source": "gcmt",
                        "preferredWeight": 10,
                        "updateTime": 100,
                        "properties": {"scalar-moment": "2.0e20"},
                    },
                    {
                        "code": "invalid",
                        "source": "us",
                        "preferredWeight": 200,
                        "updateTime": 300,
                        "properties": {"scalar-moment": "not-a-number"},
                    },
                    {
                        "code": "selected",
                        "source": "us",
                        "preferredWeight": 100,
                        "updateTime": 200,
                        "properties": {"scalar-moment": str(high_weight_moment)},
                    },
                ]
            },
        },
    }

    selected = extract_usgs_magnitude(detail)

    assert selected is not None
    assert selected.mw == pytest.approx(moment_to_mw(high_weight_moment))
    assert selected.source == "usgs_moment_tensor"
    assert selected.source_rank == 2
    assert selected.magnitude_type == "Mw(moment_tensor)"
    assert selected.product_id == "selected"
    assert selected.contributor == "us"
    assert selected.scalar_moment_nm == high_weight_moment


def test_gcmt_then_scardec_preserve_scalar_supervision() -> None:
    gcmt = resolve_magnitude(
        usgs_detail=None,
        gcmt_mw=7.1,
        stf_native_mw=7.0,
    )
    scardec = resolve_magnitude(
        usgs_detail=None,
        gcmt_mw=math.nan,
        stf_native_mw=6.9,
    )

    assert (gcmt.mw_selected, gcmt.source, gcmt.source_rank) == (
        7.1,
        "gcmt",
        3,
    )
    assert (scardec.mw_selected, scardec.source, scardec.source_rank) == (
        6.9,
        "scardec_native_stf",
        4,
    )


def test_complete_source_absence_is_a_hard_failure() -> None:
    with pytest.raises(MissingMagnitudeSource, match="scalar magnitude"):
        resolve_magnitude(
            usgs_detail=None,
            gcmt_mw=math.nan,
            stf_native_mw=math.nan,
        )


def test_unique_usgs_match_uses_time_distance_and_magnitude_evidence() -> None:
    local = _local_event()
    result = match_usgs_event(
        local,
        [
            _feature("too-late", seconds=31.0),
            _feature("too-far", longitude=21.1),
            _feature("too-different", magnitude=7.61),
            _feature("selected", seconds=10.0, longitude=20.2, magnitude=7.3),
        ],
    )

    assert result.usgs_event_id == "selected"
    assert result.time_delta_sec == pytest.approx(10.0)
    assert 0.0 < result.distance_km < 100.0
    assert result.magnitude_delta == pytest.approx(0.2)
    assert result.explicit_mapping_id == ""


def test_ambiguous_or_absent_usgs_matches_are_rejected() -> None:
    local = _local_event()

    with pytest.raises(AmbiguousUSGSMatch, match="2 acceptable"):
        match_usgs_event(local, [_feature("a"), _feature("b", seconds=1.0)])
    with pytest.raises(NoUSGSMatch, match="no acceptable"):
        match_usgs_event(local, [_feature("late", seconds=31.0)])


def test_explicit_mapping_validates_evidence_and_allows_duplicate_physical_event() -> None:
    feature = _feature("official-tohoku", magnitude=9.1)
    first = _local_event(event="Iwate2011", gcmt_mw=9.1)
    second = _local_event(event="Tohoku2011", gcmt_mw=9.1)

    first_match = match_usgs_event(
        first,
        [feature],
        explicit_event_id="official-tohoku",
        explicit_mapping_id="map-v1-001",
    )
    second_match = match_usgs_event(
        second,
        [feature],
        explicit_event_id="official-tohoku",
        explicit_mapping_id="map-v1-002",
    )

    assert first_match.usgs_event_id == second_match.usgs_event_id
    assert first_match.explicit_mapping_id == "map-v1-001"
    assert second_match.explicit_mapping_id == "map-v1-002"


def test_explicit_mapping_cannot_bypass_match_thresholds() -> None:
    with pytest.raises(NoUSGSMatch, match="explicit USGS event"):
        match_usgs_event(
            _local_event(),
            [_feature("wrong", seconds=31.0)],
            explicit_event_id="wrong",
            explicit_mapping_id="map-v1-bad",
        )


@pytest.mark.parametrize(
    ("values", "warning", "review_required"),
    [
        ({"usgs": 7.2, "gcmt": 7.1}, False, False),
        ({"usgs": 7.21, "gcmt": 7.1}, True, False),
        ({"usgs": 7.3, "gcmt": 7.1}, True, True),
    ],
)
def test_conflict_thresholds_are_exact(
    values: dict[str, float],
    warning: bool,
    review_required: bool,
) -> None:
    result = assess_source_conflicts(values)

    assert result.warning is warning
    assert result.review_required is review_required
    assert len(result.differences) == 1


def test_native_stf_uses_causal_numeric_rows_and_records_traceability(tmp_path) -> None:
    path = tmp_path / "event.stf"
    path.write_text(
        "2020 01 02 03 04 05.0 10.0 20.0\n"
        "10. 1.000E+19 6.6 1 2 3 4 5 6\n"
        "-1.0 9.0e18\n"
        "0.0 0.0\n"
        "1.0 2.0e19\n"
        "2.0 0.0\n",
        encoding="ascii",
    )

    result = read_native_stf(path)

    assert result.scalar_moment_nm == pytest.approx(2.0e19)
    assert result.mw == pytest.approx(moment_to_mw(2.0e19))
    assert result.traceable_header is True
    assert result.header_mw == pytest.approx(6.6)
    assert result.sha256


def test_native_stf_without_source_header_remains_visible_as_warning(tmp_path) -> None:
    path = tmp_path / "untraceable.stf"
    path.write_text("0 0\n1 1e18\n2 0\n", encoding="ascii")

    result = read_native_stf(path)

    assert result.mw == pytest.approx(moment_to_mw(1.0e18))
    assert result.traceable_header is False
    assert result.warning == "missing_source_header"
