from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math

import numpy as np
import pytest

from scripts.data.build_usgs_magnitude_labels import (
    CacheMiss,
    PublicationBlocked,
    build_snapshot,
    load_or_fetch_json,
    validate_publication_rows,
)
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
    validate_candidate_npz,
    write_candidate_npz,
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
        explicit_review_disposition="identity_verified_duplicate_event",
    )
    second_match = match_usgs_event(
        second,
        [feature],
        explicit_event_id="official-tohoku",
        explicit_mapping_id="map-v1-002",
        explicit_review_disposition="identity_verified_duplicate_event",
    )

    assert first_match.usgs_event_id == second_match.usgs_event_id
    assert first_match.explicit_mapping_id == "map-v1-001"
    assert second_match.explicit_mapping_id == "map-v1-002"


def test_explicit_mapping_can_resolve_threshold_exception_with_review() -> None:
    result = match_usgs_event(
        _local_event(),
        [_feature("reviewed", seconds=31.0)],
        explicit_event_id="reviewed",
        explicit_mapping_id="map-v1-reviewed",
        explicit_review_disposition="identity_verified_from_authoritative_id",
    )

    assert result.time_delta_sec == pytest.approx(31.0)
    assert result.review_disposition == "identity_verified_from_authoritative_id"


def test_explicit_mapping_requires_review_disposition() -> None:
    with pytest.raises(ValueError, match="review disposition"):
        match_usgs_event(
            _local_event(),
            [_feature("mapped")],
            explicit_event_id="mapped",
            explicit_mapping_id="map-v1-mapped",
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


def test_cached_json_replay_never_calls_network(tmp_path) -> None:
    raw = b'{"features":[],"type":"FeatureCollection"}\n'
    cache_path = tmp_path / "query.json"
    cache_path.write_bytes(raw)

    def forbidden_fetch(_url: str) -> bytes:
        raise AssertionError("offline replay attempted network access")

    cached = load_or_fetch_json(
        cache_path,
        "https://example.test/query",
        offline=True,
        fetch_bytes=forbidden_fetch,
    )

    assert cached.payload["type"] == "FeatureCollection"
    assert cached.sha256 == hashlib.sha256(raw).hexdigest()
    assert cached.from_cache is True


def test_offline_cache_miss_is_explicit(tmp_path) -> None:
    with pytest.raises(CacheMiss, match="offline cache miss"):
        load_or_fetch_json(
            tmp_path / "missing.json",
            "https://example.test/query",
            offline=True,
            fetch_bytes=lambda _url: b"{}",
        )


def test_download_retries_are_bounded_and_invalid_json_is_not_persisted(tmp_path) -> None:
    attempts = 0

    def transient_fetch(_url: str) -> bytes:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("transient")
        return b'{"id":"ok"}\n'

    cache_path = tmp_path / "retry.json"
    downloaded = load_or_fetch_json(
        cache_path,
        "https://example.test/detail",
        offline=False,
        fetch_bytes=transient_fetch,
        retry_delays_sec=(0.0, 0.0),
    )

    assert attempts == 3
    assert downloaded.payload == {"id": "ok"}
    assert downloaded.from_cache is False
    assert cache_path.read_bytes() == b'{"id":"ok"}\n'

    invalid_path = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="valid JSON"):
        load_or_fetch_json(
            invalid_path,
            "https://example.test/invalid",
            offline=False,
            fetch_bytes=lambda _url: b"not-json",
            retry_delays_sec=(),
        )
    assert not invalid_path.exists()


def _label_rows() -> list[dict[str, object]]:
    return [
        {
            "event_index": 0,
            "event": "A",
            "mw_selected": 7.2,
            "mw_usgs": 7.2,
            "mw_gcmt": 7.1,
            "mw_stf_native": 7.0,
            "mw_source": "usgs_preferred",
            "mw_source_rank": 1,
            "mw_type": "mww",
            "usgs_event_id": "us-a",
            "product_id": "",
        },
        {
            "event_index": 1,
            "event": "B",
            "mw_selected": 6.9,
            "mw_usgs": math.nan,
            "mw_gcmt": 6.9,
            "mw_stf_native": math.nan,
            "mw_source": "gcmt",
            "mw_source_rank": 3,
            "mw_type": "Mw",
            "usgs_event_id": "",
            "product_id": "",
        },
    ]


def _source_npz(path) -> None:
    station_payload = np.empty(2, dtype=object)
    station_payload[0] = {"S1": {"E": np.array([1.0, 2.0])}}
    station_payload[1] = {"S2": {"E": np.array([3.0, 4.0])}}
    np.savez(
        path,
        events=np.asarray(["A", "B"], dtype=object),
        magnitude=np.asarray([7.1, 6.9]),
        latitude=np.asarray([1.0, 2.0]),
        enu=station_payload,
    )


def test_candidate_npz_is_deterministic_and_changes_only_label_arrays(tmp_path) -> None:
    source = tmp_path / "source.npz"
    first = tmp_path / "candidate-a.npz"
    second = tmp_path / "candidate-b.npz"
    _source_npz(source)

    first_hash = write_candidate_npz(source, first, _label_rows())
    second_hash = write_candidate_npz(source, second, _label_rows())

    assert first_hash == second_hash
    assert first.read_bytes() == second.read_bytes()
    validate_candidate_npz(source, first)
    with np.load(first, allow_pickle=True) as candidate:
        assert candidate.files == [
            "events",
            "magnitude",
            "latitude",
            "enu",
            "magnitude_selected",
            "magnitude_usgs",
            "magnitude_gcmt",
            "magnitude_stf_native",
            "magnitude_source",
            "magnitude_source_rank",
            "magnitude_type",
            "usgs_event_id",
            "usgs_product_id",
        ]
        assert np.array_equal(candidate["magnitude"], candidate["magnitude_selected"])
        assert candidate["magnitude_source"].tolist() == ["usgs_preferred", "gcmt"]
        assert candidate["magnitude_source_rank"].tolist() == [1, 3]


def test_candidate_validator_detects_non_label_mutation(tmp_path) -> None:
    source = tmp_path / "source.npz"
    candidate = tmp_path / "candidate.npz"
    tampered = tmp_path / "tampered.npz"
    _source_npz(source)
    write_candidate_npz(source, candidate, _label_rows())
    with np.load(candidate, allow_pickle=True) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["latitude"] = np.asarray([99.0, 2.0])
    np.savez(tampered, **arrays)

    with pytest.raises(ValueError, match="non-label array changed: latitude"):
        validate_candidate_npz(source, tampered)


def _publication_row(
    index: int,
    event: str,
    *,
    usgs_event_id: str,
    review_required: bool = False,
    review_disposition: str = "",
    explicit_mapping_id: str = "",
) -> dict[str, object]:
    return {
        "event_index": index,
        "event": event,
        "mw_selected": 7.0,
        "mw_source": "usgs_preferred",
        "mw_source_rank": 1,
        "mw_type": "mww",
        "usgs_event_id": usgs_event_id,
        "review_required": review_required,
        "review_disposition": review_disposition,
        "explicit_mapping_id": explicit_mapping_id,
    }


def test_publication_requires_complete_reviewed_rows_and_explicit_duplicates() -> None:
    expected = ["A", "B"]
    with pytest.raises(ValueError, match="resolution rows"):
        validate_publication_rows(
            [_publication_row(0, "A", usgs_event_id="us-a")],
            expected_events=expected,
            accepted_events={"A"},
        )

    with pytest.raises(ValueError, match="review disposition"):
        validate_publication_rows(
            [
                _publication_row(
                    0,
                    "A",
                    usgs_event_id="us-a",
                    review_required=True,
                ),
                _publication_row(1, "B", usgs_event_id="us-b"),
            ],
            expected_events=expected,
            accepted_events={"A"},
        )

    duplicate_rows = [
        _publication_row(0, "A", usgs_event_id="same"),
        _publication_row(1, "B", usgs_event_id="same"),
    ]
    with pytest.raises(ValueError, match="duplicate physical USGS event"):
        validate_publication_rows(
            duplicate_rows,
            expected_events=expected,
            accepted_events={"A", "B"},
        )

    for index, row in enumerate(duplicate_rows):
        row["explicit_mapping_id"] = f"map-v1-{index}"
        row["review_disposition"] = "identity_verified_duplicate_event"
    validate_publication_rows(
        duplicate_rows,
        expected_events=expected,
        accepted_events={"A", "B"},
    )


def _buildable_source_npz(path) -> None:
    empty = np.empty(1, dtype=object)
    empty[0] = {}
    np.savez(
        path,
        events=np.asarray(["A"], dtype=object),
        country=np.asarray(["X"], dtype=object),
        longitude=np.asarray([20.0]),
        latitude=np.asarray([10.0]),
        depth_km=np.asarray([12.0]),
        magnitude=np.asarray([7.1]),
        mechanism=np.asarray(["Reverse"], dtype=object),
        strike=np.asarray([1.0]),
        dip=np.asarray([2.0]),
        rake=np.asarray([3.0]),
        origin_time=np.asarray(["2020-01-02T03:04:05"], dtype=object),
        station_count=np.asarray([0], dtype=np.int32),
        enu=empty,
        station_info=empty.copy(),
    )


def _snapshot_inputs(tmp_path):
    source = tmp_path / "source.npz"
    gcmt = tmp_path / "gcmt.csv"
    stf_dir = tmp_path / "stf"
    accepted = tmp_path / "accepted.csv"
    snapshot = tmp_path / "snapshot"
    _buildable_source_npz(source)
    gcmt.write_text("Event,Magnitude\nA,7.1\n", encoding="ascii")
    stf_dir.mkdir()
    (stf_dir / "A.stf").write_text(
        "2020 01 02 03 04 05.0 10.0 20.0\n"
        "10. 1.0E+20 7.2 1 2 3 4 5 6\n"
        "0 0\n1 1e20\n2 0\n",
        encoding="ascii",
    )
    accepted.write_text("event,accepted\nA,True\n", encoding="ascii")
    return source, gcmt, stf_dir, accepted, snapshot


def _fake_usgs_fetch(magnitude: float):
    feature = _feature("us-a", magnitude=magnitude)
    detail = {
        **feature,
        "properties": {
            **feature["properties"],
            "mag": magnitude,
            "magType": "mww",
            "updated": 1_600_000_000_000,
            "net": "us",
            "products": {},
        },
    }

    def fetch(url: str) -> bytes:
        payload = detail if "eventid=" in url else {
            "type": "FeatureCollection",
            "features": [feature],
        }
        return (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")

    return fetch


def test_snapshot_build_publishes_candidate_and_replays_offline(tmp_path) -> None:
    source, gcmt, stf_dir, accepted, snapshot = _snapshot_inputs(tmp_path)

    result = build_snapshot(
        source_npz=source,
        gcmt_csv=gcmt,
        stf_dir=stf_dir,
        accepted_manifest=accepted,
        external_event_root=None,
        snapshot_root=snapshot,
        mapping_table=None,
        offline=False,
        verify_existing=False,
        fetch_bytes=_fake_usgs_fetch(7.2),
    )

    assert result["event_count"] == 1
    assert result["accepted_event_count"] == 1
    assert result["source_counts"] == {"usgs_preferred": 1}
    assert (snapshot / "magnitude_labels.csv").is_file()
    assert (snapshot / "magnitude_labels.json").is_file()
    assert (snapshot / "artifact_hashes.json").is_file()
    candidate = snapshot / "gnss_events_matched.usgs_priority.npz"
    validate_candidate_npz(source, candidate)

    replay = build_snapshot(
        source_npz=source,
        gcmt_csv=gcmt,
        stf_dir=stf_dir,
        accepted_manifest=accepted,
        external_event_root=None,
        snapshot_root=snapshot,
        mapping_table=None,
        offline=True,
        verify_existing=True,
        fetch_bytes=lambda _url: (_ for _ in ()).throw(
            AssertionError("offline replay attempted network")
        ),
    )
    assert replay["verified_existing"] is True
    assert replay["candidate_sha256"] == result["candidate_sha256"]


def test_snapshot_blocks_unreviewed_conflict_then_accepts_versioned_review(tmp_path) -> None:
    source, gcmt, stf_dir, accepted, snapshot = _snapshot_inputs(tmp_path)

    with pytest.raises(PublicationBlocked, match="review queue"):
        build_snapshot(
            source_npz=source,
            gcmt_csv=gcmt,
            stf_dir=stf_dir,
            accepted_manifest=accepted,
            external_event_root=None,
            snapshot_root=snapshot,
            mapping_table=None,
            offline=False,
            verify_existing=False,
            fetch_bytes=_fake_usgs_fetch(7.4),
        )
    assert (snapshot / "review_queue.csv").is_file()
    assert not (snapshot / "gnss_events_matched.usgs_priority.npz").exists()

    mapping = snapshot / "usgs_event_map.v1.csv"
    mapping.write_text(
        "mapping_id,event_index,event,usgs_event_id,review_disposition\n"
        ",0,A,us-a,identity_verified_source_difference\n",
        encoding="ascii",
    )
    result = build_snapshot(
        source_npz=source,
        gcmt_csv=gcmt,
        stf_dir=stf_dir,
        accepted_manifest=accepted,
        external_event_root=None,
        snapshot_root=snapshot,
        mapping_table=mapping,
        offline=True,
        verify_existing=False,
        fetch_bytes=lambda _url: (_ for _ in ()).throw(
            AssertionError("offline publication attempted network")
        ),
    )
    assert result["source_counts"] == {"usgs_preferred": 1}
