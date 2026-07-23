from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.magnitude_relabel import (  # noqa: E402
    AmbiguousUSGSMatch,
    LocalEvent,
    MagnitudeRelabelError,
    NoUSGSMatch,
    assess_source_conflicts,
    extract_usgs_magnitude,
    match_usgs_event,
    read_native_stf,
    resolve_magnitude,
    validate_candidate_npz,
    write_candidate_npz,
)


USGS_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
LABEL_FIELDS = (
    "event_index",
    "event",
    "origin_time",
    "latitude",
    "longitude",
    "mw_selected",
    "mw_usgs",
    "mw_gcmt",
    "mw_stf_native",
    "mw_source",
    "mw_source_rank",
    "mw_type",
    "usgs_event_id",
    "product_id",
    "contributor",
    "source_update_time",
    "scalar_moment_nm",
    "query_url",
    "detail_url",
    "query_sha256",
    "detail_sha256",
    "selected_source_sha256",
    "source_npz_sha256",
    "gcmt_csv_sha256",
    "stf_path",
    "stf_sha256",
    "stf_traceable_header",
    "stf_warning",
    "match_time_delta_sec",
    "match_distance_km",
    "match_magnitude_delta",
    "explicit_mapping_id",
    "delta_usgs_vs_gcmt",
    "delta_usgs_vs_stf_native",
    "delta_gcmt_vs_stf_native",
    "warning",
    "review_required",
    "review_disposition",
)
EXTERNAL_LABEL_FIELDS = (
    "event",
    "event_dir",
    "usgs_event_id",
    "mw_original",
    "mw_selected",
    "mw_source",
    "mw_source_rank",
    "mw_type",
    "product_id",
    "contributor",
    "source_update_time",
    "scalar_moment_nm",
    "detail_url",
    "detail_sha256",
    "match_time_delta_sec",
    "match_distance_km",
    "match_magnitude_delta",
    "review_disposition",
)
REVIEW_FIELDS = (
    "event_index",
    "event",
    "usgs_event_id",
    "reason",
    "mw_usgs",
    "mw_gcmt",
    "mw_stf_native",
    "match_time_delta_sec",
    "match_distance_km",
    "match_magnitude_delta",
    "suggested_review_disposition",
)


class CacheMiss(FileNotFoundError):
    pass


class PublicationBlocked(MagnitudeRelabelError):
    pass


@dataclass(frozen=True)
class CachedJSON:
    payload: Any
    sha256: str
    from_cache: bool
    path: Path
    url: str


def _decode_json(raw: bytes, *, source: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{source} is not valid JSON") from error


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _http_fetch_bytes(url: str) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "PINN-Mag-USGS-relabel/1.0 (research cache)"},
    )
    with urlopen(request, timeout=60) as response:
        return response.read()


def load_or_fetch_json(
    cache_path: str | Path,
    url: str,
    *,
    offline: bool,
    fetch_bytes: Callable[[str], bytes] = _http_fetch_bytes,
    retry_delays_sec: Sequence[float] = (1.0, 2.0),
) -> CachedJSON:
    path = Path(cache_path)
    if path.is_file():
        raw = path.read_bytes()
        return CachedJSON(
            payload=_decode_json(raw, source=str(path)),
            sha256=hashlib.sha256(raw).hexdigest(),
            from_cache=True,
            path=path,
            url=url,
        )
    if offline:
        raise CacheMiss(f"offline cache miss: {path}")

    delays = [0.0, *map(float, retry_delays_sec)]
    for attempt, delay in enumerate(delays):
        if delay > 0.0:
            time.sleep(delay)
        try:
            raw = fetch_bytes(url)
        except OSError as error:
            if attempt + 1 < len(delays):
                continue
            raise
        if not isinstance(raw, bytes):
            raise TypeError("JSON fetcher must return bytes")
        payload = _decode_json(raw, source=url)
        _atomic_write_bytes(path, raw)
        return CachedJSON(
            payload=payload,
            sha256=hashlib.sha256(raw).hexdigest(),
            from_cache=False,
            path=path,
            url=url,
        )
    raise RuntimeError("JSON retry loop ended without a result")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes"}


def validate_publication_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_events: Sequence[str],
    accepted_events: set[str],
) -> None:
    if len(rows) != len(expected_events):
        raise ValueError(
            f"resolution rows {len(rows)} do not cover {len(expected_events)} events"
        )
    seen_events: set[str] = set()
    for index, (expected_event, row) in enumerate(
        zip(expected_events, rows, strict=True)
    ):
        event = str(row.get("event", ""))
        if int(row.get("event_index", -1)) != index or event != expected_event:
            raise ValueError(f"resolution rows are out of order at event {index}")
        if event in seen_events:
            raise ValueError(f"duplicate resolution event: {event}")
        seen_events.add(event)
        try:
            magnitude = float(row["mw_selected"])
            source_rank = int(row["mw_source_rank"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid scalar resolution for {event}") from error
        if (
            not math.isfinite(magnitude)
            or not str(row.get("mw_source", ""))
            or not str(row.get("mw_type", ""))
            or source_rank not in {1, 2, 3, 4}
        ):
            raise ValueError(f"invalid scalar resolution for {event}")
        if _truthy(row.get("review_required")) and not str(
            row.get("review_disposition", "")
        ):
            raise ValueError(f"review disposition is required for {event}")
    missing_accepted = accepted_events - seen_events
    if missing_accepted:
        raise ValueError(
            f"accepted events lack scalar resolution: {sorted(missing_accepted)}"
        )
    usgs_ids = [str(row.get("usgs_event_id", "")) for row in rows]
    duplicate_ids = {
        event_id
        for event_id, count in Counter(usgs_ids).items()
        if event_id and count > 1
    }
    for event_id in duplicate_ids:
        duplicate_rows = [
            row for row in rows if str(row.get("usgs_event_id", "")) == event_id
        ]
        if any(
            not str(row.get("explicit_mapping_id", ""))
            or not str(row.get("review_disposition", ""))
            for row in duplicate_rows
        ):
            raise ValueError(
                f"duplicate physical USGS event {event_id} requires explicit mappings"
            )


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            _json_ready(payload),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fieldnames),
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(_json_ready(row))
    return stream.getvalue().encode("utf-8")


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    _atomic_write_bytes(path, _csv_bytes(rows, fieldnames))


def _parse_origin(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    result = datetime.fromisoformat(text)
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def _query_url(event: LocalEvent) -> str:
    origin = _parse_origin(event.origin_time)
    return f"{USGS_QUERY_URL}?" + urlencode(
        {
            "format": "geojson",
            "starttime": _iso_z(origin - timedelta(seconds=30)),
            "endtime": _iso_z(origin + timedelta(seconds=30)),
            "latitude": f"{event.latitude:.6f}",
            "longitude": f"{event.longitude:.6f}",
            "maxradiuskm": "100",
            "minmagnitude": f"{event.gcmt_mw - 0.5:.3f}",
            "maxmagnitude": f"{event.gcmt_mw + 0.5:.3f}",
            "orderby": "time-asc",
        }
    )


def _detail_url(event_id: str) -> str:
    return f"{USGS_QUERY_URL}?" + urlencode(
        {"eventid": event_id, "format": "geojson"}
    )


def _query_cache_path(root: Path, event: LocalEvent) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", event.event)
    return root / "raw_usgs" / "queries" / (
        f"{event.event_index:02d}-{safe_name}.json"
    )


def _detail_cache_path(root: Path, event_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", event_id):
        raise ValueError(f"invalid USGS event ID: {event_id!r}")
    return root / "raw_usgs" / "details" / f"{event_id}.json"


def _load_local_events(source_npz: Path) -> list[LocalEvent]:
    with np.load(source_npz, allow_pickle=True) as data:
        required = (
            "events",
            "origin_time",
            "latitude",
            "longitude",
            "magnitude",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"source NPZ lacks event metadata: {missing}")
        count = len(data["events"])
        arrays = {name: data[name] for name in required}
    return [
        LocalEvent(
            event_index=index,
            event=str(arrays["events"][index]),
            origin_time=str(arrays["origin_time"][index]),
            latitude=float(arrays["latitude"][index]),
            longitude=float(arrays["longitude"][index]),
            gcmt_mw=float(arrays["magnitude"][index]),
        )
        for index in range(count)
    ]


def _load_accepted_events(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    accepted = {
        str(row.get("event", ""))
        for row in rows
        if _truthy(row.get("accepted")) and str(row.get("event", ""))
    }
    if not accepted:
        raise ValueError("accepted manifest contains no accepted events")
    return accepted


def _load_mapping_table(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            index = int(row.get("event_index", ""))
        except (TypeError, ValueError) as error:
            raise ValueError("mapping table has an invalid event_index") from error
        if index in result:
            raise ValueError(f"duplicate mapping row for event index {index}")
        result[index] = {
            "mapping_id": str(row.get("mapping_id", "")).strip(),
            "event": str(row.get("event", "")).strip(),
            "usgs_event_id": str(row.get("usgs_event_id", "")).strip(),
            "review_disposition": str(
                row.get("review_disposition", "")
            ).strip(),
        }
    return result


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _resolve_internal_rows(
    *,
    events: Sequence[LocalEvent],
    snapshot_root: Path,
    gcmt_csv_sha256: str,
    source_npz_sha256: str,
    stf_dir: Path,
    mappings: Mapping[int, Mapping[str, str]],
    offline: bool,
    fetch_bytes: Callable[[str], bytes],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    for event in events:
        mapping = mappings.get(event.event_index, {})
        if mapping and mapping.get("event") not in {"", event.event}:
            raise ValueError(
                f"mapping event mismatch at index {event.event_index}"
            )
        query_url = _query_url(event)
        query = load_or_fetch_json(
            _query_cache_path(snapshot_root, event),
            query_url,
            offline=offline,
            fetch_bytes=fetch_bytes,
        )
        if not isinstance(query.payload, Mapping):
            raise ValueError(f"USGS query is not an object for {event.event}")
        features = query.payload.get("features", [])
        if not isinstance(features, Sequence):
            raise ValueError(f"USGS query lacks features for {event.event}")
        mapped_id = str(mapping.get("usgs_event_id", ""))
        mapping_id = str(mapping.get("mapping_id", ""))
        review_disposition = str(mapping.get("review_disposition", ""))
        try:
            automatic_match = match_usgs_event(event, features)
        except (NoUSGSMatch, AmbiguousUSGSMatch):
            automatic_match = None

        if mapping_id:
            if not mapped_id:
                raise ValueError(
                    f"mapping {mapping_id} lacks a USGS event ID"
                )
            detail = load_or_fetch_json(
                _detail_cache_path(snapshot_root, mapped_id),
                _detail_url(mapped_id),
                offline=offline,
                fetch_bytes=fetch_bytes,
            )
            if not isinstance(detail.payload, Mapping):
                raise ValueError(f"USGS detail is not an object for {event.event}")
            match = match_usgs_event(
                event,
                [detail.payload],
                explicit_event_id=mapped_id,
                explicit_mapping_id=mapping_id,
                explicit_review_disposition=review_disposition,
            )
        else:
            if automatic_match is None:
                review_queue.append(
                    {
                        "event_index": event.event_index,
                        "event": event.event,
                        "usgs_event_id": mapped_id,
                        "reason": "unresolved_or_ambiguous_identity",
                        "mw_gcmt": event.gcmt_mw,
                        "suggested_review_disposition": (
                            "identity_verified_from_authoritative_evidence"
                        ),
                    }
                )
                continue
            if mapped_id and mapped_id != automatic_match.usgs_event_id:
                raise ValueError(
                    f"review row USGS ID differs from automatic match for {event.event}"
                )
            match = automatic_match
            detail = load_or_fetch_json(
                _detail_cache_path(snapshot_root, match.usgs_event_id),
                _detail_url(match.usgs_event_id),
                offline=offline,
                fetch_bytes=fetch_bytes,
            )
        if not isinstance(detail.payload, Mapping):
            raise ValueError(f"USGS detail is not an object for {event.event}")
        if str(detail.payload.get("id", "")) != match.usgs_event_id:
            raise ValueError(f"USGS detail event ID mismatch for {event.event}")

        stf_path = stf_dir / f"{event.event}.stf"
        stf = read_native_stf(stf_path) if stf_path.is_file() else None
        stf_mw = stf.mw if stf is not None else math.nan
        usgs = extract_usgs_magnitude(detail.payload)
        usgs_mw = usgs.mw if usgs is not None else math.nan
        resolution = resolve_magnitude(
            usgs_detail=detail.payload,
            gcmt_mw=event.gcmt_mw,
            stf_native_mw=stf_mw,
        )
        conflicts = assess_source_conflicts(
            {
                "usgs": usgs_mw,
                "gcmt": event.gcmt_mw,
                "stf_native": stf_mw,
            }
        )
        warnings: list[str] = []
        if conflicts.warning:
            warnings.append("source_difference_gt_0.1_mw")
        if usgs is None:
            warnings.append("usgs_has_no_usable_mw")
        if stf is None:
            warnings.append("missing_exact_stf")
        elif stf.warning:
            warnings.append(stf.warning)
        if conflicts.review_required and not review_disposition:
            review_queue.append(
                {
                    "event_index": event.event_index,
                    "event": event.event,
                    "usgs_event_id": match.usgs_event_id,
                    "reason": "source_difference_ge_0.2_mw",
                    "mw_usgs": _float_or_none(usgs_mw),
                    "mw_gcmt": event.gcmt_mw,
                    "mw_stf_native": _float_or_none(stf_mw),
                    "match_time_delta_sec": match.time_delta_sec,
                    "match_distance_km": match.distance_km,
                    "match_magnitude_delta": match.magnitude_delta,
                    "suggested_review_disposition": (
                        "identity_verified_source_difference_retained_priority"
                    ),
                }
            )
        selected_source_hash = {
            1: detail.sha256,
            2: detail.sha256,
            3: gcmt_csv_sha256,
            4: stf.sha256 if stf is not None else "",
        }[resolution.source_rank]
        differences = conflicts.differences
        rows.append(
            {
                "event_index": event.event_index,
                "event": event.event,
                "origin_time": event.origin_time,
                "latitude": event.latitude,
                "longitude": event.longitude,
                "mw_selected": resolution.mw_selected,
                "mw_usgs": _float_or_none(usgs_mw),
                "mw_gcmt": event.gcmt_mw,
                "mw_stf_native": _float_or_none(stf_mw),
                "mw_source": resolution.source,
                "mw_source_rank": resolution.source_rank,
                "mw_type": resolution.magnitude_type,
                "usgs_event_id": match.usgs_event_id,
                "product_id": resolution.product_id,
                "contributor": resolution.contributor,
                "source_update_time": resolution.source_update_time,
                "scalar_moment_nm": _float_or_none(
                    resolution.scalar_moment_nm
                ),
                "query_url": query_url,
                "detail_url": detail.url,
                "query_sha256": query.sha256,
                "detail_sha256": detail.sha256,
                "selected_source_sha256": selected_source_hash,
                "source_npz_sha256": source_npz_sha256,
                "gcmt_csv_sha256": gcmt_csv_sha256,
                "stf_path": str(stf_path) if stf is not None else "",
                "stf_sha256": stf.sha256 if stf is not None else "",
                "stf_traceable_header": (
                    stf.traceable_header if stf is not None else False
                ),
                "stf_warning": stf.warning if stf is not None else "missing",
                "match_time_delta_sec": match.time_delta_sec,
                "match_distance_km": match.distance_km,
                "match_magnitude_delta": match.magnitude_delta,
                "explicit_mapping_id": mapping_id,
                "delta_usgs_vs_gcmt": differences.get("usgs_vs_gcmt"),
                "delta_usgs_vs_stf_native": differences.get(
                    "usgs_vs_stf_native"
                ),
                "delta_gcmt_vs_stf_native": differences.get(
                    "gcmt_vs_stf_native"
                ),
                "warning": ";".join(warnings),
                "review_required": conflicts.review_required,
                "review_disposition": review_disposition,
            }
        )

    id_counts = Counter(str(row["usgs_event_id"]) for row in rows)
    for row in rows:
        event_id = str(row["usgs_event_id"])
        if id_counts[event_id] > 1 and not row["explicit_mapping_id"]:
            review_queue.append(
                {
                    "event_index": row["event_index"],
                    "event": row["event"],
                    "usgs_event_id": event_id,
                    "reason": "duplicate_physical_usgs_event",
                    "mw_usgs": row["mw_usgs"],
                    "mw_gcmt": row["mw_gcmt"],
                    "mw_stf_native": row["mw_stf_native"],
                    "match_time_delta_sec": row["match_time_delta_sec"],
                    "match_distance_km": row["match_distance_km"],
                    "match_magnitude_delta": row["match_magnitude_delta"],
                    "suggested_review_disposition": (
                        "identity_verified_duplicate_event"
                    ),
                }
            )
    return rows, review_queue


def _external_display_name(
    *,
    event_name: str,
    event_dir_name: str,
    magnitude: float,
) -> str:
    parts = [
        part
        for part in event_dir_name.replace("_", "-").strip().lower().split("-")
        if part
    ]
    year = next(
        (part for part in parts if len(part) == 4 and part.isdigit()),
        "",
    )
    stopwords = {
        "aftershock", "mainshock", "foreshock", "doublet", "eq1", "eq2",
        "earthquake", "china", "japan", "greece", "alaska", "chile",
        "mexico", "burma", "myanmar", "new", "zealand", "costa", "rica",
        "plateau", "southern", "tibetan", "of", "km", "sw", "se", "nw",
        "ne", "n", "s", "e", "w",
    }
    candidates = [
        part for part in parts if not part.isdigit() and part not in stopwords
    ]
    place = candidates[0].capitalize() if candidates else event_name.strip()
    suffix = f" {year}" if year else ""
    return f"{place}{suffix} M{magnitude:.1f}"


def _resolve_external_rows(
    *,
    external_event_root: Path | None,
    snapshot_root: Path,
    offline: bool,
    fetch_bytes: Callable[[str], bytes],
) -> list[dict[str, Any]]:
    if external_event_root is None:
        return []
    event_paths = sorted(external_event_root.glob("*/event.json"))
    if not event_paths:
        raise ValueError("external event root contains no event.json files")
    rows: list[dict[str, Any]] = []
    for event_path in event_paths:
        metadata = json.loads(event_path.read_text(encoding="utf-8"))
        event_id = str(metadata.get("usgs_event_id", "")).strip()
        if not event_id:
            raise ValueError(f"external event lacks USGS ID: {event_path}")
        event_dir = event_path.parent.name
        original_mw = float(metadata["magnitude"])
        local = LocalEvent(
            event_index=-1,
            event=event_dir,
            origin_time=str(metadata["date"]),
            latitude=float(metadata["latitude"]),
            longitude=float(metadata["longitude"]),
            gcmt_mw=original_mw,
        )
        detail = load_or_fetch_json(
            _detail_cache_path(snapshot_root, event_id),
            _detail_url(event_id),
            offline=offline,
            fetch_bytes=fetch_bytes,
        )
        if not isinstance(detail.payload, Mapping):
            raise ValueError(f"external USGS detail is invalid: {event_dir}")
        match = match_usgs_event(
            local,
            [detail.payload],
            explicit_event_id=event_id,
            explicit_mapping_id=f"external-event-json:{event_dir}",
            explicit_review_disposition="authoritative_event_json_id_verified",
        )
        selected = extract_usgs_magnitude(detail.payload)
        if selected is None:
            raise ValueError(
                f"external event has no usable USGS scalar magnitude: {event_dir}"
            )
        rows.append(
            {
                "event": _external_display_name(
                    event_name=str(metadata.get("event", event_dir)),
                    event_dir_name=event_dir,
                    magnitude=original_mw,
                ),
                "event_dir": event_dir,
                "usgs_event_id": event_id,
                "mw_original": original_mw,
                "mw_selected": selected.mw,
                "mw_source": selected.source,
                "mw_source_rank": selected.source_rank,
                "mw_type": selected.magnitude_type,
                "product_id": selected.product_id,
                "contributor": selected.contributor,
                "source_update_time": selected.source_update_time,
                "scalar_moment_nm": _float_or_none(
                    selected.scalar_moment_nm
                ),
                "detail_url": detail.url,
                "detail_sha256": detail.sha256,
                "match_time_delta_sec": match.time_delta_sec,
                "match_distance_km": match.distance_km,
                "match_magnitude_delta": match.magnitude_delta,
                "review_disposition": match.review_disposition,
            }
        )
    return rows


def _artifact_hashes(
    *,
    source_npz: Path,
    gcmt_csv: Path,
    stf_dir: Path,
    accepted_manifest: Path,
    mapping_table: Path | None,
    snapshot_root: Path,
    staged_files: Sequence[Path],
) -> dict[str, Any]:
    inputs: dict[str, str] = {
        str(source_npz): _sha256_file(source_npz),
        str(gcmt_csv): _sha256_file(gcmt_csv),
        str(accepted_manifest): _sha256_file(accepted_manifest),
    }
    if mapping_table is not None and mapping_table.is_file():
        inputs[str(mapping_table)] = _sha256_file(mapping_table)
    for path in sorted(stf_dir.glob("*.stf")):
        inputs[str(path)] = _sha256_file(path)
    raw_responses = {
        str(path.relative_to(snapshot_root)): _sha256_file(path)
        for path in sorted((snapshot_root / "raw_usgs").rglob("*.json"))
    }
    outputs = {path.name: _sha256_file(path) for path in staged_files}
    return {
        "inputs": inputs,
        "raw_usgs": raw_responses,
        "outputs": outputs,
    }


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    accepted_events: set[str],
    candidate_sha256: str,
    verified_existing: bool,
) -> dict[str, Any]:
    return {
        "event_count": len(rows),
        "accepted_event_count": len(accepted_events),
        "source_counts": dict(
            sorted(Counter(str(row["mw_source"]) for row in rows).items())
        ),
        "candidate_sha256": candidate_sha256,
        "verified_existing": verified_existing,
    }


def build_snapshot(
    *,
    source_npz: str | Path,
    gcmt_csv: str | Path,
    stf_dir: str | Path,
    accepted_manifest: str | Path,
    external_event_root: str | Path | None,
    snapshot_root: str | Path,
    mapping_table: str | Path | None,
    offline: bool,
    verify_existing: bool,
    fetch_bytes: Callable[[str], bytes] = _http_fetch_bytes,
) -> dict[str, Any]:
    source_path = Path(source_npz).resolve()
    gcmt_path = Path(gcmt_csv).resolve()
    stf_path = Path(stf_dir).resolve()
    accepted_path = Path(accepted_manifest).resolve()
    external_root = (
        Path(external_event_root).resolve()
        if external_event_root is not None
        else None
    )
    snapshot = Path(snapshot_root).resolve()
    snapshot.mkdir(parents=True, exist_ok=True)
    default_mapping = snapshot / "usgs_event_map.v1.csv"
    mapping_path = (
        Path(mapping_table).resolve()
        if mapping_table is not None
        else (default_mapping if default_mapping.is_file() else None)
    )
    for required in (source_path, gcmt_path, accepted_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not stf_path.is_dir():
        raise FileNotFoundError(stf_path)
    if verify_existing and not offline:
        raise ValueError("existing snapshot verification must be offline")

    events = _load_local_events(source_path)
    accepted_events = _load_accepted_events(accepted_path)
    mappings = _load_mapping_table(mapping_path)
    source_hash = _sha256_file(source_path)
    gcmt_hash = _sha256_file(gcmt_path)
    rows, review_queue = _resolve_internal_rows(
        events=events,
        snapshot_root=snapshot,
        gcmt_csv_sha256=gcmt_hash,
        source_npz_sha256=source_hash,
        stf_dir=stf_path,
        mappings=mappings,
        offline=offline,
        fetch_bytes=fetch_bytes,
    )
    if review_queue:
        _write_csv(snapshot / "review_queue.csv", review_queue, REVIEW_FIELDS)
        raise PublicationBlocked(
            f"review queue contains {len(review_queue)} unresolved rows"
        )
    validate_publication_rows(
        rows,
        expected_events=[event.event for event in events],
        accepted_events=accepted_events,
    )
    external_rows = _resolve_external_rows(
        external_event_root=external_root,
        snapshot_root=snapshot,
        offline=offline,
        fetch_bytes=fetch_bytes,
    )

    label_json = _json_bytes(rows)
    external_json = _json_bytes(external_rows)
    candidate_name = "gnss_events_matched.usgs_priority.npz"
    final_candidate = snapshot / candidate_name
    final_labels_json = snapshot / "magnitude_labels.json"
    final_external_json = snapshot / "external_magnitude_labels.json"

    if verify_existing:
        for required in (
            final_candidate,
            final_labels_json,
            final_external_json,
            snapshot / "artifact_hashes.json",
            snapshot / "COMPLETE",
        ):
            if not required.is_file():
                raise FileNotFoundError(required)
        if final_labels_json.read_bytes() != label_json:
            raise ValueError("offline replay label manifest differs")
        if final_external_json.read_bytes() != external_json:
            raise ValueError("offline replay external manifest differs")
        with tempfile.TemporaryDirectory(
            dir=snapshot,
            prefix=".replay-",
        ) as temporary_dir:
            replay_candidate = Path(temporary_dir) / candidate_name
            replay_hash = write_candidate_npz(
                source_path,
                replay_candidate,
                rows,
            )
        candidate_hash = _sha256_file(final_candidate)
        if replay_hash != candidate_hash:
            raise ValueError("offline replay candidate hash differs")
        validate_candidate_npz(source_path, final_candidate)
        return _summary(
            rows,
            accepted_events=accepted_events,
            candidate_sha256=candidate_hash,
            verified_existing=True,
        )

    final_names = (
        candidate_name,
        "magnitude_labels.csv",
        "magnitude_labels.json",
        "external_magnitude_labels.csv",
        "external_magnitude_labels.json",
        "source_manifest.json",
        "artifact_hashes.json",
        "COMPLETE",
    )
    existing = [name for name in final_names if (snapshot / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite snapshot outputs: {existing}")

    with tempfile.TemporaryDirectory(
        dir=snapshot,
        prefix=".publish-",
    ) as temporary_dir:
        stage = Path(temporary_dir)
        staged_candidate = stage / candidate_name
        candidate_hash = write_candidate_npz(
            source_path,
            staged_candidate,
            rows,
        )
        staged_labels_csv = stage / "magnitude_labels.csv"
        staged_labels_json = stage / "magnitude_labels.json"
        staged_external_csv = stage / "external_magnitude_labels.csv"
        staged_external_json = stage / "external_magnitude_labels.json"
        staged_source_manifest = stage / "source_manifest.json"
        staged_hashes = stage / "artifact_hashes.json"
        _atomic_write_bytes(staged_labels_csv, _csv_bytes(rows, LABEL_FIELDS))
        _atomic_write_bytes(staged_labels_json, label_json)
        _atomic_write_bytes(
            staged_external_csv,
            _csv_bytes(external_rows, EXTERNAL_LABEL_FIELDS),
        )
        _atomic_write_bytes(staged_external_json, external_json)
        source_manifest = {
            "source_npz": {"path": str(source_path), "sha256": source_hash},
            "gcmt_csv": {"path": str(gcmt_path), "sha256": gcmt_hash},
            "accepted_manifest": {
                "path": str(accepted_path),
                "sha256": _sha256_file(accepted_path),
            },
            "stf_dir": str(stf_path),
            "mapping_table": (
                {
                    "path": str(mapping_path),
                    "sha256": _sha256_file(mapping_path),
                }
                if mapping_path is not None
                else None
            ),
        }
        _atomic_write_bytes(staged_source_manifest, _json_bytes(source_manifest))
        staged_outputs = (
            staged_candidate,
            staged_labels_csv,
            staged_labels_json,
            staged_external_csv,
            staged_external_json,
            staged_source_manifest,
        )
        hashes = _artifact_hashes(
            source_npz=source_path,
            gcmt_csv=gcmt_path,
            stf_dir=stf_path,
            accepted_manifest=accepted_path,
            mapping_table=mapping_path,
            snapshot_root=snapshot,
            staged_files=staged_outputs,
        )
        _atomic_write_bytes(staged_hashes, _json_bytes(hashes))
        for staged in (*staged_outputs, staged_hashes):
            os.replace(staged, snapshot / staged.name)
        _atomic_write_bytes(
            snapshot / "COMPLETE",
            f"candidate_sha256={candidate_hash}\n".encode("ascii"),
        )
    validate_candidate_npz(source_path, final_candidate)
    return _summary(
        rows,
        accepted_events=accepted_events,
        candidate_sha256=candidate_hash,
        verified_existing=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an audited USGS-priority magnitude snapshot",
    )
    parser.add_argument("--source-npz", required=True, type=Path)
    parser.add_argument("--gcmt-csv", required=True, type=Path)
    parser.add_argument("--stf-dir", required=True, type=Path)
    parser.add_argument("--accepted-manifest", required=True, type=Path)
    parser.add_argument("--external-event-root", type=Path)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--mapping-table", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_snapshot(
        source_npz=args.source_npz,
        gcmt_csv=args.gcmt_csv,
        stf_dir=args.stf_dir,
        accepted_manifest=args.accepted_manifest,
        external_event_root=args.external_event_root,
        snapshot_root=args.snapshot_root,
        mapping_table=args.mapping_table,
        offline=args.offline,
        verify_existing=args.verify_existing,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
