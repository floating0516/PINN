from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import io
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np

from src.data.stf import moment_to_mw


MW_FAMILY = frozenset({"mw", "mww", "mwc", "mwr", "mwb"})
MAX_TIME_DELTA_SEC = 30.0
MAX_DISTANCE_KM = 100.0
MAX_MAGNITUDE_DELTA = 0.5
WARNING_DELTA_MW = 0.1
REVIEW_DELTA_MW = 0.2
_THRESHOLD_EPSILON = 1.0e-12
CANDIDATE_ADDED_ARRAYS = (
    "magnitude_selected",
    "magnitude_usgs",
    "magnitude_gcmt",
    "magnitude_stf_native",
    "magnitude_source",
    "magnitude_source_rank",
    "magnitude_type",
    "usgs_event_id",
    "usgs_product_id",
)


class MagnitudeRelabelError(ValueError):
    pass


class NoUSGSMatch(MagnitudeRelabelError):
    pass


class AmbiguousUSGSMatch(MagnitudeRelabelError):
    pass


class MissingMagnitudeSource(MagnitudeRelabelError):
    pass


@dataclass(frozen=True)
class LocalEvent:
    event_index: int
    event: str
    origin_time: str
    latitude: float
    longitude: float
    gcmt_mw: float


@dataclass(frozen=True)
class USGSMatch:
    usgs_event_id: str
    detail_url: str
    time_delta_sec: float
    distance_km: float
    magnitude_delta: float
    explicit_mapping_id: str
    review_disposition: str


@dataclass(frozen=True)
class USGSMagnitude:
    mw: float
    source: str
    source_rank: int
    magnitude_type: str
    usgs_event_id: str
    product_id: str = ""
    contributor: str = ""
    source_update_time: str = ""
    scalar_moment_nm: float = float("nan")


@dataclass(frozen=True)
class MagnitudeResolution:
    mw_selected: float
    source: str
    source_rank: int
    magnitude_type: str
    usgs_event_id: str = ""
    product_id: str = ""
    contributor: str = ""
    source_update_time: str = ""
    scalar_moment_nm: float = float("nan")


@dataclass(frozen=True)
class ConflictAssessment:
    differences: dict[str, float]
    warning: bool
    review_required: bool


@dataclass(frozen=True)
class NativeSTFMagnitude:
    mw: float
    scalar_moment_nm: float
    sha256: str
    traceable_header: bool
    header_mw: float
    header_scalar_moment_nm: float
    warning: str


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _origin_datetime(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _haversine_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    lat_a = math.radians(latitude_a)
    lat_b = math.radians(latitude_b)
    delta_lat = lat_b - lat_a
    delta_lon = math.radians(longitude_b - longitude_a)
    haversine = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * 6371.0088 * math.asin(min(1.0, math.sqrt(haversine)))


def _candidate_evidence(
    local_event: LocalEvent,
    feature: Mapping[str, Any],
) -> USGSMatch | None:
    event_id = str(feature.get("id", "")).strip()
    properties = feature.get("properties")
    geometry = feature.get("geometry")
    if (
        not event_id
        or not isinstance(properties, Mapping)
        or not isinstance(geometry, Mapping)
    ):
        return None
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, Sequence) or len(coordinates) < 2:
        return None
    epoch_ms = _finite_float(properties.get("time"))
    magnitude = _finite_float(properties.get("mag"))
    longitude = _finite_float(coordinates[0])
    latitude = _finite_float(coordinates[1])
    local_magnitude = _finite_float(local_event.gcmt_mw)
    if None in (epoch_ms, magnitude, longitude, latitude, local_magnitude):
        return None
    event_time = datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
    time_delta = abs((event_time - _origin_datetime(local_event.origin_time)).total_seconds())
    distance = _haversine_km(
        float(local_event.latitude),
        float(local_event.longitude),
        latitude,
        longitude,
    )
    magnitude_delta = abs(magnitude - local_magnitude)
    return USGSMatch(
        usgs_event_id=event_id,
        detail_url=str(properties.get("detail", "")),
        time_delta_sec=time_delta,
        distance_km=distance,
        magnitude_delta=magnitude_delta,
        explicit_mapping_id="",
        review_disposition="",
    )


def _within_automatic_match_thresholds(evidence: USGSMatch) -> bool:
    return (
        evidence.time_delta_sec <= MAX_TIME_DELTA_SEC + _THRESHOLD_EPSILON
        and evidence.distance_km <= MAX_DISTANCE_KM + _THRESHOLD_EPSILON
        and evidence.magnitude_delta
        <= MAX_MAGNITUDE_DELTA + _THRESHOLD_EPSILON
    )


def match_usgs_event(
    local_event: LocalEvent,
    features: Sequence[Mapping[str, Any]],
    *,
    explicit_event_id: str = "",
    explicit_mapping_id: str = "",
    explicit_review_disposition: str = "",
) -> USGSMatch:
    if explicit_event_id:
        selected = [
            feature
            for feature in features
            if str(feature.get("id", "")) == explicit_event_id
        ]
        if len(selected) != 1:
            raise NoUSGSMatch(
                f"explicit USGS event {explicit_event_id!r} is not present exactly once"
            )
        evidence = _candidate_evidence(local_event, selected[0])
        if evidence is None:
            raise NoUSGSMatch(
                f"explicit USGS event {explicit_event_id!r} lacks match evidence"
            )
        if not explicit_mapping_id:
            raise MagnitudeRelabelError("explicit mapping requires a mapping ID")
        if not explicit_review_disposition:
            raise MagnitudeRelabelError(
                "explicit mapping requires a review disposition"
            )
        return replace(
            evidence,
            explicit_mapping_id=explicit_mapping_id,
            review_disposition=explicit_review_disposition,
        )

    acceptable = [
        evidence
        for feature in features
        if (evidence := _candidate_evidence(local_event, feature)) is not None
        and _within_automatic_match_thresholds(evidence)
    ]
    if not acceptable:
        raise NoUSGSMatch(f"no acceptable USGS match for {local_event.event}")
    if len(acceptable) != 1:
        raise AmbiguousUSGSMatch(
            f"{len(acceptable)} acceptable USGS matches for {local_event.event}"
        )
    return acceptable[0]


def _timestamp_text(value: Any) -> str:
    epoch_ms = _finite_float(value)
    if epoch_ms is None:
        return ""
    return (
        datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _moment_tensor_products(properties: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    products = properties.get("products")
    if not isinstance(products, Mapping):
        return []
    values = products.get("moment-tensor")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    result = [value for value in values if isinstance(value, Mapping)]
    return sorted(
        result,
        key=lambda item: (
            -(_finite_float(item.get("preferredWeight")) or 0.0),
            -(_finite_float(item.get("updateTime")) or 0.0),
            str(item.get("code", "")),
        ),
    )


def extract_usgs_magnitude(
    detail: Mapping[str, Any],
) -> USGSMagnitude | None:
    event_id = str(detail.get("id", "")).strip()
    properties = detail.get("properties")
    if not isinstance(properties, Mapping):
        return None
    preferred_mw = _finite_float(properties.get("mag"))
    preferred_type = str(properties.get("magType", "")).strip()
    if preferred_mw is not None and preferred_type.casefold() in MW_FAMILY:
        return USGSMagnitude(
            mw=preferred_mw,
            source="usgs_preferred",
            source_rank=1,
            magnitude_type=preferred_type,
            usgs_event_id=event_id,
            contributor=str(properties.get("net", "")),
            source_update_time=_timestamp_text(properties.get("updated")),
        )

    for product in _moment_tensor_products(properties):
        product_properties = product.get("properties")
        if not isinstance(product_properties, Mapping):
            continue
        moment = None
        for key in ("scalar-moment", "scalar_moment", "scalarMoment"):
            moment = _finite_float(product_properties.get(key))
            if moment is not None:
                break
        if moment is None or moment <= 0.0:
            continue
        return USGSMagnitude(
            mw=moment_to_mw(moment),
            source="usgs_moment_tensor",
            source_rank=2,
            magnitude_type="Mw(moment_tensor)",
            usgs_event_id=event_id,
            product_id=str(product.get("code", "")),
            contributor=str(
                product.get("source", product_properties.get("eventsource", ""))
            ),
            source_update_time=_timestamp_text(product.get("updateTime")),
            scalar_moment_nm=moment,
        )
    return None


def resolve_magnitude(
    *,
    usgs_detail: Mapping[str, Any] | None,
    gcmt_mw: Any,
    stf_native_mw: Any,
) -> MagnitudeResolution:
    usgs = extract_usgs_magnitude(usgs_detail) if usgs_detail is not None else None
    if usgs is not None:
        return MagnitudeResolution(
            mw_selected=usgs.mw,
            source=usgs.source,
            source_rank=usgs.source_rank,
            magnitude_type=usgs.magnitude_type,
            usgs_event_id=usgs.usgs_event_id,
            product_id=usgs.product_id,
            contributor=usgs.contributor,
            source_update_time=usgs.source_update_time,
            scalar_moment_nm=usgs.scalar_moment_nm,
        )
    gcmt = _finite_float(gcmt_mw)
    if gcmt is not None:
        return MagnitudeResolution(
            mw_selected=gcmt,
            source="gcmt",
            source_rank=3,
            magnitude_type="Mw",
        )
    stf = _finite_float(stf_native_mw)
    if stf is not None:
        return MagnitudeResolution(
            mw_selected=stf,
            source="scardec_native_stf",
            source_rank=4,
            magnitude_type="Mw(STF_integral)",
        )
    raise MissingMagnitudeSource("no finite scalar magnitude source is available")


def assess_source_conflicts(values: Mapping[str, Any]) -> ConflictAssessment:
    finite = [
        (str(name), value_float)
        for name, value in values.items()
        if (value_float := _finite_float(value)) is not None
    ]
    differences: dict[str, float] = {}
    for left_index, (left_name, left_value) in enumerate(finite):
        for right_name, right_value in finite[left_index + 1 :]:
            differences[f"{left_name}_vs_{right_name}"] = abs(
                left_value - right_value
            )
    return ConflictAssessment(
        differences=differences,
        warning=any(
            value > WARNING_DELTA_MW + _THRESHOLD_EPSILON
            for value in differences.values()
        ),
        review_required=any(
            value >= REVIEW_DELTA_MW - _THRESHOLD_EPSILON
            for value in differences.values()
        ),
    )


def _source_header(lines: Sequence[str]) -> tuple[bool, float, float]:
    if len(lines) < 2:
        return False, float("nan"), float("nan")
    origin_tokens = lines[0].replace("D", "E").split()
    source_tokens = lines[1].replace("D", "E").split()
    if len(origin_tokens) < 8 or len(source_tokens) < 3:
        return False, float("nan"), float("nan")
    try:
        datetime(
            int(origin_tokens[0]),
            int(origin_tokens[1]),
            int(origin_tokens[2]),
            int(origin_tokens[3]),
            int(origin_tokens[4]),
            int(float(origin_tokens[5])),
        )
        header_moment = float(source_tokens[1])
        header_mw = float(source_tokens[2])
    except (TypeError, ValueError):
        return False, float("nan"), float("nan")
    traceable = (
        math.isfinite(header_moment)
        and header_moment > 0.0
        and math.isfinite(header_mw)
    )
    return traceable, header_moment, header_mw


def read_native_stf(path: str | Path) -> NativeSTFMagnitude:
    source = Path(path)
    raw = source.read_bytes()
    lines = raw.decode("utf-8", errors="replace").splitlines()
    rows: list[tuple[float, float]] = []
    for line in lines:
        tokens = line.replace("D", "E").split()
        if len(tokens) != 2:
            continue
        try:
            time_sec, rate = map(float, tokens)
        except ValueError:
            continue
        if math.isfinite(time_sec) and math.isfinite(rate) and time_sec >= 0.0:
            rows.append((time_sec, max(0.0, rate)))
    if len(rows) < 2:
        raise MagnitudeRelabelError(f"STF has fewer than two causal rows: {source}")
    values = np.asarray(rows, dtype=np.float64)
    order = np.argsort(values[:, 0], kind="mergesort")
    sorted_time = values[order, 0]
    sorted_rate = values[order, 1]
    unique_time, inverse = np.unique(sorted_time, return_inverse=True)
    if unique_time.size < 2:
        raise MagnitudeRelabelError(
            f"STF has fewer than two unique causal timestamps: {source}"
        )
    rate = np.bincount(inverse, weights=sorted_rate) / np.bincount(inverse)
    moment = float(np.trapezoid(rate, unique_time))
    if not math.isfinite(moment) or moment <= 0.0:
        raise MagnitudeRelabelError(f"STF moment is not positive and finite: {source}")
    traceable, header_moment, header_mw = _source_header(lines)
    return NativeSTFMagnitude(
        mw=moment_to_mw(moment),
        scalar_moment_nm=moment,
        sha256=hashlib.sha256(raw).hexdigest(),
        traceable_header=traceable,
        header_mw=header_mw,
        header_scalar_moment_nm=header_moment,
        warning="" if traceable else "missing_source_header",
    )


def _event_text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="strict")
    return str(value)


def _float_array(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = [
        (
            float(value)
            if (value := _finite_float(row.get(key))) is not None
            else math.nan
        )
        for row in rows
    ]
    return np.asarray(values, dtype=np.float64)


def _deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate NPZ: {path}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name, array in arrays.items():
                payload = io.BytesIO()
                np.save(payload, np.asarray(array), allow_pickle=True)
                info = zipfile.ZipInfo(
                    filename=f"{name}.npy",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(
                    info,
                    payload.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_candidate_npz(
    source_path: str | Path,
    destination_path: str | Path,
    label_rows: Sequence[Mapping[str, Any]],
) -> str:
    source = Path(source_path)
    destination = Path(destination_path)
    with np.load(source, allow_pickle=True) as data:
        source_names = list(data.files)
        source_arrays = {name: data[name] for name in source_names}
    if "events" not in source_arrays or "magnitude" not in source_arrays:
        raise MagnitudeRelabelError("source NPZ lacks events or magnitude")
    events = source_arrays["events"]
    if len(label_rows) != len(events):
        raise MagnitudeRelabelError(
            f"label row count {len(label_rows)} does not match {len(events)} events"
        )
    for index, (event_value, row) in enumerate(zip(events, label_rows, strict=True)):
        if int(row.get("event_index", -1)) != index:
            raise MagnitudeRelabelError(f"label event index mismatch at {index}")
        if str(row.get("event", "")) != _event_text(event_value):
            raise MagnitudeRelabelError(f"label event name mismatch at {index}")
    selected = _float_array(label_rows, "mw_selected")
    if not np.all(np.isfinite(selected)):
        raise MissingMagnitudeSource("candidate contains a non-finite selected label")
    source_magnitude = np.asarray(source_arrays["magnitude"])
    gcmt = _float_array(label_rows, "mw_gcmt")
    if not np.array_equal(gcmt, source_magnitude.astype(np.float64), equal_nan=True):
        raise MagnitudeRelabelError("manifest GCMT labels differ from source magnitude")
    selected = selected.astype(source_magnitude.dtype, copy=False)
    additions = {
        "magnitude_selected": selected,
        "magnitude_usgs": _float_array(label_rows, "mw_usgs"),
        "magnitude_gcmt": gcmt,
        "magnitude_stf_native": _float_array(label_rows, "mw_stf_native"),
        "magnitude_source": np.asarray(
            [str(row.get("mw_source", "")) for row in label_rows],
            dtype=np.str_,
        ),
        "magnitude_source_rank": np.asarray(
            [int(row.get("mw_source_rank", 0)) for row in label_rows],
            dtype=np.int16,
        ),
        "magnitude_type": np.asarray(
            [str(row.get("mw_type", "")) for row in label_rows],
            dtype=np.str_,
        ),
        "usgs_event_id": np.asarray(
            [str(row.get("usgs_event_id", "")) for row in label_rows],
            dtype=np.str_,
        ),
        "usgs_product_id": np.asarray(
            [str(row.get("product_id", "")) for row in label_rows],
            dtype=np.str_,
        ),
    }
    arrays: dict[str, np.ndarray] = {}
    for name in source_names:
        arrays[name] = selected if name == "magnitude" else source_arrays[name]
    arrays.update(additions)
    _deterministic_npz(destination, arrays)
    validate_candidate_npz(source, destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def _deep_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        if left.dtype == object:
            return all(
                _deep_equal(left_value, right_value)
                for left_value, right_value in zip(
                    left.flat,
                    right.flat,
                    strict=True,
                )
            )
        return bool(np.array_equal(left, right, equal_nan=True))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return left.keys() == right.keys() and all(
            _deep_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(
            _deep_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    try:
        if math.isnan(float(left)) and math.isnan(float(right)):
            return True
    except (TypeError, ValueError):
        pass
    return bool(left == right)


def validate_candidate_npz(
    source_path: str | Path,
    candidate_path: str | Path,
) -> None:
    with np.load(source_path, allow_pickle=True) as source, np.load(
        candidate_path,
        allow_pickle=True,
    ) as candidate:
        expected_names = [*source.files, *CANDIDATE_ADDED_ARRAYS]
        if candidate.files != expected_names:
            raise MagnitudeRelabelError(
                f"candidate arrays differ: expected={expected_names}, "
                f"actual={candidate.files}"
            )
        for name in source.files:
            if name != "magnitude" and not _deep_equal(source[name], candidate[name]):
                raise MagnitudeRelabelError(f"non-label array changed: {name}")
        event_count = len(source["events"])
        for name in ("magnitude", *CANDIDATE_ADDED_ARRAYS):
            if len(candidate[name]) != event_count:
                raise MagnitudeRelabelError(
                    f"candidate label array has wrong length: {name}"
                )
        if not np.array_equal(
            candidate["magnitude"],
            candidate["magnitude_selected"],
            equal_nan=True,
        ):
            raise MagnitudeRelabelError(
                "candidate magnitude differs from magnitude_selected"
            )
        if not np.all(np.isfinite(candidate["magnitude_selected"])):
            raise MissingMagnitudeSource("candidate selected labels are non-finite")
        if not np.array_equal(
            candidate["magnitude_gcmt"],
            source["magnitude"],
            equal_nan=True,
        ):
            raise MagnitudeRelabelError(
                "candidate magnitude_gcmt differs from source magnitude"
            )
        if any(not str(value) for value in candidate["magnitude_source"]):
            raise MagnitudeRelabelError("candidate contains an empty magnitude source")
        ranks = np.asarray(candidate["magnitude_source_rank"], dtype=np.int64)
        if not bool(np.all((ranks >= 1) & (ranks <= 4))):
            raise MagnitudeRelabelError("candidate contains an invalid source rank")
