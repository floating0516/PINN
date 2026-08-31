from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from typing import Any, Iterable
import zipfile

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_v2 import CorrectedEarthquakeDataset
from src.data.external_records import record_from_external_bundle
from src.data.geometry import compute_source_station_geometry
from src.data.manifest import audit_passes, write_dataset_audit
from src.data.sample_builder import SampleRejected, build_station_sample
from src.data.waveform import waveform_config_from_v2
from src.evaluation.evaluate_unseen import (
    EventBundle,
    StationWaveform,
    _phase_preserving_external_waveform_config,
    load_event_bundle,
)


PROJECT_DATA_ROOT = Path("/home/lihe/PINN_Mag/data")
OLD_NPZ = PROJECT_DATA_ROOT / (
    "magnitude-label-snapshots/"
    "usgs-priority-20260723T044422Z-40d808a/"
    "gnss_events_matched.usgs_priority.npz"
)
OLD_MANIFEST = Path(
    "/home/lihe/PINN_Mag/runs/"
    "phase39-manuscript-stf-glehman-scalar-20260726T111256Z-7fbb7a7/"
    "preflight/dataset_manifest.csv"
)
BASE_CONFIG = Path(
    "/home/lihe/PINN_Mag/worktrees/run-phase39-confirmatory-121197d/"
    "configs/experiments/manuscript_station_stf_usgs_glehman_scalar.yaml"
)
STF_ROOT = PROJECT_DATA_ROOT / "STF_SCARDEC"
SCARDEC_DB = STF_ROOT / (
    ".cache/SCARDEC_STF_DB/"
    "ALL_MOY_and_OPTI_2025_MAJ_till_31122023"
)
NORMALIZED_ROOT = Path(
    "/home/lihe/projects/gnss-earthscope-pipeline/exports/"
    "normalized-ok-stations-us-nz"
)
PAPER_REVIEW_ROOT = Path(
    "/home/lihe/projects/gnss-earthscope-pipeline/exports/"
    "paper-reruns-review-20260806/normalized"
)
NOTO_ARCHIVE = Path("/home/lihe/openclaw-normalized-backup.tar.gz")
NOTO_ARCHIVE_PREFIX = (
    "openclaw-gnss-collector-agent/data/gnss_data/normalized/"
    "noto-2024-japan"
)

EXPECTED_OLD_HASH = (
    "2e1fa4c12fc1eb03ffc8bf9235491f0886d4b0d360ebcb3486baca4d948cfd6a"
)
EXPECTED_EXISTING_EVENTS = 31
EXPECTED_EXISTING_ACCEPTED = 2558
EXPECTED_NEW_EVENTS = 8
EXPECTED_NEW_EVENT_ACCEPTED = 64
EXPECTED_SAME_EVENT_ADDITIONS = 50
EXPECTED_NOTO_ADDITIONS = 22
EXPECTED_ACCEPTED_EVENTS = 39
EXPECTED_ACCEPTED_STATIONS = 2694
MAX_EPICENTRAL_DISTANCE_KM = 800.0
NOTO_ALIAS_DISTANCE_KM = 1.0

EXTERNAL_UNSEEN_DIRECTORY_NAMES = {
    "iquique-aftershock-2014-chile",
    "kodiak-2018-alaska",
    "luding-2022-china",
    "myanmar-2025-mandalay",
    "nepal-aftershock-2015",
    "samos-2020-greece",
    "sand-point-2025-alaska",
    "xizang-2025-southern-tibetan-plateau",
}


@dataclass(frozen=True)
class NewEventSpec:
    event_id: str
    package_dir: Path
    scardec_dir: str


@dataclass(frozen=True)
class ExistingAdditionSpec:
    event_name: str
    package_dir: Path
    expected_additions: int


@dataclass(frozen=True)
class ScardecHeader:
    origin_time: str
    latitude: float
    longitude: float
    depth_km: float
    moment_nm: float
    magnitude: float
    strike: float
    dip: float
    rake: float


NEW_EVENT_SPECS = (
    NewEventSpec(
        "us7000i9bw",
        NORMALIZED_ROOT
        / "us-us7000i9bw-m7-6-20220919-35-km-ssw-of-aguililla-mexico",
        "FCTs_20220919_180508_MICHOACAN__MEXICO",
    ),
    NewEventSpec(
        "2013p613797",
        NORMALIZED_ROOT
        / "nz-2013p613797-m6-5-20130816-5-km-south-east-of-seddon",
        "FCTs_20130816_023105_COOK_STRAIT__NEW_ZEALAND",
    ),
    NewEventSpec(
        "2021p169083",
        NORMALIZED_ROOT
        / "nz-2021p169083-m7-2-20210304-140-km-east-of-te-araroa",
        "FCTs_20210304_132734_OFF_E._COAST_OF_N._ISLAND__N.Z",
    ),
    NewEventSpec(
        "nc73821036",
        NORMALIZED_ROOT
        / "us-nc73821036-m6-4-20221220-15km-wsw-of-ferndale-ca",
        "FCTs_20221220_103424_NEAR_COAST_OF_NORTHERN_CALIF",
    ),
    NewEventSpec(
        "2013p543824",
        NORMALIZED_ROOT
        / "nz-2013p543824-m6-5-20130721-20-km-east-of-seddon",
        "FCTs_20130721_050931_COOK_STRAIT__NEW_ZEALAND",
    ),
    NewEventSpec(
        "2016p661332",
        NORMALIZED_ROOT
        / "nz-2016p661332-m7-1-20160901-120-km-north-east-of-te-araroa",
        "FCTs_20160901_163757_OFF_E._COAST_OF_N._ISLAND__N.Z",
    ),
    NewEventSpec(
        "us6000ah9t",
        NORMALIZED_ROOT
        / "us-us6000ah9t-m7-4-20200623-9-km-se-of-santa-mar-a-xadani-mexico",
        "FCTs_20200623_152904_NEAR_COAST_OF_OAXACA__MEXICO",
    ),
    NewEventSpec(
        "ak014cbigci8",
        NORMALIZED_ROOT
        / "us-ak014cbigci8-m6-2-20140925-22-km-wsw-of-skwentna-alaska",
        "FCTs_20140925_175117_SOUTHERN_ALASKA",
    ),
)

EXISTING_ADDITION_SPECS = (
    ExistingAdditionSpec(
        "Kaikoura2016",
        NORMALIZED_ROOT
        / "nz-2016p858000-m7-8-20161113-15-km-north-east-of-culverden",
        5,
    ),
    ExistingAdditionSpec(
        "Ridgecrest2019",
        PAPER_REVIEW_ROOT
        / "us-ci38457511-m7-1-20190706-ridgecrest-earthquake-sequence",
        29,
    ),
    ExistingAdditionSpec(
        "Napa2014",
        PAPER_REVIEW_ROOT
        / "us-nc72282711-m6-02-20140824-the-2014-south-napa-california-earthquake",
        8,
    ),
    ExistingAdditionSpec(
        "Eureka2014",
        PAPER_REVIEW_ROOT
        / "us-nc72182046-m6-8-20140310-77-km-wnw-of-indianola-california",
        7,
    ),
    ExistingAdditionSpec(
        "Puebla2017",
        NORMALIZED_ROOT
        / "us-us2000ar20-m7-1-20170919-1-km-s-of-matzaco-mexico",
        1,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_station_code(value: str) -> str:
    code = str(value).strip()
    if code.lower().endswith(".gnss"):
        code = code[:-5]
    return code


def station_identity(value: str) -> str:
    return normalize_station_code(value).casefold()


def normalize_identity(value: str) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def mechanism_from_rake(rake: float) -> str:
    normalized = ((float(rake) + 180.0) % 360.0) - 180.0
    if abs(normalized) <= 30.0 or abs(normalized) >= 150.0:
        return "Strike slip"
    if 30.0 <= normalized <= 150.0:
        return "Reverse"
    return "Normal"


def parse_scardec_header(path: Path) -> ScardecHeader:
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        first = stream.readline().split()
        second = stream.readline().replace("D", "E").split()
    if len(first) < 8 or len(second) < 9:
        raise ValueError(f"invalid SCARDEC header: {path}")
    year, month, day, hour, minute = (int(first[index]) for index in range(5))
    second_value = float(first[5])
    whole_second = int(second_value)
    microsecond = int(round((second_value - whole_second) * 1_000_000))
    origin = datetime(
        year,
        month,
        day,
        hour,
        minute,
        whole_second,
        microsecond,
        tzinfo=timezone.utc,
    )
    return ScardecHeader(
        origin_time=origin.isoformat().replace("+00:00", "Z"),
        latitude=float(first[6]),
        longitude=float(first[7]),
        depth_km=float(second[0]),
        moment_nm=float(second[1]),
        magnitude=float(second[2]),
        strike=float(second[3]),
        dip=float(second[4]),
        rake=float(second[5]),
    )


def find_scardec_source(directory_name: str) -> Path:
    directory = SCARDEC_DB / directory_name
    matches = sorted(directory.glob("fctoptsource*"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one fctoptsource file in {directory}, found {len(matches)}"
        )
    return matches[0]


def deterministic_savez_compressed(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.asanyarray(arrays[key]),
                allow_pickle=True,
            )
            info = zipfile.ZipInfo(f"{key}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def _read_old_accepted() -> tuple[list[str], dict[str, set[str]]]:
    accepted: dict[str, set[str]] = {}
    with OLD_MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row["accepted"]).casefold() != "true":
                continue
            accepted.setdefault(str(row["event"]), set()).add(
                station_identity(row["station"])
            )
    count = sum(len(stations) for stations in accepted.values())
    if len(accepted) != EXPECTED_EXISTING_EVENTS or count != EXPECTED_EXISTING_ACCEPTED:
        raise ValueError(
            "old accepted manifest changed: "
            f"events={len(accepted)}, stations={count}"
        )
    return sorted(accepted), accepted


def _copy_station_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: np.asarray(value).copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
        for key, value in payload.items()
    }


def _event_metadata_from_old(data: Any, index: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in (
        "events",
        "country",
        "longitude",
        "latitude",
        "depth_km",
        "magnitude",
        "mechanism",
        "strike",
        "dip",
        "rake",
        "origin_time",
        "magnitude_selected",
        "magnitude_usgs",
        "magnitude_gcmt",
        "magnitude_stf_native",
        "magnitude_source",
        "magnitude_source_rank",
        "magnitude_type",
        "usgs_event_id",
        "usgs_product_id",
    ):
        metadata[key] = copy.deepcopy(data[key][index])
    return metadata


def _bundle_with_existing_metadata(
    bundle: EventBundle,
    metadata: dict[str, Any],
) -> EventBundle:
    return replace(
        bundle,
        event_name=str(metadata["events"]),
        magnitude=float(metadata["magnitude"]),
        latitude=float(metadata["latitude"]),
        longitude=float(metadata["longitude"]),
        depth_km=float(metadata["depth_km"]),
        mechanism=str(metadata["mechanism"]),
        strike=float(metadata["strike"]),
        dip=float(metadata["dip"]),
        rake=float(metadata["rake"]),
    )


def _bundle_with_scardec_metadata(
    bundle: EventBundle,
    event_id: str,
    header: ScardecHeader,
) -> EventBundle:
    return replace(
        bundle,
        event_name=event_id,
        mechanism=mechanism_from_rake(header.rake),
        strike=header.strike,
        dip=header.dip,
        rake=header.rake,
    )


def _epicentral_distance_km(bundle: EventBundle, station: StationWaveform) -> float:
    geometry = compute_source_station_geometry(
        bundle.latitude,
        bundle.longitude,
        bundle.depth_km,
        station.latitude,
        station.longitude,
    )
    return float(geometry.epicentral_distance_m / 1000.0)


def _accepted_external_station(
    bundle: EventBundle,
    station: StationWaveform,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, float, str, str]:
    distance_km = _epicentral_distance_km(bundle, station)
    if distance_km > MAX_EPICENTRAL_DISTANCE_KM:
        return None, float("nan"), "beyond_800_km", f"{distance_km:.6f}"
    base_waveform_config = waveform_config_from_v2(config)
    effective_config = _phase_preserving_external_waveform_config(
        station,
        base_waveform_config,
    )
    try:
        sample = build_station_sample(
            record_from_external_bundle(bundle, station),
            units="m",
            waveform_config=effective_config,
            alpha_m_per_s=float(config["physics"]["alpha"]),
            radial_peak_min_cm=float(config["dataset"]["radial_peak_min_cm"]),
        )
    except SampleRejected as exc:
        return None, effective_config.start_sec, exc.reason, exc.detail
    return sample, effective_config.start_sec, "", ""


def _stored_station_payload(
    station: StationWaveform,
    *,
    waveform_start_sec: float,
) -> dict[str, Any]:
    return {
        "t": np.asarray(station.t, dtype=np.float64).copy(),
        "E": np.asarray(station.e_m, dtype=np.float64) * 1000.0,
        "N": np.asarray(station.n_m, dtype=np.float64) * 1000.0,
        "U": np.asarray(station.u_m, dtype=np.float64) * 1000.0,
        "waveform_start_sec": float(waveform_start_sec),
    }


def _nearest_old_station(
    station: StationWaveform,
    old_coordinates: list[tuple[str, float, float]],
) -> tuple[str, float]:
    nearest_code = ""
    nearest_distance = float("inf")
    for code, latitude, longitude in old_coordinates:
        geometry = compute_source_station_geometry(
            station.latitude,
            station.longitude,
            0.0,
            latitude,
            longitude,
        )
        distance_km = geometry.epicentral_distance_m / 1000.0
        if distance_km < nearest_distance:
            nearest_code = code
            nearest_distance = distance_km
    return nearest_code, float(nearest_distance)


def _source_package_files(package_dir: Path) -> list[Path]:
    return [
        package_dir / "event.json",
        package_dir / "stations.csv",
        package_dir / "waveforms.csv.gz",
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _metadata_arrays(events: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    object_fields = {"events", "country", "mechanism", "origin_time", "enu", "station_info"}
    float_fields = {
        "longitude",
        "latitude",
        "depth_km",
        "magnitude",
        "strike",
        "dip",
        "rake",
        "magnitude_selected",
        "magnitude_usgs",
        "magnitude_gcmt",
        "magnitude_stf_native",
    }
    arrays: dict[str, np.ndarray] = {}
    for key in object_fields:
        arrays[key] = np.asarray([event[key] for event in events], dtype=object)
    for key in float_fields:
        arrays[key] = np.asarray([event[key] for event in events], dtype=np.float64)
    arrays["station_count"] = np.asarray(
        [len(event["enu"]) for event in events], dtype=np.int32
    )
    arrays["magnitude_source"] = np.asarray(
        [event["magnitude_source"] for event in events], dtype="<U32"
    )
    arrays["magnitude_source_rank"] = np.asarray(
        [event["magnitude_source_rank"] for event in events], dtype=np.int16
    )
    arrays["magnitude_type"] = np.asarray(
        [event["magnitude_type"] for event in events], dtype="<U16"
    )
    arrays["usgs_event_id"] = np.asarray(
        [event["usgs_event_id"] for event in events], dtype="<U64"
    )
    arrays["usgs_product_id"] = np.asarray(
        [event["usgs_product_id"] for event in events], dtype="<U64"
    )
    return arrays


def _write_source_file_manifest(path: Path, sources: dict[Path, str]) -> None:
    rows = []
    for source_path in sorted(sources, key=lambda item: str(item)):
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        rows.append(
            {
                "role": sources[source_path],
                "path": str(source_path.resolve()),
                "size_bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
            }
        )
    _write_csv(path, rows, ["role", "path", "size_bytes", "sha256"])


def _write_hash_manifest(snapshot_dir: Path) -> None:
    paths = [
        path
        for path in snapshot_dir.rglob("*")
        if path.is_file() and path.name != "hashes.sha256"
    ]
    lines = [
        f"{sha256_file(path)}  {path.relative_to(snapshot_dir).as_posix()}"
        for path in sorted(paths)
    ]
    (snapshot_dir / "hashes.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _copy_snapshot_stfs(
    snapshot_stf_dir: Path,
    existing_event_names: Iterable[str],
    new_stf_sources: dict[str, Path],
    source_files: dict[Path, str],
) -> None:
    snapshot_stf_dir.mkdir(parents=True, exist_ok=False)
    for event_name in sorted(existing_event_names):
        source = STF_ROOT / f"{event_name}.stf"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, snapshot_stf_dir / f"{event_name}.stf")
        source_files[source] = "existing_event_stf"
    for event_id, source in sorted(new_stf_sources.items()):
        shutil.copyfile(source, snapshot_stf_dir / f"{event_id}.stf")
        source_files[source] = "new_event_scardec_stf"


def _load_noto_bundle() -> tuple[tempfile.TemporaryDirectory[str], EventBundle]:
    temporary = tempfile.TemporaryDirectory(prefix="phase39-noto-")
    destination = Path(temporary.name)
    wanted = {
        f"{NOTO_ARCHIVE_PREFIX}/event.json",
        f"{NOTO_ARCHIVE_PREFIX}/stations.csv",
        f"{NOTO_ARCHIVE_PREFIX}/waveforms.csv.gz",
    }
    with tarfile.open(NOTO_ARCHIVE, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.name in wanted]
        if {member.name for member in members} != wanted:
            temporary.cleanup()
            raise FileNotFoundError("Noto package is incomplete in the public archive")
        archive.extractall(destination, members=members, filter="data")
    return temporary, load_event_bundle(destination / NOTO_ARCHIVE_PREFIX)


def build_snapshot(
    *,
    output_root: Path,
    snapshot_name: str | None = None,
) -> Path:
    old_hash_before = sha256_file(OLD_NPZ)
    if old_hash_before != EXPECTED_OLD_HASH:
        raise ValueError(f"old NPZ hash changed: {old_hash_before}")

    accepted_event_names, accepted_by_event = _read_old_accepted()
    base_config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    source_files: dict[Path, str] = {
        OLD_NPZ: "legacy_source_npz",
        OLD_MANIFEST: "legacy_phase39_manifest",
        BASE_CONFIG: "phase39_base_config",
        NOTO_ARCHIVE: "paper_public_archive",
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if snapshot_name is None:
        snapshot_name = f"phase39-expanded-{timestamp}-{old_hash_before[:8]}"
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / snapshot_name
    if final_dir.exists():
        raise FileExistsError(final_dir)
    temporary_dir = output_root / f".{snapshot_name}.tmp-{os.getpid()}"
    temporary_dir.mkdir(parents=False, exist_ok=False)

    event_rows: list[dict[str, Any]] = []
    station_source_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    event_source_rows: list[dict[str, Any]] = []
    new_stf_sources: dict[str, Path] = {}

    try:
        with np.load(OLD_NPZ, allow_pickle=True) as old_data:
            old_names = [str(value) for value in old_data["events"]]
            for event_name in accepted_event_names:
                index = old_names.index(event_name)
                metadata = _event_metadata_from_old(old_data, index)
                metadata["enu"] = {
                    str(code): _copy_station_payload(payload)
                    for code, payload in old_data["enu"][index].items()
                }
                metadata["station_info"] = copy.deepcopy(
                    old_data["station_info"][index]
                )
                event_rows.append(metadata)
                for station in sorted(metadata["enu"]):
                    station_source_rows.append(
                        {
                            "event": event_name,
                            "station": station,
                            "source_role": "legacy_snapshot",
                            "source_event_id": str(metadata["usgs_event_id"]),
                            "source_path": str(OLD_NPZ.resolve()),
                            "source_station": station,
                            "waveform_start_sec": "",
                            "source_units": "mm",
                            "stored_units": "mm",
                            "old_phase39_accepted": (
                                station_identity(station)
                                in accepted_by_event[event_name]
                            ),
                        }
                    )
                event_source_rows.append(
                    {
                        "event": event_name,
                        "role": "existing_phase39_event",
                        "source_event_id": str(metadata["usgs_event_id"]),
                        "source_path": str(OLD_NPZ.resolve()),
                        "raw_station_count": len(metadata["enu"]),
                        "added_station_count": 0,
                        "catalog_magnitude": float(metadata["magnitude"]),
                        "stf_source": str((STF_ROOT / f"{event_name}.stf").resolve()),
                    }
                )

        event_map = {str(row["events"]): row for row in event_rows}

        for spec in EXISTING_ADDITION_SPECS:
            source_files.update(
                {path: "same_event_normalized_package" for path in _source_package_files(spec.package_dir)}
            )
            metadata = event_map[spec.event_name]
            bundle = _bundle_with_existing_metadata(
                load_event_bundle(spec.package_dir),
                metadata,
            )
            existing_codes = {station_identity(code) for code in metadata["enu"]}
            additions = 0
            for station in sorted(bundle.stations, key=lambda item: station_identity(item.station)):
                output_code = normalize_station_code(station.station)
                identity = station_identity(output_code)
                if identity in existing_codes:
                    exclusion_rows.append(
                        {
                            "event": spec.event_name,
                            "station": output_code,
                            "source_path": str(spec.package_dir.resolve()),
                            "reason": "duplicate_raw_station_code",
                            "detail": "",
                            "nearest_old_station": output_code,
                            "nearest_old_distance_km": 0.0,
                        }
                    )
                    continue
                sample, phase_start, reason, detail = _accepted_external_station(
                    bundle,
                    station,
                    base_config,
                )
                if sample is None:
                    exclusion_rows.append(
                        {
                            "event": spec.event_name,
                            "station": output_code,
                            "source_path": str(spec.package_dir.resolve()),
                            "reason": reason,
                            "detail": detail,
                            "nearest_old_station": "",
                            "nearest_old_distance_km": "",
                        }
                    )
                    continue
                metadata["enu"][output_code] = _stored_station_payload(
                    station,
                    waveform_start_sec=phase_start,
                )
                metadata["station_info"][output_code] = {
                    "lat": float(station.latitude),
                    "lon": float(station.longitude),
                }
                existing_codes.add(identity)
                additions += 1
                station_source_rows.append(
                    {
                        "event": spec.event_name,
                        "station": output_code,
                        "source_role": "same_event_addition",
                        "source_event_id": bundle.event_dir_name,
                        "source_path": str(spec.package_dir.resolve()),
                        "source_station": station.station,
                        "waveform_start_sec": phase_start,
                        "source_units": "m",
                        "stored_units": "mm",
                        "old_phase39_accepted": False,
                    }
                )
            if additions != spec.expected_additions:
                raise ValueError(
                    f"{spec.event_name} additions changed: "
                    f"expected={spec.expected_additions}, actual={additions}"
                )
            for row in event_source_rows:
                if row["event"] == spec.event_name:
                    row["added_station_count"] = additions
                    row["addition_source_path"] = str(spec.package_dir.resolve())
                    break
            del bundle
            gc.collect()

        noto_metadata = event_map["Noto2024"]
        noto_codes = {station_identity(code) for code in noto_metadata["enu"]}
        noto_coordinates = [
            (
                str(code),
                float(value["lat"]),
                float(value["lon"]),
            )
            for code, value in noto_metadata["station_info"].items()
        ]
        noto_temporary, noto_source_bundle = _load_noto_bundle()
        try:
            noto_bundle = _bundle_with_existing_metadata(
                noto_source_bundle,
                noto_metadata,
            )
            noto_additions = 0
            for station in sorted(
                noto_bundle.stations,
                key=lambda item: station_identity(item.station),
            ):
                output_code = normalize_station_code(station.station)
                identity = station_identity(output_code)
                if identity in noto_codes:
                    continue
                sample, phase_start, reason, detail = _accepted_external_station(
                    noto_bundle,
                    station,
                    base_config,
                )
                if sample is None:
                    exclusion_rows.append(
                        {
                            "event": "Noto2024",
                            "station": output_code,
                            "source_path": f"{NOTO_ARCHIVE.resolve()}::{NOTO_ARCHIVE_PREFIX}",
                            "reason": reason,
                            "detail": detail,
                            "nearest_old_station": "",
                            "nearest_old_distance_km": "",
                        }
                    )
                    continue
                nearest_code, nearest_distance = _nearest_old_station(
                    station,
                    noto_coordinates,
                )
                if nearest_distance < NOTO_ALIAS_DISTANCE_KM:
                    exclusion_rows.append(
                        {
                            "event": "Noto2024",
                            "station": output_code,
                            "source_path": f"{NOTO_ARCHIVE.resolve()}::{NOTO_ARCHIVE_PREFIX}",
                            "reason": "coordinate_alias_below_1_km",
                            "detail": "",
                            "nearest_old_station": nearest_code,
                            "nearest_old_distance_km": nearest_distance,
                        }
                    )
                    continue
                noto_metadata["enu"][output_code] = _stored_station_payload(
                    station,
                    waveform_start_sec=phase_start,
                )
                noto_metadata["station_info"][output_code] = {
                    "lat": float(station.latitude),
                    "lon": float(station.longitude),
                }
                noto_codes.add(identity)
                noto_additions += 1
                station_source_rows.append(
                    {
                        "event": "Noto2024",
                        "station": output_code,
                        "source_role": "noto_public_archive_addition",
                        "source_event_id": "noto-2024-japan",
                        "source_path": f"{NOTO_ARCHIVE.resolve()}::{NOTO_ARCHIVE_PREFIX}",
                        "source_station": station.station,
                        "waveform_start_sec": phase_start,
                        "source_units": "m",
                        "stored_units": "mm",
                        "old_phase39_accepted": False,
                    }
                )
            if noto_additions != EXPECTED_NOTO_ADDITIONS:
                raise ValueError(
                    f"Noto additions changed: expected={EXPECTED_NOTO_ADDITIONS}, "
                    f"actual={noto_additions}"
                )
            for row in event_source_rows:
                if row["event"] == "Noto2024":
                    row["added_station_count"] = noto_additions
                    row["addition_source_path"] = (
                        f"{NOTO_ARCHIVE.resolve()}::{NOTO_ARCHIVE_PREFIX}"
                    )
                    break
        finally:
            noto_temporary.cleanup()
            del noto_source_bundle
            gc.collect()

        new_event_selected_count = 0
        for spec in NEW_EVENT_SPECS:
            source_files.update(
                {path: "new_event_normalized_package" for path in _source_package_files(spec.package_dir)}
            )
            scardec_source = find_scardec_source(spec.scardec_dir)
            header = parse_scardec_header(scardec_source)
            new_stf_sources[spec.event_id] = scardec_source
            bundle = _bundle_with_scardec_metadata(
                load_event_bundle(spec.package_dir),
                spec.event_id,
                header,
            )
            if abs(bundle.magnitude - header.magnitude) > 0.4:
                raise ValueError(
                    f"catalog/SCARDEC magnitude mismatch for {spec.event_id}"
                )
            enu: dict[str, dict[str, Any]] = {}
            station_info: dict[str, dict[str, float]] = {}
            selected_codes: set[str] = set()
            for station in sorted(bundle.stations, key=lambda item: station_identity(item.station)):
                output_code = normalize_station_code(station.station)
                identity = station_identity(output_code)
                if identity in selected_codes:
                    exclusion_rows.append(
                        {
                            "event": spec.event_id,
                            "station": output_code,
                            "source_path": str(spec.package_dir.resolve()),
                            "reason": "duplicate_package_station_code",
                            "detail": "",
                            "nearest_old_station": "",
                            "nearest_old_distance_km": "",
                        }
                    )
                    continue
                sample, phase_start, reason, detail = _accepted_external_station(
                    bundle,
                    station,
                    base_config,
                )
                if sample is None:
                    exclusion_rows.append(
                        {
                            "event": spec.event_id,
                            "station": output_code,
                            "source_path": str(spec.package_dir.resolve()),
                            "reason": reason,
                            "detail": detail,
                            "nearest_old_station": "",
                            "nearest_old_distance_km": "",
                        }
                    )
                    continue
                enu[output_code] = _stored_station_payload(
                    station,
                    waveform_start_sec=phase_start,
                )
                station_info[output_code] = {
                    "lat": float(station.latitude),
                    "lon": float(station.longitude),
                }
                selected_codes.add(identity)
                station_source_rows.append(
                    {
                        "event": spec.event_id,
                        "station": output_code,
                        "source_role": "new_event",
                        "source_event_id": spec.event_id,
                        "source_path": str(spec.package_dir.resolve()),
                        "source_station": station.station,
                        "waveform_start_sec": phase_start,
                        "source_units": "m",
                        "stored_units": "mm",
                        "old_phase39_accepted": False,
                    }
                )
            if not enu:
                raise ValueError(f"new event has no accepted stations: {spec.event_id}")
            event_meta = json.loads(
                (spec.package_dir / "event.json").read_text(encoding="utf-8")
            )
            origin_time = str(event_meta.get("event_time") or event_meta.get("date"))
            event_rows.append(
                {
                    "events": spec.event_id,
                    "country": str(event_meta.get("country", "")),
                    "longitude": float(bundle.longitude),
                    "latitude": float(bundle.latitude),
                    "depth_km": float(bundle.depth_km),
                    "magnitude": float(bundle.magnitude),
                    "mechanism": mechanism_from_rake(header.rake),
                    "strike": header.strike,
                    "dip": header.dip,
                    "rake": header.rake,
                    "origin_time": origin_time,
                    "enu": enu,
                    "station_info": station_info,
                    "magnitude_selected": float(bundle.magnitude),
                    "magnitude_usgs": float(bundle.magnitude),
                    "magnitude_gcmt": float("nan"),
                    "magnitude_stf_native": header.magnitude,
                    "magnitude_source": "event_package",
                    "magnitude_source_rank": 1,
                    "magnitude_type": str(event_meta.get("magnitude_type") or "mw"),
                    "usgs_event_id": spec.event_id,
                    "usgs_product_id": "",
                }
            )
            new_event_selected_count += len(enu)
            event_source_rows.append(
                {
                    "event": spec.event_id,
                    "role": "new_event",
                    "source_event_id": spec.event_id,
                    "source_path": str(spec.package_dir.resolve()),
                    "raw_station_count": len(bundle.stations),
                    "added_station_count": len(enu),
                    "catalog_magnitude": float(bundle.magnitude),
                    "stf_source": str(scardec_source.resolve()),
                    "scardec_magnitude": header.magnitude,
                    "scardec_strike": header.strike,
                    "scardec_dip": header.dip,
                    "scardec_rake": header.rake,
                }
            )
            del bundle
            gc.collect()

        if new_event_selected_count != EXPECTED_NEW_EVENT_ACCEPTED:
            raise ValueError(
                "new-event station count changed: "
                f"expected={EXPECTED_NEW_EVENT_ACCEPTED}, "
                f"actual={new_event_selected_count}"
            )
        same_event_total = sum(spec.expected_additions for spec in EXISTING_ADDITION_SPECS)
        if same_event_total != EXPECTED_SAME_EVENT_ADDITIONS:
            raise AssertionError("same-event expected counts are inconsistent")

        included_identities = {normalize_identity(row["events"]) for row in event_rows}
        overlap = included_identities & {
            normalize_identity(name) for name in EXTERNAL_UNSEEN_DIRECTORY_NAMES
        }
        if overlap:
            raise ValueError(f"external unseen events entered the dataset: {sorted(overlap)}")

        event_rows.sort(key=lambda row: str(row["events"]))
        if len(event_rows) != EXPECTED_ACCEPTED_EVENTS:
            raise ValueError(f"candidate event count changed: {len(event_rows)}")

        npz_path = temporary_dir / "gnss_events_matched.phase39_expanded.npz"
        deterministic_savez_compressed(npz_path, _metadata_arrays(event_rows))

        snapshot_stf_dir = temporary_dir / "stf"
        _copy_snapshot_stfs(
            snapshot_stf_dir,
            accepted_event_names,
            new_stf_sources,
            source_files,
        )

        saved_config = copy.deepcopy(base_config)
        saved_config["dataset"]["blacklist_events"] = []
        saved_config["paths"]["data_path"] = str(
            final_dir / npz_path.name
        )
        saved_config["dataset"]["stf"]["path"] = str(final_dir / "stf")
        saved_config_path = temporary_dir / "phase39_expanded_preflight.yaml"
        saved_config_path.write_text(
            yaml.safe_dump(saved_config, sort_keys=False),
            encoding="utf-8",
        )

        runtime_config = copy.deepcopy(saved_config)
        runtime_config["paths"]["data_path"] = str(npz_path)
        runtime_config["dataset"]["stf"]["path"] = str(snapshot_stf_dir)
        dataset = CorrectedEarthquakeDataset(runtime_config)
        preflight_dir = temporary_dir / "preflight"
        summary = write_dataset_audit(
            dataset,
            manifest_path=preflight_dir / "dataset_manifest.csv",
            summary_path=preflight_dir / "dataset_summary.json",
        )
        if not audit_passes(summary):
            raise ValueError("expanded dataset audit invariants failed")
        if int(summary["accepted_event_count"]) != EXPECTED_ACCEPTED_EVENTS:
            raise ValueError(
                f"preflight accepted events changed: {summary['accepted_event_count']}"
            )
        if int(summary["accepted_station_count"]) != EXPECTED_ACCEPTED_STATIONS:
            raise ValueError(
                "preflight accepted stations changed: "
                f"{summary['accepted_station_count']}"
            )

        provenance_dir = temporary_dir / "provenance"
        _write_csv(
            provenance_dir / "event_sources.csv",
            event_source_rows,
            [
                "event",
                "role",
                "source_event_id",
                "source_path",
                "raw_station_count",
                "added_station_count",
                "catalog_magnitude",
                "stf_source",
                "addition_source_path",
                "scardec_magnitude",
                "scardec_strike",
                "scardec_dip",
                "scardec_rake",
            ],
        )
        _write_csv(
            provenance_dir / "station_sources.csv",
            station_source_rows,
            [
                "event",
                "station",
                "source_role",
                "source_event_id",
                "source_path",
                "source_station",
                "waveform_start_sec",
                "source_units",
                "stored_units",
                "old_phase39_accepted",
            ],
        )
        _write_csv(
            provenance_dir / "exclusions.csv",
            exclusion_rows,
            [
                "event",
                "station",
                "source_path",
                "reason",
                "detail",
                "nearest_old_station",
                "nearest_old_distance_km",
            ],
        )
        _write_source_file_manifest(
            provenance_dir / "source_files.csv",
            source_files,
        )

        old_hash_after = sha256_file(OLD_NPZ)
        if old_hash_after != old_hash_before:
            raise ValueError("old source NPZ changed during the build")
        build_summary = {
            "schema_version": "phase39-expanded-snapshot/v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_name": snapshot_name,
            "old_source_npz": str(OLD_NPZ.resolve()),
            "old_source_sha256_before": old_hash_before,
            "old_source_sha256_after": old_hash_after,
            "dataset_npz": npz_path.name,
            "dataset_npz_sha256": sha256_file(npz_path),
            "accepted_event_count": int(summary["accepted_event_count"]),
            "accepted_station_count": int(summary["accepted_station_count"]),
            "existing_event_count": EXPECTED_EXISTING_EVENTS,
            "existing_accepted_station_count": EXPECTED_EXISTING_ACCEPTED,
            "new_event_count": EXPECTED_NEW_EVENTS,
            "new_event_accepted_station_count": new_event_selected_count,
            "same_event_addition_count": EXPECTED_SAME_EVENT_ADDITIONS,
            "noto_addition_count": EXPECTED_NOTO_ADDITIONS,
            "external_unseen_events_preserved": sorted(
                EXTERNAL_UNSEEN_DIRECTORY_NAMES
            ),
            "formal_training_started": False,
            "preflight_audit_passed": True,
        }
        (temporary_dir / "build_summary.json").write_text(
            json.dumps(build_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_hash_manifest(temporary_dir)
        os.replace(temporary_dir, final_dir)
        return final_dir
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the immutable 39-event Phase 39 expanded dataset snapshot."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_DATA_ROOT / "magnitude-label-snapshots",
    )
    parser.add_argument("--snapshot-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = build_snapshot(
        output_root=args.output_root.resolve(),
        snapshot_name=args.snapshot_name,
    )
    print(snapshot)


if __name__ == "__main__":
    main()
