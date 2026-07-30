#!/usr/bin/env python3
"""Build the causal Crowell PGD hint cache for Phase67 train/validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.run_phase40_streaming_consistency import (  # noqa: E402
    load_frozen_config,
)
from scripts.experiments.run_phase43_streaming_adapter import (  # noqa: E402
    EXPECTED_SPLIT_ASSIGNMENT_SHA256,
    HORIZONS,
    CacheBundle,
    load_cache,
)
from scripts.experiments.run_phase50_stateful_incremental_model import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_VALIDATION_COUNT,
)
from src.baseline.causal_pgd import (  # noqa: E402
    build_raw_pgd_records,
    causal_pgd_3d,
)
from src.baseline.scaling_laws import predict_mw  # noqa: E402
from src.data.loaders_v2 import get_data_loaders_v2  # noqa: E402
from src.utils.provenance import (  # noqa: E402
    current_git_commit,
    git_is_dirty,
    sha256_file,
)


CACHE_VERSION = 1
PROCESSING_DELAY_SEC = 6.0
PGD_FILENAME = "pgd_3d_m.npy"
MW_FILENAME = "crowell_mw.npy"
P_ARRIVED_FILENAME = "p_arrived.npy"
RECORDS_FILENAME = "records.json"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class PGDHintCache:
    root: Path
    pgd_3d_m: np.ndarray
    crowell_mw: np.ndarray
    p_arrived: np.ndarray
    records: list[dict[str, Any]]
    manifest: dict[str, Any]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_new_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output directory must be new or empty: {path}")


def _selected_samples(
    config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    train_loader, validation_loader, test_loader, split_manifest = (
        get_data_loaders_v2(config)
    )
    if split_manifest["assignment_sha256"] != EXPECTED_SPLIT_ASSIGNMENT_SHA256:
        raise ValueError("Phase67 split assignment changed")
    full_dataset = train_loader.dataset.dataset
    train_indices = [int(value) for value in train_loader.dataset.indices]
    validation_indices = [
        int(value) for value in validation_loader.dataset.indices
    ]
    if len(train_indices) != EXPECTED_TRAIN_COUNT:
        raise ValueError("Phase67 training count changed")
    if len(validation_indices) != EXPECTED_VALIDATION_COUNT:
        raise ValueError("Phase67 validation count changed")
    samples = {
        "train": [full_dataset.samples[index] for index in train_indices],
        "validation": [
            full_dataset.samples[index] for index in validation_indices
        ],
    }
    del train_loader, validation_loader, test_loader
    return samples, split_manifest


def _cache_records(cache: CacheBundle) -> list[dict[str, Any]]:
    records = [
        {
            "cache_index": index,
            "event": str(record["event"]),
            "station": str(record["station"]),
            "split": str(record["split"]),
        }
        for index, record in enumerate(cache.records)
    ]
    expected_splits = {"train", "validation"}
    if {record["split"] for record in records} != expected_splits:
        raise ValueError("Phase67 cache record splits changed")
    return records


def build_hint_cache(
    *,
    output_root: Path,
    phase43_cache_root: Path = DEFAULT_CACHE_ROOT,
) -> dict[str, Any]:
    _validate_new_directory(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_frozen_config()
    samples, split_manifest = _selected_samples(config)
    phase43 = load_cache(phase43_cache_root)
    records = _cache_records(phase43)
    raw_records = build_raw_pgd_records(config, samples)
    record_keys = {(row["event"], row["station"]) for row in records}
    if record_keys != set(raw_records):
        raise ValueError("Phase67 PGD/cache record cohort changed")

    shape = (len(records), len(HORIZONS))
    pgd_path = output_root / PGD_FILENAME
    mw_path = output_root / MW_FILENAME
    p_arrived_path = output_root / P_ARRIVED_FILENAME
    pgd_values = np.lib.format.open_memmap(
        pgd_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    mw_values = np.lib.format.open_memmap(
        mw_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    p_arrived = np.lib.format.open_memmap(
        p_arrived_path,
        mode="w+",
        dtype=np.bool_,
        shape=shape,
    )
    for record_row in records:
        cache_index = int(record_row["cache_index"])
        key = (record_row["event"], record_row["station"])
        raw = raw_records[key]
        for horizon_index, horizon in enumerate(HORIZONS):
            pgd_m, _, _ = causal_pgd_3d(
                raw,
                observation_horizon_sec=horizon,
                processing_delay_sec=PROCESSING_DELAY_SEC,
            )
            pgd_values[cache_index, horizon_index] = pgd_m
            mw_values[cache_index, horizon_index] = predict_mw(
                law_name="crowell",
                pgd_m=pgd_m,
                source_distance_km=raw.source_distance_km,
            )
            p_arrived[cache_index, horizon_index] = (
                float(horizon) >= raw.p_arrival_sec
            )
        if (cache_index + 1) % 256 == 0:
            pgd_values.flush()
            mw_values.flush()
            p_arrived.flush()
    pgd_values.flush()
    mw_values.flush()
    p_arrived.flush()
    if not bool(np.isfinite(pgd_values).all()):
        raise ValueError("Phase67 PGD cache contains nonfinite values")
    if not bool(np.isfinite(mw_values).all()):
        raise ValueError("Phase67 Crowell cache contains nonfinite values")
    if bool(np.any(pgd_values <= 0.0)):
        raise ValueError("Phase67 PGD cache contains nonpositive values")

    records_path = output_root / RECORDS_FILENAME
    _write_json(records_path, records)
    manifest = {
        "cache_version": CACHE_VERSION,
        "status": "complete",
        "evaluation_role": "train_validation_only_causal_crowell_hint",
        "record_count": len(records),
        "train_count": EXPECTED_TRAIN_COUNT,
        "validation_count": EXPECTED_VALIDATION_COUNT,
        "horizons_sec": list(HORIZONS),
        "shape": list(shape),
        "dtypes": {
            PGD_FILENAME: str(pgd_values.dtype),
            MW_FILENAME: str(mw_values.dtype),
            P_ARRIVED_FILENAME: str(p_arrived.dtype),
        },
        "contract": {
            "components": ["east", "north", "up"],
            "observation": "0 <= t < h",
            "processing_delay_sec": PROCESSING_DELAY_SEC,
            "baseline": "component-wise pre-P mean available by release",
            "filter": "none",
            "interpolation": "none",
            "pgd": "max sqrt(E^2 + N^2 + U^2)",
            "distance": "hypocentral/source distance",
            "scaling_law": "crowell",
        },
        "source": {
            "git_commit": current_git_commit(REPO_ROOT),
            "git_dirty": git_is_dirty(REPO_ROOT),
            "split_assignment_sha256": split_manifest["assignment_sha256"],
            "phase43_raw_rates_sha256": phase43.manifest["raw_rates_sha256"],
            "dataset_path": str(config["paths"]["data_path"]),
            "dataset_sha256": sha256_file(Path(config["paths"]["data_path"])),
        },
        "outputs": {
            PGD_FILENAME: sha256_file(pgd_path),
            MW_FILENAME: sha256_file(mw_path),
            P_ARRIVED_FILENAME: sha256_file(p_arrived_path),
            RECORDS_FILENAME: sha256_file(records_path),
        },
        "hidden_data": {
            "internal_test_predictions_generated": False,
            "external_loaded": False,
            "grouped_test_loaded": False,
        },
    }
    _write_json(output_root / MANIFEST_FILENAME, manifest)
    return manifest


def load_hint_cache(
    cache_root: Path,
    *,
    phase43_cache: CacheBundle,
) -> PGDHintCache:
    manifest_path = cache_root / MANIFEST_FILENAME
    records_path = cache_root / RECORDS_FILENAME
    paths = {
        PGD_FILENAME: cache_root / PGD_FILENAME,
        MW_FILENAME: cache_root / MW_FILENAME,
        P_ARRIVED_FILENAME: cache_root / P_ARRIVED_FILENAME,
        RECORDS_FILENAME: records_path,
    }
    for path in (manifest_path, *paths.values()):
        if not path.is_file():
            raise FileNotFoundError(f"missing Phase67 hint artifact: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["cache_version"]) != CACHE_VERSION:
        raise ValueError("Phase67 hint cache version changed")
    if manifest["status"] != "complete":
        raise ValueError("Phase67 hint cache is incomplete")
    if manifest["contract"]["scaling_law"] != "crowell":
        raise ValueError("Phase67 hint law changed")
    if manifest["source"]["split_assignment_sha256"] != (
        EXPECTED_SPLIT_ASSIGNMENT_SHA256
    ):
        raise ValueError("Phase67 hint split assignment changed")
    if manifest["source"]["phase43_raw_rates_sha256"] != (
        phase43_cache.manifest["raw_rates_sha256"]
    ):
        raise ValueError("Phase67 hint/Phase43 cache source changed")
    for name, path in paths.items():
        if sha256_file(path) != manifest["outputs"][name]:
            raise ValueError(f"Phase67 hint hash changed: {name}")
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if records != _cache_records(phase43_cache):
        raise ValueError("Phase67 hint record order changed")

    pgd_values = np.load(paths[PGD_FILENAME], mmap_mode="r")
    mw_values = np.load(paths[MW_FILENAME], mmap_mode="r")
    p_arrived = np.load(paths[P_ARRIVED_FILENAME], mmap_mode="r")
    expected_shape = (len(phase43_cache.records), len(HORIZONS))
    if pgd_values.shape != expected_shape or pgd_values.dtype != np.float32:
        raise ValueError("Phase67 PGD array shape or dtype changed")
    if mw_values.shape != expected_shape or mw_values.dtype != np.float32:
        raise ValueError("Phase67 Mw array shape or dtype changed")
    if p_arrived.shape != expected_shape or p_arrived.dtype != np.bool_:
        raise ValueError("Phase67 P-arrival array shape or dtype changed")
    return PGDHintCache(
        root=cache_root,
        pgd_3d_m=pgd_values,
        crowell_mw=mw_values,
        p_arrived=p_arrived,
        records=records,
        manifest=manifest,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Phase67 causal Crowell train/validation cache."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--phase43-cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = build_hint_cache(
        output_root=args.output_root.resolve(),
        phase43_cache_root=args.phase43_cache_root.resolve(),
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_root": str(args.output_root.resolve()),
                "shape": manifest["shape"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
