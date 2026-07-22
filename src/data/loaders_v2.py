from __future__ import annotations

import copy
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from src.data.dataset_v2 import CorrectedEarthquakeDataset
from src.data.splits import (
    EventGroupSplit,
    assert_no_event_overlap,
    make_event_balanced_weights,
    make_event_group_split,
    make_within_event_station_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_data_path(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    repository_candidate = PROJECT_ROOT / path
    if repository_candidate.exists():
        return str(repository_candidate)
    data_root = os.environ.get("PINN_DATA_ROOT")
    if data_root:
        parts = path.parts[1:] if path.parts[:1] == ("data",) else path.parts
        candidate = Path(data_root).joinpath(*parts)
        if candidate.exists():
            return str(candidate)
    return str(repository_candidate)


def _runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    if "paths" in result and "data_path" in result["paths"]:
        result["paths"]["data_path"] = _resolve_data_path(
            result["paths"]["data_path"]
        )
    stf = result.get("dataset", {}).get("stf", {})
    if "path" in stf:
        stf["path"] = _resolve_data_path(stf["path"])
    return result


def _loeo_split(
    events: list[str],
    *,
    leave_out_event: str,
    validation_fraction: float,
    seed: int,
) -> EventGroupSplit:
    test_indices = [
        index for index, event in enumerate(events) if event == leave_out_event
    ]
    if not test_indices:
        raise ValueError(f"leave_out_event not found: {leave_out_event}")
    remaining_indices = [
        index for index, event in enumerate(events) if event != leave_out_event
    ]
    remaining_events = [events[index] for index in remaining_indices]
    remaining_split = make_event_group_split(
        remaining_events,
        validation_fraction=validation_fraction,
        test_fraction=0.0,
        seed=seed,
    )
    split = EventGroupSplit(
        train_indices=[remaining_indices[index] for index in remaining_split.train_indices],
        validation_indices=[
            remaining_indices[index]
            for index in remaining_split.validation_indices
        ],
        test_indices=test_indices,
    )
    assert_no_event_overlap(events, split)
    return split


def _worker_init(worker_id: int, seed: int) -> None:
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _split_manifest(
    protocol: str,
    seed: int,
    samples: list[dict[str, Any]],
    split: EventGroupSplit,
) -> dict[str, Any]:
    events = [str(sample["event"]) for sample in samples]
    sample_keys = [
        f"{sample['event']}::{sample['station']}" for sample in samples
    ]
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("duplicate event/station sample key in dataset")
    split_indices = {
        "train": split.train_indices,
        "validation": split.validation_indices,
        "test": split.test_indices,
    }
    keys_by_split = {
        name: sorted(sample_keys[index] for index in indices)
        for name, indices in split_indices.items()
    }
    per_event_counts: dict[str, dict[str, int]] = {}
    for event in sorted(set(events)):
        per_event_counts[event] = {
            name: sum(events[index] == event for index in indices)
            for name, indices in split_indices.items()
        }

    def magnitude_summary(indices: list[int]) -> dict[str, float | int | None]:
        values = [
            float(samples[index]["magnitude_catalog"])
            for index in indices
            if math.isfinite(float(samples[index]["magnitude_catalog"]))
        ]
        return {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        }

    canonical_assignments = json.dumps(
        keys_by_split,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "protocol": protocol,
        "seed": seed,
        "train_events": sorted({events[index] for index in split.train_indices}),
        "validation_events": sorted(
            {events[index] for index in split.validation_indices}
        ),
        "test_events": sorted({events[index] for index in split.test_indices}),
        "train_record_count": len(split.train_indices),
        "validation_record_count": len(split.validation_indices),
        "test_record_count": len(split.test_indices),
        "sample_keys": keys_by_split,
        "per_event_station_counts": per_event_counts,
        "catalog_mw_summary": {
            name: magnitude_summary(indices)
            for name, indices in split_indices.items()
        },
        "assignment_sha256": hashlib.sha256(
            canonical_assignments
        ).hexdigest(),
    }


def _validate_active_split_manifest(
    manifest: dict[str, Any],
    *,
    validation_fraction: float,
    test_fraction: float,
) -> None:
    total = sum(
        int(manifest[f"{name}_record_count"])
        for name in ("train", "validation", "test")
    )
    expected = {
        "train": 1.0 - validation_fraction - test_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    for name, fraction in expected.items():
        observed = int(manifest[f"{name}_record_count"]) / total
        if abs(observed - fraction) > 0.01:
            raise ValueError(
                f"{name} split fraction {observed:.6f} differs from "
                f"{fraction:.6f}"
            )

    summaries = manifest["catalog_mw_summary"]
    all_means = [
        float(summary["mean"])
        for summary in summaries.values()
        if summary["mean"] is not None
    ]
    full_mean = sum(
        float(summary["mean"]) * int(summary["count"])
        for summary in summaries.values()
        if summary["mean"] is not None
    ) / sum(int(summary["count"]) for summary in summaries.values())
    if any(abs(mean - full_mean) > 0.05 for mean in all_means):
        raise ValueError("station-weighted catalog Mw split means differ by > 0.05")

    for event, counts in manifest["per_event_station_counts"].items():
        event_total = sum(int(value) for value in counts.values())
        if event_total >= 3 and any(int(value) < 1 for value in counts.values()):
            raise ValueError(f"event {event} is missing a split assignment")


def get_data_loaders_v2(
    config: dict[str, Any],
    *,
    leave_out_event: str | None = None,
    max_events: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    full_dataset = CorrectedEarthquakeDataset(_runtime_config(config))
    full_samples = list(full_dataset.samples)
    full_events = [str(sample["event"]) for sample in full_samples]
    dataset: Dataset = full_dataset
    samples = full_samples
    events = full_events
    if max_events is not None:
        if isinstance(max_events, bool) or int(max_events) != max_events or max_events < 2:
            raise ValueError("max_events must be an integer of at least 2")
        selected_events = set(sorted(set(full_events))[: int(max_events)])
        selected_indices = [
            index
            for index, event in enumerate(full_events)
            if event in selected_events
        ]
        if len(selected_events) < 2:
            raise ValueError("max_events retained fewer than two events")
        dataset = Subset(full_dataset, selected_indices)
        samples = [full_samples[index] for index in selected_indices]
        events = [full_events[index] for index in selected_indices]
    training = config["training"]
    protocol = str(training["split_protocol"])
    validation_fraction = float(training["validation_event_fraction"])
    test_fraction = float(training["test_event_fraction"])
    seed = int(training["random_seed"])
    if protocol == "grouped_event":
        split = make_event_group_split(
            events,
            validation_fraction,
            test_fraction,
            seed,
        )
    elif protocol == "within_event_station":
        split = make_within_event_station_split(
            [
                (str(sample["event"]), str(sample["station"]))
                for sample in samples
            ],
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    elif protocol == "loeo":
        if leave_out_event is None:
            raise ValueError("loeo protocol requires leave_out_event")
        split = _loeo_split(
            events,
            leave_out_event=leave_out_event,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    else:
        raise ValueError(f"unsupported v2 split protocol: {protocol}")

    train_dataset = Subset(dataset, split.train_indices)
    validation_dataset = Subset(dataset, split.validation_indices)
    test_dataset = Subset(dataset, split.test_indices)
    batch_size = int(training["batch_size"])
    num_workers = int(training.get("num_workers", 0))
    generator = torch.Generator().manual_seed(seed)
    worker_init_fn = partial(_worker_init, seed=seed)
    sampler = None
    shuffle = True
    if bool(training.get("event_balanced_sampling", False)):
        train_events = [events[index] for index in split.train_indices]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(
                make_event_balanced_weights(train_events),
                dtype=torch.double,
            ),
            num_samples=len(split.train_indices),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    split_manifest = _split_manifest(protocol, seed, samples, split)
    if config.get("workflow") == "station_random_shifted_stf":
        _validate_active_split_manifest(
            split_manifest,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
    return (
        train_loader,
        validation_loader,
        test_loader,
        split_manifest,
    )


def list_events_v2(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = CorrectedEarthquakeDataset(_runtime_config(config))
    counts: dict[tuple[int, str], int] = {}
    for sample in dataset.samples:
        key = (int(sample["event_index"]), str(sample["event"]))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"event_index": index, "event": event, "n_stations": count}
        for (index, event), count in sorted(counts.items())
    ]
