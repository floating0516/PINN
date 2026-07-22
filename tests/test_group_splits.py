from collections import defaultdict

import pytest
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data import loaders_v2
from src.data import splits as splits_module
from src.data.loaders_v2 import get_data_loaders_v2
from src.data.splits import (
    assert_no_event_overlap,
    make_event_balanced_weights,
    make_event_group_split,
)


def test_group_split_has_no_event_overlap() -> None:
    events = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]

    split = make_event_group_split(
        events,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )

    assert_no_event_overlap(events, split)
    train_events = {events[index] for index in split.train_indices}
    validation_events = {
        events[index] for index in split.validation_indices
    }
    test_events = {events[index] for index in split.test_indices}
    assert len(train_events) == 3
    assert len(validation_events) == 1
    assert len(test_events) == 1


def test_group_split_is_deterministic_and_retains_a_train_event() -> None:
    events = ["A", "B", "C"]

    first = make_event_group_split(events, 0.8, 0.8, seed=7)
    second = make_event_group_split(events, 0.8, 0.8, seed=7)

    assert first == second
    assert len(first.train_indices) >= 1
    assert_no_event_overlap(events, first)


def test_event_balanced_weights_sum_equally_per_event() -> None:
    events = ["A", "A", "A", "B"]

    weights = make_event_balanced_weights(events)

    assert abs(sum(weights[:3]) - weights[3]) < 1.0e-12


def test_station_split_is_order_independent_and_has_no_key_overlap() -> None:
    keys = [
        ("A", "A03"),
        ("A", "A01"),
        ("A", "A02"),
        ("B", "B03"),
        ("B", "B01"),
        ("B", "B02"),
    ]
    first = splits_module.make_within_event_station_split(
        keys,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=42,
    )
    reversed_keys = list(reversed(keys))
    second = splits_module.make_within_event_station_split(
        reversed_keys,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=42,
    )

    def assigned(split, source):
        return (
            {source[index] for index in split.train_indices},
            {source[index] for index in split.validation_indices},
            {source[index] for index in split.test_indices},
        )

    first_sets = assigned(first, keys)
    second_sets = assigned(second, reversed_keys)
    assert first_sets == second_sets
    assert not first_sets[0] & first_sets[1]
    assert not first_sets[0] & first_sets[2]
    assert not first_sets[1] & first_sets[2]
    assert set.union(*first_sets) == set(keys)


def test_station_split_uses_per_event_seventy_fifteen_fifteen_counts() -> None:
    keys = [
        (event, f"{event}{index:02d}")
        for event in ("A", "B")
        for index in range(20)
    ]

    split = splits_module.make_within_event_station_split(
        keys,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=17,
    )

    for event in ("A", "B"):
        assert sum(keys[index][0] == event for index in split.train_indices) == 14
        assert sum(keys[index][0] == event for index in split.validation_indices) == 3
        assert sum(keys[index][0] == event for index in split.test_indices) == 3


def test_station_split_rejects_duplicate_sample_keys() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        splits_module.make_within_event_station_split(
            [("A", "STA"), ("A", "STA")],
            validation_fraction=0.15,
            test_fraction=0.15,
            seed=42,
        )


class _FakeDataset(Dataset):
    def __init__(self) -> None:
        self.samples = [
            {
                "event": event,
                "station": f"S{index:02d}",
                "magnitude_catalog": 7.0 + 0.1 * index,
                "value": torch.tensor(index),
            }
            for index, event in enumerate(
                ["A", "A", "A", "B", "B", "C", "C", "D", "E", "E"]
            )
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def _loader_config(protocol: str) -> dict:
    return {
        "pipeline_version": 2,
        "training": {
            "split_protocol": protocol,
            "validation_event_fraction": 0.2,
            "test_event_fraction": 0.2,
            "random_seed": 42,
            "batch_size": 2,
            "event_balanced_sampling": True,
            "num_workers": 0,
        },
    }


def _patch_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        loaders_v2,
        "CorrectedEarthquakeDataset",
        lambda config: _FakeDataset(),
    )


def test_grouped_loader_uses_balanced_sampler_and_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch)

    train_loader, validation_loader, test_loader, manifest = (
        get_data_loaders_v2(_loader_config("grouped_event"))
    )

    assert isinstance(train_loader.sampler, WeightedRandomSampler)
    assert_no_event_overlap(
        [sample["event"] for sample in train_loader.dataset.dataset.samples],
        loaders_v2.EventGroupSplit(
            train_indices=list(train_loader.dataset.indices),
            validation_indices=list(validation_loader.dataset.indices),
            test_indices=list(test_loader.dataset.indices),
        ),
    )
    assert manifest["protocol"] == "grouped_event"
    assert manifest["train_record_count"] == len(train_loader.dataset)
    assert manifest["validation_record_count"] == len(validation_loader.dataset)
    assert manifest["test_record_count"] == len(test_loader.dataset)
    assert set(manifest) == {
        "protocol",
        "seed",
        "train_events",
        "validation_events",
        "test_events",
        "train_record_count",
        "validation_record_count",
        "test_record_count",
        "sample_keys",
        "per_event_station_counts",
        "catalog_mw_summary",
        "assignment_sha256",
    }

    event_weight_sums: dict[str, float] = defaultdict(float)
    for subset_position, weight in enumerate(train_loader.sampler.weights):
        dataset_index = train_loader.dataset.indices[subset_position]
        event = train_loader.dataset.dataset.samples[dataset_index]["event"]
        event_weight_sums[event] += float(weight)
    assert len(set(round(value, 12) for value in event_weight_sums.values())) == 1


def test_loeo_reserves_named_event_for_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch)

    train_loader, validation_loader, test_loader, manifest = (
        get_data_loaders_v2(
            _loader_config("loeo"),
            leave_out_event="C",
        )
    )

    assert manifest["test_events"] == ["C"]
    assert manifest["test_record_count"] == 2
    assert "C" not in manifest["train_events"]
    assert "C" not in manifest["validation_events"]
    assert len(test_loader.dataset) == 2


def test_loeo_requires_leave_out_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch)

    with pytest.raises(ValueError, match="leave_out_event"):
        get_data_loaders_v2(_loader_config("loeo"))


def test_within_event_station_keeps_each_event_in_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch)

    config = _loader_config("within_event_station")
    config["training"]["event_balanced_sampling"] = False
    train_loader, validation_loader, test_loader, manifest = (
        get_data_loaders_v2(config)
    )

    assert manifest["protocol"] == "within_event_station"
    assert manifest["train_events"] == ["A", "B", "C", "D", "E"]
    assert not isinstance(train_loader.sampler, WeightedRandomSampler)
    key_sets = {
        name: set(manifest["sample_keys"][name])
        for name in ("train", "validation", "test")
    }
    assert not key_sets["train"] & key_sets["validation"]
    assert not key_sets["train"] & key_sets["test"]
    assert not key_sets["validation"] & key_sets["test"]
    assert len(set.union(*key_sets.values())) == len(_FakeDataset())
    assert len(manifest["assignment_sha256"]) == 64
    assert manifest["catalog_mw_summary"]["train"]["count"] == len(
        train_loader.dataset
    )
    assert manifest["catalog_mw_summary"]["validation"]["count"] == len(
        validation_loader.dataset
    )
    assert manifest["catalog_mw_summary"]["test"]["count"] == len(
        test_loader.dataset
    )
