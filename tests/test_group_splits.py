from collections import Counter, defaultdict

import pytest
import torch
from torch.utils.data import Dataset, RandomSampler, WeightedRandomSampler

from src.data import loaders_v2
from src.data import splits as splits_module
from src.data.loaders_v2 import get_data_loaders_v2
from src.data.splits import (
    INVERSE_COUNT_FULL_DATA_ESTIMATOR,
    REPLACEMENT_SAMPLING_ESTIMATOR,
    EventGroupSplit,
    assert_no_event_overlap,
    make_event_balanced_weights,
    make_event_group_split,
    make_event_inverse_count_weights,
    resolve_event_balance_exponent,
    resolve_event_balance_estimator,
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


def test_inverse_count_weights_have_unit_mean_and_equal_event_mass() -> None:
    events = ["A", "A", "A", "B"]

    weights = make_event_inverse_count_weights(events)

    assert sum(weights) == pytest.approx(len(events))
    assert sum(weights[:3]) == pytest.approx(len(events) / 2)
    assert weights[3] == pytest.approx(len(events) / 2)
    assert weights == pytest.approx([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0])


def test_inverse_count_exponent_one_is_exactly_backward_compatible() -> None:
    events = ["A", "A", "A", "B", "C", "C"]
    counts = Counter(events)
    normalization = len(events) / len(counts)
    legacy_weights = [normalization / counts[event] for event in events]

    assert make_event_inverse_count_weights(events) == legacy_weights
    assert (
        make_event_inverse_count_weights(events, exponent=1.0)
        == legacy_weights
    )


def test_tempered_inverse_count_weights_have_unit_mean_and_analytic_mass() -> None:
    events = ["A", "A", "A", "B"]
    exponent = 0.5
    counts = Counter(events)
    normalization = len(events) / sum(
        count ** (1.0 - exponent) for count in counts.values()
    )

    weights = make_event_inverse_count_weights(events, exponent=exponent)

    expected = [normalization * counts[event] ** (-exponent) for event in events]
    assert weights == pytest.approx(expected)
    assert sum(weights) == pytest.approx(len(events))
    assert sum(weights[:3]) == pytest.approx(
        normalization * counts["A"] ** (1.0 - exponent)
    )
    assert weights[3] == pytest.approx(
        normalization * counts["B"] ** (1.0 - exponent)
    )
    assert sum(weights[:3]) > weights[3]


def test_event_balance_exponent_defaults_and_validates() -> None:
    assert resolve_event_balance_exponent({}) == 1.0
    assert resolve_event_balance_exponent(
        {
            "event_balanced_sampling": True,
            "event_balance_estimator": INVERSE_COUNT_FULL_DATA_ESTIMATOR,
            "event_balance_exponent": 0.5,
        }
    ) == 0.5

    for value in (True, "0.5", float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError, match="event_balance_exponent"):
            resolve_event_balance_exponent(
                {"event_balance_exponent": value}
            )

    for training in (
        {
            "event_balanced_sampling": False,
            "event_balance_exponent": 0.5,
        },
        {
            "event_balanced_sampling": True,
            "event_balance_estimator": REPLACEMENT_SAMPLING_ESTIMATOR,
            "event_balance_exponent": 0.5,
        },
    ):
        with pytest.raises(ValueError, match="requires.*inverse_count_full_data"):
            resolve_event_balance_exponent(training)

    assert resolve_event_balance_exponent(
        {
            "event_balanced_sampling": False,
            "event_balance_exponent": 1.0,
        }
    ) == 1.0


def test_event_balance_estimator_defaults_to_legacy_and_rejects_conflicts() -> None:
    assert resolve_event_balance_estimator(
        {"event_balanced_sampling": True}
    ) == REPLACEMENT_SAMPLING_ESTIMATOR
    assert resolve_event_balance_estimator(
        {
            "event_balanced_sampling": True,
            "event_balance_estimator": INVERSE_COUNT_FULL_DATA_ESTIMATOR,
        }
    ) == INVERSE_COUNT_FULL_DATA_ESTIMATOR
    with pytest.raises(ValueError, match="requires.*event_balanced_sampling=true"):
        resolve_event_balance_estimator(
            {
                "event_balanced_sampling": False,
                "event_balance_estimator": INVERSE_COUNT_FULL_DATA_ESTIMATOR,
            }
        )


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


def _sampled_stations(loader) -> list[str]:
    return [
        str(station)
        for batch in loader
        for station in batch["station"]
    ]


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


def test_grouped_loader_accepts_explicit_complete_event_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch)
    split = EventGroupSplit(
        train_indices=[0, 1, 2, 3, 4],
        validation_indices=[5, 6],
        test_indices=[7, 8, 9],
    )

    first = _loader_config("grouped_event")
    first["training"]["random_seed"] = 17
    _, _, _, first_manifest = get_data_loaders_v2(
        first,
        explicit_split=split,
    )
    second = _loader_config("grouped_event")
    second["training"]["random_seed"] = 73
    _, _, _, second_manifest = get_data_loaders_v2(
        second,
        explicit_split=split,
    )

    assert first_manifest["train_events"] == ["A", "B"]
    assert first_manifest["validation_events"] == ["C"]
    assert first_manifest["test_events"] == ["D", "E"]
    assert first_manifest["assignment_sha256"] == second_manifest[
        "assignment_sha256"
    ]
    assert first_manifest["seed"] == 17
    assert second_manifest["seed"] == 73


@pytest.mark.parametrize(
    ("split", "message"),
    [
        (
            EventGroupSplit(
                train_indices=[0, 1, 2, 3, 4],
                validation_indices=[5, 6],
                test_indices=[7, 8],
            ),
            "cover every dataset index",
        ),
        (
            EventGroupSplit(
                train_indices=[0, 1, 2, 3, 4],
                validation_indices=[4, 5, 6],
                test_indices=[7, 8, 9],
            ),
            "exactly one split",
        ),
        (
            EventGroupSplit(
                train_indices=[0, 1, 2, 3, 4],
                validation_indices=[5, 6],
                test_indices=[7, 8, 10],
            ),
            "out-of-range",
        ),
        (
            EventGroupSplit(
                train_indices=[0, 1, 2, 3],
                validation_indices=[4, 5, 6],
                test_indices=[7, 8, 9],
            ),
            "events overlap",
        ),
    ],
)
def test_grouped_loader_rejects_invalid_explicit_split(
    monkeypatch: pytest.MonkeyPatch,
    split: EventGroupSplit,
    message: str,
) -> None:
    _patch_dataset(monkeypatch)

    with pytest.raises(ValueError, match=message):
        get_data_loaders_v2(
            _loader_config("grouped_event"),
            explicit_split=split,
        )


def test_explicit_split_requires_grouped_event_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch)
    split = EventGroupSplit(
        train_indices=[0, 1, 2, 3, 4],
        validation_indices=[5, 6],
        test_indices=[7, 8, 9],
    )

    with pytest.raises(ValueError, match="only supported for grouped_event"):
        get_data_loaders_v2(
            _loader_config("within_event_station"),
            explicit_split=split,
        )


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


def test_within_event_station_balanced_sampler_is_event_equal_and_resumable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch)
    config = _loader_config("within_event_station")

    first, first_validation, first_test, _ = get_data_loaders_v2(config)
    assert isinstance(first.sampler, WeightedRandomSampler)
    assert first.sampler.replacement is True
    assert first.sampler.num_samples == len(first.dataset)
    assert not isinstance(first_validation.sampler, WeightedRandomSampler)
    assert not isinstance(first_test.sampler, WeightedRandomSampler)

    event_weight_sums: dict[str, float] = defaultdict(float)
    for subset_position, weight in enumerate(first.sampler.weights):
        dataset_index = first.dataset.indices[subset_position]
        event = first.dataset.dataset.samples[dataset_index]["event"]
        event_weight_sums[event] += float(weight)
    assert len(set(round(value, 12) for value in event_weight_sums.values())) == 1

    first_epoch = _sampled_stations(first)
    resume_state = first.generator.get_state().clone()
    second_epoch = _sampled_stations(first)

    replay, _, _, _ = get_data_loaders_v2(config)
    assert _sampled_stations(replay) == first_epoch
    assert _sampled_stations(replay) == second_epoch

    resumed, _, _, _ = get_data_loaders_v2(config)
    resumed.generator.set_state(resume_state)
    assert _sampled_stations(resumed) == second_epoch

    different_seed = _loader_config("within_event_station")
    different_seed["training"]["random_seed"] = 43
    other, _, _, _ = get_data_loaders_v2(different_seed)
    assert _sampled_stations(other) != first_epoch


def test_explicit_replacement_estimator_preserves_legacy_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch)
    legacy_config = _loader_config("grouped_event")
    explicit_config = _loader_config("grouped_event")
    explicit_config["training"][
        "event_balance_estimator"
    ] = REPLACEMENT_SAMPLING_ESTIMATOR

    legacy, _, _, _ = get_data_loaders_v2(legacy_config)
    explicit, _, _, _ = get_data_loaders_v2(explicit_config)

    assert isinstance(legacy.sampler, WeightedRandomSampler)
    assert isinstance(explicit.sampler, WeightedRandomSampler)
    torch.testing.assert_close(legacy.sampler.weights, explicit.sampler.weights)
    assert torch.equal(legacy.generator.get_state(), explicit.generator.get_state())
    assert _sampled_stations(legacy) == _sampled_stations(explicit)


def test_inverse_count_full_data_sampler_covers_all_records_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch)
    config = _loader_config("grouped_event")
    config["training"][
        "event_balance_estimator"
    ] = INVERSE_COUNT_FULL_DATA_ESTIMATOR

    first, validation, test, _ = get_data_loaders_v2(config)

    assert isinstance(first.sampler, RandomSampler)
    assert first.sampler.replacement is False
    assert first.sampler.num_samples == len(first.dataset)
    assert not isinstance(validation.sampler, WeightedRandomSampler)
    assert not isinstance(test.sampler, WeightedRandomSampler)
    assert not hasattr(validation, "event_balance_weights_by_event")
    assert not hasattr(test, "event_balance_weights_by_event")

    weights_by_event = first.event_balance_weights_by_event
    assert weights_by_event is not None
    event_mass: dict[str, float] = defaultdict(float)
    for dataset_index in first.dataset.indices:
        event = first.dataset.dataset.samples[dataset_index]["event"]
        event_mass[event] += weights_by_event[event]
    expected_event_mass = len(first.dataset) / len(event_mass)
    assert list(event_mass.values()) == pytest.approx(
        [expected_event_mass] * len(event_mass)
    )
    assert sum(event_mass.values()) == pytest.approx(len(first.dataset))

    first_epoch = _sampled_stations(first)
    assert len(first_epoch) == len(first.dataset)
    assert len(set(first_epoch)) == len(first.dataset)
    resume_state = first.generator.get_state().clone()
    second_epoch = _sampled_stations(first)
    assert len(set(second_epoch)) == len(first.dataset)

    replay, _, _, _ = get_data_loaders_v2(config)
    assert _sampled_stations(replay) == first_epoch
    assert _sampled_stations(replay) == second_epoch

    resumed, _, _, _ = get_data_loaders_v2(config)
    resumed.generator.set_state(resume_state)
    assert _sampled_stations(resumed) == second_epoch


def test_inverse_count_exponent_one_loader_is_exactly_backward_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch)
    default_config = _loader_config("grouped_event")
    default_config["training"][
        "event_balance_estimator"
    ] = INVERSE_COUNT_FULL_DATA_ESTIMATOR
    explicit_config = _loader_config("grouped_event")
    explicit_config["training"].update(
        event_balance_estimator=INVERSE_COUNT_FULL_DATA_ESTIMATOR,
        event_balance_exponent=1.0,
    )

    default, _, _, _ = get_data_loaders_v2(default_config)
    explicit, _, _, _ = get_data_loaders_v2(explicit_config)

    assert default.event_balance_exponent == 1.0
    assert explicit.event_balance_exponent == 1.0
    assert (
        default.event_balance_weights_by_event
        == explicit.event_balance_weights_by_event
    )
    assert torch.equal(
        default.generator.get_state(),
        explicit.generator.get_state(),
    )


def test_tempered_inverse_count_weights_are_train_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch)
    baseline_config = _loader_config("grouped_event")
    baseline_config["training"][
        "event_balance_estimator"
    ] = INVERSE_COUNT_FULL_DATA_ESTIMATOR
    config = _loader_config("grouped_event")
    config["training"].update(
        event_balance_estimator=INVERSE_COUNT_FULL_DATA_ESTIMATOR,
        event_balance_exponent=0.5,
    )

    baseline_train, baseline_validation, baseline_test, baseline_manifest = (
        get_data_loaders_v2(baseline_config)
    )
    train, validation, test, manifest = get_data_loaders_v2(config)

    assert train.event_balance_exponent == 0.5
    assert manifest == baseline_manifest
    assert validation.dataset.indices == baseline_validation.dataset.indices
    assert test.dataset.indices == baseline_test.dataset.indices
    assert torch.equal(
        train.generator.get_state(),
        baseline_train.generator.get_state(),
    )
    assert not hasattr(validation, "event_balance_exponent")
    assert not hasattr(test, "event_balance_exponent")
    assert not hasattr(validation, "event_balance_weights_by_event")
    assert not hasattr(test, "event_balance_weights_by_event")

    events = [
        str(train.dataset.dataset.samples[index]["event"])
        for index in train.dataset.indices
    ]
    counts = Counter(events)
    assert len(set(counts.values())) > 1
    normalization = len(events) / sum(
        count ** 0.5 for count in counts.values()
    )
    expected_weights = {
        event: normalization * count ** -0.5
        for event, count in counts.items()
    }
    assert train.event_balance_weights_by_event == pytest.approx(
        expected_weights
    )
    event_masses = {
        event: count * train.event_balance_weights_by_event[event]
        for event, count in counts.items()
    }
    assert sum(event_masses.values()) == pytest.approx(len(events))
    assert event_masses == pytest.approx(
        {
            event: normalization * count ** 0.5
            for event, count in counts.items()
        }
    )
    assert min(event_masses.values()) < max(event_masses.values())


def test_loader_rejects_inverse_count_estimator_when_balancing_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch)
    config = _loader_config("grouped_event")
    config["training"].update(
        event_balanced_sampling=False,
        event_balance_estimator=INVERSE_COUNT_FULL_DATA_ESTIMATOR,
    )

    with pytest.raises(ValueError, match="requires.*event_balanced_sampling=true"):
        get_data_loaders_v2(config)


def test_loader_rejects_tempered_exponent_for_replacement_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch)
    config = _loader_config("grouped_event")
    config["training"]["event_balance_exponent"] = 0.5

    with pytest.raises(ValueError, match="requires.*inverse_count_full_data"):
        get_data_loaders_v2(config)
