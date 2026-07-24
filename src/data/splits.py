from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np


REPLACEMENT_SAMPLING_ESTIMATOR = "replacement_sampling"
INVERSE_COUNT_FULL_DATA_ESTIMATOR = "inverse_count_full_data"
EVENT_BALANCE_ESTIMATORS = frozenset(
    {
        REPLACEMENT_SAMPLING_ESTIMATOR,
        INVERSE_COUNT_FULL_DATA_ESTIMATOR,
    }
)


@dataclass(frozen=True)
class EventGroupSplit:
    train_indices: list[int]
    validation_indices: list[int]
    test_indices: list[int]


def _requested_group_count(total: int, fraction: float) -> int:
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("split fractions must be finite and in [0, 1]")
    if fraction == 0.0:
        return 0
    return max(1, int(round(total * fraction)))


def make_event_group_split(
    events: list[str],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> EventGroupSplit:
    if not events:
        raise ValueError("events must not be empty")
    unique_events = sorted(set(events))
    permutation = np.random.default_rng(seed).permutation(unique_events).tolist()
    validation_count = _requested_group_count(
        len(unique_events), validation_fraction
    )
    test_count = _requested_group_count(len(unique_events), test_fraction)
    maximum_held_out = len(unique_events) - 1
    while validation_count + test_count > maximum_held_out:
        if validation_count >= test_count and validation_count > 0:
            validation_count -= 1
        elif test_count > 0:
            test_count -= 1

    test_events = set(permutation[:test_count])
    validation_events = set(
        permutation[test_count : test_count + validation_count]
    )
    train_events = set(permutation) - test_events - validation_events
    split = EventGroupSplit(
        train_indices=[
            index for index, event in enumerate(events) if event in train_events
        ],
        validation_indices=[
            index
            for index, event in enumerate(events)
            if event in validation_events
        ],
        test_indices=[
            index for index, event in enumerate(events) if event in test_events
        ],
    )
    assert_no_event_overlap(events, split)
    return split


def make_within_event_station_split(
    sample_keys: list[tuple[str, str]],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> EventGroupSplit:
    if not sample_keys:
        raise ValueError("sample_keys must not be empty")
    normalized_keys = [
        (str(event), str(station)) for event, station in sample_keys
    ]
    if len(set(normalized_keys)) != len(normalized_keys):
        raise ValueError("duplicate event/station sample key")

    by_event: dict[str, list[tuple[str, int]]] = {}
    for index, (event, station) in enumerate(normalized_keys):
        by_event.setdefault(event, []).append((station, index))

    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for event in sorted(by_event):
        ordered = sorted(by_event[event])
        permutation = rng.permutation(len(ordered)).tolist()
        shuffled = [ordered[position][1] for position in permutation]
        maximum_held_out = max(0, len(shuffled) - 1)
        validation_count = _requested_group_count(
            len(shuffled), validation_fraction
        )
        test_count = _requested_group_count(len(shuffled), test_fraction)
        while validation_count + test_count > maximum_held_out:
            if validation_count >= test_count and validation_count > 0:
                validation_count -= 1
            elif test_count > 0:
                test_count -= 1

        test.extend(shuffled[:test_count])
        validation.extend(
            shuffled[test_count : test_count + validation_count]
        )
        train.extend(shuffled[test_count + validation_count :])

    return EventGroupSplit(
        train_indices=sorted(train),
        validation_indices=sorted(validation),
        test_indices=sorted(test),
    )


def assert_no_event_overlap(
    events: list[str],
    split: EventGroupSplit,
) -> None:
    train_events = {events[index] for index in split.train_indices}
    validation_events = {
        events[index] for index in split.validation_indices
    }
    test_events = {events[index] for index in split.test_indices}
    if train_events & validation_events:
        raise AssertionError("train and validation events overlap")
    if train_events & test_events:
        raise AssertionError("train and test events overlap")
    if validation_events & test_events:
        raise AssertionError("validation and test events overlap")


def make_event_balanced_weights(events: list[str]) -> list[float]:
    counts = Counter(events)
    return [1.0 / counts[event] for event in events]


def resolve_event_balance_estimator(training: Mapping[str, object]) -> str:
    if "event_balance_estimator" not in training:
        estimator = REPLACEMENT_SAMPLING_ESTIMATOR
    else:
        value = training["event_balance_estimator"]
        if not isinstance(value, str) or value not in EVENT_BALANCE_ESTIMATORS:
            raise ValueError(
                "training.event_balance_estimator must be one of: "
                + ", ".join(sorted(EVENT_BALANCE_ESTIMATORS))
            )
        estimator = value

    enabled = training.get("event_balanced_sampling", False)
    if not isinstance(enabled, bool):
        raise ValueError("training.event_balanced_sampling must be boolean")
    if not enabled and estimator != REPLACEMENT_SAMPLING_ESTIMATOR:
        raise ValueError(
            "training.event_balance_estimator=inverse_count_full_data "
            "requires training.event_balanced_sampling=true"
        )
    return estimator


def make_event_inverse_count_weights(events: list[str]) -> list[float]:
    if not events:
        raise ValueError("events must not be empty")
    counts = Counter(events)
    normalization = len(events) / len(counts)
    return [normalization / counts[event] for event in events]
