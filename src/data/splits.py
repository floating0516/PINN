from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np


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
