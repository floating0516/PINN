from __future__ import annotations

import math

import pytest

from scripts.evaluation import run_unseen_eval


def test_summary_uses_explicit_catalog_error_field() -> None:
    metrics = run_unseen_eval.summarize_event_error_metrics(
        [
            {"error_vs_catalog": -0.2},
            {"error_vs_catalog": 0.4},
        ]
    )

    assert metrics["mae"] == pytest.approx(0.3)
    assert metrics["rmse"] == pytest.approx(math.sqrt(0.1))
    assert metrics["bias"] == pytest.approx(0.1)
