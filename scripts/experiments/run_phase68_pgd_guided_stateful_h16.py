#!/usr/bin/env python3
"""Run the Phase68 hidden-size-only PGD-guided stateful iteration."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments import (  # noqa: E402
    run_phase67_pgd_guided_stateful as campaign,
)


PHASE68_HIDDEN_SIZE = 16
PHASE68_TRANSITION_PARAMETER_COUNT = 2_958


def configure_phase68() -> None:
    campaign.EXPERIMENT_PHASE = "Phase68"
    campaign.STATE_HIDDEN_SIZE = PHASE68_HIDDEN_SIZE
    campaign.EXPECTED_TRANSITION_PARAMETER_COUNT = (
        PHASE68_TRANSITION_PARAMETER_COUNT
    )
    campaign.EXPECTED_TOTAL_PARAMETER_COUNT = (
        campaign.EXPECTED_BASE_PARAMETER_COUNT
        + PHASE68_TRANSITION_PARAMETER_COUNT
    )


def main(argv: Sequence[str] | None = None) -> int:
    configure_phase68()
    return campaign.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
