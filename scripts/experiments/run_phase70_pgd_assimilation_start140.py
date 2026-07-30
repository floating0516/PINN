#!/usr/bin/env python3
"""Run Phase70 with the late proposal assimilation starting at 140 s."""

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
from scripts.experiments.run_phase69_pgd_guided_late_assimilation import (  # noqa: E402
    PHASE69_TRANSITION_PARAMETER_COUNT,
    configure_phase69,
)


PHASE70_ASSIMILATION_START_SEC = 140


def configure_phase70() -> None:
    configure_phase69()
    campaign.EXPERIMENT_PHASE = "Phase70"
    campaign.LATE_PROPOSAL_ASSIMILATION_START_SEC = (
        PHASE70_ASSIMILATION_START_SEC
    )
    campaign.EXPECTED_TRANSITION_PARAMETER_COUNT = (
        PHASE69_TRANSITION_PARAMETER_COUNT
    )
    campaign.EXPECTED_TOTAL_PARAMETER_COUNT = (
        campaign.EXPECTED_BASE_PARAMETER_COUNT
        + PHASE69_TRANSITION_PARAMETER_COUNT
    )


def main(argv: Sequence[str] | None = None) -> int:
    configure_phase70()
    return campaign.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
