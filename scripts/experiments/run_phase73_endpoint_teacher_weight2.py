#!/usr/bin/env python3
"""Run Phase73 with stronger final Phase39 endpoint supervision."""

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
from scripts.experiments.run_phase72_post160_band_weight025 import (  # noqa: E402
    configure_phase72,
)


PHASE73_ENDPOINT_TEACHER_WEIGHT = 2.0


def configure_phase73() -> None:
    configure_phase72()
    campaign.EXPERIMENT_PHASE = "Phase73"
    campaign.LOSS_WEIGHTS = {
        **campaign.LOSS_WEIGHTS,
        "endpoint_teacher": PHASE73_ENDPOINT_TEACHER_WEIGHT,
    }


def main(argv: Sequence[str] | None = None) -> int:
    configure_phase73()
    return campaign.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
