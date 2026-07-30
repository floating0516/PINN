#!/usr/bin/env python3
"""Run Phase74 with stronger late-only Phase39 proposal assimilation."""

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


PHASE74_PROPOSAL_ASSIMILATION_SCALE = 1.5


def configure_phase74() -> None:
    configure_phase72()
    campaign.EXPERIMENT_PHASE = "Phase74"
    campaign.PROPOSAL_ASSIMILATION_SCALE = (
        PHASE74_PROPOSAL_ASSIMILATION_SCALE
    )


def main(argv: Sequence[str] | None = None) -> int:
    configure_phase74()
    return campaign.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
