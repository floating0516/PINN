#!/usr/bin/env python3
"""Run Phase71 with a gate-aligned post-160 magnitude-band loss."""

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
from scripts.experiments.run_phase70_pgd_assimilation_start140 import (  # noqa: E402
    configure_phase70,
)


PHASE71_BAND_LOSS_WEIGHT = 1.0
PHASE71_BAND_NORMALIZER_FLOOR = 1.0e-3


def configure_phase71() -> None:
    configure_phase70()
    campaign.EXPERIMENT_PHASE = "Phase71"
    campaign.USE_PLATEAU_BAND_LOSS = True
    campaign.LOSS_WEIGHTS = {
        **campaign.LOSS_WEIGHTS,
        "plateau_band": PHASE71_BAND_LOSS_WEIGHT,
    }
    campaign.NORMALIZER_FLOORS = {
        **campaign.NORMALIZER_FLOORS,
        "plateau_band": PHASE71_BAND_NORMALIZER_FLOOR,
    }


def main(argv: Sequence[str] | None = None) -> int:
    configure_phase71()
    return campaign.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
