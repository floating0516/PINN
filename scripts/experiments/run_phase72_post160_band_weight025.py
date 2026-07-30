#!/usr/bin/env python3
"""Run Phase72 with a lighter post-160 magnitude-band objective."""

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
from scripts.experiments.run_phase71_post160_band_loss import (  # noqa: E402
    configure_phase71,
)


PHASE72_BAND_LOSS_WEIGHT = 0.25


def configure_phase72() -> None:
    configure_phase71()
    campaign.EXPERIMENT_PHASE = "Phase72"
    campaign.LOSS_WEIGHTS = {
        **campaign.LOSS_WEIGHTS,
        "plateau_band": PHASE72_BAND_LOSS_WEIGHT,
    }


def main(argv: Sequence[str] | None = None) -> int:
    configure_phase72()
    return campaign.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
