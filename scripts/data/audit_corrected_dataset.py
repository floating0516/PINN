from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_v2 import CorrectedEarthquakeDataset  # noqa: E402
from src.data.manifest import audit_passes, write_dataset_audit  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the corrected v2 dataset and write its manifest.",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser


def _resolve_data_path(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    repository_candidate = PROJECT_ROOT / path
    if repository_candidate.exists():
        return str(repository_candidate)

    data_root_value = os.environ.get("PINN_DATA_ROOT")
    if data_root_value:
        relative_parts = path.parts[1:] if path.parts[:1] == ("data",) else path.parts
        data_candidate = Path(data_root_value).joinpath(*relative_parts)
        if data_candidate.exists():
            return str(data_candidate)
    return str(repository_candidate)


def _load_runtime_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("config must contain a mapping")
    config = copy.deepcopy(loaded)
    config["paths"]["data_path"] = _resolve_data_path(
        config["paths"]["data_path"]
    )
    config["dataset"]["stf"]["path"] = _resolve_data_path(
        config["dataset"]["stf"]["path"]
    )
    return config


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _load_runtime_config(args.config.resolve())
    dataset = CorrectedEarthquakeDataset(config)
    summary = write_dataset_audit(
        dataset,
        manifest_path=args.manifest,
        summary_path=args.summary,
    )
    passed = audit_passes(summary)
    print(
        json.dumps(
            {
                "accepted_station_count": summary["accepted_station_count"],
                "candidate_station_count": summary["candidate_station_count"],
                "invariants": summary["invariants"],
                "passed": passed,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
