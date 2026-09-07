from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments import run_phase39_causal_direct as direct  # noqa: E402
from scripts.experiments import run_phase39_expanded_fixed_split as fixed  # noqa: E402
from src.data.dataset_v2 import CorrectedEarthquakeDataset  # noqa: E402
from src.data.loaders_v2 import get_data_loaders_v2  # noqa: E402
from src.utils.config_v2 import validate_config_on_startup  # noqa: E402
from src.utils.provenance import sha256_file  # noqa: E402


DEFAULT_EXPERIMENT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "phase39_causal_expanded_fixed_split.yaml"
)
EXPECTED_ENDPOINT_CHECKPOINT_SHA256 = (
    "5905ccaabcbcfc151d6f2dfa8aea277725d409fb62bd16c437ac5288bcafb4fe"
)


def _load_fixed_source(
    experiment: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    summary_path = Path(
        str(experiment["fixed_endpoint_candidate_summary"])
    ).resolve()
    summary = direct._read_json(summary_path)
    seed = int(experiment["seed"])
    required = {
        "status": "complete",
        "seed": seed,
        "test_evaluated": False,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"fixed endpoint source {key} changed: "
                f"expected={expected!r}, actual={summary.get(key)!r}"
            )

    checkpoint_path = Path(str(summary["best_model_path"])).resolve()
    config_path = Path(str(summary["config_snapshot_path"])).resolve()
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != str(summary["checkpoint_sha256"]):
        raise ValueError("fixed endpoint checkpoint hash changed")
    if checkpoint_hash != EXPECTED_ENDPOINT_CHECKPOINT_SHA256:
        raise ValueError("unexpected fixed endpoint comparison checkpoint")
    if sha256_file(config_path) != str(summary["config_snapshot_sha256"]):
        raise ValueError("fixed endpoint config hash changed")

    config = direct._read_yaml(config_path)
    validate_config_on_startup(config)
    loss = config["training"]["stf_rate_loss"]
    required_loss = {
        "lambda_synth": 0.5,
        "synth_polarity_mode": "global_invariant",
        "radiation_coefficient_contract": "glehman_scalar",
    }
    for key, expected in required_loss.items():
        if loss.get(key) != expected:
            raise ValueError(f"fixed endpoint loss contract changed: {key}")

    samples = [
        dict(sample)
        for sample in CorrectedEarthquakeDataset(copy.deepcopy(config)).samples
    ]
    split, fixed_manifest = fixed.build_fixed_split(samples)
    expected_assignment = str(experiment["expected_split_assignment_sha256"])
    if fixed_manifest["assignment_sha256"] != expected_assignment:
        raise ValueError("expanded fixed split assignment changed")
    if str(summary["split_assignment_sha256"]) != expected_assignment:
        raise ValueError("endpoint candidate split assignment changed")

    train_loader, validation_loader, test_loader, loader_manifest = (
        get_data_loaders_v2(config, explicit_split=split)
    )
    if loader_manifest["assignment_sha256"] != expected_assignment:
        raise ValueError("expanded fixed loader assignment changed")
    if set(loader_manifest["train_events"]) != set(fixed.TRAIN_EVENTS):
        raise ValueError("expanded fixed training events changed")
    if set(loader_manifest["validation_events"]) != set(
        fixed.VALIDATION_EVENTS
    ):
        raise ValueError("expanded fixed validation events changed")
    if set(loader_manifest["test_events"]) != set(fixed.TEST_EVENTS):
        raise ValueError("expanded fixed test events changed")
    del test_loader

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return {
        "summary_path": summary_path,
        "summary": summary,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "config": config,
        "state": state,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "split_manifest": loader_manifest,
    }


def _enrich_protocol(output_root: Path) -> None:
    protocol_path = output_root / "protocol.json"
    protocol = direct._read_json(protocol_path)
    protocol.update(
        {
            "split_protocol": "expanded_fixed_24_train_6_validation_9_test",
            "development_role": "fixed_validation_only",
            "train_event_count": len(fixed.TRAIN_EVENTS),
            "validation_event_count": len(fixed.VALIDATION_EVENTS),
            "held_out_test_event_count": len(fixed.TEST_EVENTS),
            "held_out_test_loader_iterated": False,
            "compatibility_fold_field_only": True,
        }
    )
    direct._write_json(protocol_path, protocol)


def run_experiment(
    *,
    experiment_config_path: Path,
    output_root: Path,
    smoke: bool,
    device: torch.device,
) -> dict[str, Any]:
    original_loader = direct._load_source
    direct._load_source = _load_fixed_source
    try:
        summary = direct.run_experiment(
            experiment_config_path=experiment_config_path,
            output_root=output_root,
            smoke=smoke,
            device=device,
        )
    finally:
        direct._load_source = original_loader
    _enrich_protocol(output_root)
    summary["split_protocol"] = "expanded_fixed_24_train_6_validation_9_test"
    summary["development_role"] = "fixed_validation_only"
    summary["held_out_test_loader_iterated"] = False
    direct._write_json(output_root / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train true causal-prefix Phase 39 on the frozen expanded fixed "
            "training and validation events without iterating the test cohort."
        )
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root or (
        Path("/home/lihe/PINN_Mag/runs")
        / f"phase39-causal-expanded-fixed-{timestamp}"
    )
    summary = run_experiment(
        experiment_config_path=args.experiment_config.resolve(),
        output_root=output_root.resolve(),
        smoke=bool(args.smoke),
        device=device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
