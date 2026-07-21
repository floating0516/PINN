#!/usr/bin/env python3
"""Rename e1_4 logs and e1_5 directories from timestamps to semantic names.

Mapping derived from experiment_summary.csv in each experiment.
Run with --dry-run first to preview changes.
"""

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "outputs_experiments"


# ── e1_4: only logs need renaming ──────────────────────────────────────────
E1_4_LOG_MAP = {
    # timestamp → semantic name (matching models/results dirs)
    "training_log_20260416_100421.csv": "training_log_lm015_int.csv",
    "training_log_20260416_105255.csv": "training_log_lm020_int.csv",
    "training_log_20260416_111313.csv": "training_log_lm030_int.csv",
    "training_log_20260416_114037.csv": "training_log_lm010_noint.csv",
    "training_log_20260416_120553.csv": "training_log_lm015_noint.csv",
}

# ── e1_5: logs, models, results all need renaming ─────────────────────────
E1_5_MAP = {
    # timestamp → semantic name
    "20260416_142216": "ls010_simplified",
    "20260416_144200": "ls030_simplified",
    "20260416_150233": "ls050_simplified",
    "20260416_151630": "ls070_simplified",
    "20260416_153015": "ls100_simplified",
    "20260416_155458": "ls010_full",
    "20260416_162150": "ls030_full",
    "20260416_164717": "ls050_full",
    "20260416_171247": "ls070_full",
    "20260416_173747": "ls100_full",
}


def rename_e1_4_logs(dry_run: bool) -> None:
    logs_dir = ROOT / "e1_4" / "logs"
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}=== e1_4 logs ===")
    for old_name, new_name in E1_4_LOG_MAP.items():
        src = logs_dir / old_name
        dst = logs_dir / new_name
        if not src.exists():
            print(f"  SKIP (not found): {src.name}")
            continue
        if dst.exists():
            print(f"  SKIP (target exists): {dst.name}")
            continue
        print(f"  {src.name}  →  {dst.name}")
        if not dry_run:
            src.rename(dst)


def rename_e1_5_dirs(dry_run: bool) -> None:
    base = ROOT / "e1_5"
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}=== e1_5 logs / models / results ===")
    for subdir_name in ("logs", "models", "results"):
        subdir = base / subdir_name
        if not subdir.exists():
            continue
        for ts, semantic in E1_5_MAP.items():
            if subdir_name == "logs":
                old_name = f"training_log_{ts}.csv"
                new_name = f"training_log_{semantic}.csv"
            else:
                old_name = ts
                new_name = semantic

            src = subdir / old_name
            dst = subdir / new_name
            if not src.exists():
                continue
            if dst.exists():
                print(f"  SKIP (target exists): {subdir_name}/{new_name}")
                continue
            print(f"  {subdir_name}/{old_name}  →  {subdir_name}/{new_name}")
            if not dry_run:
                src.rename(dst)


def update_e1_4_summary(dry_run: bool) -> None:
    """Update experiment_summary.csv: run_id and models_dir columns."""
    csv_path = ROOT / "e1_4" / "results" / "experiment_summary.csv"
    if not csv_path.exists():
        return
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}=== e1_4 experiment_summary.csv ===")

    # Build reverse map: timestamp → semantic name
    ts_to_sem = {
        "20260416_100421": "lm015_int",
        "20260416_105255": "lm020_int",
        "20260416_111313": "lm030_int",
        "20260416_114037": "lm010_noint",
        "20260416_120553": "lm015_noint",
    }

    text = csv_path.read_text()
    new_text = text
    for ts, sem in ts_to_sem.items():
        old_models_dir = f"outputs_experiments/e1_4/models/{ts}"
        new_models_dir = f"outputs_experiments/e1_4/models/{sem}"
        if old_models_dir in new_text:
            print(f"  models_dir: {ts} → {sem}")
            new_text = new_text.replace(old_models_dir, new_models_dir)

    if new_text != text and not dry_run:
        csv_path.write_text(new_text)


def update_e1_5_summary(dry_run: bool) -> None:
    """Update experiment_summary.csv: run_id and models_dir columns."""
    csv_path = ROOT / "e1_5" / "results" / "experiment_summary.csv"
    if not csv_path.exists():
        return
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}=== e1_5 experiment_summary.csv ===")

    text = csv_path.read_text()
    new_text = text
    for ts, sem in E1_5_MAP.items():
        # Update run_id column
        if ts in new_text:
            print(f"  {ts} → {sem}")
            new_text = new_text.replace(ts, sem)

    if new_text != text and not dry_run:
        csv_path.write_text(new_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview renames without executing them",
    )
    args = parser.parse_args()

    rename_e1_4_logs(args.dry_run)
    rename_e1_5_dirs(args.dry_run)
    update_e1_4_summary(args.dry_run)
    update_e1_5_summary(args.dry_run)

    if args.dry_run:
        print("\n✓ Dry-run complete. Re-run without --dry-run to apply.")
    else:
        print("\n✓ All renames applied.")


if __name__ == "__main__":
    main()
