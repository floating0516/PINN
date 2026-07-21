#!/usr/bin/env python3
"""Migrate unseen evaluation results to unified structure.

Changes:
1. e1_1_v2 / e1_2_v2 / e1_3_v2:
   - Move results/{run}/unseen_8events_cm* → unseen/{run}/unseen_events_8_*
   - Move results/unseen_*_summary.csv   → unseen/*_summary.csv
   - Rename cm0→all_stations, cm1→1cm, cm2→2cm

2. e1_4:
   - Rename unseen/ run dirs from experiment names to model-dir names
     (farOnly_lp100→lm010_noint, etc.)

3. e1_5: already correct after prior rename.

Run with --dry-run first to preview.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "outputs_experiments"

# ── subfolder name mapping ─────────────────────────────────────────────────
CM_MAP = {
    "unseen_8events_cm0": "unseen_events_8_all_stations",
    "unseen_8events_cm1": "unseen_events_8_1cm",
    "unseen_8events_cm2": "unseen_events_8_2cm",
    "unseen_2cm":         "unseen_events_8_2cm",  # legacy alias in lp040_simplified
}

SUMMARY_MAP = {
    "unseen_8events_cm0_summary.csv": "unseen_events_8_all_stations_summary.csv",
    "unseen_8events_cm1_summary.csv": "unseen_events_8_1cm_summary.csv",
    "unseen_8events_cm2_summary.csv": "unseen_events_8_2cm_summary.csv",
}

# ── e1_4 run-name mapping ─────────────────────────────────────────────────
E1_4_RUN_MAP = {
    "farOnly_lp100": "lm010_noint",
    "farOnly_lp150": "lm015_noint",
    "full_lp150":    "lm015_int",
    "full_lp200":    "lm020_int",
    "full_lp300":    "lm030_int",
}


def migrate_123(exp_name: str, dry_run: bool) -> None:
    """Move unseen dirs from results/{run}/ into unseen/{run}/."""
    base = ROOT / exp_name
    results_dir = base / "results"
    unseen_dir = base / "unseen"

    if not results_dir.exists():
        return

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}=== {exp_name}: move unseen dirs ===")

    # Find all run directories under results/
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for old_name, new_name in CM_MAP.items():
            src = run_dir / old_name
            if not src.exists():
                continue
            dst = unseen_dir / run_dir.name / new_name
            if dst.exists():
                print(f"  SKIP (target exists): unseen/{run_dir.name}/{new_name}")
                continue
            print(f"  results/{run_dir.name}/{old_name}  →  unseen/{run_dir.name}/{new_name}")
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

    # Move summary CSVs
    for old_name, new_name in SUMMARY_MAP.items():
        src = results_dir / old_name
        if not src.exists():
            continue
        dst = unseen_dir / new_name
        if dst.exists():
            print(f"  SKIP (target exists): unseen/{new_name}")
            continue
        print(f"  results/{old_name}  →  unseen/{new_name}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))


def rename_e1_4_unseen(dry_run: bool) -> None:
    """Rename e1_4 unseen run dirs to match model-dir naming."""
    unseen_dir = ROOT / "e1_4" / "unseen"
    if not unseen_dir.exists():
        return

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}=== e1_4: rename unseen run dirs ===")
    for old_name, new_name in E1_4_RUN_MAP.items():
        src = unseen_dir / old_name
        dst = unseen_dir / new_name
        if not src.exists():
            continue
        if dst.exists():
            print(f"  SKIP (target exists): {new_name}")
            continue
        print(f"  unseen/{old_name}  →  unseen/{new_name}")
        if not dry_run:
            src.rename(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without executing them")
    args = parser.parse_args()

    migrate_123("e1_1", args.dry_run)
    migrate_123("e1_2", args.dry_run)
    migrate_123("e1_3", args.dry_run)
    rename_e1_4_unseen(args.dry_run)

    if args.dry_run:
        print("\n✓ Dry-run complete. Re-run without --dry-run to apply.")
    else:
        print("\n✓ All migrations applied.")


if __name__ == "__main__":
    main()
