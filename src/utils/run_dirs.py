from __future__ import annotations

from datetime import datetime
from pathlib import Path


RUN_ID_FORMAT = "%Y%m%d_%H%M%S"


def make_run_id() -> str:
    return datetime.now().strftime(RUN_ID_FORMAT)


def create_run_dir(base_dir: str | Path, run_id: str | None = None) -> tuple[str, Path]:
    resolved_run_id = run_id or make_run_id()
    run_dir = Path(base_dir) / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return resolved_run_id, run_dir


def get_latest_run_dir(base_dir: str | Path) -> Path | None:
    root = Path(base_dir)
    if not root.exists():
        return None
    run_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def resolve_model_path(models_base_dir: str | Path, filename: str = "best_model.pth") -> Path | None:
    latest_run_dir = get_latest_run_dir(models_base_dir)
    if latest_run_dir is not None:
        candidate = latest_run_dir / filename
        if candidate.exists():
            return candidate
    fallback = Path(models_base_dir) / filename
    if fallback.exists():
        return fallback
    return None
