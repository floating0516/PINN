from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


RUN_MANIFEST_FIELDS = (
    "pipeline_version",
    "git_commit",
    "git_dirty",
    "config_sha256",
    "dataset_manifest_sha256",
    "split_sha256",
    "checkpoint_sha256",
    "python_version",
    "torch_version",
    "numpy_version",
    "random_seed",
    "started_at_utc",
    "completed_at_utc",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_if_file(path: str | Path | None) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    return sha256_file(candidate) if candidate.is_file() else ""


def current_git_commit(root: str | Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(root),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_is_dirty(root: str | Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(root),
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    return output


def file_reference(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {"path": "", "sha256": ""}
    candidate = Path(path)
    return {
        "path": str(candidate),
        "sha256": sha256_if_file(candidate),
    }


def configured_dataset_manifest_path(
    config: dict[str, Any],
    *,
    root: str | Path,
) -> Path | None:
    value = (config.get("paths", {}) or {}).get("dataset_manifest_path")
    if value is None:
        value = (config.get("dataset", {}) or {}).get("manifest_path")
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def split_protocol_from_manifest(
    split_manifest_path: str | Path | None,
    *,
    configured_protocol: str,
) -> str:
    if split_manifest_path is None or not Path(split_manifest_path).is_file():
        return str(configured_protocol)
    with Path(split_manifest_path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    protocol = payload.get("protocol") if isinstance(payload, dict) else None
    return str(protocol) if protocol else str(configured_protocol)
