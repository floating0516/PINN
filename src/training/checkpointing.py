from __future__ import annotations

import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import random
import signal
import tempfile
from typing import Any, Iterator

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel


CHECKPOINT_FORMAT_VERSION = 1

_REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "format_version",
        "created_at_utc",
        "reason",
        "signal_number",
        "completed_epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "early_stop_state",
        "swa_state_dict",
        "rng_state",
        "run_state",
        "provenance",
    }
)


class CheckpointValidationError(ValueError):
    pass


class TrainingInterrupted(SystemExit):
    def __init__(self, signum: int, checkpoint_path: str | Path) -> None:
        self.signum = int(signum)
        self.checkpoint_path = Path(checkpoint_path)
        super().__init__(128 + self.signum)


class TrainingSignalState:
    def __init__(self) -> None:
        self.pending_signal: int | None = None

    def request(self, signum: int) -> None:
        if self.pending_signal is None:
            self.pending_signal = int(signum)

    def checkpoint_and_raise(
        self,
        last_checkpoint: str | Path,
        emergency_checkpoint: str | Path,
    ) -> None:
        if self.pending_signal is None:
            return
        write_emergency_checkpoint(
            last_checkpoint,
            emergency_checkpoint,
            self.pending_signal,
        )
        raise TrainingInterrupted(self.pending_signal, emergency_checkpoint)


@contextmanager
def install_training_signal_handlers(
    state: TrainingSignalState,
) -> Iterator[None]:
    previous_handlers: dict[int, Any] = {}

    def _record_signal(signum: int, _frame: Any) -> None:
        state.request(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _record_signal)
    try:
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _numpy_rng_state() -> dict[str, Any]:
    bit_generator, state, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": bit_generator,
        "state": torch.from_numpy(state.copy()),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def capture_rng_state(
    loader_generator: torch.Generator | None,
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": _numpy_rng_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
        "loader": loader_generator.get_state().clone()
        if loader_generator is not None
        else None,
    }


def restore_rng_state(
    state: dict[str, Any],
    loader_generator: torch.Generator | None,
) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "loader"}
    missing = required.difference(state)
    if missing:
        raise CheckpointValidationError(
            f"RNG state missing keys: {', '.join(sorted(missing))}"
        )

    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())

    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise CheckpointValidationError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise CheckpointValidationError(
                "checkpoint CUDA RNG state count does not match visible devices"
            )
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])

    loader_state = state["loader"]
    if loader_state is None:
        if loader_generator is not None:
            raise CheckpointValidationError(
                "checkpoint has no train loader generator state"
            )
    elif loader_generator is None:
        raise CheckpointValidationError(
            "checkpoint requires a train loader generator"
        )
    else:
        loader_generator.set_state(loader_state.cpu())


def build_full_checkpoint(
    *,
    completed_epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    early_stop_state: dict[str, Any],
    swa_model: AveragedModel | None,
    loader_generator: torch.Generator | None,
    run_state: dict[str, Any],
    provenance: dict[str, Any],
    reason: str,
    signal_number: int | None = None,
) -> dict[str, Any]:
    if completed_epoch < 0:
        raise ValueError("completed_epoch must be non-negative")
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": str(reason),
        "signal_number": signal_number,
        "completed_epoch": int(completed_epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "early_stop_state": copy.deepcopy(early_stop_state),
        "swa_state_dict": swa_model.state_dict() if swa_model is not None else None,
        "rng_state": capture_rng_state(loader_generator),
        "run_state": copy.deepcopy(run_state),
        "provenance": copy.deepcopy(provenance),
    }


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_checkpoint_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CheckpointValidationError("checkpoint payload must be a mapping")
    missing = _REQUIRED_CHECKPOINT_KEYS.difference(payload)
    if missing:
        raise CheckpointValidationError(
            f"checkpoint missing keys: {', '.join(sorted(missing))}"
        )
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointValidationError(
            f"unsupported checkpoint format_version: {payload['format_version']}"
        )
    if (
        not isinstance(payload["completed_epoch"], int)
        or payload["completed_epoch"] < 0
    ):
        raise CheckpointValidationError("checkpoint completed_epoch is invalid")
    return payload


def load_full_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointValidationError(f"unable to load checkpoint: {path}") from exc
    return _validate_checkpoint_payload(payload)


def write_emergency_checkpoint(
    last_checkpoint: str | Path,
    destination: str | Path,
    signum: int,
) -> None:
    payload = load_full_checkpoint(last_checkpoint)
    emergency_payload = dict(payload)
    emergency_payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    emergency_payload["reason"] = "signal"
    emergency_payload["signal_number"] = int(signum)
    atomic_torch_save(emergency_payload, destination)


def validate_checkpoint_provenance(
    payload: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    actual = payload.get("provenance")
    if not isinstance(actual, dict):
        raise CheckpointValidationError("checkpoint provenance must be a mapping")
    for key, expected_value in expected.items():
        if key not in actual:
            raise CheckpointValidationError(f"checkpoint provenance missing {key}")
        if actual[key] != expected_value:
            raise CheckpointValidationError(
                f"checkpoint provenance mismatch for {key}"
            )


def restore_training_state(
    payload: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    swa_model: AveragedModel | None,
    loader_generator: torch.Generator | None,
) -> None:
    _validate_checkpoint_payload(payload)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    swa_state = payload["swa_state_dict"]
    if swa_state is None:
        if swa_model is not None:
            raise CheckpointValidationError(
                "checkpoint has no SWA state but an SWA model was provided"
            )
    elif swa_model is None:
        raise CheckpointValidationError("checkpoint contains SWA state")
    else:
        swa_model.load_state_dict(swa_state, strict=True)
    restore_rng_state(payload["rng_state"], loader_generator)
