from __future__ import annotations

import random
from pathlib import Path
import copy
import csv
import signal
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel
import yaml

from src.training.checkpointing import (
    CheckpointValidationError,
    atomic_torch_save,
    build_full_checkpoint,
    capture_rng_state,
    load_full_checkpoint,
    restore_rng_state,
    restore_training_state,
    TrainingInterrupted,
    TrainingSignalState,
    install_training_signal_handlers,
    validate_checkpoint_provenance,
    write_emergency_checkpoint,
)
from src.training.train import train


def _make_training_objects() -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    AveragedModel,
]:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    swa_model = AveragedModel(model)
    swa_model.update_parameters(model)
    return model, optimizer, scheduler, swa_model


def _draw_rng_values(generator: torch.Generator) -> dict[str, object]:
    values: dict[str, object] = {
        "python": random.random(),
        "numpy": np.random.random(3),
        "torch_cpu": torch.rand(3),
        "loader": torch.rand(3, generator=generator),
    }
    if torch.cuda.is_available():
        values["torch_cuda"] = torch.rand(3, device="cuda").cpu()
    return values


def test_rng_state_round_trip_restores_all_generators() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    loader_generator = torch.Generator().manual_seed(17)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(17)

    state = capture_rng_state(loader_generator)
    expected = _draw_rng_values(loader_generator)
    _draw_rng_values(loader_generator)

    restore_rng_state(state, loader_generator)
    actual = _draw_rng_values(loader_generator)

    assert actual["python"] == expected["python"]
    np.testing.assert_array_equal(actual["numpy"], expected["numpy"])
    torch.testing.assert_close(actual["torch_cpu"], expected["torch_cpu"])
    torch.testing.assert_close(actual["loader"], expected["loader"])
    if torch.cuda.is_available():
        torch.testing.assert_close(actual["torch_cuda"], expected["torch_cuda"])


def test_atomic_checkpoint_round_trips_required_training_state(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler, swa_model = _make_training_objects()
    loader_generator = torch.Generator().manual_seed(29)
    early_stop_state = {
        "metric": "mw_mae",
        "best_val_loss": 0.4,
        "best_mw_mae": 0.2,
        "counter": 3,
        "patience": 8,
        "min_delta": 1.0e-4,
    }
    provenance = {
        "git_commit": "abc123",
        "git_dirty": False,
        "config_sha256": "config-sha",
        "dataset_manifest_sha256": "dataset-sha",
        "split_sha256": "split-sha",
    }
    run_state = {
        "run_id": "run-1",
        "models_dir": str(tmp_path / "models"),
        "results_dir": str(tmp_path / "results"),
        "log_file": str(tmp_path / "training.csv"),
        "best_model_path": str(tmp_path / "best_model.pth"),
        "best_model_swa_path": None,
    }
    payload = build_full_checkpoint(
        completed_epoch=4,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stop_state=early_stop_state,
        swa_model=swa_model,
        loader_generator=loader_generator,
        run_state=run_state,
        provenance=provenance,
        reason="epoch",
    )
    checkpoint_path = tmp_path / "last_state.pth"

    atomic_torch_save(payload, checkpoint_path)
    loaded = load_full_checkpoint(checkpoint_path)

    assert loaded["format_version"] == 1
    assert loaded["completed_epoch"] == 4
    assert loaded["early_stop_state"] == early_stop_state
    assert loaded["run_state"] == run_state
    assert loaded["provenance"] == provenance
    assert loaded["reason"] == "epoch"
    assert loaded["swa_state_dict"] is not None

    restored_model, restored_optimizer, restored_scheduler, restored_swa = (
        _make_training_objects()
    )
    restore_training_state(
        loaded,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        swa_model=restored_swa,
        loader_generator=torch.Generator(),
    )
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(actual, expected)
    assert restored_optimizer.state_dict()["param_groups"] == optimizer.state_dict()[
        "param_groups"
    ]
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_swa.state_dict().keys() == swa_model.state_dict().keys()
    assert int(restored_swa.n_averaged) == int(swa_model.n_averaged)


def test_resume_rejects_provenance_mismatch(tmp_path: Path) -> None:
    model, optimizer, scheduler, _ = _make_training_objects()
    payload = build_full_checkpoint(
        completed_epoch=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stop_state={},
        swa_model=None,
        loader_generator=None,
        run_state={},
        provenance={"git_commit": "expected", "git_dirty": False},
        reason="initial",
    )

    with pytest.raises(CheckpointValidationError, match="git_commit"):
        validate_checkpoint_provenance(
            payload,
            {"git_commit": "different", "git_dirty": False},
        )


def _training_config(tmp_path: Path) -> dict:
    config = yaml.safe_load(
        Path("configs/config_v2.yaml").read_text(encoding="utf-8")
    )
    config["model"].update(
        {
            "hidden_dim": 8,
            "num_layers": 1,
            "num_tcn_blocks": 1,
            "transformer_num_layers": 1,
            "dropout": 0.0,
        }
    )
    config["paths"].update(
        {
            "logs_dir": str(tmp_path / "logs"),
            "models_dir": str(tmp_path / "models"),
            "results_dir": str(tmp_path / "results"),
        }
    )
    config["training"].update(
        {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 1.0e-4,
            "early_stop_metric": "mw_mae",
            "early_stop_patience": 0,
            "swa_start": 1,
        }
    )
    return config


def _training_batch() -> dict[str, torch.Tensor]:
    return {
        "radial": torch.full((1, 1, 200), 1.0e-3),
        "source_distance_m": torch.tensor([100_000.0]),
        "theta_deg": torch.tensor([40.0]),
        "azimuth_deg": torch.tensor([20.0]),
        "phi_slip_deg": torch.tensor([10.0]),
        "stf": torch.full((1, 300), 1.0e18),
        "stf_dt_sec": torch.tensor([1.0]),
        "waveform_dt_sec": torch.tensor([1.0]),
        "waveform_valid_mask": torch.ones(1, 200, dtype=torch.bool),
        "has_stf": torch.tensor([True]),
        "mw_stf_native": torch.tensor([7.0]),
        "magnitude_catalog": torch.tensor([7.0]),
    }


def _injected_loaders() -> tuple[list[dict], list[dict], list[dict], dict]:
    batch = _training_batch()
    return [batch], [batch], [], {
        "protocol": "grouped_event_test",
        "seed": 42,
        "train_events": ["Synthetic"],
        "validation_events": ["SyntheticValidation"],
        "test_events": [],
        "train_record_count": 1,
        "validation_record_count": 1,
        "test_record_count": 0,
    }


def test_train_commits_full_epoch_state_and_resume_reuses_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.training.train.get_preferred_device", lambda: torch.device("cpu"))
    config = _training_config(tmp_path)

    first = train(config=copy.deepcopy(config), data_loaders=_injected_loaders())
    last_state = Path(first["models_dir"]) / "last_state.pth"
    payload = load_full_checkpoint(last_state)

    assert payload["completed_epoch"] == 1
    assert payload["swa_state_dict"] is not None
    assert payload["early_stop_state"]["counter"] == 0
    assert first["last_checkpoint_path"] == last_state

    resumed = train(
        config=copy.deepcopy(config),
        data_loaders=_injected_loaders(),
        resume_checkpoint=last_state,
    )

    assert resumed["run_id"] == first["run_id"]
    assert resumed["models_dir"] == first["models_dir"]
    assert resumed["resumed_from_epoch"] == 1
    assert list((tmp_path / "models").iterdir()) == [Path(first["models_dir"])]
    with Path(first["log_file"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 2
    assert rows[1][0] == "1"


def test_resume_and_pretrain_are_mutually_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.training.train.get_preferred_device", lambda: torch.device("cpu"))
    config = _training_config(tmp_path)
    first = train(config=copy.deepcopy(config), data_loaders=_injected_loaders())
    resume_path = Path(first["models_dir"]) / "last_state.pth"
    pretrain_path = tmp_path / "pretrain.pth"
    torch.save({}, pretrain_path)
    config["training"]["pretrain_path"] = str(pretrain_path)

    with pytest.raises(ValueError, match="pretrain_path.*resume"):
        train(
            config=config,
            data_loaders=_injected_loaders(),
            resume_checkpoint=resume_path,
        )


def _write_toy_checkpoint(tmp_path: Path) -> Path:
    model, optimizer, scheduler, _ = _make_training_objects()
    payload = build_full_checkpoint(
        completed_epoch=2,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stop_state={},
        swa_model=None,
        loader_generator=None,
        run_state={},
        provenance={},
        reason="epoch",
    )
    path = tmp_path / "last_state.pth"
    atomic_torch_save(payload, path)
    return path


def test_signal_handler_only_records_pending_signal() -> None:
    state = TrainingSignalState()
    previous = signal.getsignal(signal.SIGTERM)

    with install_training_signal_handlers(state):
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        assert state.pending_signal == signal.SIGTERM

    assert signal.getsignal(signal.SIGTERM) == previous


def test_pending_signal_writes_emergency_checkpoint_and_exits(
    tmp_path: Path,
) -> None:
    last_state = _write_toy_checkpoint(tmp_path)
    emergency_state = tmp_path / "emergency_state.pth"
    state = TrainingSignalState()
    state.request(signal.SIGTERM)

    with pytest.raises(TrainingInterrupted) as interrupted:
        state.checkpoint_and_raise(last_state, emergency_state)

    assert interrupted.value.code == 143
    payload = load_full_checkpoint(emergency_state)
    assert payload["completed_epoch"] == 2
    assert payload["reason"] == "signal"
    assert payload["signal_number"] == signal.SIGTERM


@pytest.mark.parametrize(
    ("signum", "expected_code"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_real_signal_writes_loadable_emergency_checkpoint(
    tmp_path: Path,
    signum: int,
    expected_code: int,
) -> None:
    last_state = _write_toy_checkpoint(tmp_path)
    emergency_state = tmp_path / "emergency_state.pth"
    code = """
import signal
import sys
import time
from pathlib import Path
from src.training.checkpointing import TrainingSignalState, install_training_signal_handlers

state = TrainingSignalState()
with install_training_signal_handlers(state):
    print('READY', flush=True)
    while state.pending_signal is None:
        time.sleep(0.01)
    state.checkpoint_and_raise(Path(sys.argv[1]), Path(sys.argv[2]))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(last_state), str(emergency_state)],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "READY"
    process.send_signal(signum)
    _, stderr = process.communicate(timeout=10)

    assert process.returncode == expected_code, stderr
    assert load_full_checkpoint(emergency_state)["signal_number"] == signum


def test_train_checks_pending_signal_at_batch_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.training.train.get_preferred_device", lambda: torch.device("cpu"))
    pending = TrainingSignalState()
    pending.request(signal.SIGTERM)
    monkeypatch.setattr("src.training.train.TrainingSignalState", lambda: pending)

    with pytest.raises(TrainingInterrupted) as interrupted:
        train(config=_training_config(tmp_path), data_loaders=_injected_loaders())

    assert interrupted.value.code == 143
    run_dirs = list((tmp_path / "models").iterdir())
    assert len(run_dirs) == 1
    emergency = run_dirs[0] / "emergency_state.pth"
    payload = load_full_checkpoint(emergency)
    assert payload["completed_epoch"] == 0


def test_interrupted_resume_matches_uninterrupted_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.training.train.get_preferred_device", lambda: torch.device("cpu"))
    baseline_config = _training_config(tmp_path / "baseline")
    resumed_config = _training_config(tmp_path / "resumed")

    baseline = train(
        config=copy.deepcopy(baseline_config),
        data_loaders=_injected_loaders(),
    )

    pending = TrainingSignalState()
    pending.request(signal.SIGTERM)
    monkeypatch.setattr("src.training.train.TrainingSignalState", lambda: pending)
    with pytest.raises(TrainingInterrupted):
        train(
            config=copy.deepcopy(resumed_config),
            data_loaders=_injected_loaders(),
        )
    interrupted_run = next((tmp_path / "resumed" / "models").iterdir())
    emergency = interrupted_run / "emergency_state.pth"

    monkeypatch.setattr(
        "src.training.train.TrainingSignalState",
        TrainingSignalState,
    )
    resumed = train(
        config=copy.deepcopy(resumed_config),
        data_loaders=_injected_loaders(),
        resume_checkpoint=emergency,
    )

    baseline_state = load_full_checkpoint(baseline["last_checkpoint_path"])
    resumed_state = load_full_checkpoint(resumed["last_checkpoint_path"])
    assert baseline_state["completed_epoch"] == resumed_state["completed_epoch"] == 1
    for key, expected in baseline_state["model_state_dict"].items():
        torch.testing.assert_close(resumed_state["model_state_dict"][key], expected)
    assert Path(baseline["log_file"]).read_text(encoding="utf-8") == Path(
        resumed["log_file"]
    ).read_text(encoding="utf-8")
