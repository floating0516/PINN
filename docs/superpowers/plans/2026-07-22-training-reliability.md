# Training Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add atomic epoch-boundary full-state checkpoints, deterministic resume, and safe SIGINT/SIGTERM shutdown.

**Architecture:** A focused `src/training/checkpointing.py` module owns serialization, RNG, atomic replacement, provenance validation, and signal state. `train.py` commits that state before epoch 1 and after each complete epoch, restores it into the same run, and rolls an interrupted partial epoch back to the latest committed boundary.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, POSIX signals.

---

### Task 1: Full-State Checkpoint Primitives

**Files:**
- Create: `src/training/checkpointing.py`
- Create: `tests/test_training_checkpointing.py`

- [ ] **Step 1: Write focused failing tests**

```python
def test_rng_state_round_trip_restores_python_numpy_torch_and_loader(): ...
def test_atomic_checkpoint_round_trips_all_required_state(): ...
def test_resume_rejects_provenance_mismatch(): ...
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_training_checkpointing.py -q`

Expected: collection fails because `src.training.checkpointing` does not exist.

- [ ] **Step 3: Implement the checkpoint API**

```python
CHECKPOINT_FORMAT_VERSION = 1

def capture_rng_state(loader_generator: torch.Generator | None) -> dict: ...
def restore_rng_state(state: dict, loader_generator: torch.Generator | None) -> None: ...
def atomic_torch_save(payload: dict, path: Path) -> None: ...
def load_full_checkpoint(path: Path) -> dict: ...
def validate_checkpoint_provenance(payload: dict, expected: dict) -> None: ...
```

Normalize NumPy state for `weights_only=True`, include CPU/CUDA RNG, reject
unsupported versions or missing fields, and fsync before `os.replace`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_training_checkpointing.py -q`

Expected: all focused primitive tests pass.

### Task 2: Transactional Train/Resume Integration

**Files:**
- Modify: `src/training/train.py`
- Modify: `tests/test_training_checkpointing.py`

- [ ] **Step 1: Add failing integration tests**

```python
def test_train_commits_initial_and_completed_epoch_state(tmp_path): ...
def test_resume_reuses_run_and_restores_optimizer_scheduler_early_stop_and_swa(tmp_path): ...
def test_resume_appends_log_without_duplicate_epochs(tmp_path): ...
def test_pretrain_and_resume_are_mutually_exclusive(tmp_path): ...
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_training_checkpointing.py -q`

Expected: the new `resume_checkpoint` behavior and stable checkpoint files are absent.

- [ ] **Step 3: Integrate the stable transaction**

```python
def train(
    config: dict | None = None,
    data_loaders: tuple | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, object]: ...
```

Fresh runs create the existing unique output layout, frozen config/split, log
header, and epoch-zero checkpoint. Resume loads the sibling config when needed,
reconstructs and validates loaders/split/provenance, reuses run paths, restores
all state immediately before iteration, and starts at `completed_epoch`.

After validation, scheduler, early-stop, best-weight, and SWA updates, save the
new `last_state.pth`. Preserve `best_model.pth` as weights-only.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_training_checkpointing.py tests/test_provenance.py -q`

Expected: all checkpoint and existing provenance tests pass.

### Task 3: Signal Exit And Final Gates

**Files:**
- Modify: `src/training/checkpointing.py`
- Modify: `src/training/train.py`
- Modify: `tests/test_training_checkpointing.py`

- [ ] **Step 1: Add failing signal tests**

```python
def test_signal_handler_only_records_pending_signal(): ...
def test_pending_signal_clones_last_stable_state_and_exits_143(tmp_path): ...
def test_sigterm_subprocess_writes_loadable_emergency_checkpoint(tmp_path): ...
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_training_checkpointing.py -q`

Expected: signal controller and emergency checkpoint behavior are absent.

- [ ] **Step 3: Implement safe signal handling**

```python
class TrainingInterrupted(SystemExit): ...
class TrainingSignalState: ...

@contextmanager
def install_training_signal_handlers(state: TrainingSignalState): ...

def write_emergency_checkpoint(last_state: Path, destination: Path, signum: int) -> None: ...
```

Handlers only set state. Train and validation loops check after each batch,
atomically write `emergency_state.pth` from the last committed epoch, restore
prior handlers, and exit with `128 + signum`.

- [ ] **Step 4: Run final verification**

Run: `python -m pytest tests/test_training_checkpointing.py -q`

Run: `python -m pytest tests/ -q`

Run: `python -W error -m compileall -q src scripts`

Run a short CUDA uninterrupted-versus-resume smoke using injected deterministic
loaders. Require identical final model state, finite metrics, and loadable
`last_state.pth`/`emergency_state.pth`.

- [ ] **Step 5: Commit the implementation**

```bash
git add src/training/checkpointing.py src/training/train.py tests/test_training_checkpointing.py
git commit -m "feat: add deterministic full-state training resume"
```
