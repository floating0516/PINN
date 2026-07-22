# Full-State Training Reliability Design

## Goal

Make formal training restartable and reproducible after a normal failure,
`SIGINT`, or `SIGTERM`, without changing the scientific training semantics.

## Scope

This unit adds full-state checkpoints, deterministic resume, and signal-aware
emergency shutdown to `src/training/train.py`. It does not implement Task 17,
start formal training, add batch-level resume, or change `best_model.pth`.

`best_model.pth` remains a weights-only evaluation artifact. Resume uses two
separate files in the run's model directory:

- `last_state.pth`: the most recent fully committed epoch boundary.
- `emergency_state.pth`: an atomic copy of the last stable state, annotated
  with the signal that stopped the process.

## Transaction Boundary

Training treats an epoch as a transaction. A stable checkpoint is written
before epoch 1 and after every fully completed epoch, after validation,
scheduler, early-stop, best-model, and SWA updates have all finished.

If a signal arrives during training or validation, the handler records only the
signal number. At the next batch boundary, normal computation stops and an
emergency checkpoint is created from `last_state.pth`. Work performed in the
incomplete epoch is deliberately discarded. Resume therefore repeats at most
one partial epoch and never continues from a mixed state.

Checkpoint writes use a temporary file in the destination directory, flush and
sync it, then replace the destination with `os.replace`. A crash can leave the
previous complete checkpoint or the new complete checkpoint, but not a
partially written official checkpoint.

## Checkpoint Contract

The checkpoint has a versioned top-level mapping with these state groups:

- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `completed_epoch` and the next epoch implied by it
- `early_stop_state`: metric name, best validation loss, best magnitude MAE,
  counter, patience, and minimum delta
- `swa_state_dict`, or `null` before SWA starts
- `rng_state`: Python, NumPy, Torch CPU, and all Torch CUDA generators
- `train_loader_generator_state`, when the loader exposes a generator
- `run_state`: run ID, model/results/log paths, best-model paths, and signal
  metadata
- `provenance`: Git commit and dirty flag, config SHA-256,
  dataset-manifest SHA-256, and split SHA-256

NumPy RNG arrays are normalized into weights-only-safe primitives or tensors so
the locally produced checkpoint can be loaded with `weights_only=True`.

## Resume Flow

`train` gains an explicit `resume_checkpoint` argument. Machine-local resume
paths do not enter the portable tracked YAML.

When `config` is omitted during resume, the sibling frozen `config.yaml` is
loaded. When a config is supplied, its canonical snapshot hash must equal the
checkpoint hash. Resume reconstructs the same data loaders and then rejects any
Git commit, Git dirty state, dataset manifest, or split hash mismatch. Formal
resume requires both the recorded and current worktrees to be clean.

The original run ID and output directories are reused. The training CSV is
opened in append mode and must agree with `completed_epoch`; configuration,
split, and manifest files are not silently overwritten. Model, optimizer,
scheduler, early-stop, SWA, loader generator, and global RNG state are restored
after all runtime objects have been constructed and immediately before the next
epoch iterator is created.

A small append-only resume history records checkpoint hash, epoch, timestamp,
and reason. The existing run manifest remains the primary run identity and is
completed only when training finishes normally.

## Signal Behavior

Signal handlers perform no Torch or filesystem work. They set a pending signal
flag and return. Training checks the flag after each train or validation batch.
The safe-point path atomically writes `emergency_state.pth` from the last stable
checkpoint and exits with `128 + signal_number` (`130` for `SIGINT`, `143` for
`SIGTERM`). Previous process handlers are restored when `train` returns or
raises so tests and embedding callers are not polluted.

An initial `last_state.pth` is committed before the first batch. A signal during
epoch 1 is therefore still resumable.

## Failure Policy

Resume fails closed for an unsupported checkpoint version, missing required
state, corrupt payload, incompatible model/optimizer/SWA state, provenance
mismatch, non-restorable loader generator, or a training log ahead of or behind
the checkpoint. It never falls back to weights-only loading.

The existing `pretrain_path` behavior remains transfer learning and is mutually
exclusive with resume.

## Verification

Focused tests must prove:

1. Python, NumPy, Torch CPU, conditional Torch CUDA, and loader RNG round trips.
2. Model, optimizer, scheduler, early-stop, and SWA state round trips.
3. Atomic replacement leaves a loadable versioned checkpoint.
4. An uninterrupted run and an epoch-boundary interrupted/resumed run produce
   identical final state and logs under deterministic settings.
5. Real `SIGINT` and `SIGTERM` subprocesses produce loadable emergency
   checkpoints and the expected exit status.
6. Config, Git, dataset, split, and log mismatches are rejected.
7. A short CUDA resume-equivalence smoke passes before the feature is accepted.

After focused RED/GREEN tests, run the complete test suite and warnings-as-errors
source compilation once. Do not start formal training as part of this unit.
