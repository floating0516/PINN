# R-only Event Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a controlled three-seed R-only campaign whose saved checkpoints are selected by validation Event MAE.

**Architecture:** Reuse the existing training and corrected-matrix runners. Widen only the active-workflow config contract to permit the already-supported `event_mae_catalog` checkpoint metric, then run one complete USGS-priority configuration for seeds 17, 42, and 73.

**Tech Stack:** Python 3.12, PyTorch, pytest, YAML, tmux, systemd-inhibit.

---

### Task 1: Permit Event-level Checkpoint Selection

**Files:**
- Modify: `tests/test_config_v2.py`
- Modify: `src/utils/config_v2.py`

- [x] **Step 1: Write the focused failing test**

Add `test_active_station_workflow_accepts_event_checkpoint_metric`, set the active config's checkpoint metric to `event_mae_catalog`, and call `validate_config_v2`. Change the conflicting-value parametrization to reject `val_loss` instead of the newly accepted value.

- [x] **Step 2: Verify the test fails for the intended reason**

Run: `source /home/lihe/.config/pinn/server.env && /home/lihe/PINN_Mag/venv/bin/python -m pytest tests/test_config_v2.py::test_active_station_workflow_accepts_event_checkpoint_metric -q`

Expected: FAIL because the active workflow currently requires `station_mae_catalog` exactly.

- [x] **Step 3: Implement the minimal validator change**

Read `training.checkpoint_metric` with `_required` and accept exactly `station_mae_catalog` or `event_mae_catalog`; raise a `ValueError` containing `training.checkpoint_metric` for any other value.

- [x] **Step 4: Verify the narrow behavior**

Run the focused test, `tests/test_config_v2.py`, and `tests/test_training_checkpointing.py::test_active_training_logs_station_catalog_metric_and_runs_all_epochs`.

Expected: all selected tests pass with no warnings or errors.

### Task 2: Freeze The Formal Experiment Configuration

**Files:**
- Create: `configs/experiments_v2/V2-USGS-EVENT-CHECKPOINT.yaml`

- [x] **Step 1: Add the complete formal configuration**

Copy the verified phase-9 USGS-priority configuration and change only `training.checkpoint_metric` to `event_mae_catalog`.

- [x] **Step 2: Validate the scientific diff**

Load both YAML files, remove `training.checkpoint_metric`, and assert the remaining mappings are equal. Run `validate_config_v2` on the new configuration.

Expected: validator passes and the only value difference is `station_mae_catalog -> event_mae_catalog`.

- [x] **Step 3: Commit the implementation**

Commit the test, validator, formal config, spec, and plan with message `feat: select radial checkpoints by event mae`.

### Task 3: Run The Three-seed Campaign

**Files:**
- Create at runtime: `/home/lihe/PINN_Mag/runs/phase12-r-event-checkpoint-<stamp>-<commit>/`
- Create at runtime: `/home/lihe/PINN_Mag/worktrees/run-<commit>/`

- [x] **Step 1: Create a detached run worktree**

Create the worktree at the implementation commit and confirm its tracked tree is clean.

- [x] **Step 2: Launch only from the formal tmux pane**

In `pinn-run:train`, change to the detached worktree, load `/home/lihe/.config/pinn/server.env`, and run `scripts/experiments/run_corrected_matrix.py` in `within-event-station` mode for seeds `17 42 73`, 200 epochs, with the frozen dataset manifest and unique output root under `systemd-inhibit --what=sleep --mode=block`.

- [x] **Step 3: Confirm startup**

Confirm the campaign manifest records the committed revision, `git_dirty=false`, the three seeds, the USGS-priority NPZ, and `checkpoint_metric=event_mae_catalog`. Confirm one CUDA training process is active and the log begins without exceptions.

- [x] **Step 4: Evaluate after all seeds finish**

Join the three internal `event_summary.csv` files by event, average each event's three predictions, and compare the ensemble Event MAE with `0.1609267`. Run the fixed external eight-event evaluation only if the internal result is `<= 0.1509267`.
