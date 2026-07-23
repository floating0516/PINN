# R-only Event-balanced Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one controlled three-seed R-only campaign with equal expected sampling weight per training event.

**Architecture:** Reuse the existing `WeightedRandomSampler` implementation and corrected-matrix runner. Widen only the active config contract to permit a boolean event-balanced switch, freeze a complete USGS configuration with the switch enabled, then train and evaluate seeds 17, 42, and 73.

**Tech Stack:** Python 3.12, PyTorch, pytest, YAML, tmux, systemd-inhibit.

---

### Task 1: Permit Event-balanced Sampling

**Files:**
- Modify: `tests/test_config_v2.py`
- Modify: `src/utils/config_v2.py`

- [ ] **Step 1: Write the focused failing test**

Add `test_active_station_workflow_accepts_event_balanced_sampling`, set the active workflow switch to `True`, and call `validate_config_v2`. Change the conflicting-value parametrization to reject the string `"true"` instead of the accepted boolean.

- [ ] **Step 2: Verify RED**

Run the new test with the project environment loaded. Expected: FAIL because the active workflow currently requires `False` exactly.

- [ ] **Step 3: Implement the minimal validator change**

Require `training.event_balanced_sampling` to be a boolean, accepting both `False` and `True`; preserve every other active-workflow invariant.

- [ ] **Step 4: Verify GREEN narrowly**

Run `tests/test_config_v2.py`, `test_event_balanced_weights_sum_equally_per_event`, and `test_grouped_loader_uses_balanced_sampler_and_manifest`. Expected: all selected tests pass.

### Task 2: Freeze The Formal Configuration

**Files:**
- Create: `configs/experiments_v2/V2-USGS-EVENT-BALANCED.yaml`

- [ ] **Step 1: Add the full USGS configuration**

Copy the verified phase-9 R-only config, keep `checkpoint_metric: station_mae_catalog`, and set only `event_balanced_sampling: true`.

- [ ] **Step 2: Audit the scientific diff**

Validate the YAML, compare it structurally with the phase-9 config, and assert that the only difference is `training.event_balanced_sampling`.

- [ ] **Step 3: Commit**

Commit with message `feat: balance radial training by event`.

### Task 3: Train And Decide

**Files:**
- Create at runtime: `/home/lihe/PINN_Mag/runs/phase13-r-event-balanced-<stamp>-<commit>/`
- Create at runtime: `/home/lihe/PINN_Mag/worktrees/run-<commit>/`

- [ ] **Step 1: Create a detached run worktree**

Create it at the implementation commit and confirm the tracked tree is clean.

- [ ] **Step 2: Launch the formal campaign**

In `pinn-run:train`, run the corrected matrix in `within-event-station` mode for seeds `17 42 73` and 200 epochs under `systemd-inhibit`, using the frozen phase-9 dataset manifest.

- [ ] **Step 3: Audit startup**

Confirm R-only input, USGS NPZ, `event_balanced_sampling=true`, `checkpoint_metric=station_mae_catalog`, clean commit provenance, and split hashes identical to phase 9.

- [ ] **Step 4: Evaluate the internal ensemble**

Verify three complete 200-epoch logs and finite outputs, join event predictions across seeds, and compare with `0.1609267`. Run the external eight-event evaluation only if the result is `<= 0.1509267` and the sparse/low-magnitude strata do not materially regress.
