# Gated Tangential Magnitude Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one controlled seed-42 experiment in which T can only add a gated magnitude residual to a frozen R-only model.

**Architecture:** Preserve the existing R encoder, STF head, and magnitude head. Add a shallow T encoder plus metadata-conditioned residual head, combine only the catalog magnitude at a zero-initialized scalar gate, and warm-start/freeze the radial path.

**Tech Stack:** Python 3.12, PyTorch, pytest, existing corrected-v2 training and evaluation pipeline.

---

### Task 1: Implement And Run The Gated T Residual

**Files:**
- Modify: `tests/test_model_forward.py`
- Modify: `src/models/model.py`
- Use: `src/training/train.py`
- Create at runtime: `runs/phase11-rt-gated-seed42-<stamp>-<commit>/`

- [x] **Step 1: Write the focused failing test**

Add a test that constructs an R-only model and a gated R+T model, loads the R state with `strict=False`, and asserts that the zero gate produces identical STF and catalog-Mw predictions, all base parameters are frozen, and the gate receives a finite nonzero gradient.

- [x] **Step 2: Verify the focused test fails**

Run: `python -m pytest tests/test_model_forward.py::test_gated_tangential_residual_starts_at_frozen_radial_prediction -q`

Expected: FAIL because `magnitude_gated_residual` is not implemented and the current two-channel embedding cannot load the radial checkpoint.

- [x] **Step 3: Implement the minimal gated residual**

In `PINNModel`, recognize `model.input_fusion == "magnitude_gated_residual"`, require R+T input, keep `embed` at one input channel, add the independent T temporal encoder, metadata-conditioned scalar residual head, and zero-initialized tanh gate. Use only `x[:, :1]` for the existing sequence/STF path and freeze every parameter whose name does not begin with `tangential_` when `model.freeze_radial_backbone` is true.

- [x] **Step 4: Verify implementation once**

Run the focused test, one real-data CUDA forward/loss/backward smoke, then `python -m pytest -q`. Expected: focused test PASS, finite CUDA smoke, and full suite with zero failures.

- [x] **Step 5: Commit and train once**

Commit the implementation, create a detached worktree at that commit, derive the USGS seed-42 config with R+T gated fusion, frozen radial backbone, and `training.pretrain_path` set to the formal R-only seed-42 checkpoint, then run 200 epochs in `pinn-run:train` under `systemd-inhibit`.

- [x] **Step 6: Evaluate and decide**

Run one internal mechanism summary and one fixed eight-event external evaluation. If any internal gate fails, record the result and end T experiments without seeds 17/73.
