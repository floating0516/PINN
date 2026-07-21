# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

- Install dependencies: `pip install -r requirements.txt`
- Before running training or evaluation, update the placeholder absolute paths in `configs/config.yaml`:
  - `paths.data_path`
  - `dataset.stf_path`
- Train: `python -c "from src.training.train import train; train()"`
- Evaluate: `python -c "from src.evaluation.evaluate import evaluate; evaluate()"`
- Run all tests: `pytest tests/ -v`
- Run one test file: `pytest tests/test_data_loader.py -v`
- Run one test by name: `pytest tests/test_data_loader.py -k radial_peak_filter -v`
- Build the matched GNSS NPZ dataset: `python scripts/data/build_gnss_event_npz.py`
- Run experiment sweeps/searches: `python scripts/experiments/hyperparam_search.py` or `python scripts/experiments/sweep_baseline_params.py`

Notes:
- Run commands from the repository root.
- No dedicated lint or package build configuration is present in this repo; it is a script-driven research codebase.

## Architecture

- This is a config-driven PyTorch research project for estimating earthquake moment magnitude from GNSS displacement data with physics-informed losses. `configs/config.yaml` is the control plane for dataset paths, preprocessing, model hyperparameters, physics constants, training settings, and output directories.

- `src/data/data_loader.py` is the core ingestion pipeline. It reads the training NPZ, supports both legacy `enu` + `station_info` format and newer `stations` format, projects E/N into a radial component, preprocesses waveform windows, unit conversion, filtering, optional P-wave baseline correction, STF alignment/resampling, and emits tensors plus metadata (`distance`, `theta_deg`, `phi_deg`, `mechanism`, `has_stf`).

- `scripts/build_gnss_event_npz.py` is the dataset-construction bridge between raw GNSS catalogs and the training NPZ. It combines metadata from a CSV with waveform data loaded through `src/data/gnss_dataset_loader.py`. This script is experiment-oriented and currently has hardcoded local default paths.

- The model predicts a moment-rate / STF sequence from the radial displacement sequence rather than directly regressing scalar magnitude. The shared architecture in `src/models/model.py` is a 1D convolutional encoder followed by residual dilated TCN blocks, squeeze-excitation channel attention, and a Transformer encoder, with optional metadata embedding (`[log(dist), sin/cos(theta), sin/cos(phi)]`) and a sequence head that outputs the rate representation.

- Training and evaluation both use `src.models.model.PINNModel`. When changing architecture or checkpoint loading behavior, keep train/eval compatibility aligned in this single model definition.

- Training is orchestrated by `src/training/train.py`. It seeds Python/NumPy/PyTorch, creates `outputs/` subdirectories from config, builds train/val/test loaders, and chooses the loss path from `training.loss_name`:
  - `src/training/physics.py` for STF/data-space loss plus Mw consistency
  - `src/training/loss_stf_rate.py` for waveform-based physics loss using an EEW_0012-style forward model from predicted moment rate to observed radial displacement

- Evaluation in `src/evaluation/evaluate.py` reconstructs Mw from the predicted rate sequence, compares against STF-derived Mw when available, writes metrics and figures to `outputs/results`, and compares the network against the analytic baseline in `src/baseline/__init__.py`.

## Repo-specific notes

- Several scripts are one-off experiment utilities with hardcoded assumptions or local paths. Check them before reuse, especially `scripts/data/build_gnss_event_npz.py`, `scripts/evaluation/final_eval.py`, and `src/evaluation/evaluate.py`.
- `src/evaluation/evaluate.py` currently looks for `outputs/models/best_model_B3.pth`, not `best_model.pth`.
- Tests in `tests/test_data_loader.py` build synthetic NPZ fixtures and do not require the real GNSS dataset.
- Runtime artifacts are expected under `outputs/` as configured in `configs/config.yaml`.
