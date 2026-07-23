# Radial + Tangential Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backward-compatible R+T waveform input path and run one controlled USGS-priority seed-42 pilot without changing accepted samples, split assignments, physics targets, or training hyperparameters.

**Architecture:** Compute tangential motion beside the existing radial trace, retain radial-only sample acceptance and radial physics loss, and assemble model inputs through one shared config-driven helper. Derive a unique R+T pilot config from the frozen USGS seed-42 config, validate the approved internal gates, then evaluate the fixed external events once only if the internal gate passes.

**Tech Stack:** Python 3.12, NumPy, PyTorch, PyYAML, pytest, existing corrected-v2 loaders/train/evaluation/campaign utilities.

---

### Task 1: Validate and assemble configured waveform components

**Files:**
- Create: `src/data/model_input.py`
- Modify: `src/utils/config_v2.py`
- Modify: `tests/test_config_v2.py`
- Create: `tests/test_model_input.py`

- [ ] **Step 1: Write failing config and tensor-assembly tests**

Add tests proving that a missing key defaults to radial-only, the only accepted values are `['radial']` and `['radial', 'tangential']`, duplicate/reordered/unknown components fail, and R+T tensors are stacked in canonical order:

```python
def test_waveform_components_default_to_radial() -> None:
    assert waveform_input_components_from_config(_minimal_v2()) == ("radial",)


def test_waveform_components_accept_canonical_rt() -> None:
    config = _minimal_v2()
    config.setdefault("model", {})["input_components"] = [
        "radial",
        "tangential",
    ]
    validate_config_v2(config)
    assert waveform_input_components_from_config(config) == (
        "radial",
        "tangential",
    )


@pytest.mark.parametrize(
    "value",
    ["radial", [], ["tangential"], ["tangential", "radial"],
     ["radial", "radial"], ["radial", "vertical"]],
)
def test_waveform_components_reject_ambiguous_values(value: object) -> None:
    config = _minimal_v2()
    config.setdefault("model", {})["input_components"] = value
    with pytest.raises(ValueError, match="input_components"):
        validate_config_v2(config)


def test_assemble_model_input_stacks_rt_in_declared_order() -> None:
    config = {"model": {"input_components": ["radial", "tangential"]}}
    batch = {
        "radial": torch.ones(2, 1, 4),
        "tangential": torch.full((2, 1, 4), 2.0),
    }
    result = assemble_model_input(batch, config)
    assert result.shape == (2, 2, 4)
    torch.testing.assert_close(result[:, 0], batch["radial"][:, 0])
    torch.testing.assert_close(result[:, 1], batch["tangential"][:, 0])
```

- [ ] **Step 2: Run tests and confirm the red state**

Run:

```bash
source /home/lihe/.config/pinn/server.env
/home/lihe/PINN_Mag/venv/bin/pytest -q tests/test_config_v2.py tests/test_model_input.py
```

Expected: collection/import failures because the new helpers do not exist.

- [ ] **Step 3: Implement strict component parsing and shared assembly**

Add to `src/utils/config_v2.py` and call it from `validate_config_v2`:

```python
def waveform_input_components_from_config(
    config: dict[str, Any],
) -> tuple[str, ...]:
    value = (config.get("model", {}) or {}).get(
        "input_components",
        ["radial"],
    )
    if not isinstance(value, (list, tuple)):
        raise ValueError("model.input_components must be a sequence")
    components = tuple(value)
    if components not in {
        ("radial",),
        ("radial", "tangential"),
    }:
        raise ValueError(
            "model.input_components must be ['radial'] or "
            "['radial', 'tangential']"
        )
    return components
```

Create `src/data/model_input.py`:

```python
from collections.abc import Mapping
from typing import Any

import torch

from src.utils.config_v2 import waveform_input_components_from_config


def _single_channel(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value.unsqueeze(1)
    if value.ndim == 3 and value.shape[1] == 1:
        return value
    raise ValueError(
        f"{name} must have shape (batch,time) or (batch,1,time)"
    )


def assemble_model_input(
    values: Mapping[str, torch.Tensor],
    config: dict[str, Any],
) -> torch.Tensor:
    channels = [
        _single_channel(name, values[name])
        for name in waveform_input_components_from_config(config)
    ]
    reference = channels[0].shape
    if any(channel.shape != reference for channel in channels[1:]):
        raise ValueError("configured waveform component shapes differ")
    output = torch.cat(channels, dim=1)
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError("model waveform input is non-finite")
    return output
```

- [ ] **Step 4: Run focused tests and commit**

Run the Step 2 command; expect all tests to pass. Commit:

```bash
git add src/utils/config_v2.py src/data/model_input.py tests/test_config_v2.py tests/test_model_input.py
git commit -m "feat: configure waveform input components"
```

### Task 2: Compute and expose the tangential waveform without changing sample acceptance

**Files:**
- Modify: `src/data/sample_builder.py`
- Modify: `src/data/dataset_v2.py`
- Modify: `tests/test_corrected_pipeline_integration.py`
- Modify: `tests/test_unseen_event_eval.py`

- [ ] **Step 1: Write failing R/T rotation and dataset-contract tests**

Cover canonical azimuths, energy preservation, dataset tensor shape, training/external equivalence, and radial-only rejection:

```python
def test_rotate_horizontal_to_rt_preserves_energy() -> None:
    east = np.array([1.0, 2.0, -3.0])
    north = np.array([4.0, -5.0, 6.0])
    radial, tangential = rotate_horizontal_to_rt(east, north, 37.0)
    np.testing.assert_allclose(
        radial**2 + tangential**2,
        east**2 + north**2,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_corrected_dataset_exposes_radial_and_tangential_channels(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "events.npz"
    _write_stations_npz(npz_path)
    stf_dir = tmp_path / "stf"
    _write_stf(stf_dir, "eventa")
    config = make_v2_dataset_config(
        npz_path=npz_path,
        stf_dir=stf_dir,
    )
    dataset = CorrectedEarthquakeDataset(config)
    item = dataset[0]
    assert item["radial"].shape == (1, 200)
    assert item["tangential"].shape == (1, 200)
    assert torch.isfinite(item["tangential"]).all()


def test_rt_addition_keeps_radial_threshold_authoritative(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "events.npz"
    _write_stations_npz(npz_path)
    stf_dir = tmp_path / "stf"
    _write_stf(stf_dir, "eventa")
    config = make_v2_dataset_config(
        npz_path=npz_path,
        stf_dir=stf_dir,
        radial_peak_min_cm=1.0e9,
    )
    dataset = CorrectedEarthquakeDataset(config)
    assert len(dataset) == 0
    assert {
        row["rejection_reason"]
        for row in dataset.manifest_rows
    } == {"below_radial_peak_threshold"}


def test_tangential_preprocessing_failure_is_not_a_silent_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    npz_path = tmp_path / "events.npz"
    _write_stations_npz(npz_path)
    stf_dir = tmp_path / "stf"
    _write_stf(stf_dir, "eventa")
    real = sample_builder.preprocess_waveform
    calls = 0

    def fail_third_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("synthetic tangential failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(
        sample_builder,
        "preprocess_waveform",
        fail_third_call,
    )
    with pytest.raises(ValueError, match="tangential.*eventa"):
        CorrectedEarthquakeDataset(
            make_v2_dataset_config(
                npz_path=npz_path,
                stf_dir=stf_dir,
            )
        )
```

Extend the existing external adapter parity test with:

```python
assert np.allclose(
    training_sample["tangential"],
    external_sample["tangential"],
)
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```bash
source /home/lihe/.config/pinn/server.env
/home/lihe/PINN_Mag/venv/bin/pytest -q \
  tests/test_corrected_pipeline_integration.py \
  tests/test_unseen_event_eval.py -k 'tangential or external_adapter or radial_threshold'
```

Expected: missing `rotate_horizontal_to_rt`/`tangential` failures.

- [ ] **Step 3: Implement the orthonormal E/N to R/T rotation**

Add a pure function and use it in `build_station_sample`:

```python
def rotate_horizontal_to_rt(
    east: np.ndarray,
    north: np.ndarray,
    azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    azimuth = math.radians(float(azimuth_deg))
    radial = east * math.sin(azimuth) + north * math.cos(azimuth)
    tangential = east * math.cos(azimuth) - north * math.sin(azimuth)
    return radial, tangential
```

Preserve the current order of radial/vertical preprocessing and radial peak rejection. Only after radial acceptance, preprocess tangential with the same `WaveformConfig`. If tangential preprocessing fails, raise a contextual `ValueError` rather than `SampleRejected`:

```python
try:
    tangential = preprocess_waveform(
        time_rel,
        tangential_raw,
        units=units,
        p_arrival_sec=p_arrival_sec,
        config=waveform_config,
    )
except ValueError as exc:
    raise ValueError(
        f"tangential preprocessing failed for "
        f"{record.event}/{record.station}: {exc}"
    ) from exc

sample["tangential"] = tangential.values_m
sample["tangential_peak_cm"] = float(
    np.max(np.abs(tangential.values_m)) * 100.0
)
```

Expose it from `CorrectedEarthquakeDataset.__getitem__` as `(1,T)`:

```python
"tangential": torch.from_numpy(sample["tangential"])
    .float()
    .unsqueeze(0),
```

- [ ] **Step 4: Run focused tests and commit**

Run the Step 2 command; expect pass. Commit:

```bash
git add src/data/sample_builder.py src/data/dataset_v2.py \
  tests/test_corrected_pipeline_integration.py tests/test_unseen_event_eval.py
git commit -m "feat: add tangential waveform preprocessing"
```

### Task 3: Feed R+T to the model while retaining radial physics supervision

**Files:**
- Modify: `src/models/model.py`
- Modify: `src/training/train.py`
- Modify: `src/evaluation/evaluate.py`
- Modify: `scripts/experiments/run_corrected_matrix.py`
- Modify: `tests/test_model_forward.py`
- Modify: `tests/test_corrected_experiment_matrix.py`
- Modify: `tests/test_corrected_pipeline_integration.py`

- [ ] **Step 1: Write failing two-channel model, batch, and one-epoch integration tests**

Add tests that configure `['radial', 'tangential']`, verify `Conv1d.in_channels == 2`, check finite gradients through both channels, and assert `_PreparedV2Batch` keeps both `model_input` and radial-only physics data:

```python
def test_rt_model_accepts_two_channels_and_backpropagates() -> None:
    config = yaml.safe_load(Path("configs/config_v2.yaml").read_text())
    config["model"]["input_components"] = ["radial", "tangential"]
    model = PINNModel(config).train()
    waveform = torch.randn(2, 2, 200, requires_grad=True)
    prediction = model.predict_heads(waveform, meta=_make_meta(2, torch.device("cpu")))
    (prediction.stf_encoded.mean() + prediction.catalog_mw.mean()).backward()
    assert model.embed[0].in_channels == 2
    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()


def test_prepare_v2_batch_stacks_rt_but_keeps_radial_physics() -> None:
    config = rt_config()
    batch = _prepared_batch()
    batch["tangential"] = torch.full((1, 1, 200), 2.0)
    prepared = _prepare_v2_batch(batch, config, torch.device("cpu"))
    assert prepared.model_input.shape == (1, 2, 200)
    assert prepared.radial.shape == (1, 1, 200)


def test_radial_checkpoint_is_strictly_compatible_only_with_radial_config() -> None:
    radial_config = yaml.safe_load(Path("configs/config_v2.yaml").read_text())
    radial_state = PINNModel(radial_config).state_dict()
    PINNModel(radial_config).load_state_dict(radial_state, strict=True)

    rt_config = copy.deepcopy(radial_config)
    rt_config["model"]["input_components"] = ["radial", "tangential"]
    with pytest.raises(RuntimeError, match="size mismatch"):
        PINNModel(rt_config).load_state_dict(radial_state, strict=True)
```

Make the existing one-epoch active workflow integration test run once with R+T and still strictly reload/evaluate its checkpoint.

- [ ] **Step 2: Run focused tests and confirm channel failures**

Run:

```bash
source /home/lihe/.config/pinn/server.env
/home/lihe/PINN_Mag/venv/bin/pytest -q \
  tests/test_model_forward.py \
  tests/test_corrected_experiment_matrix.py \
  tests/test_corrected_pipeline_integration.py -k 'rt or active_workflow or backpropagates'
```

Expected: model input-channel mismatch and missing `model_input` failures.

- [ ] **Step 3: Make model input channels config-driven**

In `PINNModel.__init__`, retain radial default compatibility:

```python
from src.utils.config_v2 import waveform_input_components_from_config

self.input_components = waveform_input_components_from_config(config)
self.input_channels = len(self.input_components)
self.embed = nn.Sequential(
    nn.Conv1d(
        in_channels=self.input_channels,
        out_channels=self.hidden_dim,
        kernel_size=7,
        padding=3,
    ),
    nn.GELU(),
    nn.GroupNorm(num_groups=8, num_channels=self.hidden_dim),
)
```

- [ ] **Step 4: Route shared model input through train/evaluate/smoke**

Add `model_input: torch.Tensor` to `_PreparedV2Batch` and build it with `assemble_model_input(batch, config).to(device)`. Pass `prepared_v2.model_input` to model heads in both training and validation, but continue passing `prepared_v2.radial` as `radial_obs` to `STFRateWaveformLossV2`.

In `src/evaluation/evaluate.py`, assemble the v2 model input for the main loop and sample-grid loop; retain radial for baseline metrics and plots. In `run_corrected_matrix._assert_backpropagates`, pass `prepared.model_input` to the model and `prepared.radial` to the criterion.

The essential separation is:

```python
heads = model.predict_heads(
    prepared.model_input,
    meta=prepared.metadata,
)
loss, metrics = criterion(
    heads.stf_encoded,
    pred_catalog_mw=heads.catalog_mw,
    radial_obs=prepared.radial,
    source_distance_m=prepared.source_distance_m,
    theta_deg=prepared.theta_deg,
    phi_slip_deg=prepared.phi_slip_deg,
    source_dt_sec=prepared.source_dt_sec,
    observation_dt_sec=prepared.observation_dt_sec,
    waveform_valid_mask=prepared.waveform_valid_mask,
    stf_true=prepared.stf_true,
    has_stf=prepared.has_stf,
    true_mag=prepared.true_mag,
)
```

- [ ] **Step 5: Run focused tests and commit**

Run the Step 2 command; expect pass. Commit:

```bash
git add src/models/model.py src/training/train.py src/evaluation/evaluate.py \
  scripts/experiments/run_corrected_matrix.py tests/test_model_forward.py \
  tests/test_corrected_experiment_matrix.py \
  tests/test_corrected_pipeline_integration.py
git commit -m "feat: support radial tangential model input"
```

### Task 4: Use identical R+T ordering for external-event inference

**Files:**
- Modify: `src/evaluation/evaluate_unseen.py`
- Modify: `tests/test_unseen_event_eval.py`

- [ ] **Step 1: Write a failing external R+T channel-order test**

Extend the active unseen test with an R+T config and record the waveform received by the fake model:

```python
received: list[torch.Tensor] = []

class ScalarHeadModel:
    def predict_heads(self, waveform, meta=None):
        received.append(waveform.detach().cpu())
        return PINNPrediction(
            stf_encoded=torch.zeros(waveform.shape[0], 300),
            catalog_mw=torch.full((waveform.shape[0],), 7.3),
        )

sample = {
    "radial": np.ones(200, dtype=np.float32),
    "tangential": np.full(200, 2.0, dtype=np.float32),
    "source_distance_m": 20_000.0,
    "epicentral_distance_m": 17_000.0,
    "theta_deg": 30.0,
    "azimuth_deg": 45.0,
    "waveform_dt_sec": 1.0,
    "radial_peak_cm": 3.0,
}

assert received[0].shape == (1, 2, 200)
torch.testing.assert_close(received[0][0, 0], torch.ones(200))
torch.testing.assert_close(received[0][0, 1], torch.full((200,), 2.0))
```

- [ ] **Step 2: Run and confirm the existing radial-only construction fails**

Run:

```bash
source /home/lihe/.config/pinn/server.env
/home/lihe/PINN_Mag/venv/bin/pytest -q \
  tests/test_unseen_event_eval.py -k 'active_unseen or external_adapter'
```

Expected: received waveform has one channel instead of two.

- [ ] **Step 3: Assemble external input with the shared helper**

Convert sample components to `(1,1,T)` tensors and call `assemble_model_input` before `_predict_outputs`:

```python
component_tensors = {
    name: torch.as_tensor(
        sample[name],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, -1)
    for name in waveform_input_components_from_config(config)
}
waveform = assemble_model_input(component_tensors, config)
waveform = _ensure_time_steps(waveform, time_steps)
```

Continue storing radial traces for PGD, plots, and radial-peak reporting.

- [ ] **Step 4: Run focused tests and commit**

Run the Step 2 command; expect pass. Commit:

```bash
git add src/evaluation/evaluate_unseen.py tests/test_unseen_event_eval.py
git commit -m "feat: evaluate external events with configured components"
```

### Task 5: Add mechanism summaries and the bounded seed-42 R+T pilot runner

**Files:**
- Create: `src/evaluation/mechanism_metrics.py`
- Create: `scripts/experiments/run_rt_pilot.py`
- Create: `tests/test_mechanism_metrics.py`
- Create: `tests/test_rt_pilot.py`

- [ ] **Step 1: Write failing mechanism-classification and gate tests**

Use the repository's existing rake boundaries and approved thresholds:

```python
@pytest.mark.parametrize(
    ("rake", "expected"),
    [(0.0, "strike-slip"), (180.0, "strike-slip"),
     (90.0, "reverse"), (-90.0, "normal"),
     (float("nan"), "unknown")],
)
def test_mechanism_from_rake(rake: float, expected: str) -> None:
    assert mechanism_from_rake(rake) == expected


def test_rt_gate_requires_all_three_approved_metrics() -> None:
    gate = evaluate_rt_gate(
        station_mae=0.0885,
        event_mae=0.1613,
        strike_slip_event_mae=0.1800,
    )
    assert gate.passed
    assert not evaluate_rt_gate(
        station_mae=0.0886,
        event_mae=0.1500,
        strike_slip_event_mae=0.1700,
    ).passed
```

Also test that the runner rejects config differences other than `model.input_components`, a changed split, non-200 epochs, non-finite metrics, unexpected accepted counts, and a nonempty output without `--resume`.

- [ ] **Step 2: Run new tests and confirm missing-module failures**

Run:

```bash
source /home/lihe/.config/pinn/server.env
/home/lihe/PINN_Mag/venv/bin/pytest -q \
  tests/test_mechanism_metrics.py tests/test_rt_pilot.py
```

Expected: import failures for the new modules.

- [ ] **Step 3: Implement deterministic mechanism summaries**

Create `src/evaluation/mechanism_metrics.py` with `mechanism_from_rake` using the current `30/150` degree boundaries and a summarizer returning event count, station count, MAE, bias, RMSE, and `abs(error)>0.33` count. Reject missing/non-finite prediction errors instead of silently dropping them.

```python
def mechanism_from_rake(rake_deg: float) -> str:
    rake = float(rake_deg)
    if not math.isfinite(rake):
        return "unknown"
    rake = ((rake + 180.0) % 360.0) - 180.0
    if abs(rake) <= 30.0 or abs(rake) >= 150.0:
        return "strike-slip"
    if 30.0 <= rake <= 150.0:
        return "reverse"
    if -150.0 <= rake <= -30.0:
        return "normal"
    return "unknown"
```

- [ ] **Step 4: Implement the isolated pilot runner**

`run_rt_pilot.py` must:

1. require the completed phase-9 USGS baseline, candidate NPZ, baseline config, baseline seed-42 split, external label manifest, event root, unique output root, and exactly 200 epochs;
2. require a clean Git worktree and refuse overwrite unless `--resume`;
3. deep-copy the baseline config and add only `model.input_components: [radial, tangential]`;
4. build both baseline and R+T datasets, assert identical ordered `event::station` keys and radial arrays, assert 31 events/2483 samples, and record SHA-256 for ordered R and R+T arrays;
5. run CPU and CUDA finite forward/loss/backward smoke;
6. call `run_corrected_matrix.run_matrix` for seed 42 only, 200 epochs, and `within-event-station` mode;
7. strictly compare the new split manifest with the frozen seed-42 split;
8. summarize internal mechanism metrics and the `Mw <= 6.9` strike-slip subset from the generated event CSV and candidate NPZ rakes;
9. apply the approved gate below;
10. only on a passing gate, evaluate the fixed eight external events once with threshold override `0.0`, pair USGS labels, and write overall/mechanism/Luding summaries;
11. atomically write config, audit, hashes, gate, summary, and either `COMPLETE` or `INTERNAL_GATE_FAILED`.

The CLI must also accept `--preflight-only`, which performs items 1–5, writes `PREFLIGHT_COMPLETE`, and exits without creating a training checkpoint.

Use an immutable gate result:

```python
@dataclass(frozen=True)
class RTPilotGate:
    passed: bool
    station_mae: float
    event_mae: float
    strike_slip_event_mae: float
    station_limit: float = 0.0885
    event_limit: float = 0.1613
    strike_slip_limit: float = 0.1801


def evaluate_rt_gate(
    *,
    station_mae: float,
    event_mae: float,
    strike_slip_event_mae: float,
) -> RTPilotGate:
    values = tuple(map(float, (
        station_mae,
        event_mae,
        strike_slip_event_mae,
    )))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("R+T pilot gate metrics must be finite")
    return RTPilotGate(
        passed=(
            values[0] <= 0.0885
            and values[1] <= 0.1613
            and values[2] < 0.1801
        ),
        station_mae=values[0],
        event_mae=values[1],
        strike_slip_event_mae=values[2],
    )
```

- [ ] **Step 5: Run focused tests and commit**

Run the Step 2 command; expect pass. Commit:

```bash
git add src/evaluation/mechanism_metrics.py \
  scripts/experiments/run_rt_pilot.py \
  tests/test_mechanism_metrics.py tests/test_rt_pilot.py
git commit -m "feat: add controlled radial tangential pilot"
```

### Task 6: Verify, freeze the implementation, and run the seed-42 pilot

**Files:**
- Modify: `.planning/2026-07-23-usgs-priority-magnitude-relabeling/task_plan.md` (untracked operational state only)
- Modify: `.planning/2026-07-23-usgs-priority-magnitude-relabeling/findings.md` (untracked operational state only)
- Modify: `.planning/2026-07-23-usgs-priority-magnitude-relabeling/progress.md` (untracked operational state only)

- [ ] **Step 1: Run focused and complete verification**

```bash
source /home/lihe/.config/pinn/server.env
/home/lihe/PINN_Mag/venv/bin/pytest -q \
  tests/test_config_v2.py tests/test_model_input.py \
  tests/test_corrected_pipeline_integration.py \
  tests/test_model_forward.py tests/test_corrected_experiment_matrix.py \
  tests/test_unseen_event_eval.py tests/test_mechanism_metrics.py \
  tests/test_rt_pilot.py
/home/lihe/PINN_Mag/venv/bin/pytest -q
/home/lihe/PINN_Mag/venv/bin/python -m compileall -q src scripts tests
```

Expected: focused tests pass, complete suite has no failures, compileall exits 0.

- [ ] **Step 2: Run real-data audit and CPU/CUDA finite smoke without training**

Run the pilot runner's preflight mode against the immutable USGS snapshot:

```bash
source /home/lihe/.config/pinn/server.env
/home/lihe/PINN_Mag/venv/bin/python scripts/experiments/run_rt_pilot.py \
  --snapshot-root /home/lihe/PINN_Mag/data/magnitude-label-snapshots/usgs-priority-20260723T044422Z-40d808a \
  --baseline-run /home/lihe/PINN_Mag/runs/phase9-usgs-relabel-20260723T051044Z-bb6d640 \
  --event-root /home/lihe/PINN_Mag/incoming/legacy-task17/GNSS_EQDATA \
  --output-root /home/lihe/PINN_Mag/runs/phase10-rt-preflight-9b45fb4 \
  --epochs 200 \
  --preflight-only
```

Expected: 31 events, 2483 accepted samples, unchanged ordered sample keys and radial hash, two-channel input hash recorded, finite CPU/CUDA forward/loss/gradients, and `PREFLIGHT_COMPLETE`.

- [ ] **Step 3: Commit final implementation-only adjustments**

Check `git diff --check`, confirm `.planning/` remains untracked, and commit any verified implementation adjustments. Require a clean tracked worktree before training.

- [ ] **Step 4: Create a detached run worktree and unique run directory**

Use `superpowers:using-git-worktrees` at execution time. Create a detached worktree at the verified implementation commit and a run directory named with UTC timestamp plus commit prefix. Do not modify or delete any existing worktree/run/snapshot.

```bash
RUN_COMMIT="$(git rev-parse HEAD)"
RUN_PREFIX="$(git rev-parse --short=7 HEAD)"
RUN_WORKTREE="/home/lihe/PINN_Mag/worktrees/run-rt-${RUN_PREFIX}"
RUN_ID="phase10-rt-seed42-$(date -u +%Y%m%dT%H%M%SZ)-${RUN_PREFIX}"
RUN_ROOT="/home/lihe/PINN_Mag/runs/${RUN_ID}"
git worktree add --detach "$RUN_WORKTREE" "$RUN_COMMIT"
mkdir -p "$RUN_ROOT"
```

- [ ] **Step 5: Run seed 42 only under the approved tmux/systemd boundary**

Launch the exact pilot command in `pinn-run:train`:

```bash
tmux send-keys -t pinn-run:train C-c
tmux send-keys -t pinn-run:train \
  "cd '$RUN_WORKTREE' && source /home/lihe/.config/pinn/server.env && systemd-inhibit --what=sleep --mode=block /home/lihe/PINN_Mag/venv/bin/python scripts/experiments/run_rt_pilot.py --snapshot-root /home/lihe/PINN_Mag/data/magnitude-label-snapshots/usgs-priority-20260723T044422Z-40d808a --baseline-run /home/lihe/PINN_Mag/runs/phase9-usgs-relabel-20260723T051044Z-bb6d640 --event-root /home/lihe/PINN_Mag/incoming/legacy-task17/GNSS_EQDATA --output-root '$RUN_ROOT' --epochs 200 2>&1 | tee '$RUN_ROOT/train.log'" C-m
tmux send-keys -t pinn-run:monitor C-c
tmux send-keys -t pinn-run:monitor \
  "watch -n 10 'nvidia-smi; find $RUN_ROOT -maxdepth 2 -type f -printf \"%TY-%Tm-%Td %TH:%TM:%TS %p\\n\" | sort | tail -20'" C-m
tmux send-keys -t pinn-run:logs C-c
tmux send-keys -t pinn-run:logs \
  "tail -F '$RUN_ROOT/train.log'" C-m
```

Mirror live status into `pinn-run:monitor` and logs into `pinn-run:logs`; do not rely on the chat process for training persistence.

- [ ] **Step 6: Validate and report without starting more seeds**

Independently reload the checkpoint strictly, verify the split hash, scan all 200 log rows and prediction values for finite values, verify artifact hashes/markers, and report:

- R-only versus R+T station/event MAE;
- mechanism-stratified internal MAE and low-Mw strike-slip MAE;
- gate pass/fail for all three approved criteria;
- if gate passed, fixed-eight-event overall/mechanism results and Luding station/event comparison.

Stop after reporting. Do not train seeds 17 or 73 without a new user decision.
