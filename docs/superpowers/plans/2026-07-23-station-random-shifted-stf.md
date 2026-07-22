# Station-Random Shifted-STF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现站内随机拆分、P 到时平移且固定 300 秒截断的 STF、独立目录 Mw 标量头，以及与之匹配的训练和评估流程。

**Architecture:** 保留 200 秒 canonical source STF 作为事件级来源，在 dataset 层使用唯一旅行时 provider 派生每个台站的 300 步窗口 STF；模型共享编码器后分为 STF 序列头和目录 Mw 标量头。新的 station-random workflow 使用显式双头接口，历史 `forward()` 和旧配置继续可读，但不能进入新的正式训练入口。

**Tech Stack:** Python 3.12、NumPy、PyTorch、PyYAML、pytest、现有 checkpoint/provenance 基础设施。

---

### Task 1: Active Workflow Config And Travel-Time Provider

**Files:**
- Create: `src/physics/__init__.py`
- Create: `src/physics/travel_time.py`
- Create: `tests/test_travel_time_provider.py`
- Modify: `src/utils/config_v2.py`
- Modify: `configs/config_v2.yaml`
- Modify: `tests/test_config_v2.py`

- [x] **Step 1: Write failing config and provider tests**

```python
def test_constant_velocity_provider_returns_p_s_and_relative_delays() -> None:
    provider = ConstantVelocityTravelTime(alpha_m_per_s=8.0, beta_m_per_s=4.0)
    delays = provider.delays(torch.tensor([8.0, 16.0]))
    assert torch.equal(delays.p_sec, torch.tensor([1.0, 2.0]))
    assert torch.equal(delays.s_sec, torch.tensor([2.0, 4.0]))
    assert torch.equal(delays.s_after_p_sec, torch.tensor([1.0, 2.0]))


def test_active_station_workflow_requires_fixed_contract() -> None:
    config = _minimal_v2()
    config["workflow"] = "station_random_shifted_stf"
    config["dataset"]["stf"]["station_window_duration_sec"] = 300.0
    config["dataset"]["stf"]["station_alignment"] = "p_arrival"
    config["dataset"]["stf"]["station_preserve_integral"] = False
    config["physics"]["travel_time_model"] = "constant_velocity"
    config["physics"]["delay_mode"] = "p_aligned_relative"
    config["model"] = {"predict_catalog_mw": True}
    config["training"].update(
        split_protocol="within_event_station",
        event_balanced_sampling=False,
        early_stop_patience=0,
        checkpoint_metric="station_mae_catalog",
    )
    validate_config_v2(config)
    assert stf_output_steps_from_config(config) == 300
```

- [x] **Step 2: Verify RED**

Run: `source /home/lihe/.config/pinn/server.env && "$PINN_ENV/bin/python" -m pytest tests/test_config_v2.py tests/test_travel_time_provider.py -q`

Expected: FAIL because the workflow keys and `src.physics.travel_time` do not exist.

- [x] **Step 3: Implement the provider and strict active-workflow branch**

```python
@dataclass(frozen=True)
class TravelTimeDelays:
    p_sec: Any
    s_sec: Any
    s_after_p_sec: Any


@dataclass(frozen=True)
class ConstantVelocityTravelTime:
    alpha_m_per_s: float
    beta_m_per_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha_m_per_s) or self.alpha_m_per_s <= 0.0:
            raise ValueError("alpha_m_per_s must be positive and finite")
        if not math.isfinite(self.beta_m_per_s) or self.beta_m_per_s <= 0.0:
            raise ValueError("beta_m_per_s must be positive and finite")

    def delays(self, source_distance_m: Any) -> TravelTimeDelays:
        p_sec = source_distance_m / self.alpha_m_per_s
        s_sec = source_distance_m / self.beta_m_per_s
        return TravelTimeDelays(
            p_sec=p_sec,
            s_sec=s_sec,
            s_after_p_sec=s_sec - p_sec,
        )


def travel_time_from_config(config: dict[str, Any]) -> ConstantVelocityTravelTime:
    if config["physics"]["travel_time_model"] != "constant_velocity":
        raise ValueError("unsupported physics.travel_time_model")
    return ConstantVelocityTravelTime(
        alpha_m_per_s=float(config["physics"]["alpha"]),
        beta_m_per_s=float(config["physics"]["beta"]),
    )
```

The active `workflow=station_random_shifted_stf` branch must require exactly the
approved keys and values. Historical configs without `workflow` retain their
existing validation semantics for immutable diagnostic compatibility.

- [x] **Step 4: Verify GREEN and commit**

Run the focused command from Step 2. Expected: all focused tests pass.

```bash
git add src/physics src/utils/config_v2.py configs/config_v2.yaml tests/test_config_v2.py tests/test_travel_time_provider.py
git commit -m "feat: define station-aligned travel-time contract"
```

### Task 2: Station-Window STF And Dataset Audit

**Files:**
- Modify: `src/data/stf.py`
- Modify: `src/data/dataset_v2.py`
- Modify: `src/data/manifest.py`
- Modify: `scripts/data/audit_corrected_dataset.py`
- Modify: `tests/test_stf_targets.py`
- Modify: `tests/test_corrected_pipeline_integration.py`
- Modify: `tests/test_dataset_manifest.py`

- [ ] **Step 1: Write failing station-shift tests**

```python
def test_station_window_shifts_fractionally_and_truncates_without_rescaling() -> None:
    source = ProcessedSTF(
        time_sec=np.arange(4.0),
        rate_nm_per_s=np.array([0.0, 1.0, 2.0, 1.0]),
        dt_sec=1.0,
        native_moment_nm=4.0,
        grid_moment_before_rescale_nm=4.0,
        retained_moment_fraction=1.0,
        mw_native=moment_to_mw(4.0),
    )
    shifted = shift_source_stf_to_station_window(
        source,
        p_delay_sec=2.5,
        duration_sec=5.0,
        sample_rate_hz=1.0,
    )
    np.testing.assert_allclose(shifted.rate_nm_per_s, [0.0, 0.0, 0.0, 0.5, 1.5])
    assert shifted.window_moment_nm == pytest.approx(2.0)
    assert shifted.retained_moment_fraction == pytest.approx(0.5)


def test_dataset_same_event_stations_share_full_moment_but_not_shifted_stf(tmp_path: Path) -> None:
    dataset = CorrectedEarthquakeDataset(make_station_workflow_config(tmp_path))
    first, second = dataset.samples
    assert first["full_event_moment_nm"] == second["full_event_moment_nm"]
    assert not np.array_equal(first["stf"], second["stf"])
    assert first["p_arrival_sec"] != second["p_arrival_sec"]
    assert first["stf"].shape == second["stf"].shape == (300,)
```

- [ ] **Step 2: Verify RED**

Run: `source /home/lihe/.config/pinn/server.env && "$PINN_ENV/bin/python" -m pytest tests/test_stf_targets.py tests/test_corrected_pipeline_integration.py tests/test_dataset_manifest.py -q`

Expected: FAIL because station-window STF types, fields, and audit semantics are absent.

- [ ] **Step 3: Implement the station-window derivation**

```python
@dataclass(frozen=True)
class StationWindowSTF:
    time_sec: np.ndarray
    rate_nm_per_s: np.ndarray
    dt_sec: float
    full_event_moment_nm: float
    window_moment_nm: float
    retained_moment_fraction: float
    mw_window: float


def shift_source_stf_to_station_window(
    source: ProcessedSTF,
    *,
    p_delay_sec: float,
    duration_sec: float,
    sample_rate_hz: float,
) -> StationWindowSTF:
    delay = _finite_real(p_delay_sec, "p_delay_sec")
    if delay < 0.0:
        raise ValueError("p_delay_sec must be nonnegative")
    duration = _finite_real(duration_sec, "duration_sec")
    sample_rate = _finite_real(sample_rate_hz, "sample_rate_hz")
    if duration <= 0.0 or sample_rate <= 0.0:
        raise ValueError("station target duration and sample rate must be positive")
    dt_sec = 1.0 / sample_rate
    steps = int(round(duration * sample_rate))
    time_sec = np.arange(steps, dtype=np.float64) * dt_sec
    rate = np.interp(
        time_sec - delay,
        source.time_sec,
        source.rate_nm_per_s,
        left=0.0,
        right=0.0,
    )
    full_moment = float(np.sum(source.rate_nm_per_s) * source.dt_sec)
    window_moment = float(np.sum(rate) * dt_sec)
    fraction = window_moment / full_moment
    tolerance = 1.0e-10
    if not math.isfinite(fraction) or fraction < -tolerance or fraction > 1.0 + tolerance:
        raise ValueError("station retained moment fraction is outside [0, 1]")
    fraction = min(1.0, max(0.0, fraction))
    return StationWindowSTF(
        time_sec=time_sec,
        rate_nm_per_s=rate,
        dt_sec=dt_sec,
        full_event_moment_nm=full_moment,
        window_moment_nm=window_moment,
        retained_moment_fraction=fraction,
        mw_window=(
            moment_to_mw(window_moment)
            if window_moment > 0.0
            else float("nan")
        ),
    )
```

Dataset processing must build waveform/geometry first, obtain P/S delays from
`travel_time_from_config`, then shift the cached canonical STF. Do not cache the
station target and do not scale it to catalog magnitude.

- [ ] **Step 4: Replace audit semantics**

Manifest rows add `p_arrival_sec`, `s_arrival_sec`,
`full_event_moment_nm`, `station_window_moment_nm`, and `mw_stf_window`.
`build_dataset_summary` must report finite fraction min/mean/max and verify full
event moment equality within each event. `audit_passes(summary)` must not accept
or compare a minimum retained fraction.

- [ ] **Step 5: Verify GREEN and commit**

Run the focused command from Step 2. Expected: all focused tests pass.

```bash
git add src/data/stf.py src/data/dataset_v2.py src/data/manifest.py scripts/data/audit_corrected_dataset.py tests/test_stf_targets.py tests/test_corrected_pipeline_integration.py tests/test_dataset_manifest.py
git commit -m "feat: derive truncated station STF targets"
```

### Task 3: Stable Within-Event Station Split And Complete Manifest

**Files:**
- Modify: `src/data/splits.py`
- Modify: `src/data/loaders_v2.py`
- Modify: `tests/test_group_splits.py`
- Modify: `tests/test_provenance.py`

- [ ] **Step 1: Write failing split tests**

```python
def test_station_split_is_order_independent_and_has_no_key_overlap() -> None:
    keys = [("A", "A03"), ("A", "A01"), ("A", "A02"), ("B", "B01")]
    first = make_within_event_station_split(keys, 0.15, 0.15, seed=42)
    reversed_keys = list(reversed(keys))
    second = make_within_event_station_split(reversed_keys, 0.15, 0.15, seed=42)

    def assigned(split: EventGroupSplit, source: list[tuple[str, str]]) -> tuple[set, set, set]:
        return (
            {source[index] for index in split.train_indices},
            {source[index] for index in split.validation_indices},
            {source[index] for index in split.test_indices},
        )

    first_sets = assigned(first, keys)
    second_sets = assigned(second, reversed_keys)
    assert first_sets == second_sets
    assert not (first_sets[0] & first_sets[1])
    assert not (first_sets[0] & first_sets[2])
    assert not (first_sets[1] & first_sets[2])


def test_real_split_contract_has_expected_seed_42_counts(real_config: dict) -> None:
    _, _, _, manifest = get_data_loaders_v2(real_config)
    assert manifest["train_record_count"] == 1735
    assert manifest["validation_record_count"] == 374
    assert manifest["test_record_count"] == 374
    assert manifest["station_weighted_catalog_mw_mean"] == pytest.approx(
        {"train": 8.0469, "validation": 8.0377, "test": 8.0377},
        abs=1.0e-4,
    )
```

- [ ] **Step 2: Verify RED**

Run: `source /home/lihe/.config/pinn/server.env && "$PINN_ENV/bin/python" -m pytest tests/test_group_splits.py tests/test_provenance.py -q`

Expected: FAIL because split ownership and complete manifest fields are absent.

- [ ] **Step 3: Implement stable key assignment**

```python
def make_within_event_station_split(
    sample_keys: list[tuple[str, str]],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> EventGroupSplit:
    by_event: dict[str, list[tuple[str, int]]] = {}
    for index, (event, station) in enumerate(sample_keys):
        by_event.setdefault(event, []).append((station, index))
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for event in sorted(by_event):
        ordered = sorted(by_event[event])
        order = rng.permutation(len(ordered)).tolist()
        shuffled = [ordered[position][1] for position in order]
        maximum_held_out = max(0, len(shuffled) - 1)
        validation_count = _requested_station_count(len(shuffled), validation_fraction)
        test_count = _requested_station_count(len(shuffled), test_fraction)
        while validation_count + test_count > maximum_held_out:
            if validation_count >= test_count and validation_count > 0:
                validation_count -= 1
            else:
                test_count -= 1
        test.extend(shuffled[:test_count])
        validation.extend(shuffled[test_count:test_count + validation_count])
        train.extend(shuffled[test_count + validation_count:])
    return EventGroupSplit(sorted(train), sorted(validation), sorted(test))
```

The loader manifest must include per-split sample keys, per-event counts,
catalog-Mw distribution summaries, seed, protocol, and SHA-256 over canonical
JSON. Fail on duplicate/missing keys, invalid total fractions, or catalog-Mw mean
drift above 0.05. Disable event-balanced sampling for the active workflow.

- [ ] **Step 4: Verify GREEN and commit**

Run the focused command from Step 2. Expected: all focused tests pass.

```bash
git add src/data/splits.py src/data/loaders_v2.py tests/test_group_splits.py tests/test_provenance.py
git commit -m "feat: freeze within-event station splits"
```

### Task 4: Dual-Head Model And P-Aligned Physics Loss

**Files:**
- Modify: `src/models/model.py`
- Modify: `src/training/loss_stf_rate_v2.py`
- Modify: `tests/test_model_forward.py`
- Modify: `tests/test_forward_operator.py`
- Modify: `tests/test_stf_rate_loss_v2.py`

- [ ] **Step 1: Write failing dual-head and loss tests**

```python
def _active_config() -> dict:
    return yaml.safe_load(
        Path("configs/config_v2.yaml").read_text(encoding="utf-8")
    )


def test_predict_heads_returns_300_step_stf_and_scalar_catalog_mw() -> None:
    config = _active_config()
    prediction = PINNModel(config).predict_heads(
        torch.randn(3, 1, 200),
        meta=torch.randn(3, 5),
    )
    assert prediction.stf_encoded.shape == (3, 300)
    assert prediction.catalog_mw.shape == (3,)


def test_magnitude_loss_uses_scalar_head_not_window_integral() -> None:
    criterion = STFRateWaveformLossV2(_active_config())
    pred_stf = torch.zeros(2, 300, requires_grad=True)
    pred_mw = torch.tensor([7.0, 8.0], requires_grad=True)
    loss, metrics = criterion(
        pred_stf,
        pred_catalog_mw=pred_mw,
        radial_obs=torch.zeros(2, 1, 200),
        source_distance_m=torch.tensor([1000.0, 2000.0]),
        theta_deg=torch.tensor([30.0, 45.0]),
        phi_slip_deg=torch.tensor([10.0, 20.0]),
        source_dt_sec=torch.ones(2),
        observation_dt_sec=torch.ones(2),
        waveform_valid_mask=torch.ones(2, 200, dtype=torch.bool),
        stf_true=torch.zeros(2, 300),
        has_stf=torch.tensor([True, True]),
        true_mag=torch.tensor([7.5, 8.5]),
    )
    loss.backward()
    assert pred_mw.grad is not None
    assert metrics["L_mag"] == pytest.approx(0.25)


def test_p_aligned_forward_uses_zero_p_and_relative_s_delay() -> None:
    rate = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    zeros = torch.zeros(1)
    ones = torch.ones(1)
    displacement = forward_displacement_from_rate(
        rate,
        source_dt_sec=torch.tensor([1.0]),
        observation_dt_sec=torch.tensor([1.0]),
        observation_steps=5,
        source_distance_m=torch.tensor([8.0]),
        travel_time=ConstantVelocityTravelTime(8.0, 4.0),
        C_int_P=zeros,
        C_int_S=zeros,
        C_far_P=ones,
        C_far_S=ones,
        include_intermediate=False,
        include_far_P=True,
        include_far_S=True,
        include_intermediate_P=False,
        include_intermediate_S=False,
    )
    assert torch.equal(
        displacement,
        torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0]]),
    )
```

- [ ] **Step 2: Verify RED**

Run: `source /home/lihe/.config/pinn/server.env && "$PINN_ENV/bin/python" -m pytest tests/test_model_forward.py tests/test_forward_operator.py tests/test_stf_rate_loss_v2.py -q`

Expected: FAIL because `predict_heads`, the scalar head, and P-aligned loss are absent.

- [ ] **Step 3: Implement the shared encoder and dual heads**

```python
@dataclass(frozen=True)
class PINNPrediction:
    stf_encoded: torch.Tensor
    catalog_mw: torch.Tensor


def predict_heads(self, x: torch.Tensor, meta: Optional[torch.Tensor] = None) -> PINNPrediction:
    if self.magnitude_head is None:
        raise RuntimeError("catalog magnitude head is disabled")
    sequence = self._encode_sequence(x, meta)
    stf_sequence = self._resize_source_time(sequence)
    stf_encoded = self.rate_head(stf_sequence).squeeze(-1)
    pooled = sequence.mean(dim=1)
    catalog_mw = self.magnitude_head(pooled).squeeze(-1)
    return PINNPrediction(stf_encoded=stf_encoded, catalog_mw=catalog_mw)


def forward(self, x: torch.Tensor, meta: Optional[torch.Tensor] = None) -> torch.Tensor:
    sequence = self._encode_sequence(x, meta)
    stf_sequence = self._resize_source_time(sequence)
    return self.rate_head(stf_sequence).squeeze(-1)
```

The scalar head is `Linear -> GELU -> Dropout -> Linear` with its last bias
initialized from `model.catalog_mw_initial_bias=8.0`; it remains unconstrained
after initialization. Legacy models without `predict_catalog_mw` do not create
the head, preserving historical state-dict compatibility.

- [ ] **Step 4: Implement P-zero/S-relative synthesis and scalar magnitude loss**

`STFRateWaveformLossV2.forward` receives required `pred_catalog_mw` in the active
workflow. `L_mag` is MSE against finite catalog Mw. The detached window integral
is returned as `window_mw_mean`; it never enters `L_mag`.

- [ ] **Step 5: Verify GREEN and commit**

Run the focused command from Step 2. Expected: all focused tests pass and every
active model parameter receives gradient from the combined loss.

```bash
git add src/models/model.py src/training/loss_stf_rate_v2.py tests/test_model_forward.py tests/test_forward_operator.py tests/test_stf_rate_loss_v2.py
git commit -m "feat: add catalog magnitude prediction head"
```

### Task 5: Station-Level Training, Evaluation, And Full-Epoch Control

**Files:**
- Modify: `src/training/train.py`
- Modify: `src/evaluation/evaluate.py`
- Modify: `src/evaluation/evaluate_unseen.py`
- Modify: `tests/test_training_checkpointing.py`
- Modify: `tests/test_event_metrics.py`
- Modify: `tests/test_unseen_event_eval.py`

- [ ] **Step 1: Write failing active-workflow integration tests**

```python
def test_v2_batch_uses_catalog_magnitude_as_scalar_target(tmp_path: Path) -> None:
    config = _training_config(tmp_path)
    batch = _training_batch()
    prepared = _prepare_v2_batch(batch, config, torch.device("cpu"))
    assert torch.equal(prepared.true_mag, batch["magnitude_catalog"])


def test_active_training_selects_best_by_station_catalog_mae_without_stopping(tmp_path: Path) -> None:
    config = _training_config(tmp_path)
    config["training"]["epochs"] = 2
    config["training"]["early_stop_patience"] = 0
    result = train(
        config=config,
        data_loaders=_injected_loaders(),
    )
    with Path(result["log_file"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert "validation_station_mae_catalog" in rows[0]
    assert result["checkpoint_metric"] == "station_mae_catalog"
```

- [ ] **Step 2: Verify RED**

Run: `source /home/lihe/.config/pinn/server.env && "$PINN_ENV/bin/python" -m pytest tests/test_training_checkpointing.py tests/test_event_metrics.py tests/test_unseen_event_eval.py -q`

Expected: FAIL because active callers still integrate STF and the log omits station catalog MAE.

- [ ] **Step 3: Update training**

For the active workflow, call `model.predict_heads`, pass both heads to the loss,
accumulate station-level catalog MAE from `prediction.catalog_mw`, and save
`best_model.pth` whenever that MAE improves. Keep `early_stop_patience=0`, loop
through every configured epoch, and preserve the existing full-state checkpoint,
RNG, scheduler, signal, and resume behavior. Training CSV columns must include
`validation_station_mae_catalog`, `validation_event_mae_catalog`, and
`window_mw_mean`.

- [ ] **Step 4: Update internal and external evaluation**

Use `prediction.catalog_mw` as `mw_pred`. Decode and integrate the STF only into
`mw_window_pred`/time-series diagnostics. Internal primary output remains the
the `station_mae` value returned by `summarize_predictions`; event median metrics and
the eight external events remain secondary diagnostics.

- [ ] **Step 5: Verify GREEN and commit**

Run the focused command from Step 2. Expected: all focused tests pass.

```bash
git add src/training/train.py src/evaluation/evaluate.py src/evaluation/evaluate_unseen.py tests/test_training_checkpointing.py tests/test_event_metrics.py tests/test_unseen_event_eval.py
git commit -m "feat: train and evaluate station catalog magnitude"
```

### Task 6: End-To-End Gates And Real-Data Smoke

**Files:**
- Modify: `tests/test_corrected_pipeline_integration.py`
- Modify: `tests/test_corrected_experiment_matrix.py`
- Modify: `docs/superpowers/specs/2026-07-23-station-random-shifted-stf-design.md` only if implementation reveals a wording mismatch

- [ ] **Step 1: Add one end-to-end active workflow test**

The test builds a small real-contract dataset, freezes a station split, runs one
CPU epoch, reloads the best checkpoint strictly, evaluates the locked test
loader, and asserts finite dual-head metrics and complete manifests.

- [ ] **Step 2: Run focused integration and full regression**

Run: `source /home/lihe/.config/pinn/server.env && "$PINN_ENV/bin/python" -m pytest tests/test_corrected_pipeline_integration.py tests/test_corrected_experiment_matrix.py -q`

Run: `source /home/lihe/.config/pinn/server.env && "$PINN_ENV/bin/python" -m pytest tests/ -q`

Run: `source /home/lihe/.config/pinn/server.env && "$PINN_ENV/bin/python" -W error -m compileall -q src scripts`

Expected: all tests pass, one existing skip remains, compilation emits no warning.

- [ ] **Step 3: Run real-data preflight**

Build the active dataset and freeze split manifests for seeds 17, 42, and 73
without training. Require exactly 2,483 accepted samples, seed-42 counts
1,735/374/374, 300-step finite STF targets,
finite `[0,1]` retained fractions, invariant full moment per event, no key
overlap, and complete config/dataset/split hashes.

- [ ] **Step 4: Run finite single-GPU smoke**

From a clean commit, train Model A for one epoch on a bounded real-data subset,
strictly reload `best_model.pth`, and evaluate a bounded locked test subset.
Require finite loss/components/gradients/scalar Mw/STF, a 300-step STF output,
loadable full-state checkpoint, `git_dirty=false`, and an idle GPU after exit.

- [ ] **Step 5: Final implementation commit and handoff**

If integration-only fixes were necessary, commit them atomically:

```bash
git add tests/test_corrected_pipeline_integration.py tests/test_corrected_experiment_matrix.py
git commit -m "test: verify station-random pipeline end to end"
```

Do not start the formal three-seed/two-model campaign. Prepare the clean commit
SHA and smoke evidence; formal comparison remains gated on the exact Cross1
model artifact and a detached run worktree. CRUST1.0 remains a deferred travel-
time provider until its layer data and interpolation/ray semantics are supplied
and reviewed.
