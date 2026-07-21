# STF 时间轴、几何元数据与事件级评估整改 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立一个时间定义统一、STF 积分守恒、训练与推断几何语义一致、事件级泛化可复现的第二版训练与评估管线，并用修正后的管线重新生成论文核心结果。

**Architecture:** 保留现有 TCN–Transformer 主干，但把几何、波形预处理、STF 处理、数据划分和指标聚合从 `data_loader.py`、`evaluate.py` 的重复逻辑中抽离为单一职责模块。第二版管线采用发震时刻为统一时间原点：参考 STF 保持源时间坐标，不做台站相关平移；P/S 绝对传播时延只在可微正演中施加；训练、标准评估、LOEO 和外部事件统一调用同一几何与样本构建函数。

**Tech Stack:** Python 3.12、NumPy、PyTorch、PyYAML、scikit-learn、pytest、CSV/JSON 结果清单。

---

## 固定设计决策

1. **不原地覆盖旧管线。** 新实现使用 `pipeline_version: 2` 和 `configs/config_v2.yaml`；旧权重与旧结果仅作为 legacy diagnostics。
2. **统一使用发震时刻坐标。** 波形时间轴和 STF 时间轴均以 event origin 为零点；STF 不再按台站距离平移。
3. **传播时延只出现一次。** 正演算子使用震源距 \(R=\sqrt{\Delta^2+h^2}\) 施加绝对 P/S 时延；第二版配置不允许 `skip_travel_delays`。
4. **STF 标签按事件唯一。** 同一事件所有台站共享完全相同的 STF、完整积分矩和 STF-derived Mw。
5. **STF 积分必须守恒。** 原始 SCARDEC STF 插值到规则网格后按离散积分重新标定；若固定窗口保留的原始矩比例低于 99.5%，数据审计直接失败。
6. **波形显式重采样到 1 Hz。** 不再把“200 个原始样点”解释成“200 s”。
7. **基线校正只做减法，不清零物理信号。** 优先使用发震前 `[-60, 0)` s；无发震前数据时使用真正 P 到时之前的早期段作为显式 fallback，并记录来源。
8. **滤波参数消除术语歧义。** 使用 `cutoff_hz: 0.1`、`num_taps: 7`；论文描述为“7-tap, sixth-order FIR”。
9. **距离字段不再复用。** 数据样本分别保存 `epicentral_distance_m` 与 `source_distance_m`；正演、传播时延和 PGD 标度律使用 `source_distance_m`。
10. **网络元数据由唯一函数生成。** 第二版主配置使用 `[log R, sin(theta), cos(theta), sin(azimuth), cos(azimuth)]`；训练、LOEO 和外部事件不得自行拼接。
11. **主震级指标以目录 Mw 为参考。** 同时报告完整 SCARDEC STF-derived Mw，二者不得混在同一 `mw_true` 列中。
12. **主泛化指标按事件等权。** 每个事件先取台站预测中位数，再对事件计算 MAE；station-record MAE 仅作为补充。
13. **现有八事件属于开发期外部验证。** 不再称为 fully blind test；最终跨事件证据以修正后的 LOEO 和新盲测事件为准。
14. **2 cm 阈值先作为 legacy-matched cohort 保留。** 修正结果稳定后再比较 cm0/cm1/cm2 或 SNR 筛选，避免一次同时改变标签、时间轴和样本分布。
15. **固定幅值因子改为单一 `amplitude_gain`。** 主配置为 1.0；1.44 只作为明确标注的敏感性实验。
16. **删除死损失 `L_nonneg`。** 非负性继续由模型输出参数化保证。
17. **当前工作区有未提交论文改动。** 所有实现必须在独立 worktree 中进行，不得覆盖当前 `main` 工作树。

## 合并单元与依赖顺序

- **PR 1 — Data Contract:** Tasks 1–7。产出可审计、事件不变的样本与标签。
- **PR 2 — Physics and Training:** Tasks 8–11。产出时间一致、积分一致的训练路径。
- **PR 3 — Evaluation and Experiments:** Tasks 12–19。产出事件级指标、外部验证、受控重训、鲁棒性结果与来源链。
- **PR 4 — Manuscript Regeneration:** Task 20。只消费前三个 PR 冻结后的结果，不再手工抄数字。
- **Release Verification:** Task 21。独立执行全测试、数据不变量、结果语义和可复现性检查。

PR 2 依赖 PR 1；PR 3 依赖 PR 1 和 PR 2；PR 4 依赖前三个 PR 的冻结产物；Task 21 是发布前硬门槛。

---

### Task 1: 建立隔离 worktree 并冻结 legacy 基线

**Files:**
- Create: `docs/audits/2026-07-20-legacy-pipeline-baseline.md`
- Create: `configs/legacy/config_legacy_2026-07-20.yaml`
- Do not modify: 当前工作树中已修改的 `paper/srl/**`

- [ ] **Step 1: 从当前 HEAD 创建独立分支和 worktree**

```bash
git worktree add ../demo-corrected-pipeline -b fix/corrected-stf-geometry-eval HEAD
mkdir -p ../demo-corrected-pipeline/docs/superpowers/plans
cp docs/superpowers/plans/2026-07-20-corrected-stf-geometry-evaluation-pipeline.md ../demo-corrected-pipeline/docs/superpowers/plans/
cd ../demo-corrected-pipeline
```

Expected: 新 worktree 位于 `../demo-corrected-pipeline`；本计划文件已复制到新 worktree；当前含未提交论文改动的工作树保持不变。

- [ ] **Step 2: 建立可执行 Python 环境**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import numpy, torch, yaml, sklearn; print(torch.__version__)"
```

Expected: 命令退出码为 0，能够导入 NumPy、PyTorch、PyYAML 和 scikit-learn。

- [ ] **Step 3: 记录 legacy 代码与结果来源**

`docs/audits/2026-07-20-legacy-pipeline-baseline.md` 必须包含：

```markdown
# Legacy Pipeline Baseline

- Git commit: `e9955220a407126956a4e77c60bd064fb3afeb4e`
- Headline checkpoint: `outputs_experiments/e1_4/models/lm010_noint/best_model.pth`
- Headline config: `outputs_experiments/e1_4/models/lm010_noint/config.yaml`
- Headline station-split CSV: `outputs/results/test_set_predictions_far_only.csv`
- Legacy test MAE label: within-event held-out-station MAE against station-cropped STF Mw
- Legacy external-event role: development-time validation, not blind test
- Known invalid assumptions:
  1. STF shifted by `distance / 4500`.
  2. Shifted STF cropped to 200 samples.
  3. External-event azimuth passed as takeoff angle and geographic azimuth set to zero.
  4. Standard split contains event-group overlap.
```

- [ ] **Step 4: 冻结旧配置副本**

```bash
cp outputs_experiments/e1_4/models/lm010_noint/config.yaml configs/legacy/config_legacy_2026-07-20.yaml
```

- [ ] **Step 5: 提交隔离与基线文档**

```bash
git add \
  docs/superpowers/plans/2026-07-20-corrected-stf-geometry-evaluation-pipeline.md \
  docs/audits/2026-07-20-legacy-pipeline-baseline.md \
  configs/legacy/config_legacy_2026-07-20.yaml
git commit -m "docs: freeze legacy pipeline and correction plan"
```

---

### Task 2: 新增第二版配置合同与启动时校验

**Files:**
- Create: `configs/config_v2.yaml`
- Create: `src/utils/config_v2.py`
- Create: `tests/test_config_v2.py`
- Modify: `src/training/train.py`
- Modify: `src/evaluation/evaluate.py`
- Modify: `src/evaluation/evaluate_unseen.py`

- [ ] **Step 1: 写入失败测试，要求第二版配置拒绝旧语义键**

```python
# tests/test_config_v2.py
import pytest

from src.utils.config_v2 import validate_config_v2


def _minimal_v2() -> dict:
    return {
        "pipeline_version": 2,
        "dataset": {
            "sample_rate_hz": 1.0,
            "waveform": {"start_sec": 0.0, "duration_sec": 200.0, "min_valid_fraction": 0.99},
            "baseline": {
                "method": "median",
                "pre_event_start_sec": -60.0,
                "pre_event_end_sec": 0.0,
                "fallback": "pre_p",
                "fallback_max_sec": 30.0,
                "min_samples": 10,
            },
            "filter": {"type": "lowpass", "cutoff_hz": 0.1, "num_taps": 7},
            "stf": {
                "start_sec": 0.0,
                "duration_sec": 200.0,
                "min_retained_moment_fraction": 0.995,
                "preserve_integral": True,
                "m_ref": 1.0e18,
            },
        },
        "physics": {
            "rho": 3400.0,
            "alpha": 7900.0,
            "beta": 4533.0,
            "distance_mode": "hypocentral",
            "delay_mode": "absolute",
            "amplitude_gain": 1.0,
        },
        "training": {
            "rate_representation": "log1p",
            "stf_rate_loss": {
                "lambda_MSE": 1.0,
                "lambda_synth": 0.5,
                "lambda_mag": 1.0,
                "lambda_shape": 0.1,
            },
        },
        "evaluation": {"primary_reference": "catalog", "aggregation": "event_median"},
    }


def test_valid_v2_config_passes():
    validate_config_v2(_minimal_v2())


@pytest.mark.parametrize(
    "section,key",
    [
        ("dataset", "p_velocity_mps"),
        ("physics", "attenuation"),
        ("physics", "geometrical_spreading_factor"),
        ("physics", "free_surface_factor"),
    ],
)
def test_v2_rejects_legacy_top_level_keys(section: str, key: str):
    cfg = _minimal_v2()
    cfg[section][key] = 1.0
    with pytest.raises(ValueError, match=key):
        validate_config_v2(cfg)


def test_v2_rejects_skip_delays_and_nonneg_loss():
    cfg = _minimal_v2()
    cfg["training"]["stf_rate_loss"]["skip_travel_delays"] = True
    cfg["training"]["stf_rate_loss"]["lambda_nonneg"] = 0.5
    with pytest.raises(ValueError, match="skip_travel_delays|lambda_nonneg"):
        validate_config_v2(cfg)


def test_v2_requires_explicit_log1p_representation():
    cfg = _minimal_v2()
    cfg["training"]["rate_representation"] = "auto"
    with pytest.raises(ValueError, match="rate_representation"):
        validate_config_v2(cfg)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/bin/python -m pytest tests/test_config_v2.py -v
```

Expected: FAIL，因为 `src.utils.config_v2` 尚不存在。

- [ ] **Step 3: 实现严格配置校验**

```python
# src/utils/config_v2.py
from __future__ import annotations

from typing import Any


_FORBIDDEN_PATHS = (
    ("dataset", "p_velocity_mps"),
    ("physics", "attenuation"),
    ("physics", "geometrical_spreading_factor"),
    ("physics", "free_surface_factor"),
    ("training", "stf_rate_loss", "skip_travel_delays"),
    ("training", "stf_rate_loss", "lambda_nonneg"),
)


def _lookup(config: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False, None
        node = node[key]
    return True, node


def validate_config_v2(config: dict[str, Any]) -> None:
    if int(config.get("pipeline_version", 0)) != 2:
        raise ValueError("pipeline_version 必须为 2")
    for path in _FORBIDDEN_PATHS:
        exists, _ = _lookup(config, path)
        if exists:
            raise ValueError(f"第二版配置禁止旧语义键: {'.'.join(path)}")

    ds = config["dataset"]
    sample_rate_hz = float(ds["sample_rate_hz"])
    if sample_rate_hz != 1.0:
        raise ValueError("第二版主管线要求 dataset.sample_rate_hz == 1.0")
    if int(ds["filter"]["num_taps"]) % 2 != 1:
        raise ValueError("dataset.filter.num_taps 必须为奇数")
    if float(ds["stf"]["min_retained_moment_fraction"]) < 0.995:
        raise ValueError("STF 矩保留阈值不得低于 0.995")
    if float(ds["stf"]["m_ref"]) <= 0.0:
        raise ValueError("dataset.stf.m_ref 必须为正")

    training = config["training"]
    if str(training.get("rate_representation", "")).lower() != "log1p":
        raise ValueError("第二版主管线要求 training.rate_representation=log1p")

    phys = config["physics"]
    if phys["distance_mode"] != "hypocentral":
        raise ValueError("第二版主配置要求 physics.distance_mode=hypocentral")
    if phys["delay_mode"] != "absolute":
        raise ValueError("第二版主配置要求 physics.delay_mode=absolute")


def stf_m_ref_from_config(config: dict[str, Any]) -> float:
    validate_config_v2(config)
    return float(config["dataset"]["stf"]["m_ref"])


def stf_output_steps_from_config(config: dict[str, Any]) -> int:
    validate_config_v2(config)
    dataset = config["dataset"]
    return int(round(float(dataset["stf"]["duration_sec"]) * float(dataset["sample_rate_hz"])))
```

- [ ] **Step 4: 新增明确的 `configs/config_v2.yaml`**

```yaml
pipeline_version: 2
project_name: PINN_Earthquake_Magnitude_Prediction_v2

dataset:
  blacklist_events:
    - N.Honshu2011
    - N.Honshu2012
    - N.Honshu2013
    - E.Fukushima2011
    - Iwate2011
  units: mm
  sample_rate_hz: 1.0
  radial_peak_min_cm: 2.0
  allow_missing_stf: false
  waveform:
    start_sec: 0.0
    duration_sec: 200.0
    min_valid_fraction: 0.99
    max_interpolation_gap_sec: 2.5
  baseline:
    method: median
    pre_event_start_sec: -60.0
    pre_event_end_sec: 0.0
    fallback: pre_p
    fallback_max_sec: 30.0
    min_samples: 10
  filter:
    type: lowpass
    cutoff_hz: 0.1
    num_taps: 7
    window: hamming
  stf:
    path: /Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/PINN_Mag/dataset/STF_SCARDEC
    start_sec: 0.0
    duration_sec: 200.0
    min_retained_moment_fraction: 0.995
    preserve_integral: true
    m_ref: 1.0e18
    magnitude_target: stf_native

geometry:
  network_distance: hypocentral

physics:
  rho: 3400.0
  alpha: 7900.0
  beta: 4533.0
  distance_mode: hypocentral
  delay_mode: absolute
  amplitude_gain: 1.0

model:
  hidden_dim: 128
  num_layers: 3
  num_tcn_blocks: 6
  transformer_num_layers: 3
  dropout: 0.2
  use_meta: true

paths:
  data_path: /Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/PINN_Mag/dataset/gnss_events_matched.gcmt.npz
  output_dir: ./outputs_v2
  logs_dir: ./outputs_v2/logs
  models_dir: ./outputs_v2/models
  results_dir: ./outputs_v2/results

training:
  loss_name: stf_rate
  rate_representation: log1p
  random_seed: 42
  batch_size: 64
  epochs: 200
  learning_rate: 0.0001
  weight_decay: 1.0e-05
  grad_clip_norm: 1.0
  split_protocol: grouped_event
  validation_event_fraction: 0.15
  test_event_fraction: 0.0
  event_balanced_sampling: true
  early_stop_metric: event_mae_catalog
  early_stop_patience: 50
  early_stop_min_delta: 0.0001
  stf_rate_loss:
    lambda_MSE: 1.0
    lambda_synth: 0.5
    lambda_mag: 1.0
    lambda_shape: 0.1
    include_intermediate_field: false
    radiation_pattern_mode: full

evaluation:
  primary_reference: catalog
  secondary_reference: stf_native
  aggregation: event_median
  station_thresholds_cm: [0.0, 1.0, 2.0]
  external_role: validation
```

- [ ] **Step 5: 在所有第二版入口加载配置后立即校验**

在 `train()`、`evaluate()` 和 `evaluate_unseen_events()` 中，在构建数据集或模型之前调用：

```python
if int(config.get("pipeline_version", 1)) == 2:
    from src.utils.config_v2 import validate_config_v2
    validate_config_v2(config)
```

- [ ] **Step 6: 运行配置测试**

```bash
.venv/bin/python -m pytest tests/test_config_v2.py -v
```

Expected: PASS。

- [ ] **Step 7: 提交**

```bash
git add configs/config_v2.yaml src/utils/config_v2.py tests/test_config_v2.py src/training/train.py src/evaluation/evaluate.py src/evaluation/evaluate_unseen.py
git commit -m "feat: add strict v2 pipeline config"
```

---

### Task 3: 建立唯一的几何与网络元数据实现

**Files:**
- Create: `src/data/geometry.py`
- Create: `src/data/metadata.py`
- Create: `tests/test_geometry.py`
- Modify later consumers only after tests pass

- [ ] **Step 1: 写入几何失败测试**

```python
# tests/test_geometry.py
import math

import torch

from src.data.geometry import compute_source_station_geometry
from src.data.metadata import build_metadata_tensor


def test_collocated_station_uses_depth_as_source_distance():
    g = compute_source_station_geometry(35.0, 140.0, 20.0, 35.0, 140.0)
    assert g.epicentral_distance_m == 0.0
    assert g.source_distance_m == 20_000.0
    assert g.takeoff_angle_deg == 0.0


def test_due_north_station_has_northward_azimuth():
    g = compute_source_station_geometry(35.0, 140.0, 10.0, 36.0, 140.0)
    assert 110_000.0 < g.epicentral_distance_m < 112_000.0
    assert abs(g.azimuth_deg - 0.0) < 1.0e-6
    assert abs(g.back_azimuth_deg - 180.0) < 1.0e-6
    expected = math.degrees(math.atan2(g.epicentral_distance_m, 10_000.0))
    assert abs(g.takeoff_angle_deg - expected) < 1.0e-6


def test_metadata_order_is_log_r_theta_azimuth():
    source_distance = torch.tensor([100_000.0])
    theta = torch.tensor([30.0])
    azimuth = torch.tensor([90.0])
    meta = build_metadata_tensor(source_distance, theta, azimuth)
    assert meta.shape == (1, 5)
    assert torch.allclose(meta[0, 0], torch.log(source_distance)[0])
    assert torch.allclose(meta[0, 1], torch.tensor(0.5), atol=1.0e-6)
    assert torch.allclose(meta[0, 2], torch.tensor(math.sqrt(3.0) / 2.0), atol=1.0e-6)
    assert torch.allclose(meta[0, 3], torch.tensor(1.0), atol=1.0e-6)
    assert torch.allclose(meta[0, 4], torch.tensor(0.0), atol=1.0e-6)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/bin/python -m pytest tests/test_geometry.py -v
```

Expected: FAIL，因为两个模块尚不存在。

- [ ] **Step 3: 实现明确命名的几何数据类**

```python
# src/data/geometry.py
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SourceStationGeometry:
    epicentral_distance_m: float
    source_distance_m: float
    azimuth_deg: float
    back_azimuth_deg: float
    takeoff_angle_deg: float


def compute_source_station_geometry(
    event_lat: float,
    event_lon: float,
    depth_km: float,
    station_lat: float,
    station_lon: float,
) -> SourceStationGeometry:
    if not all(math.isfinite(v) for v in (event_lat, event_lon, depth_km, station_lat, station_lon)):
        raise ValueError("事件、深度和台站坐标必须为有限值")
    if depth_km < 0.0:
        raise ValueError("depth_km 不得为负")

    earth_radius_m = 6_371_000.0
    phi1 = math.radians(event_lat)
    phi2 = math.radians(station_lat)
    dphi = math.radians(station_lat - event_lat)
    dlambda = math.radians(station_lon - event_lon)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    epicentral_distance_m = earth_radius_m * c

    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    azimuth_deg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    back_azimuth_deg = (azimuth_deg + 180.0) % 360.0

    depth_m = depth_km * 1000.0
    source_distance_m = math.hypot(epicentral_distance_m, depth_m)
    takeoff_angle_deg = math.degrees(math.atan2(epicentral_distance_m, max(depth_m, 1.0e-12)))
    return SourceStationGeometry(
        epicentral_distance_m=epicentral_distance_m,
        source_distance_m=source_distance_m,
        azimuth_deg=azimuth_deg,
        back_azimuth_deg=back_azimuth_deg,
        takeoff_angle_deg=takeoff_angle_deg,
    )
```

- [ ] **Step 4: 实现唯一的 metadata builder**

```python
# src/data/metadata.py
from __future__ import annotations

import torch


def build_metadata_tensor(
    source_distance_m: torch.Tensor,
    takeoff_angle_deg: torch.Tensor,
    azimuth_deg: torch.Tensor,
) -> torch.Tensor:
    r = source_distance_m.reshape(-1).clamp_min(1.0)
    theta = torch.deg2rad(takeoff_angle_deg.reshape(-1))
    azimuth = torch.deg2rad(azimuth_deg.reshape(-1))
    return torch.stack(
        [torch.log(r), torch.sin(theta), torch.cos(theta), torch.sin(azimuth), torch.cos(azimuth)],
        dim=1,
    )
```

- [ ] **Step 5: 运行测试**

```bash
.venv/bin/python -m pytest tests/test_geometry.py -v
```

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add src/data/geometry.py src/data/metadata.py tests/test_geometry.py
git commit -m "feat: centralize source station geometry"
```

---

### Task 4: 重建波形时间轴、基线、滤波和有效样点掩码

**Files:**
- Create: `src/data/waveform.py`
- Create: `tests/test_waveform_preprocessing.py`
- Do not modify: `src/data/data_loader.py` 的 legacy 预处理

- [ ] **Step 1: 写入失败测试，覆盖 1 Hz 重采样、基线减法和禁止整段清零**

```python
# tests/test_waveform_preprocessing.py
import numpy as np

from src.data.waveform import WaveformConfig, preprocess_waveform


def _config() -> WaveformConfig:
    return WaveformConfig(
        sample_rate_hz=1.0,
        start_sec=0.0,
        duration_sec=20.0,
        min_valid_fraction=0.9,
        max_interpolation_gap_sec=2.5,
        baseline_method="mean",
        pre_event_start_sec=-5.0,
        pre_event_end_sec=0.0,
        baseline_fallback="pre_p",
        baseline_fallback_max_sec=10.0,
        baseline_min_samples=3,
        filter_type="none",
        cutoff_hz=0.1,
        num_taps=7,
        filter_window="hamming",
    )


def test_irregular_waveform_is_resampled_to_one_hz():
    t = np.arange(-5.0, 20.0, 0.5)
    x = 0.01 * t
    out = preprocess_waveform(t, x, units="m", p_arrival_sec=8.0, config=_config())
    assert out.values_m.shape == (20,)
    assert out.valid_mask.shape == (20,)
    assert out.dt_sec == 1.0
    assert np.allclose(out.time_sec, np.arange(20.0))


def test_pre_event_baseline_is_subtracted_from_all_samples():
    t = np.arange(-5.0, 20.0)
    x = np.where(t < 0.0, 2.0, 2.0 + 0.1 * t)
    out = preprocess_waveform(t, x, units="m", p_arrival_sec=8.0, config=_config())
    assert out.baseline_source == "pre_event"
    assert abs(out.baseline_m - 2.0) < 1.0e-6
    assert abs(out.values_m[0] - 0.0) < 1.0e-6
    assert out.values_m[5] > out.values_m[1]


def test_pre_p_fallback_does_not_flatten_pre_p_segment():
    t = np.arange(0.0, 20.0)
    x = 1.0 + 0.01 * t
    out = preprocess_waveform(t, x, units="m", p_arrival_sec=8.0, config=_config())
    assert out.baseline_source == "pre_p"
    assert np.std(out.values_m[:8]) > 0.0
    assert not np.allclose(out.values_m[:8], 0.0)


def test_short_record_below_valid_fraction_is_rejected():
    t = np.arange(0.0, 10.0)
    x = np.ones_like(t)
    try:
        preprocess_waveform(t, x, units="m", p_arrival_sec=8.0, config=_config())
    except ValueError as exc:
        assert "valid fraction" in str(exc)
    else:
        raise AssertionError("不足 90% 有效覆盖率的记录必须被拒绝")
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/bin/python -m pytest tests/test_waveform_preprocessing.py -v
```

Expected: FAIL。

- [ ] **Step 3: 实现纯函数式波形处理模块**

`src/data/waveform.py` 必须实现以下公开合同：

```python
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class WaveformConfig:
    sample_rate_hz: float
    start_sec: float
    duration_sec: float
    min_valid_fraction: float
    max_interpolation_gap_sec: float
    baseline_method: str
    pre_event_start_sec: float
    pre_event_end_sec: float
    baseline_fallback: str
    baseline_fallback_max_sec: float
    baseline_min_samples: int
    filter_type: str
    cutoff_hz: float
    num_taps: int
    filter_window: str


@dataclass(frozen=True)
class ProcessedWaveform:
    time_sec: np.ndarray
    values_m: np.ndarray
    valid_mask: np.ndarray
    dt_sec: float
    raw_dt_sec: float
    baseline_m: float
    baseline_source: str
    valid_fraction: float


def _convert_to_metres(values: np.ndarray, units: str) -> np.ndarray:
    key = units.lower()
    if key == "m":
        factor = 1.0
    elif key == "cm":
        factor = 1.0e-2
    elif key == "mm":
        factor = 1.0e-3
    else:
        raise ValueError(f"第二版管线不接受自动单位推断: {units}")
    return values.astype(np.float64, copy=False) * factor


def _sort_and_average_duplicates(
    time_sec: np.ndarray,
    values_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(time_sec) & np.isfinite(values_m)
    t = np.asarray(time_sec, dtype=np.float64)[finite]
    x = np.asarray(values_m, dtype=np.float64)[finite]
    if t.size < 2:
        raise ValueError("有效波形样点少于 2")
    order = np.argsort(t, kind="mergesort")
    t = t[order]
    x = x[order]
    unique_t, inverse = np.unique(t, return_inverse=True)
    sums = np.bincount(inverse, weights=x)
    counts = np.bincount(inverse)
    unique_x = sums / counts
    if unique_t.size < 2:
        raise ValueError("去重后有效时间戳少于 2")
    return unique_t, unique_x


def _estimate_baseline(
    time_sec: np.ndarray,
    values_m: np.ndarray,
    *,
    p_arrival_sec: float,
    config: WaveformConfig,
) -> tuple[float, str]:
    pre_event = (
        (time_sec >= config.pre_event_start_sec)
        & (time_sec < config.pre_event_end_sec)
    )
    if int(pre_event.sum()) >= config.baseline_min_samples:
        selected = values_m[pre_event]
        source = "pre_event"
    elif config.baseline_fallback == "pre_p":
        fallback_end = min(float(p_arrival_sec), config.baseline_fallback_max_sec)
        pre_p = (time_sec >= 0.0) & (time_sec < fallback_end)
        if int(pre_p.sum()) < config.baseline_min_samples:
            raise ValueError("insufficient baseline: pre-event and pre-P samples are insufficient")
        selected = values_m[pre_p]
        source = "pre_p"
    else:
        raise ValueError(f"不支持的 baseline fallback: {config.baseline_fallback}")

    if config.baseline_method == "median":
        baseline = float(np.median(selected))
    elif config.baseline_method == "mean":
        baseline = float(np.mean(selected))
    else:
        raise ValueError(f"不支持的 baseline method: {config.baseline_method}")
    if not math.isfinite(baseline):
        raise ValueError("baseline 不是有限值")
    return baseline, source


def _fir_lowpass(values: np.ndarray, config: WaveformConfig) -> np.ndarray:
    if config.filter_type == "none":
        return values
    if config.filter_type != "lowpass":
        raise ValueError(f"第二版主管线仅支持 none/lowpass: {config.filter_type}")
    taps = int(config.num_taps)
    if taps < 3 or taps % 2 == 0:
        raise ValueError("num_taps 必须是至少为 3 的奇数")
    nyquist_hz = 0.5 * config.sample_rate_hz
    if not 0.0 < config.cutoff_hz < nyquist_hz:
        raise ValueError("cutoff_hz 必须位于 (0, Nyquist) 内")

    n = np.arange(taps, dtype=np.float64)
    midpoint = 0.5 * (taps - 1)
    normalized_cutoff = config.cutoff_hz / config.sample_rate_hz
    kernel = 2.0 * normalized_cutoff * np.sinc(
        2.0 * normalized_cutoff * (n - midpoint)
    )
    if config.filter_window == "hamming":
        kernel *= np.hamming(taps)
    elif config.filter_window in {"hann", "hanning"}:
        kernel *= np.hanning(taps)
    else:
        raise ValueError(f"不支持的 FIR window: {config.filter_window}")
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def preprocess_waveform(
    time_sec: np.ndarray,
    values: np.ndarray,
    *,
    units: str,
    p_arrival_sec: float,
    config: WaveformConfig,
) -> ProcessedWaveform:
    if config.sample_rate_hz <= 0.0 or config.duration_sec <= 0.0:
        raise ValueError("sample_rate_hz 和 duration_sec 必须为正")
    values_m = _convert_to_metres(np.asarray(values), units)
    t, x = _sort_and_average_duplicates(np.asarray(time_sec), values_m)
    positive_diffs = np.diff(t)
    positive_diffs = positive_diffs[positive_diffs > 0.0]
    raw_dt_sec = float(np.median(positive_diffs))

    baseline_m, baseline_source = _estimate_baseline(
        t,
        x,
        p_arrival_sec=p_arrival_sec,
        config=config,
    )
    x = x - baseline_m

    dt_sec = 1.0 / config.sample_rate_hz
    sample_count = int(round(config.duration_sec * config.sample_rate_hz))
    grid = config.start_sec + np.arange(sample_count, dtype=np.float64) * dt_sec
    interpolated = np.interp(grid, t, x)

    supported = (grid >= t[0]) & (grid <= t[-1])
    left = np.searchsorted(t, grid, side="right") - 1
    left_safe = np.clip(left, 0, t.size - 1)
    exact = supported & np.isclose(grid, t[left_safe], rtol=0.0, atol=1.0e-8)
    right = left + 1
    interior = supported & (left >= 0) & (right < t.size)
    gap_ok = np.zeros_like(supported, dtype=bool)
    gap_ok[exact] = True
    interior_nonexact = interior & ~exact
    gap_ok[interior_nonexact] = (
        t[right[interior_nonexact]] - t[left[interior_nonexact]]
        <= config.max_interpolation_gap_sec
    )
    valid_mask = supported & gap_ok

    network_values = np.where(valid_mask, interpolated, 0.0)
    filtered = _fir_lowpass(network_values, config)
    filtered = np.where(valid_mask, filtered, 0.0)
    valid_fraction = float(np.mean(valid_mask))
    if valid_fraction < config.min_valid_fraction:
        raise ValueError(
            f"valid fraction {valid_fraction:.4f} is below "
            f"{config.min_valid_fraction:.4f}"
        )

    return ProcessedWaveform(
        time_sec=grid.astype(np.float32),
        values_m=filtered.astype(np.float32),
        valid_mask=valid_mask.astype(bool),
        dt_sec=float(dt_sec),
        raw_dt_sec=raw_dt_sec,
        baseline_m=baseline_m,
        baseline_source=baseline_source,
        valid_fraction=valid_fraction,
    )
```

实现顺序必须为：

1. 删除非有限值并按时间排序；同一时间戳取均值。
2. 若记录提供绝对 `origin`，调用方先传入 `time_rel = time - origin`，而不是只做 `time >= origin`。
3. 单位显式转为米；`mm -> /1000`，`cm -> /100`，`m -> unchanged`。
4. 基线优先取 `[-60, 0)`；样点不足时取 `[0, min(tP, 30)]`；两者均不足时拒绝记录。
5. 仅从整个原始序列减去一个基线常数，不得写入 `values[:ip] = baseline`。
6. 建立 `np.arange(start, start + duration, 1/sample_rate)` 规则网格。
7. 只在原始数据支持范围内插值；跨越大于 `max_interpolation_gap_sec` 的内部缺口时将对应 mask 置为 False。
8. 对插值后的数值执行 7-tap、0.1 Hz Hamming FIR；mask 不随卷积扩张。
9. mask=False 的网络输入值置零；若有效覆盖率低于 0.99，拒绝记录。

- [ ] **Step 4: 运行波形测试**

```bash
.venv/bin/python -m pytest tests/test_waveform_preprocessing.py -v
```

Expected: PASS。

- [ ] **Step 5: 明确 legacy 测试与第二版测试的边界**

保留 `tests/test_data_loader.py::test_p_baseline_correction_zeroes_pre_p`，因为它记录的是旧管线的实际数值行为，供 legacy 结果复现使用。禁止在第二版入口导入或调用 `_apply_p_baseline()`；第二版“不清零前段”的要求只由 `tests/test_waveform_preprocessing.py::test_pre_p_fallback_does_not_flatten_pre_p_segment` 约束。

- [ ] **Step 6: 提交**

```bash
git add src/data/waveform.py tests/test_waveform_preprocessing.py
git commit -m "feat: add canonical v2 waveform preprocessing"
```

---

### Task 5: 建立源时间 STF、积分守恒和事件唯一标签

**Files:**
- Create: `src/data/stf.py`
- Create: `tests/test_stf_targets.py`
- Do not modify: `src/data/data_loader.py` 的 legacy STF shift/crop 路径

- [ ] **Step 1: 写入失败测试**

```python
# tests/test_stf_targets.py
import numpy as np
import pytest

from src.data.stf import STFWindowTooShort, resample_source_stf


def test_source_stf_preserves_discrete_moment():
    t = np.arange(0.0, 100.5, 0.5)
    q = np.maximum(0.0, 1.0 - np.abs(t - 40.0) / 30.0) * 1.0e18
    out = resample_source_stf(
        t,
        q,
        start_sec=0.0,
        duration_sec=120.0,
        sample_rate_hz=1.0,
        min_retained_moment_fraction=0.995,
        preserve_integral=True,
    )
    discrete_m0 = float(np.sum(out.rate_nm_per_s) * out.dt_sec)
    assert abs(discrete_m0 - out.native_moment_nm) / out.native_moment_nm < 1.0e-10
    assert out.retained_moment_fraction >= 0.995


def test_same_event_stf_has_no_station_shift_parameter():
    t = np.arange(0.0, 50.0)
    q = np.ones_like(t) * 1.0e18
    a = resample_source_stf(t, q, start_sec=0.0, duration_sec=60.0, sample_rate_hz=1.0,
                            min_retained_moment_fraction=0.995, preserve_integral=True)
    b = resample_source_stf(t, q, start_sec=0.0, duration_sec=60.0, sample_rate_hz=1.0,
                            min_retained_moment_fraction=0.995, preserve_integral=True)
    assert np.array_equal(a.rate_nm_per_s, b.rate_nm_per_s)
    assert a.mw_native == b.mw_native


def test_window_that_loses_moment_fails():
    t = np.arange(0.0, 300.0)
    q = np.ones_like(t) * 1.0e18
    with pytest.raises(STFWindowTooShort):
        resample_source_stf(t, q, start_sec=0.0, duration_sec=100.0, sample_rate_hz=1.0,
                            min_retained_moment_fraction=0.995, preserve_integral=True)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/bin/python -m pytest tests/test_stf_targets.py -v
```

Expected: FAIL。

- [ ] **Step 3: 实现完整的源时间 STF 模块**

```python
# src/data/stf.py
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class STFWindowTooShort(ValueError):
    """固定源时间窗口未达到规定的矩保留比例。"""


@dataclass(frozen=True)
class ProcessedSTF:
    time_sec: np.ndarray
    rate_nm_per_s: np.ndarray
    dt_sec: float
    native_moment_nm: float
    grid_moment_before_rescale_nm: float
    retained_moment_fraction: float
    mw_native: float


def moment_to_mw(moment_nm: float) -> float:
    if not math.isfinite(moment_nm) or moment_nm <= 0.0:
        raise ValueError("moment_nm 必须为正且有限")
    return (2.0 / 3.0) * (math.log10(moment_nm) - 9.1)


def _sort_and_average_duplicates(
    time_sec: np.ndarray,
    rate_nm_per_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(time_sec) & np.isfinite(rate_nm_per_s)
    causal = finite & (time_sec >= 0.0)
    t = np.asarray(time_sec, dtype=np.float64)[causal]
    q = np.maximum(np.asarray(rate_nm_per_s, dtype=np.float64)[causal], 0.0)
    if t.size < 2:
        raise ValueError("有效且因果的 STF 样点少于 2")
    order = np.argsort(t, kind="mergesort")
    t = t[order]
    q = q[order]
    unique_t, inverse = np.unique(t, return_inverse=True)
    sums = np.bincount(inverse, weights=q)
    counts = np.bincount(inverse)
    unique_q = sums / counts
    if unique_t.size < 2:
        raise ValueError("STF 去重后时间戳少于 2")
    return unique_t, unique_q


def _integrate_interval(
    time_sec: np.ndarray,
    rate_nm_per_s: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> float:
    left = max(float(start_sec), float(time_sec[0]))
    right = min(float(end_sec), float(time_sec[-1]))
    if right <= left:
        return 0.0
    interior = (time_sec > left) & (time_sec < right)
    t_eval = np.concatenate(([left], time_sec[interior], [right]))
    q_eval = np.interp(t_eval, time_sec, rate_nm_per_s)
    return float(np.trapz(q_eval, t_eval))


def resample_source_stf(
    time_sec: np.ndarray,
    rate_nm_per_s: np.ndarray,
    *,
    start_sec: float,
    duration_sec: float,
    sample_rate_hz: float,
    min_retained_moment_fraction: float,
    preserve_integral: bool,
) -> ProcessedSTF:
    if sample_rate_hz <= 0.0 or duration_sec <= 0.0:
        raise ValueError("sample_rate_hz 和 duration_sec 必须为正")
    if not 0.0 < min_retained_moment_fraction <= 1.0:
        raise ValueError("min_retained_moment_fraction 必须位于 (0, 1]")

    t, q = _sort_and_average_duplicates(
        np.asarray(time_sec),
        np.asarray(rate_nm_per_s),
    )
    native_moment_nm = float(np.trapz(q, t))
    if not math.isfinite(native_moment_nm) or native_moment_nm <= 0.0:
        raise ValueError("STF 原始积分矩必须为正且有限")

    end_sec = start_sec + duration_sec
    retained_moment_nm = _integrate_interval(t, q, start_sec, end_sec)
    retained_fraction = retained_moment_nm / native_moment_nm
    if retained_fraction + 1.0e-12 < min_retained_moment_fraction:
        raise STFWindowTooShort(
            f"STF window [{start_sec}, {end_sec}) retains "
            f"{retained_fraction:.6f}, below {min_retained_moment_fraction:.6f}"
        )

    dt_sec = 1.0 / sample_rate_hz
    sample_count = int(round(duration_sec * sample_rate_hz))
    target_time = start_sec + np.arange(sample_count, dtype=np.float64) * dt_sec
    target_rate = np.interp(target_time, t, q, left=0.0, right=0.0)
    grid_moment_before_rescale_nm = float(np.sum(target_rate) * dt_sec)
    if grid_moment_before_rescale_nm <= 0.0:
        raise ValueError("STF 在目标网格上的离散积分为零")

    if preserve_integral:
        target_rate = target_rate * (
            native_moment_nm / grid_moment_before_rescale_nm
        )

    return ProcessedSTF(
        time_sec=target_time,
        rate_nm_per_s=target_rate,
        dt_sec=float(dt_sec),
        native_moment_nm=native_moment_nm,
        grid_moment_before_rescale_nm=grid_moment_before_rescale_nm,
        retained_moment_fraction=float(retained_fraction),
        mw_native=moment_to_mw(native_moment_nm),
    )
```

- [ ] **Step 4: 确认 API 不接受任何台站相关参数**

运行：

```bash
.venv/bin/python - <<'PY'
import inspect
from src.data.stf import resample_source_stf

parameters = set(inspect.signature(resample_source_stf).parameters)
for forbidden in {"distance", "distance_m", "p_shift_sec", "p_velocity_mps", "station"}:
    assert forbidden not in parameters, forbidden
print(sorted(parameters))
PY
```

Expected: 输出只包含源 STF、源时间网格和积分门槛参数；断言全部通过。

- [ ] **Step 5: 运行测试**

```bash
.venv/bin/python -m pytest tests/test_stf_targets.py -v
```

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add src/data/stf.py tests/test_stf_targets.py
git commit -m "feat: preserve event invariant source time functions"
```

---

### Task 6: 新建与 legacy 隔离的第二版数据集和样本构建管线

**Files:**
- Create: `src/data/records_v2.py`
- Create: `src/data/sample_builder.py`
- Create: `src/data/dataset_v2.py`
- Create: `tests/test_corrected_pipeline_integration.py`
- Do not modify: `src/data/data_loader.py` 的 legacy 数值路径
- Do not modify: `tests/test_data_loader.py` 的 legacy 行为断言

- [ ] **Step 1: 写入两事件集成失败测试**

```python
# tests/test_corrected_pipeline_integration.py
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.dataset_v2 import CorrectedEarthquakeDataset


def test_same_event_has_one_stf_and_one_reference_mw(tmp_path: Path):
    # fixture 构造一个事件、两个不同距离台站和一个三角形 STF。
    t = np.arange(-10.0, 210.0, dtype=np.float32)
    stations = np.empty(1, dtype=object)
    stations[0] = [
        {"name": "NEAR", "lat": 35.1, "lon": 140.0, "t": t, "E": np.sin(t / 20), "N": np.zeros_like(t), "U": np.zeros_like(t), "origin": 0.0},
        {"name": "FAR", "lat": 37.0, "lon": 140.0, "t": t, "E": np.sin(t / 20), "N": np.zeros_like(t), "U": np.zeros_like(t), "origin": 0.0},
    ]
    npz_path = tmp_path / "events.npz"
    np.savez(npz_path, events=np.array(["EventA"], dtype=object), magnitude=np.array([7.0]),
             latitude=np.array([35.0]), longitude=np.array([140.0]), depth_km=np.array([20.0]), stations=stations)

    stf_dir = tmp_path / "stf"
    stf_dir.mkdir()
    q_t = np.arange(0.0, 100.0)
    q = np.maximum(0.0, 1.0 - np.abs(q_t - 40.0) / 30.0) * 1.0e18
    with (stf_dir / "eventa.stf").open("w", encoding="utf-8") as f:
        for ti, qi in zip(q_t, q):
            f.write(f"{ti} {qi}\n")

    config = make_v2_dataset_config(npz_path=npz_path, stf_dir=stf_dir)
    ds = CorrectedEarthquakeDataset(config)
    assert len(ds) == 2
    assert np.array_equal(ds.samples[0]["stf"], ds.samples[1]["stf"])
    assert ds.samples[0]["mw_stf_native"] == ds.samples[1]["mw_stf_native"]
    assert ds.samples[0]["source_distance_m"] != ds.samples[1]["source_distance_m"]
    assert ds.samples[0]["azimuth_deg"] == ds.samples[0]["phi_deg"]
```

`make_v2_dataset_config()` 必须在同一测试文件中返回完整最小配置，不依赖真实数据路径。

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/bin/python -m pytest tests/test_corrected_pipeline_integration.py -v
```

Expected: FAIL，因为 `src.data.dataset_v2.CorrectedEarthquakeDataset` 尚不存在。

- [ ] **Step 3: 合并 `enu` 和 `stations` 两套重复循环**

在 `src/data/records_v2.py` 中实现独立解析器，并先定义不可变记录类型：

```python
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


def _unwrap_object(value: Any) -> Any:
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.reshape(-1)[0]
    return value


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(_unwrap_object(value))
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _optional_float(value: Any) -> float | None:
    result = _as_float(value)
    return result if np.isfinite(result) else None


def _get_field(payload: Any, keys: list[str]) -> Any:
    payload = _unwrap_object(payload)
    if not isinstance(payload, dict):
        return None
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _iter_stations_container(container: Any) -> list[tuple[str, dict[str, Any]]]:
    container = _unwrap_object(container)
    if isinstance(container, dict):
        result: list[tuple[str, dict[str, Any]]] = []
        for name, raw_payload in container.items():
            payload = _unwrap_object(raw_payload)
            if isinstance(payload, dict):
                result.append((str(name), payload))
        return result
    if isinstance(container, np.ndarray):
        container = container.tolist()
    if isinstance(container, (list, tuple)):
        result: list[tuple[str, dict[str, Any]]] = []
        for index, raw in enumerate(container):
            station = _unwrap_object(raw)
            if not isinstance(station, dict):
                continue
            name = station.get("name", station.get("station", station.get("id", f"st_{index}")))
            result.append((str(name), station))
        return result
    return []


def _normalize_station_info(info: Any) -> dict[str, dict[str, Any]]:
    info = _unwrap_object(info)
    if isinstance(info, dict):
        return {str(name): _unwrap_object(payload) for name, payload in info.items()}
    if isinstance(info, np.ndarray):
        info = info.tolist()
    if isinstance(info, (list, tuple)):
        mapping: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(info):
            station = _unwrap_object(raw)
            if not isinstance(station, dict):
                continue
            name = station.get("name", station.get("station", station.get("id", f"st_{index}")))
            mapping[str(name)] = station
        return mapping
    return {}


def mechanism_to_code(value: Any) -> int:
    value = _unwrap_object(value)
    if value is None:
        return -1
    if isinstance(value, (int, np.integer)):
        integer = int(value)
        if integer in {0, 1, 2}:
            return integer
        if integer in {1, 2, 3}:
            return integer - 1
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if "normal" in text or text in {"nf", "正断", "正斷"}:
        return 0
    if "strike" in text or text in {"strikeslip", "ss", "走滑"}:
        return 1
    if "reverse" in text or "thrust" in text or text in {"rv", "逆冲", "逆衝", "冲断", "衝斷"}:
        return 2
    return -1


def _rake_to_code(value: Any) -> int:
    try:
        rake = float(_unwrap_object(value))
    except (TypeError, ValueError):
        return -1
    if not np.isfinite(rake):
        return -1
    rake = ((rake + 180.0) % 360.0) - 180.0
    if abs(rake) <= 30.0 or abs(rake) >= 150.0:
        return 1
    if 30.0 <= rake <= 150.0:
        return 2
    if -150.0 <= rake <= -30.0:
        return 0
    return -1


def _event_mechanism_code(data: np.lib.npyio.NpzFile, event_index: int) -> int:
    for key in ("mechanism", "fault_type", "fm_type", "source_mechanism", "focal_mechanism", "mech"):
        if key in data:
            return mechanism_to_code(data[key][event_index])
    for key in ("rake", "rake_deg"):
        if key in data:
            return _rake_to_code(data[key][event_index])
    return -1


@dataclass(frozen=True)
class NormalizedStationRecord:
    event_index: int
    event: str
    magnitude_catalog: float
    event_lat: float
    event_lon: float
    depth_km: float
    strike: float
    dip: float
    rake: float
    mechanism: int
    station: str
    station_lat: float
    station_lon: float
    time_sec: np.ndarray
    east: np.ndarray
    north: np.ndarray
    vertical: np.ndarray
    origin_sec: float | None


def _iter_normalized_station_records(
    data: np.lib.npyio.NpzFile,
) -> Iterator[NormalizedStationRecord]:
    """把 `enu/station_info` 与 `stations` 两种 NPZ 结构归一为同一记录流。"""
    events = data["events"]
    magnitudes = data["magnitude"]
    event_lats = data["latitude"]
    event_lons = data["longitude"]
    depths = data["depth_km"] if "depth_km" in data else np.full(len(events), np.nan)
    strikes = data["strike"] if "strike" in data else np.full(len(events), np.nan)
    dips = data["dip"] if "dip" in data else np.full(len(events), np.nan)
    rakes = data["rake"] if "rake" in data else np.full(len(events), np.nan)

    if "enu" in data and "station_info" in data:
        event_containers = data["enu"]
        station_metadata = data["station_info"]
        layout = "enu"
    elif "stations" in data:
        event_containers = data["stations"]
        station_metadata = None
        layout = "stations"
    else:
        raise ValueError("NPZ 必须包含 `enu/station_info` 或 `stations`")

    for event_index, event_name in enumerate(events):
        container = event_containers[event_index]
        station_items = _iter_stations_container(container)
        metadata_map = (
            _normalize_station_info(station_metadata[event_index])
            if layout == "enu"
            else {}
        )
        for station_name, payload in station_items:
            metadata = metadata_map.get(str(station_name), {})
            station_lat = payload.get(
                "lat", payload.get("latitude", metadata.get("lat", metadata.get("latitude", np.nan)))
            )
            station_lon = payload.get(
                "lon", payload.get("longitude", metadata.get("lon", metadata.get("longitude", np.nan)))
            )
            time_sec = _get_field(payload, ["t", "time"])
            east = _get_field(payload, ["E", "east"])
            north = _get_field(payload, ["N", "north"])
            vertical = _get_field(payload, ["U", "up", "vertical"])
            origin_sec = _get_field(
                payload,
                ["origin", "origin_s", "origin_time", "origin_epoch", "origin_ts", "t0", "origin_sec"],
            )
            if time_sec is None or east is None or north is None or vertical is None:
                continue
            yield NormalizedStationRecord(
                event_index=int(event_index),
                event=str(event_name),
                magnitude_catalog=_as_float(magnitudes[event_index]),
                event_lat=_as_float(event_lats[event_index]),
                event_lon=_as_float(event_lons[event_index]),
                depth_km=_as_float(depths[event_index]),
                strike=_as_float(strikes[event_index]),
                dip=_as_float(dips[event_index]),
                rake=_as_float(rakes[event_index]),
                mechanism=_event_mechanism_code(data, event_index),
                station=str(station_name),
                station_lat=_as_float(station_lat),
                station_lon=_as_float(station_lon),
                time_sec=np.asarray(time_sec),
                east=np.asarray(east),
                north=np.asarray(north),
                vertical=np.asarray(vertical),
                origin_sec=_optional_float(origin_sec),
            )
```

`_event_mechanism_code()` 复用现有 mechanism/rake 解析规则，但只负责返回单个事件的 `0/1/2/-1` 编码；不得在两套 NPZ 分支中复制解析逻辑。

- [ ] **Step 4: 每个事件只处理一次 STF**

在数据加载循环外维护：

```python
event_stf_cache: dict[str, ProcessedSTF | None] = {}
```

缓存键为规范化事件名。调用 `resample_source_stf()` 时不得传入距离、P 速度或台站信息。

- [ ] **Step 5: 在 `src/data/sample_builder.py` 实现唯一的台站样本构建函数**

```python
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.data.geometry import compute_source_station_geometry
from src.data.records_v2 import NormalizedStationRecord
from src.data.waveform import WaveformConfig, preprocess_waveform


class SampleRejected(ValueError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _compute_phi_slip_deg(
    azimuth_deg: float,
    strike_deg: float,
    dip_deg: float,
    rake_deg: float,
) -> float:
    if not all(math.isfinite(v) for v in (strike_deg, dip_deg, rake_deg)):
        return float(azimuth_deg)
    strike = math.radians(strike_deg)
    dip = math.radians(dip_deg)
    rake = math.radians(rake_deg)
    slip_east = (
        math.cos(rake) * math.sin(strike)
        - math.sin(rake) * math.cos(dip) * math.cos(strike)
    )
    slip_north = (
        math.cos(rake) * math.cos(strike)
        + math.sin(rake) * math.cos(dip) * math.sin(strike)
    )
    slip_azimuth_deg = math.degrees(math.atan2(slip_east, slip_north))
    return float(azimuth_deg - slip_azimuth_deg)


def build_station_sample(
    record: NormalizedStationRecord,
    *,
    units: str,
    waveform_config: WaveformConfig,
    alpha_m_per_s: float,
    radial_peak_min_cm: float,
) -> dict[str, Any]:
    if not math.isfinite(record.station_lat) or not math.isfinite(record.station_lon):
        raise SampleRejected("missing_station_coordinates")
    if not math.isfinite(record.event_lat) or not math.isfinite(record.event_lon):
        raise SampleRejected("invalid_geometry", "missing event coordinates")
    if not math.isfinite(record.depth_km) or record.depth_km < 0.0:
        raise SampleRejected("invalid_geometry", "missing or negative source depth")
    if alpha_m_per_s <= 0.0:
        raise ValueError("alpha_m_per_s 必须为正")

    geometry = compute_source_station_geometry(
        record.event_lat,
        record.event_lon,
        record.depth_km,
        record.station_lat,
        record.station_lon,
    )
    time_rel = np.asarray(record.time_sec, dtype=np.float64)
    if record.origin_sec is not None:
        time_rel = time_rel - float(record.origin_sec)

    east = np.asarray(record.east, dtype=np.float64)
    north = np.asarray(record.north, dtype=np.float64)
    vertical_raw = np.asarray(record.vertical, dtype=np.float64)
    if not (time_rel.size == east.size == north.size == vertical_raw.size):
        raise SampleRejected("invalid_waveform", "component lengths differ")

    azimuth_rad = math.radians(geometry.azimuth_deg)
    radial_raw = east * math.sin(azimuth_rad) + north * math.cos(azimuth_rad)
    p_arrival_sec = geometry.source_distance_m / alpha_m_per_s
    try:
        radial = preprocess_waveform(
            time_rel,
            radial_raw,
            units=units,
            p_arrival_sec=p_arrival_sec,
            config=waveform_config,
        )
        vertical = preprocess_waveform(
            time_rel,
            vertical_raw,
            units=units,
            p_arrival_sec=p_arrival_sec,
            config=waveform_config,
        )
    except ValueError as exc:
        text = str(exc)
        reason = (
            "insufficient_baseline"
            if "baseline" in text
            else "insufficient_valid_fraction"
            if "valid fraction" in text
            else "invalid_waveform"
        )
        raise SampleRejected(reason, text) from exc

    radial_peak_cm = float(np.max(np.abs(radial.values_m)) * 100.0)
    if radial_peak_cm <= radial_peak_min_cm:
        raise SampleRejected(
            "below_radial_peak_threshold",
            f"{radial_peak_cm:.6f} <= {radial_peak_min_cm:.6f} cm",
        )

    return {
        "event": record.event,
        "event_index": record.event_index,
        "station": record.station,
        "mechanism": record.mechanism,
        "magnitude_catalog": record.magnitude_catalog,
        "radial": radial.values_m,
        "vertical": vertical.values_m,
        "waveform_valid_mask": radial.valid_mask,
        "waveform_dt_sec": radial.dt_sec,
        "raw_dt_sec": radial.raw_dt_sec,
        "valid_fraction": radial.valid_fraction,
        "baseline_source": radial.baseline_source,
        "radial_peak_cm": radial_peak_cm,
        "epicentral_distance_m": geometry.epicentral_distance_m,
        "source_distance_m": geometry.source_distance_m,
        "theta_deg": geometry.takeoff_angle_deg,
        "azimuth_deg": geometry.azimuth_deg,
        "phi_slip_deg": _compute_phi_slip_deg(
            geometry.azimuth_deg,
            record.strike,
            record.dip,
            record.rake,
        ),
    }
```

- [ ] **Step 6: 在 `src/data/dataset_v2.py` 实现事件级 STF 缓存和无歧义样本字段**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.records_v2 import NormalizedStationRecord, _iter_normalized_station_records
from src.data.sample_builder import SampleRejected, build_station_sample
from src.data.stf import ProcessedSTF, STFWindowTooShort, resample_source_stf
from src.data.waveform import WaveformConfig
from src.utils.config_v2 import (
    stf_m_ref_from_config,
    stf_output_steps_from_config,
    validate_config_v2,
)


def _normalize_event_name(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _load_stf_files(stf_dir: str | Path) -> dict[str, tuple[Path, np.ndarray, np.ndarray]]:
    root = Path(stf_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"STF directory not found: {root}")
    mapping: dict[str, tuple[Path, np.ndarray, np.ndarray]] = {}
    for path in sorted(root.glob("*.stf")):
        rows: list[tuple[float, float]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                tokens = line.replace("D", "E").split()
                if len(tokens) != 2:
                    continue
                try:
                    rows.append((float(tokens[0]), float(tokens[1])))
                except ValueError:
                    continue
        if not rows:
            continue
        key = _normalize_event_name(path.stem)
        if key in mapping:
            raise ValueError(f"duplicate normalized STF key: {key}")
        values = np.asarray(rows, dtype=np.float64)
        mapping[key] = (path, values[:, 0], values[:, 1])
    return mapping


def _match_stf(
    event_name: str,
    mapping: dict[str, tuple[Path, np.ndarray, np.ndarray]],
) -> tuple[Path, np.ndarray, np.ndarray] | None:
    key = _normalize_event_name(event_name)
    if key in mapping:
        return mapping[key]
    candidates = [item for candidate, item in mapping.items() if key in candidate or candidate in key]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        paths = ", ".join(str(item[0]) for item in candidates)
        raise ValueError(f"ambiguous STF match for {event_name}: {paths}")
    return None


def _waveform_config(config: dict[str, Any]) -> WaveformConfig:
    dataset = config["dataset"]
    waveform = dataset["waveform"]
    baseline = dataset["baseline"]
    filter_config = dataset["filter"]
    return WaveformConfig(
        sample_rate_hz=float(dataset["sample_rate_hz"]),
        start_sec=float(waveform["start_sec"]),
        duration_sec=float(waveform["duration_sec"]),
        min_valid_fraction=float(waveform["min_valid_fraction"]),
        max_interpolation_gap_sec=float(waveform["max_interpolation_gap_sec"]),
        baseline_method=str(baseline["method"]),
        pre_event_start_sec=float(baseline["pre_event_start_sec"]),
        pre_event_end_sec=float(baseline["pre_event_end_sec"]),
        baseline_fallback=str(baseline["fallback"]),
        baseline_fallback_max_sec=float(baseline["fallback_max_sec"]),
        baseline_min_samples=int(baseline["min_samples"]),
        filter_type=str(filter_config["type"]),
        cutoff_hz=float(filter_config["cutoff_hz"]),
        num_taps=int(filter_config["num_taps"]),
        filter_window=str(filter_config["window"]),
    )


class CorrectedEarthquakeDataset(Dataset):
    def __init__(self, config: dict[str, Any]) -> None:
        validate_config_v2(config)
        self.config = config
        self.samples: list[dict[str, Any]] = []
        self.rejections: list[dict[str, Any]] = []
        dataset_config = config["dataset"]
        stf_config = dataset_config["stf"]
        self.stf_m_ref = stf_m_ref_from_config(config)
        self.stf_output_steps = stf_output_steps_from_config(config)
        self.waveform_config = _waveform_config(config)
        self.blacklist = {str(value) for value in dataset_config.get("blacklist_events", [])}
        self.allow_missing_stf = bool(dataset_config.get("allow_missing_stf", False))
        self.stf_files = _load_stf_files(stf_config["path"])
        self.event_stf_cache: dict[str, tuple[ProcessedSTF, str] | None] = {}

        with np.load(config["paths"]["data_path"], allow_pickle=True) as data:
            for record in _iter_normalized_station_records(data):
                self._consume_record(record)

        accepted_events = {sample["event"] for sample in self.samples}
        print(
            f"Loaded {len(self.samples)} corrected samples from "
            f"{len(accepted_events)} accepted events"
        )

    def _event_stf(self, event_name: str) -> tuple[ProcessedSTF, str] | None:
        if event_name in self.event_stf_cache:
            return self.event_stf_cache[event_name]
        match = _match_stf(event_name, self.stf_files)
        if match is None:
            self.event_stf_cache[event_name] = None
            return None
        path, source_time, source_rate = match
        config = self.config["dataset"]["stf"]
        processed = resample_source_stf(
            source_time,
            source_rate,
            start_sec=float(config["start_sec"]),
            duration_sec=float(config["duration_sec"]),
            sample_rate_hz=float(self.config["dataset"]["sample_rate_hz"]),
            min_retained_moment_fraction=float(config["min_retained_moment_fraction"]),
            preserve_integral=bool(config["preserve_integral"]),
        )
        result = (processed, str(path))
        self.event_stf_cache[event_name] = result
        return result

    def _consume_record(self, record: NormalizedStationRecord) -> None:
        if record.event in self.blacklist:
            self.rejections.append(
                {"event": record.event, "station": record.station, "reason": "blacklisted_event"}
            )
            return
        try:
            event_stf = self._event_stf(record.event)
        except STFWindowTooShort as exc:
            self.rejections.append(
                {"event": record.event, "station": record.station, "reason": "stf_window_too_short", "detail": str(exc)}
            )
            return
        if event_stf is None and not self.allow_missing_stf:
            self.rejections.append(
                {"event": record.event, "station": record.station, "reason": "missing_stf"}
            )
            return
        try:
            sample = build_station_sample(
                record,
                units=str(self.config["dataset"]["units"]),
                waveform_config=self.waveform_config,
                alpha_m_per_s=float(self.config["physics"]["alpha"]),
                radial_peak_min_cm=float(self.config["dataset"]["radial_peak_min_cm"]),
            )
        except SampleRejected as exc:
            self.rejections.append(
                {"event": record.event, "station": record.station, "reason": exc.reason, "detail": exc.detail}
            )
            return

        if event_stf is None:
            stf = np.zeros(self.stf_output_steps, dtype=np.float64)
            stf_log = np.zeros(self.stf_output_steps, dtype=np.float32)
            stf_dt_sec = 1.0 / float(self.config["dataset"]["sample_rate_hz"])
            mw_stf_native = float("nan")
            retained_fraction = float("nan")
            stf_path = ""
            has_stf = False
        else:
            processed_stf, stf_path = event_stf
            stf = processed_stf.rate_nm_per_s
            stf_log = np.log10(1.0 + stf / self.stf_m_ref).astype(np.float32)
            stf_dt_sec = processed_stf.dt_sec
            mw_stf_native = processed_stf.mw_native
            retained_fraction = processed_stf.retained_moment_fraction
            has_stf = True

        sample.update(
            {
                "stf": stf.astype(np.float32),
                "stf_log": stf_log,
                "stf_dt_sec": float(stf_dt_sec),
                "mw_stf_native": float(mw_stf_native),
                "stf_retained_moment_fraction": float(retained_fraction),
                "stf_path": stf_path,
                "has_stf": has_stf,
            }
        )
        self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "event": sample["event"],
            "station": sample["station"],
            "event_index": torch.tensor(sample["event_index"], dtype=torch.long),
            "mechanism": torch.tensor(sample["mechanism"], dtype=torch.long),
            "radial": torch.from_numpy(sample["radial"]).float().unsqueeze(0),
            "vertical": torch.from_numpy(sample["vertical"]).float(),
            "waveform_valid_mask": torch.from_numpy(sample["waveform_valid_mask"]).bool(),
            "waveform_dt_sec": torch.tensor(sample["waveform_dt_sec"], dtype=torch.float32),
            "raw_dt_sec": torch.tensor(sample["raw_dt_sec"], dtype=torch.float32),
            "epicentral_distance_m": torch.tensor(sample["epicentral_distance_m"], dtype=torch.float32),
            "source_distance_m": torch.tensor(sample["source_distance_m"], dtype=torch.float32),
            "theta_deg": torch.tensor(sample["theta_deg"], dtype=torch.float32),
            "azimuth_deg": torch.tensor(sample["azimuth_deg"], dtype=torch.float32),
            "phi_slip_deg": torch.tensor(sample["phi_slip_deg"], dtype=torch.float32),
            "magnitude_catalog": torch.tensor(sample["magnitude_catalog"], dtype=torch.float32),
            "stf": torch.from_numpy(sample["stf"]).float(),
            "stf_log": torch.from_numpy(sample["stf_log"]).float(),
            "stf_dt_sec": torch.tensor(sample["stf_dt_sec"], dtype=torch.float32),
            "mw_stf_native": torch.tensor(sample["mw_stf_native"], dtype=torch.float32),
            "stf_retained_moment_fraction": torch.tensor(
                sample["stf_retained_moment_fraction"], dtype=torch.float32
            ),
            "has_stf": torch.tensor(sample["has_stf"], dtype=torch.bool),
            "valid_fraction": torch.tensor(sample["valid_fraction"], dtype=torch.float32),
            "radial_peak_cm": torch.tensor(sample["radial_peak_cm"], dtype=torch.float32),
            "baseline_source": sample["baseline_source"],
            "stf_path": sample["stf_path"],
        }
```

第二版不得输出含义不明的 `distance`、`dt`、`magnitude`、`theta` 或 `phi` 键；所有距离、时间步和角度字段必须带物理语义与单位。

- [ ] **Step 7: 修正事件计数和加载摘要**

最终事件数必须由成功样本计算：

```python
accepted_events = {sample["event"] for sample in self.samples}
print(f"Loaded {len(self.samples)} samples from {len(accepted_events)} accepted events")
```

- [ ] **Step 8: 运行数据层测试**

```bash
.venv/bin/python -m pytest tests/test_geometry.py tests/test_waveform_preprocessing.py tests/test_stf_targets.py tests/test_data_loader.py tests/test_corrected_pipeline_integration.py -v
```

Expected: PASS。

- [ ] **Step 9: 提交**

```bash
git add \
  src/data/records_v2.py \
  src/data/sample_builder.py \
  src/data/dataset_v2.py \
  tests/test_corrected_pipeline_integration.py
git commit -m "feat: add isolated invariant v2 earthquake dataset"
```

---

### Task 7: 生成完整数据 manifest 和排除原因

**Files:**
- Create: `src/data/manifest.py`
- Create: `scripts/data/audit_corrected_dataset.py`
- Create: `tests/test_dataset_manifest.py`
- Modify: `src/data/dataset_v2.py`
- Do not modify: `src/data/data_loader.py`

- [ ] **Step 1: 定义每个候选台站都必须输出的 manifest 字段**

```python
MANIFEST_FIELDS = [
    "event_index", "event", "station", "accepted", "rejection_reason",
    "magnitude_catalog", "mw_stf_native", "has_stf",
    "epicentral_distance_km", "source_distance_km", "theta_deg", "azimuth_deg",
    "raw_dt_sec", "waveform_dt_sec", "valid_fraction", "baseline_source",
    "radial_peak_cm", "stf_retained_moment_fraction",
]
```

拒绝原因只能来自固定枚举：

```python
REJECTION_REASONS = {
    "blacklisted_event", "missing_station_coordinates", "invalid_waveform",
    "insufficient_baseline", "insufficient_valid_fraction", "below_radial_peak_threshold",
    "missing_stf", "stf_window_too_short", "invalid_geometry",
}
```

- [ ] **Step 2: 写入 manifest 测试**

测试必须构造一条通过样本和一条低于阈值样本，并断言两者均出现在 CSV 中，且后者为：

```python
assert row["accepted"] == "False"
assert row["rejection_reason"] == "below_radial_peak_threshold"
```

- [ ] **Step 3: 实现审计脚本**

```bash
.venv/bin/python scripts/data/audit_corrected_dataset.py \
  --config configs/config_v2.yaml \
  --manifest outputs_v2/audit/dataset_manifest.csv \
  --summary outputs_v2/audit/dataset_summary.json
```

`dataset_summary.json` 必须包含：

```json
{
  "candidate_event_count": 0,
  "accepted_event_count": 0,
  "candidate_station_count": 0,
  "accepted_station_count": 0,
  "rejection_counts": {},
  "events": {},
  "invariants": {
    "all_waveform_dt_equal_1s": false,
    "one_stf_per_event": false,
    "one_stf_mw_per_event": false,
    "min_stf_retained_fraction": 0.0
  }
}
```

脚本用真实统计替换上述初始值；任一 invariant 为 false 时退出码必须非零。

- [ ] **Step 4: 运行测试和真实只读审计**

```bash
.venv/bin/python -m pytest tests/test_dataset_manifest.py -v
.venv/bin/python scripts/data/audit_corrected_dataset.py --config configs/config_v2.yaml --manifest outputs_v2/audit/dataset_manifest.csv --summary outputs_v2/audit/dataset_summary.json
```

Expected: 测试 PASS；真实审计只有在 200 s 源时间窗口对每个 STF 保留至少 99.5% 矩时才成功。

- [ ] **Step 5: 对 STF 窗口审计失败采用确定性扩展规则**

若真实审计因 STF 窗口不足失败，执行以下规则而不是人工挑选：

1. 依次测试 `duration_sec = 240, 300, 360, 420, 480, 600`。
2. 选择第一个使所有纳入事件 `retained_moment_fraction >= 0.995` 的窗口。
3. 将该值写入 `configs/config_v2.yaml`。
4. Task 8 的模型输出长度测试必须覆盖该 STF 长度与 200 点输入长度不同的情况。

- [ ] **Step 6: 提交**

```bash
git add src/data/manifest.py scripts/data/audit_corrected_dataset.py tests/test_dataset_manifest.py src/data/dataset_v2.py configs/config_v2.yaml
git commit -m "feat: add reproducible dataset manifest and invariants"
```

---

### Task 8: 允许模型输出长度与波形输入长度分离

**Files:**
- Modify: `src/models/model.py`
- Modify: `tests/test_model_forward.py`

- [ ] **Step 1: 增加输出长度失败测试**

```python
from pathlib import Path

import yaml


def test_output_length_can_differ_from_input_length():
    config = yaml.safe_load(Path("configs/config_v2.yaml").read_text(encoding="utf-8"))
    config["dataset"]["stf"]["duration_sec"] = 300.0
    model = PINNModel(config).eval()
    x = torch.randn(2, 1, 200)
    meta = _make_meta(2, torch.device("cpu"))
    with torch.no_grad():
        out = model(x, meta=meta)
    assert out.shape == (2, 300)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/bin/python -m pytest tests/test_model_forward.py -k output_length_can_differ -v
```

Expected: FAIL，当前输出仍为输入长度。

- [ ] **Step 3: 在输出头前统一调整 source-time latent 长度**

在 `PINNModel.__init__` 中替换当前 `rate_representation == "auto"` 推断，并同时设置输出长度：

```python
training_cfg = config.get("training", {}) or {}
dataset_cfg = config.get("dataset", {}) or {}
pipeline_version = int(config.get("pipeline_version", 1))

if pipeline_version == 2:
    from src.utils.config_v2 import stf_output_steps_from_config, validate_config_v2

    validate_config_v2(config)
    rate_representation = str(training_cfg["rate_representation"]).lower()
    self.output_time_steps: int | None = stf_output_steps_from_config(config)
else:
    rate_representation = str(training_cfg.get("rate_representation", "auto")).lower()
    if rate_representation == "auto":
        rate_representation = "log1p" if "stf_m_ref" in dataset_cfg else "linear"
    self.output_time_steps = None

if rate_representation not in {"log1p", "linear"}:
    raise ValueError(f"不支持的 rate_representation: {rate_representation}")
```

在 `forward()` 的输出头前：

```python
seq_time = feat.transpose(1, 2)
if self.output_time_steps is not None and seq_time.size(1) != self.output_time_steps:
    seq_time = torch.nn.functional.interpolate(
        seq_time.transpose(1, 2),
        size=self.output_time_steps,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
rate = self.rate_head(seq_time)
```

同步更新 `debug_forward()`。

- [ ] **Step 4: 运行模型测试**

```bash
.venv/bin/python -m pytest tests/test_model_forward.py -v
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/models/model.py tests/test_model_forward.py
git commit -m "feat: decouple source STF and waveform lengths"
```

---

### Task 9: 新建第二版正演损失并用源时间到观测时间的可微采样

**Files:**
- Create: `src/training/time_sampling.py`
- Create: `src/training/loss_stf_rate_v2.py`
- Create: `tests/test_forward_operator.py`
- Do not modify: `src/training/loss_stf_rate.py` 的 legacy 正演与损失

- [ ] **Step 1: 写入延迟采样失败测试**

```python
# tests/test_forward_operator.py
import torch

from src.training.time_sampling import sample_source_history


def test_fractional_delay_uses_linear_interpolation():
    source = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    sampled = sample_source_history(
        source,
        source_dt_sec=torch.tensor([1.0]),
        observation_dt_sec=torch.tensor([1.0]),
        delay_sec=torch.tensor([1.5]),
        observation_steps=5,
    )
    assert sampled.shape == (1, 5)
    assert torch.allclose(sampled[0], torch.tensor([0.0, 0.0, 0.5, 0.5, 0.0]), atol=1.0e-6)


def test_delay_larger_than_window_returns_zeros():
    source = torch.ones(1, 4)
    sampled = sample_source_history(source, torch.tensor([1.0]), torch.tensor([1.0]),
                                    torch.tensor([10.0]), observation_steps=5)
    assert torch.count_nonzero(sampled) == 0
```

- [ ] **Step 2: 实现批次线性采样**

```python
# src/training/time_sampling.py
from __future__ import annotations

import torch


def sample_source_history(
    source: torch.Tensor,
    source_dt_sec: torch.Tensor,
    observation_dt_sec: torch.Tensor,
    delay_sec: torch.Tensor,
    observation_steps: int,
) -> torch.Tensor:
    batch, source_steps = source.shape
    obs_index = torch.arange(observation_steps, device=source.device, dtype=source.dtype).view(1, -1)
    obs_time = obs_index * observation_dt_sec.reshape(batch, 1)
    source_position = (obs_time - delay_sec.reshape(batch, 1)) / source_dt_sec.reshape(batch, 1)
    left = torch.floor(source_position).to(torch.long)
    right = left + 1
    weight_right = source_position - left.to(source.dtype)
    valid = (left >= 0) & (right < source_steps)
    left_safe = left.clamp(0, source_steps - 1)
    right_safe = right.clamp(0, source_steps - 1)
    left_value = torch.gather(source, 1, left_safe)
    right_value = torch.gather(source, 1, right_safe)
    interpolated = left_value * (1.0 - weight_right) + right_value * weight_right
    return torch.where(valid, interpolated, torch.zeros_like(interpolated))
```

- [ ] **Step 3: 改写正演接口**

`forward_displacement_from_rate()` 新接口必须为：

```python
def forward_displacement_from_rate(
    rate_hat: torch.Tensor,
    source_dt_sec: torch.Tensor,
    observation_dt_sec: torch.Tensor,
    observation_steps: int,
    source_distance_m: torch.Tensor,
    alpha: float,
    beta: float,
    C_int_P: torch.Tensor,
    C_int_S: torch.Tensor,
    C_far_P: torch.Tensor,
    C_far_S: torch.Tensor,
    *,
    include_intermediate: bool,
    include_far_P: bool,
    include_far_S: bool,
    include_intermediate_P: bool,
    include_intermediate_S: bool,
) -> torch.Tensor:
```

内部逻辑：

```python
moment_hat = torch.cumsum(rate_hat * source_dt_sec.reshape(-1, 1), dim=1)
tp = source_distance_m / alpha
ts = source_distance_m / beta
rate_p = sample_source_history(rate_hat, source_dt_sec, observation_dt_sec, tp, observation_steps)
rate_s = sample_source_history(rate_hat, source_dt_sec, observation_dt_sec, ts, observation_steps)
moment_p = sample_source_history(moment_hat, source_dt_sec, observation_dt_sec, tp, observation_steps)
moment_s = sample_source_history(moment_hat, source_dt_sec, observation_dt_sec, ts, observation_steps)
```

第二版代码中删除 `skip_delays` 分支。

- [ ] **Step 4: 用源距替换震中距**

`compute_physical_coefficients()` 参数重命名为 `source_distance_m`，并在 docstring 中明确它是 \(R\)，而不是 Δ。

- [ ] **Step 5: 运行正演测试**

```bash
.venv/bin/python -m pytest tests/test_forward_operator.py -v
```

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add src/training/time_sampling.py src/training/loss_stf_rate_v2.py tests/test_forward_operator.py
git commit -m "fix: apply absolute travel delays on source time STF"
```

---

### Task 10: 完成第二版损失中的逐样本时间步、mask、幅值增益和死损失清理

**Files:**
- Modify: `src/training/loss_stf_rate_v2.py`
- Modify: `src/training/train.py`
- Modify: `scripts/plotting/plot_training_curves.py`
- Modify: `scripts/experiments/run_experiment.py`
- Create: `tests/test_stf_rate_loss_v2.py`
- Do not modify: `src/training/loss_stf_rate.py` 的 legacy `L_nonneg` 与 delay 行为

- [ ] **Step 1: 写入逐样本积分失败测试**

```python
# tests/test_stf_rate_loss_v2.py
import torch

from src.training.loss_stf_rate_v2 import moment_magnitude_from_rate


def test_per_sample_source_dt_changes_only_its_own_moment():
    rate = torch.ones(2, 4) * 1.0e18
    mw = moment_magnitude_from_rate(rate, torch.tensor([1.0, 2.0]))
    expected_delta = (2.0 / 3.0) * torch.log10(torch.tensor(2.0))
    assert torch.allclose(mw[1] - mw[0], expected_delta, atol=1.0e-6)


def test_invalid_waveform_samples_do_not_enter_synth_loss():
    u_hat = torch.tensor([[1.0, 100.0]])
    u_obs = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, False]])
    from src.training.loss_stf_rate_v2 import masked_normalized_waveform_mse
    loss = masked_normalized_waveform_mse(u_hat, u_obs, mask)
    assert loss.item() == 0.0
```

- [ ] **Step 2: 实现逐样本震级积分**

```python
def moment_magnitude_from_rate(rate_nm_per_s: torch.Tensor, dt_sec: torch.Tensor) -> torch.Tensor:
    dt = dt_sec.reshape(-1, 1).to(device=rate_nm_per_s.device, dtype=rate_nm_per_s.dtype)
    moment = torch.sum(torch.clamp(rate_nm_per_s, min=0.0) * dt, dim=1).clamp_min(1.0e10)
    return (2.0 / 3.0) * (torch.log10(moment) - 9.1)
```

不得再使用 `dt_b[0]` 或 batch mean。

- [ ] **Step 3: 实现有效样点 mask 的合成波形损失**

```python
def masked_normalized_waveform_mse(
    u_hat: torch.Tensor,
    u_obs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    mask = valid_mask.to(device=u_obs.device, dtype=u_obs.dtype)
    observed_abs = torch.where(valid_mask, u_obs.abs(), torch.zeros_like(u_obs))
    scale = observed_abs.amax(dim=1, keepdim=True).clamp_min(1.0e-12)
    squared = ((u_hat - u_obs) / scale).pow(2) * mask
    return squared.sum() / mask.sum().clamp_min(1.0)
```

- [ ] **Step 4: 把三个固定因子替换为单一增益**

`compute_physical_coefficients()` 只接受：

```python
amplitude_gain: float = 1.0
```

并使用：

```python
scale = float(amplitude_gain)
```

删除 `geom`、`free_surface`、`attenuation` 参数与成员变量。`STFRateWaveformLoss.__init__` 同时改为显式读取第二版嵌套 STF 配置：

```python
pipeline_version = int(config.get("pipeline_version", 1))
if pipeline_version == 2:
    from src.utils.config_v2 import stf_m_ref_from_config, validate_config_v2

    validate_config_v2(config)
    self.rate_representation = str(config["training"]["rate_representation"]).lower()
    self.stf_m_ref = stf_m_ref_from_config(config)
    self.amplitude_gain = float(config["physics"]["amplitude_gain"])
else:
    dataset_cfg = config.get("dataset", {}) or {}
    training_cfg = config.get("training", {}) or {}
    self.rate_representation = str(training_cfg.get("rate_representation", "auto")).lower()
    if self.rate_representation == "auto":
        self.rate_representation = "log1p" if "stf_m_ref" in dataset_cfg else "linear"
    self.stf_m_ref = float(dataset_cfg.get("stf_m_ref", 1.0e18))
    physics_cfg = config.get("physics", {}) or {}
    self.amplitude_gain = float(
        physics_cfg.get("geometrical_spreading_factor", 1.0)
        * physics_cfg.get("free_surface_factor", 1.0)
        * physics_cfg.get("attenuation", 1.0)
    )
```

第二版评估代码也必须调用 `stf_m_ref_from_config(config)`，不得继续使用 `dataset.get("stf_m_ref", 1.0e18)`。

- [ ] **Step 5: 删除 `L_nonneg` 全链路**

第二版实现从以下位置彻底排除该项：

- `pinn_loss_stf_rate_v2()` 的参数和总损失；
- `STFRateWaveformLossV2.__init__`；
- 第二版 `loss_dict`；
- `configs/config_v2.yaml` 与 `configs/experiments_v2/**`；
- 第二版实验 CSV；
- 第二版训练曲线标签。

legacy 的 `src/training/loss_stf_rate.py`、`configs/legacy/**` 和旧输出配置不修改，以保证旧权重仍可加载并复现实验。

- [ ] **Step 6: 训练入口按版本选择损失，并使用第二版显式字段**

初始化损失时：

```python
if int(config.get("pipeline_version", 1)) == 2:
    from src.training.loss_stf_rate_v2 import STFRateWaveformLossV2

    criterion_2 = STFRateWaveformLossV2(config).to(device)
else:
    from src.training.loss_stf_rate import STFRateWaveformLoss

    criterion_2 = STFRateWaveformLoss(config).to(device)
```

第二版 batch 处理为：

```python
source_distance_m = batch["source_distance_m"].to(device)
source_dt_sec = batch["stf_dt_sec"].to(device)
observation_dt_sec = batch["waveform_dt_sec"].to(device)
waveform_valid_mask = batch["waveform_valid_mask"].to(device)
meta = build_metadata_tensor(
    source_distance_m,
    batch["theta_deg"].to(device),
    batch["azimuth_deg"].to(device),
)
```

`true_mag` 按配置选择：

```python
if magnitude_target == "stf_native":
    true_mag = batch["mw_stf_native"].to(device)
elif magnitude_target == "catalog":
    true_mag = batch["magnitude_catalog"].to(device)
else:
    raise ValueError(f"未知 magnitude_target: {magnitude_target}")
```

- [ ] **Step 7: 运行训练损失测试**

```bash
.venv/bin/python -m pytest tests/test_forward_operator.py tests/test_stf_rate_loss_v2.py -v
```

Expected: PASS。

- [ ] **Step 8: 提交**

```bash
git add src/training/loss_stf_rate_v2.py src/training/train.py scripts/plotting/plot_training_curves.py scripts/experiments/run_experiment.py tests/test_stf_rate_loss_v2.py configs/config_v2.yaml
git commit -m "fix: use per sample time axes and explicit amplitude gain"
```

---

### Task 11: 建立事件级划分和事件均衡采样

**Files:**
- Create: `src/data/splits.py`
- Create: `src/data/loaders_v2.py`
- Create: `tests/test_group_splits.py`
- Modify: `src/training/train.py`
- Modify: `scripts/experiments/loeo_cv.py`
- Do not modify: `src/data/data_loader.py` 的 legacy `random_split()` 路径

- [ ] **Step 1: 写入无事件重叠测试**

```python
# tests/test_group_splits.py
from src.data.splits import make_event_group_split, make_event_balanced_weights


def test_group_split_has_no_event_overlap():
    events = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]
    split = make_event_group_split(events, validation_fraction=0.2, test_fraction=0.2, seed=42)
    train_events = {events[i] for i in split.train_indices}
    val_events = {events[i] for i in split.validation_indices}
    test_events = {events[i] for i in split.test_indices}
    assert train_events.isdisjoint(val_events)
    assert train_events.isdisjoint(test_events)
    assert val_events.isdisjoint(test_events)


def test_event_balanced_weights_sum_equally_per_event():
    events = ["A", "A", "A", "B"]
    weights = make_event_balanced_weights(events)
    assert abs(sum(weights[:3]) - weights[3]) < 1.0e-12
```

- [ ] **Step 2: 实现确定性事件划分**

`make_event_group_split()` 必须：

1. 对唯一事件名排序；
2. 使用 `np.random.default_rng(seed).permutation()`；
3. 按事件数而不是记录数分配 validation/test；
4. 至少保留一个训练事件；
5. 返回索引数据类；
6. 提供 `assert_no_event_overlap()`。

- [ ] **Step 3: 实现事件均衡权重**

```python
def make_event_balanced_weights(events: list[str]) -> list[float]:
    from collections import Counter
    counts = Counter(events)
    return [1.0 / counts[event] for event in events]
```

训练 loader 使用：

```python
sampler = torch.utils.data.WeightedRandomSampler(
    weights=torch.as_tensor(weights, dtype=torch.double),
    num_samples=len(train_indices),
    replacement=True,
    generator=torch.Generator().manual_seed(seed),
)
```

- [ ] **Step 4: 保留三种显式协议**

`src/data/loaders_v2.py` 暴露 `get_data_loaders_v2(config, *, leave_out_event=None)`，只接受：

- `grouped_event`：第二版主训练/内部验证；
- `within_event_station`：论文补充的同事件台站插值；
- `loeo`：跨事件主评估，必须提供 `leave_out_event`。

`src/training/train.py` 按版本路由，确保 legacy 数值路径仍可复现：

```python
if int(config.get("pipeline_version", 1)) == 2:
    from src.data.loaders_v2 import get_data_loaders_v2

    train_loader, val_loader, test_loader, split_manifest = get_data_loaders_v2(config)
else:
    from src.data.data_loader import get_data_loaders

    train_loader, val_loader, test_loader = get_data_loaders(config)
    split_manifest = None
```

第二版不得调用 legacy 的无事件分组 `random_split()`；旧函数保留只为复现历史结果。

- [ ] **Step 5: 每次运行保存 split manifest**

保存 `split.json`：

```json
{
  "protocol": "grouped_event",
  "seed": 42,
  "train_events": [],
  "validation_events": [],
  "test_events": [],
  "train_record_count": 0,
  "validation_record_count": 0,
  "test_record_count": 0
}
```

- [ ] **Step 6: 运行测试**

```bash
.venv/bin/python -m pytest tests/test_group_splits.py tests/test_data_loader.py -v
```

Expected: PASS。

- [ ] **Step 7: 提交**

```bash
git add src/data/splits.py src/data/loaders_v2.py src/training/train.py scripts/experiments/loeo_cv.py tests/test_group_splits.py
git commit -m "feat: add event grouped splits and balanced sampling"
```

---

### Task 12: 统一训练、标准评估和外部事件的 metadata 构建

**Files:**
- Modify: `src/training/train.py`
- Modify: `src/evaluation/evaluate.py`
- Modify: `src/evaluation/evaluate_no_stf.py`
- Modify: `src/evaluation/evaluate_unseen.py`
- Modify: `scripts/experiments/loeo_event_level_eval.py`
- Modify: `scripts/robustness/latency_analysis.py`
- Modify: `paper/srl/figure_sources/fig3_new/gen_test_csv_far_only.py`
- Create: `tests/test_metadata_call_sites.py`

- [ ] **Step 1: 替换所有手写 `torch.stack` 元数据**

所有调用统一为：

```python
meta = build_metadata_tensor(
    batch["source_distance_m"].to(device),
    batch["theta_deg"].to(device),
    batch["azimuth_deg"].to(device),
)
```

外部单台站样本先构造 1 元素 tensor，再调用同一函数。

- [ ] **Step 2: 添加静态回归测试**

```python
# tests/test_metadata_call_sites.py
from pathlib import Path


FILES = [
    "src/training/train.py",
    "src/evaluation/evaluate.py",
    "src/evaluation/evaluate_no_stf.py",
    "src/evaluation/evaluate_unseen.py",
    "scripts/experiments/loeo_event_level_eval.py",
    "scripts/robustness/latency_analysis.py",
]


def test_no_production_file_manually_stacks_geometry_metadata():
    for name in FILES:
        text = Path(name).read_text(encoding="utf-8")
        assert "build_metadata_tensor" in text
        assert "torch.stack([\n                dist_log" not in text
        assert "phi_deg = ds_helper.default_phi_deg" not in text
```

- [ ] **Step 3: 删除未见事件中的错误变量命名**

以下代码必须消失：

```python
dist_m, theta_deg, _ = ds_helper._calculate_geodetics(...)
```

替换为 `compute_source_station_geometry()` 返回的数据类，径向投影使用 `geometry.azimuth_deg`，网络角度使用 `geometry.takeoff_angle_deg`，第五维使用 `geometry.azimuth_deg`。

- [ ] **Step 4: 运行元数据测试**

```bash
.venv/bin/python -m pytest tests/test_geometry.py tests/test_metadata_call_sites.py tests/test_unseen_event_eval.py -v
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/training/train.py src/evaluation/evaluate.py src/evaluation/evaluate_no_stf.py src/evaluation/evaluate_unseen.py scripts/experiments/loeo_event_level_eval.py scripts/robustness/latency_analysis.py paper/srl/figure_sources/fig3_new/gen_test_csv_far_only.py tests/test_metadata_call_sites.py
git commit -m "fix: unify geometry metadata across inference paths"
```

---

### Task 13: 外部事件通过适配器复用第二版样本构建器

**Files:**
- Create: `src/data/external_records.py`
- Modify: `src/data/waveform.py`
- Modify: `src/data/dataset_v2.py`
- Modify: `src/evaluation/evaluate_unseen.py`
- Modify: `scripts/robustness/noise_robustness.py`
- Modify: `scripts/robustness/station_dropout.py`
- Modify: `scripts/robustness/latency_analysis.py`
- Modify: `scripts/plotting/fig2_real_event_panels.py`
- Modify: `tests/test_unseen_event_eval.py`
- Do not modify: `src/data/data_loader.py`

- [ ] **Step 1: 把 `WaveformConfig` 构造器提升为公共函数**

把 Task 6 中 `dataset_v2._waveform_config()` 移到 `src/data/waveform.py` 并改名为：

```python
def waveform_config_from_v2(config: dict[str, Any]) -> WaveformConfig:
    dataset = config["dataset"]
    waveform = dataset["waveform"]
    baseline = dataset["baseline"]
    filter_config = dataset["filter"]
    return WaveformConfig(
        sample_rate_hz=float(dataset["sample_rate_hz"]),
        start_sec=float(waveform["start_sec"]),
        duration_sec=float(waveform["duration_sec"]),
        min_valid_fraction=float(waveform["min_valid_fraction"]),
        max_interpolation_gap_sec=float(waveform["max_interpolation_gap_sec"]),
        baseline_method=str(baseline["method"]),
        pre_event_start_sec=float(baseline["pre_event_start_sec"]),
        pre_event_end_sec=float(baseline["pre_event_end_sec"]),
        baseline_fallback=str(baseline["fallback"]),
        baseline_fallback_max_sec=float(baseline["fallback_max_sec"]),
        baseline_min_samples=int(baseline["min_samples"]),
        filter_type=str(filter_config["type"]),
        cutoff_hz=float(filter_config["cutoff_hz"]),
        num_taps=int(filter_config["num_taps"]),
        filter_window=str(filter_config["window"]),
    )
```

`CorrectedEarthquakeDataset` 和外部事件评估均导入这一函数，不得各自解析 YAML。

- [ ] **Step 2: 实现 `EventBundle` 到 `NormalizedStationRecord` 的无损适配器**

```python
# src/data/external_records.py
from __future__ import annotations

from src.data.records_v2 import NormalizedStationRecord, mechanism_to_code


def record_from_external_bundle(bundle, station) -> NormalizedStationRecord:
    return NormalizedStationRecord(
        event_index=-1,
        event=str(bundle.event_name),
        magnitude_catalog=float(bundle.magnitude),
        event_lat=float(bundle.latitude),
        event_lon=float(bundle.longitude),
        depth_km=float(bundle.depth_km),
        strike=float(bundle.strike),
        dip=float(bundle.dip),
        rake=float(bundle.rake),
        mechanism=mechanism_to_code(bundle.mechanism),
        station=str(station.station),
        station_lat=float(station.latitude),
        station_lon=float(station.longitude),
        time_sec=station.t.copy(),
        east=station.e_m.copy(),
        north=station.n_m.copy(),
        vertical=station.u_m.copy(),
        origin_sec=0.0,
    )
```

`records_v2.py` 中的 `mechanism_to_code()` 必须保持公共、无副作用，并由 NPZ 解析器与外部事件适配器共同调用。外部事件随后直接调用 Task 6 已实现的：

```python
sample = build_station_sample(
    record_from_external_bundle(bundle, station),
    units="m",
    waveform_config=waveform_config_from_v2(config),
    alpha_m_per_s=float(config["physics"]["alpha"]),
    radial_peak_min_cm=effective_threshold_cm,
)
```

该路径不加载 STF；网络预测积分后的 Mw 只与 `bundle.magnitude` 和可选的事件级参考 STF 分列比较。

- [ ] **Step 3: 删除 `_build_unseen_dataset_helper()` 的伪数据集实例化**

外部事件不得为了复用若干方法而构造 `EarthquakeDataset(mock.npz)`。改为从配置构造 `WaveformConfig`，直接调用纯函数。

- [ ] **Step 4: 添加训练/外部路径等价测试**

同一合成事件和台站分别通过训练 NPZ 入口与外部 `EventBundle` 入口处理，断言：

```python
assert np.allclose(training_sample["radial"], external_sample["radial"])
assert np.array_equal(training_sample["waveform_valid_mask"], external_sample["waveform_valid_mask"])
assert training_sample["source_distance_m"] == external_sample["source_distance_m"]
assert training_sample["theta_deg"] == external_sample["theta_deg"]
assert training_sample["azimuth_deg"] == external_sample["azimuth_deg"]
```

- [ ] **Step 5: 运行外部评估测试**

```bash
.venv/bin/python -m pytest tests/test_unseen_event_eval.py tests/test_corrected_pipeline_integration.py -v
```

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add src/data/external_records.py src/data/waveform.py src/data/dataset_v2.py src/data/records_v2.py src/evaluation/evaluate_unseen.py scripts/robustness/noise_robustness.py scripts/robustness/station_dropout.py scripts/robustness/latency_analysis.py scripts/plotting/fig2_real_event_panels.py tests/test_unseen_event_eval.py
git commit -m "refactor: share station preprocessing across all datasets"
```

---

### Task 14: 建立事件级指标和明确的目标列

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `tests/test_event_metrics.py`
- Modify: `src/evaluation/evaluate.py`
- Modify: `src/evaluation/evaluate_unseen.py`
- Modify: `scripts/experiments/loeo_event_level_eval.py`
- Modify: `src/evaluation/bootstrap.py`

- [ ] **Step 1: 写入事件等权失败测试**

```python
# tests/test_event_metrics.py
from src.evaluation.metrics import aggregate_event_predictions, summarize_predictions


def test_dense_event_does_not_dominate_event_mae():
    rows = [
        {"event": "A", "mw_pred": 8.0, "mw_catalog": 7.0},
        {"event": "A", "mw_pred": 8.0, "mw_catalog": 7.0},
        {"event": "A", "mw_pred": 8.0, "mw_catalog": 7.0},
        {"event": "B", "mw_pred": 7.0, "mw_catalog": 7.0},
    ]
    events = aggregate_event_predictions(rows, reference_key="mw_catalog")
    metrics = summarize_predictions(rows, events, reference_key="mw_catalog")
    assert metrics["event_mae"] == 0.5
    assert metrics["station_mae"] == 0.75


def test_event_prediction_is_station_median():
    rows = [
        {"event": "A", "mw_pred": 6.0, "mw_catalog": 7.0},
        {"event": "A", "mw_pred": 7.0, "mw_catalog": 7.0},
        {"event": "A", "mw_pred": 10.0, "mw_catalog": 7.0},
    ]
    events = aggregate_event_predictions(rows, reference_key="mw_catalog")
    assert events[0]["mw_pred_median"] == 7.0
```

- [ ] **Step 2: 实现指标模块**

每次评估必须同时输出：

```python
{
    "reference": "catalog",
    "event_count": int,
    "station_count": int,
    "event_mae": float,
    "event_rmse": float,
    "event_bias": float,
    "station_mae": float,
    "station_rmse": float,
    "station_bias": float,
}
```

- [ ] **Step 3: 删除含义混合的 `mw_true`**

台站 CSV 使用以下明确列：

```text
event,station,mw_pred,mw_catalog,mw_stf_native,error_vs_catalog,error_vs_stf_native,
epicentral_distance_km,source_distance_km,theta_deg,azimuth_deg,threshold_cm
```

事件 CSV 使用：

```text
event,mw_pred_median,mw_catalog,mw_stf_native,error_vs_catalog,error_vs_stf_native,
n_stations,pred_std,pred_iqr
```

- [ ] **Step 4: bootstrap 只对事件行重采样**

`src/evaluation/bootstrap.py` 输入必须是事件级表；若收到同一事件多行则抛出错误。

- [ ] **Step 5: 运行指标测试**

```bash
.venv/bin/python -m pytest tests/test_event_metrics.py tests/test_unseen_event_eval.py -v
```

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add src/evaluation/metrics.py src/evaluation/evaluate.py src/evaluation/evaluate_unseen.py scripts/experiments/loeo_event_level_eval.py src/evaluation/bootstrap.py tests/test_event_metrics.py
git commit -m "feat: make event level magnitude metrics primary"
```

---

### Task 15: 修正 PGD 标度律的距离合同与来源命名

**Files:**
- Modify: `src/baseline/scaling_laws.py`
- Modify: `src/evaluation/evaluate_unseen.py`
- Modify: `scripts/evaluation/evaluate_pgd_scaling_laws.py`
- Modify: `tests/test_scaling_laws.py`
- Modify: `tests/test_pgd_eval_helpers.py`

- [ ] **Step 1: 扩展标度律规格**

```python
@dataclass(frozen=True)
class ScalingLawSpec:
    name: str
    citation_key: str
    pgd_unit: str
    distance_kind: str
    a: float
    b: float
    c: float
```

三个当前标度律均显式设置：

```python
distance_kind="hypocentral"
```

Crowell 常量名称与稿件引用年份统一；代码、表格和 citation key 不得出现 `CROWELL_2013` 与 “Crowell 2016” 并存。

- [ ] **Step 2: 改写预测接口**

```python
def predict_mw(*, law_name: str, pgd_m: float, source_distance_km: float) -> float:
```

删除含义模糊的 `distance_km` 参数名。

- [ ] **Step 3: 外部评估用同一 source distance**

```python
source_distance_km = sample["source_distance_m"] / 1000.0
p_arrival_sec = sample["source_distance_m"] / config["physics"]["alpha"]
```

不得再重新计算 epicentral distance 后传给 PGD。

- [ ] **Step 4: 增加近震中深源测试**

构造 `epicentral=10 km, depth=30 km`，断言传给标度律的是 `sqrt(10^2+30^2)` km，而不是 10 km。

- [ ] **Step 5: 运行测试并提交**

```bash
.venv/bin/python -m pytest tests/test_scaling_laws.py tests/test_pgd_eval_helpers.py -v
git add src/baseline/scaling_laws.py src/evaluation/evaluate_unseen.py scripts/evaluation/evaluate_pgd_scaling_laws.py tests/test_scaling_laws.py tests/test_pgd_eval_helpers.py
git commit -m "fix: use explicit source distance in pgd baselines"
```

---

### Task 16: 加入运行来源链和结果注册表

**Files:**
- Create: `src/utils/provenance.py`
- Create: `tests/test_provenance.py`
- Modify: `src/training/train.py`
- Modify: `src/evaluation/evaluate.py`
- Modify: `scripts/experiments/run_experiment.py`

- [ ] **Step 1: 每次训练写入 `run_manifest.json`**

字段固定为：

```json
{
  "pipeline_version": 2,
  "git_commit": "",
  "git_dirty": false,
  "config_sha256": "",
  "dataset_manifest_sha256": "",
  "split_sha256": "",
  "checkpoint_sha256": "",
  "python_version": "",
  "torch_version": "",
  "numpy_version": "",
  "random_seed": 42,
  "started_at_utc": "",
  "completed_at_utc": ""
}
```

- [ ] **Step 2: 实现 SHA-256 和 git 信息函数**

```python
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit(root: str | Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(root),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_is_dirty(root: str | Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(root),
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip())
```

- [ ] **Step 3: 评估结果写入 `metrics.json` 和 `result_registry.json`**

`result_registry.json` 记录：

- checkpoint path/hash；
- config path/hash；
- dataset manifest path/hash；
- split protocol；
- primary/secondary reference；
- station/event metrics；
- 所有生成 CSV 和图件路径。

- [ ] **Step 4: 测试哈希在文件变化后变化**

```python
def test_sha256_changes_when_file_changes(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("a", encoding="utf-8")
    first = sha256_file(path)
    path.write_text("b", encoding="utf-8")
    assert sha256_file(path) != first
```

- [ ] **Step 5: 运行测试并提交**

```bash
.venv/bin/python -m pytest tests/test_provenance.py -v
git add src/utils/provenance.py src/training/train.py src/evaluation/evaluate.py scripts/experiments/run_experiment.py tests/test_provenance.py
git commit -m "feat: record reproducible result provenance"
```

---

### Task 17: 先量化旧模型角度错误，再启动重训

**Files:**
- Create: `scripts/evaluation/diagnose_legacy_metadata.py`
- Create: `tests/test_legacy_metadata_diagnostic.py`
- Output only: `outputs_v2/diagnostics/legacy_metadata/**`

- [ ] **Step 1: 对同一旧 checkpoint 定义四种只推断模式**

```text
legacy_exact      = [log Delta, sin azimuth, cos azimuth, 0, 1]
theta_only_fixed  = [log Delta, sin theta, cos theta, 0, 1]
geometry_fixed    = [log Delta, sin theta, cos theta, sin azimuth, cos azimuth]
metadata_disabled = model receives no metadata contribution
```

该脚本只用于估算外部事件 metadata bug 的影响，不用于更新论文最终性能。

- [ ] **Step 2: 每种模式输出相同事件/台站集合**

若某模式的样本集合不同，脚本必须失败。四份 CSV 按 `(event, station)` 排序后键集合必须完全相同。

- [ ] **Step 3: 输出差异摘要**

```json
{
  "legacy_exact_event_mae_catalog": 0.0,
  "theta_only_fixed_event_mae_catalog": 0.0,
  "geometry_fixed_event_mae_catalog": 0.0,
  "metadata_disabled_event_mae_catalog": 0.0,
  "median_absolute_prediction_change_theta_only": 0.0,
  "median_absolute_prediction_change_geometry_fixed": 0.0
}
```

- [ ] **Step 4: 运行脚本**

```bash
.venv/bin/python scripts/evaluation/diagnose_legacy_metadata.py \
  --model-dir outputs_experiments/e1_4/models/lm010_noint \
  --output-dir outputs_v2/diagnostics/legacy_metadata
```

- [ ] **Step 5: 提交脚本，不提交大型结果文件**

```bash
git add scripts/evaluation/diagnose_legacy_metadata.py tests/test_legacy_metadata_diagnostic.py
git commit -m "analysis: quantify legacy unseen metadata mismatch"
```

---

### Task 18: 执行受控的修正实验矩阵

**Files:**
- Create: `configs/experiments_v2/`
- Create: `scripts/experiments/run_corrected_matrix.py`
- Create: `tests/test_corrected_experiment_matrix.py`
- Output only: `outputs_experiments_v2/`

- [ ] **Step 1: 固定随机种子和实验集合**

统一种子：

```python
SEEDS = [17, 42, 73]
```

首轮只运行以下 8 个配置，禁止恢复旧 78 组全扫描：

| ID | 改动 | 目的 |
|---|---|---|
| V2-BASE | far P+S，gain=1，meta on，source distance | 修正主模型 |
| V2-FULL | 中场+远场 | 检查中场项 |
| V2-NOSYNTH | `lambda_synth=0` | 正演项价值 |
| V2-NOSTF | `lambda_MSE=lambda_shape=0` | STF 监督价值 |
| V2-NOMETA | `use_meta=false` | 几何捷径与元数据价值 |
| V2-GAIN144 | `amplitude_gain=1.44` | 固定增益敏感性 |
| V2-DELTA-META | metadata 第一维用 `log Delta`，正演仍用 R | 网络距离定义敏感性 |
| V2-CATALOG-SCALED-STF | 每事件 STF 按目录矩统一缩放 | 目标来源敏感性 |

- [ ] **Step 2: 分三层运行，前一层失败不得进入下一层**

**Layer A — smoke:**

```bash
.venv/bin/python scripts/experiments/run_corrected_matrix.py --config-dir configs/experiments_v2 --mode smoke --epochs 2 --max-events 3
```

验收：无 NaN/Inf；每个配置可保存并重载 checkpoint；所有损失可反向传播。

**Layer B — development validation:**

```bash
.venv/bin/python scripts/experiments/run_corrected_matrix.py --config-dir configs/experiments_v2 --mode external-validation --seeds 17 42 73
```

验收：八事件按目录 Mw 计算事件级指标，cm0/cm1/cm2 分开报告；模型选择只使用预注册的 `cm2 event_mae_catalog`，平局时依次比较绝对 bias、参数量。

**Layer C — final cross-event evaluation:**

```bash
.venv/bin/python scripts/experiments/loeo_cv.py --config configs/experiments_v2/V2-SELECTED.yaml --output-root outputs_experiments_v2/loeo_selected
```

验收：每 fold 的留出事件不出现在 train/val；主汇总为事件级目录 Mw MAE。

- [ ] **Step 3: 同事件台站插值只作为补充运行**

```bash
.venv/bin/python scripts/experiments/run_corrected_matrix.py --config configs/experiments_v2/V2-SELECTED.yaml --mode within-event-station --seeds 17 42 73
```

结果名称必须为 `within_event_station_*`，不得再用无修饰的 `test_mae`。

- [ ] **Step 4: 定义选模与报告规则**

1. 八事件仅用于选模和开发期外部验证。
2. 33-fold LOEO 是主要跨事件结果。
3. 三个 seed 分别报告，不得只保留最好 seed。
4. 事件 bootstrap 以 fold/event 为单位。
5. 不要求修正结果复制 0.079；任何数据不变量失败均优先于数值性能。

- [ ] **Step 5: 运行矩阵结构测试**

```bash
.venv/bin/python -m pytest tests/test_corrected_experiment_matrix.py -v
```

测试必须确认 8 个 ID 唯一、每个配置 `pipeline_version=2`、没有 forbidden key、每个配置只改变表中声明的因素。

- [ ] **Step 6: 提交实验定义**

```bash
git add configs/experiments_v2 scripts/experiments/run_corrected_matrix.py tests/test_corrected_experiment_matrix.py
git commit -m "exp: define controlled corrected pipeline matrix"
```

---

### Task 19: 重跑所有依赖外部样本构建器的鲁棒性与图件

**Files:**
- Modify: `scripts/robustness/latency_analysis.py`
- Modify: `scripts/robustness/noise_robustness.py`
- Modify: `scripts/robustness/station_dropout.py`
- Modify: `scripts/plotting/fig2_real_event_panels.py`
- Modify: `scripts/plotting/plot_unseen_scatter.py`
- Modify: `scripts/plotting/plot_pgd_comparison.py`
- Modify: `scripts/plotting/plot_bootstrap_ci.py`

- [ ] **Step 1: 使用选定第二版 checkpoint 重新生成延迟实验**

预注册窗口：

```text
30, 60, 90, 120, 150, 180, 200 s after origin
```

每个窗口都保留原始 200 点输入结构，窗口外样点置零并更新 valid mask；不得重新运行不同滤波或基线。

- [ ] **Step 2: 重新生成噪声实验**

噪声标准差按每台站 pre-event/pre-P baseline RMS 的倍数定义：

```text
0.0, 0.5, 1.0, 2.0 times baseline RMS
```

每级至少运行固定 seed 17、42、73。

- [ ] **Step 3: 重新生成台站丢失实验**

事件级聚合在随机保留 `1, 2, 4, 8, 16, all` 个台站时计算；台站不足时使用全部台站并记录实际数量。

- [ ] **Step 4: 所有图件只读取第二版 CSV/JSON**

图脚本不得硬编码 `0.079`、`0.142` 或任一实验数值。增加测试：

```bash
rg "0\.079|0\.142" scripts paper/srl/figure_sources
```

Expected: 只允许 legacy audit 文档出现，生产图脚本不得匹配。

- [ ] **Step 5: 生成结果注册表并提交代码**

大型 PNG/PDF/CSV 是否提交按仓库现有策略处理；脚本、配置和小型 JSON 注册表必须提交。

```bash
git add scripts/robustness scripts/plotting
git commit -m "exp: regenerate robustness analyses on v2 samples"
```

---

### Task 20: 用结果注册表重写论文 Methods、Results 和图注

**Files:**
- Create: `paper/srl/results_registry_v2.yaml`
- Modify: `paper/srl/sections/data_methods.tex`
- Modify: `paper/srl/sections/srl_data_methods.tex`
- Modify: `paper/srl/sections/results.tex`
- Modify: `paper/srl/sections/conclusions.tex`
- Modify: `paper/srl/PINN_Magnitude_SRL.tex`
- Modify: `paper/srl/PINN_Magnitude_SRL_CN.tex`
- Modify: `paper/srl/figure_sources/fig2_method_overview.drawio`
- Modify: `paper/srl/figure_sources/fig2_method_overview.svg`
- Modify: `paper/srl/figures/fig2_method_overview.pdf`

- [ ] **Step 1: 冻结唯一数字来源**

`paper/srl/results_registry_v2.yaml` 必须由生成脚本从 `result_registry.json` 和 `dataset_summary.json` 写入，不允许人工编辑。注册表字段和校验规则固定如下：

| YAML 路径 | 来源 | 校验规则 |
|---|---|---|
| `pipeline_version` | selected run manifest | 必须严格等于整数 `2` |
| `dataset_manifest_path` | dataset audit | 必须存在且位于 `outputs_v2/audit/` |
| `dataset_manifest_sha256` | dataset audit | 必须匹配 64 位小写十六进制字符串 |
| `selected_run_manifest_path` | selected experiment | 必须存在 |
| `selected_run_manifest_sha256` | selected experiment | 必须匹配实际文件哈希 |
| `accepted_station_records` | dataset summary | 必须为正整数 |
| `accepted_events` | dataset summary | 必须等于实际有保留样本的事件数 |
| `within_event_station.event_mae_catalog_mean` | 三 seed 补充实验 | 必须为有限非负数 |
| `within_event_station.station_mae_catalog_mean` | 三 seed 补充实验 | 必须为有限非负数 |
| `external_validation_8events.role` | 固定文本 | 必须为 `development_validation` |
| `external_validation_8events.cm0_event_mae_catalog` | 八事件结果 | 必须为有限非负数 |
| `external_validation_8events.cm1_event_mae_catalog` | 八事件结果 | 必须为有限非负数 |
| `external_validation_8events.cm2_event_mae_catalog` | 八事件结果 | 必须为有限非负数 |
| `loeo.event_count` | LOEO summary | 必须等于成功完成的 fold 数，不得硬编码 33 |
| `loeo.event_mae_catalog` | LOEO summary | 必须为有限非负数 |
| `loeo.event_bias_catalog` | LOEO summary | 必须为有限数 |
| `loeo.bootstrap_ci95` | event bootstrap | 必须是两个递增的有限数 |

生成脚本对任何缺失路径、哈希不一致、非有限指标、零事件数或失败 fold 返回非零退出码。

- [ ] **Step 2: Methods 按真实实现逐条改写**

必须明确写出：

- 波形显式重采样到 1 Hz；
- 200 s 是物理时间窗；
- 基线窗口和 fallback；
- 7 taps / sixth-order / 0.1 Hz；
- 短记录的有效覆盖率规则；
- 2 cm 是处理后径向绝对峰值严格大于阈值；
- STF 保持源时间坐标、不做台站平移；
- STF 离散积分重标定与 99.5% 保留门槛；
- \(R=\sqrt{\Delta^2+h^2}\)、theta 和 geographic azimuth；
- 正演中 P/S 绝对延迟；
- `amplitude_gain` 的选定值；
- 四项损失，不含 `L_nonneg`；
- 八事件是开发期验证，LOEO 是跨事件主评估。

- [ ] **Step 3: Results 中移除或降级所有 legacy headline 数字**

若保留 0.079，只能写成：

```text
A legacy station-level pipeline produced 0.079 Mw against station-cropped STF-derived labels; this value is not used as evidence for the corrected model.
```

不得在摘要、结论或主结果表中把它作为第二版性能。

- [ ] **Step 4: 方法图显示唯一时间坐标**

图中应为：

```text
SCARDEC source STF q_e(tau) -- absolute P/S delays in forward operator --> station waveform u_es(t)
```

不得出现“STF shifted to P arrival”或 `skip_travel_delays=true`。

- [ ] **Step 5: 重新构建论文并检查引用和数字**

```bash
cd paper/srl
latexmk -pdf -interaction=nonstopmode PINN_Magnitude_SRL.tex
```

然后执行：

```bash
rg "4,?280|35 development|0\.079|skip_travel_delays|lambda_nonneg|order~6, 0\.2" paper/srl
```

Expected: 旧数字和旧术语只允许出现在明确标注的 legacy 说明中。

- [ ] **Step 6: 在干净 worktree 中合并当前主工作树的论文修改**

由于原始工作树已有未提交论文改动，先导出 patch 或手工三方比较，再把第二版 Methods/Results 改动合入；不得直接覆盖用户现有修改。

- [ ] **Step 7: 提交**

```bash
git add paper/srl
git commit -m "docs: align manuscript with corrected v2 pipeline"
```

---

### Task 21: 最终全链路验证和发布门槛

**Files:**
- Modify only when a verification failure exposes a defect
- Output: `outputs_v2/verification/final_verification.json`

- [ ] **Step 1: 运行全测试集**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: 全部 PASS，无 xfail 用于掩盖第二版缺陷。

- [ ] **Step 2: 运行数据审计**

```bash
.venv/bin/python scripts/data/audit_corrected_dataset.py --config configs/config_v2.yaml --manifest outputs_v2/audit/dataset_manifest.csv --summary outputs_v2/audit/dataset_summary.json
```

必须满足：

```text
all_waveform_dt_equal_1s = true
one_stf_per_event = true
one_stf_mw_per_event = true
min_stf_retained_fraction >= 0.995
accepted_event_count equals the number of events with accepted samples
```

- [ ] **Step 3: 搜索禁止模式**

```bash
rg "p_velocity_mps|skip_travel_delays|lambda_nonneg|batch\['dt'\]\.mean|dt_b\[0\]|phi_deg = ds_helper\.default_phi_deg" src scripts configs/config_v2.yaml configs/experiments_v2
```

Expected: 第二版生产路径零匹配；legacy 配置目录可匹配。

- [ ] **Step 4: 验证所有划分无事件重叠**

对每个 `split.json` 执行自动检查，任一事件同时出现于 train/validation/test 时失败。

- [ ] **Step 5: 验证结果表目标语义**

所有第二版 station/event CSV 必须同时包含 `mw_catalog` 和 `mw_stf_native`，且不得包含无修饰的 `mw_true`。

- [ ] **Step 6: 验证同一事件标签不随台站变化**

对 dataset manifest 和评估 CSV 分组：

```python
assert manifest.groupby("event")["mw_stf_native"].nunique().max() == 1
assert station_results.groupby("event")["mw_catalog"].nunique().max() == 1
assert station_results.groupby("event")["mw_stf_native"].nunique().max() == 1
```

- [ ] **Step 7: 验证可复现性**

同一 seed 对 smoke 配置运行两次，要求：

- split hash 相同；
- dataset manifest hash 相同；
- 初始 epoch 的 batch 顺序相同；
- CPU 环境下预测与指标逐元素一致；MPS/CUDA 环境允许 `1e-5` 数值容差。

- [ ] **Step 8: 写入最终验证记录**

`final_verification.json` 必须列出每条检查的命令、退出码、产物路径和 SHA-256。

- [ ] **Step 9: 最终提交**

```bash
git add outputs_v2/verification/final_verification.json
git commit -m "chore: record v2 pipeline verification evidence"
```

---

## 必须重算的结果范围

修正后以下结果一律作废并重新生成：

1. 标准 station-record split 的预测 CSV、MAE、散点图和时间演化图；
2. 八个外部事件的所有台站级与事件级预测；
3. cm0/cm1/cm2 阈值比较；
4. 延迟窗口实验；
5. 噪声鲁棒性；
6. 台站丢失实验；
7. 使用 `_station_sample_from_bundle()` 的波形/物理图；
8. 33-fold LOEO；
9. PGD 标度律比较；
10. 物理项、backbone、lambda 和辐射模式的论文主表/主图。

以下内容不要求因自身原因重训，但需要清理或重新解释：

- 删除 `L_nonneg`；
- FIR 阶数/抽头术语；
- 35 候选事件与 33 实际事件的计数说明；
- 1.44 固定增益的披露；
- within-event interpolation 的限定语。

## Go/No-Go 门槛

### Gate A — 允许启动训练

只有下列条件全部满足才能启动第二版完整训练：

- 数据审计退出码为 0；
- 同事件 STF 和 Mw 不随台站变化；
- 所有波形 dt 为 1 s；
- 外部与训练样本构建等价测试通过；
- metadata call-site 静态测试通过；
- 正演延迟测试和逐样本积分测试通过。

### Gate B — 允许选模型

- 8 个配置 smoke 均无 NaN/Inf；
- 三个 seed 均产生完整结果；
- 所有配置使用完全相同的事件/台站集合；
- 选模规则在运行前已写入配置矩阵；
- 八事件明确标记为 development validation。

### Gate C — 允许更新论文数字

- 选定配置和 run manifest 已冻结；
- 33-fold LOEO 完整，无静默失败 fold；
- station/event CSV 目标列无歧义；
- 事件 bootstrap 完成；
- results registry 的所有数字均能追溯到文件 hash；
- 主文中不存在从旧 CSV 手工复制的 headline 数字。

## 实施风险与控制

1. **源 STF 200 s 仍不足。** 由 Task 7 的固定扩展序列解决，不允许静默截尾。
2. **输出长度改变导致 checkpoint 不兼容。** 第二版 checkpoint 使用独立目录和 `pipeline_version=2`；旧 checkpoint 只走 legacy diagnostic 脚本。
3. **预事件数据不足导致样本减少。** manifest 明确记录 `baseline_source` 和拒绝原因；先报告变化，不降低基线样本门槛来追求旧样本数。
4. **修正后 MAE 变差。** 不回退到错误标签；先报告 catalog/STF 两套参考和各事件误差，再分析模型容量与物理项。
5. **LOEO 计算量大。** 只对预注册选定模型运行完整 LOEO，不对全部消融做 33-fold。
6. **现有论文工作树冲突。** 代码和论文更新均在独立 worktree；最终使用三方比较合并，不强制 checkout 或覆盖。
7. **幅值增益不可识别。** 主值固定 1.0，1.44 仅一项敏感性实验；不把三个常数拆成可解释物理贡献。
8. **2 cm 选择偏差。** 首轮保持阈值用于错误归因；第二轮单独比较 cm0/cm1/cm2，论文明确其条件样本含义。

## 完成定义

本整改只有在以下事实同时成立时才算完成：

- 任一事件的参考 STF、完整矩和参考 Mw 对所有台站完全相同；
- 波形与 STF 均有明确的物理时间坐标；
- P/S 时延只在正演中施加一次；
- 训练、标准评估、LOEO、八事件和鲁棒性脚本使用同一几何与预处理函数；
- 主评估 train/validation/test 在事件层面不重叠；
- 事件级目录 Mw MAE 是主指标，台站级指标为补充；
- 每个论文数字能定位到 checkpoint、配置、数据 manifest、split 和代码 commit；
- 论文不再把 legacy 0.079 或八事件开发期结果表述为无偏新事件测试性能。
