# USGS-Priority Magnitude Relabeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成 USGS 优先、可离线回放且保留全部标量监督的候选 NPZ，并完成冻结预测对照、seed-42 门控、三种子重训和外部复评。

**Architecture:** `src/data/magnitude_relabel.py` 负责纯解析、匹配、选择、哈希和 NPZ 不变量；三个薄命令分别负责采集发布、冻结预测重算和门控训练。USGS 原始响应、映射、manifest、候选数据和运行结果都写入唯一快照目录，现有正式数据与结果只读。

**Tech Stack:** Python 3.12 标准库、NumPy、PyTorch、PyYAML、pytest、USGS ComCat GeoJSON。

---

### Task 1: 确定性标签解析与选择

**Files:**
- Create: `src/data/magnitude_relabel.py`
- Create: `tests/test_magnitude_relabel.py`

- [ ] 先写失败测试，覆盖：Mw/Mww/Mwc/Mwr/Mwb preferred、非 Mw preferred 回退 USGS moment tensor、GCMT、SCARDEC、无来源硬失败、唯一/歧义匹配、0.1 warning、0.2 review 和同一物理事件显式双映射。

```python
resolution = resolve_magnitude(
    local_event=local_event,
    usgs_detail=detail,
    gcmt_mw=7.1,
    stf_native_mw=7.0,
)
assert (resolution.mw_selected, resolution.source_rank) == (7.2, 1)
```

- [ ] 运行并确认因模块尚不存在而失败：

```bash
source /home/lihe/.config/pinn/server.env
"$PINN_ENV/bin/python" -m pytest tests/test_magnitude_relabel.py -q
```

- [ ] 实现最小纯函数接口：`match_candidates`、`extract_usgs_magnitude`、`resolve_magnitude`、`read_native_stf`、`source_differences`；全部有限值校验，Mw-family 大小写不敏感，矩震级统一调用 `src.data.stf.moment_to_mw`。

- [ ] 运行聚焦测试与全套测试，均为 0 failure；提交：

```bash
git add src/data/magnitude_relabel.py tests/test_magnitude_relabel.py
git commit -m "feat: add deterministic magnitude label resolver"
```

### Task 2: USGS 快照、审计 manifest 与候选 NPZ

**Files:**
- Create: `scripts/data/build_usgs_magnitude_labels.py`
- Modify: `tests/test_magnitude_relabel.py`

- [ ] 先写失败测试，使用临时缓存和小型 NPZ 验证：有缓存时零网络、原始 JSON SHA-256、40 事件逐事件解析、未审阅冲突阻止发布、`magnitude == magnitude_selected`、新增九个字段，以及全部非标签数组深度相等。

- [ ] 实现命令：以 ±30 秒、100 km、±0.5 Mw 查询 ComCat；候选必须唯一；原子保存 query/detail JSON；显式映射 CSV 只修正身份；从 exact-stem STF 文件积分；以固定 ZIP 元数据确定性发布候选 NPZ。manifest 必须写入来源/等级/类型、事件与产品 ID、contributor/update time/scalar moment、原始与选中来源 hash、来源差值、warning/review disposition 和匹配证据；再次从缓存回放并核对选择、manifest 与候选 hash。

```bash
export SNAPSHOT_ID="usgs-priority-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"
export SNAPSHOT_ROOT="/home/lihe/PINN_Mag/data/magnitude-label-snapshots/$SNAPSHOT_ID"
"$PINN_ENV/bin/python" scripts/data/build_usgs_magnitude_labels.py \
  --source-npz /home/lihe/PINN_Mag/data/gnss_events_matched.gcmt.npz \
  --gcmt-csv /home/lihe/PINN_Mag/data/Global_Earthquakes_List.gcmt.csv \
  --stf-dir /home/lihe/PINN_Mag/data/STF_SCARDEC \
  --accepted-manifest /home/lihe/PINN_Mag/runs/phase8f-model-a-formal-20260723T023729Z-063e44d/preflight/dataset_manifest.csv \
  --external-event-root /home/lihe/PINN_Mag/incoming/legacy-task17/GNSS_EQDATA \
  --snapshot-root "$SNAPSHOT_ROOT"
```

- [ ] 首次下载后只为歧义、重复物理事件或 `review_required` 条目写入快照内 `usgs_event_map.v1.csv`，逐条核验时间、距离和 USGS ID，再离线重放并要求 40/40 有标量标签、31/31 正式事件有来源元数据。

- [ ] 聚焦测试和全套测试通过后提交代码；下载数据不提交 Git：

```bash
git add src/data/magnitude_relabel.py scripts/data/build_usgs_magnitude_labels.py tests/test_magnitude_relabel.py
git commit -m "feat: publish audited USGS magnitude snapshots"
```

### Task 3: 冻结预测的同行标签对照

**Files:**
- Create: `scripts/evaluation/recompute_relabel_metrics.py`
- Modify: `tests/test_magnitude_relabel.py`

- [ ] 先写失败测试：按事件严格 join 标签，重复/缺失事件硬失败；预测值和行序不变；输出 station 与 event-median 的 MAE/RMSE/bias、逐事件标签差和按来源分组指标。

- [ ] 实现命令并对三种子正式预测和外部八事件预测分别运行；输出 paired CSV/JSON 和输入 SHA-256，不改冻结 registry。

```bash
export SNAPSHOT_ROOT="$(find /home/lihe/PINN_Mag/data/magnitude-label-snapshots -mindepth 1 -maxdepth 1 -type d -name 'usgs-priority-*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
"$PINN_ENV/bin/python" scripts/evaluation/recompute_relabel_metrics.py \
  --label-manifest "$SNAPSHOT_ROOT/magnitude_labels.csv" \
  --external-label-manifest "$SNAPSHOT_ROOT/external_magnitude_labels.csv" \
  --formal-run /home/lihe/PINN_Mag/runs/phase8f-model-a-formal-20260723T023729Z-063e44d \
  --external-run /home/lihe/PINN_Mag/runs/phase8e-external-sanity-corrected-20260723T031718Z-4f6efa8 \
  --output-dir "$SNAPSHOT_ROOT/frozen-model-comparison"
```

- [ ] 核对每个 seed 的站点/事件行集合与冻结表完全一致，记录 seed-42 新标签 event MAE 作为 pilot 门槛基线；测试通过后提交。

### Task 4: Seed-42 门控重训与外部复评

**Files:**
- Create: `scripts/experiments/run_relabel_campaign.py`
- Modify: `tests/test_magnitude_relabel.py`

- [ ] 先写失败测试，证明运行配置除候选 `paths.data_path`、输出路径和 seed 外与冻结配置相同，并验证 `pilot_event_mae > frozen_new_label_mae + 0.05` 会阻止正式阶段。

- [ ] 实现可恢复编排：先生成新 dataset audit；做 CPU/CUDA finite smoke；调用现有 `run_matrix` 跑 seed 42 共 200 epoch；严格重载和验证后应用门槛；通过才顺序运行 seeds 17/42/73，再一次性评估固定八事件。

```bash
export SNAPSHOT_ROOT="$(find /home/lihe/PINN_Mag/data/magnitude-label-snapshots -mindepth 1 -maxdepth 1 -type d -name 'usgs-priority-*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
export RELABEL_RUN_ID="phase9-usgs-relabel-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"
"$PINN_ENV/bin/python" scripts/experiments/run_relabel_campaign.py \
  --snapshot-root "$SNAPSHOT_ROOT" \
  --frozen-config /home/lihe/PINN_Mag/runs/phase8f-model-a-formal-20260723T023729Z-063e44d/preflight/config_v2.yaml \
  --frozen-comparison "$SNAPSHOT_ROOT/frozen-model-comparison/summary.json" \
  --output-root "/home/lihe/PINN_Mag/runs/$RELABEL_RUN_ID" \
  --event-root /home/lihe/PINN_Mag/incoming/legacy-task17/GNSS_EQDATA \
  --epochs 200
```

- [ ] 用 `py_compile -W error`、聚焦测试、全套 316+ 新测试、快照离线回放、NPZ 不变量、checkpoint hash/strict load 和最终指标表做 fresh verification；提交代码和实验说明，运行产物保持在数据/运行根目录。
