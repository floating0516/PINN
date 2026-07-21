# PINN

基于 GNSS 位移数据的 PINN 地震震级预测项目。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据路径

从干净 clone 后，编辑 `configs/config.yaml`，至少确认这两个**本地**路径（仓库内为开发机占位绝对路径，换机必改）：

```yaml
dataset:
  stf_path: /path/to/STF_SCARDEC
paths:
  data_path: /path/to/gnss_events_matched.gcmt.npz
```

部分实验脚本（未见事件评估、LOEO 配置）里也可能写有本机路径，运行前请按需替换。

### 3. 训练

```bash
python -c "from src.training.train import train; train()"
```

### 4. 评估

```bash
python -c "from src.evaluation.evaluate import evaluate; evaluate()"
```

### 5. 运行测试

```bash
pytest tests/ -v
```

## 入口一览

| 用途 | 路径 |
|------|------|
| 主配置 | `configs/config.yaml` |
| 训练 | `src/training/train.py` |
| 评估 | `src/evaluation/evaluate.py` |
| 未见事件评估 | `src/evaluation/evaluate_unseen.py` |
| Bootstrap CI | `src/evaluation/bootstrap.py` |
| 批量实验 / sweep | `scripts/experiments/run_experiment.py` |
| 超参搜索 | `scripts/experiments/hyperparam_search.py` |
| LOEO 交叉验证 | `scripts/experiments/loeo_cv.py` |
| 构建训练 NPZ | `scripts/data/build_gnss_event_npz.py` |
| PGD 标度律对比 | `scripts/evaluation/evaluate_pgd_scaling_laws.py` |

### 实验配置

- 基线 / 消融 / 课程学习：`configs/sweep_e1_*.yaml`、`configs/sweep_e2_4_curriculum.yaml`
- Backbone / 高 λ / synth 扫描：`configs/sweep_e1_backbone.yaml`、`configs/sweep_e1_4_high_lambda.yaml`、`configs/sweep_e1_5_lambda_synth.yaml`
- LOEO：`configs/config_loeo_sanity.yaml`、`configs/config_loeo_faronly_lp100.yaml`

### 常用脚本补充

- `scripts/plotting/`：论文图与诊断图
- `scripts/robustness/`：台站丢弃 / 噪声 / 延迟鲁棒性
- `scripts/evaluation/run_unseen_eval.py`、`batch_unseen_8events.py`：未见事件批处理

## 输出目录

运行结果默认保存在 `outputs/`（已 gitignore）：

- `outputs/models/`：模型权重
- `outputs/logs/`：训练日志
- `outputs/results/`：评估结果与图表
- `outputs/figures/`：波形或诊断图

批量实验另可写入 `outputs_experiments/`（已 gitignore）。

## 说明

- 数据 NPZ、模型 `.pth`、实验运行目录不入库；换机需自备数据并改路径。
- `paper/srl/` 为 SRL 稿件源；LaTeX 中间产物与官方 Sample Files 已忽略。
- PGD 标度律对比支持 Melgar、Ruhl、Crowell 三种经验关系。
