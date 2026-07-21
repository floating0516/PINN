# PINN

基于 GNSS 位移数据的物理信息神经网络地震震级预测项目。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

其中 PyGMT 还需要系统安装 GMT。只进行核心训练时，可按实际用途安装所需依赖。

### 2. 配置数据路径

数据不纳入 Git。编辑 `configs/config.yaml`，至少设置以下两个服务器本地路径：

```yaml
dataset:
  stf_path: /path/to/STF_SCARDEC
paths:
  data_path: /path/to/gnss_events_matched.gcmt.npz
```

LOEO 配置及部分一次性实验脚本也可能需要单独设置外部数据路径。

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
python -m pytest tests/ -v
```

测试使用合成 fixture，不要求真实 GNSS 训练数据。

## 入口一览

| 用途 | 路径 |
|---|---|
| 主配置 | `configs/config.yaml` |
| 训练 | `src/training/train.py` |
| 评估 | `src/evaluation/evaluate.py` |
| 未见事件评估 | `src/evaluation/evaluate_unseen.py` |
| Bootstrap CI | `src/evaluation/bootstrap.py` |
| 构建训练 NPZ | `scripts/data/build_gnss_event_npz.py` |
| 批量实验 / sweep | `scripts/experiments/run_experiment.py` |
| 超参搜索 | `scripts/experiments/hyperparam_search.py` |
| LOEO 交叉验证 | `scripts/experiments/loeo_cv.py` |
| PGD 标度律对比 | `scripts/evaluation/evaluate_pgd_scaling_laws.py` |

实验配置位于 `configs/`：

- 基线、消融和课程学习：`configs/sweep_e1_*.yaml`、`configs/sweep_e2_4_curriculum.yaml`
- LOEO：`configs/config_loeo_sanity.yaml`、`configs/config_loeo_faronly_lp100.yaml`

其他脚本：

- `scripts/plotting/`：诊断图与结果图
- `scripts/robustness/`：台站丢弃、噪声和延迟鲁棒性
- `scripts/evaluation/`：PGD 与未见事件评估

## 输出目录

运行结果默认写入以下未跟踪目录：

- `outputs/models/`：模型权重
- `outputs/logs/`：训练日志
- `outputs/results/`：评估结果
- `outputs/figures/`：波形或诊断图
- `outputs_experiments/`：批量实验结果

## 说明

- 数据 NPZ、STF、模型权重和实验输出不纳入 Git，需要在服务器单独准备。
- `configs/config.yaml` 是数据、模型、损失、训练和输出路径的主控制文件。
- 多个脚本属于一次性研究工具，运行前应检查其默认路径和参数。
