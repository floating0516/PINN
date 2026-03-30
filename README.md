# PINN

基于 GNSS 位移数据的 PINN 地震震级预测项目。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据路径

编辑 `configs/config.yaml`，至少确认这两个路径：

```yaml
dataset:
  stf_path: /path/to/STF_SCARDEC
paths:
  data_path: /path/to/gnss_events_matched.gcmt.npz
```

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

## 常用脚本

- `scripts/build_gnss_event_npz.py`：构建 GNSS 训练用 NPZ 数据
- `scripts/plot_gnss_record_section.py`：绘制事件三分量波形图
- `scripts/evaluate_pgd_scaling_laws.py`：运行 PGD-Mw 标度律对比

## 输出目录

运行结果默认保存在 `outputs/`：

- `outputs/models/`：模型权重
- `outputs/logs/`：训练日志
- `outputs/results/`：评估结果与图表
- `outputs/figures/`：波形或诊断图

## 说明

- 主配置文件：`configs/config.yaml`
- 主训练入口：`src/training/train.py`
- 主评估入口：`src/evaluation/evaluate.py`
- 当前项目包含 PGD 标度律对比脚本，可用于 Melgar、Ruhl、Crowell 三种经验关系比较
