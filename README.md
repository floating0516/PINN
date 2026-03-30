# 物理信息神经网络（PINN）用于地震震级预测

本项目基于 GNSS 位移数据，采用物理信息神经网络（PINN）预测矩震级（$M_w$），并通过矩率相关的物理约束提升模型一致性与泛化。

## 项目结构

```
more_eq/
├── configs/
│   └── config.yaml              # 训练配置（路径、物理常数、超参数）
├── src/                         # 核心源代码
│   ├── data/
│   │   ├── gnss_dataset_loader.py   # GNSS 原始数据加载器
│   │   └── data_loader.py          # PyTorch 数据集与预处理
│   ├── models/
│   │   └── model.py                # PINNModel（TCN + Transformer + meta embedding）
│   ├── training/
│   │   ├── train.py                # 训练入口
│   │   ├── physics.py              # 物理约束损失
│   │   └── loss_stf_rate.py        # STF 矩率损失
│   ├── evaluation/
│   │   ├── evaluate.py             # 主评估脚本
│   │   ├── evaluate_no_stf.py      # 无 STF 样本评估
│   │   └── evaluate_baseline_raw.py # 基线评估
│   ├── visualization/
│   │   └── visualize.py            # 绘图工具
│   └── baseline/
│       └── __init__.py             # 基线模型
├── scripts/                     # 可执行脚本
│   ├── build_gnss_event_npz.py     # 构建 NPZ 数据集
│   ├── hyperparam_search.py        # 超参数搜索
│   ├── sweep_baseline_params.py    # 基线参数扫描
│   ├── final_eval.py               # 最终评估
│   ├── plot_training_curves.py     # 训练曲线绘图
│   └── ...
├── tests/
│   ├── test_data_loader.py         # 数据加载测试
│   └── test_model_forward.py       # 模型前向传播测试
├── notebooks/
│   └── gnss_dataset_loader_demo.ipynb
├── outputs/                     # 运行时生成（已 gitignore）
│   ├── models/                     # 训练权重 (.pth)
│   ├── logs/                       # 训练日志 (.csv)
│   ├── results/                    # 评估结果
│   └── figures/                    # 生成图表
├── .env
├── conftest.py
└── requirements.txt
```

## 在新机器上运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据路径

编辑 `configs/config.yaml`，修改以下两个 `TODO` 项为本机实际路径：

```yaml
dataset:
  stf_path: /path/to/STF_SCARDEC        # SCARDEC STF 数据目录
paths:
  data_path: /path/to/gnss_events_matched.gcmt.npz  # NPZ 数据文件
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

## 方法说明

### 数据处理

- **输入**：来自 `gnss_events_matched.gcmt.npz` 的 GNSS 位移波形
- **特征提取**：径向分量（东/北投影）+ 垂直分量（物理约束）
- **预处理**：位移统一转换为毫米，填充/截断到 200 个时间步

### 模型架构

- **类型**：多尺度膨胀卷积 (TCN) + 轻量 Transformer + SE 注意力
- **输入**：径向位移时间序列 $(B, 1, T)$
- **输出**：矩率序列 $\dot{M}_0(t)$

### 物理约束

训练损失为复合形式：
$$ L = L_{data} + \lambda \cdot L_{physics} $$

## 配置说明

在 `configs/config.yaml` 中可调整：

- **物理常数**：`rho`, `alpha`, `beta`
- **训练参数**：`batch_size`, `learning_rate`, `epochs`
- **模型超参数**：`hidden_dim`, `num_layers`, `dropout`
