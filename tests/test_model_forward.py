"""
test_model_forward.py
---------------------
PINNModel 前向传播自动化测试，不依赖真实数据集或配置文件。

验证项目：
1. 输出形状严格等于 (B, T)
2. log1p 模式下输出无负值（ReLU 约束生效）
3. 传入 meta 与不传入 meta 时，输出数值不同（元数据确实影响预测）
4. loss.backward() 后所有参数梯度均不为 None（梯度流无断点）
5. 模型参数总量在预期范围内（防止配置失控导致参数爆炸）
6. use_meta=False 时模型正常前向（向后兼容）
7. transformer_num_layers 配置生效（层数变化导致参数量变化）
"""

import copy
import math
from pathlib import Path
import sys

import torch
import torch.nn as nn
import pytest
import yaml

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from src.models.model import PINNModel


# ---------------------------------------------------------------------------
# 公共 config 构造辅助函数
# ---------------------------------------------------------------------------

def _make_config(
    hidden_dim: int = 32,
    num_layers: int = 1,
    num_tcn_blocks: int = 2,
    transformer_num_layers: int = 1,
    dropout: float = 0.0,
    use_meta: bool = True,
    rate_representation: str = 'log1p',
) -> dict:
    """返回最小化的 PINNModel 配置字典。"""
    return {
        'model': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_tcn_blocks': num_tcn_blocks,
            'transformer_num_layers': transformer_num_layers,
            'dropout': dropout,
            'use_meta': use_meta,
        },
        'training': {
            'rate_representation': rate_representation,
        },
        'dataset': {
            'stf_m_ref': 1.0e18,  # 存在此 key 则 auto 模式推断为 log1p
        },
    }


def _make_meta(B: int, device: torch.device) -> torch.Tensor:
    """构造标准 meta 张量 (B, 5)：[log(dist), sin(θ), cos(θ), sin(φ), cos(φ)]"""
    dist_log = torch.full((B,), math.log(100_000.0))   # 100 km
    theta_r  = torch.full((B,), math.radians(45.0))
    phi_r    = torch.full((B,), math.radians(30.0))
    meta = torch.stack([
        dist_log,
        torch.sin(theta_r), torch.cos(theta_r),
        torch.sin(phi_r),   torch.cos(phi_r),
    ], dim=1).to(device)
    return meta


# ---------------------------------------------------------------------------
# 测试 1：输出形状
# 验证：模型输出张量的形状是否符合预期的 (批次大小, 时间步数)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,T", [(1, 200), (4, 200), (8, 50)])
def test_output_shape(B: int, T: int):
    """验证：模型输出张量的形状是否符合预期的 (批次大小, 时间步数)"""
    config = _make_config(hidden_dim=32, num_tcn_blocks=2)
    model = PINNModel(config).eval()
    x = torch.randn(B, 1, T)
    meta = _make_meta(B, torch.device('cpu'))

    with torch.no_grad():
        out = model(x, meta=meta)

    assert out.shape == (B, T), (
        f"输出形状应为 ({B}, {T})，实际为 {tuple(out.shape)}"
    )


def test_output_length_can_differ_from_input_length():
    config = yaml.safe_load(
        Path("configs/config_v2.yaml").read_text(encoding="utf-8")
    )
    config["dataset"]["stf"]["duration_sec"] = 300.0
    model = PINNModel(config).eval()
    x = torch.randn(2, 1, 200)
    meta = _make_meta(2, torch.device("cpu"))

    with torch.no_grad():
        output = model(x, meta=meta)

    assert output.shape == (2, 300)


def test_predict_heads_returns_station_stf_and_catalog_magnitude() -> None:
    config = yaml.safe_load(
        Path("configs/config_v2.yaml").read_text(encoding="utf-8")
    )
    model = PINNModel(config).eval()
    waveform = torch.randn(3, 1, 200)
    metadata = _make_meta(3, torch.device("cpu"))

    with torch.no_grad():
        prediction = model.predict_heads(waveform, meta=metadata)
        legacy_output = model(waveform, meta=metadata)

    assert prediction.stf_encoded.shape == (3, 300)
    assert prediction.catalog_mw.shape == (3,)
    assert torch.equal(prediction.stf_encoded, legacy_output)


def test_rt_model_accepts_two_channels_and_backpropagates() -> None:
    config = yaml.safe_load(
        Path("configs/config_v2.yaml").read_text(encoding="utf-8")
    )
    config["model"]["input_components"] = ["radial", "tangential"]
    model = PINNModel(config).train()
    waveform = torch.randn(2, 2, 200, requires_grad=True)

    prediction = model.predict_heads(
        waveform,
        meta=_make_meta(2, torch.device("cpu")),
    )
    (prediction.stf_encoded.mean() + prediction.catalog_mw.mean()).backward()

    assert model.embed[0].in_channels == 2
    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()


def test_gated_tangential_residual_starts_at_frozen_radial_prediction() -> None:
    radial_config = yaml.safe_load(
        Path("configs/config_v2.yaml").read_text(encoding="utf-8")
    )
    gated_config = copy.deepcopy(radial_config)
    gated_config["model"].update(
        {
            "input_components": ["radial", "tangential"],
            "input_fusion": "magnitude_gated_residual",
            "freeze_radial_backbone": True,
        }
    )

    torch.manual_seed(7)
    radial_model = PINNModel(radial_config).eval()
    gated_model = PINNModel(gated_config).eval()
    incompatible = gated_model.load_state_dict(
        radial_model.state_dict(),
        strict=False,
    )
    waveform_r = torch.randn(3, 1, 200)
    waveform_t = torch.randn(3, 1, 200)
    metadata = _make_meta(3, torch.device("cpu"))

    with torch.no_grad():
        radial = radial_model.predict_heads(waveform_r, meta=metadata)
        gated = gated_model.predict_heads(
            torch.cat((waveform_r, waveform_t), dim=1),
            meta=metadata,
        )

    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(key.startswith("tangential_") for key in incompatible.missing_keys)
    torch.testing.assert_close(
        gated.stf_encoded,
        radial.stf_encoded,
        rtol=0.0,
        atol=5e-7,
    )
    assert torch.equal(gated.catalog_mw, radial.catalog_mw)
    assert gated_model.tangential_gate_logit.item() == 0.0
    assert all(
        not parameter.requires_grad
        for name, parameter in gated_model.named_parameters()
        if not name.startswith("tangential_")
    )

    gated_model.train()
    with torch.no_grad():
        gated_model.tangential_magnitude_head[-1].bias.fill_(1.0)
    prediction = gated_model.predict_heads(
        torch.cat((waveform_r, waveform_t), dim=1),
        meta=metadata,
    )
    prediction.catalog_mw.sum().backward()

    gate_gradient = gated_model.tangential_gate_logit.grad
    assert gate_gradient is not None
    assert torch.isfinite(gate_gradient)
    assert gate_gradient.abs() > 0.0


def test_radial_checkpoint_is_strictly_compatible_only_with_radial_config(
) -> None:
    radial_config = yaml.safe_load(
        Path("configs/config_v2.yaml").read_text(encoding="utf-8")
    )
    radial_state = PINNModel(radial_config).state_dict()
    PINNModel(radial_config).load_state_dict(radial_state, strict=True)

    rt_config = copy.deepcopy(radial_config)
    rt_config["model"]["input_components"] = ["radial", "tangential"]

    with pytest.raises(RuntimeError, match="size mismatch"):
        PINNModel(rt_config).load_state_dict(radial_state, strict=True)


def test_catalog_magnitude_head_bias_and_gradient_flow() -> None:
    config = yaml.safe_load(
        Path("configs/config_v2.yaml").read_text(encoding="utf-8")
    )
    model = PINNModel(config).train()
    final_linear = model.magnitude_head[-1]

    assert torch.equal(final_linear.bias, torch.tensor([8.0]))

    prediction = model.predict_heads(
        torch.randn(2, 1, 200),
        meta=_make_meta(2, torch.device("cpu")),
    )
    (prediction.stf_encoded.mean() + prediction.catalog_mw.mean()).backward()

    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []


def test_legacy_model_has_no_catalog_magnitude_head() -> None:
    model = PINNModel(_make_config())

    assert model.magnitude_head is None
    with pytest.raises(RuntimeError, match="disabled"):
        model.predict_heads(torch.randn(1, 1, 20))


def test_debug_forward_uses_v2_output_length():
    config = yaml.safe_load(
        Path("configs/config_v2.yaml").read_text(encoding="utf-8")
    )
    config["dataset"]["stf"]["duration_sec"] = 300.0
    model = PINNModel(config).eval()

    with torch.no_grad():
        report = model.debug_forward(torch.randn(2, 1, 200))

    assert report["output"].shape == (2, 300)
    assert report["shapes"]["to_time"] == [2, 300, 128]


# ---------------------------------------------------------------------------
# 测试 2：log1p 模式下输出无负值
# 验证：log1p 模式下 ReLU 激活函数是否正确约束输出为非负值
# ---------------------------------------------------------------------------

def test_output_nonnegative_log1p():
    """验证：log1p 模式下 ReLU 激活函数是否正确约束输出为非负值"""
    B, T = 8, 200
    config = _make_config(rate_representation='log1p')
    model = PINNModel(config).eval()
    x = torch.randn(B, 1, T) * 2.0  # 较大方差输入，更易触发负值
    meta = _make_meta(B, torch.device('cpu'))

    with torch.no_grad():
        out = model(x, meta=meta)

    min_val = float(out.min().item())
    assert min_val >= 0.0, (
        f"log1p 模式下输出应 >= 0，实际最小值为 {min_val:.6f}"
    )


# ---------------------------------------------------------------------------
# 测试 3：linear 模式输出经 Softplus 保证正值
# 验证：linear 模式下 Softplus 激活函数是否正确保证输出为正值
# ---------------------------------------------------------------------------

def test_output_positive_linear():
    """验证：linear 模式下 Softplus 激活函数是否正确保证输出为正值"""
    B, T = 4, 100
    config = _make_config(rate_representation='linear')
    # linear 模式下 dataset 中不需要 stf_m_ref
    config['dataset'] = {}
    config['training']['rate_representation'] = 'linear'
    model = PINNModel(config).eval()
    x = torch.randn(B, 1, T) * 3.0
    meta = _make_meta(B, torch.device('cpu'))

    with torch.no_grad():
        out = model(x, meta=meta)

    min_val = float(out.min().item())
    assert min_val > 0.0, (
        f"linear 模式下 Softplus 输出应 > 0，实际最小值为 {min_val:.6f}"
    )


# ---------------------------------------------------------------------------
# 测试 4：meta 确实影响输出
# 验证：元数据（距离、方位角等）是否真正影响模型预测结果
# ---------------------------------------------------------------------------

def test_meta_influences_output():
    """验证：元数据（距离、方位角等）是否真正影响模型预测结果"""
    B, T = 4, 200
    torch.manual_seed(0)
    config = _make_config(use_meta=True)
    model = PINNModel(config).eval()

    x = torch.randn(B, 1, T)

    meta_a = _make_meta(B, torch.device('cpu'))
    # meta_b：距离差异很大（10 m 而非 100 km）
    meta_b = meta_a.clone()
    meta_b[:, 0] = math.log(10.0)

    with torch.no_grad():
        out_no_meta = model(x, meta=None)
        out_meta_a  = model(x, meta=meta_a)
        out_meta_b  = model(x, meta=meta_b)

    assert not torch.allclose(out_no_meta, out_meta_a, atol=1e-5), (
        "传入 meta 与不传 meta 时输出应不同（meta 嵌入未生效）"
    )
    assert not torch.allclose(out_meta_a, out_meta_b, atol=1e-5), (
        "不同 meta 值应产生不同输出（距离信息未被区分）"
    )


# ---------------------------------------------------------------------------
# 测试 5：梯度流——所有参数 .grad 均不为 None
# 验证：反向传播时梯度是否正确流向所有可训练参数，无梯度断点
# ---------------------------------------------------------------------------

def test_gradient_flow():
    """验证：反向传播时梯度是否正确流向所有可训练参数，无梯度断点"""
    B, T = 4, 200
    config = _make_config()
    model = PINNModel(config).train()
    x = torch.randn(B, 1, T, requires_grad=False)
    meta = _make_meta(B, torch.device('cpu'))

    out = model(x, meta=meta)         # (B, T)
    loss = out.mean()
    loss.backward()

    broken = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            broken.append(name)

    assert len(broken) == 0, (
        f"以下参数梯度为 None（梯度流断点）：\n" + "\n".join(broken)
    )


# ---------------------------------------------------------------------------
# 测试 6：参数量在合理范围内
# 验证：不同配置下模型参数总量是否在合理范围，防止配置错误导致参数爆炸
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hidden_dim,expected_min,expected_max", [
    (32,   5_000,    500_000),    # 小模型
    (128,  50_000,  5_000_000),   # 默认模型
])
def test_parameter_count_in_range(hidden_dim: int, expected_min: int, expected_max: int):
    """验证：不同配置下模型参数总量是否在合理范围，防止配置错误导致参数爆炸"""
    config = _make_config(hidden_dim=hidden_dim, num_tcn_blocks=4, transformer_num_layers=2)
    model = PINNModel(config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert expected_min <= total_params <= expected_max, (
        f"hidden_dim={hidden_dim} 时参数量 {total_params:,} "
        f"超出预期范围 [{expected_min:,}, {expected_max:,}]"
    )


# ---------------------------------------------------------------------------
# 测试 7：use_meta=False 向后兼容
# 验证：关闭元数据功能后模型是否仍能正常工作，保证向后兼容性
# ---------------------------------------------------------------------------

def test_no_meta_backward_compat():
    """验证：关闭元数据功能后模型是否仍能正常工作，保证向后兼容性"""
    B, T = 4, 200
    config = _make_config(use_meta=False)
    model = PINNModel(config).eval()
    x = torch.randn(B, 1, T)

    with torch.no_grad():
        out = model(x)   # 不传 meta

    assert out.shape == (B, T), (
        f"use_meta=False 时输出形状应为 ({B}, {T})，实际为 {tuple(out.shape)}"
    )


# ---------------------------------------------------------------------------
# 测试 8：transformer_num_layers 配置实际生效
# 验证：Transformer 层数配置是否真正影响模型结构（通过参数量变化判断）
# ---------------------------------------------------------------------------

def test_transformer_num_layers_affects_param_count():
    """验证：Transformer 层数配置是否真正影响模型结构（通过参数量变化判断）"""
    config_1 = _make_config(transformer_num_layers=1)
    config_3 = _make_config(transformer_num_layers=3)

    params_1 = sum(p.numel() for p in PINNModel(config_1).parameters())
    params_3 = sum(p.numel() for p in PINNModel(config_3).parameters())

    assert params_3 > params_1, (
        f"transformer_num_layers=3 的参数量 ({params_3}) 应大于 1 层 ({params_1})"
    )


# ---------------------------------------------------------------------------
# 测试 9：debug_forward 输出各阶段形状一致性
# 验证：调试模式下各中间层输出形状是否合法，且最终结果与正常前向传播一致
# ---------------------------------------------------------------------------

def test_debug_forward_shapes():
    """验证：调试模式下各中间层输出形状是否合法，且最终结果与正常前向传播一致"""
    B, T = 2, 200
    config = _make_config(hidden_dim=32, num_tcn_blocks=2)
    model = PINNModel(config).eval()
    x = torch.randn(B, 1, T)

    with torch.no_grad():
        report = model.debug_forward(x)
        out_debug = report['output']
        out_forward = model(x)

    assert out_debug.shape == (B, T)
    assert torch.allclose(out_debug, out_forward, atol=1e-6), (
        "debug_forward 输出与 forward 输出应完全一致"
    )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
