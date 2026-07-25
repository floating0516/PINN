from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.utils.config_v2 import (
    moment_head_dropout_from_config,
    moment_linear_skip_from_config,
    radial_dynamic_range_stem_from_config,
    waveform_input_components_from_config,
)


@dataclass(frozen=True)
class PINNPrediction:
    stf_encoded: torch.Tensor
    catalog_mw: torch.Tensor


class RadialAsinhZeroConv(nn.Module):
    """Zero-initialized wide-dynamic-range residual for the radial stem."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_dim, 1, 7))

    def forward(self, radial: torch.Tensor) -> torch.Tensor:
        if radial.ndim != 3 or radial.size(1) != 1:
            raise ValueError(
                'radial asinh residual expects input shape (B, 1, T)'
            )
        compressed = torch.asinh(radial / 0.01)
        return F.conv1d(compressed, self.weight, bias=None, padding=3)


class MomentLinearSkip(nn.Module):
    """Zero-initialized, bias-free residual for the factorized log moment."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        if (
            pooled_features.ndim != 2
            or pooled_features.size(1) != self.weight.size(1)
        ):
            raise ValueError(
                'moment linear skip expects input shape (B, hidden_dim)'
            )
        return F.linear(pooled_features, self.weight, bias=None)


class PINNModel(nn.Module):
    """
    基于 1D CNN 的 PINN 主干网络：输入径向位移时间序列，输出震级预测。

    设计动机（Why）
    - 卷积在时间域具有局部模式提取能力，适合捕捉早期波形与后续释放的多尺度特征；
    - 引入多尺度膨胀卷积 + 轻量 Transformer 编码器，兼顾局部与全局依赖；
    - 使用 Squeeze-Excitation 注意力，以突出与震级相关的通道响应。
    """

    def __init__(self, config: dict):
        super(PINNModel, self).__init__()

        self.hidden_dim = config['model']['hidden_dim']
        self.dropout_p = config['model']['dropout']
        self.num_layers = config['model']['num_layers']
        training_cfg = config.get('training', {}) or {}
        ds_cfg = config.get('dataset', {}) or {}
        pipeline_version = int(config.get('pipeline_version', 1))
        if pipeline_version == 2:
            from src.utils.config_v2 import (
                stf_output_steps_from_config,
                validate_config_v2,
            )

            validate_config_v2(config)
            rate_representation = str(
                training_cfg['rate_representation']
            ).lower()
            self.output_time_steps: int | None = (
                stf_output_steps_from_config(config)
            )
        else:
            rate_representation = str(
                training_cfg.get('rate_representation', 'auto')
            ).lower()
            if rate_representation == 'auto':
                rate_representation = (
                    'log1p' if 'stf_m_ref' in ds_cfg else 'linear'
                )
            self.output_time_steps = None
        if rate_representation not in {'log1p', 'linear'}:
            raise ValueError(
                f'unsupported rate_representation: {rate_representation}'
            )

        self.stf_output_parameterization = str(
            config['model'].get('stf_output_parameterization', 'direct')
        ).lower()
        if self.stf_output_parameterization not in {
            'direct',
            'moment_shape_factorized',
        }:
            raise ValueError(
                'model.stf_output_parameterization must be direct or '
                'moment_shape_factorized'
            )
        self.use_moment_linear_skip = moment_linear_skip_from_config(config)
        self.use_moment_head_dropout = moment_head_dropout_from_config(config)
        self.factorized_source_dt_sec: float | None = None
        self.factorized_m_ref: float | None = None
        if self.stf_output_parameterization == 'moment_shape_factorized':
            if pipeline_version != 2:
                raise ValueError(
                    'moment_shape_factorized requires pipeline_version=2'
                )
            if rate_representation != 'log1p':
                raise ValueError(
                    'moment_shape_factorized requires log1p rate representation'
                )
            if bool(config['model'].get('predict_catalog_mw', False)):
                raise ValueError(
                    'moment_shape_factorized forbids an independent catalog Mw head'
                )
            self.factorized_source_dt_sec = 1.0 / float(
                config['dataset']['sample_rate_hz']
            )
            self.factorized_m_ref = float(config['dataset']['stf']['m_ref'])

        self.use_meta = bool(config['model'].get('use_meta', True))
        if self.use_meta:
            self.meta_embed = nn.Sequential(
                nn.Linear(5, self.hidden_dim),
                nn.GELU(),
            )

        # 嵌入层：将配置的波形分量映射到隐藏维度
        self.input_components = waveform_input_components_from_config(config)
        self.radial_dynamic_range_stem = (
            radial_dynamic_range_stem_from_config(config)
        )
        self.input_channels = len(self.input_components)
        self.input_fusion = str(
            config['model'].get('input_fusion', 'early')
        ).lower()
        self.gated_tangential_residual = (
            self.input_fusion == 'magnitude_gated_residual'
        )
        if self.gated_tangential_residual and self.input_components != (
            'radial',
            'tangential',
        ):
            raise ValueError(
                'magnitude_gated_residual requires radial and tangential inputs'
            )
        embed_input_channels = (
            1 if self.gated_tangential_residual else self.input_channels
        )
        self.embed = nn.Sequential(
            nn.Conv1d(in_channels=embed_input_channels, out_channels=self.hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.GroupNorm(num_groups=8, num_channels=self.hidden_dim),
        )

        # 多尺度膨胀卷积残差块（TCN）- 可配置深度
        num_tcn_blocks = config['model'].get('num_tcn_blocks', 4)
        self.tcn_blocks = nn.ModuleList([
            ResidualDilatedBlock(self.hidden_dim, dilation=2**i, dropout=self.dropout_p)
            for i in range(num_tcn_blocks)
        ])

        # 通道注意力（SE）
        self.se = SqueezeExcitation(self.hidden_dim, reduction_ratio=4)

        # 轻量 Transformer 编码器（捕捉长程依赖）
        n_trans_layers = int(config['model'].get('transformer_num_layers', self.num_layers))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=4,
            dim_feedforward=self.hidden_dim * 2,
            dropout=self.dropout_p,
            batch_first=True,
            activation='gelu',
        )
        # transformer_num_layers=0 -> Transformer branch disabled (TCN-only backbone)
        self.transformer = (
            nn.TransformerEncoder(encoder_layer, num_layers=n_trans_layers)
            if n_trans_layers > 0 else None
        )
        self.positional_encoding = SinusoidalPositionalEncoding(self.hidden_dim)

        # Transformer 后添加 LayerNorm 稳定输出
        self.post_transformer_norm = nn.LayerNorm(self.hidden_dim)

        # 序列输出头：生成随时间变化的矩率序列 dot_M0(t)
        rate_head_layers: list[nn.Module] = [
            nn.Linear(self.hidden_dim, max(16, self.hidden_dim // 4)),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(max(16, self.hidden_dim // 4), 1),
        ]
        if self.stf_output_parameterization == 'direct':
            if rate_representation == 'linear':
                rate_head_layers.append(nn.Softplus())
            elif rate_representation == 'log1p':
                rate_head_layers.append(nn.ReLU())
            self.rate_head: nn.Sequential | None = nn.Sequential(
                *rate_head_layers
            )
            self.shape_head: nn.Sequential | None = None
            self.log10_moment_head: nn.Sequential | None = None
        else:
            self.rate_head = None
            self.shape_head = nn.Sequential(*rate_head_layers)
            magnitude_hidden = max(16, self.hidden_dim // 4)
            self.log10_moment_head = nn.Sequential(
                nn.Linear(self.hidden_dim, magnitude_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(magnitude_hidden, 1),
            )
            nn.init.zeros_(self.log10_moment_head[-1].weight)
            nn.init.constant_(
                self.log10_moment_head[-1].bias,
                1.5 * 8.0 + 9.1,
            )

        if (
            self.stf_output_parameterization == 'direct'
            and bool(config['model'].get('predict_catalog_mw', False))
        ):
            magnitude_hidden = max(16, self.hidden_dim // 4)
            self.magnitude_head: nn.Sequential | None = nn.Sequential(
                nn.Linear(self.hidden_dim, magnitude_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(magnitude_hidden, 1),
            )
            nn.init.constant_(
                self.magnitude_head[-1].bias,
                float(config['model'].get('catalog_mw_initial_bias', 8.0)),
            )
        else:
            self.magnitude_head = None

        if self.gated_tangential_residual:
            if self.magnitude_head is None:
                raise ValueError(
                    'magnitude_gated_residual requires the catalog magnitude head'
                )
            magnitude_hidden = max(16, self.hidden_dim // 4)
            self.tangential_encoder = nn.Sequential(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=self.hidden_dim,
                    kernel_size=7,
                    padding=3,
                ),
                nn.GELU(),
                nn.GroupNorm(num_groups=8, num_channels=self.hidden_dim),
                ResidualDilatedBlock(
                    self.hidden_dim,
                    dilation=1,
                    dropout=self.dropout_p,
                ),
            )
            self.tangential_pool = nn.AdaptiveAvgPool1d(1)
            self.tangential_meta_embed = nn.Sequential(
                nn.Linear(5, self.hidden_dim),
                nn.GELU(),
            )
            self.tangential_magnitude_head = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, magnitude_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(magnitude_hidden, 1),
            )
            self.tangential_gate_logit = nn.Parameter(torch.zeros(()))

            if bool(config['model'].get('freeze_radial_backbone', False)):
                for name, parameter in self.named_parameters():
                    if not name.startswith('tangential_'):
                        parameter.requires_grad_(False)

        # Register zero residuals last to preserve common state and RNG streams.
        self.radial_asinh_zero_conv: RadialAsinhZeroConv | None
        if self.radial_dynamic_range_stem == 'asinh_residual':
            self.radial_asinh_zero_conv = RadialAsinhZeroConv(self.hidden_dim)
        else:
            self.radial_asinh_zero_conv = None

        self.moment_linear_skip: MomentLinearSkip | None
        if self.use_moment_linear_skip:
            self.moment_linear_skip = MomentLinearSkip(self.hidden_dim)
        else:
            self.moment_linear_skip = None

    def _embed_backbone_input(self, backbone_input: torch.Tensor) -> torch.Tensor:
        pre_activation = self.embed[0](backbone_input)
        if self.radial_asinh_zero_conv is not None:
            pre_activation = pre_activation + self.radial_asinh_zero_conv(
                backbone_input
            )
        return self.embed[2](self.embed[1](pre_activation))

    def _resize_source_time(self, seq_time: torch.Tensor) -> torch.Tensor:
        if (
            self.output_time_steps is None
            or seq_time.size(1) == self.output_time_steps
        ):
            return seq_time
        return F.interpolate(
            seq_time.transpose(1, 2),
            size=self.output_time_steps,
            mode='linear',
            align_corners=False,
        ).transpose(1, 2)

    def _encode_sequence(
        self,
        x: torch.Tensor,
        meta: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.gated_tangential_residual:
            if x.ndim != 3 or x.size(1) != 2:
                raise ValueError(
                    'magnitude_gated_residual expects input shape (B, 2, T)'
                )
            backbone_input = x[:, :1].contiguous()
        else:
            backbone_input = x
        feat = self._embed_backbone_input(backbone_input)  # (B, C, T)
        for block in self.tcn_blocks:
            feat = block(feat)               # (B, C, T)
        feat = self.se(feat)                 # (B, C, T)

        # 序列编码（添加位置编码）
        seq = feat.transpose(1, 2)           # (B, T, C)
        seq = self.positional_encoding(seq)  # (B, T, C)
        if self.use_meta and meta is not None:
            meta_emb = self.meta_embed(meta)
            seq = seq + meta_emb.unsqueeze(1)
        if self.transformer is not None:
            seq = self.transformer(seq)      # (B, T, C)
        seq = self.post_transformer_norm(seq)  # LayerNorm 稳定输出
        return seq

    def _predict_stf(self, sequence: torch.Tensor) -> torch.Tensor:
        if self.stf_output_parameterization == 'moment_shape_factorized':
            stf_encoded, _ = self._predict_factorized_stf(sequence)
            return stf_encoded
        sequence = self._resize_source_time(sequence)
        if self.rate_head is None:
            raise RuntimeError('direct STF head is unavailable')
        return self.rate_head(sequence).squeeze(-1)

    def _predict_factorized_stf(
        self,
        sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.shape_head is None
            or self.log10_moment_head is None
            or self.factorized_source_dt_sec is None
            or self.factorized_m_ref is None
        ):
            raise RuntimeError('factorized STF head is unavailable')

        shape_sequence = self._resize_source_time(sequence)
        shape_logits = self.shape_head(shape_sequence).squeeze(-1)
        positive_shape = F.softplus(shape_logits)
        normalized_shape = positive_shape / (
            positive_shape.sum(dim=1, keepdim=True)
            * self.factorized_source_dt_sec
        ).clamp_min(torch.finfo(positive_shape.dtype).tiny)

        pooled_features = sequence.mean(dim=1)
        moment_features = self.log10_moment_head[1](
            self.log10_moment_head[0](pooled_features)
        )
        # Both variants execute dropout so their training RNG streams stay aligned.
        dropped_moment_features = self.log10_moment_head[2](moment_features)
        moment_head_input = (
            dropped_moment_features
            if self.use_moment_head_dropout
            else moment_features
        )
        log10_moment = self.log10_moment_head[3](moment_head_input).squeeze(-1)
        if self.moment_linear_skip is not None:
            log10_moment = log10_moment + self.moment_linear_skip(
                pooled_features
            ).squeeze(-1)
        ln_10 = math.log(10.0)
        log_rate_over_reference = (
            log10_moment.unsqueeze(1) * ln_10
            + torch.log(
                normalized_shape.clamp_min(
                    torch.finfo(normalized_shape.dtype).tiny
                )
            )
            - math.log(self.factorized_m_ref)
        )
        stf_encoded = F.softplus(log_rate_over_reference) / ln_10
        catalog_mw = (2.0 / 3.0) * (log10_moment - 9.1)
        return stf_encoded, catalog_mw

    def _predict_tangential_magnitude_residual(
        self,
        x: torch.Tensor,
        meta: Optional[torch.Tensor],
    ) -> torch.Tensor:
        tangential = self.tangential_encoder(x[:, 1:2])
        waveform_features = self.tangential_pool(tangential).squeeze(-1)
        if meta is None:
            metadata_features = torch.zeros_like(waveform_features)
        else:
            metadata_features = self.tangential_meta_embed(meta)
        delta_mw = self.tangential_magnitude_head(
            torch.cat((waveform_features, metadata_features), dim=1)
        ).squeeze(-1)
        return torch.tanh(self.tangential_gate_logit) * delta_mw

    def predict_heads(
        self,
        x: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
    ) -> PINNPrediction:
        if self.stf_output_parameterization == 'moment_shape_factorized':
            sequence = self._encode_sequence(x, meta)
            stf_encoded, catalog_mw = self._predict_factorized_stf(sequence)
            return PINNPrediction(
                stf_encoded=stf_encoded,
                catalog_mw=catalog_mw,
            )
        if self.magnitude_head is None:
            raise RuntimeError("catalog magnitude head is disabled")
        sequence = self._encode_sequence(x, meta)
        catalog_mw = self.magnitude_head(sequence.mean(dim=1)).squeeze(-1)
        if self.gated_tangential_residual:
            catalog_mw = catalog_mw + self._predict_tangential_magnitude_residual(
                x,
                meta,
            )
        return PINNPrediction(
            stf_encoded=self._predict_stf(sequence),
            catalog_mw=catalog_mw,
        )

    def forward(self, x: torch.Tensor, meta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向计算：从径向波形（及可选元数据）到矩率序列。

        参数
        - x:    (B, 1, T) 归一或尺度调整后的径向分量
        - meta: (B, 5) 可选元数据 [log(dist), sin(θ), cos(θ), sin(φ), cos(φ)]

        返回
        - (B, T) 预测矩率序列（log1p 或 linear 编码）
        """
        return self._predict_stf(self._encode_sequence(x, meta))

    def debug_forward(self, x: torch.Tensor) -> dict:
        shapes = {}
        shapes['input'] = list(x.shape)
        backbone_input = (
            x[:, :1].contiguous() if self.gated_tangential_residual else x
        )
        feat = self._embed_backbone_input(backbone_input)
        shapes['embed'] = list(feat.shape)
        for i, block in enumerate(self.tcn_blocks):
            feat = block(feat)
            shapes[f'tcn_{i}'] = list(feat.shape)
        feat = self.se(feat)
        shapes['se'] = list(feat.shape)
        seq = feat.transpose(1, 2)
        shapes['to_seq'] = list(seq.shape)
        seq = self.positional_encoding(seq)
        shapes['pos'] = list(seq.shape)
        if self.transformer is not None:
            seq = self.transformer(seq)
        shapes['transformer'] = list(seq.shape)
        seq = self.post_transformer_norm(seq)
        feat = seq.transpose(1, 2)
        shapes['to_feat'] = list(feat.shape)
        seq_time = feat.transpose(1, 2)
        seq_time = self._resize_source_time(seq_time)
        shapes['to_time'] = list(seq_time.shape)
        if self.stf_output_parameterization == 'moment_shape_factorized':
            out, _ = self._predict_factorized_stf(seq)
            shapes['shape_head'] = [*out.shape, 1]
        else:
            if self.rate_head is None:
                raise RuntimeError('direct STF head is unavailable')
            rate = self.rate_head(seq_time)
            shapes['rate_head'] = list(rate.shape)
            out = rate.squeeze(-1)
        shapes['output'] = list(out.shape)
        return {'shapes': shapes, 'output': out}


class ResidualDilatedBlock(nn.Module):
    """
    残差膨胀卷积块：以不同膨胀率扩展感受野，增强对多时标能量释放的刻画。
    """
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=2 * dilation, dilation=2 * dilation)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act2 = nn.GELU()
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)
        
        y = self.dropout(y)
        return y + residual


class SqueezeExcitation(nn.Module):
    """
    Squeeze-Excitation 注意力：提升与震级关联的通道权重。
    """
    def __init__(self, channels: int, reduction_ratio: int = 4):
        super().__init__()
        reduced = max(8, channels // reduction_ratio)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)
        self.act = nn.ReLU()
        self.gate = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        s = x.mean(dim=2)               # (B, C)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.gate(s)                # (B, C)
        s = s.unsqueeze(-1)             # (B, C, 1)
        return x * s


class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码：为 Transformer 引入时序位置信息。
    """
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (max_len, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        t = x.size(1)
        return x + self.pe[:t, :].unsqueeze(0)

if __name__ == '__main__':
    config = {'model': {'hidden_dim': 64, 'dropout': 0.1, 'num_layers': 2}}
    model = PINNModel(config)
    B, T = 4, 250
    x = torch.randn(B, 1, T)
    report = model.debug_forward(x)
    print('Output shape:', tuple(report['output'].shape))
    for k, v in report['shapes'].items():
        print(k, tuple(v))
