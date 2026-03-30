import torch
import torch.nn as nn
from typing import Optional

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
        training_cfg = (config.get('training', {}) or {})
        ds_cfg = (config.get('dataset', {}) or {})
        rate_representation = str(training_cfg.get('rate_representation', 'auto')).lower()
        if rate_representation == 'auto':
            rate_representation = 'log1p' if 'stf_m_ref' in ds_cfg else 'linear'

        self.use_meta = bool(config['model'].get('use_meta', True))
        if self.use_meta:
            self.meta_embed = nn.Sequential(
                nn.Linear(5, self.hidden_dim),
                nn.GELU(),
            )

        # 嵌入层：将原始径向分量映射到隐藏维度
        self.embed = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.hidden_dim, kernel_size=7, padding=3),
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_trans_layers)
        self.positional_encoding = SinusoidalPositionalEncoding(self.hidden_dim)

        # Transformer 后添加 LayerNorm 稳定输出
        self.post_transformer_norm = nn.LayerNorm(self.hidden_dim)

        # 序列输出头：生成随时间变化的矩率序列 dot_M0(t)
        rate_head_layers = [
            nn.Linear(self.hidden_dim, max(16, self.hidden_dim // 4)),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(max(16, self.hidden_dim // 4), 1),
        ]
        if rate_representation == 'linear':
            rate_head_layers.append(nn.Softplus())
        elif rate_representation == 'log1p':
            rate_head_layers.append(nn.ReLU())
        self.rate_head = nn.Sequential(*rate_head_layers)

    def forward(self, x: torch.Tensor, meta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向计算：从径向波形（及可选元数据）到矩率序列。

        参数
        - x:    (B, 1, T) 归一或尺度调整后的径向分量
        - meta: (B, 5) 可选元数据 [log(dist), sin(θ), cos(θ), sin(φ), cos(φ)]

        返回
        - (B, T) 预测矩率序列（log1p 或 linear 编码）
        """
        # 特征嵌入
        feat = self.embed(x)                 # (B, C, T)
        for block in self.tcn_blocks:
            feat = block(feat)               # (B, C, T)
        feat = self.se(feat)                 # (B, C, T)

        # 序列编码（添加位置编码）
        seq = feat.transpose(1, 2)           # (B, T, C)
        seq = self.positional_encoding(seq)  # (B, T, C)
        if self.use_meta and meta is not None:
            meta_emb = self.meta_embed(meta)
            seq = seq + meta_emb.unsqueeze(1)
        seq = self.transformer(seq)          # (B, T, C)
        seq = self.post_transformer_norm(seq)  # LayerNorm 稳定输出
        feat = seq.transpose(1, 2)           # (B, C, T)

        # 逐时刻回归矩率序列
        seq_time = feat.transpose(1, 2)      # (B, T, C)
        rate = self.rate_head(seq_time)      # (B, T, 1)
        return rate.squeeze(-1)              # (B, T)

    def debug_forward(self, x: torch.Tensor) -> dict:
        shapes = {}
        shapes['input'] = list(x.shape)
        feat = self.embed(x)
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
        seq = self.transformer(seq)
        shapes['transformer'] = list(seq.shape)
        feat = seq.transpose(1, 2)
        shapes['to_feat'] = list(feat.shape)
        seq_time = feat.transpose(1, 2)
        shapes['to_time'] = list(seq_time.shape)
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
