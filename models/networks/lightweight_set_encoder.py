# models/networks/lightweight_set_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightSetEncoder(nn.Module):
    """轻量级集合编码器"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super(LightweightSetEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.aggregator = nn.AdaptiveAvgPool1d(1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features)
        # 集合聚合
        pooled = self.aggregator(encoded.transpose(0, 1)).squeeze()
        return pooled


class SetTransformer(nn.Module):
    """集合变换器"""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4):
        super(SetTransformer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 多头注意力
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1
        )

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, set_size, input_dim)
        batch_size, set_size, _ = x.shape

        # 输入投影
        x = self.input_projection(x)  # (batch_size, set_size, hidden_dim)

        # 多头注意力
        x = x.transpose(0, 1)  # (set_size, batch_size, hidden_dim)
        attn_output, _ = self.multihead_attention(x, x, x)
        x = x + attn_output  # 残差连接
        x = self.layer_norm1(x)

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = x + ff_output  # 残差连接
        x = self.layer_norm2(x)

        # 聚合 (取平均)
        x = x.transpose(0, 1)  # (batch_size, set_size, hidden_dim)
        x = torch.mean(x, dim=1)  # (batch_size, hidden_dim)

        return x
