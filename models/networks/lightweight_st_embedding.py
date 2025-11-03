# models/networks/lightweight_st_embedding.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LightweightSTEmbedding(nn.Module):
    """轻量级时空嵌入模块"""

    def __init__(self, node_feature_dim: int, hidden_dim: int):
        super(LightweightSTEmbedding, self).__init__()

        # 空间编码器 - 图注意力网络
        self.spatial_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 时序编码器 - LSTM
        self.temporal_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor,
                task_sequence: torch.Tensor) -> torch.Tensor:
        # 空间编码
        spatial_embeddings = self.spatial_encoder(node_features)

        # 应用邻接矩阵进行图卷积（简化实现）
        graph_embeddings = torch.matmul(adjacency_matrix, spatial_embeddings)

        # 时序编码
        # 注意：这里需要根据实际任务序列调整输入格式
        temporal_input = graph_embeddings.unsqueeze(0)  # 添加批次维度
        temporal_output, _ = self.temporal_encoder(temporal_input)

        # 全局特征提取
        global_features = self.global_pool(graph_embeddings.transpose(0, 1)).squeeze()

        # 结合时序和全局特征
        combined_features = torch.cat([temporal_output.squeeze(0)[-1], global_features], dim=-1)

        return combined_features


class GraphAttentionLayer(nn.Module):
    """图注意力层"""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        N = Wh.size()[0]  # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)
