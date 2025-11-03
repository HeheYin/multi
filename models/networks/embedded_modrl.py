# models/networks/embedded_modrl.py
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
        # 确保输入是3D张量 (batch_size, seq_len, input_size)
        if graph_embeddings.dim() == 2:
            # 如果是2D，添加批次维度
            temporal_input = graph_embeddings.unsqueeze(0)  # (1, num_nodes, hidden_dim)
        elif graph_embeddings.dim() == 3:
            temporal_input = graph_embeddings
        else:
            # 如果维度更高，只保留最后两维
            temporal_input = graph_embeddings.view(-1, graph_embeddings.shape[-2], graph_embeddings.shape[-1])

        temporal_output, _ = self.temporal_encoder(temporal_input)

        # 全局特征提取
        # 确保输入到global_pool的是3D张量 (batch, channels, length)
        if graph_embeddings.dim() == 2:
            # (num_nodes, hidden_dim) -> (1, hidden_dim, num_nodes)
            global_input = graph_embeddings.transpose(0, 1).unsqueeze(0)
        elif graph_embeddings.dim() == 3:
            # (batch, num_nodes, hidden_dim) -> (batch, hidden_dim, num_nodes)
            global_input = graph_embeddings.transpose(1, 2)
        else:
            # 处理高维情况，先压缩到3D
            compressed = graph_embeddings.view(-1, graph_embeddings.shape[-2], graph_embeddings.shape[-1])
            global_input = compressed.transpose(1, 2)

        # 确保global_input是3D张量
        if global_input.dim() != 3:
            # 如果仍不是3D，强制转换
            if global_input.dim() < 3:
                global_input = global_input.unsqueeze(0)
            else:
                global_input = global_input.view(-1, global_input.shape[-2], global_input.shape[-1])

        global_features = self.global_pool(global_input)

        # 结合时序和全局特征
        # 处理时序特征
        if temporal_output.dim() > 2:
            temporal_features = temporal_output[:, -1, :]  # 取最后一个时间步
        else:
            temporal_features = temporal_output[-1, :] if temporal_output.dim() == 2 else temporal_output

        # 处理全局特征，确保输出是2D
        if global_features.dim() > 2:
            global_features = global_features.squeeze()
            # 如果squeeze后仍有多维，需要进一步处理
            if global_features.dim() > 2:
                global_features = global_features.view(global_features.size(0), -1)

        # 如果global_features是1D，扩展为2D
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)

        # 如果temporal_features是1D，扩展为2D
        if temporal_features.dim() == 1:
            temporal_features = temporal_features.unsqueeze(0)

        # 确保两个特征维度匹配后再拼接
        # 取第一个批次的特征进行拼接
        if temporal_features.dim() > 2:
            temporal_features = temporal_features[0]  # 取第一个批次
        if global_features.dim() > 2:
            global_features = global_features[0]  # 取第一个批次

        # 如果仍需要扩展维度
        if temporal_features.dim() == 1:
            temporal_features = temporal_features.unsqueeze(0)
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)

        # 确保两个张量的批次维度相同
        if temporal_features.shape[0] != global_features.shape[0]:
            # 如果维度不匹配，复制较小的张量
            if temporal_features.shape[0] == 1:
                temporal_features = temporal_features.repeat(global_features.shape[0], 1)
            elif global_features.shape[0] == 1:
                global_features = global_features.repeat(temporal_features.shape[0], 1)

        combined_features = torch.cat([temporal_features, global_features], dim=-1)

        return combined_features


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


class EmbeddedMODRL(nn.Module):
    """嵌入式多目标深度强化学习模型"""

    def __init__(self, node_feature_dim: int, hardware_feature_dim: int,
                 hidden_dim: int, num_hardware: int, num_actions: int):
        super(EmbeddedMODRL, self).__init__()

        # 时空嵌入模块
        self.st_embedding = LightweightSTEmbedding(node_feature_dim, hidden_dim)

        # 硬件特征编码器
        self.hardware_encoder = LightweightSetEncoder(hardware_feature_dim, hidden_dim)

        # 状态表示维度
        state_dim = hidden_dim * 2 + hidden_dim  # 任务特征 + 硬件特征

        # 价值流 (状态价值函数)
        self.value_stream = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 优势流 (动作优势函数)
        self.advantage_stream = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor,
                task_sequence: torch.Tensor, hardware_features: torch.Tensor) -> torch.Tensor:
        # 任务特征嵌入
        task_embedding = self.st_embedding(node_features, adjacency_matrix, task_sequence)

        # 硬件特征编码
        hardware_embedding = self.hardware_encoder(hardware_features)

        # 调试信息（可选，帮助理解张量形状）
        # print(f"task_embedding shape: {task_embedding.shape}")
        # print(f"hardware_embedding shape: {hardware_embedding.shape}")

        # 确保两个张量在批次维度上匹配
        if task_embedding.dim() == 2 and hardware_embedding.dim() == 2:
            # 两个都是 (batch, features) 形状
            batch_size = max(task_embedding.shape[0], hardware_embedding.shape[0])

            # 扩展较小的批次维度
            if task_embedding.shape[0] == 1 and batch_size > 1:
                task_embedding = task_embedding.expand(batch_size, -1)
            if hardware_embedding.shape[0] == 1 and batch_size > 1:
                hardware_embedding = hardware_embedding.expand(batch_size, -1)

        elif task_embedding.dim() == 3 and hardware_embedding.dim() == 2:
            # task_embedding 是 (batch, seq, features)，hardware_embedding 是 (batch, features)
            batch_size = task_embedding.shape[0]
            if hardware_embedding.shape[0] == 1 and batch_size > 1:
                hardware_embedding = hardware_embedding.expand(batch_size, -1)
            # 添加序列维度
            hardware_embedding = hardware_embedding.unsqueeze(1).expand(-1, task_embedding.shape[1], -1)

        elif task_embedding.dim() == 2 and hardware_embedding.dim() == 3:
            # task_embedding 是 (batch, features)，hardware_embedding 是 (batch, seq, features)
            batch_size = hardware_embedding.shape[0]
            if task_embedding.shape[0] == 1 and batch_size > 1:
                task_embedding = task_embedding.expand(batch_size, -1)
            # 添加序列维度
            task_embedding = task_embedding.unsqueeze(1).expand(-1, hardware_embedding.shape[1], -1)

        # 状态表示
        # 确保张量形状兼容后再拼接
        if task_embedding.shape[:-1] == hardware_embedding.shape[:-1]:
            state_embedding = torch.cat([task_embedding, hardware_embedding], dim=-1)
        else:
            # 如果形状仍不匹配，进行展平处理
            task_flat = task_embedding.view(task_embedding.shape[0], -1)
            hardware_flat = hardware_embedding.view(hardware_embedding.shape[0], -1)

            # 确保批次维度匹配
            if task_flat.shape[0] != hardware_flat.shape[0]:
                max_batch = max(task_flat.shape[0], hardware_flat.shape[0])
                if task_flat.shape[0] == 1:
                    task_flat = task_flat.expand(max_batch, -1)
                if hardware_flat.shape[0] == 1:
                    hardware_flat = hardware_flat.expand(max_batch, -1)

            state_embedding = torch.cat([task_flat, hardware_flat], dim=-1)

        # 价值和优势计算
        value = self.value_stream(state_embedding)
        advantages = self.advantage_stream(state_embedding)

        # 组合价值和优势: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))

        return q_values


class DuelingDQN(nn.Module):
    """Dueling DQN网络架构"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DuelingDQN, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 价值流 (状态价值函数)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 优势流 (动作优势函数)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.feature_layer(state)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # 组合价值和优势: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values
