import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import random

from .base_agent import BaseAgent
from .experience_replay import PrioritizedReplayBuffer, Transition


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

    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor,
                task_sequence: torch.Tensor, hardware_features: torch.Tensor) -> torch.Tensor:
        # 任务特征嵌入
        task_embedding = self.st_embedding(node_features, adjacency_matrix, task_sequence)

        # 硬件特征编码
        hardware_embedding = self.hardware_encoder(hardware_features)

        # 确保两个嵌入具有兼容的批次维度
        # 检查并调整维度以匹配
        if task_embedding.dim() != hardware_embedding.dim():
            # 如果维度不匹配，调整较小的张量
            if task_embedding.dim() < hardware_embedding.dim():
                while task_embedding.dim() < hardware_embedding.dim():
                    task_embedding = task_embedding.unsqueeze(0)
            else:
                while hardware_embedding.dim() < task_embedding.dim():
                    hardware_embedding = hardware_embedding.unsqueeze(0)

        # 确保批次维度匹配
        if task_embedding.shape[0] != hardware_embedding.shape[0]:
            # 扩展较小的批次维度
            if task_embedding.shape[0] == 1:
                task_embedding = task_embedding.expand(hardware_embedding.shape[0], *task_embedding.shape[1:])
            elif hardware_embedding.shape[0] == 1:
                hardware_embedding = hardware_embedding.expand(task_embedding.shape[0], *hardware_embedding.shape[1:])

        # 状态表示 - 确保在正确的维度上拼接
        if task_embedding.dim() == 2 and hardware_embedding.dim() == 2:
            # 两个都是2D张量 (batch, features)
            state_embedding = torch.cat([task_embedding, hardware_embedding], dim=-1)
        elif task_embedding.dim() == 3 and hardware_embedding.dim() == 3:
            # 两个都是3D张量 (batch, seq, features)
            # 需要确保序列维度匹配或进行适当的处理
            if task_embedding.shape[1] != hardware_embedding.shape[1]:
                # 如果序列长度不匹配，可能需要对硬件嵌入进行扩展
                if hardware_embedding.shape[1] == 1:
                    hardware_embedding = hardware_embedding.expand(-1, task_embedding.shape[1], -1)

            state_embedding = torch.cat([task_embedding, hardware_embedding], dim=-1)
        else:
            # 处理其他情况，确保可以拼接
            # 简化处理：展平后拼接
            task_flat = task_embedding.view(task_embedding.shape[0], -1)
            hardware_flat = hardware_embedding.view(hardware_embedding.shape[0], -1)
            state_embedding = torch.cat([task_flat, hardware_flat], dim=-1)

        # 价值和优势计算
        value = self.value_stream(state_embedding)
        advantages = self.advantage_stream(state_embedding)

        # 组合价值和优势: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))

        return q_values


class D3QNAgent(BaseAgent):
    """Dueling Double DQN智能体"""

    def __init__(self, model: nn.Module, target_model: nn.Module, config: Dict):
        """
        初始化D3QN智能体

        Args:
            model: 评估网络
            target_model: 目标网络
            config: 训练配置
        """
        # 估计状态和动作维度
        action_dim = config.get('action_dim', self._estimate_action_dim(model))
        state_dim = self._estimate_state_dim(model)

        super().__init__(state_dim, action_dim, config)

        # 网络模型
        self.model = model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.hard_update(self.target_model, self.model)  # 初始硬更新

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 经验回放缓冲区
        self.memory = PrioritizedReplayBuffer(
            capacity=config.get('memory_size', 10000),
            alpha=config.get('alpha', 0.6),
            beta=config.get('beta', 0.4),
            beta_increment=config.get('beta_increment', 0.001)
        )

        # 训练参数
        self.target_update = config.get('target_update', 100)  # 目标网络更新频率
        self.learning_starts = config.get('learning_starts', 1000)  # 开始学习的最小经验数

        print(f"✅ D3QN智能体初始化完成")
        print(f"  设备: {self.device}")
        print(f"  状态维度: {state_dim}, 动作维度: {action_dim}")
        print(f"  记忆容量: {self.memory.capacity}")

    def _estimate_state_dim(self, model: nn.Module) -> int:
        """估计状态维度"""
        # 这里需要根据实际模型结构来估计
        # 简化实现，实际应该从模型输入维度获取
        return 256  # 假设的维度

    def _estimate_action_dim(self, model: nn.Module) -> int:
        """估计动作维度"""
        # 从模型输出层获取动作维度
        for module in model.modules():
            if isinstance(module, nn.Linear):
                return module.out_features
        return 4  # 默认值

    def act(self, state: Any, training: bool = True) -> int:
        """
        ε-greedy策略选择动作

        Args:
            state: 当前状态
            training: 是否为训练模式

        Returns:
            action: 选择的动作
        """
        self.step_count += 1

        if training and random.random() <= self.epsilon:
            # 探索: 随机选择动作
            return random.randrange(self.action_dim)
        else:
            # 利用: 选择Q值最大的动作
            with torch.no_grad():
                state_tensor = self._prepare_state(state)
                q_values = self.model(state_tensor)
                return q_values.argmax().item()

    def _prepare_state(self, state: Any) -> torch.Tensor:
        """
        准备状态输入

        Args:
            state: 原始状态

        Returns:
            state_tensor: 处理后的状态张量
        """
        if isinstance(state, tuple):
            # 处理多个输入的状态
            processed_states = []
            for s in state:
                if isinstance(s, torch.Tensor):
                    processed_states.append(s.unsqueeze(0).to(self.device))
                else:
                    processed_states.append(torch.FloatTensor(s).unsqueeze(0).to(self.device))
            return processed_states
        else:
            return self.preprocess_state(state)

    def remember(self, state: Any, action: int, reward: float,
                 next_state: Any, done: bool) -> None:
        """
        存储经验到优先经验回放缓冲区

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        transition = Transition(state, action, reward, next_state, done)
        self.memory.push(transition)

        # 更新总奖励
        self.total_reward += reward

    def replay(self) -> Optional[float]:
        """
        从经验回放中学习

        Returns:
            loss: 损失值，如果没有学习则返回None
        """
        # 检查是否有足够经验
        if len(self.memory) < self.learning_starts or len(self.memory) < self.batch_size:
            return None

        # 采样批次
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self._unpack_batch(batch)

        # 转换为张量
        states = self._batch_to_tensor(states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = self._batch_to_tensor(next_states)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # 计算当前Q值
        if isinstance(states, tuple) and len(states) == 4:
            # 对于EmbeddedMODRL模型，传入4个参数
            current_q_values = self.model(states[0], states[1], states[2], states[3]).gather(1, actions.unsqueeze(1))
        else:
            # 对于其他模型，保持原有逻辑
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值 (Double DQN)
        with torch.no_grad():
            # 使用评估网络选择动作
            if isinstance(next_states, tuple) and len(next_states) == 4:
                # 对于EmbeddedMODRL模型
                next_actions = self.model(next_states[0], next_states[1], next_states[2], next_states[3]).argmax(1,
                                                                                                                 keepdim=True)
                # 使用目标网络评估动作价值
                next_q_values = self.target_model(next_states[0], next_states[1], next_states[2],
                                                  next_states[3]).gather(1, next_actions)
            else:
                # 对于其他模型
                next_actions = self.model(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_model(next_states).gather(1, next_actions)

            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
            # 计算TD误差和损失
            td_errors = target_q_values - current_q_values
            loss = (td_errors.pow(2) * weights.unsqueeze(1)).mean()

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            # 更新优先级
            priorities = td_errors.abs().detach().cpu().numpy().flatten()
            self.memory.update_priorities(indices, priorities)

            return loss.item()

        #
    def _unpack_batch(self, batch: List[Transition]) -> Tuple:
        """解包批次数据"""
        states = [t.state for t in batch]
        actions = [t.action for t in batch]
        rewards = [t.reward for t in batch]
        next_states = [t.next_state for t in batch]
        dones = [t.done for t in batch]

        return states, actions, rewards, next_states, dones

    def _batch_to_tensor(self, batch_states: List) -> Tuple:
        """将批次状态转换为张量"""
        if isinstance(batch_states[0], tuple) and len(batch_states[0]) == 4:
            # 处理4个输入的状态 (node_features, adjacency_matrix, task_sequence, hardware_features)
            node_features_list = [state[0] for state in batch_states]
            adjacency_matrices_list = [state[1] for state in batch_states]
            task_sequences_list = [state[2] for state in batch_states]
            hardware_features_list = [state[3] for state in batch_states]

            # 转换为张量
            node_features = torch.stack(node_features_list).to(self.device)
            adjacency_matrices = torch.stack(adjacency_matrices_list).to(self.device)
            task_sequences = torch.stack(task_sequences_list).to(self.device)
            hardware_features = torch.stack(hardware_features_list).to(self.device)

            return (node_features, adjacency_matrices, task_sequences, hardware_features)
        else:
            # 单个状态输入
            if isinstance(batch_states[0], torch.Tensor):
                return torch.stack(batch_states).to(self.device)
            else:
                return torch.FloatTensor(np.array(batch_states)).to(self.device)



    def update_target_network(self) -> None:
        """软更新目标网络"""
        self.soft_update(self.target_model, self.model)

    def save_model(self, filepath: str) -> None:
        """
        保存模型和优化器状态

        Args:
            filepath: 保存路径
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        print(f"✅ 模型已保存: {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        加载模型和优化器状态

        Args:
            filepath: 加载路径
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ 模型已加载: {filepath}")
        except FileNotFoundError:
            print(f"⚠️ 模型文件不存在: {filepath}")

    def get_q_values(self, state: Any) -> np.ndarray:
        """
        获取状态的所有动作Q值

        Args:
            state: 当前状态

        Returns:
            q_values: 所有动作的Q值数组
        """
        with torch.no_grad():
            state_tensor = self._prepare_state(state)
            q_values = self.model(state_tensor)
            return q_values.cpu().numpy().flatten()

    def train_mode(self) -> None:
        """设置为训练模式"""
        self.model.train()
        self.target_model.train()

    def eval_mode(self) -> None:
        """设置为评估模式"""
        self.model.eval()
        self.target_model.eval()

    def get_memory_info(self) -> Dict[str, Any]:
        """获取记忆缓冲区信息"""
        return {
            'memory_size': len(self.memory),
            'memory_capacity': self.memory.capacity,
            'beta': self.memory.beta
        }