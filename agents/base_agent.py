import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import random


class BaseAgent(ABC):
    """强化学习智能体基类"""

    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        """
        初始化基类智能体

        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            config: 训练配置参数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # 训练参数
        self.batch_size = config.get('batch_size', 32)
        self.gamma = config.get('gamma', 0.99)  # 折扣因子
        self.lr =  float(config.get('learning_rate', 1e-4))   # 学习率
        self.tau = config.get('tau', 0.005)  # 目标网络软更新参数

        # 探索参数
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)

        # 训练状态
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0
        self.losses = []

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def act(self, state: Any, training: bool = True) -> int:
        """
        根据状态选择动作

        Args:
            state: 当前状态
            training: 是否为训练模式

        Returns:
            action: 选择的动作
        """
        pass

    @abstractmethod
    def remember(self, state: Any, action: int, reward: float,
                 next_state: Any, done: bool) -> None:
        """
        存储经验到回放缓冲区

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        pass

    @abstractmethod
    def replay(self) -> Optional[float]:
        """
        从经验回放中学习

        Returns:
            loss: 损失值，如果没有学习则返回None
        """
        pass

    @abstractmethod
    def update_target_network(self) -> None:
        """更新目标网络"""
        pass

    def update_epsilon(self) -> None:
        """更新探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_checkpoint(self, filepath: str) -> None:
        """
        保存智能体状态

        Args:
            filepath: 保存路径
        """
        checkpoint = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'total_reward': self.total_reward,
            'losses': self.losses
        }
        torch.save(checkpoint, filepath)
        print(f"✅ 智能体状态已保存: {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """
        加载智能体状态

        Args:
            filepath: 加载路径
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.step_count = checkpoint.get('step_count', 0)
            self.episode_count = checkpoint.get('episode_count', 0)
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.total_reward = checkpoint.get('total_reward', 0)
            self.losses = checkpoint.get('losses', [])
            print(f"✅ 智能体状态已加载: {filepath}")
        except FileNotFoundError:
            print(f"⚠️ 检查点文件不存在: {filepath}")

    def get_training_info(self) -> Dict[str, Any]:
        """获取训练信息"""
        return {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'total_reward': self.total_reward,
            'average_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'device': str(self.device)
        }

    def reset_episode(self) -> None:
        """重置回合统计"""
        self.total_reward = 0
        self.episode_count += 1

    def soft_update(self, target_net: torch.nn.Module,
                    source_net: torch.nn.Module) -> None:
        """
        软更新目标网络参数

        Args:
            target_net: 目标网络
            source_net: 源网络
        """
        for target_param, source_param in zip(target_net.parameters(),
                                              source_net.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def hard_update(self, target_net: torch.nn.Module,
                    source_net: torch.nn.Module) -> None:
        """
        硬更新目标网络参数

        Args:
            target_net: 目标网络
            source_net: 源网络
        """
        target_net.load_state_dict(source_net.state_dict())

    def preprocess_state(self, state: Any) -> torch.Tensor:
        """
        预处理状态为模型输入格式

        Args:
            state: 原始状态

        Returns:
            processed_state: 处理后的状态张量
        """
        if isinstance(state, np.ndarray):
            return torch.FloatTensor(state).unsqueeze(0).to(self.device)
        elif isinstance(state, (list, tuple)):
            return torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        elif isinstance(state, torch.Tensor):
            return state.unsqueeze(0).to(self.device)
        else:
            raise ValueError(f"不支持的state类型: {type(state)}")