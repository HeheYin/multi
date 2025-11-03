from typing import List, Tuple, Optional, Any, Dict
import random
import numpy as np
import torch
from collections import namedtuple

# 定义经验转换数据结构
Transition = namedtuple('Transition',
                        ['state', 'action', 'reward', 'next_state', 'done'])


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""

    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta: float = 0.4, beta_increment: float = 0.001):
        """
        初始化优先经验回放缓冲区

        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数 (0=均匀采样，1=完全按优先级)
            beta: 重要性采样权重参数
            beta_increment: beta的增量
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        # 存储结构
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

        # 最小优先级，避免零概率
        self.min_priority = 1e-6

    def push(self, transition: Transition) -> None:
        """
        添加经验到缓冲区

        Args:
            transition: 经验转换
        """
        # 如果是第一个经验，设置最高优先级
        if self.size == 0:
            priority = 1.0
        else:
            priority = np.max(self.priorities) if self.size > 0 else 1.0

        # 添加或替换经验
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        # 更新优先级
        self.priorities[self.position] = priority

        # 更新位置指针
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        采样批次经验

        Args:
            batch_size: 批次大小

        Returns:
            batch: 经验批次
            indices: 采样索引
            weights: 重要性采样权重
        """
        if self.size < batch_size:
            batch_size = self.size

        # 计算采样概率
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # 采样
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)

        # 获取批次
        batch = [self.buffer[i] for i in indices]

        # 计算重要性采样权重
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化

        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        更新经验的优先级

        Args:
            indices: 经验索引
            td_errors: TD误差
        """
        for idx, td_error in zip(indices, td_errors):
            # 使用TD误差的绝对值作为优先级，加上小常数避免零优先级
            priority = float(np.abs(td_error)) + self.min_priority
            self.priorities[idx] = priority

    def __len__(self) -> int:
        """返回当前缓冲区大小"""
        return self.size

    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        if self.size == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'avg_priority': 0,
                'max_priority': 0,
                'min_priority': 0
            }

        priorities = self.priorities[:self.size]
        return {
            'size': self.size,
            'capacity': self.capacity,
            'avg_priority': np.mean(priorities),
            'max_priority': np.max(priorities),
            'min_priority': np.min(priorities),
            'alpha': self.alpha,
            'beta': self.beta
        }

    def save_buffer(self, filepath: str) -> None:
        """
        保存缓冲区到文件

        Args:
            filepath: 文件路径
        """
        buffer_data = {
            'buffer': self.buffer,
            'priorities': self.priorities,
            'position': self.position,
            'size': self.size,
            'alpha': self.alpha,
            'beta': self.beta
        }
        torch.save(buffer_data, filepath)
        print(f"✅ 经验回放缓冲区已保存: {filepath}")

    def load_buffer(self, filepath: str) -> None:
        """
        从文件加载缓冲区

        Args:
            filepath: 文件路径
        """
        try:
            buffer_data = torch.load(filepath)
            self.buffer = buffer_data['buffer']
            self.priorities = buffer_data['priorities']
            self.position = buffer_data['position']
            self.size = buffer_data['size']
            self.alpha = buffer_data.get('alpha', self.alpha)
            self.beta = buffer_data.get('beta', self.beta)
            print(f"✅ 经验回放缓冲区已加载: {filepath}")
        except FileNotFoundError:
            print(f"⚠️ 缓冲区文件不存在: {filepath}")


class UniformReplayBuffer:
    """均匀经验回放缓冲区（简化版本）"""

    def __init__(self, capacity: int):
        """
        初始化均匀经验回放缓冲区

        Args:
            capacity: 缓冲区容量
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition: Transition) -> None:
        """添加经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        """均匀采样"""
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer))
        else:
            return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def get_stats(self) -> Dict[str, Any]:
        return {
            'size': len(self.buffer),
            'capacity': self.capacity
        }


class MultiStepBuffer:
    """多步学习缓冲区"""

    def __init__(self, n_step: int = 3, gamma: float = 0.99):
        """
        初始化多步缓冲区

        Args:
            n_step: 多步数
            gamma: 折扣因子
        """
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = []

    def push(self, transition: Transition) -> Optional[Transition]:
        """
        添加经验，返回完整的n步转换（如果可用）

        Args:
            transition: 单步经验

        Returns:
            n_step_transition: n步经验（如果完成），否则为None
        """
        self.buffer.append(transition)

        if len(self.buffer) < self.n_step:
            return None

        # 计算n步回报
        reward = 0
        for i in range(self.n_step):
            reward += (self.gamma ** i) * self.buffer[i].reward

        # 创建n步转换
        n_step_transition = Transition(
            state=self.buffer[0].state,
            action=self.buffer[0].action,
            reward=reward,
            next_state=self.buffer[-1].next_state,
            done=self.buffer[-1].done
        )

        # 移除最早的经验
        self.buffer.pop(0)

        return n_step_transition

    def clear(self) -> None:
        """清空缓冲区"""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class ExperienceReplayManager:
    """经验回放管理器（组合多步和优先回放）"""

    def __init__(self, capacity: int, n_step: int = 3,
                 alpha: float = 0.6, beta: float = 0.4):
        """
        初始化经验回放管理器

        Args:
            capacity: 缓冲区容量
            n_step: 多步数
            alpha: 优先级参数
            beta: 重要性采样参数
        """
        self.n_step_buffer = MultiStepBuffer(n_step)
        self.prioritized_buffer = PrioritizedReplayBuffer(capacity, alpha, beta)

    def push(self, transition: Transition) -> None:
        """
        添加经验到多步缓冲区，并将完成的n步经验添加到优先缓冲区
        """
        n_step_transition = self.n_step_buffer.push(transition)

        if n_step_transition is not None:
            self.prioritized_buffer.push(n_step_transition)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """从优先缓冲区采样"""
        return self.prioritized_buffer.sample(batch_size)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """更新优先级"""
        self.prioritized_buffer.update_priorities(indices, td_errors)

    def __len__(self) -> int:
        return len(self.prioritized_buffer)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.prioritized_buffer.get_stats()
        stats['n_step_buffer_size'] = len(self.n_step_buffer)
        stats['n_step'] = self.n_step_buffer.n_step
        return stats

    def save(self, filepath: str) -> None:
        """保存管理器状态"""
        manager_data = {
            'n_step_buffer': self.n_step_buffer.buffer,
            'prioritized_buffer': {
                'buffer': self.prioritized_buffer.buffer,
                'priorities': self.prioritized_buffer.priorities,
                'position': self.prioritized_buffer.position,
                'size': self.prioritized_buffer.size
            }
        }
        torch.save(manager_data, filepath)
        print(f"✅ 经验回放管理器已保存: {filepath}")

    def load(self, filepath: str) -> None:
        """加载管理器状态"""
        try:
            manager_data = torch.load(filepath)
            self.n_step_buffer.buffer = manager_data['n_step_buffer']

            pb_data = manager_data['prioritized_buffer']
            self.prioritized_buffer.buffer = pb_data['buffer']
            self.prioritized_buffer.priorities = pb_data['priorities']
            self.prioritized_buffer.position = pb_data['position']
            self.prioritized_buffer.size = pb_data['size']

            print(f"✅ 经验回放管理器已加载: {filepath}")
        except FileNotFoundError:
            print(f"⚠️ 管理器文件不存在: {filepath}")