# models/networks/multi_objective_reward.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple


class MultiObjectiveReward(nn.Module):
    """多目标奖励函数"""

    def __init__(self, weights: Dict[str, float] = None):
        super(MultiObjectiveReward, self).__init__()

        if weights is None:
            weights = {
                'makespan': 0.3,
                'energy': 0.3,
                'load_balance': 0.2,
                'deadline': 0.2
            }

        self.weights = weights

    def forward(self, metrics: Dict[str, float]) -> float:
        """
        计算多目标奖励值

        Args:
            metrics: 性能指标字典

        Returns:
            reward: 奖励值
        """
        reward = 0.0

        # Makespan奖励 (越小越好)
        if 'makespan' in metrics:
            reward -= self.weights['makespan'] * metrics['makespan'] / 1000.0

        # 能耗奖励 (越小越好)
        if 'energy' in metrics:
            reward -= self.weights['energy'] * metrics['energy'] / 100.0

        # 负载均衡奖励 (越大越好)
        if 'load_balance' in metrics:
            reward += self.weights['load_balance'] * metrics['load_balance']

        # 截止时间满足奖励 (越大越好)
        if 'deadline_satisfaction' in metrics:
            reward += self.weights['deadline'] * metrics['deadline_satisfaction']

        return reward


class RewardShaping(nn.Module):
    """奖励塑形"""

    def __init__(self):
        super(RewardShaping, self).__init__()
        self.gamma = 0.99  # 折扣因子

    def potential_based_reward_shaping(self,
                                       state_potential: float,
                                       next_state_potential: float,
                                       immediate_reward: float) -> float:
        """
        基于势能的奖励塑形

        Args:
            state_potential: 当前状态势能
            next_state_potential: 下一状态势能
            immediate_reward: 即时奖励

        Returns:
            shaped_reward: 塑形后的奖励
        """
        potential_difference = next_state_potential - state_potential
        shaped_reward = immediate_reward + self.gamma * potential_difference
        return shaped_reward

    def compute_potential(self, metrics: Dict[str, float]) -> float:
        """
        计算状态势能

        Args:
            metrics: 性能指标

        Returns:
            potential: 状态势能
        """
        potential = 0.0

        # 完成时间势能
        if 'makespan' in metrics:
            potential -= metrics['makespan'] / 1000.0

        # 能耗势能
        if 'energy' in metrics:
            potential -= metrics['energy'] / 100.0

        # 负载均衡势能
        if 'load_balance' in metrics:
            potential += metrics['load_balance']

        return potential
