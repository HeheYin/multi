# models/core/embedded_dag.py
import torch
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any


class HardwareType(Enum):
    CPU = 0
    GPU = 1
    FPGA = 2
    MCU = 3


@dataclass
class TaskNode:
    """任务节点定义"""
    node_id: int
    task_type: str
    computation_cost: Dict[HardwareType, float]  # 各硬件执行时间
    data_dependencies: List[int]
    hardware_constraints: Dict[str, Any]
    deadline: float = None
    period: float = None
    energy_consumption: Dict[HardwareType, float] = None

    def __post_init__(self):
        if self.energy_consumption is None:
            self.energy_consumption = {}


@dataclass
class EmbeddedDAG:
    """嵌入式DAG任务图"""
    nodes: List[TaskNode]
    edges: List[tuple]  # (source, target, data_size)
    communication_cost: Dict[tuple, Dict[str, float]]  # 硬件间通信成本

    def __init__(self):
        self.nodes = []
        self.edges = []
        self.communication_cost = {}

    def add_node(self, node: TaskNode):
        self.nodes.append(node)

    def add_edge(self, source: int, target: int, data_size: float):
        self.edges.append((source, target, data_size))

    def add_hardware_communication_cost(self, src_hw: HardwareType,
                                        dst_hw: HardwareType,
                                        latency: float, bandwidth: float):
        self.communication_cost[(src_hw, dst_hw)] = {
            'latency': latency,
            'bandwidth': bandwidth
        }

    def get_adjacency_matrix(self) -> torch.Tensor:
        """获取邻接矩阵"""
        num_nodes = len(self.nodes)
        adj_matrix = torch.zeros((num_nodes, num_nodes))

        for src, dst, _ in self.edges:
            adj_matrix[src, dst] = 1

        return adj_matrix

    def get_node_features(self) -> torch.Tensor:
        """获取节点特征矩阵"""
        features = []
        for node in self.nodes:
            feature_vector = [
                node.hardware_constraints.get('priority', 0),
                len(node.data_dependencies),
                node.deadline or 0,
                node.period or 0,
                0.0,  # 特征5
                0.0,  # 特征6
                0.0,  # 特征7
                0.0,  # 特征8
                0.0,  # 特征9
                1.0  # 特征10
            ]
            features.append(feature_vector)

        return torch.tensor(features, dtype=torch.float32)

    def get_communication_cost(self, src_hw: HardwareType, dst_hw: HardwareType) -> Dict[str, float]:
        """获取硬件间通信成本"""
        return self.communication_cost.get((src_hw, dst_hw), {'latency': 0.0, 'bandwidth': 1e9})

    def validate_dag(self) -> bool:
        """验证DAG有效性（无环）"""
        # 简单实现：检查是否存在环（可进一步完善）
        adjacency = self.get_adjacency_matrix()
        num_nodes = len(self.nodes)

        # 检查对角线元素（自环）
        if torch.trace(adjacency) > 0:
            return False

        # 检查是否为DAG（可进一步完善）
        return True

    def get_topological_order(self) -> List[int]:
        """获取拓扑排序"""
        # 简化的拓扑排序实现
        in_degree = {node.node_id: 0 for node in self.nodes}

        # 计算入度
        for src, dst, _ in self.edges:
            in_degree[dst] += 1

        # 找到入度为0的节点
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            # 减少相邻节点的入度
            for src, dst, _ in self.edges:
                if src == node_id:
                    in_degree[dst] -= 1
                    if in_degree[dst] == 0:
                        queue.append(dst)

        return result

    def get_predecessors(self, node_id: int) -> List[int]:
        """获取节点的前驱节点"""
        return [src for src, dst, _ in self.edges if dst == node_id]

    def get_successors(self, node_id: int) -> List[int]:
        """获取节点的后继节点"""
        return [dst for src, dst, _ in self.edges if src == node_id]
