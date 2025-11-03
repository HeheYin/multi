import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from models.core.embedded_dag import EmbeddedDAG, TaskNode


class EmbeddedDAGDataset(Dataset):
    """嵌入式DAG数据集"""

    def __init__(self, dag_list: List[EmbeddedDAG]):
        """
        初始化数据集

        Args:
            dag_list: DAG列表
        """
        self.dag_list = dag_list

    def __len__(self):
        return len(self.dag_list)

    def __getitem__(self, idx: int) -> EmbeddedDAG:
        return self.dag_list[idx]


class DataLoader:
    """数据加载器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据加载器

        Args:
            config: 配置参数
        """
        self.config = config
        self.batch_size = config.get('batch_size', 32)
        self.shuffle = config.get('shuffle', True)
        self.num_workers = config.get('num_workers', 0)

    def create_dataloader(self, dag_list: List[EmbeddedDAG],
                          batch_size: Optional[int] = None) -> TorchDataLoader:
        """
        创建PyTorch数据加载器

        Args:
            dag_list: DAG列表
            batch_size: 批次大小

        Returns:
            dataloader: PyTorch数据加载器
        """
        dataset = EmbeddedDAGDataset(dag_list)
        batch_size = batch_size or self.batch_size

        dataloader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )

        return dataloader

    def _collate_fn(self, batch: List[EmbeddedDAG]) -> Dict[str, Any]:
        """
        自定义批处理函数

        Args:
            batch: 批次数据

        Returns:
            batched_data: 批次化数据
        """
        batched_data = {
            'node_features': [],
            'adjacency_matrices': [],
            'task_sequences': [],
            'hardware_features': []
        }

        for dag in batch:
            batched_data['node_features'].append(dag.get_node_features())
            batched_data['adjacency_matrices'].append(dag.get_adjacency_matrix())
            # 其他特征可以根据需要添加

        return batched_data

    def load_from_file(self, file_path: str) -> List[EmbeddedDAG]:
        """
        从文件加载DAG数据

        Args:
            file_path: 文件路径

        Returns:
            dag_list: DAG列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        _, ext = os.path.splitext(file_path)

        if ext == '.json':
            return self._load_from_json(file_path)
        elif ext == '.pt' or ext == '.pth':
            return self._load_from_torch(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def _load_from_json(self, file_path: str) -> List[EmbeddedDAG]:
        """从JSON文件加载"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        dag_list = []
        for dag_data in data:
            dag = self._deserialize_dag(dag_data)
            dag_list.append(dag)

        return dag_list

    def _load_from_torch(self, file_path: str) -> List[EmbeddedDAG]:
        """从PyTorch文件加载"""
        data = torch.load(file_path)
        if isinstance(data, list):
            return data
        else:
            return [data]

    def _deserialize_dag(self, dag_data: Dict) -> EmbeddedDAG:
        """
        反序列化DAG数据

        Args:
            dag_data: DAG数据字典

        Returns:
            dag: EmbeddedDAG对象
        """
        dag = EmbeddedDAG()

        # 反序列化节点
        for node_data in dag_data['nodes']:
            node = TaskNode(
                node_id=node_data['node_id'],
                task_type=node_data['task_type'],
                computation_cost=node_data['computation_cost'],
                data_dependencies=node_data['data_dependencies'],
                hardware_constraints=node_data['hardware_constraints'],
                deadline=node_data.get('deadline'),
                period=node_data.get('period'),
                energy_consumption=node_data.get('energy_consumption', {})
            )
            dag.add_node(node)

        # 反序列化边
        for edge_data in dag_data['edges']:
            dag.add_edge(
                source=edge_data['source'],
                target=edge_data['target'],
                data_size=edge_data['data_size']
            )

        # 反序列化通信成本
        if 'communication_cost' in dag_data:
            for (src_hw, dst_hw), cost_data in dag_data['communication_cost'].items():
                dag.add_hardware_communication_cost(
                    src_hw=src_hw,
                    dst_hw=dst_hw,
                    latency=cost_data['latency'],
                    bandwidth=cost_data['bandwidth']
                )

        return dag

    def save_to_file(self, dag_list: List[EmbeddedDAG], file_path: str):
        """
        保存DAG数据到文件

        Args:
            dag_list: DAG列表
            file_path: 文件路径
        """
        _, ext = os.path.splitext(file_path)

        if ext == '.json':
            self._save_to_json(dag_list, file_path)
        elif ext == '.pt' or ext == '.pth':
            self._save_to_torch(dag_list, file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def _save_to_json(self, dag_list: List[EmbeddedDAG], file_path: str):
        """保存到JSON文件"""
        data = []
        for dag in dag_list:
            dag_data = {
                'nodes': [],
                'edges': [],
                'communication_cost': {}
            }

            # 序列化节点
            for node in dag.nodes:
                node_data = {
                    'node_id': node.node_id,
                    'task_type': node.task_type,
                    'computation_cost': node.computation_cost,
                    'data_dependencies': node.data_dependencies,
                    'hardware_constraints': node.hardware_constraints,
                    'deadline': node.deadline,
                    'period': node.period,
                    'energy_consumption': node.energy_consumption
                }
                dag_data['nodes'].append(node_data)

            # 序列化边
            for edge in dag.edges:
                edge_data = {
                    'source': edge[0],
                    'target': edge[1],
                    'data_size': edge[2]
                }
                dag_data['edges'].append(edge_data)

            # 序列化通信成本
            for (src_hw, dst_hw), cost_data in dag.communication_cost.items():
                dag_data['communication_cost'][(src_hw, dst_hw)] = cost_data

            data.append(dag_data)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_to_torch(self, dag_list: List[EmbeddedDAG], file_path: str):
        """保存到PyTorch文件"""
        torch.save(dag_list, file_path)
