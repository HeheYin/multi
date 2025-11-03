import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .embedded_dag_generator import EmbeddedDAGGenerator
from models.core.embedded_dag import EmbeddedDAG, TaskNode, HardwareType


class RandomDAGGenerator(EmbeddedDAGGenerator):
    """随机DAG生成器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化随机DAG生成器

        Args:
            config: 配置参数
        """
        super().__init__(config)

        # 随机DAG特定参数
        self.complexity_levels = {
            'simple': {'task_range': (3, 8), 'density': 0.2},
            'medium': {'task_range': (8, 15), 'density': 0.3},
            'complex': {'task_range': (15, 25), 'density': 0.4}
        }

    def generate_random_dag(self,
                            complexity: str = 'medium',
                            custom_params: Optional[Dict[str, Any]] = None) -> EmbeddedDAG:
        """
        生成随机DAG

        Args:
            complexity: 复杂度等级 ('simple', 'medium', 'complex')
            custom_params: 自定义参数

        Returns:
            dag: 生成的随机DAG
        """
        if custom_params:
            return self._generate_with_custom_params(custom_params)

        if complexity not in self.complexity_levels:
            complexity = 'medium'

        params = self.complexity_levels[complexity]
        return self.generate(
            task_count_range=params['task_range'],
            density=params['density']
        )

    def _generate_with_custom_params(self, params: Dict[str, Any]) -> EmbeddedDAG:
        """
        使用自定义参数生成DAG

        Args:
            params: 自定义参数

        Returns:
            dag: 生成的DAG
        """
        task_count_range = params.get('task_count_range', (5, 15))
        density = params.get('density', 0.3)
        software_type = params.get('software_type')

        return self.generate(
            task_count_range=task_count_range,
            density=density,
            software_type=software_type,
            **{k: v for k, v in params.items()
               if k not in ['task_count_range', 'density', 'software_type']}
        )

    def generate_fan_in_fan_out_dag(self,
                                    levels: int = 3,
                                    nodes_per_level: int = 3) -> EmbeddedDAG:
        """
        生成扇入扇出结构的DAG

        Args:
            levels: 层数
            nodes_per_level: 每层节点数

        Returns:
            dag: 生成的扇入扇出DAG
        """
        dag = EmbeddedDAG()

        # 生成节点
        node_id = 0
        level_nodes = {}  # 记录每层的节点

        for level in range(levels):
            level_nodes[level] = []
            for _ in range(nodes_per_level):
                task_type = random.choice(self.task_types)
                computation_cost = self._generate_computation_cost(task_type)
                hardware_constraints = self._generate_hardware_constraints(task_type)
                energy_consumption = self._generate_energy_consumption(task_type)

                node = TaskNode(
                    node_id=node_id,
                    task_type=task_type,
                    computation_cost=computation_cost,
                    data_dependencies=[],
                    hardware_constraints=hardware_constraints,
                    deadline=None,
                    period=None,
                    energy_consumption=energy_consumption
                )

                dag.add_node(node)
                level_nodes[level].append(node_id)
                node_id += 1

        # 添加边：每层节点连接到下一层所有节点
        for level in range(levels - 1):
            for src_node in level_nodes[level]:
                for dst_node in level_nodes[level + 1]:
                    data_size = random.uniform(1, 50)
                    dag.add_edge(src_node, dst_node, data_size)

        # 生成通信成本
        self._generate_communication_costs(dag)

        return dag

    def generate_pipeline_dag(self,
                              stages: int = 5,
                              nodes_per_stage: int = 2) -> EmbeddedDAG:
        """
        生成流水线结构的DAG

        Args:
            stages: 流水线阶段数
            nodes_per_stage: 每阶段节点数

        Returns:
            dag: 生成的流水线DAG
        """
        dag = EmbeddedDAG()

        # 生成节点
        node_id = 0
        stage_nodes = {}  # 记录每阶段的节点

        for stage in range(stages):
            stage_nodes[stage] = []
            for _ in range(nodes_per_stage):
                task_type = random.choice(self.task_types)
                computation_cost = self._generate_computation_cost(task_type)
                hardware_constraints = self._generate_hardware_constraints(task_type)
                energy_consumption = self._generate_energy_consumption(task_type)

                node = TaskNode(
                    node_id=node_id,
                    task_type=task_type,
                    computation_cost=computation_cost,
                    data_dependencies=[],
                    hardware_constraints=hardware_constraints,
                    deadline=None,
                    period=None,
                    energy_consumption=energy_consumption
                )

                dag.add_node(node)
                stage_nodes[stage].append(node_id)
                node_id += 1

        # 添加边：流水线连接
        for stage in range(stages - 1):
            # 当前阶段的每个节点连接到下一阶段的每个节点
            for src_node in stage_nodes[stage]:
                for dst_node in stage_nodes[stage + 1]:
                    data_size = random.uniform(5, 30)
                    dag.add_edge(src_node, dst_node, data_size)

        # 生成通信成本
        self._generate_communication_costs(dag)

        return dag

    def generate_tree_dag(self,
                          max_depth: int = 4,
                          branching_factor: int = 2) -> EmbeddedDAG:
        """
        生成树形结构的DAG

        Args:
            max_depth: 最大深度
            branching_factor: 分支因子

        Returns:
            dag: 生成的树形DAG
        """
        dag = EmbeddedDAG()

        # 生成根节点
        root_task_type = random.choice(self.task_types)
        root_computation_cost = self._generate_computation_cost(root_task_type)
        root_hardware_constraints = self._generate_hardware_constraints(root_task_type)
        root_energy_consumption = self._generate_energy_consumption(root_task_type)

        root_node = TaskNode(
            node_id=0,
            task_type=root_task_type,
            computation_cost=root_computation_cost,
            data_dependencies=[],
            hardware_constraints=root_hardware_constraints,
            deadline=None,
            period=None,
            energy_consumption=root_energy_consumption
        )

        dag.add_node(root_node)

        # 递归生成子树
        self._generate_subtree(dag, 0, 1, max_depth, branching_factor, 1)

        # 生成通信成本
        self._generate_communication_costs(dag)

        return dag

    def _generate_subtree(self,
                          dag: EmbeddedDAG,
                          parent_id: int,
                          current_depth: int,
                          max_depth: int,
                          branching_factor: int,
                          next_id: int) -> int:
        """
        递归生成子树

        Args:
            dag: DAG对象
            parent_id: 父节点ID
            current_depth: 当前深度
            max_depth: 最大深度
            branching_factor: 分支因子
            next_id: 下一个节点ID

        Returns:
            next_id: 更新后的下一个节点ID
        """
        if current_depth >= max_depth:
            return next_id

        # 生成子节点
        num_children = random.randint(1, branching_factor)
        children_ids = []

        for _ in range(num_children):
            task_type = random.choice(self.task_types)
            computation_cost = self._generate_computation_cost(task_type)
            hardware_constraints = self._generate_hardware_constraints(task_type)
            energy_consumption = self._generate_energy_consumption(task_type)

            node = TaskNode(
                node_id=next_id,
                task_type=task_type,
                computation_cost=computation_cost,
                data_dependencies=[parent_id],
                hardware_constraints=hardware_constraints,
                deadline=None,
                period=None,
                energy_consumption=energy_consumption
            )

            dag.add_node(node)
            dag.add_edge(parent_id, next_id, random.uniform(1, 20))
            children_ids.append(next_id)
            next_id += 1

        # 递归生成孙子节点
        for child_id in children_ids:
            next_id = self._generate_subtree(
                dag, child_id, current_depth + 1, max_depth, branching_factor, next_id
            )

        return next_id

    def generate_mixed_pattern_dag(self) -> EmbeddedDAG:
        """
        生成混合模式的DAG

        Returns:
            dag: 生成的混合模式DAG
        """
        dag = EmbeddedDAG()

        # 生成不同模式的子DAG
        patterns = [
            self.generate_fan_in_fan_out_dag(levels=2, nodes_per_level=2),
            self.generate_pipeline_dag(stages=3, nodes_per_stage=2),
            self.generate_tree_dag(max_depth=3, branching_factor=2)
        ]

        # 合并子DAG
        node_offset = 0
        for pattern_dag in patterns:
            # 添加节点
            for node in pattern_dag.nodes:
                new_node = TaskNode(
                    node_id=node.node_id + node_offset,
                    task_type=node.task_type,
                    computation_cost=node.computation_cost,
                    data_dependencies=[dep + node_offset for dep in node.data_dependencies],
                    hardware_constraints=node.hardware_constraints,
                    deadline=node.deadline,
                    period=node.period,
                    energy_consumption=node.energy_consumption
                )
                dag.add_node(new_node)

            # 添加边
            for edge in pattern_dag.edges:
                dag.add_edge(
                    source=edge[0] + node_offset,
                    target=edge[1] + node_offset,
                    data_size=edge[2]
                )

            node_offset += len(pattern_dag.nodes)

        # 添加一些随机连接以增加复杂性
        self._add_random_connections(dag)

        # 生成通信成本
        self._generate_communication_costs(dag)

        return dag

    def _add_random_connections(self, dag: EmbeddedDAG):
        """
        添加随机连接以增加DAG复杂性

        Args:
            dag: DAG对象
        """
        node_count = len(dag.nodes)
        if node_count < 3:
            return

        # 添加随机边
        num_random_edges = random.randint(1, node_count // 2)
        for _ in range(num_random_edges):
            src = random.randint(0, node_count - 1)
            dst = random.randint(0, node_count - 1)

            # 确保不是自环且不违反DAG性质
            if src != dst:
                # 简单检查避免自环，实际应用中可能需要更复杂的环检测
                data_size = random.uniform(1, 20)
                dag.add_edge(src, dst, data_size)

    def generate_dataset_with_patterns(self,
                                       num_dags: int,
                                       pattern_weights: Optional[Dict[str, float]] = None) -> List[EmbeddedDAG]:
        """
        生成包含不同模式的DAG数据集

        Args:
            num_dags: DAG数量
            pattern_weights: 模式权重

        Returns:
            dag_list: DAG列表
        """
        if pattern_weights is None:
            pattern_weights = {
                'random': 0.3,
                'fan_in_fan_out': 0.2,
                'pipeline': 0.2,
                'tree': 0.15,
                'mixed': 0.15
            }

        patterns = list(pattern_weights.keys())
        weights = list(pattern_weights.values())

        dag_list = []
        for _ in range(num_dags):
            pattern = random.choices(patterns, weights=weights)[0]

            if pattern == 'random':
                dag = self.generate_random_dag()
            elif pattern == 'fan_in_fan_out':
                dag = self.generate_fan_in_fan_out_dag()
            elif pattern == 'pipeline':
                dag = self.generate_pipeline_dag()
            elif pattern == 'tree':
                dag = self.generate_tree_dag()
            elif pattern == 'mixed':
                dag = self.generate_mixed_pattern_dag()
            else:
                dag = self.generate_random_dag()

            dag_list.append(dag)

        return dag_list
