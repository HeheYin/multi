import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import torch

from .base_environment import BaseSchedulingEnvironment, TaskState, HardwareState
from models.core.embedded_dag import EmbeddedDAG, TaskNode


class EmbeddedSchedulingEnvironment(BaseSchedulingEnvironment):
    """嵌入式任务调度环境"""

    def __init__(self, config: Dict):
        super().__init__(config)

        # 嵌入式特定参数
        self.energy_weight = config.get('energy_weight', 0.3)
        self.deadline_weight = config.get('deadline_weight', 0.4)
        self.makespan_weight = config.get('makespan_weight', 0.3)

        # 当前DAG任务
        self.current_dag = None
        self.task_sequence = []
        self.current_task_index = 0

        # 状态空间维度
        self.state_dim = self._calculate_state_dim()

    def reset(self, dag: Optional[EmbeddedDAG] = None) -> Tuple:
        """
        重置环境

        Args:
            dag: DAG任务图

        Returns:
            state: 初始状态
        """
        # 重置基础状态
        self.current_time = 0.0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_reward = 0.0

        # 重置硬件
        self.hardware_resources = self._initialize_hardware()

        # 重置任务队列
        self.task_queue = []
        self.running_tasks = []
        self.completed_task_list = []

        # 设置新的DAG
        if dag is not None:
            self.current_dag = dag
            self._initialize_dag_tasks(dag)
        else:
            # 如果没有提供DAG，生成一个随机DAG
            from data.datasets.embedded_dag_generator import EmbeddedDAGGenerator
            generator = EmbeddedDAGGenerator(self.config)
            self.current_dag = generator.generate()
            self._initialize_dag_tasks(self.current_dag)

        # 生成任务执行序列
        self.task_sequence = self._generate_task_sequence()
        self.current_task_index = 0

        print(f"✅ 环境重置完成，DAG包含 {len(self.task_sequence)} 个任务")

        return self.get_state()

    def _initialize_dag_tasks(self, dag: EmbeddedDAG) -> None:
        """初始化DAG任务"""
        self.task_queue = []

        for i, node in enumerate(dag.nodes):
            task = {
                'id': node.node_id,
                'node': node,
                'state': TaskState.WAITING,
                'dependencies': node.data_dependencies,
                'estimated_time': self._get_task_estimation(node),
                'executed_time': 0.0,
                'remaining_time': 0.0,
                'assigned_hardware': None,
                'start_time': None,
                'completion_time': None,
                'deadline': node.deadline,
                'priority': node.hardware_constraints.get('priority', 1.0)
            }
            task['remaining_time'] = task['estimated_time']
            self.task_queue.append(task)

    def _get_task_estimation(self, node: TaskNode) -> float:
        """获取任务执行时间估计"""
        # 使用平均执行时间作为估计
        if node.computation_cost:
            return np.mean(list(node.computation_cost.values()))
        else:
            return 10.0  # 默认执行时间

    def _generate_task_sequence(self) -> List[int]:
        """生成任务执行序列"""
        if self.current_dag is None:
            return []

        # 使用拓扑排序生成任务序列
        task_order = self._topological_sort()
        return task_order

    def _topological_sort(self) -> List[int]:
        """拓扑排序"""
        if self.current_dag is None:
            return []

        # 构建依赖图
        graph = {}
        in_degree = {}

        for node in self.current_dag.nodes:
            graph[node.node_id] = []
            in_degree[node.node_id] = 0

        for edge in self.current_dag.edges:
            src, dst, _ = edge
            graph[src].append(dst)
            in_degree[dst] = in_degree.get(dst, 0) + 1

        # 拓扑排序
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for neighbor in graph.get(node_id, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """
        执行调度动作

        Args:
            action: 选择的硬件索引

        Returns:
            state: 新状态
            reward: 奖励值
            done: 是否结束
            info: 附加信息
        """
        if self.is_done():
            return self.get_state(), 0.0, True, {}

        # 获取当前任务
        if self.current_task_index >= len(self.task_sequence):
            # 没有更多任务要调度，推进时间
            self._advance_time()
            return self.get_state(), 0.0, self.is_done(), {}

        current_task_id = self.task_sequence[self.current_task_index]
        current_task = self._get_task_by_id(current_task_id)

        if current_task is None or current_task['state'] != TaskState.WAITING:
            # 任务不可用，跳过
            self.current_task_index += 1
            return self.get_state(), -0.1, False, {'info': 'Task not available'}

        # 检查依赖是否满足
        if not self._check_task_dependencies(current_task):
            # 依赖未满足，跳过此任务
            self.current_task_index += 1
            return self.get_state(), -0.5, False, {'info': 'Dependencies not satisfied'}

        # 执行调度
        hardware_types = list(self.hardware_resources.keys())
        if action < 0 or action >= len(hardware_types):
            action = 0  # 默认选择第一个硬件

        selected_hardware = hardware_types[action]

        # 检查硬件是否可用
        hw_info = self.hardware_resources[selected_hardware]
        if hw_info['state'] == HardwareState.FAILED:
            reward = -1.0
            info = {'info': 'Hardware failed'}
        else:
            # 分配任务到硬件
            success = self._assign_task_to_hardware(current_task, selected_hardware)
            if success:
                reward = self._calculate_reward(current_task, selected_hardware)
                info = {'info': f'Task {current_task_id} assigned to {selected_hardware}'}
            else:
                reward = -0.5
                info = {'info': 'Assignment failed'}

        # 移动到下一个任务
        self.current_task_index += 1

        # 如果所有任务都已考虑，推进时间
        if self.current_task_index >= len(self.task_sequence):
            self._advance_time()

        # 更新总奖励
        self.total_reward += reward

        # 检查是否完成
        done = self.is_done()

        return self.get_state(), reward, done, info

    def _assign_task_to_hardware(self, task: Dict, hardware_type: str) -> bool:
        """
        分配任务到硬件

        Args:
            task: 任务信息
            hardware_type: 硬件类型

        Returns:
            success: 是否分配成功
        """
        # 检查硬件约束
        node = task['node']
        allowed_hardware = node.hardware_constraints.get('allowed_hardware', [])
        if allowed_hardware and hardware_type not in [hw.value if isinstance(hw, HardwareState) else hw for hw in
                                                      allowed_hardware]:
            return False

        # 更新任务状态
        task['state'] = TaskState.RUNNING
        task['assigned_hardware'] = hardware_type
        task['start_time'] = self.current_time

        # 更新硬件状态
        self.hardware_resources[hardware_type]['current_task'] = task
        self.hardware_resources[hardware_type]['state'] = HardwareState.BUSY

        # 从等待队列移动到运行队列
        if task in self.task_queue:
            self.task_queue.remove(task)
        self.running_tasks.append(task)

        return True

    def _calculate_reward(self, task: Dict, hardware_type: str) -> float:
        """
        计算调度奖励

        Args:
            task: 任务信息
            hardware_type: 硬件类型

        Returns:
            reward: 奖励值
        """
        reward = 0.0

        # 1. 执行时间奖励（负值，越小越好）
        estimated_time = task['estimated_time']
        hw_specs = self.hardware_resources[hardware_type]['specifications']
        compute_power = hw_specs.get('compute_power', 1.0)

        # 考虑硬件计算能力的执行时间
        effective_time = estimated_time / compute_power
        time_reward = -effective_time * self.makespan_weight
        reward += time_reward

        # 2. 能耗奖励（负值，越小越好）
        energy = self._calculate_energy_consumption(hardware_type, effective_time)
        energy_reward = -energy * self.energy_weight
        reward += energy_reward

        # 3. 截止时间奖励
        if task['deadline'] is not None:
            expected_completion = self.current_time + effective_time
            if expected_completion <= task['deadline']:
                deadline_reward = 1.0 * self.deadline_weight
            else:
                # 超时惩罚
                overtime = expected_completion - task['deadline']
                deadline_reward = -overtime * self.deadline_weight
            reward += deadline_reward

        # 4. 负载均衡奖励
        hw_utilization = self.hardware_resources[hardware_type]['utilization']
        if hw_utilization > 0.8:
            # 高负载惩罚
            load_penalty = -(hw_utilization - 0.8) * 0.5
            reward += load_penalty

        return reward

    def get_state(self) -> Tuple:
        """获取环境状态"""
        if self.current_dag is None:
            return self._get_default_state()

        # 节点特征
        node_features = self.current_dag.get_node_features()

        # 邻接矩阵
        adjacency_matrix = self.current_dag.get_adjacency_matrix()

        # 任务序列
        task_sequence = torch.tensor(self.task_sequence, dtype=torch.long)

        # 硬件特征
        hardware_features = self._get_hardware_features()

        return node_features, adjacency_matrix, task_sequence, hardware_features

    def _get_hardware_features(self) -> torch.Tensor:
        """获取硬件特征"""
        features = []
        for hw_type, hw_info in self.hardware_resources.items():
            hw_feature = [
                hw_info['utilization'],
                hw_info['energy_consumption'],
                hw_info['completed_tasks'],
                hw_info['total_working_time'],
                hw_info['specifications'].get('compute_power', 1.0),
                hw_info['specifications'].get('energy_efficiency', 0.8),
                hw_info['specifications'].get('memory', 4096),
                0.0
            ]
            features.append(hw_feature)

        return torch.tensor(features, dtype=torch.float32)

    def _get_default_state(self) -> Tuple:
        """获取默认状态"""
        # 当没有DAG时的默认状态
        node_features = torch.zeros((1, 10))  # 默认节点特征
        adjacency_matrix = torch.zeros((1, 1))  # 默认邻接矩阵
        task_sequence = torch.zeros(1, dtype=torch.long)  # 默认任务序列
        hardware_features = self._get_hardware_features()

        return node_features, adjacency_matrix, task_sequence, hardware_features

    def _calculate_state_dim(self) -> int:
        """计算状态空间维度"""
        # 简化的状态维度计算
        return 256  # 根据实际模型调整

    def render(self, mode: str = 'human') -> Optional[Any]:
        """渲染环境状态"""
        if mode == 'human':
            self._render_text()
        elif mode == 'rgb_array':
            return self._render_rgb_array()
        else:
            super().render(mode)

    def _render_text(self) -> None:
        """文本模式渲染"""
        print(f"\n=== 嵌入式调度环境状态 ===")
        print(f"当前时间: {self.current_time:.2f} ms")
        print(f"已完成任务: {self.completed_tasks}")
        print(f"运行中任务: {len(self.running_tasks)}")
        print(f"等待中任务: {len(self.task_queue)}")

        print(f"\n硬件状态:")
        for hw_type, hw_info in self.hardware_resources.items():
            state_str = hw_info['state'].name
            utilization = hw_info['utilization'] * 100
            print(f"  {hw_type}: {state_str}, 利用率: {utilization:.1f}%")

        if self.running_tasks:
            print(f"\n运行中任务:")
            for task in self.running_tasks:
                progress = (task['executed_time'] / task['estimated_time']) * 100
                print(f"  任务{task['id']}: {progress:.1f}% 完成")

    def _render_rgb_array(self) -> np.ndarray:
        """RGB数组渲染（用于可视化）"""
        # 简化的RGB渲染，实际应该生成图像
        height, width = 400, 600
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # 这里可以添加具体的渲染逻辑
        # 例如绘制甘特图、硬件状态图等

        return image

    def get_current_task(self) -> Optional[Dict]:
        """获取当前要调度的任务"""
        if self.current_task_index < len(self.task_sequence):
            task_id = self.task_sequence[self.current_task_index]
            return self._get_task_by_id(task_id)
        return None

    def get_hardware_utilization(self) -> Dict[str, float]:
        """获取硬件利用率"""
        utilization = {}
        for hw_type, hw_info in self.hardware_resources.items():
            utilization[hw_type] = hw_info['utilization']
        return utilization