import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import torch
import random
from collections import defaultdict

from .embedded_scheduling_env import EmbeddedSchedulingEnvironment
from models.core.embedded_dag import EmbeddedDAG


class MultiSoftwareEnvironment(EmbeddedSchedulingEnvironment):
    """多软件并发环境"""

    def __init__(self, config: Dict):
        super().__init__(config)

        # 多软件参数
        self.max_concurrent_software = config.get('max_concurrent_software', 3)
        self.software_priorities = config.get('software_priorities', {})
        self.resource_isolation = config.get('resource_isolation', False)

        # 多软件状态
        self.software_apps = {}  # 软件应用字典
        self.software_tasks = defaultdict(list)  # 按软件分组的任务
        self.current_software_id = None

        # 资源分配
        self.hardware_allocations = {}  # 硬件资源分配
        self.software_deadlines = {}  # 软件截止时间

        # 性能指标
        self.software_metrics = defaultdict(dict)

    def reset(self, software_apps: Optional[Dict] = None) -> Tuple:
        """
        重置环境

        Args:
            software_apps: 软件应用字典 {sw_id: {'dag': DAG, 'priority': float, ...}}

        Returns:
            state: 初始状态
        """
        # 重置基础环境
        super().reset()

        # 重置多软件状态
        self.software_apps = {}
        self.software_tasks = defaultdict(list)
        self.current_software_id = None
        self.hardware_allocations = {}
        self.software_deadlines = {}
        self.software_metrics = defaultdict(dict)

        # 设置软件应用
        if software_apps is not None:
            self.software_apps = software_apps
        else:
            self.software_apps = self._generate_software_apps()

        # 初始化软件任务
        self._initialize_software_tasks()

        # 初始化资源分配
        self._initialize_resource_allocations()

        print(f"✅ 多软件环境重置完成，{len(self.software_apps)} 个软件应用")

        return self.get_state()

    def _generate_software_apps(self) -> Dict[str, Dict]:
        """生成软件应用"""
        from data.datasets.embedded_dag_generator import EmbeddedDAGGenerator
        generator = EmbeddedDAGGenerator(self.config)

        software_types = ['control_software', 'ai_inference', 'sensing_software', 'communication_software']
        software_apps = {}

        for i, sw_type in enumerate(software_types[:self.max_concurrent_software]):
            sw_id = f"sw_{i}"

            # 生成DAG
            dag = generator.generate(
                task_count_range=(5, 15),
                software_type=sw_type
            )

            # 设置软件属性
            priority = self.software_priorities.get(sw_type, 1.0)
            deadline = self.current_time + random.uniform(1000, 5000)

            software_apps[sw_id] = {
                'dag': dag,
                'type': sw_type,
                'priority': priority,
                'deadline': deadline,
                'arrival_time': self.current_time,
                'completed': False
            }

        return software_apps

    def _initialize_software_tasks(self) -> None:
        """初始化软件任务"""
        self.task_queue = []
        self.software_tasks = defaultdict(list)

        for sw_id, sw_info in self.software_apps.items():
            dag = sw_info['dag']

            for i, node in enumerate(dag.nodes):
                task = {
                    'id': f"{sw_id}_task_{node.node_id}",
                    'software_id': sw_id,
                    'node': node,
                    'state': self.TaskState.WAITING,
                    'dependencies': node.data_dependencies,
                    'estimated_time': self._get_task_estimation(node),
                    'executed_time': 0.0,
                    'remaining_time': 0.0,
                    'assigned_hardware': None,
                    'start_time': None,
                    'completion_time': None,
                    'deadline': node.deadline or sw_info['deadline'],
                    'priority': node.hardware_constraints.get('priority', 1.0) * sw_info['priority']
                }
                task['remaining_time'] = task['estimated_time']

                self.task_queue.append(task)
                self.software_tasks[sw_id].append(task)

            # 为每个软件生成任务序列
            self.software_apps[sw_id]['task_sequence'] = self._generate_software_task_sequence(sw_id)
            self.software_apps[sw_id]['current_task_index'] = 0

    def _generate_software_task_sequence(self, sw_id: str) -> List[str]:
        """生成软件任务序列"""
        tasks = self.software_tasks[sw_id]
        if not tasks:
            return []

        # 简化的任务排序：按优先级和依赖关系
        sorted_tasks = sorted(tasks, key=lambda x: x['priority'], reverse=True)
        return [task['id'] for task in sorted_tasks]

    def _initialize_resource_allocations(self) -> None:
        """初始化资源分配"""
        hardware_types = list(self.hardware_resources.keys())

        if self.resource_isolation:
            # 资源隔离模式：为每个软件分配专用硬件
            for i, sw_id in enumerate(self.software_apps.keys()):
                if i < len(hardware_types):
                    allocated_hw = hardware_types[i]
                    self.hardware_allocations[sw_id] = [allocated_hw]
                else:
                    # 硬件不足，共享剩余硬件
                    self.hardware_allocations[sw_id] = hardware_types[i % len(hardware_types):]
        else:
            # 资源共享模式：所有软件共享所有硬件
            for sw_id in self.software_apps.keys():
                self.hardware_allocations[sw_id] = hardware_types

    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """
        执行调度动作（考虑多软件优先级）

        Args:
            action: 调度动作

        Returns:
            state: 新状态
            reward: 奖励值
            done: 是否结束
            info: 附加信息
        """
        if self.is_done():
            return self.get_state(), 0.0, True, {}

        # 选择当前要调度的软件和任务
        current_software, current_task = self._select_next_task()

        if current_task is None:
            # 没有可调度任务，推进时间
            self._advance_time()
            return self.get_state(), 0.0, self.is_done(), {'info': 'No task available'}

        # 执行调度
        hardware_types = self.hardware_allocations.get(current_software, list(self.hardware_resources.keys()))
        if action < 0 or action >= len(hardware_types):
            action = 0

        selected_hardware = hardware_types[action]

        # 检查硬件是否可用且允许该软件使用
        hw_info = self.hardware_resources[selected_hardware]
        if (hw_info['state'] == self.HardwareState.FAILED or
                selected_hardware not in self.hardware_allocations[current_software]):
            reward = -1.0
            info = {'info': 'Hardware not available for this software'}
        else:
            # 分配任务到硬件
            success = self._assign_task_to_hardware(current_task, selected_hardware)
            if success:
                reward = self._calculate_multi_software_reward(current_task, selected_hardware, current_software)
                info = {'info': f'Task {current_task["id"]} assigned to {selected_hardware}'}

                # 更新软件任务索引
                sw_info = self.software_apps[current_software]
                sw_info['current_task_index'] += 1
            else:
                reward = -0.5
                info = {'info': 'Assignment failed'}

        # 推进时间（如果所有软件都没有可调度任务）
        if self._no_tasks_available():
            self._advance_time()

        self.total_reward += reward
        done = self.is_done()

        return self.get_state(), reward, done, info

    def _select_next_task(self) -> Tuple[Optional[str], Optional[Dict]]:
        """选择下一个要调度的任务（考虑软件优先级）"""
        available_tasks = []

        for sw_id, sw_info in self.software_apps.items():
            if sw_info['completed']:
                continue

            current_index = sw_info['current_task_index']
            task_sequence = sw_info['task_sequence']

            if current_index < len(task_sequence):
                task_id = task_sequence[current_index]
                task = self._get_task_by_id(task_id)

                if task and task['state'] == self.TaskState.WAITING and self._check_task_dependencies(task):
                    available_tasks.append((sw_id, task, sw_info['priority']))

        if not available_tasks:
            return None, None

        # 按软件优先级和任务优先级选择
        available_tasks.sort(key=lambda x: x[2] * x[1]['priority'], reverse=True)
        return available_tasks[0][0], available_tasks[0][1]

    def _calculate_multi_software_reward(self, task: Dict, hardware_type: str, software_id: str) -> float:
        """
        计算多软件环境下的奖励

        Args:
            task: 任务信息
            hardware_type: 硬件类型
            software_id: 软件ID

        Returns:
            reward: 奖励值
        """
        base_reward = super()._calculate_reward(task, hardware_type)

        # 软件优先级加权
        software_priority = self.software_apps[software_id]['priority']
        priority_multiplier = 1.0 + (software_priority - 1.0) * 0.5

        # 资源竞争惩罚
        hw_utilization = self.hardware_resources[hardware_type]['utilization']
        if hw_utilization > 0.7:
            competition_penalty = -(hw_utilization - 0.7) * software_priority
            base_reward += competition_penalty

        # 软件级截止时间奖励
        sw_deadline = self.software_apps[software_id]['deadline']
        if self.current_time > sw_deadline:
            deadline_penalty = -(self.current_time - sw_deadline) * software_priority * 0.1
            base_reward += deadline_penalty

        return base_reward * priority_multiplier

    def _no_tasks_available(self) -> bool:
        """检查是否没有可调度任务"""
        for sw_id, sw_info in self.software_apps.items():
            if sw_info['completed']:
                continue

            current_index = sw_info['current_task_index']
            if current_index < len(sw_info['task_sequence']):
                return False

        return True

    def get_state(self) -> Tuple:
        """获取多软件环境状态"""
        base_state = super().get_state()

        # 添加多软件信息
        multi_software_info = self._get_multi_software_info()

        return base_state + (multi_software_info,)

    def _get_multi_software_info(self) -> torch.Tensor:
        """获取多软件信息"""
        info = [
            len(self.software_apps),  # 软件数量
            sum(1 for sw_info in self.software_apps.values() if not sw_info['completed']),  # 活跃软件数量
        ]

        # 添加每个软件的状态
        for sw_id, sw_info in self.software_apps.items():
            completed_tasks = sum(1 for task in self.software_tasks[sw_id]
                                  if task['state'] == self.TaskState.COMPLETED)
            total_tasks = len(self.software_tasks[sw_id])
            progress = completed_tasks / total_tasks if total_tasks > 0 else 0.0

            info.extend([
                sw_info['priority'],
                progress,
                1.0 if sw_info['completed'] else 0.0
            ])

        # 填充到固定长度
        max_software = self.max_concurrent_software
        current_length = len(info)
        target_length = 2 + max_software * 3  # 基础信息 + 每个软件的3个指标

        if current_length < target_length:
            info.extend([0.0] * (target_length - current_length))

        return torch.tensor(info, dtype=torch.float32)

    def is_done(self) -> bool:
        """检查环境是否结束"""
        # 所有软件都完成
        all_completed = all(sw_info.get('completed', False)
                            for sw_info in self.software_apps.values())

        max_time = self.config.get('max_simulation_time', 10000.0)
        time_exceeded = self.current_time >= max_time

        return all_completed or time_exceeded

    def _update_running_tasks(self, time_increment: float) -> None:
        """更新运行中的任务状态（扩展以更新软件状态）"""
        super()._update_running_tasks(time_increment)

        # 检查软件是否完成
        for sw_id, sw_info in self.software_apps.items():
            if sw_info.get('completed', False):
                continue

            completed_count = sum(1 for task in self.software_tasks[sw_id]
                                  if task['state'] == self.TaskState.COMPLETED)
            total_count = len(self.software_tasks[sw_id])

            if completed_count == total_count:
                sw_info['completed'] = True
                sw_info['completion_time'] = self.current_time
                print(f"✅ 软件 {sw_id} 已完成")

    def get_software_metrics(self) -> Dict[str, Dict]:
        """获取软件级指标"""
        for sw_id, sw_info in self.software_apps.items():
            tasks = self.software_tasks[sw_id]
            completed_tasks = [t for t in tasks if t['state'] == self.TaskState.COMPLETED]

            if completed_tasks:
                completion_times = [t['completion_time'] for t in completed_tasks]
                makespan = max(completion_times) if completion_times else 0.0

                self.software_metrics[sw_id] = {
                    'completed_tasks': len(completed_tasks),
                    'total_tasks': len(tasks),
                    'completion_rate': len(completed_tasks) / len(tasks),
                    'makespan': makespan,
                    'met_deadline': makespan <= sw_info['deadline'] if sw_info.get('deadline') else True,
                    'priority': sw_info['priority']
                }

        return dict(self.software_metrics)

    def _render_text(self) -> None:
        """文本模式渲染（扩展显示多软件信息）"""
        super()._render_text()

        print(f"\n多软件状态:")
        for sw_id, sw_info in self.software_apps.items():
            completed = sw_info.get('completed', False)
            status = "已完成" if completed else "运行中"
            priority = sw_info['priority']

            tasks = self.software_tasks[sw_id]
            completed_count = sum(1 for t in tasks if t['state'] == self.TaskState.COMPLETED)
            progress = completed_count / len(tasks) * 100

            print(f"  软件 {sw_id}: {status}, 优先级: {priority:.2f}, 进度: {progress:.1f}%")