import gymnasium
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


class HardwareState(Enum):
    """硬件状态枚举"""
    IDLE = 0
    BUSY = 1
    OVERLOADED = 2
    FAILED = 3


class TaskState(Enum):
    """任务状态枚举"""
    WAITING = 0
    READY = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4


class BaseSchedulingEnvironment(ABC):
    """基础调度环境抽象基类"""

    def __init__(self, config: Dict):
        """
        初始化基础环境

        Args:
            config: 环境配置参数
        """
        self.config = config
        self.max_tasks = config.get('max_tasks', 50)
        self.max_hardware = config.get('max_hardware', 4)
        self.time_slot = config.get('time_slot', 1.0)  # 时间片长度(ms)

        # 环境状态
        self.current_time = 0.0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_reward = 0.0

        # 硬件资源
        self.hardware_resources = self._initialize_hardware()

        # 任务队列
        self.task_queue = []
        self.running_tasks = []
        self.completed_task_list = []

        # 监控指标
        self.metrics = {
            'total_makespan': 0.0,
            'total_energy': 0.0,
            'hardware_utilization': {},
            'task_completion_stats': {}
        }

        # 随机种子
        self.seed = config.get('seed', 42)
        self._set_random_seed()

    @abstractmethod
    def reset(self, dag=None) -> Any:
        """
        重置环境状态

        Args:
            dag: 可选的DAG任务图

        Returns:
            state: 初始状态
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """
        执行一步调度动作

        Args:
            action: 调度动作

        Returns:
            state: 新状态
            reward: 奖励值
            done: 是否结束
            info: 附加信息
        """
        pass

    @abstractmethod
    def get_state(self) -> Any:
        """获取当前环境状态"""
        pass

    @abstractmethod
    def render(self, mode: str = 'human') -> Optional[Any]:
        """
        渲染环境状态

        Args:
            mode: 渲染模式

        Returns:
            渲染结果
        """
        pass

    def _initialize_hardware(self) -> Dict[str, Dict]:
        """
        初始化硬件资源

        Returns:
            hardware_dict: 硬件资源字典
        """
        hardware_types = self.config.get('hardware', {}).get('types', ['CPU', 'GPU', 'FPGA', 'MCU'])
        hardware_resources = {}

        for i, hw_type in enumerate(hardware_types):
            hardware_resources[hw_type] = {
                'id': i,
                'type': hw_type,
                'state': HardwareState.IDLE,
                'current_task': None,
                'utilization': 0.0,
                'energy_consumption': 0.0,
                'completed_tasks': 0,
                'total_working_time': 0.0,
                'specifications': self._get_hardware_specs(hw_type)
            }

        return hardware_resources

    def _get_hardware_specs(self, hw_type: str) -> Dict[str, Any]:
        """
        获取硬件规格

        Args:
            hw_type: 硬件类型

        Returns:
            specs: 硬件规格字典
        """
        specs_config = self.config.get('hardware', {}).get('capabilities', {})

        default_specs = {
            'compute_power': 1.0,
            'energy_efficiency': 0.8,
            'memory': 4096,
            'max_power': 100.0,
            'base_power': 10.0
        }

        return specs_config.get(hw_type, default_specs)

    def _set_random_seed(self) -> None:
        """设置随机种子"""
        np.random.seed(self.seed)

    def _update_hardware_state(self) -> None:
        """更新硬件状态"""
        for hw_id, hw_info in self.hardware_resources.items():
            # 更新硬件利用率
            if hw_info['current_task'] is not None:
                hw_info['state'] = HardwareState.BUSY
                hw_info['utilization'] = min(1.0, hw_info['utilization'] + 0.1)
            else:
                hw_info['state'] = HardwareState.IDLE
                hw_info['utilization'] = max(0.0, hw_info['utilization'] - 0.05)

            # 检查是否过载
            if hw_info['utilization'] > 0.9:
                hw_info['state'] = HardwareState.OVERLOADED

    def _calculate_energy_consumption(self, hw_type: str, execution_time: float) -> float:
        """
        计算能耗

        Args:
            hw_type: 硬件类型
            execution_time: 执行时间

        Returns:
            energy: 能耗值
        """
        specs = self.hardware_resources[hw_type]['specifications']
        base_power = specs.get('base_power', 10.0)
        max_power = specs.get('max_power', 100.0)
        utilization = self.hardware_resources[hw_type]['utilization']

        # 简化的能耗模型: 基础功耗 + 动态功耗
        dynamic_power = (max_power - base_power) * utilization
        total_power = base_power + dynamic_power

        # 能耗 = 功率 × 时间
        energy = total_power * execution_time / 1000.0  # 转换为焦耳

        return energy

    def _check_task_dependencies(self, task: Dict) -> bool:
        """
        检查任务依赖是否满足

        Args:
            task: 任务信息

        Returns:
            ready: 是否就绪
        """
        if 'dependencies' not in task:
            return True

        for dep_task_id in task['dependencies']:
            dep_task = self._get_task_by_id(dep_task_id)
            if dep_task is None or dep_task.get('state') != TaskState.COMPLETED:
                return False

        return True

    def _get_task_by_id(self, task_id: int) -> Optional[Dict]:
        """根据ID获取任务"""
        all_tasks = self.task_queue + self.running_tasks + self.completed_task_list
        for task in all_tasks:
            if task.get('id') == task_id:
                return task
        return None

    def _advance_time(self, time_increment: float = None) -> None:
        """
        推进仿真时间

        Args:
            time_increment: 时间增量，None则使用默认时间片
        """
        if time_increment is None:
            time_increment = self.time_slot

        self.current_time += time_increment

        # 更新运行中的任务
        self._update_running_tasks(time_increment)

        # 更新硬件状态
        self._update_hardware_state()

    def _update_running_tasks(self, time_increment: float) -> None:
        """更新运行中的任务状态"""
        completed_tasks = []

        for task in self.running_tasks:
            if task['state'] == TaskState.RUNNING:
                # 更新任务进度
                task['executed_time'] += time_increment
                task['remaining_time'] = max(0, task['estimated_time'] - task['executed_time'])

                # 检查是否完成
                if task['remaining_time'] <= 0:
                    task['state'] = TaskState.COMPLETED
                    task['completion_time'] = self.current_time
                    completed_tasks.append(task)

                    # 更新硬件统计
                    hw_type = task['assigned_hardware']
                    self.hardware_resources[hw_type]['completed_tasks'] += 1
                    self.hardware_resources[hw_type]['total_working_time'] += task['executed_time']

                    # 计算能耗
                    energy = self._calculate_energy_consumption(hw_type, task['executed_time'])
                    self.hardware_resources[hw_type]['energy_consumption'] += energy
                    self.metrics['total_energy'] += energy

        # 移动已完成任务
        for task in completed_tasks:
            self.running_tasks.remove(task)
            self.completed_task_list.append(task)
            self.completed_tasks += 1

    def get_metrics(self) -> Dict[str, Any]:
        """获取环境指标"""
        # 计算硬件利用率
        hw_utilization = {}
        for hw_type, hw_info in self.hardware_resources.items():
            hw_utilization[hw_type] = {
                'utilization': hw_info['utilization'],
                'completed_tasks': hw_info['completed_tasks'],
                'energy_consumption': hw_info['energy_consumption'],
                'total_working_time': hw_info['total_working_time']
            }

        # 计算整体完成时间
        if self.completed_task_list:
            completion_times = [task['completion_time'] for task in self.completed_task_list]
            self.metrics['total_makespan'] = max(completion_times) if completion_times else 0.0
        else:
            self.metrics['total_makespan'] = self.current_time

        self.metrics['hardware_utilization'] = hw_utilization
        self.metrics['task_completion_stats'] = {
            'completed': self.completed_tasks,
            'failed': self.failed_tasks,
            'total': self.completed_tasks + self.failed_tasks
        }

        return self.metrics

    def get_available_actions(self) -> List[int]:
        """获取可用动作列表"""
        return list(range(len(self.hardware_resources)))

    def is_done(self) -> bool:
        """检查环境是否结束"""
        # 所有任务都完成或没有更多任务
        return (len(self.task_queue) == 0 and
                len(self.running_tasks) == 0 and
                self.completed_tasks > 0)

    def get_hardware_info(self) -> Dict[str, Any]:
        """获取硬件信息"""
        return {
            'hardware_count': len(self.hardware_resources),
            'hardware_types': list(self.hardware_resources.keys()),
            'hardware_states': {hw: info['state'].value for hw, info in self.hardware_resources.items()},
            'hardware_utilization': {hw: info['utilization'] for hw, info in self.hardware_resources.items()}
        }

    def get_task_info(self) -> Dict[str, Any]:
        """获取任务信息"""
        return {
            'waiting_tasks': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks
        }