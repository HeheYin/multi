import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import torch
import random

from .embedded_scheduling_env import EmbeddedSchedulingEnvironment
from models.core.embedded_dag import EmbeddedDAG


class DynamicTaskEnvironment(EmbeddedSchedulingEnvironment):
    """动态任务到达环境"""

    def __init__(self, config: Dict):
        super().__init__(config)

        # 动态任务参数
        self.task_arrival_rate = config.get('task_arrival_rate', 0.1)  # 任务到达率
        self.max_concurrent_dags = config.get('max_concurrent_dags', 3)  # 最大并发DAG数
        self.task_buffer_size = config.get('task_buffer_size', 10)  # 任务缓冲区大小

        # 动态任务队列
        self.pending_dags = []  # 等待处理的DAG
        self.active_dags = []  # 活跃的DAG
        self.completed_dags = []  # 已完成的DAG

        # 任务到达统计
        self.arrived_tasks = 0
        self.arrival_times = []

        # 部分可观测状态
        self.observation_history = []
        self.max_history_length = config.get('max_history_length', 10)

    def reset(self, dag: Optional[EmbeddedDAG] = None) -> Tuple:
        """
        重置环境

        Args:
            dag: 初始DAG任务图

        Returns:
            state: 初始状态
        """
        # 重置基础环境
        super().reset(dag)

        # 重置动态任务状态
        self.pending_dags = []
        self.active_dags = []
        self.completed_dags = []

        self.arrived_tasks = 0
        self.arrival_times = []

        self.observation_history = []

        # 如果提供了初始DAG，添加到活跃DAG
        if dag is not None:
            self.active_dags.append({
                'dag': dag,
                'arrival_time': self.current_time,
                'deadline': self.current_time + 1000.0,  # 默认截止时间
                'priority': 1.0
            })

        # 生成一些初始动态任务
        self._generate_initial_tasks()

        print(f"✅ 动态任务环境重置完成")

        return self.get_state()

    def _generate_initial_tasks(self) -> None:
        """生成初始动态任务"""
        from data.datasets.embedded_dag_generator import EmbeddedDAGGenerator
        generator = EmbeddedDAGGenerator(self.config)

        num_initial_tasks = random.randint(1, 3)
        for _ in range(num_initial_tasks):
            dag = generator.generate()
            arrival_delay = random.expovariate(self.task_arrival_rate)

            self.pending_dags.append({
                'dag': dag,
                'arrival_time': self.current_time + arrival_delay,
                'deadline': self.current_time + arrival_delay + random.uniform(500, 2000),
                'priority': random.uniform(0.5, 1.5)
            })

    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """
        执行调度动作（支持动态任务到达）

        Args:
            action: 调度动作

        Returns:
            state: 新状态
            reward: 奖励值
            done: 是否结束
            info: 附加信息
        """
        # 处理动态任务到达
        self._process_task_arrivals()

        # 如果没有活跃任务且动作是等待，推进时间
        if (action == len(self.hardware_resources) and
                not self.active_dags and
                len(self.running_tasks) == 0):
            # 等待动作，推进时间到下一个任务到达
            if self.pending_dags:
                next_arrival = min(dag['arrival_time'] for dag in self.pending_dags)
                time_skip = max(0, next_arrival - self.current_time)
                self._advance_time(time_skip)
            else:
                self._advance_time(self.time_slot)

            reward = -0.1  # 等待的小惩罚
            return self.get_state(), reward, self.is_done(), {'info': 'Waiting'}

        # 执行正常的调度步骤
        state, reward, done, info = super().step(action)

        # 更新历史观察
        self._update_observation_history(state)

        # 检查是否需要添加新任务到调度队列
        if self.current_task_index >= len(self.task_sequence) and self.active_dags:
            self._activate_next_dag()

        return state, reward, done, info

    def _process_task_arrivals(self) -> None:
        """处理任务到达事件"""
        arrived_dags = []

        for dag_info in self.pending_dags[:]:
            if dag_info['arrival_time'] <= self.current_time:
                # 任务到达
                if len(self.active_dags) < self.max_concurrent_dags:
                    self.active_dags.append(dag_info)
                    arrived_dags.append(dag_info)
                    self.pending_dags.remove(dag_info)
                    self.arrived_tasks += 1
                    self.arrival_times.append(self.current_time)

        # 如果有新DAG到达且当前没有活跃任务，激活一个DAG
        if arrived_dags and not self.active_dags:
            self._activate_next_dag()

    def _activate_next_dag(self) -> None:
        """激活下一个DAG进行调度"""
        if not self.active_dags:
            return

        # 选择优先级最高的DAG
        next_dag_info = max(self.active_dags, key=lambda x: x['priority'])
        self.active_dags.remove(next_dag_info)

        # 设置当前DAG
        self.current_dag = next_dag_info['dag']
        self._initialize_dag_tasks(self.current_dag)
        self.task_sequence = self._generate_task_sequence()
        self.current_task_index = 0

        print(f"📥 激活新DAG，包含 {len(self.task_sequence)} 个任务")

    def _update_observation_history(self, state: Any) -> None:
        """更新观察历史"""
        self.observation_history.append(state)
        if len(self.observation_history) > self.max_history_length:
            self.observation_history.pop(0)

    def get_state(self) -> Tuple:
        """获取部分可观测状态"""
        base_state = super().get_state()

        # 添加动态任务信息到状态
        dynamic_info = self._get_dynamic_task_info()

        # 组合状态
        full_state = base_state + (dynamic_info,)

        return full_state

    def _get_dynamic_task_info(self) -> torch.Tensor:
        """获取动态任务信息"""
        info = [
            len(self.pending_dags),  # 等待中的DAG数量
            len(self.active_dags),  # 活跃的DAG数量
            self.arrived_tasks,  # 已到达任务总数
            self.task_arrival_rate,  # 任务到达率
            min([dag['arrival_time'] for dag in self.pending_dags]) if self.pending_dags else 0.0,  # 下一个到达时间
        ]

        # 添加历史统计
        if self.arrival_times:
            avg_arrival_interval = np.mean(np.diff(self.arrival_times[-10:])) if len(self.arrival_times) > 1 else 0.0
            info.append(avg_arrival_interval)
        else:
            info.append(0.0)

        return torch.tensor(info, dtype=torch.float32)

    def is_done(self) -> bool:
        """检查环境是否结束"""
        # 在动态环境中，可以设置基于时间或任务数量的结束条件
        max_simulation_time = self.config.get('max_simulation_time', 10000.0)
        max_completed_tasks = self.config.get('max_completed_tasks', 100)

        time_condition = self.current_time >= max_simulation_time
        task_condition = self.completed_tasks >= max_completed_tasks
        no_more_tasks = (len(self.pending_dags) == 0 and
                         len(self.active_dags) == 0 and
                         len(self.running_tasks) == 0 and
                         len(self.task_queue) == 0)

        return time_condition or task_condition or no_more_tasks

    def get_available_actions(self) -> List[int]:
        """获取可用动作列表（包含等待动作）"""
        base_actions = super().get_available_actions()
        # 添加等待动作
        base_actions.append(len(base_actions))
        return base_actions

    def _render_text(self) -> None:
        """文本模式渲染（扩展显示动态任务信息）"""
        super()._render_text()

        print(f"\n动态任务信息:")
        print(f"等待中DAG: {len(self.pending_dags)}")
        print(f"活跃中DAG: {len(self.active_dags)}")
        print(f"已完成DAG: {len(self.completed_dags)}")
        print(f"总到达任务: {self.arrived_tasks}")

        if self.pending_dags:
            next_arrival = min(dag['arrival_time'] for dag in self.pending_dags)
            print(f"下一个任务到达: {next_arrival - self.current_time:.2f} ms后")