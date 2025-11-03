import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from models.core.embedded_dag import EmbeddedDAG, TaskNode, HardwareType


class EmbeddedDAGGenerator:
    """嵌入式DAG生成器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化DAG生成器

        Args:
            config: 配置参数
        """
        self.config = config

        # DAG生成参数
        self.default_task_count_range = config.get('task_count_range', (5, 15))
        self.default_density = config.get('dag_density', 0.3)  # 边密度
        self.default_complexity = config.get('dag_complexity', 'medium')  # 复杂度

        # 硬件配置
        self.hardware_types = config.get('hardware', {}).get('types', ['CPU', 'GPU', 'FPGA', 'MCU'])

        # 任务类型配置
        self.task_types = config.get('task_types', [
            'computation', 'communication', 'control', 'sensing', 'storage'
        ])

        # 软件类型配置
        self.software_types = config.get('software_types', [
            'control_software', 'ai_inference', 'sensing_software', 'communication_software'
        ])

    def generate(self,
                 task_count_range: Optional[Tuple[int, int]] = None,
                 density: Optional[float] = None,
                 software_type: Optional[str] = None,
                 **kwargs) -> EmbeddedDAG:
        """
        生成嵌入式DAG

        Args:
            task_count_range: 任务数量范围
            density: DAG密度
            software_type: 软件类型
            **kwargs: 其他参数

        Returns:
            dag: 生成的DAG
        """
        task_count_range = task_count_range or self.default_task_count_range
        density = density or self.default_density

        # 随机生成任务数量
        task_count = random.randint(*task_count_range)

        # 创建DAG
        dag = EmbeddedDAG()

        # 生成节点
        nodes = self._generate_nodes(task_count, software_type, **kwargs)
        for node in nodes:
            dag.add_node(node)

        # 生成边
        edges = self._generate_edges(nodes, density, **kwargs)
        for source, target, data_size in edges:
            dag.add_edge(source, target, data_size)

        # 生成硬件通信成本
        self._generate_communication_costs(dag, **kwargs)

        return dag

    def _generate_nodes(self, task_count: int,
                        software_type: Optional[str] = None,
                        **kwargs) -> List[TaskNode]:
        """
        生成任务节点

        Args:
            task_count: 任务数量
            software_type: 软件类型
            **kwargs: 其他参数

        Returns:
            nodes: 任务节点列表
        """
        nodes = []

        # 根据软件类型设置参数
        if software_type:
            task_type_weights = self._get_task_type_weights(software_type)
        else:
            task_type_weights = {task_type: 1.0 for task_type in self.task_types}

        for i in range(task_count):
            # 随机选择任务类型
            task_type = random.choices(
                list(task_type_weights.keys()),
                weights=list(task_type_weights.values())
            )[0]

            # 生成计算成本
            computation_cost = self._generate_computation_cost(task_type)

            # 生成数据依赖
            data_dependencies = []  # 在生成边时填充

            # 生成硬件约束
            hardware_constraints = self._generate_hardware_constraints(task_type)

            # 生成截止时间和周期
            deadline = self._generate_deadline(task_type, **kwargs)
            period = self._generate_period(task_type, **kwargs)

            # 生成能耗
            energy_consumption = self._generate_energy_consumption(task_type)

            node = TaskNode(
                node_id=i,
                task_type=task_type,
                computation_cost=computation_cost,
                data_dependencies=data_dependencies,
                hardware_constraints=hardware_constraints,
                deadline=deadline,
                period=period,
                energy_consumption=energy_consumption
            )

            nodes.append(node)

        return nodes

    def _generate_computation_cost(self, task_type: str) -> Dict[HardwareType, float]:
        """
        生成任务在不同硬件上的计算成本

        Args:
            task_type: 任务类型

        Returns:
            computation_cost: 计算成本字典
        """
        computation_cost = {}

        # 根据任务类型设置不同的计算成本模式
        if task_type == 'computation':
            # 计算密集型任务在GPU/FPGA上更快
            computation_cost[HardwareType.CPU] = random.uniform(50, 100)
            computation_cost[HardwareType.GPU] = random.uniform(10, 30)
            computation_cost[HardwareType.FPGA] = random.uniform(15, 35)
            computation_cost[HardwareType.MCU] = random.uniform(80, 150)
        elif task_type == 'ai_inference':
            # AI推理任务在GPU/FPGA上最优
            computation_cost[HardwareType.CPU] = random.uniform(100, 200)
            computation_cost[HardwareType.GPU] = random.uniform(20, 50)
            computation_cost[HardwareType.FPGA] = random.uniform(25, 60)
            computation_cost[HardwareType.MCU] = random.uniform(300, 500)
        elif task_type == 'control':
            # 控制任务在MCU/CPU上较快
            computation_cost[HardwareType.CPU] = random.uniform(5, 15)
            computation_cost[HardwareType.GPU] = random.uniform(10, 25)
            computation_cost[HardwareType.FPGA] = random.uniform(8, 20)
            computation_cost[HardwareType.MCU] = random.uniform(3, 10)
        elif task_type == 'sensing':
            # 传感任务在MCU上最优
            computation_cost[HardwareType.CPU] = random.uniform(20, 40)
            computation_cost[HardwareType.GPU] = random.uniform(30, 60)
            computation_cost[HardwareType.FPGA] = random.uniform(25, 50)
            computation_cost[HardwareType.MCU] = random.uniform(10, 25)
        else:
            # 默认计算成本
            for hw_type in HardwareType:
                computation_cost[hw_type] = random.uniform(20, 80)

        return computation_cost

    def _generate_hardware_constraints(self, task_type: str) -> Dict[str, Any]:
        """
        生成硬件约束

        Args:
            task_type: 任务类型

        Returns:
            hardware_constraints: 硬件约束字典
        """
        constraints = {
            'priority': random.uniform(0.1, 1.0)
        }

        # 根据任务类型设置硬件约束
        if task_type == 'ai_inference':
            constraints['allowed_hardware'] = [HardwareType.GPU, HardwareType.FPGA]
        elif task_type == 'control':
            constraints['allowed_hardware'] = [HardwareType.MCU, HardwareType.CPU]
        elif task_type == 'sensing':
            constraints['allowed_hardware'] = [HardwareType.MCU]
        else:
            # 允许所有硬件
            constraints['allowed_hardware'] = list(HardwareType)

        return constraints

    def _generate_deadline(self, task_type: str, **kwargs) -> Optional[float]:
        """
        生成任务截止时间

        Args:
            task_type: 任务类型
            **kwargs: 其他参数

        Returns:
            deadline: 截止时间
        """
        deadline_strictness = kwargs.get('deadline_strictness', 'medium')

        if deadline_strictness == 'high':
            deadline_factor = random.uniform(1.2, 1.8)
        elif deadline_strictness == 'very_high':
            deadline_factor = random.uniform(1.0, 1.3)
        elif deadline_strictness == 'low':
            deadline_factor = random.uniform(2.0, 4.0)
        else:  # medium
            deadline_factor = random.uniform(1.5, 2.5)

        # 基于任务类型的平均执行时间估算截止时间
        base_time = 50.0  # 基准时间
        return base_time * deadline_factor

    def _generate_period(self, task_type: str, **kwargs) -> Optional[float]:
        """
        生成任务周期

        Args:
            task_type: 任务类型
            **kwargs: 其他参数

        Returns:
            period: 任务周期
        """
        periodic_tasks_ratio = kwargs.get('periodic_tasks_ratio', 0.3)

        if random.random() < periodic_tasks_ratio:
            if task_type == 'control':
                # 控制任务通常周期较短
                return random.uniform(10, 100)
            else:
                # 其他任务周期较长
                return random.uniform(100, 1000)
        else:
            return None

    def _generate_energy_consumption(self, task_type: str) -> Dict[HardwareType, float]:
        """
        生成任务在不同硬件上的能耗

        Args:
            task_type: 任务类型

        Returns:
            energy_consumption: 能耗字典
        """
        energy_consumption = {}

        # 根据任务类型设置不同的能耗模式
        if task_type == 'computation':
            energy_consumption[HardwareType.CPU] = random.uniform(20, 40)
            energy_consumption[HardwareType.GPU] = random.uniform(50, 100)
            energy_consumption[HardwareType.FPGA] = random.uniform(30, 60)
            energy_consumption[HardwareType.MCU] = random.uniform(5, 15)
        elif task_type == 'ai_inference':
            energy_consumption[HardwareType.CPU] = random.uniform(40, 80)
            energy_consumption[HardwareType.GPU] = random.uniform(80, 150)
            energy_consumption[HardwareType.FPGA] = random.uniform(50, 100)
            energy_consumption[HardwareType.MCU] = random.uniform(100, 200)
        elif task_type == 'control':
            energy_consumption[HardwareType.CPU] = random.uniform(5, 15)
            energy_consumption[HardwareType.GPU] = random.uniform(15, 30)
            energy_consumption[HardwareType.FPGA] = random.uniform(10, 25)
            energy_consumption[HardwareType.MCU] = random.uniform(2, 8)
        elif task_type == 'sensing':
            energy_consumption[HardwareType.CPU] = random.uniform(10, 25)
            energy_consumption[HardwareType.GPU] = random.uniform(20, 40)
            energy_consumption[HardwareType.FPGA] = random.uniform(15, 35)
            energy_consumption[HardwareType.MCU] = random.uniform(3, 10)
        else:
            # 默认能耗
            for hw_type in HardwareType:
                energy_consumption[hw_type] = random.uniform(10, 50)

        return energy_consumption

    def _get_task_type_weights(self, software_type: str) -> Dict[str, float]:
        """
        根据软件类型获取任务类型权重

        Args:
            software_type: 软件类型

        Returns:
            weights: 任务类型权重字典
        """
        weights = {task_type: 1.0 for task_type in self.task_types}

        if software_type == 'control_software':
            weights['control'] = 3.0
            weights['computation'] = 0.5
        elif software_type == 'ai_inference':
            weights['ai_inference'] = 3.0
            weights['computation'] = 1.5
        elif software_type == 'sensing_software':
            weights['sensing'] = 3.0
            weights['communication'] = 1.5
        elif software_type == 'communication_software':
            weights['communication'] = 3.0
            weights['storage'] = 1.5

        return weights

    def _generate_edges(self, nodes: List[TaskNode],
                        density: float, **kwargs) -> List[Tuple[int, int, float]]:
        """
        生成DAG边

        Args:
            nodes: 节点列表
            density: 边密度
            **kwargs: 其他参数

        Returns:
            edges: 边列表
        """
        edges = []
        node_count = len(nodes)

        # 计算期望边数
        max_edges = node_count * (node_count - 1) // 2  # 有向无环图的最大边数
        expected_edges = int(max_edges * density)

        # 使用拓扑排序确保无环
        node_ids = [node.node_id for node in nodes]
        random.shuffle(node_ids)  # 随机排列节点顺序

        # 生成边确保DAG性质
        added_edges = 0
        for i in range(node_count):
            for j in range(i + 1, node_count):
                if added_edges >= expected_edges:
                    break

                # 根据概率添加边
                if random.random() < density:
                    source_id = node_ids[i]
                    target_id = node_ids[j]

                    # 生成数据大小
                    data_size = random.uniform(1, 100)  # KB

                    edges.append((source_id, target_id, data_size))
                    added_edges += 1

        return edges

    def _generate_communication_costs(self, dag: EmbeddedDAG, **kwargs):
        """
        生成硬件间通信成本

        Args:
            dag: DAG对象
            **kwargs: 其他参数
        """
        # 为每对硬件类型生成通信成本
        for i, src_hw in enumerate(HardwareType):
            for j, dst_hw in enumerate(HardwareType):
                if i != j:  # 不同硬件类型之间
                    # 随机生成延迟和带宽
                    latency = random.uniform(0.1, 5.0)  # ms
                    bandwidth = random.uniform(10, 1000)  # MB/s

                    dag.add_hardware_communication_cost(
                        src_hw=src_hw,
                        dst_hw=dst_hw,
                        latency=latency,
                        bandwidth=bandwidth
                    )

    def generate_dataset(self,
                         num_dags: int,
                         task_count_range: Optional[Tuple[int, int]] = None,
                         density: Optional[float] = None,
                         software_type: Optional[str] = None) -> List[EmbeddedDAG]:
        """
        生成DAG数据集

        Args:
            num_dags: DAG数量
            task_count_range: 任务数量范围
            density: DAG密度
            software_type: 软件类型

        Returns:
            dag_list: DAG列表
        """
        dag_list = []

        for _ in range(num_dags):
            dag = self.generate(
                task_count_range=task_count_range,
                density=density,
                software_type=software_type
            )
            dag_list.append(dag)

        return dag_list

    def generate_industrial_control_dag(self) -> EmbeddedDAG:
        """生成工业控制DAG"""
        return self.generate(
            task_count_range=(8, 15),
            density=0.4,
            software_type='control_software',
            deadline_strictness='high',
            periodic_tasks_ratio=0.7
        )

    def generate_edge_ai_dag(self) -> EmbeddedDAG:
        """生成边缘AI推理DAG"""
        return self.generate(
            task_count_range=(10, 20),
            density=0.3,
            software_type='ai_inference',
            deadline_strictness='medium',
            computation_intensity='high'
        )

    def generate_autonomous_driving_dag(self) -> EmbeddedDAG:
        """生成自动驾驶系统DAG"""
        return self.generate(
            task_count_range=(12, 25),
            density=0.35,
            software_type='sensing_software',
            deadline_strictness='very_high',
            reliability_requirement='high'
        )

    def generate_smart_surveillance_dag(self) -> EmbeddedDAG:
        """生成智能监控系统DAG"""
        return self.generate(
            task_count_range=(6, 12),
            density=0.25,
            software_type='communication_software',
            energy_sensitivity='high',
            continuous_operation=True
        )
