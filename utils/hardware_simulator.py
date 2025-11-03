import numpy as np
from typing import Dict, Any, List
import random


class HardwareSimulator:
    """硬件模拟器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hardware_specs = config.get('hardware', {}).get('capabilities', {})
        self.hardware_types = config.get('hardware', {}).get('types', ['CPU', 'GPU', 'FPGA', 'MCU'])

        # 硬件状态
        self.hardware_status = {}
        self._initialize_hardware()

    def _initialize_hardware(self):
        """初始化硬件状态"""
        for hw_type in self.hardware_types:
            self.hardware_status[hw_type] = {
                'available': True,
                'utilization': 0.0,
                'temperature': 25.0,  # 初始温度25°C
                'power_consumption': 0.0,
                'failure_probability': 0.0
            }

    def simulate_execution(self, task: Dict[str, Any], hardware_type: str) -> Dict[str, Any]:
        """
        模拟任务在硬件上的执行

        Args:
            task: 任务信息
            hardware_type: 硬件类型

        Returns:
            execution_result: 执行结果
        """
        hw_spec = self.hardware_specs.get(hardware_type, {})
        hw_status = self.hardware_status[hardware_type]

        # 检查硬件是否可用
        if not hw_status['available']:
            return {
                'success': False,
                'error': 'Hardware not available',
                'execution_time': 0,
                'energy_consumption': 0
            }

        # 模拟硬件故障
        if random.random() < hw_status['failure_probability']:
            hw_status['available'] = False
            return {
                'success': False,
                'error': 'Hardware failure',
                'execution_time': 0,
                'energy_consumption': 0
            }

        # 计算执行时间和能耗
        base_time = task.get('estimated_time', 10.0)
        compute_power = hw_spec.get('compute_power', 1.0)

        # 考虑硬件计算能力和当前负载
        effective_time = base_time / (compute_power * (1.0 - hw_status['utilization'] * 0.5))

        # 能耗计算
        base_power = hw_spec.get('base_power', 10.0)
        max_power = hw_spec.get('max_power', 100.0)
        dynamic_power = (max_power - base_power) * (hw_status['utilization'] + 0.1)
        total_power = base_power + dynamic_power
        energy_consumption = total_power * effective_time / 1000.0  # 转换为焦耳

        # 更新硬件状态
        self._update_hardware_status(hardware_type, effective_time)

        return {
            'success': True,
            'execution_time': effective_time,
            'energy_consumption': energy_consumption,
            'hardware_utilization': hw_status['utilization']
        }

    def _update_hardware_status(self, hardware_type: str, execution_time: float):
        """更新硬件状态"""
        hw_status = self.hardware_status[hardware_type]

        # 更新利用率
        hw_status['utilization'] = min(1.0, hw_status['utilization'] + 0.1)

        # 更新温度（简化模型）
        hw_status['temperature'] += execution_time * 0.01
        if hw_status['temperature'] > 80:
            hw_status['failure_probability'] += 0.001

        # 更新功耗
        hw_spec = self.hardware_specs.get(hardware_type, {})
        base_power = hw_spec.get('base_power', 10.0)
        max_power = hw_spec.get('max_power', 100.0)
        hw_status['power_consumption'] = base_power + (max_power - base_power) * hw_status['utilization']

    def get_hardware_info(self) -> Dict[str, Any]:
        """获取硬件信息"""
        return {
            'hardware_types': self.hardware_types,
            'hardware_status': self.hardware_status,
            'total_hardware': len(self.hardware_types)
        }

    def simulate_hardware_failure(self, hardware_type: str, failure_rate: float = 0.1):
        """模拟硬件故障"""
        if hardware_type in self.hardware_status:
            self.hardware_status[hardware_type]['failure_probability'] = failure_rate
            if random.random() < failure_rate:
                self.hardware_status[hardware_type]['available'] = False

    def recover_hardware(self, hardware_type: str):
        """恢复硬件"""
        if hardware_type in self.hardware_status:
            self.hardware_status[hardware_type]['available'] = True
            self.hardware_status[hardware_type]['failure_probability'] = 0.0
            self.hardware_status[hardware_type]['temperature'] = 25.0

    def get_performance_benchmark(self) -> Dict[str, float]:
        """获取硬件性能基准测试结果"""
        benchmark_results = {}

        for hw_type in self.hardware_types:
            hw_spec = self.hardware_specs.get(hw_type, {})
            compute_power = hw_spec.get('compute_power', 1.0)
            energy_efficiency = hw_spec.get('energy_efficiency', 0.8)

            # 综合性能评分
            performance_score = compute_power * energy_efficiency
            benchmark_results[hw_type] = performance_score

        return benchmark_results
