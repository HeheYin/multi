import numpy as np
from typing import Dict, Any
from environments.base_environment import BaseSchedulingEnvironment


class SchedulingMetrics:
    """调度性能指标计算器"""

    def __init__(self):
        pass

    def calculate_metrics(self, environment: BaseSchedulingEnvironment) -> Dict[str, float]:
        """
        计算调度性能指标

        Args:
            environment: 调度环境

        Returns:
            metrics: 性能指标字典
        """
        metrics = {}

        # 获取环境指标
        env_metrics = environment.get_metrics()

        # 1. 完成时间 (Makespan)
        metrics['makespan'] = env_metrics['total_makespan']

        # 2. 能耗
        metrics['energy_consumption'] = env_metrics['total_energy']

        # 3. 负载均衡
        hw_utilization = env_metrics['hardware_utilization']
        utilizations = [info['utilization'] for info in hw_utilization.values()]
        if utilizations:
            metrics['load_balance'] = 1.0 - np.std(utilizations)  # 标准差越小，负载越均衡
        else:
            metrics['load_balance'] = 0.0

        # 4. 截止时间满足率
        task_stats = env_metrics['task_completion_stats']
        total_tasks = task_stats['total']
        if total_tasks > 0:
            metrics['deadline_satisfaction'] = task_stats['completed'] / total_tasks
        else:
            metrics['deadline_satisfaction'] = 0.0

        # 5. 吞吐量 (任务/时间单位)
        if env_metrics['total_makespan'] > 0:
            metrics['throughput'] = task_stats['completed'] / env_metrics['total_makespan']
        else:
            metrics['throughput'] = 0.0

        # 6. 资源利用率
        avg_utilization = np.mean(utilizations) if utilizations else 0.0
        metrics['resource_utilization'] = avg_utilization

        return metrics

    def calculate_multi_objective_score(self, metrics: Dict[str, float],
                                        weights: Dict[str, float] = None) -> float:
        """
        计算多目标综合得分

        Args:
            metrics: 性能指标
            weights: 各指标权重

        Returns:
            score: 综合得分
        """
        if weights is None:
            weights = {
                'makespan': 0.3,
                'energy_consumption': 0.2,
                'load_balance': 0.2,
                'deadline_satisfaction': 0.2,
                'throughput': 0.1
            }

        # 归一化处理
        normalized_metrics = self._normalize_metrics(metrics)

        # 计算加权得分
        score = 0.0
        for metric_name, weight in weights.items():
            if metric_name in normalized_metrics:
                score += weight * normalized_metrics[metric_name]

        return score

    def _normalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """归一化指标（简化实现）"""
        normalized = {}

        # 对于越小越好的指标（如makespan, energy）
        smaller_better = ['makespan', 'energy_consumption']
        for metric in smaller_better:
            if metric in metrics:
                # 简化处理，实际应根据基准值归一化
                normalized[metric] = max(0, 1.0 - metrics[metric] / 1000.0)

        # 对于越大越好的指标
        larger_better = ['load_balance', 'deadline_satisfaction', 'throughput', 'resource_utilization']
        for metric in larger_better:
            if metric in metrics:
                normalized[metric] = min(1.0, metrics[metric])

        return normalized

    def compare_algorithms(self, algorithm_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        比较不同算法的性能

        Args:
            algorithm_results: 各算法结果 {algorithm_name: {metric_name: value}}

        Returns:
            comparison: 比较结果
        """
        comparison = {}

        # 找出每个指标的最佳算法
        metric_names = set()
        for results in algorithm_results.values():
            metric_names.update(results.keys())

        best_algorithms = {}
        for metric in metric_names:
            if metric in ['load_balance', 'deadline_satisfaction', 'throughput', 'resource_utilization']:
                # 越大越好
                best_algo = max(algorithm_results.keys(),
                                key=lambda x: algorithm_results[x].get(metric, -float('inf')))
            else:
                # 越小越好
                best_algo = min(algorithm_results.keys(),
                                key=lambda x: algorithm_results[x].get(metric, float('inf')))
            best_algorithms[metric] = best_algo

        comparison['best_algorithms'] = best_algorithms

        # 计算相对性能
        relative_performance = {}
        for algo, results in algorithm_results.items():
            relative_performance[algo] = {}
            for metric, value in results.items():
                if metric in ['load_balance', 'deadline_satisfaction', 'throughput', 'resource_utilization']:
                    # 基于最佳值的相对性能
                    best_value = algorithm_results[best_algorithms[metric]][metric]
                    if best_value > 0:
                        relative_performance[algo][metric] = value / best_value
                    else:
                        relative_performance[algo][metric] = 1.0
                else:
                    # 越小越好
                    best_value = algorithm_results[best_algorithms[metric]][metric]
                    if value > 0:
                        relative_performance[algo][metric] = best_value / value
                    else:
                        relative_performance[algo][metric] = 1.0

        comparison['relative_performance'] = relative_performance

        return comparison
