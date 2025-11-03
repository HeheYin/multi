import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from tqdm import tqdm
import time
import os

from models.core.embedded_dag import EmbeddedDAG
from environments.embedded_scheduling_env import EmbeddedSchedulingEnvironment
from agents.d3qn_agent import D3QNAgent
from utils.metrics import SchedulingMetrics
from utils.visualization import plot_comparison_results


class BaselineComparator:
    """基线算法比较器"""

    def __init__(self, config):
        self.config = config
        self.metrics_calculator = SchedulingMetrics()
        self.results = {}

        # 初始化基线算法
        self.baselines = self._initialize_baselines()

    def _initialize_baselines(self) -> Dict[str, Any]:
        """初始化所有基线算法"""
        baselines = {}

        # HEFT算法
        baselines['HEFT'] = HEFTBaseline()

        # CPOP算法
        baselines['CPOP'] = CPOPBaseline()

        # EDF算法（嵌入式实时调度）
        baselines['EDF'] = EDFBaseline()

        # RM算法（速率单调调度）
        baselines['RM'] = RMBaseline()

        # 随机调度
        baselines['Random'] = RandomBaseline()

        return baselines

    def load_trained_model(self, model_path: str):
        """加载训练好的MODRL模型"""
        from models.networks.embedded_modrl import EmbeddedMODRL

        model_config = self.config['model']
        self.modrl_model = EmbeddedMODRL(
            node_feature_dim=model_config['node_feature_dim'],
            hardware_feature_dim=model_config['hardware_feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_hardware=model_config['num_hardware'],
            num_actions=model_config['num_actions']
        )

        checkpoint = torch.load(model_path, map_location='cpu')
        self.modrl_model.load_state_dict(checkpoint['model_state_dict'])
        self.modrl_model.eval()

        print(f"✅ 成功加载MODRL模型: {model_path}")

    def run_comparison(self, test_datasets: Dict[str, List[EmbeddedDAG]],
                       num_runs: int = 10) -> Dict[str, Dict[str, float]]:
        """
        运行完整的比较实验

        Args:
            test_datasets: 测试数据集 {'dataset_name': [dag1, dag2, ...]}
            num_runs: 每个算法在每个DAG上的运行次数
        """
        print("🚀 开始基线算法比较实验...")

        all_results = {}

        for dataset_name, dags in test_datasets.items():
            print(f"\n📊 测试数据集: {dataset_name}, DAG数量: {len(dags)}")
            dataset_results = {}

            # 测试所有基线算法
            for baseline_name, baseline in self.baselines.items():
                print(f"  正在测试 {baseline_name}...")
                baseline_metrics = self._evaluate_baseline(
                    baseline, dags, num_runs, baseline_name)
                dataset_results[baseline_name] = baseline_metrics

            # 测试MODRL模型
            if hasattr(self, 'modrl_model'):
                print("  正在测试 MODRL...")
                modrl_metrics = self._evaluate_modrl(dags, num_runs)
                dataset_results['MODRL'] = modrl_metrics

            all_results[dataset_name] = dataset_results

            # 保存当前数据集结果
            self._save_dataset_results(dataset_name, dataset_results)

        self.results = all_results
        return all_results

    def _evaluate_baseline(self, baseline, dags: List[EmbeddedDAG],
                           num_runs: int, baseline_name: str) -> Dict[str, float]:
        """评估单个基线算法"""
        env = EmbeddedSchedulingEnvironment(self.config)
        all_metrics = []

        for dag in tqdm(dags, desc=f"{baseline_name}", leave=False):
            dag_metrics = []

            for run in range(num_runs):
                # 重置环境
                state = env.reset(dag)
                done = False

                while not done:
                    # 基线算法决策
                    action = baseline.schedule(env.current_state, env.available_hardware)
                    state, reward, done, info = env.step(action)

                # 收集指标
                metrics = self.metrics_calculator.calculate_metrics(env)
                dag_metrics.append(metrics)

            # 计算DAG的平均指标
            avg_metrics = self._average_metrics(dag_metrics)
            all_metrics.append(avg_metrics)

        # 计算整体平均指标
        return self._average_metrics(all_metrics)

    def _evaluate_modrl(self, dags: List[EmbeddedDAG], num_runs: int) -> Dict[str, float]:
        """评估MODRL模型"""
        env = EmbeddedSchedulingEnvironment(self.config)
        all_metrics = []

        for dag in tqdm(dags, desc="MODRL", leave=False):
            dag_metrics = []

            for run in range(num_runs):
                state = env.reset(dag)
                done = False

                while not done:
                    # MODRL模型决策
                    with torch.no_grad():
                        state_tensor = self._state_to_tensor(state, env)
                        q_values = self.modrl_model(*state_tensor)
                        action = torch.argmax(q_values).item()

                    state, reward, done, info = env.step(action)

                metrics = self.metrics_calculator.calculate_metrics(env)
                dag_metrics.append(metrics)

            avg_metrics = self._average_metrics(dag_metrics)
            all_metrics.append(avg_metrics)

        return self._average_metrics(all_metrics)

    def _state_to_tensor(self, state, env):
        """将状态转换为模型输入张量"""
        # 这里需要根据实际的状态表示来实现
        node_features = torch.tensor(state['node_features'], dtype=torch.float32)
        adjacency_matrix = torch.tensor(state['adjacency_matrix'], dtype=torch.float32)
        task_sequence = torch.tensor(state['task_sequence'], dtype=torch.long)
        hardware_features = torch.tensor(state['hardware_features'], dtype=torch.float32)

        return node_features, adjacency_matrix, task_sequence, hardware_features

    def _average_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """计算指标平均值"""
        if not metrics_list:
            return {}

        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [metrics[key] for metrics in metrics_list]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f"{key}_std"] = np.std(values)

        return avg_metrics

    def _save_dataset_results(self, dataset_name: str, results: Dict):
        """保存数据集结果到文件"""
        os.makedirs('results/comparison', exist_ok=True)

        # 保存为CSV
        df_data = []
        for algo, metrics in results.items():
            row = {'Algorithm': algo}
            row.update(metrics)
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_csv(f'results/comparison/{dataset_name}_comparison.csv', index=False)

        # 保存为JSON
        import json
        with open(f'results/comparison/{dataset_name}_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)

    def generate_report(self):
        """生成比较实验报告"""
        if not self.results:
            print("⚠️ 没有可用的结果，请先运行比较实验")
            return

        print("\n" + "=" * 80)
        print("📈 基线算法比较实验报告")
        print("=" * 80)

        for dataset_name, algorithms in self.results.items():
            print(f"\n📊 数据集: {dataset_name}")
            print("-" * 50)

            # 创建比较表格
            metrics_to_show = ['makespan', 'energy_consumption', 'load_balance', 'deadline_satisfaction']

            for metric in metrics_to_show:
                if metric in list(algorithms.values())[0]:
                    print(f"\n{metric.replace('_', ' ').title()}:")
                    for algo, metrics in algorithms.items():
                        value = metrics.get(metric, 'N/A')
                        if isinstance(value, float):
                            print(f"  {algo:15}: {value:.4f}")
                        else:
                            print(f"  {algo:15}: {value}")

        # 生成可视化图表
        self._generate_comparison_plots()

    def _generate_comparison_plots(self):
        """生成比较结果可视化图表"""
        for dataset_name, algorithms in self.results.items():
            plot_comparison_results(algorithms, dataset_name)


# 基线算法实现
class HEFTBaseline:
    """HEFT算法实现"""

    def schedule(self, state, available_hardware):
        # 简化的HEFT实现
        # 实际实现需要计算 upward rank 和选择最早完成时间的处理器
        current_task = state['current_task']
        task_priority = state['task_priorities'][current_task]

        # 选择负载最低的硬件
        hardware_loads = state['hardware_loads']
        best_hardware = np.argmin(hardware_loads)

        return best_hardware


class CPOPBaseline:
    """CPOP算法实现"""

    def schedule(self, state, available_hardware):
        # 简化的CPOP实现
        current_task = state['current_task']
        is_critical = state.get('is_critical_path', {}).get(current_task, False)

        if is_critical:
            # 关键路径任务分配到专用处理器
            return 0  # 假设处理器0是专用处理器
        else:
            # 非关键路径使用HEFT策略
            hardware_loads = state['hardware_loads']
            return np.argmin(hardware_loads)


class EDFBaseline:
    """最早截止时间优先算法"""

    def schedule(self, state, available_hardware):
        current_task = state['current_task']
        deadlines = state.get('task_deadlines', {})

        # 优先分配截止时间最早的任务到可用硬件
        task_deadline = deadlines.get(current_task, float('inf'))

        # 选择最早可用的硬件
        hardware_availability = state['hardware_availability']
        best_hardware = np.argmin(hardware_availability)

        return best_hardware


class RMBaseline:
    """速率单调调度算法"""

    def schedule(self, state, available_hardware):
        current_task = state['current_task']
        task_periods = state.get('task_periods', {})

        # 周期越短，优先级越高
        task_period = task_periods.get(current_task, float('inf'))
        priority = 1.0 / task_period if task_period > 0 else 0

        # 分配到负载合适的硬件
        hardware_loads = state['hardware_loads']
        best_hardware = np.argmin(hardware_loads)

        return best_hardware


class RandomBaseline:
    """随机调度算法"""

    def schedule(self, state, available_hardware):
        return np.random.randint(0, len(available_hardware))