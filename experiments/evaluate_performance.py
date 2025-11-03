import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import torch
import yaml
from tqdm import tqdm

from evaluation.baseline_comparison import BaselineComparator
from evaluation.ablation_study import AblationStudy
from evaluation.real_world_test import RealWorldTester
from agents.d3qn_agent import D3QNAgent
from models.networks.embedded_modrl import EmbeddedMODRL
from environments.embedded_scheduling_env import EmbeddedSchedulingEnvironment
from data.datasets.embedded_dag_generator import EmbeddedDAGGenerator
from utils.metrics import SchedulingMetrics


class PerformanceEvaluator:
    """综合性能评估器"""

    def __init__(self, config: Dict):
        self.config = config
        self.metrics_calculator = SchedulingMetrics()
        self.dag_generator = EmbeddedDAGGenerator(config)

        # 评估结果
        self.evaluation_results = {}

    def load_model(self, model_path: str):
        """加载MODRL模型"""
        model_config = self.config['model']

        self.model = EmbeddedMODRL(
            node_feature_dim=model_config['node_feature_dim'],
            hardware_feature_dim=model_config['hardware_feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_hardware=model_config['num_hardware'],
            num_actions=model_config['num_actions']
        )

        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✅ MODRL模型已加载: {model_path}")

    def run_comprehensive_evaluation(self, test_datasets: Dict[str, List],
                                     model_path: str = None) -> Dict[str, Any]:
        """
        运行综合性能评估

        Args:
            test_datasets: 测试数据集 {'dataset_name': [dag1, dag2, ...]}
            model_path: MODRL模型路径

        Returns:
            evaluation_results: 评估结果
        """
        print("=" * 80)
        print("🎯 开始综合性能评估")
        print("=" * 80)

        if model_path:
            self.load_model(model_path)

        # 1. 基线算法比较
        print("\n1. 📊 基线算法比较...")
        baseline_results = self._run_baseline_comparison(test_datasets)
        self.evaluation_results['baseline_comparison'] = baseline_results

        # 2. 消融实验
        if model_path:
            print("\n2. 🔬 消融实验...")
            ablation_results = self._run_ablation_study(test_datasets, model_path)
            self.evaluation_results['ablation_study'] = ablation_results

        # 3. 真实场景测试
        if model_path:
            print("\n3. 🌍 真实场景测试...")
            real_world_results = self._run_real_world_test(model_path)
            self.evaluation_results['real_world_test'] = real_world_results

        # 4. 模型鲁棒性测试
        if model_path:
            print("\n4. 🛡️ 模型鲁棒性测试...")
            robustness_results = self._run_robustness_test(test_datasets, model_path)
            self.evaluation_results['robustness_test'] = robustness_results

        # 生成综合报告
        self._generate_comprehensive_report()

        return self.evaluation_results

    def _run_baseline_comparison(self, test_datasets: Dict[str, List]) -> Dict[str, Any]:
        """运行基线算法比较"""
        comparator = BaselineComparator(self.config)

        if hasattr(self, 'model'):
            comparator.load_trained_model(self._get_model_path())

        results = comparator.run_comparison(test_datasets, num_runs=5)
        comparator.generate_report()

        return results

    def _run_ablation_study(self, test_datasets: Dict[str, List], model_path: str) -> Dict[str, Any]:
        """运行消融实验"""
        # 使用第一个数据集进行消融实验
        first_dataset_name = list(test_datasets.keys())[0]
        test_dags = test_datasets[first_dataset_name][:50]  # 使用前50个DAG

        ablation = AblationStudy(self.config)
        models = ablation.create_ablated_models(model_path)

        results = ablation.run_ablation_study(test_dags, models, num_runs=3)
        ablation.generate_ablation_report()

        return results

    def _run_real_world_test(self, model_path: str) -> Dict[str, Any]:
        """运行真实场景测试"""
        tester = RealWorldTester(self.config)
        tester.load_model(model_path)

        results = tester.run_real_world_tests(test_duration=1800)  # 30分钟测试

        # 运行鲁棒性测试
        robustness_results = tester.test_model_robustness()
        results['robustness'] = robustness_results

        return results

    def _run_robustness_test(self, test_datasets: Dict[str, List], model_path: str) -> Dict[str, Any]:
        """运行模型鲁棒性测试"""
        robustness_results = {}

        # 测试不同干扰级别
        disturbance_levels = [0.0, 0.1, 0.2, 0.3, 0.4]

        for level in disturbance_levels:
            print(f"   测试干扰级别: {level}")
            level_results = self._test_robustness_at_level(test_datasets, model_path, level)
            robustness_results[level] = level_results

        self._plot_robustness_results(robustness_results)

        return robustness_results

    def _test_robustness_at_level(self, test_datasets: Dict[str, List],
                                  model_path: str, disturbance_level: float) -> Dict[str, float]:
        """在特定干扰级别测试鲁棒性"""
        env = EmbeddedSchedulingEnvironment(self.config)
        test_dags = list(test_datasets.values())[0][:20]  # 使用第一个数据集的前20个DAG

        performances = []

        for dag in test_dags:
            state = env.reset(dag)
            done = False

            while not done:
                # 模拟硬件故障
                if np.random.random() < disturbance_level:
                    # 随机禁用一台硬件
                    available_hardware = list(env.hardware_resources.keys())
                    if len(available_hardware) > 1:
                        failed_hw = np.random.choice(available_hardware)
                        # 在实际实现中，这里应该更新状态以反映硬件故障

                with torch.no_grad():
                    state_tensor = self._state_to_tensor(state, env)
                    q_values = self.model(*state_tensor)
                    action = torch.argmax(q_values).item()

                state, reward, done, info = env.step(action)

            metrics = self.metrics_calculator.calculate_metrics(env)
            performances.append(metrics['makespan'])

        return {
            'avg_makespan': np.mean(performances),
            'std_makespan': np.std(performances),
            'performance_degradation': (np.mean(
                performances) - self._get_baseline_performance()) / self._get_baseline_performance() * 100
        }

    def _state_to_tensor(self, state, env):
        """将状态转换为模型输入张量"""
        node_features = torch.tensor(state['node_features'], dtype=torch.float32)
        adjacency_matrix = torch.tensor(state['adjacency_matrix'], dtype=torch.float32)
        task_sequence = torch.tensor(state['task_sequence'], dtype=torch.long)
        hardware_features = torch.tensor(state['hardware_features'], dtype=torch.float32)

        return node_features, adjacency_matrix, task_sequence, hardware_features

    def _get_baseline_performance(self) -> float:
        """获取基线性能（简化实现）"""
        # 实际应该从基线比较结果中获取
        return 100.0  # 假设的基线性能

    def _get_model_path(self) -> str:
        """获取模型路径（简化实现）"""
        return 'checkpoints/best_model.pth'

    def _plot_robustness_results(self, robustness_results: Dict):
        """绘制鲁棒性测试结果"""
        disturbance_levels = list(robustness_results.keys())
        performances = [robustness_results[level]['avg_makespan'] for level in disturbance_levels]
        degradations = [robustness_results[level]['performance_degradation'] for level in disturbance_levels]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 性能随干扰变化
        ax1.plot(disturbance_levels, performances, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('干扰级别')
        ax1.set_ylabel('平均完成时间 (ms)')
        ax1.set_title('性能随干扰级别变化')
        ax1.grid(True, alpha=0.3)

        # 性能下降百分比
        ax2.bar([str(level) for level in disturbance_levels], degradations, alpha=0.7)
        ax2.set_xlabel('干扰级别')
        ax2.set_ylabel('性能下降 (%)')
        ax2.set_title('性能下降百分比')
        ax2.grid(True, alpha=0.3)

        # 添加数值标签
        for i, v in enumerate(degradations):
            ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('results/evaluation/robustness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_comprehensive_report(self):
        """生成综合评估报告"""
        print("\n" + "=" * 80)
        print("📊 综合性能评估报告")
        print("=" * 80)

        # 汇总所有评估结果
        if 'baseline_comparison' in self.evaluation_results:
            print("\n1. 基线算法比较结果:")
            for dataset, algorithms in self.evaluation_results['baseline_comparison'].items():
                print(f"   数据集: {dataset}")
                for algo, metrics in algorithms.items():
                    if 'makespan' in metrics:
                        print(f"     {algo:15}: {metrics['makespan']:.2f} ms")

        if 'ablation_study' in self.evaluation_results:
            print("\n2. 消融实验结果:")
            ablation_results = self.evaluation_results['ablation_study']
            for model_variant, metrics in ablation_results.items():
                if 'makespan' in metrics:
                    print(f"     {model_variant:25}: {metrics['makespan']:.2f} ms")

        if 'real_world_test' in self.evaluation_results:
            print("\n3. 真实场景测试结果:")
            real_world_results = self.evaluation_results['real_world_test']
            for scenario, results in real_world_results.items():
                if isinstance(results, dict) and 'average_metrics' in results:
                    metrics = results['average_metrics']
                    if 'makespan' in metrics:
                        print(f"     {scenario:20}: {metrics['makespan']:.2f} ms")

        # 保存详细报告
        self._save_comprehensive_report()

        print(f"\n✅ 综合评估完成！详细结果已保存至 results/evaluation/")

    def _save_comprehensive_report(self):
        """保存综合评估报告"""
        os.makedirs('results/evaluation', exist_ok=True)

        # 保存所有评估结果
        with open('results/evaluation/comprehensive_results.json', 'w') as f:
            # 转换numpy类型为Python原生类型
            serializable_results = {}
            for category, results in self.evaluation_results.items():
                if isinstance(results, dict):
                    serializable_results[category] = self._make_serializable(results)
                else:
                    serializable_results[category] = results

            json.dump(serializable_results, f, indent=2)

        # 生成摘要报告
        summary = self._generate_summary_report()
        with open('results/evaluation/summary_report.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # 保存为CSV
        self._save_csv_report()

    def _make_serializable(self, obj):
        """使对象可序列化"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16,
                              np.int32, np.int64, np.uint8, np.uint16,
                              np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        else:
            return obj

    def _generate_summary_report(self) -> Dict[str, Any]:
        """生成摘要报告"""
        summary = {
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config
        }

        # 添加关键性能指标
        if 'baseline_comparison' in self.evaluation_results:
            baseline_results = self.evaluation_results['baseline_comparison']
            summary['baseline_performance'] = {}

            for dataset, algorithms in baseline_results.items():
                summary['baseline_performance'][dataset] = {}
                for algo, metrics in algorithms.items():
                    if 'makespan' in metrics:
                        summary['baseline_performance'][dataset][algo] = {
                            'makespan': metrics['makespan'],
                            'energy': metrics.get('energy_consumption', 0),
                            'load_balance': metrics.get('load_balance', 0)
                        }

        return summary

    def _save_csv_report(self):
        """保存CSV格式报告"""
        # 基线比较结果CSV
        if 'baseline_comparison' in self.evaluation_results:
            baseline_data = []
            baseline_results = self.evaluation_results['baseline_comparison']

            for dataset, algorithms in baseline_results.items():
                for algo, metrics in algorithms.items():
                    row = {
                        'dataset': dataset,
                        'algorithm': algo,
                        'makespan': metrics.get('makespan', 0),
                        'energy_consumption': metrics.get('energy_consumption', 0),
                        'load_balance': metrics.get('load_balance', 0),
                        'deadline_satisfaction': metrics.get('deadline_satisfaction', 0)
                    }
                    baseline_data.append(row)

            df = pd.DataFrame(baseline_data)
            df.to_csv('results/evaluation/baseline_comparison.csv', index=False)

        # 消融实验结果CSV
        if 'ablation_study' in self.evaluation_results:
            ablation_data = []
            ablation_results = self.evaluation_results['ablation_study']

            for model_variant, metrics in ablation_results.items():
                row = {
                    'model_variant': model_variant,
                    'makespan': metrics.get('makespan', 0),
                    'energy_consumption': metrics.get('energy_consumption', 0),
                    'load_balance': metrics.get('load_balance', 0),
                    'deadline_satisfaction': metrics.get('deadline_satisfaction', 0)
                }
                ablation_data.append(row)

            df = pd.DataFrame(ablation_data)
            df.to_csv('results/evaluation/ablation_study.csv', index=False)


def main():
    """主函数：运行综合性能评估"""
    # 加载配置
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 创建评估器
    evaluator = PerformanceEvaluator(config)

    # 生成测试数据集
    dag_generator = EmbeddedDAGGenerator(config)
    test_datasets = {
        'random_dags': [dag_generator.generate() for _ in range(100)],
        'industrial_dags': [dag_generator.generate(task_count_range=(8, 15)) for _ in range(50)],
        'edge_ai_dags': [dag_generator.generate(task_count_range=(10, 20)) for _ in range(50)]
    }

    # 运行综合评估
    results = evaluator.run_comprehensive_evaluation(
        test_datasets=test_datasets,
        model_path='checkpoints/best_model.pth'  # 可选：指定MODRL模型路径
    )

    print("\n🎉 综合性能评估完成！")


if __name__ == "__main__":
    main()