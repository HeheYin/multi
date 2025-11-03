import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import json
from tqdm import tqdm

from environments.embedded_scheduling_env import EmbeddedSchedulingEnvironment
from data.datasets.embedded_dag_generator import EmbeddedDAGGenerator
from utils.metrics import SchedulingMetrics
from utils.logger import ExperimentLogger


class BaselineTrainer:
    """基线算法训练器"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = ExperimentLogger('baseline_training')
        self.metrics_calculator = SchedulingMetrics()

        # 初始化环境
        self.env = EmbeddedSchedulingEnvironment(config)

        # 初始化数据生成器
        self.dag_generator = EmbeddedDAGGenerator(config)

        # 训练结果
        self.training_results = {}
        self.baseline_performance = {}

    def train_heft(self, num_episodes: int = 100) -> Dict[str, Any]:
        """训练HEFT算法（实际上是运行并收集性能数据）"""
        print("🚀 开始HEFT基线性能评估...")

        results = {
            'makespans': [],
            'energy_consumptions': [],
            'load_balances': [],
            'deadline_satisfactions': [],
            'training_time': 0
        }

        start_time = time.time()

        for episode in tqdm(range(num_episodes), desc="HEFT"):
            # 生成随机DAG
            dag = self.dag_generator.generate()

            # 运行HEFT调度
            metrics = self._run_heft_scheduling(dag)

            # 记录结果
            results['makespans'].append(metrics['makespan'])
            results['energy_consumptions'].append(metrics['energy_consumption'])
            results['load_balances'].append(metrics['load_balance'])
            results['deadline_satisfactions'].append(metrics['deadline_satisfaction'])

        results['training_time'] = time.time() - start_time

        # 计算统计信息
        results['avg_makespan'] = np.mean(results['makespans'])
        results['std_makespan'] = np.std(results['makespans'])
        results['avg_energy'] = np.mean(results['energy_consumptions'])
        results['avg_load_balance'] = np.mean(results['load_balances'])
        results['avg_deadline_satisfaction'] = np.mean(results['deadline_satisfactions'])

        self.baseline_performance['HEFT'] = results
        print(f"✅ HEFT评估完成 - 平均完成时间: {results['avg_makespan']:.2f} ms")

        return results

    def _run_heft_scheduling(self, dag) -> Dict[str, float]:
        """运行HEFT调度算法"""
        # 重置环境
        state = self.env.reset(dag)

        # HEFT算法核心逻辑
        task_sequence = self._heft_task_ordering(dag)
        hardware_mapping = self._heft_processor_selection(dag, task_sequence)

        # 执行调度
        done = False
        while not done:
            # 对于HEFT，我们按照预计算的映射执行
            current_task_id = self._get_next_heft_task(task_sequence)
            if current_task_id is None:
                # 没有更多任务，推进时间
                self.env._advance_time()
                done = self.env.is_done()
                continue

            # 获取预分配的处理机
            hardware_idx = hardware_mapping.get(current_task_id, 0)
            action = hardware_idx

            state, reward, done, info = self.env.step(action)

        # 收集指标
        metrics = self.metrics_calculator.calculate_metrics(self.env)
        return metrics

    def _heft_task_ordering(self, dag) -> List[int]:
        """HEFT任务排序（向上排名）"""
        # 计算向上排名
        upward_ranks = self._calculate_upward_ranks(dag)

        # 按向上排名降序排序
        task_order = sorted(upward_ranks.keys(),
                            key=lambda x: upward_ranks[x],
                            reverse=True)

        return task_order

    def _calculate_upward_ranks(self, dag) -> Dict[int, float]:
        """计算向上排名"""
        upward_ranks = {}
        visited = set()

        def compute_rank(task_id):
            if task_id in visited:
                return upward_ranks[task_id]

            visited.add(task_id)

            # 找到后继任务
            successors = self._get_successors(dag, task_id)

            if not successors:
                # 退出任务，排名为平均执行时间
                task = self._get_task_by_id(dag, task_id)
                avg_execution = np.mean(list(task.computation_cost.values())) if task.computation_cost else 10.0
                upward_ranks[task_id] = avg_execution
            else:
                # 排名 = 平均执行时间 + max(后继排名 + 通信成本)
                task = self._get_task_by_id(dag, task_id)
                avg_execution = np.mean(list(task.computation_cost.values())) if task.computation_cost else 10.0

                max_successor_rank = 0
                for succ_id in successors:
                    succ_rank = compute_rank(succ_id)
                    # 简化的通信成本估计
                    comm_cost = 1.0  # 实际应该根据任务间数据量计算
                    max_successor_rank = max(max_successor_rank, succ_rank + comm_cost)

                upward_ranks[task_id] = avg_execution + max_successor_rank

            return upward_ranks[task_id]

        # 从入口任务开始计算
        entry_tasks = self._get_entry_tasks(dag)
        for task_id in entry_tasks:
            compute_rank(task_id)

        return upward_ranks

    def _heft_processor_selection(self, dag, task_sequence: List[int]) -> Dict[int, int]:
        """HEFT处理机选择（最早完成时间）"""
        hardware_mapping = {}
        hardware_availability = {hw: 0.0 for hw in self.env.hardware_resources.keys()}

        for task_id in task_sequence:
            task = self._get_task_by_id(dag, task_id)
            best_hardware = None
            earliest_finish = float('inf')

            # 为任务选择最早完成的处理机
            for hw_type, hw_info in self.env.hardware_resources.items():
                # 计算最早开始时间
                est = self._calculate_est(task_id, hw_type, hardware_availability, dag)

                # 计算执行时间
                exec_time = task.computation_cost.get(hw_type, 10.0) if task.computation_cost else 10.0

                # 完成时间
                eft = est + exec_time

                if eft < earliest_finish:
                    earliest_finish = eft
                    best_hardware = hw_type

            # 记录分配
            if best_hardware is not None:
                hw_idx = list(self.env.hardware_resources.keys()).index(best_hardware)
                hardware_mapping[task_id] = hw_idx
                hardware_availability[best_hardware] = earliest_finish

        return hardware_mapping

    def _calculate_est(self, task_id: int, hw_type: str, hardware_availability: Dict, dag) -> float:
        """计算最早开始时间"""
        task = self._get_task_by_id(dag, task_id)

        # 硬件可用时间
        hw_available = hardware_availability[hw_type]

        # 依赖任务的最晚完成时间
        est_from_dependencies = 0.0
        if task.data_dependencies:
            for dep_id in task.data_dependencies:
                dep_task = self._get_task_by_id(dag, dep_id)
                # 简化的通信时间估计
                comm_time = 1.0  # 实际应根据硬件间通信成本计算
                est_from_dependencies = max(est_from_dependencies,
                                            hardware_availability.get(hw_type, 0) + comm_time)

        return max(hw_available, est_from_dependencies)

    def _get_successors(self, dag, task_id: int) -> List[int]:
        """获取后继任务"""
        successors = []
        for edge in dag.edges:
            if edge[0] == task_id:  # source == task_id
                successors.append(edge[1])  # target
        return successors

    def _get_entry_tasks(self, dag) -> List[int]:
        """获取入口任务（没有依赖的任务）"""
        all_tasks = set(node.node_id for node in dag.nodes)
        dependent_tasks = set()

        for edge in dag.edges:
            dependent_tasks.add(edge[1])  # 目标任务有依赖

        entry_tasks = list(all_tasks - dependent_tasks)
        return entry_tasks

    def _get_task_by_id(self, dag, task_id: int):
        """根据ID获取任务节点"""
        for node in dag.nodes:
            if node.node_id == task_id:
                return node
        return None

    def _get_next_heft_task(self, task_sequence: List[int]) -> Optional[int]:
        """获取下一个可调度的HEFT任务"""
        for task_id in task_sequence:
            task = self.env._get_task_by_id(task_id)
            if (task and task['state'] == self.env.TaskState.WAITING and
                    self.env._check_task_dependencies(task)):
                return task_id
        return None

    def train_cpop(self, num_episodes: int = 100) -> Dict[str, Any]:
        """训练CPOP算法"""
        print("🚀 开始CPOP基线性能评估...")

        results = {
            'makespans': [],
            'energy_consumptions': [],
            'load_balances': [],
            'deadline_satisfactions': [],
            'training_time': 0
        }

        start_time = time.time()

        for episode in tqdm(range(num_episodes), desc="CPOP"):
            dag = self.dag_generator.generate()
            metrics = self._run_cpop_scheduling(dag)

            results['makespans'].append(metrics['makespan'])
            results['energy_consumptions'].append(metrics['energy_consumption'])
            results['load_balances'].append(metrics['load_balance'])
            results['deadline_satisfactions'].append(metrics['deadline_satisfaction'])

        results['training_time'] = time.time() - start_time

        # 计算统计信息
        results['avg_makespan'] = np.mean(results['makespans'])
        results['std_makespan'] = np.std(results['makespans'])
        results['avg_energy'] = np.mean(results['energy_consumptions'])
        results['avg_load_balance'] = np.mean(results['load_balances'])
        results['avg_deadline_satisfaction'] = np.mean(results['deadline_satisfactions'])

        self.baseline_performance['CPOP'] = results
        print(f"✅ CPOP评估完成 - 平均完成时间: {results['avg_makespan']:.2f} ms")

        return results

    def _run_cpop_scheduling(self, dag) -> Dict[str, float]:
        """运行CPOP调度算法"""
        # CPOP实现类似HEFT，但关键路径任务分配到专用处理器
        # 这里简化实现，实际CPOP更复杂
        return self._run_heft_scheduling(dag)

    def train_random(self, num_episodes: int = 100) -> Dict[str, Any]:
        """随机调度算法"""
        print("🚀 开始随机调度算法评估...")

        results = {
            'makespans': [],
            'energy_consumptions': [],
            'load_balances': [],
            'deadline_satisfactions': [],
            'training_time': 0
        }

        start_time = time.time()

        for episode in tqdm(range(num_episodes), desc="Random"):
            dag = self.dag_generator.generate()
            state = self.env.reset(dag)

            done = False
            while not done:
                # 随机选择硬件
                available_actions = self.env.get_available_actions()
                action = np.random.choice(available_actions)

                state, reward, done, info = self.env.step(action)

            metrics = self.metrics_calculator.calculate_metrics(self.env)

            results['makespans'].append(metrics['makespan'])
            results['energy_consumptions'].append(metrics['energy_consumption'])
            results['load_balances'].append(metrics['load_balance'])
            results['deadline_satisfactions'].append(metrics['deadline_satisfaction'])

        results['training_time'] = time.time() - start_time

        # 计算统计信息
        results['avg_makespan'] = np.mean(results['makespans'])
        results['std_makespan'] = np.std(results['makespans'])
        results['avg_energy'] = np.mean(results['energy_consumptions'])
        results['avg_load_balance'] = np.mean(results['load_balances'])
        results['avg_deadline_satisfaction'] = np.mean(results['deadline_satisfactions'])

        self.baseline_performance['Random'] = results
        print(f"✅ 随机调度评估完成 - 平均完成时间: {results['avg_makespan']:.2f} ms")

        return results

    def run_all_baselines(self, num_episodes: int = 100) -> Dict[str, Any]:
        """运行所有基线算法"""
        print("=" * 80)
        print("🎯 开始所有基线算法性能评估")
        print("=" * 80)

        # 运行各种基线算法
        self.train_heft(num_episodes)
        self.train_cpop(num_episodes)
        self.train_random(num_episodes)

        # 生成比较报告
        self.generate_baseline_report()

        return self.baseline_performance

    def generate_baseline_report(self):
        """生成基线算法性能报告"""
        if not self.baseline_performance:
            print("⚠️ 没有可用的基线性能数据")
            return

        print("\n" + "=" * 80)
        print("📊 基线算法性能比较报告")
        print("=" * 80)

        # 创建比较表格
        comparison_data = []

        for algo_name, results in self.baseline_performance.items():
            row = {
                'Algorithm': algo_name,
                'Avg Makespan': f"{results['avg_makespan']:.2f} ± {results['std_makespan']:.2f}",
                'Avg Energy': f"{results['avg_energy']:.2f}",
                'Avg Load Balance': f"{results['avg_load_balance']:.3f}",
                'Avg Deadline Satisfaction': f"{results['avg_deadline_satisfaction']:.3f}",
                'Training Time (s)': f"{results['training_time']:.2f}"
            }
            comparison_data.append(row)

        # 打印表格
        df = pd.DataFrame(comparison_data)
        print("\n性能比较:")
        print(df.to_string(index=False))

        # 保存结果
        self._save_baseline_results()

        # 生成可视化图表
        self._plot_baseline_comparison()

    def _save_baseline_results(self):
        """保存基线结果"""
        os.makedirs('results/baselines', exist_ok=True)

        # 保存详细结果
        with open('results/baselines/baseline_performance.json', 'w') as f:
            # 转换numpy类型为Python原生类型
            serializable_results = {}
            for algo, results in self.baseline_performance.items():
                serializable_results[algo] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in results.items()
                }
            json.dump(serializable_results, f, indent=2)

        # 保存摘要
        summary_data = []
        for algo, results in self.baseline_performance.items():
            summary_data.append({
                'algorithm': algo,
                'avg_makespan': results['avg_makespan'],
                'std_makespan': results['std_makespan'],
                'avg_energy': results['avg_energy'],
                'avg_load_balance': results['avg_load_balance'],
                'avg_deadline_satisfaction': results['avg_deadline_satisfaction'],
                'training_time': results['training_time']
            })

        df = pd.DataFrame(summary_data)
        df.to_csv('results/baselines/baseline_summary.csv', index=False)

        print("✅ 基线结果已保存至 results/baselines/")

    def _plot_baseline_comparison(self):
        """绘制基线算法比较图"""
        if not self.baseline_performance:
            return

        algorithms = list(self.baseline_performance.keys())

        # 创建多个子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Makespan比较
        makespans = [self.baseline_performance[algo]['avg_makespan'] for algo in algorithms]
        stds = [self.baseline_performance[algo]['std_makespan'] for algo in algorithms]

        bars = axes[0, 0].bar(algorithms, makespans, yerr=stds, capsize=5, alpha=0.7)
        axes[0, 0].set_title('平均完成时间比较')
        axes[0, 0].set_ylabel('完成时间 (ms)')

        # 添加数值标签
        for bar, value in zip(bars, makespans):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                            f'{value:.1f}', ha='center', va='bottom')

        # 能耗比较
        energies = [self.baseline_performance[algo]['avg_energy'] for algo in algorithms]
        axes[0, 1].bar(algorithms, energies, alpha=0.7, color='orange')
        axes[0, 1].set_title('平均能耗比较')
        axes[0, 1].set_ylabel('能耗')

        # 负载均衡比较
        load_balances = [self.baseline_performance[algo]['avg_load_balance'] for algo in algorithms]
        axes[1, 0].bar(algorithms, load_balances, alpha=0.7, color='green')
        axes[1, 0].set_title('平均负载均衡比较')
        axes[1, 0].set_ylabel('负载均衡指标')

        # 截止时间满足率比较
        deadlines = [self.baseline_performance[algo]['avg_deadline_satisfaction'] for algo in algorithms]
        axes[1, 1].bar(algorithms, deadlines, alpha=0.7, color='red')
        axes[1, 1].set_title('平均截止时间满足率比较')
        axes[1, 1].set_ylabel('满足率')

        plt.tight_layout()
        plt.savefig('results/baselines/baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ 基线比较图表已保存")


def main():
    """主函数：运行基线算法训练"""
    import yaml

    # 加载配置
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 创建训练器
    trainer = BaselineTrainer(config)

    # 运行所有基线算法
    results = trainer.run_all_baselines(num_episodes=50)

    print("\n🎉 基线算法训练完成！")


if __name__ == "__main__":
    main()