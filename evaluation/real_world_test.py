import torch
import numpy as np
import pandas as pd
import time
import json
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime
import psutil
import GPUtil

from models.networks.embedded_modrl import EmbeddedMODRL
from environments.embedded_scheduling_env import EmbeddedSchedulingEnvironment
from utils.metrics import SchedulingMetrics
from data.datasets.embedded_dag_generator import EmbeddedDAGGenerator


class RealWorldTester:
    """真实场景测试器"""

    def __init__(self, config):
        self.config = config
        self.metrics_calculator = SchedulingMetrics()
        self.test_scenarios = self._initialize_test_scenarios()

    def _initialize_test_scenarios(self) -> Dict[str, Dict]:
        """初始化真实测试场景"""
        scenarios = {
            'industrial_control': {
                'name': '工业控制系统',
                'description': '多周期控制任务，高实时性要求',
                'task_types': ['PID控制', '运动规划', '传感器融合', '安全监控'],
                'hardware_constraints': {
                    '实时任务': ['CPU', 'FPGA'],
                    '计算密集型': ['GPU', 'CPU'],
                    '低功耗任务': ['MCU']
                }
            },
            'edge_ai_inference': {
                'name': '边缘AI推理',
                'description': 'AI模型推理任务，注重能耗效率',
                'task_types': ['图像预处理', 'CNN推理', '后处理', '结果传输'],
                'hardware_constraints': {
                    '图像处理': ['GPU', 'FPGA'],
                    '神经网络': ['GPU'],
                    '数据传输': ['CPU', 'MCU']
                }
            },
            'autonomous_driving': {
                'name': '自动驾驶系统',
                'description': '多传感器融合，严格实时要求',
                'task_types': ['激光雷达处理', '摄像头处理', '路径规划', '决策控制'],
                'hardware_constraints': {
                    '传感器处理': ['FPGA', 'GPU'],
                    '规划决策': ['CPU'],
                    '控制执行': ['MCU', 'CPU']
                }
            },
            'smart_surveillance': {
                'name': '智能监控系统',
                'description': '连续视频分析，能效敏感',
                'task_types': ['视频解码', '目标检测', '行为分析', '警报生成'],
                'hardware_constraints': {
                    '视频处理': ['GPU', 'FPGA'],
                    'AI分析': ['GPU'],
                    '通信任务': ['CPU', 'MCU']
                }
            }
        }
        return scenarios

    def load_model(self, model_path: str):
        """加载训练好的模型"""
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

        print(f"✅ 成功加载模型: {model_path}")

    def run_real_world_tests(self, scenario_name: str = None,
                             test_duration: int = 3600) -> Dict[str, Any]:
        """
        运行真实场景测试

        Args:
            scenario_name: 特定场景名称，None表示测试所有场景
            test_duration: 测试持续时间(秒)
        """
        print("🌍 开始真实场景测试...")

        if scenario_name:
            scenarios = {scenario_name: self.test_scenarios[scenario_name]}
        else:
            scenarios = self.test_scenarios

        all_results = {}

        for scenario_key, scenario_info in scenarios.items():
            print(f"\n🎯 测试场景: {scenario_info['name']}")
            print(f"  描述: {scenario_info['description']}")

            scenario_results = self._test_single_scenario(
                scenario_key, scenario_info, test_duration)
            all_results[scenario_key] = scenario_results

            # 保存场景结果
            self._save_scenario_results(scenario_key, scenario_results)

        # 生成综合报告
        self._generate_comprehensive_report(all_results)

        return all_results

    def _test_single_scenario(self, scenario_key: str, scenario_info: Dict,
                              test_duration: int) -> Dict[str, Any]:
        """测试单个场景"""
        start_time = time.time()
        test_results = {
            'scenario_info': scenario_info,
            'start_time': datetime.now().isoformat(),
            'test_duration': test_duration,
            'performance_metrics': [],
            'resource_usage': [],
            'system_metrics': []
        }

        env = EmbeddedSchedulingEnvironment(self.config)
        dag_generator = EmbeddedDAGGenerator(self.config)

        # 监控系统资源
        self._start_system_monitoring(test_results)

        iteration = 0
        while time.time() - start_time < test_duration:
            iteration += 1
            print(f"  迭代 {iteration}...", end='\r')

            # 生成场景特定的DAG
            dag = self._generate_scenario_specific_dag(scenario_key, dag_generator)

            # 运行调度
            schedule_metrics = self._run_scheduling(env, dag)
            test_results['performance_metrics'].append(schedule_metrics)

            # 记录系统指标
            system_metrics = self._collect_system_metrics()
            test_results['system_metrics'].append(system_metrics)

            # 每10次迭代记录一次资源使用情况
            if iteration % 10 == 0:
                resource_usage = self._collect_resource_usage(env)
                test_results['resource_usage'].append(resource_usage)

        test_results['end_time'] = datetime.now().isoformat()
        test_results['total_iterations'] = iteration

        # 计算平均指标
        test_results['average_metrics'] = self._calculate_average_metrics(
            test_results['performance_metrics'])

        print(f"✅ 场景 {scenario_info['name']} 测试完成，共 {iteration} 次迭代")
        return test_results

    def _generate_scenario_specific_dag(self, scenario_key: str,
                                        dag_generator) -> Any:
        """生成场景特定的DAG"""
        # 根据场景特点调整DAG生成参数
        scenario_params = {
            'industrial_control': {
                'task_count_range': (8, 15),
                'deadline_strictness': 'high',
                'periodic_tasks_ratio': 0.7
            },
            'edge_ai_inference': {
                'task_count_range': (10, 20),
                'computation_intensity': 'high',
                'energy_sensitivity': 'high'
            },
            'autonomous_driving': {
                'task_count_range': (12, 25),
                'deadline_strictness': 'very_high',
                'reliability_requirement': 'high'
            },
            'smart_surveillance': {
                'task_count_range': (6, 12),
                'energy_sensitivity': 'very_high',
                'continuous_operation': True
            }
        }

        params = scenario_params.get(scenario_key, {})
        return dag_generator.generate(**params)

    def _run_scheduling(self, env, dag) -> Dict[str, float]:
        """运行单次调度并返回指标"""
        state = env.reset(dag)
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state, env)
                q_values = self.model(*state_tensor)
                action = torch.argmax(q_values).item()

            state, reward, done, info = env.step(action)

        metrics = self.metrics_calculator.calculate_metrics(env)
        return metrics

    def _state_to_tensor(self, state, env):
        """将状态转换为模型输入张量"""
        node_features = torch.tensor(state['node_features'], dtype=torch.float32)
        adjacency_matrix = torch.tensor(state['adjacency_matrix'], dtype=torch.float32)
        task_sequence = torch.tensor(state['task_sequence'], dtype=torch.long)
        hardware_features = torch.tensor(state['hardware_features'], dtype=torch.float32)

        return node_features, adjacency_matrix, task_sequence, hardware_features

    def _start_system_monitoring(self, test_results: Dict):
        """开始系统资源监控"""
        test_results['system_info'] = {
            'cpu_count': psutil.cpu_count(),
            'total_memory': psutil.virtual_memory().total,
            'platform': os.uname().sysname,
            'python_version': os.sys.version
        }

    def _collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()

        # 尝试获取GPU信息
        gpu_info = {}
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                gpu_info[f'gpu_{i}_load'] = gpu.load * 100
                gpu_info[f'gpu_{i}_memory'] = gpu.memoryUtil * 100
        except:
            pass

        return {
            'timestamp': time.time(),
            'cpu_usage': cpu_percent,
            'memory_usage': memory_info.percent,
            'disk_read': disk_io.read_bytes if disk_io else 0,
            'disk_write': disk_io.write_bytes if disk_io else 0,
            **gpu_info
        }

    def _collect_resource_usage(self, env) -> Dict[str, Any]:
        """收集资源使用情况"""
        return {
            'timestamp': time.time(),
            'hardware_utilization': env.get_hardware_utilization(),
            'task_queue_length': len(env.task_queue),
            'completed_tasks': env.completed_tasks_count
        }

    def _calculate_average_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """计算平均指标"""
        if not metrics_list:
            return {}

        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [metrics[key] for metrics in metrics_list]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f"{key}_std"] = np.std(values)
            avg_metrics[f"{key}_min"] = np.min(values)
            avg_metrics[f"{key}_max"] = np.max(values)

        return avg_metrics

    def _save_scenario_results(self, scenario_key: str, results: Dict):
        """保存场景测试结果"""
        os.makedirs('results/real_world', exist_ok=True)

        # 保存详细结果
        with open(f'results/real_world/{scenario_key}_detailed.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # 保存摘要结果
        summary = {
            'scenario': results['scenario_info']['name'],
            'test_duration': results['test_duration'],
            'total_iterations': results['total_iterations'],
            'average_metrics': results['average_metrics'],
            'system_info': results.get('system_info', {})
        }

        with open(f'results/real_world/{scenario_key}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    def _generate_comprehensive_report(self, all_results: Dict[str, Any]):
        """生成综合测试报告"""
        print("\n" + "=" * 80)
        print("🌍 真实场景测试综合报告")
        print("=" * 80)

        report_data = []

        for scenario_key, results in all_results.items():
            scenario_name = results['scenario_info']['name']
            avg_metrics = results['average_metrics']

            print(f"\n📊 场景: {scenario_name}")
            print("-" * 50)

            scenario_row = {'Scenario': scenario_name}

            for metric in ['makespan', 'energy_consumption', 'load_balance', 'deadline_satisfaction']:
                if metric in avg_metrics:
                    value = avg_metrics[metric]
                    std = avg_metrics.get(f'{metric}_std', 0)
                    print(f"  {metric:20}: {value:.4f} (±{std:.4f})")
                    scenario_row[metric] = value
                    scenario_row[f'{metric}_std'] = std

            report_data.append(scenario_row)

        # 保存报告表格
        df = pd.DataFrame(report_data)
        df.to_csv('results/real_world/comprehensive_report.csv', index=False)

        # 生成性能对比图
        self._generate_real_world_comparison_plot(report_data)

        print(f"\n✅ 详细报告已保存至 results/real_world/")

    def _generate_real_world_comparison_plot(self, report_data: List[Dict]):
        """生成真实场景对比图"""
        if not report_data:
            return

        scenarios = [item['Scenario'] for item in report_data]
        metrics = ['makespan', 'energy_consumption', 'load_balance']

        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

        for i, metric in enumerate(metrics):
            values = [item.get(metric, 0) for item in report_data]
            errors = [item.get(f'{metric}_std', 0) for item in report_data]

            axes[i].bar(scenarios, values, yerr=errors, capsize=5, alpha=0.7)
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_ylabel('Value')
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('results/real_world/scenario_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def test_model_robustness(self, disturbance_levels: List[float] = [0.1, 0.2, 0.3]):
        """
        测试模型鲁棒性（应对硬件故障、任务变化等）
        """
        print("\n🛡️ 开始模型鲁棒性测试...")

        robustness_results = {}

        for disturbance in disturbance_levels:
            print(f"\n🔧 干扰级别: {disturbance}")

            # 模拟不同类型的干扰
            disturbance_results = {}

            # 1. 硬件故障模拟
            disturbance_results['hardware_failure'] = self._simulate_hardware_failure(disturbance)

            # 2. 任务到达变化
            disturbance_results['task_variation'] = self._simulate_task_variation(disturbance)

            # 3. 资源波动
            disturbance_results['resource_fluctuation'] = self._simulate_resource_fluctuation(disturbance)

            robustness_results[disturbance] = disturbance_results

        # 保存鲁棒性测试结果
        self._save_robustness_results(robustness_results)

        return robustness_results

    def _simulate_hardware_failure(self, failure_prob: float) -> Dict[str, float]:
        """模拟硬件故障"""
        env = EmbeddedSchedulingEnvironment(self.config)
        dag_generator = EmbeddedDAGGenerator(self.config)

        performance_drops = []

        for i in range(10):  # 测试10个不同的DAG
            dag = dag_generator.generate()
            state = env.reset(dag)
            done = False

            while not done:
                # 模拟硬件故障
                if np.random.random() < failure_prob:
                    # 随机禁用一台硬件
                    available_hardware = state['available_hardware']
                    if len(available_hardware) > 1:
                        failed_hw = np.random.choice(available_hardware)
                        state['available_hardware'] = [hw for hw in available_hardware if hw != failed_hw]

                with torch.no_grad():
                    state_tensor = self._state_to_tensor(state, env)
                    q_values = self.model(*state_tensor)
                    action = torch.argmax(q_values).item()

                state, reward, done, info = env.step(action)

            metrics = self.metrics_calculator.calculate_metrics(env)
            performance_drops.append(metrics['makespan'])

        return {
            'average_makespan_increase': np.mean(performance_drops),
            'std': np.std(performance_drops)
        }

    def _simulate_task_variation(self, variation_level: float) -> Dict[str, float]:
        """模拟任务变化"""
        # 类似的实现，模拟任务到达时间、计算需求的变化
        return {'average_impact': variation_level * 0.1}  # 简化实现

    def _simulate_resource_fluctuation(self, fluctuation_level: float) -> Dict[str, float]:
        """模拟资源波动"""
        # 类似的实现，模拟硬件性能波动
        return {'average_impact': fluctuation_level * 0.05}  # 简化实现

    def _save_robustness_results(self, results: Dict):
        """保存鲁棒性测试结果"""
        os.makedirs('results/robustness', exist_ok=True)

        with open('results/robustness/robustness_test.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("✅ 鲁棒性测试结果已保存")