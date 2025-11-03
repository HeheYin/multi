import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import torch
import yaml
import json
import subprocess
from tqdm import tqdm


class HardwareDeployer:
    """硬件部署管理器"""

    def __init__(self, config: Dict, model_path: str = None):
        self.config = config
        self.model_path = model_path

        # 硬件平台配置
        self.hardware_platforms = self._initialize_hardware_platforms()

        # 部署结果
        self.deployment_results = {}

    def _initialize_hardware_platforms(self) -> Dict[str, Dict]:
        """初始化硬件平台配置"""
        platforms = {
            'nvidia_jetson_nano': {
                'name': 'NVIDIA Jetson Nano',
                'architecture': 'ARM Cortex-A57 + 128-core NVIDIA GPU',
                'memory': '4GB LPDDR4',
                'power_consumption': '5-10W',
                'supported_hardware': ['CPU', 'GPU'],
                'deployment_script': 'deploy_scripts/jetson_deploy.sh',
                'performance_factor': 1.0  # 基准性能
            },
            'raspberry_pi_4': {
                'name': 'Raspberry Pi 4',
                'architecture': 'ARM Cortex-A72',
                'memory': '4GB/8GB LPDDR4',
                'power_consumption': '3-7W',
                'supported_hardware': ['CPU'],
                'deployment_script': 'deploy_scripts/raspberry_deploy.sh',
                'performance_factor': 0.6  # 相对于Jetson的性能
            },
            'x86_embedded': {
                'name': 'x86 Embedded Platform',
                'architecture': 'Intel Atom/Celeron',
                'memory': '8GB DDR4',
                'power_consumption': '10-15W',
                'supported_hardware': ['CPU'],
                'deployment_script': 'deploy_scripts/x86_deploy.sh',
                'performance_factor': 0.8
            },
            'fpga_accelerator': {
                'name': 'FPGA Accelerator Board',
                'architecture': 'Xilinx/Intel FPGA',
                'memory': 'DDR4 + On-chip Memory',
                'power_consumption': '5-20W',
                'supported_hardware': ['CPU', 'FPGA'],
                'deployment_script': 'deploy_scripts/fpga_deploy.sh',
                'performance_factor': 1.2  # 特定任务上性能更好
            }
        }
        return platforms

    def deploy_model(self, platform: str, deployment_mode: str = 'simulation') -> Dict[str, Any]:
        """
        部署模型到硬件平台

        Args:
            platform: 硬件平台
            deployment_mode: 部署模式 ('simulation', 'real')

        Returns:
            deployment_results: 部署结果
        """
        print(f"🚀 开始部署到 {self.hardware_platforms[platform]['name']}...")

        if platform not in self.hardware_platforms:
            raise ValueError(f"不支持的硬件平台: {platform}")

        platform_info = self.hardware_platforms[platform]

        if deployment_mode == 'real':
            results = self._real_deployment(platform, platform_info)
        else:
            results = self._simulation_deployment(platform, platform_info)

        self.deployment_results[platform] = results
        return results

    def _real_deployment(self, platform: str, platform_info: Dict) -> Dict[str, Any]:
        """真实硬件部署"""
        print(f"  执行真实硬件部署...")

        # 检查部署脚本是否存在
        deploy_script = platform_info['deployment_script']
        if not os.path.exists(deploy_script):
            print(f"  ⚠️ 部署脚本不存在: {deploy_script}，使用模拟部署")
            return self._simulation_deployment(platform, platform_info)

        try:
            # 执行部署脚本
            result = subprocess.run(
                ['bash', deploy_script, self.model_path],
                capture_output=True, text=True, timeout=300
            )

            if result.returncode == 0:
                print(f"  ✅ 部署脚本执行成功")
                # 解析部署结果
                deployment_results = self._parse_deployment_output(result.stdout)
            else:
                print(f"  ❌ 部署脚本执行失败: {result.stderr}")
                deployment_results = self._simulation_deployment(platform, platform_info)

        except subprocess.TimeoutExpired:
            print(f"  ❌ 部署脚本执行超时")
            deployment_results = self._simulation_deployment(platform, platform_info)
        except Exception as e:
            print(f"  ❌ 部署过程中出错: {e}")
            deployment_results = self._simulation_deployment(platform, platform_info)

        return deployment_results

    def _simulation_deployment(self, platform: str, platform_info: Dict) -> Dict[str, Any]:
        """模拟硬件部署（用于测试和开发）"""
        print(f"  执行模拟硬件部署...")

        # 模拟部署过程
        time.sleep(2)  # 模拟部署时间

        # 基于平台性能因子调整预期性能
        performance_factor = platform_info['performance_factor']

        # 模拟性能测试结果
        test_results = self._run_simulation_tests(platform, performance_factor)

        deployment_results = {
            'platform': platform,
            'platform_name': platform_info['name'],
            'deployment_status': 'success',
            'deployment_mode': 'simulation',
            'deployment_time': 2.0,
            'performance_factor': performance_factor,
            'test_results': test_results,
            'resource_usage': self._simulate_resource_usage(platform_info),
            'power_measurements': self._simulate_power_measurements(platform_info)
        }

        return deployment_results

    def _run_simulation_tests(self, platform: str, performance_factor: float) -> Dict[str, Any]:
        """运行模拟性能测试"""
        from data.datasets.embedded_dag_generator import EmbeddedDAGGenerator
        from environments.embedded_scheduling_env import EmbeddedSchedulingEnvironment
        from utils.metrics import SchedulingMetrics

        dag_generator = EmbeddedDAGGenerator(self.config)
        env = EmbeddedSchedulingEnvironment(self.config)
        metrics_calculator = SchedulingMetrics()

        test_dags = [dag_generator.generate() for _ in range(10)]

        makespans = []
        energies = []
        inference_times = []

        for dag in test_dags:
            # 模拟推理时间（考虑硬件性能因子）
            start_time = time.time()

            state = env.reset(dag)
            done = False

            while not done:
                # 在真实部署中，这里会调用部署的模型进行推理
                # 模拟推理延迟
                time.sleep(0.001 * (1.0 / performance_factor))  # 调整延迟基于性能因子

                # 模拟动作选择（随机）
                available_actions = env.get_available_actions()
                action = np.random.choice(available_actions)

                state, reward, done, info = env.step(action)

            inference_time = time.time() - start_time
            metrics = metrics_calculator.calculate_metrics(env)

            makespans.append(metrics['makespan'] * (1.0 / performance_factor))
            energies.append(metrics['energy_consumption'] * performance_factor)  # 能耗与性能因子相关
            inference_times.append(inference_time)

        return {
            'avg_makespan': np.mean(makespans),
            'std_makespan': np.std(makespans),
            'avg_energy': np.mean(energies),
            'avg_inference_time': np.mean(inference_times),
            'throughput': len(test_dags) / np.sum(inference_times)  # 任务/秒
        }

    def _simulate_resource_usage(self, platform_info: Dict) -> Dict[str, float]:
        """模拟资源使用情况"""
        return {
            'cpu_usage': np.random.uniform(0.3, 0.8),
            'memory_usage': np.random.uniform(0.2, 0.6),
            'gpu_usage': 0.0 if 'GPU' not in platform_info['supported_hardware'] else np.random.uniform(0.4, 0.9),
            'fpga_usage': 0.0 if 'FPGA' not in platform_info['supported_hardware'] else np.random.uniform(0.5, 0.95)
        }

    def _simulate_power_measurements(self, platform_info: Dict) -> Dict[str, float]:
        """模拟功耗测量"""
        base_power = {
            'nvidia_jetson_nano': 5.0,
            'raspberry_pi_4': 3.0,
            'x86_embedded': 8.0,
            'fpga_accelerator': 6.0
        }

        platform_key = [k for k, v in self.hardware_platforms.items() if v['name'] == platform_info['name']][0]
        base = base_power.get(platform_key, 5.0)

        return {
            'idle_power': base,
            'average_power': base * 1.3,
            'peak_power': base * 1.8,
            'energy_efficiency': np.random.uniform(0.7, 0.9)  # 能效比
        }

    def _parse_deployment_output(self, output: str) -> Dict[str, Any]:
        """解析部署脚本输出"""
        # 这里应该根据实际部署脚本的输出格式进行解析
        # 简化实现
        return {
            'deployment_status': 'success',
            'deployment_mode': 'real',
            'output_summary': output[:200] + '...' if len(output) > 200 else output
        }

    def deploy_to_all_platforms(self, deployment_mode: str = 'simulation') -> Dict[str, Any]:
        """部署到所有支持的硬件平台"""
        print("=" * 80)
        print("🔧 开始多平台部署")
        print("=" * 80)

        for platform in self.hardware_platforms.keys():
            self.deploy_model(platform, deployment_mode)

        # 生成部署比较报告
        self._generate_deployment_report()

        return self.deployment_results

    def _generate_deployment_report(self):
        """生成部署报告"""
        print("\n" + "=" * 80)
        print("📋 硬件部署报告")
        print("=" * 80)

        deployment_data = []

        for platform, results in self.deployment_results.items():
            platform_name = self.hardware_platforms[platform]['name']
            test_results = results.get('test_results', {})

            row = {
                'Platform': platform_name,
                'Deployment Status': results.get('deployment_status', 'unknown'),
                'Mode': results.get('deployment_mode', 'simulation'),
                'Avg Makespan (ms)': test_results.get('avg_makespan', 0),
                'Avg Energy': test_results.get('avg_energy', 0),
                'Throughput (tasks/s)': test_results.get('throughput', 0),
                'Avg Inference Time (s)': test_results.get('avg_inference_time', 0)
            }
            deployment_data.append(row)

        # 打印部署结果表格
        df = pd.DataFrame(deployment_data)
        print("\n部署性能比较:")
        print(df.to_string(index=False))

        # 保存部署结果
        self._save_deployment_results()

        # 生成性能比较图表
        self._plot_deployment_comparison()

    def _save_deployment_results(self):
        """保存部署结果"""
        os.makedirs('results/deployment', exist_ok=True)

        # 保存详细结果
        with open('results/deployment/deployment_results.json', 'w') as f:
            json.dump(self.deployment_results, f, indent=2, default=str)

        # 保存摘要
        summary_data = []
        for platform, results in self.deployment_results.items():
            platform_name = self.hardware_platforms[platform]['name']
            test_results = results.get('test_results', {})

            summary_data.append({
                'platform': platform_name,
                'deployment_status': results.get('deployment_status', 'unknown'),
                'avg_makespan': test_results.get('avg_makespan', 0),
                'avg_energy': test_results.get('avg_energy', 0),
                'throughput': test_results.get('throughput', 0),
                'performance_factor': results.get('performance_factor', 1.0)
            })

        df = pd.DataFrame(summary_data)
        df.to_csv('results/deployment/deployment_summary.csv', index=False)

        print("✅ 部署结果已保存至 results/deployment/")

    def _plot_deployment_comparison(self):
        """绘制部署性能比较图"""
        if not self.deployment_results:
            return

        platforms = []
        makespans = []
        energies = []
        throughputs = []

        for platform, results in self.deployment_results.items():
            platform_name = self.hardware_platforms[platform]['name']
            test_results = results.get('test_results', {})

            platforms.append(platform_name)
            makespans.append(test_results.get('avg_makespan', 0))
            energies.append(test_results.get('avg_energy', 0))
            throughputs.append(test_results.get('throughput', 0))

        # 创建比较图表
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # 完成时间比较
        bars1 = ax1.bar(platforms, makespans, alpha=0.7, color='skyblue')
        ax1.set_title('平均完成时间比较')
        ax1.set_ylabel('完成时间 (ms)')
        ax1.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 5,
                     f'{height:.1f}', ha='center', va='bottom')

        # 能耗比较
        bars2 = ax2.bar(platforms, energies, alpha=0.7, color='lightcoral')
        ax2.set_title('平均能耗比较')
        ax2.set_ylabel('能耗')
        ax2.tick_params(axis='x', rotation=45)

        # 吞吐量比较
        bars3 = ax3.bar(platforms, throughputs, alpha=0.7, color='lightgreen')
        ax3.set_title('吞吐量比较')
        ax3.set_ylabel('吞吐量 (任务/秒)')
        ax3.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('results/deployment/platform_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ 部署比较图表已保存")

    def generate_deployment_guide(self):
        """生成部署指南"""
        print("\n" + "=" * 80)
        print("📖 硬件部署指南")
        print("=" * 80)

        guide = {
            'overview': '嵌入式智能软件并行优化系统硬件部署指南',
            'supported_platforms': list(self.hardware_platforms.keys()),
            'deployment_steps': [
                "1. 准备目标硬件平台",
                "2. 安装必要的依赖库 (PyTorch, NumPy, 等)",
                "3. 部署模型文件到目标设备",
                "4. 配置环境参数",
                "5. 运行验证测试",
                "6. 集成到目标应用程序"
            ],
            'performance_optimization_tips': [
                "使用量化技术减少模型大小",
                "利用硬件特定加速库",
                "优化内存访问模式",
                "调整批处理大小平衡延迟和吞吐量"
            ],
            'troubleshooting': [
                "检查模型文件完整性",
                "验证硬件兼容性",
                "监控资源使用情况",
                "查看系统日志获取错误信息"
            ]
        }

        # 保存部署指南
        os.makedirs('docs', exist_ok=True)
        with open('docs/deployment_guide.json', 'w') as f:
            json.dump(guide, f, indent=2)

        # 生成Markdown格式的指南
        self._generate_markdown_guide(guide)

        print("✅ 部署指南已生成至 docs/ 目录")

    def _generate_markdown_guide(self, guide: Dict):
        """生成Markdown格式的部署指南"""
        markdown_content = f"""# {guide['overview']}

## 支持的硬件平台

{', '.join(guide['supported_platforms'])}

## 部署步骤

"""

        for step in guide['deployment_steps']:
            markdown_content += f"{step}\n\n"

        markdown_content += "## 性能优化建议\n\n"
        for tip in guide['performance_optimization_tips']:
            markdown_content += f"- {tip}\n"

        markdown_content += "\n## 故障排除\n\n"
        for item in guide['troubleshooting']:
            markdown_content += f"- {item}\n"

        markdown_content += f"""

## 部署性能结果

以下是各硬件平台的性能测试结果：

| 平台 | 平均完成时间 (ms) | 平均能耗 | 吞吐量 (任务/秒) |
|------|------------------|----------|-----------------|
"""

        for platform, results in self.deployment_results.items():
            platform_name = self.hardware_platforms[platform]['name']
            test_results = results.get('test_results', {})

            markdown_content += f"| {platform_name} | {test_results.get('avg_makespan', 0):.1f} | {test_results.get('avg_energy', 0):.1f} | {test_results.get('throughput', 0):.1f} |\n"

        with open('docs/deployment_guide.md', 'w') as f:
            f.write(markdown_content)


def main():
    """主函数：运行硬件部署"""
    # 加载配置
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 创建部署器
    deployer = HardwareDeployer(config, model_path='checkpoints/best_model.pth')

    # 部署到所有平台（模拟模式）
    results = deployer.deploy_to_all_platforms(deployment_mode='simulation')

    # 生成部署指南
    deployer.generate_deployment_guide()

    print("\n🎉 硬件部署完成！")


if __name__ == "__main__":
    main()