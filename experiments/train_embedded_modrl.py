import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import torch
import yaml
from tqdm import tqdm

from agents.d3qn_agent import D3QNAgent
from models.networks.embedded_modrl import EmbeddedMODRL
from environments.embedded_scheduling_env import EmbeddedSchedulingEnvironment
from environments.dynamic_task_env import DynamicTaskEnvironment
from environments.multi_software_env import MultiSoftwareEnvironment
from data.datasets.embedded_dag_generator import EmbeddedDAGGenerator
from utils.logger import ExperimentLogger
from utils.metrics import SchedulingMetrics


class MODRLTrainer:
    """MODRL模型训练器"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = ExperimentLogger('modrl_training')
        self.metrics_calculator = SchedulingMetrics()

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 使用设备: {self.device}")

        # 初始化模型
        self.model, self.target_model = self._initialize_models()

        # 初始化智能体
        self.agent = self._initialize_agent()

        # 初始化环境
        self.env = self._initialize_environment()

        # 初始化数据生成器
        self.dag_generator = EmbeddedDAGGenerator(config)

        # 训练记录
        self.training_history = {
            'episode_rewards': [],
            'episode_losses': [],
            'episode_makespans': [],
            'episode_energies': [],
            'epsilon_history': []
        }

        # 最佳模型跟踪
        self.best_reward = -float('inf')
        self.best_model_path = None

    def _initialize_models(self) -> Tuple[EmbeddedMODRL, EmbeddedMODRL]:
        """初始化模型"""
        model_config = self.config['model']

        model = EmbeddedMODRL(
            node_feature_dim=model_config['node_feature_dim'],
            hardware_feature_dim=model_config['hardware_feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_hardware=model_config['num_hardware'],
            num_actions=model_config['num_actions']
        ).to(self.device)

        target_model = EmbeddedMODRL(
            node_feature_dim=model_config['node_feature_dim'],
            hardware_feature_dim=model_config['hardware_feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_hardware=model_config['num_hardware'],
            num_actions=model_config['num_actions']
        ).to(self.device)

        print(f"✅ 模型初始化完成")
        print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")

        return model, target_model

    def _initialize_agent(self) -> D3QNAgent:
        """初始化智能体"""
        training_config = self.config['training']
        agent = D3QNAgent(self.model, self.target_model, training_config)

        print(f"✅ D3QN智能体初始化完成")
        print(f"   记忆容量: {training_config.get('memory_size', 10000)}")
        print(f"   批次大小: {training_config.get('batch_size', 32)}")

        return agent

    def _initialize_environment(self):
        """初始化训练环境"""
        env_type = self.config.get('environment_type', 'embedded')

        if env_type == 'dynamic':
            env = DynamicTaskEnvironment(self.config)
            print("✅ 动态任务环境初始化完成")
        elif env_type == 'multi_software':
            env = MultiSoftwareEnvironment(self.config)
            print("✅ 多软件环境初始化完成")
        else:
            env = EmbeddedSchedulingEnvironment(self.config)
            print("✅ 嵌入式调度环境初始化完成")

        return env

    def train(self, num_episodes: int = 1000,
              save_interval: int = 100,
              eval_interval: int = 50) -> Dict[str, Any]:
        """
        训练MODRL模型

        Args:
            num_episodes: 训练回合数
            save_interval: 模型保存间隔
            eval_interval: 评估间隔

        Returns:
            training_results: 训练结果
        """
        print("=" * 80)
        print("🎯 开始MODRL模型训练")
        print("=" * 80)

        start_time = time.time()

        # 训练进度条
        pbar = tqdm(range(num_episodes), desc="训练进度")

        for episode in pbar:
            # 生成新的DAG
            dag = self.dag_generator.generate()

            # 重置环境
            state = self.env.reset(dag)
            self.agent.reset_episode()

            episode_reward = 0
            episode_losses = []
            done = False

            while not done:
                # 选择动作
                action = self.agent.act(state, training=True)

                # 执行动作
                next_state, reward, done, info = self.env.step(action)

                # 存储经验
                self.agent.remember(state, action, reward, next_state, done)

                # 经验回放学习
                loss = self.agent.replay()
                if loss is not None:
                    episode_losses.append(loss)

                episode_reward += reward
                state = next_state

            # 记录训练数据
            metrics = self.metrics_calculator.calculate_metrics(self.env)

            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_losses'].append(
                np.mean(episode_losses) if episode_losses else 0.0
            )
            self.training_history['episode_makespans'].append(metrics['makespan'])
            self.training_history['episode_energies'].append(metrics['energy_consumption'])
            self.training_history['epsilon_history'].append(self.agent.epsilon)

            # 更新进度条
            avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
            avg_loss = np.mean(self.training_history['episode_losses'][-10:])
            pbar.set_postfix({
                'Avg Reward': f'{avg_reward:.2f}',
                'Avg Loss': f'{avg_loss:.4f}',
                'Epsilon': f'{self.agent.epsilon:.3f}'
            })

            # 定期评估
            if episode % eval_interval == 0:
                self._evaluate_model(episode)

            # 保存最佳模型
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self._save_model(episode, is_best=True)

            # 定期保存检查点
            if episode % save_interval == 0:
                self._save_model(episode, is_best=False)
                self._save_training_history()

        # 训练完成
        training_time = time.time() - start_time

        print(f"\n✅ 训练完成！总时间: {training_time:.2f} 秒")
        print(f"   最佳奖励: {self.best_reward:.2f}")
        print(f"   平均奖励: {np.mean(self.training_history['episode_rewards']):.2f}")

        # 保存最终模型
        self._save_model(num_episodes, is_final=True)

        # 生成训练报告
        self._generate_training_report(training_time)

        return {
            'training_time': training_time,
            'best_reward': self.best_reward,
            'final_epsilon': self.agent.epsilon,
            'training_history': self.training_history
        }

    def _evaluate_model(self, episode: int):
        """在评估集上评估模型"""
        print(f"\n📊 第 {episode} 回合模型评估...")

        eval_episodes = 10
        eval_rewards = []
        eval_makespans = []
        eval_energies = []

        # 切换到评估模式
        self.agent.eval_mode()

        for _ in range(eval_episodes):
            dag = self.dag_generator.generate()
            state = self.env.reset(dag)

            episode_reward = 0
            done = False

            while not done:
                action = self.agent.act(state, training=False)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward

            metrics = self.metrics_calculator.calculate_metrics(self.env)

            eval_rewards.append(episode_reward)
            eval_makespans.append(metrics['makespan'])
            eval_energies.append(metrics['energy_consumption'])

        # 切换回训练模式
        self.agent.train_mode()

        avg_reward = np.mean(eval_rewards)
        avg_makespan = np.mean(eval_makespans)
        avg_energy = np.mean(eval_energies)

        print(f"   评估结果 - 平均奖励: {avg_reward:.2f}, "
              f"平均完成时间: {avg_makespan:.2f} ms, "
              f"平均能耗: {avg_energy:.2f}")

        # 记录评估结果
        self.training_history.setdefault('eval_rewards', []).append(avg_reward)
        self.training_history.setdefault('eval_makespans', []).append(avg_makespan)
        self.training_history.setdefault('eval_energies', []).append(avg_energy)

    def _save_model(self, episode: int, is_best: bool = False, is_final: bool = False):
        """保存模型"""
        os.makedirs('checkpoints', exist_ok=True)

        if is_best:
            filename = f'checkpoints/best_model.pth'
            print(f"💾 保存最佳模型: {filename}")
        elif is_final:
            filename = f'checkpoints/final_model_ep{episode}.pth'
            print(f"💾 保存最终模型: {filename}")
        else:
            filename = f'checkpoints/checkpoint_ep{episode}.pth'

        # 保存模型和智能体状态
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'agent_config': self.agent.get_training_info(),
            'training_config': self.config
        }

        torch.save(checkpoint, filename)

        if is_best:
            self.best_model_path = filename

    def _save_training_history(self):
        """保存训练历史"""
        os.makedirs('results/training', exist_ok=True)

        # 转换为可序列化格式
        history_serializable = {}
        for key, values in self.training_history.items():
            history_serializable[key] = [float(v) for v in values]

        with open('results/training/training_history.json', 'w') as f:
            json.dump(history_serializable, f, indent=2)

        # 保存为CSV
        df = pd.DataFrame({
            'episode': range(len(self.training_history['episode_rewards'])),
            'reward': self.training_history['episode_rewards'],
            'loss': self.training_history['episode_losses'],
            'makespan': self.training_history['episode_makespans'],
            'energy': self.training_history['episode_energies'],
            'epsilon': self.training_history['epsilon_history']
        })
        df.to_csv('results/training/training_history.csv', index=False)

    def _generate_training_report(self, training_time: float):
        """生成训练报告"""
        print("\n" + "=" * 80)
        print("📈 MODRL训练报告")
        print("=" * 80)

        # 训练统计
        final_rewards = self.training_history['episode_rewards'][-10:]
        final_losses = self.training_history['episode_losses'][-10:]

        print(f"\n📊 训练统计:")
        print(f"   总回合数: {len(self.training_history['episode_rewards'])}")
        print(f"   训练时间: {training_time:.2f} 秒")
        print(f"   最终平均奖励: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
        print(f"   最终平均损失: {np.mean(final_losses):.4f} ± {np.std(final_losses):.4f}")
        print(f"   最终探索率: {self.agent.epsilon:.4f}")
        print(f"   最佳模型: {self.best_model_path}")

        # 生成训练曲线
        self._plot_training_curves()

        # 保存训练摘要
        summary = {
            'total_episodes': len(self.training_history['episode_rewards']),
            'training_time_seconds': training_time,
            'final_avg_reward': float(np.mean(final_rewards)),
            'final_avg_loss': float(np.mean(final_losses)),
            'best_reward': float(self.best_reward),
            'final_epsilon': float(self.agent.epsilon),
            'best_model_path': self.best_model_path,
            'config': self.config
        }

        with open('results/training/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    def _plot_training_curves(self):
        """绘制训练曲线"""
        episodes = range(len(self.training_history['episode_rewards']))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 奖励曲线
        ax1.plot(episodes, self.training_history['episode_rewards'], alpha=0.6)
        # 移动平均
        window = min(50, len(episodes) // 10)
        if window > 0:
            moving_avg = pd.Series(self.training_history['episode_rewards']).rolling(window).mean()
            ax1.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'{window}回合移动平均')
        ax1.set_title('回合奖励')
        ax1.set_xlabel('回合')
        ax1.set_ylabel('奖励')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 损失曲线
        ax2.plot(episodes, self.training_history['episode_losses'], alpha=0.6, color='orange')
        if window > 0:
            moving_avg_loss = pd.Series(self.training_history['episode_losses']).rolling(window).mean()
            ax2.plot(episodes, moving_avg_loss, 'r-', linewidth=2, label=f'{window}回合移动平均')
        ax2.set_title('训练损失')
        ax2.set_xlabel('回合')
        ax2.set_ylabel('损失')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 完成时间曲线
        ax3.plot(episodes, self.training_history['episode_makespans'], alpha=0.6, color='green')
        if window > 0:
            moving_avg_makespan = pd.Series(self.training_history['episode_makespans']).rolling(window).mean()
            ax3.plot(episodes, moving_avg_makespan, 'r-', linewidth=2, label=f'{window}回合移动平均')
        ax3.set_title('完成时间')
        ax3.set_xlabel('回合')
        ax3.set_ylabel('完成时间 (ms)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 探索率曲线
        ax4.plot(episodes, self.training_history['epsilon_history'], color='purple')
        ax4.set_title('探索率衰减')
        ax4.set_xlabel('回合')
        ax4.set_ylabel('Epsilon')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/training/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ 训练曲线已保存至 results/training/training_curves.png")

    def load_model(self, model_path: str):
        """加载预训练模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 恢复训练状态
            self.agent.step_count = checkpoint['agent_config'].get('step_count', 0)
            self.agent.episode_count = checkpoint['agent_config'].get('episode_count', 0)
            self.agent.epsilon = checkpoint['agent_config'].get('epsilon', 1.0)

            print(f"✅ 模型已加载: {model_path}")
            print(f"   训练步数: {self.agent.step_count}, 回合数: {self.agent.episode_count}")

        except FileNotFoundError:
            print(f"⚠️ 模型文件不存在: {model_path}")
        except Exception as e:
            print(f"❌ 加载模型时出错: {e}")


def main():
    """主函数：运行MODRL训练"""
    # 加载配置
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 创建训练器
    trainer = MODRLTrainer(config)

    # 可选：加载预训练模型继续训练
    # trainer.load_model('checkpoints/best_model.pth')

    # 开始训练
    results = trainer.train(
        num_episodes=1000,
        save_interval=100,
        eval_interval=50
    )

    print("\n🎉 MODRL训练完成！")


if __name__ == "__main__":
    main()