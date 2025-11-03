import torch
import numpy as np
from tqdm import tqdm
from utils.logger import TrainingLogger
from agents.d3qn_agent import D3QNAgent
from environments.embedded_scheduling_env import EmbeddedSchedulingEnvironment


class MODRLTrainer:
    def __init__(self, config):
        self.config = config
        self.logger = TrainingLogger()
        self.setup_environment()
        self.setup_agent()

    def setup_environment(self):
        """设置训练环境"""
        from data.datasets.embedded_dag_generator import EmbeddedDAGGenerator

        self.dag_generator = EmbeddedDAGGenerator(self.config)
        self.env = EmbeddedSchedulingEnvironment(self.config)

    def setup_agent(self):
        """设置智能体"""
        from models.networks.embedded_modrl import EmbeddedMODRL

        model_config = self.config['model']
        self.model = EmbeddedMODRL(
            node_feature_dim=model_config['node_feature_dim'],
            hardware_feature_dim=model_config['hardware_feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_hardware=model_config['num_hardware'],
            num_actions=model_config['num_actions']
        )

        self.target_model = EmbeddedMODRL(
            node_feature_dim=model_config['node_feature_dim'],
            hardware_feature_dim=model_config['hardware_feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_hardware=model_config['num_hardware'],
            num_actions=model_config['num_actions']
        )

        training_config = self.config['training'].copy()
        training_config['action_dim'] = model_config['num_actions']

        self.agent = D3QNAgent(self.model, self.target_model, training_config)

    def train(self):
        """训练主循环"""
        training_config = self.config['training']

        for episode in tqdm(range(training_config['num_episodes'])):
            dag = self.dag_generator.generate()
            state = self.env.reset(dag)
            total_reward = 0
            done = False

            while not done:
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                self.agent.replay()

            # 记录训练信息
            self.logger.log_episode(episode, total_reward, info)

            # 更新目标网络
            if episode % training_config['target_update'] == 0:
                self.agent.update_target_network()

            # 保存检查点
            if episode % 100 == 0:
                self.save_checkpoint(episode)

    def save_checkpoint(self, episode):
        """保存训练检查点"""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'epsilon': self.agent.epsilon
        }
        torch.save(checkpoint, f'checkpoints/model_episode_{episode}.pth')