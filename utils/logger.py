import logging
import os
from typing import Dict, Any
import json
from datetime import datetime


class BaseLogger:
    """基础日志记录器"""

    def __init__(self, name: str, log_dir: str = 'logs'):
        self.name = name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 设置日志格式
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # 文件处理器
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)


class TrainingLogger(BaseLogger):
    """训练日志记录器"""

    def __init__(self, log_dir: str = 'logs/training'):
        super().__init__('training', log_dir)
        self.episode_logs = []

    def log_episode(self, episode: int, reward: float, info: Dict[str, Any]):
        """记录回合信息"""
        log_entry = {
            'episode': episode,
            'reward': reward,
            'timestamp': datetime.now().isoformat(),
            'info': info
        }
        self.episode_logs.append(log_entry)

        self.info(f"Episode {episode}: Reward={reward:.2f}")

    def log_metrics(self, metrics: Dict[str, float]):
        """记录评估指标"""
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.info(f"Metrics: {metrics_str}")

    def save_logs(self, filename: str = None):
        """保存日志到文件"""
        if filename is None:
            filename = f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.episode_logs, f, indent=2)

        self.info(f"Logs saved to {filepath}")


class ExperimentLogger(BaseLogger):
    """实验日志记录器"""

    def __init__(self, experiment_name: str, log_dir: str = 'logs/experiments'):
        super().__init__(experiment_name, log_dir)
        self.experiment_name = experiment_name
        self.experiment_logs = []

    def log_experiment_start(self, config: Dict[str, Any]):
        """记录实验开始"""
        log_entry = {
            'event': 'experiment_start',
            'experiment_name': self.experiment_name,
            'config': config,
            'start_time': datetime.now().isoformat()
        }
        self.experiment_logs.append(log_entry)
        self.info(f"Experiment '{self.experiment_name}' started")

    def log_result(self, result_type: str, result_data: Dict[str, Any]):
        """记录实验结果"""
        log_entry = {
            'event': 'result',
            'type': result_type,
            'data': result_data,
            'timestamp': datetime.now().isoformat()
        }
        self.experiment_logs.append(log_entry)
        self.info(f"Result logged: {result_type}")

    def log_experiment_end(self):
        """记录实验结束"""
        log_entry = {
            'event': 'experiment_end',
            'end_time': datetime.now().isoformat()
        }
        self.experiment_logs.append(log_entry)
        self.info(f"Experiment '{self.experiment_name}' ended")

    def save_experiment_logs(self, filename: str = None):
        """保存实验日志"""
        if filename is None:
            filename = f'{self.experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.experiment_logs, f, indent=2)

        self.info(f"Experiment logs saved to {filepath}")
