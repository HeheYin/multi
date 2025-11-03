import argparse
import yaml
import torch
from training.trainer import MODRLTrainer
from evaluation.baseline_comparison import BaselineComparator


def main():
    parser = argparse.ArgumentParser(description='嵌入式智能软件并行优化系统')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'deploy'],
                        default='train', help='运行模式')
    parser.add_argument('--model_path', type=str, help='模型路径')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.mode == 'train':
        trainer = MODRLTrainer(config)
        trainer.train()

    elif args.mode == 'eval':
        comparator = BaselineComparator(config)
        comparator.run_comparison()

    elif args.mode == 'deploy':
        from experiments.hardware_deployment import HardwareDeployer
        deployer = HardwareDeployer(config, args.model_path)
        deployer.deploy()


if __name__ == "__main__":
    main()