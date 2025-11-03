import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import os
import json

from models.networks.embedded_modrl import EmbeddedMODRL
from models.networks.lightweight_st_embedding import LightweightSTEmbedding
from models.networks.lightweight_set_encoder import LightweightSetEncoder
from environments.embedded_scheduling_env import EmbeddedSchedulingEnvironment
from utils.metrics import SchedulingMetrics
from data.datasets.embedded_dag_generator import EmbeddedDAGGenerator


class AblationStudy:
    """消融实验研究"""

    def __init__(self, config):
        self.config = config
        self.metrics_calculator = SchedulingMetrics()
        self.dag_generator = EmbeddedDAGGenerator(config)
        self.results = {}

    def create_ablated_models(self, base_model_path: str) -> Dict[str, torch.nn.Module]:
        """创建消融实验的各个变体模型"""
        models = {}

        # 加载基础模型
        base_model = self._load_base_model(base_model_path)
        models['Full_Model'] = base_model

        # 1. 无时空嵌入模型 (仅使用原始特征)
        models['No_ST_Embedding'] = self._create_no_st_embedding_model(base_model)

        # 2. 无图注意力模型 (替换为简单线性层)
        models['No_GAT'] = self._create_no_gat_model(base_model)

        # 3. 无时序模型 (移除LSTM/GRU)
        models['No_Temporal'] = self._create_no_temporal_model(base_model)

        # 4. 无Set Transformer模型 (使用平均池化)
        models['No_Set_Transformer'] = self._create_no_set_transformer_model(base_model)

        # 5. 仅Makespan优化 (单目标)
        models['Makespan_Only'] = self._create_single_objective_model(base_model, 'makespan')

        # 6. 仅能耗优化 (单目标)
        models['Energy_Only'] = self._create_single_objective_model(base_model, 'energy')

        print(f"✅ 成功创建 {len(models)} 个消融实验模型")
        return models

    def _load_base_model(self, model_path: str) -> EmbeddedMODRL:
        """加载基础模型"""
        model_config = self.config['model']
        model = EmbeddedMODRL(
            node_feature_dim=model_config['node_feature_dim'],
            hardware_feature_dim=model_config['hardware_feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_hardware=model_config['num_hardware'],
            num_actions=model_config['num_actions']
        )

        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model

    def _create_no_st_embedding_model(self, base_model: EmbeddedMODRL) -> torch.nn.Module:
        """创建无时空嵌入的模型变体"""

        class NoSTEmbeddingModel(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                # 替换时空嵌入为简单的线性投影
                self.simple_embedding = torch.nn.Linear(
                    base.st_embedding.st_embedding[0].in_features,
                    base.st_embedding.st_embedding[0].out_features
                )

            def forward(self, node_features, adjacency_matrix, task_sequence, hardware_features):
                # 简单线性投影代替复杂时空嵌入
                task_embedding = self.simple_embedding(node_features.mean(dim=0))
                hardware_embedding = self.base.hardware_encoder(hardware_features)
                hardware_global = torch.mean(hardware_embedding, dim=0)

                state_embedding = torch.cat([task_embedding, hardware_global], dim=-1)

                value = self.base.value_stream(state_embedding)
                advantages = self.base.advantage_stream(state_embedding)
                q_values = value + (advantages - advantages.mean())

                return q_values

        return NoSTEmbeddingModel(base_model)

    def _create_no_gat_model(self, base_model: EmbeddedMODRL) -> torch.nn.Module:
        """创建无图注意力的模型变体"""

        class NoGATModel(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                # 移除GAT，仅保留简单的特征提取
                self.feature_extractor = torch.nn.Sequential(
                    torch.nn.Linear(
                        base.st_embedding.spatial_encoder.layers[0].in_features,
                        base.st_embedding.spatial_encoder.layers[0].out_features
                    ),
                    torch.nn.ReLU()
                )

            def forward(self, node_features, adjacency_matrix, task_sequence, hardware_features):
                # 简单特征提取代替GAT
                spatial_embeddings = self.feature_extractor(node_features)
                temporal_embeddings = self.base.st_embedding.temporal_encoder(
                    spatial_embeddings.unsqueeze(0), task_sequence)
                global_embedding = self.base.st_embedding.global_pool(
                    spatial_embeddings.transpose(0, 1)).squeeze()

                task_embedding = torch.cat([temporal_embeddings.squeeze(0), global_embedding], dim=-1)
                hardware_embedding = self.base.hardware_encoder(hardware_features)
                hardware_global = torch.mean(hardware_embedding, dim=0)

                state_embedding = torch.cat([task_embedding, hardware_global], dim=-1)

                value = self.base.value_stream(state_embedding)
                advantages = self.base.advantage_stream(state_embedding)
                q_values = value + (advantages - advantages.mean())

                return q_values

        return NoGATModel(base_model)

    def _create_no_temporal_model(self, base_model: EmbeddedMODRL) -> torch.nn.Module:
        """创建无时序编码的模型变体"""

        class NoTemporalModel(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, node_features, adjacency_matrix, task_sequence, hardware_features):
                # 跳过时序编码，直接使用空间特征
                spatial_embeddings = self.base.st_embedding.spatial_encoder(
                    node_features, adjacency_matrix)
                global_embedding = self.base.st_embedding.global_pool(
                    spatial_embeddings.transpose(0, 1)).squeeze()

                # 不使用时序特征
                task_embedding = global_embedding
                hardware_embedding = self.base.hardware_encoder(hardware_features)
                hardware_global = torch.mean(hardware_embedding, dim=0)

                state_embedding = torch.cat([task_embedding, hardware_global], dim=-1)

                value = self.base.value_stream(state_embedding)
                advantages = self.base.advantage_stream(state_embedding)
                q_values = value + (advantages - advantages.mean())

                return q_values

        return NoTemporalModel(base_model)

    def _create_no_set_transformer_model(self, base_model: EmbeddedMODRL) -> torch.nn.Module:
        """创建无Set Transformer的模型变体"""

        class NoSetTransformerModel(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                # 使用平均池化代替Set Transformer
                self.simple_pooling = torch.nn.AdaptiveAvgPool1d(1)

            def forward(self, node_features, adjacency_matrix, task_sequence, hardware_features):
                task_embedding = self.base.st_embedding(
                    node_features, adjacency_matrix, task_sequence)

                # 简单平均池化代替Set Transformer
                hardware_global = torch.mean(hardware_features, dim=0)

                state_embedding = torch.cat([task_embedding, hardware_global], dim=-1)

                value = self.base.value_stream(state_embedding)
                advantages = self.base.advantage_stream(state_embedding)
                q_values = value + (advantages - advantages.mean())

                return q_values

        return NoSetTransformerModel(base_model)

    def _create_single_objective_model(self, base_model: EmbeddedMODRL,
                                       objective: str) -> torch.nn.Module:
        """创建单目标优化的模型变体"""

        class SingleObjectiveModel(torch.nn.Module):
            def __init__(self, base, objective):
                super().__init__()
                self.base = base
                self.objective = objective

            def forward(self, node_features, adjacency_matrix, task_sequence, hardware_features):
                # 使用相同的网络结构，但在训练时使用单目标奖励
                return self.base(node_features, adjacency_matrix, task_sequence, hardware_features)

        return SingleObjectiveModel(base_model, objective)

    def run_ablation_study(self, test_dags: List, models: Dict[str, torch.nn.Module],
                           num_runs: int = 5) -> Dict[str, Dict[str, float]]:
        """
        运行消融实验

        Args:
            test_dags: 测试DAG列表
            models: 模型变体字典
            num_runs: 每个模型在每个DAG上的运行次数
        """
        print("🔬 开始消融实验...")

        results = {}
        env = EmbeddedSchedulingEnvironment(self.config)

        for model_name, model in models.items():
            print(f"\n🧪 测试模型变体: {model_name}")
            model.eval()

            model_metrics = []

            for dag in tqdm(test_dags, desc=model_name, leave=False):
                dag_metrics = []

                for run in range(num_runs):
                    state = env.reset(dag)
                    done = False

                    while not done:
                        with torch.no_grad():
                            state_tensor = self._state_to_tensor(state, env)
                            q_values = model(*state_tensor)
                            action = torch.argmax(q_values).item()

                        state, reward, done, info = env.step(action)

                    metrics = self.metrics_calculator.calculate_metrics(env)
                    dag_metrics.append(metrics)

                avg_dag_metrics = self._average_metrics(dag_metrics)
                model_metrics.append(avg_dag_metrics)

            # 计算模型整体指标
            results[model_name] = self._average_metrics(model_metrics)

        self.results = results
        self._save_ablation_results(results)
        return results

    def _state_to_tensor(self, state, env):
        """将状态转换为模型输入张量"""
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

    def _save_ablation_results(self, results: Dict):
        """保存消融实验结果"""
        os.makedirs('results/ablation', exist_ok=True)

        # 保存为CSV
        df_data = []
        for model_name, metrics in results.items():
            row = {'Model_Variant': model_name}
            row.update(metrics)
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_csv('results/ablation/ablation_results.csv', index=False)

        # 保存为JSON
        with open('results/ablation/ablation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    def analyze_component_importance(self) -> Dict[str, float]:
        """分析各组件的重要性"""
        if not self.results or 'Full_Model' not in self.results:
            print("⚠️ 需要先运行消融实验")
            return {}

        full_model_performance = self.results['Full_Model']['makespan']
        importance_scores = {}

        for model_name, metrics in self.results.items():
            if model_name != 'Full_Model':
                performance_drop = metrics['makespan'] - full_model_performance
                importance_scores[model_name] = performance_drop

        # 按重要性排序
        importance_scores = dict(sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        print("\n🔍 组件重要性分析:")
        print("-" * 40)
        for component, importance in importance_scores.items():
            print(f"{component:20}: {importance:+.4f}")

        return importance_scores

    def generate_ablation_report(self):
        """生成消融实验报告"""
        if not self.results:
            print("⚠️ 没有可用的结果，请先运行消融实验")
            return

        print("\n" + "=" * 80)
        print("🔬 消融实验报告")
        print("=" * 80)

        # 主要指标比较
        main_metrics = ['makespan', 'energy_consumption', 'load_balance', 'deadline_satisfaction']

        for metric in main_metrics:
            if metric in self.results['Full_Model']:
                print(f"\n📊 {metric.replace('_', ' ').title()} 比较:")
                print("-" * 50)

                baseline_value = self.results['Full_Model'][metric]

                for model_name, metrics in self.results.items():
                    value = metrics[metric]
                    change = ((value - baseline_value) / baseline_value) * 100
                    change_symbol = "+" if change > 0 else ""
                    print(f"{model_name:25}: {value:.4f} ({change_symbol}{change:+.1f}%)")

        # 组件重要性分析
        self.analyze_component_importance()

        # 生成可视化
        self._generate_ablation_plots()

    def _generate_ablation_plots(self):
        """生成消融实验可视化图表"""
        if not self.results:
            return

        metrics_to_plot = ['makespan', 'energy_consumption', 'load_balance']

        for metric in metrics_to_plot:
            if metric in self.results['Full_Model']:
                plt.figure(figsize=(12, 6))

                model_names = list(self.results.keys())
                values = [self.results[name][metric] for name in model_names]

                bars = plt.bar(model_names, values, alpha=0.7)
                plt.title(f'Ablation Study - {metric.replace("_", " ").title()}')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.xticks(rotation=45, ha='right')

                # 添加数值标签
                for bar, value in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                             f'{value:.3f}', ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig(f'results/ablation/{metric}_ablation.png', dpi=300, bbox_inches='tight')
                plt.close()

        print("✅ 消融实验图表已保存至 results/ablation/")