import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import seaborn as sns


def plot_comparison_results(algorithms: Dict[str, Dict[str, float]],
                            dataset_name: str = "Dataset"):
    """
    绘制算法比较结果

    Args:
        algorithms: 算法结果字典
        dataset_name: 数据集名称
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 提取指标
    metrics = list(list(algorithms.values())[0].keys())
    algorithms_names = list(algorithms.keys())

    # 创建子图
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # 绘制每个指标的比较
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [algorithms[algo].get(metric, 0) for algo in algorithms_names]

        bars = ax.bar(algorithms_names, values, alpha=0.7)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{value:.3f}', ha='center', va='bottom')

    # 隐藏多余的子图
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'results/comparison/{dataset_name}_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(training_history: Dict[str, List[float]],
                         title: str = "Training Curves"):
    """
    绘制训练曲线

    Args:
        training_history: 训练历史数据
        title: 图表标题
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # 奖励曲线
    if 'episode_rewards' in training_history:
        rewards = training_history['episode_rewards']
        episodes = range(len(rewards))
        axes[0].plot(episodes, rewards, alpha=0.6)
        # 移动平均
        if len(rewards) > 10:
            window = min(50, len(rewards) // 10)
            moving_avg = pd.Series(rewards).rolling(window).mean()
            axes[0].plot(episodes, moving_avg, 'r-', linewidth=2,
                         label=f'{window}回合移动平均')
        axes[0].set_title('回合奖励')
        axes[0].set_xlabel('回合')
        axes[0].set_ylabel('奖励')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # 损失曲线
    if 'episode_losses' in training_history:
        losses = training_history['episode_losses']
        episodes = range(len(losses))
        axes[1].plot(episodes, losses, alpha=0.6, color='orange')
        if len(losses) > 10:
            window = min(50, len(losses) // 10)
            moving_avg = pd.Series(losses).rolling(window).mean()
            axes[1].plot(episodes, moving_avg, 'r-', linewidth=2,
                         label=f'{window}回合移动平均')
        axes[1].set_title('训练损失')
        axes[1].set_xlabel('回合')
        axes[1].set_ylabel('损失')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # 完成时间曲线
    if 'episode_makespans' in training_history:
        makespans = training_history['episode_makespans']
        episodes = range(len(makespans))
        axes[2].plot(episodes, makespans, alpha=0.6, color='green')
        if len(makespans) > 10:
            window = min(50, len(makespans) // 10)
            moving_avg = pd.Series(makespans).rolling(window).mean()
            axes[2].plot(episodes, moving_avg, 'r-', linewidth=2,
                         label=f'{window}回合移动平均')
        axes[2].set_title('完成时间')
        axes[2].set_xlabel('回合')
        axes[2].set_ylabel('完成时间 (ms)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    # 探索率曲线
    if 'epsilon_history' in training_history:
        epsilons = training_history['epsilon_history']
        episodes = range(len(epsilons))
        axes[3].plot(episodes, epsilons, color='purple')
        axes[3].set_title('探索率衰减')
        axes[3].set_xlabel('回合')
        axes[3].set_ylabel('Epsilon')
        axes[3].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('results/training/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_hardware_utilization(utilization_data: Dict[str, List[float]],
                              title: str = "Hardware Utilization"):
    """
    绘制硬件利用率

    Args:
        utilization_data: 硬件利用率数据 {hardware_type: [utilization_values]}
        title: 图表标题
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 6))

    for hw_type, values in utilization_data.items():
        time_points = range(len(values))
        ax.plot(time_points, values, label=hw_type, marker='o', markersize=3)

    ax.set_title(title)
    ax.set_xlabel('时间点')
    ax.set_ylabel('利用率')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/hardware_utilization.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_task_scheduling_gantt(tasks_data: List[Dict[str, Any]],
                               title: str = "Task Scheduling Gantt Chart"):
    """
    绘制任务调度甘特图

    Args:
        tasks_data: 任务数据列表
        title: 图表标题
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 8))

    # 按硬件分组任务
    hardware_tasks = {}
    for task in tasks_data:
        hw = task['hardware']
        if hw not in hardware_tasks:
            hardware_tasks[hw] = []
        hardware_tasks[hw].append(task)

    # 绘制甘特图
    y_ticks = []
    y_labels = []

    for i, (hw, tasks) in enumerate(hardware_tasks.items()):
        y_pos = len(hardware_tasks) - i - 1
        y_ticks.append(y_pos)
        y_labels.append(hw)

        for task in tasks:
            start = task['start_time']
            duration = task['end_time'] - task['start_time']
            ax.barh(y_pos, duration, left=start, height=0.6,
                    label=task['task_id'] if len(tasks) < 10 else None)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('时间 (ms)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/task_scheduling_gantt.png', dpi=300, bbox_inches='tight')
    plt.close()
