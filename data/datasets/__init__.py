"""
数据集模块
包含各种DAG生成器
"""

from .embedded_dag_generator import EmbeddedDAGGenerator
from .random_dag_generator import RandomDAGGenerator

__all__ = [
    'EmbeddedDAGGenerator',
    'RandomDAGGenerator'
]
