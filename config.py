"""
config.py
纯净版 TPU/JAX 全局配置中心
"""
from flax import struct

@struct.dataclass
class ModelConfig:
    """TPU Transformer 模型配置"""
    d_model: int = 256          # 测试时用 256，正式上 TRC 建议改为 1024
    nhead: int = 8              # 正式训练建议改为 16
    num_layers: int = 8         # 正式训练建议改为 16
    dim_feedforward: int = 1024 # 正式训练建议改为 4096
    dropout: float = 0.0        # 在海量自我对弈中，通常设为 0
    num_policy_outputs: int = 362
    num_value_buckets: int = 128
    use_bayesian: bool = True   # 你的核心理念开关

@struct.dataclass
class BayesianConfig:
    """贝叶斯不确定性探索配置"""
    max_candidates: int = 20
    min_candidates: int = 5
    uncertainty_threshold: float = 0.2
    exploration_weight: float = 0.3
    q_value_weight: float = 0.2