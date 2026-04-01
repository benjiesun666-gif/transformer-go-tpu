"""
config.py
Pure JAX/TPU configuration hub.
"""
from flax import struct

@struct.dataclass
class ModelConfig:
    """TPU Transformer 模型配置"""
    d_model: int = 256          # Hidden size. Increase for stronger models.
    nhead: int = 8              # Attention heads. Should divide d_model.
    num_layers: int = 8         # Number of Transformer layers.
    dim_feedforward: int = 1024 # FFN inner dimension (typically 4*d_model).
    dropout: float = 0.0        # Dropout rate. Usually 0 for self-play.
    num_policy_outputs: int = 362 # 361 board positions + pass.
    num_value_buckets: int = 128 # Discretized value targets.
    use_bayesian: bool = True   # Enable uncertainty head and pruning.

@struct.dataclass
class BayesianConfig:
    """Bayesian pruning settings."""
    max_candidates: int = 20    # Keep up to this many moves when uncertain.
    min_candidates: int = 5    # Keep at least this many moves when confident.
    uncertainty_threshold: float = 0.2    # Interpolation range for k.
    exploration_weight: float = 0.3    # How much uncertainty encourages exploration.
    q_value_weight: float = 0.2    # Q‑value bonus (reserved).
