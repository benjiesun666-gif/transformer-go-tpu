"""
config.py
Pure JAX/TPU configuration hub.
"""
import os
import json
from flax import struct


@struct.dataclass
class ModelConfig:
    """
    Unified configuration for Go models with synchronized architecture depth.
    """
    model_type: str = "transformer"
    data_dir: str = "./tpu_data"

    @property
    def checkpoint_dir(self) -> str:
        """
        Generates a unique directory path for checkpoints based on the model type.
        """
        return f"./checkpoints_{self.model_type}"

    @property
    def cnn_blocks(self) -> int:
        """
        Calculates the number of residual blocks required to match
        the target 80M parameter capacity.
        """
        return 40

    d_model: int = 768
    nhead: int = 12
    num_layers: int = 12
    dim_feedforward: int = 3072
    dropout: float = 0.0
    num_policy_outputs: int = 362
    num_value_buckets: int = 128
    use_bayesian: bool = True

@struct.dataclass
class BayesianConfig:
    """Bayesian pruning settings."""
    max_candidates: int = 20    # Keep up to this many moves when uncertain.
    min_candidates: int = 5    # Keep at least this many moves when confident.
    uncertainty_threshold: float = 0.2    # Interpolation range for k.
    exploration_weight: float = 0.3    # How much uncertainty encourages exploration.
    q_value_weight: float = 0.2    # Q‑value bonus (reserved).


def load_bayesian_config(filepath: str = "best_mcts_params.json") -> BayesianConfig:
    # Load hyperparameter configuration from a JSON file if it exists.
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                params = json.load(f)
            # Filter loaded dictionary to match dataclass fields.
            valid_keys = BayesianConfig.__dataclass_fields__.keys()
            filtered_params = {k: v for k, v in params.items() if k in valid_keys}
            return BayesianConfig(**filtered_params)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}. Using defaults. ({e})")

    # Return default configuration if file is missing or invalid.
    return BayesianConfig()
