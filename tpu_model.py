"""
tpu_model.py
Transformer model for Go, implemented in Flax, designed for TPU execution.
Uses a convolutional stem followed by a Transformer encoder, with policy, value,
and optional uncertainty heads.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple

from config import ModelConfig    # hyperparameters

BOARD_SIZE = 19
NUM_CELLS = 361

class PositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding for the flattened board.

    Adds a position-dependent vector to each token, allowing the Transformer
    to distinguish coordinates. The encoding is a learnable parameter of shape
    (height, width, d_model), broadcasted across the batch.
    """
    d_model: int
    height: int = BOARD_SIZE
    width: int = BOARD_SIZE

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(0.02),
            (self.height, self.width, self.d_model)
        )
        pos_embedding_flat = pos_embedding.reshape(-1, self.d_model)
        return x + pos_embedding_flat[jnp.newaxis, :, :]

class TransformerBlock(nn.Module):
    """Pre‑LayerNorm Transformer block with self‑attention and feed‑forward."""
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Self‑attention with pre‑norm
        norm1 = nn.LayerNorm()(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.nhead,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic
        )(norm1, norm1)
        x = x + attn_out

        # Feed‑forward with pre‑norm
        norm2 = nn.LayerNorm()(x)
        ffn_out = nn.Sequential([
            nn.Dense(self.dim_feedforward),
            nn.gelu,
            nn.Dense(self.d_model),
        ])(norm2)
        x = x + ffn_out
        return x

class GoTransformerTPU(nn.Module):
    """Main Transformer model for Go.

    Input: Pgx observation (batch, 19, 19, 17) – 17 feature planes.
    Steps:
      1. Convolutional stem: 3x3 conv, LayerNorm, gelu.
      2. Reshape to (batch, 361, d_model) and add 2D positional encoding.
      3. Stack of TransformerBlocks.
      4. Global average pooling.
      5. Policy head: output (batch, 362) logits (361 board + pass).
      6. Value head: output (batch, 128) logits (discretized value).
      7. Uncertainty head (optional): output (batch, 1) sigmoid.
    """
    config: ModelConfig

    @nn.compact
    def __call__(self, board: jnp.ndarray, deterministic: bool = True):
        batch_size = board.shape[0]
        x = board.astype(jnp.float32)

        # Convolutional stem: map 17 channels to d_model, extract local patterns
        x = nn.Conv(features=self.config.d_model, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)

        # Flatten spatial dimensions to sequence
        x = x.reshape((batch_size, NUM_CELLS, self.config.d_model))
        x = PositionalEncoding2D(d_model=self.config.d_model)(x)

        # Transformer encoder
        for _ in range(self.config.num_layers):
            x = TransformerBlock(
                d_model=self.config.d_model,
                nhead=self.config.nhead,
                dim_feedforward=self.config.dim_feedforward,
                dropout_rate=self.config.dropout
            )(x, deterministic=deterministic)

        # Global average pooling over the board
        x_pooled = jnp.mean(x, axis=1)

        # Policy head: predicts action logits (361 board positions + pass)
        policy_logits = nn.Sequential([
            nn.Dense(self.config.d_model),
            nn.gelu,
            nn.Dense(self.config.num_policy_outputs),
        ])(x_pooled)

        # Value head: predicts distribution over 128 value buckets
        value_logits = nn.Sequential([
            nn.Dense(self.config.d_model),
            nn.gelu,
            nn.Dense(self.config.num_value_buckets),
        ])(x_pooled)

        # Uncertainty head: optional, outputs scalar in [0,1]
        if self.config.use_bayesian:
            uncertainty = nn.Sequential([
                nn.Dense(self.config.d_model // 2),
                nn.gelu,
                nn.Dense(1),
                nn.sigmoid,
            ])(x_pooled)
            return policy_logits, value_logits, uncertainty

        return policy_logits, value_logits

def train_step(state, batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], config: ModelConfig):
    """Single training step, JIT‑compiled.

    Args:
        state: Flax training state (model params, optimizer).
        batch: (obs, policy_target, value_target)
        config: model config (used to determine Bayesian head).

    Returns:
        new_state, metrics (policy_loss, value_loss, uncertainty_loss, total_loss)
    """
    obs, policy_target, value_target = batch

    def loss_fn(params):
        # Forward pass with dropout (deterministic=False)
        if config.use_bayesian:
            policy_logits, value_logits, uncertainty = state.apply_fn({'params': params}, obs, deterministic=False)
        else:
            policy_logits, value_logits = state.apply_fn({'params': params}, obs, deterministic=False)
            uncertainty = None
            
        # Policy cross‑entropy
        policy_log_probs = jax.nn.log_softmax(policy_logits)
        policy_loss = -jnp.sum(policy_target * policy_log_probs, axis=-1).mean()

        # Value cross‑entropy (target is bucket index 0..127)
        value_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=value_logits, labels=value_target
        ).mean()

        total_loss = policy_loss + value_loss
        metrics = {'policy_loss': policy_loss, 'value_loss': value_loss}

        if config.use_bayesian:
            # Convert bucket index back to scalar in [-1,1] for uncertainty loss
            value_scalar_target = value_target.astype(jnp.float32) / (config.num_value_buckets - 1)
            uncertainty_target = jnp.abs(value_scalar_target - 0.5) * 2.0
            uncertainty_target = jnp.expand_dims(uncertainty_target, -1)
            uncertainty_loss = jnp.mean(jnp.square(uncertainty - uncertainty_target))

            total_loss += 0.01 * uncertainty_loss
            metrics['uncertainty_loss'] = uncertainty_loss

        metrics['total_loss'] = total_loss
        return total_loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, metrics), grads = grad_fn(state.params.fast)
    state = state.apply_gradients(grads=grads)
    return state, metrics
