"""
tpu_model.py
纯净版 TPU/JAX 围棋模型
统一从 config.py 读取配置，消除重复定义风险
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple

# ✅ 修复：统一使用全局配置
from config import ModelConfig

BOARD_SIZE = 19
NUM_CELLS = 361

class PositionalEncoding2D(nn.Module):
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
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        norm1 = nn.LayerNorm()(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.nhead,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic
        )(norm1, norm1)
        x = x + attn_out

        norm2 = nn.LayerNorm()(x)
        ffn_out = nn.Sequential([
            nn.Dense(self.dim_feedforward),
            nn.gelu,
            nn.Dense(self.d_model),
        ])(norm2)
        x = x + ffn_out
        return x

class GoTransformerTPU(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, board: jnp.ndarray, deterministic: bool = True):
        batch_size = board.shape[0]
        x = board.astype(jnp.float32)

        x = nn.Conv(features=self.config.d_model, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)

        x = x.reshape((batch_size, NUM_CELLS, self.config.d_model))
        x = PositionalEncoding2D(d_model=self.config.d_model)(x)

        for _ in range(self.config.num_layers):
            x = TransformerBlock(
                d_model=self.config.d_model,
                nhead=self.config.nhead,
                dim_feedforward=self.config.dim_feedforward,
                dropout_rate=self.config.dropout
            )(x, deterministic=deterministic)

        x_pooled = jnp.mean(x, axis=1)

        policy_logits = nn.Sequential([
            nn.Dense(self.config.d_model),
            nn.gelu,
            nn.Dense(self.config.num_policy_outputs),
        ])(x_pooled)

        value_logits = nn.Sequential([
            nn.Dense(self.config.d_model),
            nn.gelu,
            nn.Dense(self.config.num_value_buckets),
        ])(x_pooled)

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
    obs, policy_target, value_target = batch

    def loss_fn(params):
        # 已经修复了 {'params': params} 的字典传参
        if config.use_bayesian:
            policy_logits, value_logits, uncertainty = state.apply_fn({'params': params}, obs, deterministic=False)
        else:
            policy_logits, value_logits = state.apply_fn({'params': params}, obs, deterministic=False)
            uncertainty = None

        policy_log_probs = jax.nn.log_softmax(policy_logits)
        policy_loss = -jnp.sum(policy_target * policy_log_probs, axis=-1).mean()

        # 此时传进来的 value_target 已经是 int32 类型了
        value_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=value_logits, labels=value_target
        ).mean()

        total_loss = policy_loss + value_loss
        metrics = {'policy_loss': policy_loss, 'value_loss': value_loss}

        if config.use_bayesian:
            # 还原为 float 计算 MSE
            value_scalar_target = value_target.astype(jnp.float32) / (config.num_value_buckets - 1)
            uncertainty_target = jnp.abs(value_scalar_target - 0.5) * 2.0
            uncertainty_target = jnp.expand_dims(uncertainty_target, -1)
            uncertainty_loss = jnp.mean(jnp.square(uncertainty - uncertainty_target))

            total_loss += 0.01 * uncertainty_loss
            metrics['uncertainty_loss'] = uncertainty_loss

        metrics['total_loss'] = total_loss
        return total_loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, metrics), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics