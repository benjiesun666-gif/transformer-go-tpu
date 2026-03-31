import jax
import jax.numpy as jnp
import mctx
import pgx
from typing import Callable, Tuple, Optional

from jax_bayesian import JaxBayesianOptimizer, BayesianConfig
from config import ModelConfig  # 引入全局配置以获取桶数量


class PgxMctxMCTS:
    """完全符合 JAX 纯函数规范的 TPU 树搜索加速器"""

    def __init__(
            self,
            apply_fn: Callable,
            num_simulations: int = 200,
            use_bayesian: bool = True,
            bayesian_config: Optional[BayesianConfig] = None
    ):
        self.apply_fn = apply_fn
        self.num_simulations = num_simulations
        self.use_bayesian = use_bayesian

        # 🚨 获取价值分类桶的数量
        self.num_value_buckets = ModelConfig().num_value_buckets

        if self.use_bayesian:
            config = bayesian_config if bayesian_config else BayesianConfig()
            self.bayesian_opt = JaxBayesianOptimizer(config)

        self.env = pgx.make("go_19x19")
        self.search_batch = jax.jit(self._make_search_fn())

    def _apply_bayesian_mask(self, prior_logits, uncertainty, legal_mask):
        if not self.use_bayesian:
            return prior_logits

        policy_probs = jax.nn.softmax(prior_logits, axis=-1)
        exploration_weight = self.bayesian_opt.config.exploration_weight
        unc_expanded = jnp.broadcast_to(uncertainty, policy_probs.shape)

        bayesian_scores = policy_probs + (unc_expanded * exploration_weight * (1.0 - policy_probs))
        bayesian_scores = jnp.where(legal_mask, bayesian_scores, -100.0)

        threshold = self.bayesian_opt.config.uncertainty_threshold
        max_k = self.bayesian_opt.config.max_candidates
        min_k = self.bayesian_opt.config.min_candidates

        ratio = jnp.clip(uncertainty / threshold, 0.0, 1.0)
        dynamic_k = min_k + (max_k - min_k) * ratio
        dynamic_k = jnp.floor(dynamic_k).astype(jnp.int32)

        top_k_scores, _ = jax.lax.top_k(bayesian_scores, max_k)
        batch_indices = jnp.arange(bayesian_scores.shape[0])
        dynamic_k_idx = jnp.clip(dynamic_k - 1, 0, max_k - 1).flatten()
        score_thresholds = top_k_scores[batch_indices, dynamic_k_idx][:, jnp.newaxis]

        bayesian_mask = bayesian_scores >= score_thresholds
        pruned_logits = jnp.where(bayesian_mask, prior_logits, jnp.finfo(prior_logits.dtype).min)
        return pruned_logits

    def _decode_value_logits(self, value_logits):
        """🚨 核心解码：将 (batch, 128) 的桶分布还原为 (batch,) 的胜率期望值 (-1.0 ~ 1.0)"""
        value_probs = jax.nn.softmax(value_logits, axis=-1)
        value_buckets = jnp.linspace(-1.0, 1.0, self.num_value_buckets)
        return jnp.sum(value_probs * value_buckets, axis=-1)

    def _make_search_fn(self):

        def recurrent_fn(params, rng_key, action, state: pgx.State):
            next_state = jax.vmap(self.env.step)(state, action)
            out = self.apply_fn({'params': params}, next_state.observation)

            if isinstance(out, (list, tuple)) and len(out) == 3:
                prior_logits, value_logits, uncertainty = out
            else:
                prior_logits, value_logits = out
                uncertainty = None

            value_scalar = self._decode_value_logits(value_logits)

            prior_logits = jnp.where(
                next_state.legal_action_mask,
                prior_logits,
                jnp.finfo(prior_logits.dtype).min
            )

            if self.use_bayesian:
                prior_logits = self._apply_bayesian_mask(prior_logits, uncertainty, next_state.legal_action_mask)

            discount = jnp.where(next_state.terminated, 0.0, -1.0)

            # 🚨 核心修复：从 (batch, 2) 中提取出动作执行者（当前玩家）的专属奖励，变为 (batch,)
            batch_size = action.shape[0]
            step_reward = next_state.rewards[jnp.arange(batch_size), state.current_player]

            recurrent_output = mctx.RecurrentFnOutput(
                reward=step_reward,  # <--- 换成提取后的一维数组
                discount=discount,
                prior_logits=prior_logits,
                value=value_scalar
            )
            return recurrent_output, next_state

        def search(params, rng_key, state: pgx.State):
            out = self.apply_fn({'params': params}, state.observation)

            if isinstance(out, (list, tuple)) and len(out) == 3:
                prior_logits, value_logits, uncertainty = out
            else:
                prior_logits, value_logits = out
                uncertainty = None

            # 🚨 根节点同样解码出标量
            value_scalar = self._decode_value_logits(value_logits)

            prior_logits = jnp.where(
                state.legal_action_mask,
                prior_logits,
                jnp.finfo(prior_logits.dtype).min
            )
            if self.use_bayesian:
                prior_logits = self._apply_bayesian_mask(prior_logits, uncertainty, state.legal_action_mask)

            root = mctx.RootFnOutput(
                prior_logits=prior_logits,
                value=value_scalar,  # <--- 使用标量 (batch_size,)
                embedding=state
            )

            policy_output = mctx.muzero_policy(
                params=params,
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=self.num_simulations,
                dirichlet_fraction=0.25,
                dirichlet_alpha=0.3,
            )

            return policy_output.action_weights, value_scalar

        return search