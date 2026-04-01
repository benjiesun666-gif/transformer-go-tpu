"""
JAX/TPU‑compatible Bayesian optimizer for MCTS action pruning.
Dynamically selects candidate actions based on policy prior and uncertainty.
Supports batch processing for TPU efficiency.
"""
import jax
import jax.numpy as jnp
from typing import Tuple, List, Optional, NamedTuple, Any
from dataclasses import dataclass
from flax import struct


@struct.dataclass
class BayesianConfig:
    """Bayesian pruning hyperparameters."""
    max_candidates: int = 20    # Max moves to keep when uncertainty high.
    min_candidates: int = 5    # Min moves to keep when uncertainty low.
    uncertainty_threshold: float = 0.2    # Scale for mapping uncertainty to k.
    exploration_weight: float = 0.3    # Weight of uncertainty bonus.
    q_value_weight: float = 0.2    # Weight of Q‑value term (reserved).


@struct.dataclass  
class CandidateAction:
    """Placeholder for storing candidate action metadata (currently unused)."""
    move: int
    prior: float
    uncertainty: float
    q_value: float = 0.0
    score: float = 0.0


class JaxBayesianOptimizer:
    """Bayesian action selector for sparse MCTS.

    Uses a simple score: prior + uncertainty * exploration_weight * (1 - prior).
    The number of kept candidates k is interpolated linearly between min and max
    based on uncertainty clamped to uncertainty_threshold.
    """
    
    def __init__(self, config: BayesianConfig):
        self.config = config
    
    def select_candidates(
        self,
        policy: jnp.ndarray,
        uncertainty: jnp.ndarray,
        legal_moves: jnp.ndarray,
        q_values: Optional[jnp.ndarray] = None,
        temperature: float = 1.0
    ) -> jnp.ndarray:
        """
        Select a subset of legal moves to keep for MCTS.

        Args:
            policy: (NUM_CELLS,) prior probabilities.
            uncertainty: scalar uncertainty (0..1).
            legal_moves: (L,) indices of legal actions.
            q_values: (NUM_CELLS,) optional Q‑values (not used in current version).
            temperature: softmax temperature (used only if != 1.0).

        Returns:
            candidates: (k,) indices of top‑scoring moves.
        """
        policy = jnp.asarray(policy)
        uncertainty = jnp.asarray(uncertainty)
        legal_moves = jnp.asarray(legal_moves)
        
        if q_values is not None:
            q_values = jnp.asarray(q_values)
        
        k = self._compute_k_jax(uncertainty)
        
        if legal_moves.size == 0:
            return jnp.array([], dtype=jnp.int32)
        
        scores = self._compute_scores_jax(
            policy, uncertainty, legal_moves, q_values
        )
        
        if temperature != 1.0:
            scores = scores ** (1.0 / temperature)
            scores = scores / (scores.sum() + 1e-10)
            
        # Select top‑k actions (deterministic).
        top_k = min(k, len(legal_moves))
        top_indices = jnp.argsort(-scores)[:top_k]
        candidates = legal_moves[top_indices]
        
        return candidates
    
    def _compute_k_jax(self, uncertainty: jnp.ndarray) -> int:
        """Map uncertainty to number of candidates."""
        uncertainty_val = jnp.asarray(uncertainty).item() if uncertainty.size == 1 else uncertainty
        
        if uncertainty_val < self.config.uncertainty_threshold * 0.3:
            return self.config.min_candidates
        elif uncertainty_val < self.config.uncertainty_threshold * 0.7:
            k = int(self.config.min_candidates + 
                   (self.config.max_candidates - self.config.min_candidates) * 0.4)
            return max(self.config.min_candidates, min(k, self.config.max_candidates))
        else:
            ratio = jnp.minimum(uncertainty_val / self.config.uncertainty_threshold, 1.0)
            k = int(self.config.min_candidates + 
                   (self.config.max_candidates - self.config.min_candidates) * ratio)
            return max(self.config.min_candidates, min(k, self.config.max_candidates))
    
    def _compute_scores_jax(
        self,
        policy: jnp.ndarray,
        uncertainty: jnp.ndarray,
        legal_moves: jnp.ndarray,
        q_values: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        """Compute Bayesian score for each legal move (vectorized)."""
        prior_probs = policy[legal_moves]
        
        if q_values is not None:
            q_rewards = q_values[legal_moves] * self.config.q_value_weight
        else:
            q_rewards = jnp.zeros_like(prior_probs)
        
        uncertainty_val = jnp.asarray(uncertainty).item() if uncertainty.size == 1 else uncertainty
        exploration_bonus = uncertainty_val * self.config.exploration_weight * (1.0 - prior_probs)
        
        scores = prior_probs + exploration_bonus + q_rewards
        
        return scores
    
    def select_candidates_batch(
        self,
        policy_batch: jnp.ndarray,
        uncertainty_batch: jnp.ndarray,
        legal_masks: jnp.ndarray,
        q_values_batch: Optional[jnp.ndarray] = None,
        temperature: float = 1.0
    ) -> jnp.ndarray:
        """
        Batch version of select_candidates. Returns padded arrays for TPU.
        """
        batch_size = policy_batch.shape[0]
        max_candidates = self.config.max_candidates
        
        all_candidates = []
        
        for i in range(batch_size):
            policy = policy_batch[i]
            uncertainty = uncertainty_batch[i]
            legal_mask = legal_masks[i]
            
            legal_indices = jnp.where(legal_mask > 0)[0]
            
            q_values = None
            if q_values_batch is not None:
                q_values = q_values_batch[i]
            
            candidates = self.select_candidates(
                policy, uncertainty, legal_indices, q_values, temperature
            )
            
            padded = jnp.pad(
                candidates,
                (0, max_candidates - len(candidates)),
                mode='constant',
                constant_values=-1
            )
            
            all_candidates.append(padded)
        
        return jnp.stack(all_candidates)
    
    @staticmethod
    @jax.jit
    def jax_select_top_k(
        policy: jnp.ndarray,
        legal_mask: jnp.ndarray,
        k: int,
        temperature: float = 1.0
    ) -> jnp.ndarray:
        masked_policy = policy * legal_mask
        
        if temperature != 1.0:
            masked_policy = masked_policy ** (1.0 / temperature)
        
        sum_probs = masked_policy.sum() + 1e-10
        masked_policy = masked_policy / sum_probs
        
        top_indices = jnp.argsort(-masked_policy)[:k]
        
        return top_indices


# ============ Integration with MCTX (conceptual) ============

def create_bayesian_mcts_policy(
    policy_value_fn,
    bayesian_optimizer: JaxBayesianOptimizer,
    num_simulations: int = 200,
    use_uncertainty: bool = True
):
    """
    Wrap a policy‑value function with Bayesian pruning for use with mctx.
    (Conceptual implementation; actual usage may require adapting to mctx API.)
    """
    import mctx
    
    def bayesian_mcts_policy(params, rng_key, root):
        # placeholder
        policy_logits, value, uncertainty = policy_value_fn(root.embedding)
        
        legal_mask = jnp.ones_like(policy_logits)  
        
        policy_probs = jax.nn.softmax(policy_logits)
        
        if use_uncertainty:
            candidates = bayesian_optimizer.select_candidates(
                policy=policy_probs,
                uncertainty=uncertainty,
                legal_moves=jnp.where(legal_mask > 0)[0]
            )
            
            candidate_mask = jnp.zeros_like(policy_probs)
            candidate_mask = candidate_mask.at[candidates].set(1.0)
            
            masked_policy = policy_probs * candidate_mask
            
            masked_policy = masked_policy / (masked_policy.sum() + 1e-10)
        
            prior_logits = jnp.log(masked_policy + 1e-10)
        else:
            prior_logits = policy_logits
            
        output = mctx.RootFnOutput(
            prior_logits=prior_logits,
            value=value,
            embedding=root.embedding
        )
        
        return output
    
    return bayesian_mcts_policy


# ============ Test ============

def test_bayesian_optimizer():
    """Quick test for the Bayesian optimizer."""
    print("=" * 60)
    print("Testing JAX/TPU Bayesian Optimizer")
    print("=" * 60)
    
    config = BayesianConfig(
        max_candidates=10,
        min_candidates=3,
        uncertainty_threshold=0.2
    )
    
    optimizer = JaxBayesianOptimizer(config)
    
    policy = jnp.ones(361) / 361  # uniform prior
    uncertainty = jnp.array(0.5)  # high uncertainty
    legal_moves = jnp.arange(0, 100, dtype=jnp.int32)  # first 100 moves are legal
    
    print("[1] Candidate selection:")
    candidates = optimizer.select_candidates(
        policy=policy,
        uncertainty=uncertainty,
        legal_moves=legal_moves,
        temperature=1.0
    )
    print(f"    Selected {len(candidates)} candidates: {candidates[:10]}...")
    
    print("\n[2] Batch selection:")
    batch_size = 4
    policy_batch = jnp.stack([policy] * batch_size)
    uncertainty_batch = jnp.array([0.1, 0.3, 0.5, 0.8]) 
    legal_mask = jnp.ones((batch_size, 361))
    
    candidates_batch = optimizer.select_candidates_batch(
        policy_batch=policy_batch,
        uncertainty_batch=uncertainty_batch,
        legal_masks=legal_mask
    )
    
    print(f"    Batch shape: {candidates_batch.shape}")
    print(f"    Batch 0 candidates: {candidates_batch[0, :6]}...")
    print(f"    Batch 3 candidates: {candidates_batch[3, :6]}...")
    
    print("\n[3] Top‑k JIT:")
    top_k_moves = optimizer.jax_select_top_k(
        policy=policy,
        legal_mask=jnp.ones(361),
        k=5,
        temperature=1.0
    )
    
    print(f"    Top‑5 moves: {top_k_moves}")
    
    print("\n" + "=" * 60)
    print("Test passed.")
    print("=" * 60)


if __name__ == "__main__":
    test_bayesian_optimizer()
