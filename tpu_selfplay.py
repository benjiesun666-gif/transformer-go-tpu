"""
tpu_selfplay.py
Batched self-play data generator for Go, using JAX, Pgx, and mctx.
Loads the latest model checkpoint if available, runs multiple games in parallel,
and saves trajectories as .npz files for training.
"""
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import pgx
import orbax.checkpoint as ocp

from config import ModelConfig
from tpu_model import GoTransformerTPU
from pgx_mctx_bridge import PgxMctxMCTS

# ----------------------------------------------------------------------
# Configuration – adjust based on hardware and desired data volume
# ----------------------------------------------------------------------
BATCH_SIZE = 64        # Number of concurrent games. Increase to utilize TPU memory
NUM_SIMULATIONS = 800    # MCTS simulations per move. Higher -> stronger but slower.
MAX_MOVES = 300         # Maximum moves per game (for 19x19, ~300 is typical).
CHECKPOINT_DIR = "./tpu_checkpoints"

# Bayesian pruning switch – enable only after model is strong enough.
USE_BAYESIAN_IN_SELFPLAY = False 


def load_latest_params(init_params):
    """Load the most recent model checkpoint from CHECKPOINT_DIR.
    If no valid checkpoint exists, return the initial random parameters.
    """
    if not os.path.exists(CHECKPOINT_DIR):
        return init_params

    options = ocp.CheckpointManagerOptions(step_prefix='', cleanup_tmp_directories=True)
    mngr = ocp.CheckpointManager(
        os.path.abspath(CHECKPOINT_DIR),
        ocp.StandardCheckpointer(),
        options=options
    )

    latest_step = mngr.latest_step()
    if latest_step is None:
        print("⚠️ No valid checkpoint found. Using random initial parameters.")
        return init_params

    print(f"📦 Found checkpoint at step {latest_step}. Loading...")
    restored = mngr.restore(latest_step)

    # Restored may be a dict with 'params' or the params directly.
    if 'params' in restored:
        return restored['params']
    return restored

def run_selfplay():
    print(f"🚀 Initializing TPU self-play engine (Bayesian: {USE_BAYESIAN_IN_SELFPLAY})")
    rng = jax.random.PRNGKey(int(time.time()))

    # 1. Initialize model and parameters
    config = ModelConfig()
    model = GoTransformerTPU(config)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, 19, 19, 17), dtype=jnp.float32)

    
    init_params = model.init(init_rng, dummy_obs)['params']
    
    params = load_latest_params(init_params)

    # 2. Set up environment and MCTS
    env = pgx.make("go_19x19")
    mcts = PgxMctxMCTS(model.apply, num_simulations=NUM_SIMULATIONS, use_bayesian=USE_BAYESIAN_IN_SELFPLAY)

    @jax.jit
    def play_step(params, rng_key, state):
        """One step of parallel self-play for all active games."""
        rng_mcts, rng_action = jax.random.split(rng_key)
        action_weights, _ = mcts.search_batch(params, rng_mcts, state)
        action = jax.random.categorical(rng_action, jnp.log(action_weights + 1e-8))
        next_state = jax.vmap(env.step)(state, action)
        return next_state, action_weights, action

    # 3. Initialize batch of games
    rng, env_rng = jax.random.split(rng)
    state = jax.vmap(env.init)(jax.random.split(env_rng, BATCH_SIZE))
    # 4. Run the games
    trajectories = {"obs": [], "policy": [], "player": [], "mask": []}

    print(f"🔥 Starting self-play...")
    start_time = time.time()

    for step in range(MAX_MOVES):
        rng, step_rng = jax.random.split(rng)
        active_mask = ~state.terminated
        current_obs = state.observation

        state, action_weights, action = play_step(params, step_rng, state)

        # Store data for this time step (only for debugging; later we compress)
        trajectories["obs"].append(current_obs)
        trajectories["policy"].append(action_weights)
        trajectories["player"].append(state.current_player)
        trajectories["mask"].append(active_mask)

        if (step + 1) % 20 == 0:
            print(f"  > 第 {step+1} 手 | 活跃局数: {np.sum(np.array(active_mask))}")
        if not jnp.any(active_mask): break

    # 5. Back‑propagate true game outcomes to each move (AlphaZero style)
    print(f"📦 Processing data and saving...")
    final_rewards = np.array(state.rewards)    # shape (BATCH_SIZE,)
    all_obs = np.array(jnp.stack(trajectories["obs"]))    # (T, B, 19, 19, 17)
    all_policy = np.array(jnp.stack(trajectories["policy"]))    # (T, B, 362)
    all_players = np.array(jnp.stack(trajectories["player"]))    # (T, B)
    all_masks = np.array(jnp.stack(trajectories["mask"]))    # (T, B)

    seq_len = all_obs.shape[0]
    true_values = np.zeros((seq_len, BATCH_SIZE), dtype=np.float32)
    batch_idx = np.arange(BATCH_SIZE)
    for t in range(seq_len):
        # Value = final reward (from perspective of the player who moved at this step)
        true_values[t] = np.where(all_masks[t], final_rewards[batch_idx, all_players[t]], 0.0)

    # Discretize values into 128 buckets for cross‑entropy loss
    true_values_bucket = ((true_values + 1.0) / 2.0 * 127.0).astype(np.int32)

    # Save to compressed .npz file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("./tpu_data", exist_ok=True)
    save_path = f"./tpu_data/selfplay_{timestamp}.npz"
    np.savez_compressed(save_path, obs=all_obs, policy=all_policy, value=true_values_bucket, mask=all_masks)

    print(f"🎉 Done! Total time: {time.time()-start_time:.2f}s | Saved to {save_path}")

if __name__ == "__main__":
    run_selfplay()
