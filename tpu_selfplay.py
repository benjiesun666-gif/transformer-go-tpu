"""
tpu_selfplay.py
Batched self-play data generator for Go, using JAX, Pgx, and mctx.
Loads the latest model checkpoint if available, runs multiple games in parallel,
and saves trajectories as .npz files for training.
"""
import os
import time
import functools
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
BATCH_SIZE = 256        # Number of concurrent games. Increase to utilize TPU memory
NUM_SIMULATIONS = 400    # MCTS simulations per move. Higher -> stronger but slower.
MAX_MOVES = 400         # Maximum moves per game (for 19x19, ~300 is typical).

# Bayesian pruning switch – enable only after model is strong enough.
USE_BAYESIAN_IN_SELFPLAY = False


def load_latest_params(init_params, config: ModelConfig):
    """
    Loads the latest weights for self-play from the correct directory.
    """
    path = config.checkpoint_dir

    if not os.path.exists(path):
        print(f"⚠️ Path {path} not found. Starting with initial params.")
        return init_params

    options = ocp.CheckpointManagerOptions(step_prefix='', cleanup_tmp_directories=True)
    mngr = ocp.CheckpointManager(
        os.path.abspath(path),
        ocp.StandardCheckpointer(),
        options=options
    )

    latest_step = mngr.latest_step()
    if latest_step is None:
        return init_params

    print(f"📦 Loading {config.model_type} weights for self-play from {path}...")
    restored = mngr.restore(latest_step)

    if isinstance(restored, dict) and 'params' in restored:
        return restored['params']
    return restored

def run_selfplay():
    print(f"🚀 Initializing TPU self-play engine (Bayesian: {USE_BAYESIAN_IN_SELFPLAY})")
    rng = jax.random.PRNGKey(int(time.time()))
    # 1. Initialize the selected model architecture and parameters
    config = ModelConfig()

    # Conditional instantiation based on the global config type
    if config.model_type == "cnn":
        from tpu_model import GoCNNTPU
        # Use the centralized cnn_blocks property to ensure consistency
        model = GoCNNTPU(config, num_blocks=config.cnn_blocks)
    else:
        from tpu_model import GoTransformerTPU
        model = GoTransformerTPU(config)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, 19, 19, 17), dtype=jnp.float32)

    
    init_params = model.init(init_rng, dummy_obs)['params']
    
    params = load_latest_params(init_params, config)

    # 2. Set up environment and MCTS
    env = pgx.make("go_19x19")
    mcts = PgxMctxMCTS(model.apply, num_simulations=NUM_SIMULATIONS, use_bayesian=USE_BAYESIAN_IN_SELFPLAY)

    @functools.partial(jax.pmap, in_axes=(None, 0, 0))
    def play_step(params, rng_key, state):
        """Maps the computation across available devices.
        Parameters are shared; random keys and environment states are sharded along the batch dimension."""
        rng_mcts, rng_action = jax.random.split(rng_key)
        action_weights, _ = mcts.search_batch(params, rng_mcts, state)
        action = jax.random.categorical(rng_action, jnp.log(action_weights + 1e-8))
        next_state = jax.vmap(env.step)(state, action)
        return next_state, action_weights, action

    # 3. Initialize batch of games
    rng, env_rng = jax.random.split(rng)
    state = jax.vmap(env.init)(jax.random.split(env_rng, BATCH_SIZE))
    num_devices = jax.device_count()
    state = jax.tree.map(
        lambda x: x.reshape((num_devices, BATCH_SIZE // num_devices) + x.shape[1:]),
        state
    )
    # 4. Run the games
    trajectories = {"obs": [], "policy": [], "player": [], "mask": []}

    print(f"🔥 Starting self-play...")
    start_time = time.time()

    for step in range(MAX_MOVES):
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, num_devices)

        active_mask = ~state.terminated
        current_obs = state.observation

        state, action_weights, action = play_step(params, step_rngs, state)

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
    final_rewards = np.array(state.rewards).reshape((BATCH_SIZE, 2))    # shape (BATCH_SIZE,)
    all_obs = np.array(jnp.stack(trajectories["obs"])).reshape((-1, BATCH_SIZE, 19, 19, 17))
    all_policy = np.array(jnp.stack(trajectories["policy"])).reshape((-1, BATCH_SIZE, 362))
    all_players = np.array(jnp.stack(trajectories["player"])).reshape((-1, BATCH_SIZE))
    all_masks = np.array(jnp.stack(trajectories["mask"])).reshape((-1, BATCH_SIZE))

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
