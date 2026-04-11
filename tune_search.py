import os
import time
import json
import jax
import jax.numpy as jnp
import numpy as np
import pgx
import optuna
import orbax.checkpoint as ocp
import csv

from config import ModelConfig, BayesianConfig, load_bayesian_config
from tpu_model import GoTransformerTPU
from pgx_mctx_bridge import PgxMctxMCTS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BATCH_SIZE = 128
NUM_SIMULATIONS = 400
MAX_MOVES = 400
PARAMS_FILE = "best_mcts_params.json"


def load_latest_params():
    """
    Initializes the model architecture and restores parameters from
    the architecture-specific directory.
    """
    # 1. Initialize configuration and select the correct backbone
    config = ModelConfig()

    if config.model_type == "cnn":
        from tpu_model import GoCNNTPU
        # Align CNN blocks with Transformer depth (12 layers -> 40 blocks)
        model = GoCNNTPU(config, num_blocks=40)
    else:
        from tpu_model import GoTransformerTPU
        model = GoTransformerTPU(config)

    # 2. Generate initial random parameters for initialization
    dummy_obs = jnp.zeros((1, 19, 19, 17), dtype=jnp.float32)
    init_params = model.init(jax.random.PRNGKey(0), dummy_obs)['params']

    # 3. Use the dynamic path from config
    path = config.checkpoint_dir

    # 4. If directory doesn't exist, return initial random weights
    if not os.path.exists(path):
        print(f"⚠️ No checkpoint found at {path}. Using initial weights.")
        return init_params, model

    # 5. Set up Orbax CheckpointManager with standard options
    options = ocp.CheckpointManagerOptions(step_prefix='', cleanup_tmp_directories=True)
    mngr = ocp.CheckpointManager(
        os.path.abspath(path),
        ocp.StandardCheckpointer(),
        options=options
    )

    # 6. Find the latest step and restore
    latest_step = mngr.latest_step()
    if latest_step is None:
        return init_params, model

    print(f"📦 Restoring {config.model_type.upper()} weights from step {latest_step}...")
    restored = mngr.restore(latest_step)

    # 7. Extract parameter dictionary based on training state structure
    if isinstance(restored, dict) and 'params' in restored:
        actual_params = restored['params']
        # Handle Lookahead optimizer state if present
        if hasattr(actual_params, 'fast'):
            return actual_params.fast, model
        return actual_params, model

    return restored, model


def objective(trial, params, model):
    # Load baseline parameters from file.
    baseline_config = load_bayesian_config(PARAMS_FILE)

    # Sample float hyperparameters.
    exploration_weight = trial.suggest_float("exploration_weight", 0.05, 1.0)
    uncertainty_threshold = trial.suggest_float("uncertainty_threshold", 0.05, 0.5)

    # Fix max_candidates to a static integer to prevent XLA recompilation triggered by dynamic shapes.
    max_candidates = 20

    challenger_config = BayesianConfig(
        exploration_weight=exploration_weight,
        uncertainty_threshold=uncertainty_threshold,
        max_candidates=max_candidates
    )

    print(f"Trial {trial.number} started.")
    print(
        f"Params: exp_weight={exploration_weight:.3f}, unc_thresh={uncertainty_threshold:.3f}, max_cand={max_candidates}")

    start_time = time.time()

    # Initialize environments and MCTS instances once per trial.
    env = pgx.make("go_19x19")
    baseline_mcts = PgxMctxMCTS(model.apply, num_simulations=NUM_SIMULATIONS, use_bayesian=True,
                                bayesian_config=baseline_config)
    challenger_mcts = PgxMctxMCTS(model.apply, num_simulations=NUM_SIMULATIONS, use_bayesian=True,
                                  bayesian_config=challenger_config)

    # Compile play_step once. Use a flag to alternate player assignment without triggering recompilation.
    @jax.jit
    def play_step(rng_key, state, challenger_is_black_flag):
        rng_mcts, rng_action = jax.random.split(rng_key)

        # Determine if current turn belongs to the challenger based on the flag.
        is_challenger_turn = (state.current_player[0] == 0) == (challenger_is_black_flag == 1)

        def do_challenger(_):
            return challenger_mcts.search_batch(params, rng_mcts, state)[0]

        def do_baseline(_):
            return baseline_mcts.search_batch(params, rng_mcts, state)[0]

        # Dispatch MCTS evaluation conditionally.
        action_weights = jax.lax.cond(is_challenger_turn, do_challenger, do_baseline, operand=None)
        action = jax.random.categorical(rng_action, jnp.log(action_weights + 1e-8))
        next_state = jax.vmap(env.step)(state, action)

        return next_state

    def run_match(is_black: bool):
        rng = jax.random.PRNGKey(int(time.time()) + int(is_black))
        state = jax.vmap(env.init)(jax.random.split(rng, BATCH_SIZE // 2))

        # Convert boolean to integer flag for JIT compatibility.
        flag = jnp.int32(1) if is_black else jnp.int32(0)

        for _ in range(MAX_MOVES):
            rng, step_rng = jax.random.split(rng)
            state = play_step(step_rng, state, flag)
            if not jnp.any(~state.terminated):
                break

        rewards = np.array(state.rewards)
        scores = rewards[:, 0] if is_black else rewards[:, 1]
        wins = int(np.sum(scores == 1.0))
        draws = int(np.sum(scores == 0.0))
        return wins, draws

    # Execute matches. The second call hits the XLA cache generated by the first call.
    wins_b, draws_b = run_match(is_black=True)
    wins_w, draws_w = run_match(is_black=False)

    total_wins = wins_b + wins_w
    total_draws = draws_b + draws_w
    total_games = BATCH_SIZE

    win_rate = (total_wins + 0.5 * total_draws) / total_games

    print(f"Trial {trial.number} finished.")
    print(f"Result: {total_wins} W, {total_draws} D, {total_games - total_wins - total_draws} L.")
    print(f"Win Rate: {win_rate:.4f} | Time: {time.time() - start_time:.1f}s\n")

    return win_rate


def main():
    """
    Main execution loop for MCTS parameter tuning.
    Handles model initialization, Optuna optimization, and architecture-specific logging.
    """
    print("Starting MCTS parameter tuning...")

    # 1. Load configuration and model weights based on the current architecture type
    params, model = load_latest_params()
    config = ModelConfig()

    # 2. Initialize Optuna study to maximize the win rate against the baseline
    study = optuna.create_study(direction='maximize')

    # 3. Execute optimization trials for exploration_weight and uncertainty_threshold
    study.optimize(lambda trial: objective(trial, params, model), n_trials=20)

    # 4. Extract the best performing hyperparameters from the study
    best_params = study.best_trial.params
    # Ensure the static candidate limit is preserved in the final output
    best_params["max_candidates"] = 20

    # 5. Define a dynamic CSV filename to prevent data contamination between architectures
    history_file = f"win_rate_{config.model_type}_history.csv"
    file_exists = os.path.isfile(history_file)
    best_win_rate = study.best_value

    # 6. Log the tuning results into the architecture-specific CSV file
    with open(history_file, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Write header only if the file is being created for the first time
        if not file_exists:
            writer.writerow([
                "timestamp",
                "architecture",
                "best_win_rate",
                "exploration_weight",
                "uncertainty_threshold"
            ])

        # Append the best result from this iteration of the auto-loop
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            config.model_type,
            f"{best_win_rate:.4f}",
            best_params.get("exploration_weight"),
            best_params.get("uncertainty_threshold")
        ])

    # 7. Print summary of optimal parameters to the console for monitoring
    print(f"\n🏆 Tuning completed for [{config.model_type.upper()}]")
    print("Optimal parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # 8. Update the persistent JSON file used by the self-play engine for the next iteration
    with open(PARAMS_FILE, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"✅ Parameters saved to {PARAMS_FILE}.")
    print(f"📈 Performance history logged to {history_file}")

if __name__ == "__main__":
    main()
