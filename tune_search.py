import os
import time
import json
import jax
import jax.numpy as jnp
import numpy as np
import pgx
import optuna
import orbax.checkpoint as ocp

from config import ModelConfig, BayesianConfig, load_bayesian_config
from tpu_model import GoTransformerTPU
from pgx_mctx_bridge import PgxMctxMCTS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BATCH_SIZE = 128
NUM_SIMULATIONS = 200
MAX_MOVES = 300
CHECKPOINT_DIR = "./tpu_checkpoints"
PARAMS_FILE = "best_mcts_params.json"


def load_latest_params():
    # Initialize model with random parameters.
    config = ModelConfig()
    model = GoTransformerTPU(config)
    dummy_obs = jnp.zeros((1, 19, 19, 17), dtype=jnp.float32)
    init_params = model.init(jax.random.PRNGKey(0), dummy_obs)['params']

    if not os.path.exists(CHECKPOINT_DIR):
        return init_params, model

    # Setup Checkpoint manager.
    options = ocp.CheckpointManagerOptions(step_prefix='', cleanup_tmp_directories=True)
    mngr = ocp.CheckpointManager(
        os.path.abspath(CHECKPOINT_DIR),
        ocp.StandardCheckpointer(),
        options=options
    )

    latest_step = mngr.latest_step()
    if latest_step is None:
        return init_params, model

    restored = mngr.restore(latest_step)

    # Extract weights depending on optimizer state structure.
    if 'params' in restored:
        if hasattr(restored['params'], 'fast'):
            return restored['params'].fast, model
        return restored['params'], model
    return restored, model


def play_match(params, model, baseline_config, challenger_config, challenger_is_black):
    # Instantiate environment and two independent MCTS trees.
    env = pgx.make("go_19x19")
    baseline_mcts = PgxMctxMCTS(model.apply, num_simulations=NUM_SIMULATIONS, use_bayesian=True,
                                bayesian_config=baseline_config)
    challenger_mcts = PgxMctxMCTS(model.apply, num_simulations=NUM_SIMULATIONS, use_bayesian=True,
                                  bayesian_config=challenger_config)

    @jax.jit
    def play_step(rng_key, state):
        rng_mcts, rng_action = jax.random.split(rng_key)

        # Determine the active search tree based on player assignment.
        is_challenger_turn = (state.current_player[0] == 0) == challenger_is_black

        def do_challenger(_):
            return challenger_mcts.search_batch(params, rng_mcts, state)[0]

        def do_baseline(_):
            return baseline_mcts.search_batch(params, rng_mcts, state)[0]

        # Dispatch MCTS tree evaluation using JAX conditional logic.
        action_weights = jax.lax.cond(is_challenger_turn, do_challenger, do_baseline, operand=None)

        action = jax.random.categorical(rng_action, jnp.log(action_weights + 1e-8))
        next_state = jax.vmap(env.step)(state, action)
        return next_state

    # Initialize parallel environments.
    rng = jax.random.PRNGKey(int(time.time()) + int(challenger_is_black))
    half_batch = BATCH_SIZE // 2
    state = jax.vmap(env.init)(jax.random.split(rng, half_batch))

    # Execute game loop.
    for _ in range(MAX_MOVES):
        rng, step_rng = jax.random.split(rng)
        state = play_step(step_rng, state)
        if not jnp.any(~state.terminated):
            break

    # Calculate match outcomes. Pgx rewards are structured as (batch, 2).
    rewards = np.array(state.rewards)
    if challenger_is_black:
        challenger_scores = rewards[:, 0]
    else:
        challenger_scores = rewards[:, 1]

    wins = np.sum(challenger_scores == 1.0)
    draws = np.sum(challenger_scores == 0.0)
    return int(wins), int(draws)


def objective(trial, params, model):
    # Load current optimal parameters as the baseline.
    baseline_config = load_bayesian_config(PARAMS_FILE)

    # Propose new hyperparameter values.
    exploration_weight = trial.suggest_float("exploration_weight", 0.05, 1.0)
    uncertainty_threshold = trial.suggest_float("uncertainty_threshold", 0.05, 0.5)
    max_candidates = trial.suggest_int("max_candidates", 10, 30)

    challenger_config = BayesianConfig(
        exploration_weight=exploration_weight,
        uncertainty_threshold=uncertainty_threshold,
        max_candidates=max_candidates
    )

    print(f"Trial {trial.number} started.")
    print(
        f"Params: exp_weight={exploration_weight:.3f}, unc_thresh={uncertainty_threshold:.3f}, max_cand={max_candidates}")

    start_time = time.time()

    # Evaluate performance across both colors.
    wins_b, draws_b = play_match(params, model, baseline_config, challenger_config, challenger_is_black=True)
    wins_w, draws_w = play_match(params, model, baseline_config, challenger_config, challenger_is_black=False)

    total_wins = wins_b + wins_w
    total_draws = draws_b + draws_w
    total_games = BATCH_SIZE

    win_rate = (total_wins + 0.5 * total_draws) / total_games

    print(f"Trial {trial.number} finished.")
    print(f"Result: {total_wins} W, {total_draws} D, {total_games - total_wins - total_draws} L.")
    print(f"Win Rate: {win_rate:.4f} | Time: {time.time() - start_time:.1f}s\n")

    return win_rate


def main():
    print("Starting MCTS parameter tuning...")
    params, model = load_latest_params()

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, params, model), n_trials=20)

    best_params = study.best_trial.params
    print("Tuning completed. Optimal parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Persist best parameters to JSON for automatic loading in subsequent steps.
    with open(PARAMS_FILE, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Parameters saved to {PARAMS_FILE}.")


if __name__ == "__main__":
    main()