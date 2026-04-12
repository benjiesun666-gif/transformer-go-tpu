"""
tpu_train.py (8-Core Distributed Edition)
Distributed Reinforcement learning training loop for Go.
Maintains SWA, Optuna, TensorBoard, and architecture-specific persistence.
"""
import os
import time
import argparse
import functools
import json

import jax
import jax.numpy as jnp
import optax
import optuna
from flax import jax_utils
from flax.training import train_state
import orbax.checkpoint as ocp

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

from tpu_model import GoTransformerTPU, train_step
from data_utils import TPUSelfPlayDataset, create_dataloader
from config import ModelConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@jax.jit
def update_swa(swa_p, current_p, ep):
    """JIT-compiled SWA weight moving average"""
    return jax.tree.map(lambda s, p: (s * ep + p) / (ep + 1), swa_p, current_p)


def create_train_state(rng, config: ModelConfig, learning_rate: float, model_type: str = "transformer"):
    """Initializes model architecture and Lookahead optimizer."""
    if model_type == "cnn":
        from tpu_model import GoCNNTPU
        model = GoCNNTPU(config, num_blocks=config.cnn_blocks)
    else:
        from tpu_model import GoTransformerTPU
        model = GoTransformerTPU(config)

    dummy_obs = jnp.zeros((1, 19, 19, 17), dtype=jnp.float32)
    params = model.init(rng, dummy_obs)['params']

    base_opt = optax.sgd(learning_rate=learning_rate, momentum=0.9)
    tx = optax.lookahead(base_opt, sync_period=5, slow_step_size=0.5)
    lookahead_params = optax.LookaheadParams.init_synced(params)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=lookahead_params,
        tx=tx
    )


def train_and_evaluate(
        config: ModelConfig,
        lr: float,
        train_loader,
        val_loader,
        epochs: int,
        checkpoint_dir: str = None,
        log_dir: str = None,
        model_type: str = "transformer",
        trial: optuna.Trial = None
):
    num_devices = jax.local_device_count()
    rng = jax.random.PRNGKey(42)

    # 1. Initialize and replicate state for all 8 cores
    state = create_train_state(rng, config, lr, model_type=model_type)

    # Distributed Pmap Training Step
    p_train_step = jax.pmap(functools.partial(train_step, config=config), axis_name="batch")

    # Distributed Pmap Evaluation Step
    @functools.partial(jax.pmap, axis_name="batch", static_argnums=(2,))
    def p_eval_step(state, batch_jax, config: ModelConfig):
        logits_out = state.apply_fn({'params': state.params.fast}, batch_jax[0], deterministic=True)
        policy_logits, value_logits = logits_out[0], logits_out[1]

        policy_loss = -jnp.sum(batch_jax[1] * jax.nn.log_softmax(policy_logits), axis=-1).mean()
        value_loss = optax.softmax_cross_entropy_with_integer_labels(value_logits, batch_jax[2]).mean()

        total_loss = policy_loss + value_loss
        return jax.lax.pmean(total_loss, axis_name="batch")

    writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    checkpoint_manager = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_manager = ocp.CheckpointManager(
            os.path.abspath(checkpoint_dir), ocp.StandardCheckpointer(),
            options=ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
        )

    latest_step = checkpoint_manager.latest_step() if checkpoint_manager else None
    global_step = latest_step if latest_step is not None else 0

    if latest_step is not None:
        state = checkpoint_manager.restore(latest_step, args=ocp.args.StandardRestore(state))
        print(f"✅ Restored weights from step {latest_step} [{model_type.upper()}]")
    else:
        print(f"📈 Starting training from scratch [{model_type.upper()}]")

    # Replicate state across 8 cores
    p_state = jax_utils.replicate(state)
    start_time = time.time()
    best_val_loss = float('inf')

    # SWA parameters must be maintained on host (CPU)
    swa_params = jax.tree.map(lambda x: jnp.copy(x), state.params.fast)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch_pt in enumerate(train_loader):
            obs_pt, policy_pt, value_pt = batch_pt

            # Data Sharding: (Batch, ...) -> (8, Batch/8, ...)
            def shard(x): return jnp.array(x.numpy()).reshape((num_devices, -1) + x.shape[1:])
            batch_jax = (shard(obs_pt), shard(policy_pt), shard(value_pt.long()))

            # Execute Parallel Step
            p_state, p_metrics = p_train_step(p_state, batch_jax)

            # Aggregate metrics for logging
            metrics = jax_utils.unreplicate(p_metrics)
            epoch_loss += metrics['total_loss'].item()
            global_step += 1

            if writer and global_step % 10 == 0:
                for name, val in metrics.items():
                    writer.add_scalar(f'Loss/Train_{name}', val.item(), global_step)

        avg_train_loss = epoch_loss / max(1, len(train_loader))

        # --- Distributed Validation ---
        val_loss = 0.0
        if val_loader:
            for val_batch_pt in val_loader:
                v_obs, v_policy, v_value = val_batch_pt
                v_batch_jax = (shard(v_obs), shard(v_policy), shard(v_value.long()))
                v_loss_sharded = p_eval_step(p_state, v_batch_jax, config)
                val_loss += jax_utils.unreplicate(v_loss_sharded).item()
            avg_val_loss = val_loss / max(1, len(val_loader))
        else:
            avg_val_loss = avg_train_loss

        if writer:
            writer.add_scalar('Loss/Validation_Total', avg_val_loss, global_step)

        print(f"  Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {time.time() - start_time:.1f}s")

        # Update SWA on host
        current_params_host = jax_utils.unreplicate(p_state).params.fast
        swa_params = update_swa(swa_params, current_params_host, epoch)

        # Save Checkpoint
        if checkpoint_manager and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state_to_save = jax_utils.unreplicate(p_state)
            checkpoint_manager.save(global_step, args=ocp.args.StandardSave(state_to_save))

        if trial:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()

    if writer: writer.close()
    return best_val_loss if val_loader else avg_train_loss

def objective(trial, train_loader, val_loader, args):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    config = ModelConfig(use_bayesian=not args.disable_bayesian)
    return train_and_evaluate(config, lr, train_loader, val_loader, min(args.epochs, 10), trial=trial)

def main():
    parser = argparse.ArgumentParser(description='TPU distributed training')
    parser.add_argument('--data-dir', type=str, default='./tpu_data')
    parser.add_argument('--checkpoint-dir', type=str, default='./tpu_checkpoints')
    parser.add_argument('--log-dir', type=str, default='./tpu_logs')
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--disable-bayesian', action='store_true')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--n-trials', type=int, default=20)
    args = parser.parse_args()

    full_dataset = TPUSelfPlayDataset(data_dir=args.data_dir)
    if len(full_dataset) == 0: return

    train_size = int(0.9 * len(full_dataset))
    train_ds, val_ds = random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    train_loader = create_dataloader(train_ds, args.batch_size, True, num_workers=8)
    val_loader = create_dataloader(val_ds, args.batch_size, False, num_workers=8)

    if args.tune:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective(t, train_loader, val_loader, args), n_trials=args.n_trials)
        with open("best_train_params.json", "w") as f: json.dump(study.best_params, f, indent=4)
    else:
        config = ModelConfig(use_bayesian=not args.disable_bayesian)
        train_and_evaluate(config, args.lr, train_loader, val_loader, args.epochs,
                           checkpoint_dir=config.checkpoint_dir, log_dir=f"./tpu_logs/{config.model_type}")

if __name__ == "__main__":
    main()
