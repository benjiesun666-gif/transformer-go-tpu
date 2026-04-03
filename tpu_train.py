"""
tpu_train.py
Reinforcement learning training loop for the TPU-based Go Transformer.
Loads self-play data, trains the model, logs metrics to TensorBoard,
saves checkpoints using Orbax, and supports high-performance Optuna hyperparameter tuning.
"""
import os
import time
import argparse
import functools

import jax
import jax.numpy as jnp
import optax
import optuna
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


def create_train_state(rng, config: ModelConfig, learning_rate: float):
    """Initialize a Flax TrainState with the model and SGD+Lookahead optimizer."""
    model = GoTransformerTPU(config)
    dummy_obs = jnp.zeros((1, 19, 19, 17), dtype=jnp.float32)
    params = model.init(rng, dummy_obs)['params']
    base_opt = optax.sgd(learning_rate=learning_rate, momentum=0.9)
    tx = optax.lookahead(base_opt, sync_period=5, slow_step_size=0.5)
    lookahead_params = optax.LookaheadParams.init_synced(params)
    return train_state.TrainState.create(apply_fn=model.apply, params=lookahead_params, tx=tx)


@functools.partial(jax.jit, static_argnums=(2,))
def eval_step(state, batch_jax, config: ModelConfig):
    """
    JIT-compiled evaluation step.
    Calculates validation loss without applying gradients to maximize TPU throughput.
    """
    if config.use_bayesian:
        policy_logits, value_logits, uncertainty = state.apply_fn(
            {'params': state.params.fast}, batch_jax[0], deterministic=True
        )
    else:
        policy_logits, value_logits = state.apply_fn(
            {'params': state.params.fast}, batch_jax[0], deterministic=True
        )
        uncertainty = None

    policy_log_probs = jax.nn.log_softmax(policy_logits)
    policy_loss = -jnp.sum(batch_jax[1] * policy_log_probs, axis=-1).mean()

    value_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=value_logits, labels=batch_jax[2]
    ).mean()

    v_loss = policy_loss + value_loss

    if config.use_bayesian:
        value_scalar_target = batch_jax[2].astype(jnp.float32) / (config.num_value_buckets - 1)
        uncertainty_target = jnp.expand_dims(jnp.abs(value_scalar_target - 0.5) * 2.0, -1)
        v_loss += 0.01 * jnp.mean(jnp.square(uncertainty - uncertainty_target))

    return v_loss


def train_and_evaluate(
    config: ModelConfig,
    lr: float,
    train_loader,
    val_loader,
    epochs: int,
    checkpoint_dir: str = None,
    log_dir: str = None,
    trial: optuna.Trial = None
):
    """
    Core training and evaluation loop.
    Can be run standardly or managed dynamically by an Optuna trial.
    """
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, config, lr)
    jitted_train_step = jax.jit(train_step, static_argnums=(2,))

    writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    checkpoint_manager = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
        checkpoint_manager = ocp.CheckpointManager(os.path.abspath(checkpoint_dir), options=options)

    start_time = time.time()
    global_step = 0
    best_val_loss = float('inf')

    swa_params = jax.tree.map(lambda x: jnp.copy(x), state.params.fast)

    for epoch in range(epochs):
        # --- Training Phase ---
        epoch_loss = 0.0
        for batch_idx, batch_pt in enumerate(train_loader):
            obs_pt, policy_pt, value_pt = batch_pt
            value_int = value_pt.long()

            batch_jax = (
                jnp.array(obs_pt.numpy()),
                jnp.array(policy_pt.numpy()),
                jnp.array(value_int.numpy(), dtype=jnp.int32)
            )

            state, metrics = jitted_train_step(state, batch_jax, config)
            epoch_loss += metrics['total_loss'].item()
            global_step += 1

            if writer:
                writer.add_scalar('Loss/Train_Total', metrics['total_loss'].item(), global_step)
                writer.add_scalar('Loss/Train_Policy', metrics['policy_loss'].item(), global_step)
                writer.add_scalar('Loss/Train_Value', metrics['value_loss'].item(), global_step)
                if config.use_bayesian:
                    writer.add_scalar('Loss/Train_Uncertainty', metrics['uncertainty_loss'].item(), global_step)

        avg_train_loss = epoch_loss / max(1, len(train_loader))

        # --- Validation Phase ---
        val_loss = 0.0
        if val_loader:
            for val_batch_pt in val_loader:
                obs_pt, policy_pt, value_pt = val_batch_pt
                batch_jax = (
                    jnp.array(obs_pt.numpy()),
                    jnp.array(policy_pt.numpy()),
                    jnp.array(value_pt.long().numpy(), dtype=jnp.int32)
                )

                v_loss = eval_step(state, batch_jax, config)
                val_loss += v_loss.item()

            avg_val_loss = val_loss / max(1, len(val_loader))
        else:
            avg_val_loss = avg_train_loss

        if writer:
            writer.add_scalar('Loss/Validation_Total', avg_val_loss, global_step)

        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {time.time()-start_time:.1f}s")

        swa_params = update_swa(swa_params, state.params.fast, epoch)

        # Save checkpoint if it's the best model so far
        if checkpoint_manager and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_manager.save(global_step, args=ocp.args.StandardSave(state))
            checkpoint_manager.wait_until_finished()

        # Optuna Pruning step (Early Stopping for bad trials)
        if trial:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if writer:
        writer.close()

    return best_val_loss if val_loader else avg_train_loss


def objective(trial, train_loader, val_loader, args):
    """Optuna objective function for hyperparameter tuning."""
    # 1. Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    nhead = trial.suggest_categorical('nhead', [4, 8])
    num_layers = trial.suggest_int('num_layers', 4, 12, step=2)

    use_bayesian = not args.disable_bayesian
    config = ModelConfig(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
        use_bayesian=use_bayesian
    )

    # Shorten epochs for tuning to save compute
    tune_epochs = min(args.epochs, 10)

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: lr={lr:.1e}, d_model={d_model}, layers={num_layers}, nhead={nhead}")

    # 2. Run training
    val_loss = train_and_evaluate(
        config=config,
        lr=lr,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=tune_epochs,
        trial=trial
    )

    return val_loss


def main():
    parser = argparse.ArgumentParser(description='TPU reinforcement learning training')
    parser.add_argument('--data-dir', type=str, default='./tpu_data', help='Directory containing self-play .npz files')
    parser.add_argument('--checkpoint-dir', type=str, default='./tpu_checkpoints', help='Where to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='./tpu_logs', help='TensorBoard log directory')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training (multiple of TPU cores)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--disable-bayesian', action='store_true', help='Disable the uncertainty head')

    # Tuning Arguments
    parser.add_argument('--tune', action='store_true', help='Run Optuna hyperparameter tuning instead of standard training')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials to run')
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Starting TPU training engine")
    print(f"Available devices: {jax.devices()}")
    print("=" * 60)

    # Load and split dataset
    full_dataset = TPUSelfPlayDataset(data_dir=args.data_dir)
    if len(full_dataset) == 0:
        print("❌ No training data found. Please run tpu_selfplay.py first.")
        return

    # 90% Train, 10% Validation split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4) if val_size > 0 else None

    if args.tune:
        print("🔍 OPTUNA TUNING MODE ENABLED")
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: objective(trial, train_loader, val_loader, args), n_trials=args.n_trials)

        print("\n🏆 Tuning Complete!")
        print(f"Best Trial: {study.best_trial.number}")
        print("Best Hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("🎯 STANDARD TRAINING MODE")
        use_bayesian = not args.disable_bayesian
        config = ModelConfig(
            d_model=256, nhead=8, num_layers=8, use_bayesian=use_bayesian
        )
        if use_bayesian:
            print("💡 Bayesian uncertainty head enabled")

        print(f"📊 TensorBoard logs will be saved to: {args.log_dir}")
        train_and_evaluate(
            config=config,
            lr=args.lr,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir
        )
        print("🎉 Training finished.")

if __name__ == "__main__":
    main()
