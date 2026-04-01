"""
tpu_train.py
Reinforcement learning training loop for the TPU‑based Go Transformer.
Loads self‑play data, trains the model, logs metrics to TensorBoard,
and saves checkpoints using Orbax.
"""
import os
import time
import argparse
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import orbax.checkpoint as ocp
from torch.utils.tensorboard import SummaryWriter # only used for logging

from tpu_model import GoTransformerTPU, train_step
from data_utils import TPUSelfPlayDataset, create_dataloader
from config import ModelConfig, BayesianConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_train_state(rng, config: ModelConfig, learning_rate: float):
    """
    Initialize a Flax TrainState with the model and AdamW optimizer.
    The model parameters are randomly initialized.
    """
    model = GoTransformerTPU(config)
    dummy_obs = jnp.zeros((1, 19, 19, 17), dtype=jnp.float32)
    params = model.init(rng, dummy_obs)['params']
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def main():
    parser = argparse.ArgumentParser(description='TPU reinforcement learning training')
    parser.add_argument('--data-dir', type=str, default='./tpu_data', help='Directory containing self‑play .npz files')
    parser.add_argument('--checkpoint-dir', type=str, default='./tpu_checkpoints', help='Where to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='./tpu_logs', help='TensorBoard log directory')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training (should be a multiple of TPU cores)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--disable-bayesian', action='store_true', help='Disable the uncertainty head (use only policy+value)')
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Starting TPU training engine (with TensorBoard)")
    print(f"Available devices: {jax.devices()}")
    print("=" * 60)

    use_bayesian = not args.disable_bayesian
    config = ModelConfig(
        d_model=256, nhead=8, num_layers=8, use_bayesian=use_bayesian
    )
    if use_bayesian:
        print("💡 Bayesian uncertainty head enabled")

    # TensorBoard writer
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"📊 TensorBoard logs will be saved to: {args.log_dir}"))
    print("   (run: tensorboard --logdir=./tpu_logs)")

    # Load dataset and create DataLoader
    dataset = TPUSelfPlayDataset(data_dir=args.data_dir)
    if len(dataset) == 0:
        print("❌ No training data found. Please run tpu_selfplay.py first.")
        return
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize model and training state
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, config, args.lr)
    
    # JIT‑compile the training step (config is static)
    jitted_train_step = jax.jit(train_step, static_argnums=(2,))

    # Orbax checkpoint manager
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
    checkpoint_manager = ocp.CheckpointManager(os.path.abspath(args.checkpoint_dir), options=options)

    # Training loop
    start_time = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0

        for batch_idx, batch_pt in enumerate(dataloader):
            obs_pt, policy_pt, value_pt = batch_pt
            value_int = value_pt.long()    # target bucket indices (0..127)

            batch_jax = (
                jnp.array(obs_pt.numpy()),
                jnp.array(policy_pt.numpy()),
                jnp.array(value_int.numpy(), dtype=jnp.int32)
            )

            state, metrics = jitted_train_step(state, batch_jax, config)

            epoch_loss += metrics['total_loss'].item()
            global_step += 1

            # Log to TensorBoard
            writer.add_scalar('Loss/Total', metrics['total_loss'].item(), global_step)
            writer.add_scalar('Loss/Policy', metrics['policy_loss'].item(), global_step)
            writer.add_scalar('Loss/Value', metrics['value_loss'].item(), global_step)
            if use_bayesian:
                writer.add_scalar('Loss/Uncertainty', metrics['uncertainty_loss'].item(), global_step)

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1} | Step {global_step} | Loss: {metrics['total_loss']:.4f}")

        avg_loss = epoch_loss / (batch_idx + 1)
        print(f"✅ Epoch {epoch+1} finished | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.1f}s")

        checkpoint_manager.save(global_step, args=ocp.args.StandardSave(state))

        checkpoint_manager.wait_until_finished()

        writer.close()
        print("🎉 Training finished. Logs written to TensorBoard.")
        # ============================================================================
        # TODO: Future hyperparameter tuning
        # ============================================================================
        # The current fixed learning rate and MCTS parameters (c_puct, dirichlet noise,
        # temperature schedule) are reasonable starting points, but they are not optimal.
        # To further improve training efficiency and final playing strength, we recommend
        # performing automatic hyperparameter optimization using:
        #
        #   - Bayesian optimization (e.g., with Optuna or GPyTorch) – efficient for
        #     small numbers of expensive evaluations.
        #   - Evolutionary algorithms (e.g., CMA‑ES, population‑based training) –
        #     robust for noisy objectives.
        #
        # Key hyperparameters to tune:
        #   - Learning rate (initial value, schedule, warmup steps)
        #   - MCTS: c_puct, dirichlet_alpha, dirichlet_fraction, temperature threshold
        #   - Bayesian pruning: exploration_weight, uncertainty_threshold, min/max_candidates
        #   - Network architecture: d_model, num_layers, nhead, dim_feedforward
        #
        # The optimization objective could be the win rate against a fixed baseline
        # (e.g., a previous model snapshot) after a fixed number of training steps.
        # ============================================================================
if __name__ == "__main__":
    main()
