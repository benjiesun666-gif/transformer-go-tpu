# Project Design Document: Transformer-based Go AI (TPU Edition)

## 1. Overview
This project implements a complete reinforcement learning system for Go, built entirely with JAX/Flax. The core algorithm follows the AlphaZero paradigm: self‑play with Monte Carlo Tree Search (MCTS) and introduces a **Bayesian‑guided sparse MCTS** that dynamically prunes low‑potential actions using the network’s uncertainty output. The entire pipeline is optimized for TPU execution, leveraging `jax.jit`, `jax.vmap`, and the `mctx` library to fuse MCTS and neural inference into an efficient XLA computation graph. A single TPU can perform thousands of concurrent self‑play games.

The code is fully open‑source, aiming to explore Transformer architectures for Go and to provide a reproducible, scalable reinforcement learning baseline for the community.

## 2. System Architecture
The training loop alternates between two main phases:
- **Self‑play data generation**: Many games are played in parallel. At each move, MCTS guided by the neural network selects an action. For every state, the MCTS visit distribution (policy) and the final game outcome are recorded.
- **Model training**: The generated data is used to train the Transformer model. The policy loss is the cross‑entropy between the MCTS policy and the network’s policy logits; the value loss is a cross‑entropy over discretized game outcomes (buckets).

These two phases can be interleaved for continuous reinforcement learning.

The main components are:
| Module | File | Description |
|--------|------|-------------|
| Model definition | `tpu_model.py` | `GoTransformerTPU` class: Conv stem, 2D positional encoding, Transformer blocks, policy/value/uncertainty heads. |
| Bayesian optimization | `jax_bayesian.py` | Dynamic action pruning using Gaussian processes (simplified version) for sparse MCTS. |
| MCTS bridge | `pgx_mctx_bridge.py` | Wires the Pgx Go environment with the `mctx` library, providing a pure‑functional MCTS that supports Bayesian pruning and batch inference. |
| Self‑play | `tpu_selfplay.py` | Uses `jax.vmap` to run many games in parallel, calls MCTS, and saves trajectories as `.npz` files. |
| Training loop | `tpu_train.py` | Loads self‑play data, executes training steps, saves checkpoints with Orbax, and logs metrics to TensorBoard. |
| Data loading | `data_utils.py` | PyTorch `DataLoader` with multiprocessing to prefetch data and feed it to TPU efficiently. |
| Configuration | `config.py` | Hyperparameters (model, Bayesian, training) defined as `flax.struct.dataclass`. |

## 3. Detailed Module Description

### 3.1 Model Definition (`tpu_model.py`)
The model `GoTransformerTPU` takes a Pgx observation tensor of shape `(batch, 19, 19, 17)` (17 feature planes: current board, history, legal actions, etc.) and processes it as follows:
1. **Conv stem**: `nn.Conv(features=d_model, kernel_size=(3,3), padding='SAME')` maps the 17 input channels to `d_model` dimensions, extracting local features.
2. **Reshape and positional encoding**: The tensor is reshaped to `(batch, 361, d_model)` and a learnable 2D positional embedding is added.
3. **Transformer encoder**: `num_layers` blocks of `TransformerBlock` (Pre‑LayerNorm, multi‑head self‑attention, feed‑forward network).
4. **Pooling and heads**: Global average pooling over the sequence dimension yields a `(batch, d_model)` vector. Three separate heads produce:
   - Policy logits: `(batch, 362)` – 361 board positions + **1 pass**.
   - Value logits: `(batch, 128)` (for 128 buckets).
   - Uncertainty: `(batch, 1)` squashed by sigmoid to `[0,1]`.

All hyperparameters (`d_model`, `nhead`, `num_layers`, `dim_feedforward`, `use_bayesian`) are defined in `config.py`.

### 3.2 MCTS and Bayesian Pruning (`pgx_mctx_bridge.py` + `jax_bayesian.py`)
#### 3.2.1 `PgxMctxMCTS` Class
This class wraps the MCTS search. Its main method `search_batch` calls `mctx.muzero_policy` and supplies two essential functions:
- `root_fn`: evaluates the root state and returns prior logits, value, and an embedding (the state itself).
- `recurrent_fn`: given a state and an action, applies the environment step, evaluates the new state with the network, and returns reward, discount, prior logits, and value.

Both functions invoke `self.apply_fn({'params': params}, obs)`, i.e., the Flax model.

#### 3.2.2 Bayesian Pruning (`_apply_bayesian_mask`)
When `use_bayesian=True`, the policy logits are dynamically pruned in the root and recurrent nodes:
- Compute action probabilities: `policy_probs = softmax(prior_logits)`.
- For each action, compute a “Bayesian score”: `score = prior + uncertainty * exploration_weight * (1 - prior)` where `exploration_weight` is a hyperparameter and `uncertainty` is the scalar output of the uncertainty head (broadcasted to all actions).
- The number of candidates to keep `k` is determined dynamically based on uncertainty: `k = min_k + (max_k - min_k) * clip(uncertainty / threshold, 0, 1)`.
- Use `jax.lax.top_k` to obtain the top `max_k` scores across legal actions, then pick the score at the `k`-th position as a threshold. Keep actions whose score ≥ threshold.
- Mask out pruned actions by setting their logits to `-inf`, so they are never selected by MCTS.

This yields a **sparse MCTS** that concentrates search on promising actions, especially in high‑uncertainty positions (the model “is unsure” and keeps more candidates).

### 3.3 Self‑play Data Generation (`tpu_selfplay.py`)
The function `run_selfplay()` generates a batch of self‑play games:
1. Initialize model parameters randomly, create a Pgx environment, and instantiate `PgxMctxMCTS`.
2. Use `jax.vmap` to initialize `BATCH_SIZE` games in parallel.
3. Record for every step: observation, action weights, current player, and active mask.
4. After the games finish, back‑propagate the true game outcomes (`state.rewards`) to each move, flipping sign according to the player at that move (so that the value is always from the perspective of the player who made the move).
5. Map the true outcomes from `[-1,0,1]` to bucket indices `0..127`: `value_bucket = ((value + 1) / 2 * 127).astype(np.int32)`.
6. Save the data as a compressed `.npz` file with timestamped filename (containing `obs`, `policy` (shape `(T*B, 362)`), `value`, `mask`).

### 3.4 Training Loop (`tpu_train.py`)
- `TPUSelfPlayDataset` loads `.npz` files into memory (concatenating and filtering by mask). It expects policy arrays of shape `(T*B, 362)`.
- A PyTorch `DataLoader` with multiprocessing (fallback to 0 workers on Windows) feeds batches to the training step.
- The training step `train_step` is JIT‑compiled with `jax.jit` and computes:
  - Policy cross‑entropy loss (using the MCTS policy as the target, dimension `362`).
  - Value cross‑entropy loss (target is the integer bucket index).
  - Optionally, Bayesian uncertainty loss (MSE between `uncertainty` and the normalized prediction error) when `use_bayesian=True`.
- **Advanced Optimizer Stack**: The traditional AdamW optimizer has been replaced with a combination of **SGD + Lookahead** (`optax.lookahead`). This allows the optimizer to explore the loss landscape more aggressively while maintaining stability.
- **Stochastic Weight Averaging (SWA)**: An SWA mechanism is implemented manually via `jax.tree_map` across epochs. It maintains a smoothed running average of the model weights (`swa_params`), which provides significantly better generalization and robust performance in self-play compared to the raw training weights.
- **Automated Hyperparameter Tuning**: The training script integrates **Optuna** for Bayesian hyperparameter search. By passing the `--tune` flag, the system bypasses standard training and uses the Tree-structured Parzen Estimator (TPE) algorithm combined with a `MedianPruner` to dynamically find the optimal learning rate, Transformer depth, and attention heads based on Validation Loss.
- Checkpoints are saved using `orbax.checkpoint`, securely storing model parameters and states.
- Losses are logged to TensorBoard for monitoring.

### 3.5 Data Loading (`data_utils.py`)
- `TPUSelfPlayDataset` scans all `.npz` files, flattens the time and batch dimensions, and filters out invalid moves using the `mask` array. The data is concatenated into large arrays for efficient access. Policy arrays are reshaped to `(total_samples, 362)`.
- `create_dataloader` returns a PyTorch `DataLoader` with configurable number of workers, pin memory, and `drop_last` to ensure consistent batch sizes.

## 4. Key Algorithmic Details
### 4.1 Value Target Discretization
To use cross‑entropy for value prediction, the true game outcome `v ∈ [-1,1]` is discretized into 128 bins:
`bucket = floor((v + 1) / 2 * 127)`
The model outputs 128 logits; training uses `optax.softmax_cross_entropy_with_integer_labels`.

### 4.2 Uncertainty Loss
The uncertainty head produces `uncertainty ∈ [0,1]`. Its target is the normalized absolute error:
`target = |v_scalar - 0.5| * 2`
where `v_scalar` is the original value in `[-1,1]`. The loss is MSE, weighted by 0.01.

### 4.3 Batch Parallelism
- Self‑play uses `jax.vmap` to vectorize all operations over the batch of games.
- MCTS internally uses `mctx.muzero_policy`, which is designed for batch processing and runs efficiently on TPU.

### 4.4 Randomness
- Action sampling uses `jax.random.categorical` on the MCTS action weights (with a small `eps` to avoid zero probabilities).
- Root‑level Dirichlet noise is added by `mctx.muzero_policy` (parameters `dirichlet_fraction=0.25`, `dirichlet_alpha=0.3`).

## 5. Performance Optimizations
- **JIT compilation**: All heavy functions (`play_step`, `train_step`, internal MCTS functions) are compiled with `jax.jit`, eliminating Python overhead.
- **Batch processing**: `BATCH_SIZE` (number of concurrent games) can be increased to fully utilize TPU memory and compute.
- **Memory layout**: Data is stored in flat arrays and directly turned into JAX arrays without extra copies.
- **Asynchronous data loading**: PyTorch `DataLoader` prefetches batches on CPU while the TPU is busy training.
- **Zero-Overhead Dynamic Control Flow**: Features like randomized MCTS depths normally trigger expensive XLA recompilations on the TPU due to changing tensor dimensions. This project bypasses this by pre-compiling discrete branches of JIT functions (e.g., for different simulation counts) during initialization, allowing Python to dynamically dispatch calls without incurring any compilation penalty during the self-play loop.

## 6. Dependencies and Environment
- **Core**: `jax`, `jaxlib`, `flax`, `optax`, `mctx`, `pgx`, `orbax-checkpoint`
- **Support**: `torch` (only for data loading), `tensorboard`, `optuna` (for hyperparameter tuning)
- **Hardware**: TPU (v2/v3/v4) or GPU (with CUDA). Linux recommended; Windows via WSL2 is experimental.
- Installation: see `requirements.txt`.

## 7. Theoretical Comparison: Transformer vs CNN in Go AI
### 7.1 Limitations of CNNs
CNNs rely on stacking convolutional layers to gradually expand the receptive field. Information must propagate through many layers to connect distant parts of the board, which can lead to vanishing gradients or noise accumulation. To compensate, models like AlphaGo and KataGo incorporate many hand‑crafted input planes (e.g., liberties, ladder status, move history) that encode domain knowledge explicitly. While effective, this feature engineering restricts the model’s ability to discover higher‑level abstractions autonomously.

### 7.2 Advantages of Transformer Self‑Attention
The Transformer’s self‑attention mechanism allows each position to directly attend to every other position, regardless of distance. This makes it possible to capture global patterns (e.g., thickness, ladders) in a single layer and to learn concepts like “liberties”, “eyes”, and “group connectivity” from the raw board state alone, without hand‑crafted features. Moreover, the input can be extremely simple – in our case, we use the 17‑plane Pgx observation (which includes history) as a starting point. This rich input still benefits from the Transformer’s ability to model long‑range dependencies, and the architecture can later be simplified once the approach is validated.

### 7.3 Empirical Motivation
This project does not aim to “replace” CNNs, but rather to explore an alternative path: **with minimal prior knowledge, can a Transformer achieve competitive Go strength under pure reinforcement learning?** If successful, it would provide a cleaner, more generic baseline for future Go AI research.

## 8. Bayesian Optimization Activation Strategy
### 8.1 Why Not Enable Bayesian Pruning Early?
Bayesian pruning relies on the uncertainty head `uncertainty` to adjust the search width. During the initial stages of training, the model is random, and the uncertainty estimates are meaningless (often stuck near 0.5 or extreme values). Using pruning then would not only fail to focus search but might discard good moves, slowing down learning.

### 8.2 When to Enable It?
Once the model has been sufficiently trained (e.g., reaches amateur high‑dan level after supervised learning or early reinforcement learning), the uncertainty head becomes a reliable indicator of how confident the model is about a position (e.g., high uncertainty in complex life‑and‑death or ko fights). At that point, enabling Bayesian pruning can:
- In low‑uncertainty (simple) positions, aggressively prune to concentrate computation on a few likely moves.
- In high‑uncertainty (complex) positions, keep more candidates to avoid missing crucial tactics.

This **adaptive sparse search** accelerates self‑play while preserving or even improving playing strength.

### 8.3 Implementation
The training script provides a command‑line flag `--disable-bayesian` to turn off the pruning. A suggested schedule:
- **Phase 1–2**: Train with `--disable-bayesian` (standard MCTS) to quickly build a strong base.
- **Phase 3**: Re‑enable Bayesian pruning to fine‑tune and compare performance.

## 9. Conclusion
This project embodies several core design principles:
1. **Pure functional style**: All state transitions, randomness, and model evaluations are expressed as functions, enabling easy JIT compilation and massive parallelism.
2. **Bayesian‑guided pruning**: Leveraging the model’s own uncertainty to dynamically narrow the search space, leading to an efficient sparse MCTS on TPU.
3. **Architectural shift**: Replacing CNN with Transformer, learning from rich input (Pgx 17‑plane observations) and capable of reducing to simpler representations in the future.
4. **Research‑ready engineering**: Modular code, clear configuration, and detailed documentation provide a solid baseline for future experiments (e.g., hyperparameter tuning, architecture exploration).

## 10. Important Note on Policy Dimension
All policy‑related tensors in this implementation have dimension **362** (361 board intersections + 1 pass). This is consistently reflected in:
- `config.num_policy_outputs = 362`
- Model output shape `(batch, 362)`
- Self‑play saved policy arrays shape `(T*B, 362)`
- Data loader reshape using `362`

This ensures that the pass move is properly included in both search and training.
