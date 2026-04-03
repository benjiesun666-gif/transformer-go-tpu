# Bayesian AlphaZero: Pure JAX/TPU Architecture

This repository contains a highly optimized, fully XLA-compiled implementation of the AlphaZero algorithm, specifically designed for Google TPU and GPU accelerators. 

The core innovation of this project is the integration of a **Bayesian Uncertainty Exploration** mechanism directly into the XLA computation graph, replacing traditional Dirichlet noise with a dynamic, network-driven action pruning strategy during the Monte Carlo Tree Search (MCTS).

## 🏗️ Architecture Overview

The pipeline strictly adheres to the pure functional programming paradigm required by JAX, ensuring zero state mutations and zero closure captures for neural network parameters. This allows the entire self-play and training loop to be fully JIT-compiled (`@jax.jit`) or mapped across distributed devices (`jax.pmap`).

### Tech Stack
* **Compute & Neural Networks:** JAX, Flax, Optax
* **Environment State Machine:** [Pgx](https://github.com/sotetsuk/pgx) (Go 19x19, fully tensorized)
* **Tree Search:** [DeepMind MCTX](https://github.com/google-deepmind/mctx) (Batched MCTS in XLA)
* **Hyperparameter Tuning:** Optuna (Automated TPE search with Median Pruning)
* **I/O & Logging:** PyTorch `DataLoader` (CPU multi-processing) and `TensorBoard`
* **Checkpointing:** Google `orbax.checkpoint`

## 🧠 Core Technical Implementations

### 1. Bayesian Uncertainty Head & MCTS Pruning
Unlike standard dual-tower (Policy/Value) architectures, the Flax Transformer in this project features a tripartite output: `(policy_logits, value_logits, uncertainty)`. 
* **Uncertainty Tensor:** Outputted via a Sigmoid activation, representing the epistemic uncertainty of the network for a given state.
* **XLA-Native Pruning:** During the `recurrent_fn` and `root_fn` expansion phases in MCTX, the `jax_bayesian.py` module applies a dynamic tensor mask. Candidate actions are pruned based on a computed threshold: `score = prior + (uncertainty * exploration_weight * (1 - prior))`. Actions falling below the dynamic top-K threshold are masked with `-inf`, forcing the MCTS to focus compute on high-confidence branches or highly uncertain (novel) states.

### 2. Pure Functional Bridge (`pgx_mctx_bridge.py`)
The MCTS implementation explicitly passes the Flax model parameters (`params`) through the `mctx.muzero_policy` call tree. This eliminates implicit closure captures, ensuring deterministic JIT compilation caching and allowing seamless, asynchronous parameter updates between the training loop and self-play workers.

### 3. Data Pipeline & Value Retro-propagation
The self-play engine (`tpu_selfplay.py`) evaluates hundreds of parallel environments. Upon terminal states, true game outcomes `[-1, 0, 1]` are retro-propagated through the trajectory and strictly mapped to 128 categorical Value Buckets `[0, 127]` to align with the Cross-Entropy value loss. Data is serialized to compressed `.npz` files.
PyTorch's `DataLoader` is utilized purely on the CPU side (`data_utils.py`) to handle asynchronous, multi-worker I/O and batch concatenation, before converting batches to `jnp.array` for TPU ingestion.

### 4. Advanced Optimizer Stack & SWA
To jump out of sharp local minima and improve generalization, the default AdamW optimizer has been replaced with a custom stack: **SGD + Lookahead**. Furthermore, **Stochastic Weight Averaging (SWA)** is manually tracked across epochs using pure `jax.tree_map` operations, maintaining a smoothed parameter state that drastically increases self-play stability.

## 📂 Codebase Structure

| File | Description |
| :--- | :--- |
| `config.py` | `flax.struct.dataclass` defining static hyperparameters for the Transformer and Bayesian optimizer. |
| `tpu_model.py` | Flax module definitions (Conv stem -> Positional Encoding -> Transformer Blocks -> Policy/Value/Uncertainty Heads). |
| `jax_bayesian.py` | Pure JAX mathematical operations for dynamic thresholding and MCTS action masking. |
| `pgx_mctx_bridge.py` | The JAX-native bridge linking Pgx environment transitions with the MCTX search loop. |
| `tpu_selfplay.py` | Batched self-play inference engine. Generates trajectories and handles reward mapping. |
| `data_utils.py` | CPU-bound PyTorch `Dataset` and `DataLoader` for efficient `.npz` parsing and batching. |
| `tpu_train.py` | The main Optax training loop with Orbax checkpointing and TensorBoard scalar logging. |
| `utils.py` | Stateless helper functions (RNG seeding, win-rate tracking). |

## 🚀 Usage Instructions

### Dependencies
```bash
pip install jax flax optax mctx pgx orbax-checkpoint torch tensorboard optuna
(Ensure jaxlib is correctly configured for your specific TPU/CUDA environment).

1. Data Generation (Self-Play)
Execute the batched inference engine to generate training trajectories:

Bash
python tpu_selfplay.py
Outputs are saved to ./tpu_data/selfplay_batch_{timestamp}.npz.

2. Model Training
Run the training loop on the generated XLA tensors:

Bash
python tpu_train.py --batch-size 512 --epochs 50 --lr 2e-4
Checkpoints are managed by Orbax in ./tpu_checkpoints/.

3. Hyperparameter Tuning (Optuna)
Instead of standard training, you can launch an automated hyperparameter search (tuning learning rate, model depth, heads, etc.) using validation loss as the pruning metric:

```bash
python tpu_train.py --tune --n-trials 20 --batch-size 128 --epochs 5

4. Metric Monitoring
Launch TensorBoard to monitor Cross-Entropy (Policy/Value) and MSE (Uncertainty) losses:

Bash
tensorboard --logdir=./tpu_logs
