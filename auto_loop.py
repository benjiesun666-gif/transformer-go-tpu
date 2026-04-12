import os
import glob
import time
import subprocess
import json

# Enable JAX compilation cache to disk.
os.environ['JAX_COMPILATION_CACHE_DIR'] = './jax_cache'


def run_command(command: str):
    # Execute shell command and halt execution upon failure.
    print(f"Executing: {command}")
    process = subprocess.Popen(command, shell=True)
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with exit code {process.returncode}: {command}")
        exit(process.returncode)


def get_optimized_lr(default_lr="2e-4"):
    """
    Retrieves the best learning rate found by the most recent Bayesian optimization.
    """
    if os.path.exists("best_train_params.json"):
        try:
            with open("best_train_params.json", "r") as f:
                params = json.load(f)
            return params.get("lr", default_lr)
        except:
            return default_lr
    return default_lr


def main():
    """
    Full-featured automated pipeline with Bayesian LR tuning and persistent buffer management.
    """
    print("Starting high-performance automated execution pipeline...")
    iteration = 1

    from config import ModelConfig
    base_config = ModelConfig()

    # Set buffer size to 200 to ensure a robust training window.
    MAX_REPLAY_BUFFER = 200

    # Ensure the compilation cache directory exists.
    os.makedirs('./jax_cache', exist_ok=True)

    while True:
        print(f"\n--- Pipeline Iteration {iteration} [Architecture: {base_config.model_type.upper()}] ---")

        # 1. Generate self-play data.
        run_command("python3 -u tpu_selfplay.py")

        # 2. Manage the sliding window replay buffer (FIFO).
        data_files = sorted(glob.glob(os.path.join(base_config.data_dir, "*.npz")))
        if len(data_files) > MAX_REPLAY_BUFFER:
            num_to_delete = len(data_files) - MAX_REPLAY_BUFFER
            for f in data_files[:num_to_delete]:
                try:
                    os.remove(f)
                except Exception as e:
                    pass

        # Re-fetch the updated list of data files after cleanup.
        data_files = sorted(glob.glob(os.path.join(base_config.data_dir, "*.npz")))

        # 3. Dynamic Hyperparameter Tuning (The "Scientific" Part)
        # Run Bayesian optimization for the learning rate every 5 iterations.
        if iteration % 5 == 0 and len(data_files) >= 5:
            print("🔍 Running Bayesian Optimization to find the optimal LR for current data distribution...")
            run_command("python3 -u tpu_train.py --tune --n-trials 10 --batch-size 1024")

        # 4. Train the model using the LATEST optimized LR from Bayesian trials.
        # Implemented a warm-up phase to prevent overfitting to initial random play data.
        if len(data_files) < 5:
            print(
                f"⏳ Data warm-up in progress... Current files: {len(data_files)}/5. Skipping training to accumulate more data.")
        else:
            current_lr = get_optimized_lr()
            # Dynamic epochs: Train more passes when data is scarce, fewer when data is abundant.
            dynamic_epochs = 3 if len(data_files) < 20 else 1
            print(f"🎯 Applying Optimized Learning Rate: {current_lr} | Epochs: {dynamic_epochs}")
            run_command(f"python3 -u tpu_train.py --batch-size 1024 --epochs {dynamic_epochs} --lr {current_lr}")

        # 5. Tune MCTS search parameters.
        if iteration % 5 == 0 and len(data_files) >= 5:
            run_command("python3 -u tune_search.py")

        # 6. Periodic Weight Backup.
        if iteration % 5 == 0:
            src = base_config.checkpoint_dir
            dst = f"./backup_{base_config.model_type}/iter_{iteration}"
            run_command(f"mkdir -p {dst} && cp -r {src}/* {dst}/")

        iteration += 1
        time.sleep(5)


if __name__ == "__main__":
    main()
