import os
import glob
import time
import subprocess

# Enable JAX compilation cache to disk.
# This allows subsequent subprocess calls to reuse compiled XLA executables.
os.environ['JAX_COMPILATION_CACHE_DIR'] = './jax_cache'


def run_command(command: str):
    # Execute shell command and halt execution upon failure.
    print(f"Executing: {command}")
    process = subprocess.Popen(command, shell=True)
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with exit code {process.returncode}: {command}")
        exit(process.returncode)


def main():
    """
    Automated training pipeline with a sliding window (FIFO) replay buffer.
    """
    print("Starting automated execution pipeline...")
    iteration = 1

    from config import ModelConfig
    base_config = ModelConfig()

    MAX_REPLAY_BUFFER = 50

    # Ensure the compilation cache directory exists.
    os.makedirs('./jax_cache', exist_ok=True)

    while True:
        print(f"\n--- Pipeline Iteration {iteration} [Architecture: {base_config.model_type.upper()}] ---")

        # 1. Generate self-play data. Utilizes the persistent compilation cache.
        run_command("python3 -u tpu_selfplay.py")

        # 2. Manage the sliding window replay buffer (FIFO) by removing the oldest data files.
        data_files = sorted(glob.glob(os.path.join(base_config.data_dir, "*.npz")))
        if len(data_files) > MAX_REPLAY_BUFFER:
            num_to_delete = len(data_files) - MAX_REPLAY_BUFFER
            for f in data_files[:num_to_delete]:
                try:
                    os.remove(f)
                except Exception as e:
                    pass

        # 3. Train the model. Batch size is scaled to utilize 8 TPU cores.
        run_command("python3 -u tpu_train.py --batch-size 1024 --epochs 3 --lr 2e-4")

        # 4. Run hyperparameter tuning for the MCTS search. Executed every 5 iterations.
        if iteration % 5 == 0:
            run_command("python3 -u tune_search.py")

        # 5. Backup model weights periodically for recovery.
        if iteration % 5 == 0:
            src = base_config.checkpoint_dir
            dst = f"./backup_{base_config.model_type}/iter_{iteration}"
            run_command(f"mkdir -p {dst} && cp -r {src}/* {dst}/")

        iteration += 1
        time.sleep(5)


if __name__ == "__main__":
    main()
