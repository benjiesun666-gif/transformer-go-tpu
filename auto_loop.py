import os
import glob
import time
import subprocess


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
    Ensures the model trains on the most recent self-play data while
    limiting disk usage and training time.
    """
    print("Starting automated execution pipeline...")
    iteration = 1

    # Load config to access model_type and dynamic data/checkpoint paths
    from config import ModelConfig
    base_config = ModelConfig()

    # Define the maximum number of self-play files to keep in the buffer
    # 200 files approx. 1.28 million samples, providing a robust training window
    MAX_REPLAY_BUFFER = 200

    while True:
        print(f"\n--- Pipeline Iteration {iteration} [Architecture: {base_config.model_type.upper()}] ---")

        # 1. Generate new self-play data
        run_command("python tpu_selfplay.py")

        # 2. Manage the sliding window replay buffer (FIFO)
        # Search for all compressed NPZ files in the model-specific data directory
        data_files = sorted(glob.glob(os.path.join(base_config.data_dir, "*.npz")))

        if len(data_files) > MAX_REPLAY_BUFFER:
            # Calculate how many files need to be removed
            num_to_delete = len(data_files) - MAX_REPLAY_BUFFER
            files_to_delete = data_files[:num_to_delete]

            print(f"♻️ Replay buffer full. Removing {num_to_delete} oldest file(s)...")
            for f in files_to_delete:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Warning: Failed to delete {f}: {e}")

        # 3. Train the model on the current buffer
        run_command("python tpu_train.py --batch-size 1024 --epochs 3 --lr 2e-4")

        # 4. Tune MCTS search parameters
        run_command("python tune_search.py")

        # 5. Backup weights periodically (every 5 iterations) for long-term recovery
        if iteration % 5 == 0:
            src = base_config.checkpoint_dir
            dst = f"./backup_{base_config.model_type}/iter_{iteration}"
            print(f"📦 Backing up weights to {dst}...")
            run_command(f"mkdir -p {dst} && cp -r {src}/* {dst}/")

        iteration += 1
        # Brief pause to allow system cooling and I/O stabilization
        time.sleep(5)


if __name__ == "__main__":
    main()
