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
    print("Starting automated execution pipeline...")
    iteration = 1

    while True:
        print(f"\n--- Pipeline Iteration {iteration} ---")

        # Step 1: Generate self-play data.
        run_command("python tpu_selfplay.py")

        # Step 2: Train the neural network model.
        run_command("python tpu_train.py --batch-size 128 --epochs 5 --lr 2e-4")

        # Step 3: Optimize MCTS parameters via asymmetric play.
        run_command("python tune_search.py")

        iteration += 1
        time.sleep(5)


if __name__ == "__main__":
    main()