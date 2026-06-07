"""
Plot PPO Learning Curve.

Loads a saved PPO checkpoint and plots the episode rewards over training.

Usage:
    python plot_learning_curve.py out/ppo_example_grid_seed0.pt
"""

from argparse import ArgumentParser
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = ArgumentParser(description="Plot PPO learning curve.")
    p.add_argument("checkpoint", type=Path,
                   help="Path to the saved .pt checkpoint file.")
    p.add_argument("--window", type=int, default=50,
                   help="Rolling average window size (default: 50).")
    p.add_argument("--out", type=Path, default=None,
                   help="Path to save the plot. If not set, shows interactively.")
    return p.parse_args()


def rolling_average(values: list[float], window: int) -> np.ndarray:
    """Compute rolling average with the given window size."""
    result = np.zeros(len(values))
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result[i] = np.mean(values[start:i + 1])
    return result


def main():
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    episode_rewards = checkpoint.get("episode_rewards", None)

    if episode_rewards is None or len(episode_rewards) == 0:
        print("No episode rewards found in checkpoint.")
        return

    print(f"Loaded {len(episode_rewards)} episodes from {args.checkpoint}")
    print(f"  Min return:  {min(episode_rewards):.1f}")
    print(f"  Max return:  {max(episode_rewards):.1f}")
    print(f"  Final avg (last 50): {np.mean(episode_rewards[-50:]):.1f}")

    episodes = np.arange(1, len(episode_rewards) + 1)
    rewards = np.array(episode_rewards)
    smoothed = rolling_average(episode_rewards, args.window)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(episodes, rewards, color="steelblue", alpha=0.3, linewidth=0.8,
            label="Episode return")
    ax.plot(episodes, smoothed, color="steelblue", linewidth=2,
            label=f"Rolling avg (window={args.window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"PPO Learning Curve — {args.checkpoint.stem}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.out is not None:
        plt.savefig(args.out, dpi=150)
        print(f"Plot saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()