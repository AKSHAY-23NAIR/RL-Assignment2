"""
Plot PPO learning curve averaged over multiple seeds.

Usage:
    python plot_learning_curve.py `
        out/ppo_experiments/ppo_full_A1_grid_seed0.pt `
        out/ppo_experiments/ppo_full_A1_grid_seed1.pt `
        out/ppo_experiments/ppo_full_A1_grid_seed2.pt

    python plot_learning_curve.py `
        out/ppo_experiments/ppo_full_restaurant_delivery_grid_seed0.pt `
        out/ppo_experiments/ppo_full_restaurant_delivery_grid_seed1.pt `
        out/ppo_experiments/ppo_full_restaurant_delivery_grid_seed2.pt
"""

from argparse import ArgumentParser
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = ArgumentParser(description="Plot averaged PPO learning curve.")
    p.add_argument(
        "checkpoints",
        nargs="+",
        type=Path,
        help="Paths to saved .pt checkpoint files."
    )
    p.add_argument(
        "--window",
        type=int,
        default=50,
        help="Rolling average window size (default: 50)."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to save plot. If not set, shows interactively."
    )
    return p.parse_args()


def rolling_average(values, window):
    result = np.zeros(len(values))
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result[i] = np.mean(values[start:i + 1])
    return result


def main():
    args = parse_args()

    all_rewards = []

    for ckpt_path in args.checkpoints:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        rewards = checkpoint.get("episode_rewards")

        if rewards is None or len(rewards) == 0:
            print(f"No rewards found in {ckpt_path}")
            continue

        print(f"{ckpt_path.name}: {len(rewards)} episodes")
        all_rewards.append(np.array(rewards))

    if len(all_rewards) == 0:
        print("No valid reward histories found.")
        return

    # Use shortest run length so all seeds align
    min_len = min(len(r) for r in all_rewards)
    all_rewards = np.array([r[:min_len] for r in all_rewards])

    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    smoothed = rolling_average(mean_rewards, args.window)
    episodes = np.arange(1, min_len + 1)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Mean return
    ax.plot(
        episodes,
        mean_rewards,
        alpha=0.3,
        linewidth=1,
        label="Mean episode return"
    )

    # ±1 std region
    ax.fill_between(
        episodes,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        label="±1 std"
    )

    # Smoothed mean
    ax.plot(
        episodes,
        smoothed,
        linewidth=2,
        label=f"Rolling avg (window={args.window})"
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"PPO Learning Curve ({len(all_rewards)} seeds)")
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