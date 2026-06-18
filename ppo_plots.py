import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def rolling_average(values, window):
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()

def plot_dqn_convergence_like_ppo(grid_name, checkpoint_paths, out_path, window=50, color="steelblue"):
    """
    Plots DQN convergence matching the PPO visual style exactly.
    Aligns seeds by interpolating to the MAX episode length instead of truncating.
    """
    all_rewards = []
    for ckpt_path in checkpoint_paths:
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint not found: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        rewards = ckpt.get("episode_rewards", [])
        if len(rewards) > 0:
            all_rewards.append(np.array(rewards))

    if not all_rewards:
        print(f"No valid episode_rewards found for {grid_name}")
        return

    # Find the MAX length among seeds so we don't throw away data
    max_len = max(len(r) for r in all_rewards)
    target_episodes = np.arange(1, max_len + 1)
    
    aligned_rewards = []
    for r in all_rewards:
        # If a seed is shorter, smoothly stretch it to the max_len
        current_episodes = np.arange(1, len(r) + 1)
        interp_r = np.interp(target_episodes, current_episodes, r)
        aligned_rewards.append(interp_r)

    aligned = np.array(aligned_rewards)  # shape (n_seeds, max_len)

    mean_curve = aligned.mean(axis=0)
    std_curve  = aligned.std(axis=0)
    rolling    = rolling_average(mean_curve, window)

    fig, ax = plt.subplots(figsize=(10, 5))

    # 1. Thin mean line
    ax.plot(target_episodes, mean_curve, color=color, alpha=0.35,
            linewidth=0.8, label="Mean episode return")

    # 2. pm1 std shaded band
    ax.fill_between(target_episodes,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color=color, alpha=0.15, label="±1 std")

    # 3. Rolling average (thick orange line)
    ax.plot(target_episodes, rolling, color="darkorange", linewidth=2,
             label=f"Rolling avg (window={window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"DQN Convergence — {grid_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Set x-limits to perfectly frame the data
    ax.set_xlim(0, max_len)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")


# ============================================================
# GENERATE TWO BASELINE PLOTS
# ============================================================

OUT_DIR = Path("out/dqn_experiments")

# Plot 1: A1_grid Baseline (Full variant)
a1_seeds = [OUT_DIR / f"dqn_A1_grid_seed{i}.pt" for i in range(3)]
plot_dqn_convergence_like_ppo("A1_grid", a1_seeds, "out/Convergence_plot_DQN_A1.png")

# Plot 2: restaurant_delivery_grid Baseline (Full variant)
restaurant_seeds = [OUT_DIR / f"dqn_restaurant_delivery_grid_seed{i}.pt" for i in range(3)]
plot_dqn_convergence_like_ppo("restaurant_delivery_grid", restaurant_seeds, "out/Convergence_plot_DQN_Restaurant.png")