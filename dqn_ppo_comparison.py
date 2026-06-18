import csv
import pandas as pd

HEADER_16 = ["config", "grid", "seed", "entropy_coeff", "gae_lambda",
             "epsilon_clip", "trajectory_length", "hidden_size",
             "timesteps", "updates", "episodes", "final_avg_50",
             "success_rate", "avg_eval_reward", "avg_eval_steps",
             "checkpoint"]

HEADER_11 = ["variant", "grid", "seed", "timesteps", "updates",
             "episodes", "final_avg_50", "success_rate",
             "avg_eval_reward", "avg_eval_steps", "checkpoint"]

HEADER_18 = ["config", "grid", "seed", "entropy_coeff", "gae_lambda",
             "epsilon_clip", "trajectory_length", "hidden_size",
             "learning_rate_actor", "gamma", "timesteps", "updates",
             "episodes", "final_avg_50", "success_rate",
             "avg_eval_reward", "avg_eval_steps", "checkpoint"]

def load_mixed_ppo_csv(path):
    rows_11, rows_16, rows_18 = [], [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0] in ("variant", "config"):
                continue
            n = len(row)
            if n == 11:
                rows_11.append(row)
            elif n == 16:
                rows_16.append(row)
            elif n == 18:
                rows_18.append(row)
            else:
                print(f"Unknown schema, {n} cols: {row[:3]}...")

    df_11 = pd.DataFrame(rows_11, columns=HEADER_11) if rows_11 else pd.DataFrame()
    df_16 = pd.DataFrame(rows_16, columns=HEADER_16) if rows_16 else pd.DataFrame()
    df_18 = pd.DataFrame(rows_18, columns=HEADER_18) if rows_18 else pd.DataFrame()

    if not df_11.empty:
        df_11 = df_11.rename(columns={"variant": "config"})

    # numeric conversion for all three
    for df, numcols in [
        (df_11, ["seed","timesteps","updates","episodes","final_avg_50",
                 "success_rate","avg_eval_reward","avg_eval_steps"]),
        (df_16, ["seed","entropy_coeff","gae_lambda","epsilon_clip",
                 "trajectory_length","hidden_size","timesteps","updates",
                 "episodes","final_avg_50","success_rate","avg_eval_reward",
                 "avg_eval_steps"]),
        (df_18, ["seed","entropy_coeff","gae_lambda","epsilon_clip",
                 "trajectory_length","hidden_size","learning_rate_actor",
                 "gamma","timesteps","updates","episodes","final_avg_50",
                 "success_rate","avg_eval_reward","avg_eval_steps"]),
    ]:
        if not df.empty:
            for c in numcols:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    return df_11, df_16, df_18


df11, df16, df18 = load_mixed_ppo_csv("out/ppo_experiments/ppo_experiment_summary.csv")
print(f"11-col (ablations): {len(df11)} rows")
print(f"16-col (sweeps missing lr/gamma): {len(df16)} rows")
print(f"18-col (full sweeps): {len(df18)} rows")
print("\n16-col configs:", sorted(df16['config'].unique()) if not df16.empty else "none")
print("18-col configs:", sorted(df18['config'].unique()) if not df18.empty else "none")

import seaborn as sns
import matplotlib.pyplot as plt

def plot_sweep_simple(df_ppo, df_dqn, configs_order, x_label,
                       grid_filter, title_prefix, out_name):
    ppo_sub = df_ppo[(df_ppo['config'].isin(configs_order)) &
                      (df_ppo['grid'] == grid_filter)].copy()
    ppo_sub['method'] = 'PPO'

    dqn_sub = pd.DataFrame()
    if df_dqn is not None and not df_dqn.empty:
        dqn_sub = df_dqn[(df_dqn['config'].isin(configs_order)) &
                          (df_dqn['grid'] == grid_filter)].copy()
        dqn_sub['method'] = 'DQN'

    combined = pd.concat([ppo_sub, dqn_sub], ignore_index=True)
    if combined.empty:
        print(f"No data for {title_prefix} on {grid_filter}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = ['success_rate', 'avg_eval_reward', 'avg_eval_steps']
    labels = ['Success Rate', 'Avg Eval Reward', 'Avg Eval Steps']

    for ax, metric, label in zip(axes, metrics, labels):
        sns.barplot(data=combined, x='config', y=metric, hue='method',
                    order=configs_order, ax=ax, errorbar='sd',
                    capsize=0.08, edgecolor='black', linewidth=0.8)
        ax.set_title(f'{label}')
        ax.set_xlabel(x_label)
        ax.tick_params(axis='x', rotation=20)

    fig.suptitle(f'{title_prefix} — {grid_filter}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"out/{out_name}", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved out/{out_name}")


# load your DQN sweep CSVs as before (these are clean, no corruption)
dqn_hidden = pd.read_csv("out/dqn_experiments/dqn_sweep_hidden_summary.csv")
dqn_lr     = pd.read_csv("out/dqn_experiments/dqn_sweep_lr_summary.csv")
dqn_gamma  = pd.read_csv("out/dqn_experiments/dqn_sweep_discount_summary.csv")

# shared sweeps: DQN vs PPO
plot_sweep_simple(df16, dqn_hidden,
    ["hidden_64","hidden_128","hidden_256"], "Hidden Size",
    "restaurant_delivery_grid", "Hidden Size Sweep", "sweep_hidden.png")

plot_sweep_simple(df18, dqn_gamma,
    ["gamma_090","gamma_095","gamma_099"], "Gamma",
    "restaurant_delivery_grid", "Discount Factor Sweep", "sweep_gamma.png")

# PPO-only sweeps: no DQN equivalent needed
plot_sweep_simple(df16, None,
    ["entropy_0","entropy_001","entropy_005"], "Entropy Coeff",
    "restaurant_delivery_grid", "Entropy Sweep (PPO only)", "sweep_entropy.png")

plot_sweep_simple(df16, None,
    ["gae_090","gae_095","gae_099"], "GAE Lambda",
    "restaurant_delivery_grid", "GAE Sweep (PPO only)", "sweep_gae.png")

plot_sweep_simple(df16, None,
    ["clip_01","clip_02","clip_03"], "Clip Epsilon",
    "restaurant_delivery_grid", "Clip Sweep (PPO only)", "sweep_clip.png")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def rolling_average(values, window):
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()


def plot_dqn_convergence(grid_name, checkpoint_paths, out_path,
                          window=50, color="steelblue"):
    """
    Replicates the PPO convergence plot style:
    - thin line: mean episode return across seeds, per episode index
    - shaded band: ±1 std across seeds, per episode index
    - thick line: rolling average (window=50) of the mean curve

    Args:
        grid_name: label for the title (e.g. "A1_grid")
        checkpoint_paths: list of .pt file paths, one per seed,
                           for the SAME config (e.g. 'full')
        out_path: where to save the .png
    """
    all_rewards = []
    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        rewards = ckpt.get("episode_rewards", [])
        if rewards:
            all_rewards.append(rewards)

    if not all_rewards:
        print(f"No episode_rewards found for {grid_name}")
        return

    # align to the shortest run length so seeds can be averaged
    # at matching episode indices
    min_len = min(len(r) for r in all_rewards)
    aligned = np.array([r[:min_len] for r in all_rewards])  # shape (n_seeds, min_len)

    mean_curve = aligned.mean(axis=0)
    std_curve  = aligned.std(axis=0)
    rolling    = rolling_average(mean_curve, window)

    episodes = np.arange(1, min_len + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    # thin mean line
    ax.plot(episodes, mean_curve, color=color, alpha=0.35,
            linewidth=0.8, label="Mean episode return")

    # ±1 std shaded band
    ax.fill_between(episodes,
                     mean_curve - std_curve,
                     mean_curve + std_curve,
                     color=color, alpha=0.15, label="±1 std")

    # rolling average (thick line)
    ax.plot(episodes, rolling, color="darkorange", linewidth=2,
             label=f"Rolling avg (window={window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"DQN Convergence — {grid_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ============================================================
# RUN FOR BOTH GRIDS
# ============================================================

OUT_DIR = Path("out/dqn_experiments")

# A1_grid — full config, 3 seeds
a1_checkpoints = [
    OUT_DIR / "dqn_A1_grid_seed0.pt",
    OUT_DIR / "dqn_A1_grid_seed1.pt",
    OUT_DIR / "dqn_A1_grid_seed2.pt",
]
plot_dqn_convergence("A1_grid", a1_checkpoints,
                      "out/Convergence_plot_DQN_A1.png")

# restaurant_delivery_grid — full config, 3 seeds
restaurant_checkpoints = [
    OUT_DIR / "dqn_restaurant_delivery_grid_seed0.pt",
    OUT_DIR / "dqn_restaurant_delivery_grid_seed1.pt",
    OUT_DIR / "dqn_restaurant_delivery_grid_seed2.pt",
]
plot_dqn_convergence("restaurant_delivery_grid", restaurant_checkpoints,
                      "out/Convergence_plot_DQN_Restaurant.png")

