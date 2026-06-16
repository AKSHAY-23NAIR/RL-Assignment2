"""
Full comparison and plotting script for DQN vs PPO.

Handles:
- Clean DQN CSVs (single schema)
- Mixed-schema PPO CSV (11-col ablations + 18-col sweeps in same file)
- Convergence plots
- Main comparison matrix (success/reward/steps x grid x config)
- Novelty bonus ablation
- Hyperparameter sweeps (hidden, lr, gamma) for DQN vs PPO
- PPO-only sweeps (entropy, gae, clip)
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path

sns.set_theme(style="whitegrid")
OUT = Path("out")
OUT.mkdir(exist_ok=True)


# ============================================================
# LOADERS
# ============================================================

def safe_read_csv(path, **kwargs):
    p = Path(path)
    if not p.exists():
        print(f"Missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(p, **kwargs)


def load_mixed_ppo_csv(path):
    """
    Parses ppo_experiment_summary.csv, which contains two
    different row schemas mixed together:

    - 11 columns: ablation runs (full, no_action_mask, ...)
        variant,grid,seed,timesteps,updates,episodes,
        final_avg_50,success_rate,avg_eval_reward,
        avg_eval_steps,checkpoint

    - 18 columns: hyperparameter sweep runs (entropy_*, gae_*,
      clip_*, hidden_*, lr_actor_*, gamma_*)
        config,grid,seed,entropy_coeff,gae_lambda,epsilon_clip,
        trajectory_length,hidden_size,learning_rate_actor,gamma,
        timesteps,updates,episodes,final_avg_50,success_rate,
        avg_eval_reward,avg_eval_steps,checkpoint

    Returns (df_ablations, df_sweeps) as two DataFrames.
    """
    HEADER_11 = ["variant", "grid", "seed", "timesteps", "updates",
                 "episodes", "final_avg_50", "success_rate",
                 "avg_eval_reward", "avg_eval_steps", "checkpoint"]

    HEADER_18 = ["config", "grid", "seed", "entropy_coeff", "gae_lambda",
                 "epsilon_clip", "trajectory_length", "hidden_size",
                 "learning_rate_actor", "gamma", "timesteps", "updates",
                 "episodes", "final_avg_50", "success_rate",
                 "avg_eval_reward", "avg_eval_steps", "checkpoint"]

    rows_11, rows_18 = [], []

    p = Path(path)
    if not p.exists():
        print(f"Missing: {path}")
        return pd.DataFrame(), pd.DataFrame()

    with open(p, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0] in ("variant", "config"):
                continue  # skip header rows
            n = len(row)
            if n == 11:
                rows_11.append(row)
            elif n == 18:
                rows_18.append(row)
            else:
                print(f"  Skipping row with {n} cols: {row[:3]}...")

    df_11 = pd.DataFrame(rows_11, columns=HEADER_11) if rows_11 else pd.DataFrame()
    df_18 = pd.DataFrame(rows_18, columns=HEADER_18) if rows_18 else pd.DataFrame()

    numeric_11 = ["seed", "timesteps", "updates", "episodes", "final_avg_50",
                  "success_rate", "avg_eval_reward", "avg_eval_steps"]
    numeric_18 = ["seed", "entropy_coeff", "gae_lambda", "epsilon_clip",
                  "trajectory_length", "hidden_size", "learning_rate_actor",
                  "gamma", "timesteps", "updates", "episodes",
                  "final_avg_50", "success_rate", "avg_eval_reward",
                  "avg_eval_steps"]

    if not df_11.empty:
        for c in numeric_11:
            df_11[c] = pd.to_numeric(df_11[c], errors="coerce")
        df_11 = df_11.rename(columns={"variant": "config"})

    if not df_18.empty:
        for c in numeric_18:
            df_18[c] = pd.to_numeric(df_18[c], errors="coerce")

    return df_11, df_18


# ============================================================
# LOAD MAIN RESULTS
# ============================================================

# DQN main (clean single schema)
dqn_main = safe_read_csv("out/dqn_experiments/dqn_experiment_summary.csv")
if not dqn_main.empty:
    dqn_main = dqn_main[dqn_main['timesteps'] >= 10000]
    if 'variant' in dqn_main.columns:
        dqn_main = dqn_main.rename(columns={'variant': 'config'})
    dqn_main['method'] = 'DQN'

# PPO main (mixed schema)
ppo_ablations, ppo_sweeps = load_mixed_ppo_csv(
    "out/ppo_experiments/ppo_experiment_summary.csv"
)
if not ppo_ablations.empty:
    ppo_ablations = ppo_ablations[ppo_ablations['timesteps'] >= 10000]
    ppo_ablations['method'] = 'PPO'
if not ppo_sweeps.empty:
    ppo_sweeps['method'] = 'PPO'

print(f"DQN main rows: {len(dqn_main)}")
print(f"PPO ablation rows: {len(ppo_ablations)}")
print(f"PPO sweep rows: {len(ppo_sweeps)}")
if not ppo_sweeps.empty:
    print(f"PPO sweep configs: {sorted(ppo_sweeps['config'].unique())}")

# combined ablation comparison data
both = pd.concat([dqn_main, ppo_ablations], ignore_index=True)
if not both.empty:
    both = both[both['config'].isin(
        ['full', 'no_action_mask', 'no_novelty', 'long_rollout']
    )]


# ============================================================
# PLOT 1 — CONVERGENCE CURVES (per grid, full variant)
# ============================================================

def plot_convergence(df, grid_name, out_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, method in zip(axes, ['DQN', 'PPO']):
        sub = df[(df['method'] == method) &
                 (df['config'] == 'full') &
                 (df['grid'] == grid_name)]
        for _, row in sub.iterrows():
            try:
                ckpt = torch.load(row['checkpoint'], map_location='cpu')
                rewards = ckpt.get('episode_rewards', [])
                if rewards:
                    smoothed = pd.Series(rewards).rolling(50, min_periods=1).mean()
                    ax.plot(smoothed, alpha=0.8, label=f"seed {int(row['seed'])}")
            except Exception as e:
                print(f"Could not load {row['checkpoint']}: {e}")
        ax.set_title(f'{method} Convergence — {grid_name}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward (50-ep rolling avg)')
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / out_name, dpi=300, bbox_inches='tight')
    plt.close()


if not both.empty:
    for grid in both['grid'].dropna().unique():
        plot_convergence(both, grid, f"convergence_{grid}.png")


# ============================================================
# PLOT 2 — MAIN COMPARISON MATRIX
# ============================================================

if not both.empty:
    main = both[both['config'].isin(['full', 'no_novelty'])]
    grids = sorted(main['grid'].dropna().unique())
    metrics = ['success_rate', 'avg_eval_reward', 'avg_eval_steps']
    metric_labels = ['Success Rate', 'Avg Evaluation Reward', 'Avg Evaluation Steps']
    colors = {"DQN": "#1f77b4", "PPO": "#ff7f0e"}

    fig, axes = plt.subplots(len(grids), 3, figsize=(16, 5 * len(grids)))
    if len(grids) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, grid in enumerate(grids):
        grid_data = main[main['grid'] == grid]
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            sns.barplot(data=grid_data, x="config", y=metric,
                        hue="method", ax=ax, palette=colors,
                        errorbar="sd", capsize=0.08,
                        edgecolor="black", linewidth=0.8)
            ax.set_title(f"{metric_labels[col_idx]} — {grid}",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel("Config")
            ax.set_ylabel(metric_labels[col_idx])
            ax.tick_params(axis="x", rotation=15)
            if row_idx == 0 and col_idx == 0:
                ax.legend(title='Method')
            else:
                leg = ax.get_legend()
                if leg:
                    leg.remove()

    plt.tight_layout()
    plt.savefig(OUT / "main_comparison_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# PLOT 3 — NOVELTY ABLATION (A1_grid)
# ============================================================

if not both.empty:
    novelty = both[both['config'].isin(['full', 'no_novelty'])]
    a1 = novelty[novelty['grid'] == 'A1_grid']

    if not a1.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        nov_summary = a1.groupby(['method', 'config']).agg(
            steps_mean=('avg_eval_steps', 'mean'),
            steps_std=('avg_eval_steps', 'std'),
        ).reset_index()

        configs = ['full', 'no_novelty']
        x = np.arange(len(configs))
        width = 0.35

        for i, method in enumerate(['DQN', 'PPO']):
            sub = nov_summary[nov_summary['method'] == method]
            sub = sub.set_index('config').reindex(configs).reset_index()
            axes[0].bar(x + i*width, sub['steps_mean'], width, label=method)
            axes[1].bar(x + i*width, sub['steps_std'], width, label=method)

        axes[0].set_xticks(x + width/2)
        axes[0].set_xticklabels(configs)
        axes[0].set_ylabel('Avg Steps')
        axes[0].set_title('Novelty Ablation — Mean Steps (A1_grid)')
        axes[0].legend()

        axes[1].set_xticks(x + width/2)
        axes[1].set_xticklabels(configs)
        axes[1].set_ylabel('Std Dev of Steps (across seeds)')
        axes[1].set_title('Novelty Ablation — Seed Variance (A1_grid)')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(OUT / "novelty_ablation.png", dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================
# PLOT 4 — HYPERPARAMETER SWEEPS (DQN vs PPO, shared sweeps)
# ============================================================

def load_dqn_sweep(group):
    path = f"out/dqn_experiments/dqn_sweep_{group}_summary.csv"
    df = safe_read_csv(path)
    if not df.empty:
        df['method'] = 'DQN'
    return df


def plot_sweep(group, configs_order, x_label, grid_filter, out_name):
    """
    group: 'hidden', 'lr', or 'discount' (DQN) — maps to PPO
           configs via configs_order naming
    configs_order: list of config names as they appear in
                    BOTH csvs (e.g. ['hidden_64','hidden_128','hidden_256'])
    """
    dqn_sweep = load_dqn_sweep(group)
    if not dqn_sweep.empty:
        dqn_sweep = dqn_sweep[dqn_sweep['grid'] == grid_filter]

    ppo_sweep = pd.DataFrame()
    if not ppo_sweeps.empty:
        ppo_sweep = ppo_sweeps[
            (ppo_sweeps['config'].isin(configs_order)) &
            (ppo_sweeps['grid'] == grid_filter)
        ].copy()

    combined = pd.concat([dqn_sweep, ppo_sweep], ignore_index=True)
    if combined.empty:
        print(f"No data for sweep group '{group}' on grid '{grid_filter}' — skipping")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['success_rate', 'avg_eval_reward', 'avg_eval_steps']
    metric_labels = ['Success Rate', 'Avg Eval Reward', 'Avg Eval Steps']

    for ax, metric, label in zip(axes, metrics, metric_labels):
        sns.barplot(data=combined, x='config', y=metric, hue='method',
                    order=configs_order, ax=ax, errorbar='sd',
                    capsize=0.08, edgecolor='black', linewidth=0.8)
        ax.set_title(f'{label} — {group} sweep ({grid_filter})')
        ax.set_xlabel(x_label)
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(OUT / out_name, dpi=300, bbox_inches='tight')
    plt.close()


# NOTE: DQN sweep configs are 'hidden_64'/'lr_1e4'/'gamma_090' etc.,
# while PPO sweep configs are 'hidden_64'/'lr_actor_1e4'/'gamma_090'.
# 'hidden_*' and 'gamma_*' names already match between DQN and PPO.
# 'lr_*' names differ (lr_1e4 vs lr_actor_1e4), handle separately.

# Hidden size sweep — restaurant grid (names match: hidden_64/128/256)
plot_sweep('hidden', ['hidden_64', 'hidden_128', 'hidden_256'],
           'Hidden Layer Size', 'restaurant_delivery_grid',
           'sweep_hidden_size.png')

# Gamma sweep — restaurant grid (names match: gamma_090/095/099)
plot_sweep('discount', ['gamma_090', 'gamma_095', 'gamma_099'],
           'Gamma', 'restaurant_delivery_grid',
           'sweep_gamma.png')


def plot_lr_sweep(grid_filter, dqn_iter_check, out_name):
    """
    Special handling for learning rate sweep — DQN uses
    config names lr_1e4/lr_3e4/lr_1e3, PPO uses
    lr_actor_1e4/lr_actor_3e4/lr_actor_1e3. We rename PPO's
    to match DQN's for a unified x-axis.
    """
    dqn_sweep = load_dqn_sweep('lr')
    if not dqn_sweep.empty:
        dqn_sweep = dqn_sweep[dqn_sweep['grid'] == grid_filter]

    ppo_sweep = pd.DataFrame()
    if not ppo_sweeps.empty:
        ppo_lr_configs = ['lr_actor_1e4', 'lr_actor_3e4', 'lr_actor_1e3']
        ppo_sweep = ppo_sweeps[
            (ppo_sweeps['config'].isin(ppo_lr_configs)) &
            (ppo_sweeps['grid'] == grid_filter)
        ].copy()
        # rename to match DQN naming for shared x-axis
        rename_map = {'lr_actor_1e4': 'lr_1e4',
                       'lr_actor_3e4': 'lr_3e4',
                       'lr_actor_1e3': 'lr_1e3'}
        ppo_sweep['config'] = ppo_sweep['config'].map(rename_map)

    combined = pd.concat([dqn_sweep, ppo_sweep], ignore_index=True)
    if combined.empty:
        print(f"No data for lr sweep on grid '{grid_filter}' — skipping")
        return

    configs_order = ['lr_1e4', 'lr_3e4', 'lr_1e3']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['success_rate', 'avg_eval_reward', 'avg_eval_steps']
    metric_labels = ['Success Rate', 'Avg Eval Reward', 'Avg Eval Steps']

    for ax, metric, label in zip(axes, metrics, metric_labels):
        sns.barplot(data=combined, x='config', y=metric, hue='method',
                    order=configs_order, ax=ax, errorbar='sd',
                    capsize=0.08, edgecolor='black', linewidth=0.8)
        ax.set_title(f'{label} — learning rate sweep ({grid_filter})')
        ax.set_xlabel('Learning Rate')
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(OUT / out_name, dpi=300, bbox_inches='tight')
    plt.close()


plot_lr_sweep('restaurant_delivery_grid', 100000, 'sweep_lr_restaurant.png')
plot_lr_sweep('A1_grid', 500000, 'sweep_lr_A1.png')


# ============================================================
# PLOT 5 — PPO-ONLY SWEEPS (entropy, gae, clip)
# ============================================================

def plot_ppo_only_sweep(configs_order, x_label, grid_filter, out_name, group_label):
    if ppo_sweeps.empty:
        print(f"No PPO sweep data available — skipping {group_label}")
        return
    df = ppo_sweeps[
        (ppo_sweeps['config'].isin(configs_order)) &
        (ppo_sweeps['grid'] == grid_filter)
    ]
    if df.empty:
        print(f"No data for PPO-only sweep '{group_label}' on '{grid_filter}'")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['success_rate', 'avg_eval_reward', 'avg_eval_steps']
    metric_labels = ['Success Rate', 'Avg Eval Reward', 'Avg Eval Steps']

    for ax, metric, label in zip(axes, metrics, metric_labels):
        sns.barplot(data=df, x='config', y=metric, order=configs_order,
                    ax=ax, color='#ff7f0e', errorbar='sd',
                    capsize=0.08, edgecolor='black', linewidth=0.8)
        ax.set_title(f'{label} — {group_label} (PPO only, {grid_filter})')
        ax.set_xlabel(x_label)
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(OUT / out_name, dpi=300, bbox_inches='tight')
    plt.close()


plot_ppo_only_sweep(['entropy_0', 'entropy_001', 'entropy_005'],
                     'Entropy Coefficient', 'restaurant_delivery_grid',
                     'sweep_entropy_ppo.png', 'entropy')
plot_ppo_only_sweep(['gae_090', 'gae_095', 'gae_099'],
                     'GAE Lambda', 'restaurant_delivery_grid',
                     'sweep_gae_ppo.png', 'gae')
plot_ppo_only_sweep(['clip_01', 'clip_02', 'clip_03'],
                     'Clip Epsilon', 'restaurant_delivery_grid',
                     'sweep_clip_ppo.png', 'clip')


print(f"\nAll plots saved to {OUT}/")