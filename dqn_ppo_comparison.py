import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

dqn = pd.read_csv("./out/dqn_experiments/dqn_experiment_summary.csv")
ppo = pd.read_csv("./out/ppo_experiments/ppo_experiment_summary.csv")

dqn['method'] = 'DQN'
ppo['method'] = 'PPO'

both = pd.concat([dqn, ppo], ignore_index=True)
shared = both[both['variant'].isin(['full', 'no_novelty'])]

print("=== MAIN COMPARISON: full variant ===")
main = shared[shared['variant'] == 'full']
summary = main.groupby(['method', 'grid']).agg(
    success_mean=('success_rate', 'mean'),
    success_std=('success_rate', 'std'),
    reward_mean=('avg_eval_reward', 'mean'),
    reward_std=('avg_eval_reward', 'std'),
    steps_mean=('avg_eval_steps', 'mean'),
    steps_std=('avg_eval_steps', 'std'),
    episodes_mean=('episodes', 'mean'),
    episodes_std=('episodes', 'std'),
).round(3)
print(summary.to_string())

print("\n=== NOVELTY ABLATION ===")
novelty = shared.groupby(['method', 'grid', 'variant']).agg(
    success_mean=('success_rate', 'mean'),
    success_std=('success_rate', 'std'),
    reward_mean=('avg_eval_reward', 'mean'),
    reward_std=('avg_eval_reward', 'std'),
    steps_mean=('avg_eval_steps', 'mean'),
    steps_std=('avg_eval_steps', 'std'),
).round(3)
print(novelty.to_string())

print("\n=== SAMPLE EFFICIENCY (episodes to converge) ===")
print("DQN episodes (full variant):")
print(dqn[dqn['variant']=='full'][['grid','seed','episodes']].to_string())
print("\nPPO episodes (full variant):")
print(ppo[ppo['variant']=='full'][['grid','seed','episodes']].to_string())

print("\n=== KEY OBSERVATIONS ===")
dqn_a1 = dqn[(dqn['variant']=='full') & (dqn['grid']=='A1_grid')]
ppo_a1 = ppo[(ppo['variant']=='full') & (ppo['grid']=='A1_grid')]
print(f"DQN A1 episodes mean: {dqn_a1['episodes'].mean():.0f}")
print(f"PPO A1 episodes mean: {ppo_a1['episodes'].mean():.0f}")
print(f"DQN A1 steps mean: {dqn_a1['avg_eval_steps'].mean():.2f}")
print(f"PPO A1 steps mean: {ppo_a1['avg_eval_steps'].mean():.2f}")

print(f"\nDQN updates (full A1): {dqn_a1['updates'].mean():.0f}")
print(f"PPO updates (full A1): {ppo_a1['updates'].mean():.0f}")

# novelty effect
dqn_nov = dqn[(dqn['variant']=='no_novelty') & (dqn['grid']=='A1_grid')]
ppo_nov = ppo[(ppo['variant']=='no_novelty') & (ppo['grid']=='A1_grid')]
print(f"\nNovelty effect on DQN A1 success: {dqn_a1['success_rate'].mean():.2f} -> {dqn_nov['success_rate'].mean():.2f}")
print(f"Novelty effect on PPO A1 success: {ppo_a1['success_rate'].mean():.2f} -> {ppo_nov['success_rate'].mean():.2f}")
print(f"Novelty effect on DQN A1 steps: {dqn_a1['avg_eval_steps'].mean():.2f} -> {dqn_nov['avg_eval_steps'].mean():.2f}")
print(f"Novelty effect on PPO A1 steps: {ppo_a1['avg_eval_steps'].mean():.2f} -> {ppo_nov['avg_eval_steps'].mean():.2f}")

#------------------------------
#Plotting
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(16,10))

grids = ["A1_grid", "restaurant_delivery_grid"] #rows
metrics = ["success_rate", "avg_eval_reward", "avg_eval_steps"] #columns
metric_labels = ['Success Rate', 'Avg Evaluation Reward', 'Avg Evaluation Steps']

colors = {"DQN": "#1f77b4", "PPO": "#ff7f0e"}

for row_idx, grid in enumerate(grids):
    grid_data = both[both["grid"] == grid]

    for col_idx, metric in enumerate(metrics):
        ax = axes[row_idx, col_idx]
        sns.barplot(data=grid_data, x="variant", y=metric, 
                    hue="method", ax=ax, palette=colors, errorbar="sd",
                    capsize=0.08, edgecolor="black", linewidth=0.8)
        clean_grid_name = "A1 Grid" if grid == "A1_grid" else "Restaurant Delivery"

        ax.set_title(f"{metric_labels[col_idx]} - {clean_grid_name}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Hyperparameter / Variant", fontsize=10)
        ax.set_ylabel(metric_labels[col_idx], fontsize=10)

        ax.tick_params(axis="x", rotation=15, labelsize=9)

        if row_idx == 0 and col_idx == 0:
            ax.legend(title='Agent Method', frameon=True, facecolor='white', edgecolor='none')
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

plt.tight_layout()
plt.savefig("./out/hyperparameter_comparison_matrix.png", dpi=300, bbox_inches='tight')
