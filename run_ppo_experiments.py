"""
Run PPO experiments over grids, seeds, hyperparameters and ablation variants.

Examples:
    python run_ppo_experiments.py --config_names full no_action_mask
    python run_ppo_experiments.py --group entropy
    python run_ppo_experiments.py --group gae
    python run_ppo_experiments.py --group hidden
    python run_ppo_experiments.py --group ablations
"""

from __future__ import annotations
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import csv
import importlib
import random
from typing import Any
import numpy as np

import agents.ppo_config as ppo_config
from reward_functions import shaped_reward

BASE_CONFIG: dict[str, Any] = {
    "LEARNING_RATE_ACTOR": ppo_config.LEARNING_RATE_ACTOR,
    "LEARNING_RATE_CRITIC": ppo_config.LEARNING_RATE_CRITIC,
    "EPSILON_CLIP": ppo_config.EPSILON_CLIP,
    "EPOCHS_PER_UPDATE": ppo_config.EPOCHS_PER_UPDATE,
    "BATCH_SIZE": ppo_config.BATCH_SIZE,
    "TRAJECTORY_LENGTH": ppo_config.TRAJECTORY_LENGTH,
    "GAMMA": ppo_config.GAMMA,
    "GAE_LAMBDA": ppo_config.GAE_LAMBDA,
    "HIDDEN_SIZE": ppo_config.HIDDEN_SIZE,
    "STATE_DIM": ppo_config.STATE_DIM,
    "ACTION_DIM": ppo_config.ACTION_DIM,
    "ENTROPY_COEFF": ppo_config.ENTROPY_COEFF,
    "NOVELTY_BONUS": ppo_config.NOVELTY_BONUS,
    "USE_ACTION_MASKING": ppo_config.USE_ACTION_MASKING,
    "MAX_EPISODE_STEPS": ppo_config.MAX_EPISODE_STEPS,
}

EXPERIMENT_GROUPS = {
    "ablations": ["full", "no_action_mask", "no_novelty", "long_rollout",],
    "entropy": ["entropy_0", "entropy_001", "entropy_005",],
    "gae": ["gae_090", "gae_095", "gae_099",],
    "clip": ["clip_01", "clip_02", "clip_03",],
    "hidden": ["hidden_64", "hidden_128", "hidden_256", ],
    "lr": ["lr_actor_1e4", "lr_actor_3e4", "lr_actor_1e3"],
    "discount": ["gamma_090", "gamma_095", "gamma_099"]
}

CONFIGS: dict[str, dict[str, Any]] = {
    # Ablations
    "full": {},
    "no_action_mask": {"USE_ACTION_MASKING": False,},
    "no_novelty": {"NOVELTY_BONUS": 0.0,},
    "long_rollout": {"TRAJECTORY_LENGTH": 4096,},

    # Entropy sweep
    "entropy_0": {"ENTROPY_COEFF": 0.0,},
    "entropy_001": {"ENTROPY_COEFF": 0.01,},
    "entropy_005": {"ENTROPY_COEFF": 0.05,},

    # GAE sweep
    "gae_090": {"GAE_LAMBDA": 0.90,},
    "gae_095": {"GAE_LAMBDA": 0.95,},
    "gae_099": {"GAE_LAMBDA": 0.99,},

    # Clip sweep
    "clip_01": {"EPSILON_CLIP": 0.1,},
    "clip_02": {"EPSILON_CLIP": 0.2,},
    "clip_03": {"EPSILON_CLIP": 0.3,},

    # Hidden-size sweep
    "hidden_64": {"HIDDEN_SIZE": 64,},
    "hidden_128": {"HIDDEN_SIZE": 128,},
    "hidden_256": {"HIDDEN_SIZE": 256,},

    # Learning rate sweep
    "lr_actor_1e4": {"LEARNING_RATE_ACTOR": 1e-4,},
    "lr_actor_3e4": {"LEARNING_RATE_ACTOR": 3e-4,},
    "lr_actor_1e3": {"LEARNING_RATE_ACTOR": 1e-3,},

    # Discount factor sweep
    "gamma_090": {"GAMMA": 0.90,},
    "gamma_095": {"GAMMA": 0.95,},
    "gamma_099": {"GAMMA": 0.99,},
}

@dataclass
class ExperimentResult:
    config: str
    grid: str
    seed: int

    entropy_coeff: float
    gae_lambda: float
    epsilon_clip: float
    trajectory_length: int
    hidden_size: int
    learning_rate_actor: float
    gamma: float

    timesteps: int
    updates: int
    episodes: int
    final_avg_50: float
    success_rate: float
    avg_eval_reward: float
    avg_eval_steps: float
    checkpoint: str

def parse_args():
    p = ArgumentParser(description="Run PPO experiment suite.")
    p.add_argument("--grids", type=Path, nargs="+",
                   default=[Path("grid_configs/A1_grid.npy"), Path("grid_configs/restaurant_delivery_grid.npy"),],
                   help="Grid files to train/evaluate on.")
    p.add_argument("--config_names", nargs="+", choices=sorted(CONFIGS.keys()),
                   default=["full", "no_action_mask", "no_novelty",
                            "long_rollout"],
                   help="PPO configurations to run.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Random seeds to run.")
    p.add_argument("--iter", type=int, default=500_000,
                   help="Training timesteps per run.")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Environment stochasticity.")
    p.add_argument("--eval_episodes", type=int, default=50,
                   help="Evaluation episodes per trained model.")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Optional fixed start position as row,col.")
    p.add_argument("--out_dir", type=Path, default=Path("out/ppo_experiments"),
                   help="Directory for checkpoints and summary CSV.")
    p.add_argument("--log_every", type=int, default=100,
                   help="Print progress every N episodes.")
    p.add_argument("--group", choices=EXPERIMENT_GROUPS.keys(), default=None,)
    return p.parse_args()

def set_random_seeds(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def apply_config(config_name: str):
    # restore defaults
    for key, value in BASE_CONFIG.items():
        setattr(ppo_config, key, value)

    # apply overrides
    for key, value in CONFIGS[config_name].items():
        setattr(ppo_config, key, value)

    import agents.ppo_agent as ppo_agent_module
    return importlib.reload(ppo_agent_module).PPOAgent

def parse_start_pos(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    row, col = raw.split(",")
    return int(row), int(col)

def evaluate_agent(agent, grid: Path, sigma: float, start_pos: tuple[int, int],
                   seed: int, eval_episodes: int) -> tuple[float, float, float]:
    from world import Environment

    rewards = []
    steps = []
    successes = 0

    for episode_idx in range(eval_episodes):
        eval_env = Environment(
            grid,
            no_gui=True,
            sigma=sigma,
            agent_start_pos=start_pos,
            reward_fn=shaped_reward,
            random_seed=seed + 10_000 + episode_idx,
        )
        state = eval_env.reset()
        total_reward = 0.0
        terminated = False

        for step in range(ppo_config.MAX_EPISODE_STEPS):
            action = agent.take_action(
                state,
                grid=eval_env.grid,
                store=False,
                deterministic=True,
            )
            state, reward, terminated, _ = eval_env.step(action)
            total_reward += reward
            if terminated:
                break

        rewards.append(total_reward)
        steps.append(step + 1)
        successes += int(terminated)

    return (
        successes / eval_episodes,
        float(np.mean(rewards)),
        float(np.mean(steps)),
    )

def train_one(grid: Path, config_name: str, seed: int, total_timesteps: int,
              sigma: float, eval_episodes: int,
              start_pos: tuple[int, int] | None, out_dir: Path,
              log_every: int) -> ExperimentResult:
    import torch
    from world import Environment

    set_random_seeds(seed)
    PPOAgent = apply_config(config_name)

    env = Environment(
        grid,
        no_gui=True,
        sigma=sigma,
        agent_start_pos=start_pos,
        reward_fn=shaped_reward,
        random_seed=seed,
    )
    state = env.reset()
    initial_pos = env.agent_pos
    agent = PPOAgent(grid_shape=env.grid.shape)

    timestep = 0
    episode = 0
    episode_reward = 0.0
    episode_steps = 0
    episode_rewards: list[float] = []
    last_done = False
    visited_positions = {state}

    while timestep < total_timesteps:
        action = agent.take_action(state, grid=env.grid)
        state, reward, terminated, info = env.step(action)
        episode_steps += 1

        if info["agent_moved"] and state not in visited_positions:
            reward += ppo_config.NOVELTY_BONUS
            visited_positions.add(state)

        truncated = episode_steps >= ppo_config.MAX_EPISODE_STEPS
        done = terminated or truncated
        agent.update(state, reward, action, done=done, grid=env.grid)

        timestep += 1
        episode_reward += reward
        last_done = done

        if done:
            episode += 1
            episode_rewards.append(episode_reward)

            if log_every > 0 and episode % log_every == 0:
                window = episode_rewards[-min(log_every, len(episode_rewards)):]
                avg_reward = sum(window) / len(window)
                print(
                    f"  {config_name:>14} | {grid.stem:<28} | seed {seed} | "
                    f"step {timestep:>8,} | episode {episode:>5} | "
                    f"avg: {avg_reward:>7.2f} | updates: {agent.update_count}"
                )

            episode_reward = 0.0
            episode_steps = 0
            state = env.reset()
            visited_positions = {state}

    if len(agent.states) > 0:
        next_value = 0.0 if last_done else agent._state_value(state, env.grid)
        agent._ppo_update(next_value=next_value)

    success_rate, avg_eval_reward, avg_eval_steps = evaluate_agent(
        agent, grid, sigma, initial_pos, seed, eval_episodes
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"ppo_{config_name}_{grid.stem}_seed{seed}.pt"
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "update_count": agent.update_count,
        "episode_rewards": episode_rewards,
        "config_name": config_name,
        "grid": str(grid),
        "seed": seed,
        "sigma": sigma,
        "config": {
            "learning_rate_actor": ppo_config.LEARNING_RATE_ACTOR,
            "learning_rate_critic": ppo_config.LEARNING_RATE_CRITIC,
            "epsilon_clip": ppo_config.EPSILON_CLIP,
            "epochs_per_update": ppo_config.EPOCHS_PER_UPDATE,
            "batch_size": ppo_config.BATCH_SIZE,
            "trajectory_length": ppo_config.TRAJECTORY_LENGTH,
            "entropy_coeff": ppo_config.ENTROPY_COEFF,
            "novelty_bonus": ppo_config.NOVELTY_BONUS,
            "use_action_masking": ppo_config.USE_ACTION_MASKING,
        },
    }, checkpoint)

    final_avg_50 = float(np.mean(episode_rewards[-50:])) if episode_rewards else 0.0
    return ExperimentResult(
    config=config_name,
    grid=grid.stem,
    seed=seed,

    entropy_coeff=ppo_config.ENTROPY_COEFF,
    gae_lambda=ppo_config.GAE_LAMBDA,
    epsilon_clip=ppo_config.EPSILON_CLIP,
    trajectory_length=ppo_config.TRAJECTORY_LENGTH,
    hidden_size=ppo_config.HIDDEN_SIZE,
    learning_rate_actor=ppo_config.LEARNING_RATE_ACTOR,
    gamma=ppo_config.GAMMA,

    timesteps=total_timesteps,
    updates=agent.update_count,
    episodes=len(episode_rewards),
    final_avg_50=final_avg_50,
    success_rate=success_rate,
    avg_eval_reward=avg_eval_reward,
    avg_eval_steps=avg_eval_steps,
    checkpoint=str(checkpoint),
)

def append_results(csv_path: Path, results: list[ExperimentResult]):
    fieldnames = list(ExperimentResult.__dataclass_fields__.keys())
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

def main():
    args = parse_args()
    if args.group:
        configs_to_run = EXPERIMENT_GROUPS[args.group]
    else:
        configs_to_run = args.config_names
    start_pos = parse_start_pos(args.start_pos)
    all_results = []

    for grid in args.grids:
        if not grid.exists():
            print(f"Skipping missing grid: {grid}")
            continue

        for config_name in configs_to_run:
            for seed in args.seeds:
                print(
                    f"\nRunning PPO experiment: grid={grid}, "
                    f"config={config_name}, seed={seed}"
                )
                result = train_one(
                    grid=grid,
                    config_name=config_name,
                    seed=seed,
                    total_timesteps=args.iter,
                    sigma=args.sigma,
                    eval_episodes=args.eval_episodes,
                    start_pos=start_pos,
                    out_dir=args.out_dir,
                    log_every=args.log_every,
                )
                all_results.append(result)
                append_results(args.out_dir / "ppo_experiment_summary.csv",
                               [result])
                print(
                    f"Finished {config_name} / {grid.stem} / seed {seed}: "
                    f"success={result.success_rate:.2f}, "
                    f"avg_reward={result.avg_eval_reward:.2f}, "
                    f"avg_steps={result.avg_eval_steps:.1f}"
                )

    if all_results:
        print(f"\nSaved summary to {args.out_dir / 'ppo_experiment_summary.csv'}")
    else:
        print("\nNo experiments were run.")

if __name__ == "__main__":
    main()
