"""
Run DQN experiments over grids, seeds, and ablation variants.
Mirrors run_ppo_experiments.py so that results are comparable

Examples:
    python run_dqn_experiments.py --grids grid_configs/A1_grid.npy grid_configs/restaurant_delivery_grid.npy --iter 200000
    python run_dqn_experiments.py --variants full --seeds 0 1 2
"""

from __future__ import annotations
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import csv
import random
from typing import Any
import numpy as np

import agents.dqn_config as dqn_config
from reward_functions import shaped_reward

#Ablation variants
#Each variant changes one thing from full config
#Isolates the contribution of each component 

BASE_CONFIG: dict[str, Any] = {
    "LEARNING_RATE":      dqn_config.LEARNING_RATE,
    "GAMMA":              dqn_config.GAMMA,
    "EPSILON_START":      dqn_config.EPSILON_START,
    "EPSILON_END":        dqn_config.EPSILON_END,
    "EPSILON_DECAY":      dqn_config.EPSILON_DECAY,
    "BUFFER_SIZE":        dqn_config.BUFFER_SIZE,
    "BATCH_SIZE":         dqn_config.BATCH_SIZE,
    "WARMUP_STEPS":       dqn_config.WARMUP_STEPS,
    "TARGET_UPDATE_FREQ": dqn_config.TARGET_UPDATE_FREQ,
    "HIDDEN_SIZE":        dqn_config.HIDDEN_SIZE,
    "STATE_DIM":          dqn_config.STATE_DIM,
    "ACTION_DIM":         dqn_config.ACTION_DIM,
    "NOVELTY_BONUS":      dqn_config.NOVELTY_BONUS,
    "MAX_EPISODE_STEPS":  dqn_config.MAX_EPISODE_STEPS,
}

EXPERIMENT_GROUPS = {
    "ablations": ["full", "no_novelty"],
    "hidden": ["hidden_64", "hidden_128", "hidden_256"],
    "lr": ["lr_1e4", "lr_3e4", "lr_1e3"],
    "discount": ["gamma_090", "gamma_095", "gamma_099"],
}

CONFIGS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_novelty": {"NOVELTY_BONUS": 0.0}, #to test if exploration bonus helps DQN 
    "hidden_64": {"HIDDEN_SIZE": 64},
    "hidden_128": {"HIDDEN_SIZE": 128},
    "hidden_256": {"HIDDEN_SIZE": 256},
    "lr_1e4": {"LEARNING_RATE": 1e-4},
    "lr_3e4": {"LEARNING_RATE": 3e-4},
    "lr_1e3": {"LEARNING_RATE": 1e-3},
    "gamma_090": {"GAMMA": 0.90},
    "gamma_095": {"GAMMA": 0.95},
    "gamma_099": {"GAMMA": 0.99},
    "high_sigma": {}, #high stochasticity robustness test, sigma is set at environment level but document here 
}

@dataclass
class ExperimentResult:
    config: str
    grid: str
    seed: int

    hidden_size: int
    learning_rate: float
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
    p = ArgumentParser(description="Run DQN experiment suite.")
    p.add_argument("--grids", type=Path, nargs="+",
                   default=[Path("grid_configs/A1_grid.npy")],
                   help="Grid files to train/evaluate on.")
    p.add_argument("--config_names", nargs="+", choices=sorted(CONFIGS.keys()), 
                   default=["full", "no_novelty"], 
                   help="DQN variants/ablations to run.")

    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Random seeds to run.")
    p.add_argument("--iter", type=int, default=500_000,
                   help="Training timesteps per run.")
    p.add_argument("--sigma", type=float, default=0.02,
                   help="Environment stochasticity.")
    p.add_argument("--eval_episodes", type=int, default=50,
                   help="Evaluation episodes per trained model.")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Optional fixed start position as row,col.")
    p.add_argument("--out_dir", type=Path, default=Path("out/dqn_experiments"),
                   help="Directory for checkpoints and summary CSV.")
    p.add_argument("--log_every", type=int, default=100,
                   help="Print progress every N episodes.")
    p.add_argument("--group", choices=EXPERIMENT_GROUPS.keys(), default=None)
    return p.parse_args()


def set_random_seeds(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def apply_config(config_name: str):
    # reset to base
    for key, value in BASE_CONFIG.items():
        setattr(dqn_config, key, value)
    # apply overrides
    for key, value in CONFIGS[config_name].items():
        setattr(dqn_config, key, value)

def parse_start_pos(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    row, col = raw.split(",")
    return int(row), int(col)


def evaluate_agent(agent, grid: Path, sigma: float, start_pos: tuple[int, int] | None,
                   seed: int, eval_episodes: int) -> tuple[float, float, float]:
    from world import Environment

    # disable exploration during evaluation
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0

    rewards = []
    steps = []
    successes = 0

    try:
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

            for step in range(dqn_config.MAX_EPISODE_STEPS):
                action = agent.take_action(state, grid=eval_env.grid)
                state, reward, terminated, _ = eval_env.step(action)
                total_reward += reward
                if terminated:
                    break

            rewards.append(total_reward)
            steps.append(step + 1)
            successes += int(terminated)
    finally:
        # restore epsilon (even if exception occurs)
        agent.epsilon = saved_epsilon

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
    from agents.dqn_agent import DQNAgent
    from world import Environment

    set_random_seeds(seed)
    apply_config(config_name)

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
    agent = DQNAgent(grid_shape=env.grid.shape)

    timestep = 0
    episode = 0
    episode_reward = 0.0
    episode_steps = 0
    episode_rewards: list[float] = []

    visited_positions = {state}
    current_novelty = dqn_config.NOVELTY_BONUS #to help escape local optima in early training

    while timestep < total_timesteps:
        action = agent.take_action(state, grid=env.grid)
        next_state, reward, terminated, info = env.step(action)
        episode_steps += 1

        if(current_novelty > 0 and info.get("agent_moved", False) and next_state not in visited_positions):
            reward += current_novelty
            visited_positions.add(next_state)

        truncated = episode_steps >= dqn_config.MAX_EPISODE_STEPS
        done = terminated or truncated
        agent.update(next_state, reward, action, done=done, grid=env.grid)

        timestep += 1
        episode_reward += reward

        state = next_state #update state

        if done:
            episode += 1
            episode_rewards.append(episode_reward)

            if log_every > 0 and episode % log_every == 0:
                window = episode_rewards[-min(log_every, len(episode_rewards)):]
                avg_reward = sum(window) / len(window)
                print(
                    f"  {config_name:>14} | {grid.stem:<28} | seed {seed} | "
                    f"step {timestep:>8,} | episode {episode:>5} | "
                    f"avg: {avg_reward:>7.2f} | ε: {agent.epsilon:.3f} | "
                    f"| updates: {agent.update_count}"
                )

            episode_reward = 0.0
            episode_steps = 0
            state = env.reset()
            visited_positions = {state} #reset novelty tracking

    success_rate, avg_eval_reward, avg_eval_steps = evaluate_agent(
        agent, grid, sigma, initial_pos, seed, eval_episodes
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"dqn_{config_name}_{grid.stem}_seed{seed}.pt"
    torch.save({
        "policy_net": agent.policy_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "epsilon": agent.epsilon,
        "steps_done": agent.steps_done,
        "update_count": agent.update_count,
        "episode_rewards": episode_rewards,
        "config_name": config_name,
        "grid": str(grid),
        "seed": seed,
        "sigma": sigma,
        "config": {
            "learning_rate": dqn_config.LEARNING_RATE,
            "gamma": dqn_config.GAMMA,
            "hidden_size": dqn_config.HIDDEN_SIZE,
            "novelty_bonus": dqn_config.NOVELTY_BONUS,
            "state_dim": dqn_config.STATE_DIM,
        },
    }, checkpoint)

    final_avg_50 = float(np.mean(episode_rewards[-50:])) if episode_rewards else 0.0
    return ExperimentResult(
        config=config_name,
        grid=grid.stem,
        seed=seed,
        hidden_size=dqn_config.HIDDEN_SIZE,
        learning_rate=dqn_config.LEARNING_RATE,
        gamma=dqn_config.GAMMA,
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
    configs_to_run = EXPERIMENT_GROUPS[args.group] if args.group else args.config_names
    start_pos = parse_start_pos(args.start_pos)
    all_results = []

    if args.group:
        summary_filename = f"dqn_sweep_{args.group}_summary.csv"
    else:
        summary_filename = "dqn_experiment_summary.csv"
    for grid in args.grids:
        if not grid.exists():
            print(f"Skipping missing grid: {grid}")
            continue

        for config_name in configs_to_run:
            for seed in args.seeds:
                print(
                    f"\nRunning DQN experiment: grid={grid.stem}, "
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
                append_results(args.out_dir / summary_filename,
                               [result])
                print(
                    f"Finished {config_name} / {grid.stem} / seed {seed}: "
                    f"success={result.success_rate:.2f}, "
                    f"avg_reward={result.avg_eval_reward:.2f}, "
                    f"avg_steps={result.avg_eval_steps:.1f}"
                )

    if all_results:
        print(f"\nSaved summary to {args.out_dir / summary_filename}")
    else:
        print("\nNo experiments were run.")


if __name__ == "__main__":
    main()
