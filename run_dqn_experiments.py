"""
Run DQN experiments over grids, seeds, and ablation variants.

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



@dataclass
class ExperimentResult:
    variant: str
    grid: str
    seed: int
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
    p.add_argument("--out_dir", type=Path, default=Path("out/dqn_experiments"),
                   help="Directory for checkpoints and summary CSV.")
    p.add_argument("--log_every", type=int, default=100,
                   help="Print progress every N episodes.")
    return p.parse_args()


def set_random_seeds(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_start_pos(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    row, col = raw.split(",")
    return int(row), int(col)


def evaluate_agent(agent, grid: Path, sigma: float, start_pos: tuple[int, int] | None,
                   seed: int, eval_episodes: int) -> tuple[float, float, float]:
    from world import Environment

    current_epsilon = getattr(agent, "epsilon", None)
    if current_epsilon is not None:
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
        if current_epsilon is not None:
            agent.epsilon = current_epsilon

    return (
        successes / eval_episodes,
        float(np.mean(rewards)),
        float(np.mean(steps)),
    )


def train_one(grid: Path, variant: str, seed: int, total_timesteps: int,
              sigma: float, eval_episodes: int,
              start_pos: tuple[int, int] | None, out_dir: Path,
              log_every: int) -> ExperimentResult:
    import torch
    from agents.dqn_agent import DQNAgent
    from world import Environment

    set_random_seeds(seed)

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

    while timestep < total_timesteps:
        action = agent.take_action(state, grid=env.grid)
        next_state, reward, terminated, info = env.step(action)
        episode_steps += 1

        truncated = episode_steps >= dqn_config.MAX_EPISODE_STEPS
        done = terminated or truncated
        agent.update(next_state, reward, action, done=done, grid=env.grid)

        timestep += 1
        episode_reward += reward

        if done:
            episode += 1
            episode_rewards.append(episode_reward)

            if log_every > 0 and episode % log_every == 0:
                window = episode_rewards[-min(log_every, len(episode_rewards)):]
                avg_reward = sum(window) / len(window)
                print(
                    f"  {variant:>14} | {grid.stem:<28} | seed {seed} | "
                    f"step {timestep:>8,} | episode {episode:>5} | "
                    f"avg: {avg_reward:>7.2f} | updates: {agent.update_count}"
                )

            episode_reward = 0.0
            episode_steps = 0
            state = env.reset()
        else:
            state = next_state

    success_rate, avg_eval_reward, avg_eval_steps = evaluate_agent(
        agent, grid, sigma, initial_pos, seed, eval_episodes
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"dqn_{grid.stem}_seed{seed}.pt"
    torch.save({
        "policy_net": agent.policy_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "epsilon": agent.epsilon,
        "steps_done": agent.steps_done,
        "update_count": agent.update_count,
        "episode_rewards": episode_rewards,
        "variant": variant,
        "grid": str(grid),
        "seed": seed,
        "sigma": sigma,
        "config": {
            "learning_rate": dqn_config.LEARNING_RATE,
            "gamma": dqn_config.GAMMA,
            "epsilon_start": dqn_config.EPSILON_START,
            "epsilon_end": dqn_config.EPSILON_END,
            "epsilon_decay": dqn_config.EPSILON_DECAY,
            "buffer_size": dqn_config.BUFFER_SIZE,
            "batch_size": dqn_config.BATCH_SIZE,
            "warmup_steps": dqn_config.WARMUP_STEPS,
            "target_update_freq": dqn_config.TARGET_UPDATE_FREQ,
            "hidden_size": dqn_config.HIDDEN_SIZE,
            "state_dim": dqn_config.STATE_DIM,
            "action_dim": dqn_config.ACTION_DIM,
            "max_episode_steps": dqn_config.MAX_EPISODE_STEPS,
        },
    }, checkpoint)

    final_avg_50 = float(np.mean(episode_rewards[-50:])) if episode_rewards else 0.0
    return ExperimentResult(
        variant=variant,
        grid=grid.stem,
        seed=seed,
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
    start_pos = parse_start_pos(args.start_pos)
    all_results = []

    for grid in args.grids:
        if not grid.exists():
            print(f"Skipping missing grid: {grid}")
            continue

        for variant in args.variants:
            for seed in args.seeds:
                print(
                    f"\nRunning DQN experiment: grid={grid}, "
                    f"variant={variant}, seed={seed}"
                )
                result = train_one(
                    grid=grid,
                    variant=variant,
                    seed=seed,
                    total_timesteps=args.iter,
                    sigma=args.sigma,
                    eval_episodes=args.eval_episodes,
                    start_pos=start_pos,
                    out_dir=args.out_dir,
                    log_every=args.log_every,
                )
                all_results.append(result)
                append_results(args.out_dir / "dqn_experiment_summary.csv",
                               [result])
                print(
                    f"Finished {variant} / {grid.stem} / seed {seed}: "
                    f"success={result.success_rate:.2f}, "
                    f"avg_reward={result.avg_eval_reward:.2f}, "
                    f"avg_steps={result.avg_eval_steps:.1f}"
                )

    if all_results:
        print(f"\nSaved summary to {args.out_dir / 'dqn_experiment_summary.csv'}")
    else:
        print("\nNo experiments were run.")


if __name__ == "__main__":
    main()
