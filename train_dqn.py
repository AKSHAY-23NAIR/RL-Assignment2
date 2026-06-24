"""
Train DQN Agent.

DQN training script for the grid navigation task.
Mirrors the interface of train.py but uses a timestep-based
training loop suited to DQN's experience buffer design.

Usage:
    python train_dqn.py grid_configs/example_grid.npy --no_gui --iter 500000
"""

from argparse import ArgumentParser
from pathlib import Path
from agents.dqn_config import MAX_EPISODE_STEPS
import torch

from reward_functions import shaped_reward
from world import Environment
from agents.dqn_agent import DQNAgent


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer (DQN).")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster.")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--iter", type=int, default=500_000,
                   help="Total number of timesteps to train for.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Agent start position as row,col (e.g. 2,3). "
                        "If not set, the GUI lets you click to place it. "
                        "In no_gui mode, defaults to random placement.")
    p.add_argument("--eval_episodes", type=int, default=10,
                   help="Number of evaluation episodes after training.")
    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, total_timesteps: int, fps: int,
         sigma: float, random_seed: int, start_pos: tuple[int, int] | None,
         eval_episodes: int):
    """Main loop of the program."""

    for grid in grid_paths:

        print(f"\nTraining on grid: {grid}")
        print(f"Total timesteps: {total_timesteps:,}")

        # Set up the environment
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
                          agent_start_pos=start_pos,
                          reward_fn=shaped_reward,
                          random_seed=random_seed)

        # Reset before constructing the agent so env.grid.shape is available
        state = env.reset()
        initial_pos = env.agent_pos

        # Initialize DQN agent with actual grid dimensions
        agent = DQNAgent(grid_shape=env.grid.shape)

        # --- Training loop ---
        # DQN is trained over total timesteps, not episodes.
        # The experience buffer accumulates steps and triggers a DQN update
        # every TRAJECTORY_LENGTH steps (defined in dqn_config.py).

        timestep = 0
        episode = 0
        episode_reward = 0.0
        episode_steps = 0
        episode_rewards = []  # store per-episode returns for analysis
        last_done = False
        visited_positions = {state}

        while timestep < total_timesteps:

            action = agent.take_action(state, grid=env.grid)
            next_state, reward, terminated, info = env.step(action)
            episode_steps += 1

            if info["agent_moved"] and next_state not in visited_positions:
                # reward += NOVELTY_BONUS
                visited_positions.add(next_state)
            
            truncated = episode_steps >= MAX_EPISODE_STEPS
            done = terminated or truncated
            
            agent.update(state, reward, action, done=done, grid=env.grid)

            timestep += 1
            episode_reward += reward
            last_done = done
            state = next_state

            if done:
                episode += 1
                episode_rewards.append(episode_reward)

                # Print progress every 10 episodes
                if episode % 10 == 0:
                    avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
                    print(f"  Timestep {timestep:>8,} | Episode {episode:>5} | "
                          f"Return: {episode_reward:>8.1f} | "
                          f"Avg (last 10): {avg_reward:>8.1f} | "
                          f"DQN updates: {agent.update_count}")

                episode_reward = 0.0
                episode_steps = 0
                state = env.reset()
                visited_positions = {state}

        print(f"\nTraining complete. Total episodes: {episode}, "
              f"DQN updates: {agent.update_count}")

        # --- Save model ---
        out_dir = Path("out")
        out_dir.mkdir(exist_ok=True)
        save_path = out_dir / f"dqn_{grid.stem}_seed{random_seed}.pt"
        torch.save({
            'policy_net': agent.policy_net.state_dict(),
            'target_net': agent.target_net.state_dict(),
            'update_count': agent.update_count,
            'episode_rewards': episode_rewards,
        }, save_path)
        print(f"Model saved to {save_path}")

        # # --- Evaluation ---
        # # Evaluate the trained agent without learning (no update() calls).
        # Environment.evaluate_agent(grid, agent, total_timesteps, sigma,
        #                            agent_start_pos=initial_pos,
        #                            random_seed=random_seed)
        
        # --- Evaluation ---
        print("\nEvaluating agent...")
        eval_rewards = []
        eval_steps = []
        eval_successes = 0

        for eval_idx in range(eval_episodes):
            eval_env = Environment(grid, no_gui=True, sigma=sigma,
                                agent_start_pos=initial_pos,
                                reward_fn=shaped_reward,
                                random_seed=random_seed + eval_idx)
            eval_state = eval_env.reset()
            total_reward = 0.0
            terminated = False

            for step in range(500):
                action = agent.take_action(eval_state, grid=eval_env.grid)
                eval_state, reward, terminated, info = eval_env.step(action)
                total_reward += reward
                if terminated:
                    break

            eval_rewards.append(total_reward)
            eval_steps.append(step + 1)
            eval_successes += int(terminated)

        print(f"Evaluation complete over {eval_episodes} episodes.")
        print(f"  Success rate: {eval_successes}/{eval_episodes}")
        print(f"  Avg reward: {sum(eval_rewards) / len(eval_rewards):.1f}")
        print(f"  Avg steps: {sum(eval_steps) / len(eval_steps):.1f}")


if __name__ == '__main__':
    args = parse_args()
    start_pos = None
    if args.start_pos is not None:
        parts = args.start_pos.split(',')
        start_pos = (int(parts[0]), int(parts[1]))
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma,
         args.random_seed, start_pos, args.eval_episodes)
