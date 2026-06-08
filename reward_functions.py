"""
Reward functions for the grid navigation task.
Shared between PPO, DQN, and any other agents.
"""

def default_reward(grid, agent_pos) -> float:
    """Default reward: -1 per step, -5 for walls, +10 for target."""
    match grid[agent_pos]:
        case 0:
            return -1
        case 1 | 2:
            return -5
        case 3:
            return 10
        case _:
            raise ValueError(f"Unexpected grid value at {agent_pos}")


def shaped_reward(grid, agent_pos) -> float:
    """Scaled reward for PPO training."""
    match grid[agent_pos]:
        case 0:
            return -0.02
        case 1 | 2:
            return -1
        case 3:
            return 20
        case _:
            raise ValueError(f"Unexpected grid value at {agent_pos}")
