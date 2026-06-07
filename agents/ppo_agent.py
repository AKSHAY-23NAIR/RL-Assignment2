"""Proximal Policy Optimization Agent.

Implementation of PPO for the grid navigation task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from agents.base_agent import BaseAgent
from agents.neural_network import ActorNetwork, CriticNetwork
from agents.ppo_config import (
    LEARNING_RATE_ACTOR, LEARNING_RATE_CRITIC, EPSILON_CLIP,
    EPOCHS_PER_UPDATE, BATCH_SIZE, TRAJECTORY_LENGTH,
    GAMMA, GAE_LAMBDA, HIDDEN_SIZE, STATE_DIM, ACTION_DIM,
    ENTROPY_COEFF
)


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization Agent."""
    
    def __init__(self, grid_shape: tuple[int, int]):
        """Initialize the PPO agent."""
        super().__init__()
        self.grid_rows = grid_shape[0]
        self.grid_cols = grid_shape[1]
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = ActorNetwork(STATE_DIM, ACTION_DIM, HIDDEN_SIZE).to(self.device)
        self.critic = CriticNetwork(STATE_DIM, HIDDEN_SIZE).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), 
                                         lr=LEARNING_RATE_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), 
                                          lr=LEARNING_RATE_CRITIC)
        
        # Trajectory storage
        self.states = deque(maxlen=TRAJECTORY_LENGTH)
        self.actions = deque(maxlen=TRAJECTORY_LENGTH)
        self.rewards = deque(maxlen=TRAJECTORY_LENGTH)
        self.values = deque(maxlen=TRAJECTORY_LENGTH)
        self.log_probs = deque(maxlen=TRAJECTORY_LENGTH)
        self.dones = deque(maxlen=TRAJECTORY_LENGTH)
        
        self.trajectory_step = 0
        self.update_count = 0

    def _get_sensor(self, grid: np.ndarray, row: int, col: int,
                    d_row: int, d_col: int, max_range: int = 5) -> float:
        """Cast a ray in direction (d_row, d_col) and return normalised distance
        to the nearest wall or obstacle (cell value 1 or 2).

        Returns a value in [0, 1] where 0 means blocked immediately and 1 means
        nothing found within max_range steps.

        Args:
            grid:      Current grid array.
            row, col:  Agent position.
            d_row, d_col: Unit direction vector.
            max_range: Maximum number of cells to look ahead.
        """
        for dist in range(1, max_range + 1):
            r = row + d_row * dist
            c = col + d_col * dist
            if r < 0 or r >= self.grid_rows or c < 0 or c >= self.grid_cols:
                # Hit boundary — treat as wall
                return (dist - 1) / max_range
            if grid[r, c] in (1, 2):  # wall or obstacle
                return (dist - 1) / max_range
        return 1.0  # nothing found within range

    def _get_obstacle_density(self, grid: np.ndarray, row: int, col: int,
                               radius: int = 2) -> float:
        """Compute fraction of cells within a square neighbourhood that are
        walls or obstacles.

        Args:
            grid:         Current grid array.
            row, col:     Agent position.
            radius:       Chebyshev radius of the neighbourhood.

        Returns:
            Density in [0, 1].
        """
        total = 0
        blocked = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue  # skip agent cell
                r, c = row + dr, col + dc
                if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                    total += 1
                    if grid[r, c] in (1, 2):
                        blocked += 1
        return blocked / total if total > 0 else 0.0

    def _get_target_quadrant(self, grid: np.ndarray, row: int, col: int) -> np.ndarray:
        """Return a 4-element binary vector indicating which cardinal quadrant(s)
        the nearest target lies in relative to the agent.

        Vector layout: [north, south, east, west]
        A quadrant flag is 1 if the target is in that direction, 0 otherwise.
        If no target exists (all collected), returns [0, 0, 0, 0].

        Args:
            grid:      Current grid array.
            row, col:  Agent position.
        """
        target_positions = np.argwhere(grid == 3)
        if len(target_positions) == 0:
            return np.zeros(4, dtype=np.float32)

        # Find nearest target by Manhattan distance
        distances = np.abs(target_positions[:, 0] - row) + \
                    np.abs(target_positions[:, 1] - col)
        nearest = target_positions[np.argmin(distances)]
        t_row, t_col = nearest

        north = 1.0 if t_row < row else 0.0  # target is above (smaller row)
        south = 1.0 if t_row > row else 0.0
        east  = 1.0 if t_col > col else 0.0
        west  = 1.0 if t_col < col else 0.0

        return np.array([north, south, east, west], dtype=np.float32)

    def _build_state(self, agent_pos: tuple[int, int], grid: np.ndarray) -> np.ndarray:
        """Build the full state vector from agent position and grid.

        State layout (10 values):
            [0]  normalised row position
            [1]  normalised col position
            [2]  sensor facing north  (up,    d_row=-1)
            [3]  sensor facing south  (down,  d_row=+1)
            [4]  sensor facing east   (right, d_col=+1)
            [5]  sensor facing west   (left,  d_col=-1)
            [6]  local obstacle density (radius-2 neighbourhood)
            [7]  target quadrant north
            [8]  target quadrant south
            [9]  target quadrant east
            [10] target quadrant west

        All values are in [0, 1].

        Args:
            agent_pos: (row, col) tuple.
            grid:      Current grid numpy array.

        Returns:
            State vector as float32 numpy array of length STATE_DIM.
        """
        row, col = agent_pos

        # Global position
        norm_row = row / (self.grid_rows - 1)
        norm_col = col / (self.grid_cols - 1)

        # Directional sensors: north, south, east, west
        sensor_north = self._get_sensor(grid, row, col, -1,  0)
        sensor_south = self._get_sensor(grid, row, col, +1,  0)
        sensor_east  = self._get_sensor(grid, row, col,  0, +1)
        sensor_west  = self._get_sensor(grid, row, col,  0, -1)

        # Local obstacle density
        density = self._get_obstacle_density(grid, row, col, radius=2)

        # Target quadrant
        quadrant = self._get_target_quadrant(grid, row, col)

        state = np.array([
            norm_row, norm_col,
            sensor_north, sensor_south, sensor_east, sensor_west,
            density,
            quadrant[0], quadrant[1], quadrant[2], quadrant[3],
        ], dtype=np.float32)

        return state

    def _state_value(self, agent_pos: tuple[int, int], grid: np.ndarray) -> float:
        """Return the critic value for a given state."""
        state_vec = self._build_state(agent_pos, grid)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(state_tensor)
        return value.detach().cpu().numpy().flatten()[0]
    
    def take_action(self, state: tuple[int, int], grid: np.ndarray = None) -> int:
        """Select action using the policy network.
        
        Args:
            state: Current agent position (row, col).
            grid:  Current grid array (required for rich state).
            
        Returns:
            Action (0-3).
        """
        state_vec = self._build_state(state, grid)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits = self.actor(state_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            value = self.critic(state_tensor)
        
        # Store trajectory data
        self.states.append(state_vec)
        self.actions.append(action.item())
        self.log_probs.append(log_prob.detach().cpu().numpy())
        self.values.append(value.detach().cpu().numpy().flatten()[0])
        
        return action.item()
    
    def update(self, state: tuple[int, int], reward: float, action: int,
               done: bool = False, grid: np.ndarray = None):
        """Update agent with experience.
        
        Args:
            state:  New state (row, col) after action.
            reward: Reward from environment.
            action: Action that was taken.
            done:   Whether the episode is done.
            grid:   Current grid array (required for bootstrap value).
        """
        self.rewards.append(reward)
        self.dones.append(done)
        self.trajectory_step += 1
        
        # When trajectory is full, perform PPO update with proper bootstrap.
        if self.trajectory_step >= TRAJECTORY_LENGTH:
            next_value = 0.0 if done else self._state_value(state, grid)
            self._ppo_update(next_value=next_value)
            self.trajectory_step = 0

    def _compute_advantages(self, next_value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Compute advantages using Generalized Advantage Estimation."""
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        returns = np.zeros(len(self.rewards), dtype=np.float32)
        
        gae = 0.0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
            
            delta = self.rewards[t] + GAMMA * next_val * (1 - float(self.dones[t])) - self.values[t]
            gae = delta + GAMMA * GAE_LAMBDA * gae
            advantages[t] = gae
            returns[t] = gae + self.values[t]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _ppo_update(self, next_value: float = 0.0):
        """Perform PPO update on collected trajectory."""
        if len(self.states) == 0:
            return
        
        states = torch.FloatTensor(np.array(list(self.states))).to(self.device)
        actions = torch.LongTensor(np.array(list(self.actions))).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(list(self.log_probs))).to(self.device)

        advantages, returns = self._compute_advantages(next_value)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        for epoch in range(EPOCHS_PER_UPDATE):
            indices = np.random.permutation(len(self.states))
            for i in range(0, len(self.states), BATCH_SIZE):
                batch_indices = indices[i:i+BATCH_SIZE]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Actor update
                action_logits = self.actor(batch_states)
                action_probs = torch.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                new_log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - EPSILON_CLIP, 
                                   1 + EPSILON_CLIP) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEFF * entropy
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Critic update
                predicted_returns = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(predicted_returns, batch_returns)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
        
        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
        self.update_count += 1

    def load(self, path: str):
        """Load a saved checkpoint.
        
        Args:
            path: Path to the .pt checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.update_count = checkpoint['update_count']