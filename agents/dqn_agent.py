"""
DQN Agent
Extends tabular Q-learing from Assignment 1 to handle cont. state space using NN

The value function is a NN that approximates Q(s,a) for all states and actions, instead of a table
State lookup is by forward pass through the network, not by indexing a table
Update rule is Bellman + SGD

The Bellman update equation is the same as A1:
    Q(s,a) <-  Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]

In DQN this becomes a gradient descent step:
    loss = (r + γ·max Q_target(s',a') - Q_policy(s,a))²
    θ ← θ - α · ∇_θ loss

The only structural additions are:
1. Replay buffer ->  breaks temporal correlation
2. Target network ->  provides stable Bellman targets
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from agents.dqn_network import DQNNetwork
from agents.dqn_replay_buffer import ReplayBuffer
from agents.dqn_config import (
    LEARNING_RATE, GAMMA, 
    EPSILON_START, EPSILON_END, EPSILON_DECAY,
    BUFFER_SIZE, BATCH_SIZE, WARMUP_STEPS, 
    TARGET_UPDATE_FREQ, HIDDEN_SIZE, 
    STATE_DIM, ACTION_DIM, MAX_EPISODE_STEPS
)

class DQNAgent(BaseAgent):
    def __init__(self, grid_shape: tuple[int, int], learning_rate=None, gamma=None, hidden_size=None):
        """
        Initialize DQN agent 
        Args: 
            grid_shape: (n_rows, n_cols) of the gridworld environment, 
            used to normalising position features and casting sensor rays
        """
        super().__init__()
        import agents.dqn_config as dqn_config

        self.grid_rows = grid_shape[0]
        self.grid_cols = grid_shape[1]

        lr = learning_rate if learning_rate is not None else dqn_config.LEARNING_RATE
        self.gamma = gamma if gamma is not None else dqn_config.GAMMA
        hidden = hidden_size if hidden_size is not None else dqn_config.HIDDEN_SIZE

        #-- Device --
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #-- Networks --
        #trained every step, approx Q(s,a) for current policy and SELECT action
        self.policy_net = DQNNetwork(dqn_config.STATE_DIM, dqn_config.ACTION_DIM, hidden).to(self.device)

        # Target Network:
        # and identical but frozen copy of policy_net, 
        # used to compute stable Bellman targets during training
        
        #without target network, both Q(s,a) and target Q(s',a') would change every step, leading to unstable training and divergence
        #by freezing the target network and updating only every TARGET_UPDATE_FREQ steps,
        #target stays stable for a while, so that training can converge towards it before it moves again
        self.target_net = DQNNetwork(dqn_config.STATE_DIM,dqn_config. ACTION_DIM, hidden).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict()) # initialize target net with same weights as policy net

        self.target_net.eval() # set target net to eval mode

        #-- Optimizer --
        #Adam adapts to the learning rate per parameter 
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.loss_fn = nn.SmoothL1Loss() # Huber loss, less sensitive to outliers than MSE, more stable training in early stages

        #-- Replay Buffer --
        self.buffer = ReplayBuffer(BUFFER_SIZE)

        #--Exploration--
        self.epsilon = EPSILON_START
        self.steps_done = 0 # total steps across episodes
        self.update_count = 0 # no of gradient updates

        #--state tracking--
        #store the current state to use as (s,a,r,s') transition when the next action is taken
        self._last_state_vec = None

    # State Building
    def _get_sensor(self, grid: np.ndarray, row: int, col: int, d_row: int, d_col:int, max_range: int=5) -> float:
        """
        Cast a sensor ray in direction (d_row, d_col) and return the distance to the nearest wall, normalized to [0,1]
        If no wall is detected within max_range, return 1.0
        0.0 means blocked in the very next cell 
        1.0 means nothing found within max_range, free in that direction

            Args:
                grid: 2D numpy array representing the gridworld
                row, col: current position of the agent
                d_row, d_col: direction vector for the sensor ray
                max_range: maximum distance the sensor can detect (in cells)
        """
        for dist in range(1, max_range+1):
            r = row + d_row * dist
            c = col + d_col * dist
            if r < 0 or r >= self.grid_rows or c < 0 or c >= self.grid_cols:
                return (dist-1) / max_range # out of bounds, treat as wall at previous cell
            if grid[r,c] in (1,2): 
                return (dist -1)/ max_range # wall or obstacle
        return 1.0 # no wall within max range
    
    def _get_obstacle_density(self, grid: np.ndarray, row: int, col: int, radius: int=2) -> float:
        """
            Fraction of cells in radius neighbourhood that are walls or obstacles, normalized to [0,1]
            returns [0,1]

            Args:
                grid: 2D numpy array representing the gridworld
                row, col: current position of the agent
                radius: how far around the agent to check for obstacles (in cells)
        """
        total, blocked = 0,0
        for dr in range(-radius, radius+1):
            for dc in range(-radius, radius+1):
                if dr == 0 and dc == 0:
                    continue # skip the agent's own cell
                r = row + dr
                c = col + dc
                if (0 <= r < self.grid_rows) and (0 <= c < self.grid_cols):
                    total += 1
                    if grid[r,c] in (1,2):
                        blocked += 1
        return blocked / total if total > 0 else 0.0
    
    def _get_target_quadrant(self, grid: np.ndarray, row: int, col:int) -> np.ndarray:
        """
        4-element binary vector [N, S, E, W] indicating
        which direction the nearest target lies.

        NOTE: This was flagged by the teacher as potentially
        too helpful (close to giving distance to target).
        kept it consistent with PPO for fair comparison
        but take a look at this!!
        """
        targets = np.argwhere(grid == 3) # find all target cells
        if len(targets) == 0:
            return np.zeros(4, dtype=np.float32) # no targets, return all zeros
        
        dists = (np.abs(targets[:,0] - row) + np.abs(targets[:,1] - col)) #  distance to each target
        tr, tc = targets[np.argmin(dists)] # nearest target row and col

        return np.array([
            1.0 if tr < row else 0.0, # target north
            1.0 if tr > row else 0.0, # target south
            1.0 if tc > col else 0.0, # target east
            1.0 if tc < col else 0.0, # target west
        ], dtype=np.float32)
        
    def _build_state(self, agent_pos, grid):
        """
        Identical to PPOAgent._build_state() so both agents
        receive the same information to make comparison fair.

        State layout:
            [0]    normalised row position
            [1]    normalised col position
            [2]    sensor north
            [3]    sensor south
            [4]    sensor east
            [5]    sensor west
            [6]    obstacle density (radius 2)

        All values in [0, 1].
        """
        row, col = agent_pos
        norm_row = row / (self.grid_rows - 1) 
        norm_col = col / (self.grid_cols - 1)

        sensor_north = self._get_sensor(grid, row, col, -1, 0)
        sensor_south = self._get_sensor(grid, row, col, 1, 0)
        sensor_east = self._get_sensor(grid, row, col, 0, 1)
        sensor_west = self._get_sensor(grid, row, col, 0, -1)

        density = self._get_obstacle_density(grid, row, col)

        return np.array([
            norm_row, norm_col, 
            sensor_north, sensor_south, sensor_east, sensor_west, 
            density,
        ], dtype=np.float32)
    
    # Action Selection
    def take_action(self, state: tuple[int, int], grid: np.ndarray=None) -> int:
        """
        Select an action using epsilon-greedy policy
        with probability epsilon, select a random action (exploration)
        with probability 1-epsilon, select the action with highest Q-value from the policy network (exploitation)
        
        agent explores a lot early and exploits its learned policy later
        Args:
            state: current position of the agent (row, col)
            grid: current gridworld state, used to build the state vector for the network
        """
        #build and cache state vector 
        state_vec = self._build_state(state, grid)
        self._last_state_vec = state_vec # store for use in learning update when we get the next state and reward

        #explore: random
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)
        
        #exploit: select action with highest Q-value from policy network
        state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device) # shape (1, state_dim)
        with torch.no_grad():
            q_values = self.policy_net(state_t) # shape (1, action_dim)
        return int(q_values.argmax(dim=1).item()) # select action with highest Q-value, convert from tensor to int

    # Learning Update
    def update(self, state: tuple[int,int], reward: float, action:int, done: bool=False, grid:np.ndarray = None):
        """
        Store the transition and perform training step.

        Called after every environment step, mirroring the PPOAgent.update() interface.

        Args:
            state  : new state (row, col) after action
            reward : reward received from environment
            action : action that was taken
            done   : True if episode ended
            grid   : current grid array
        """
        next_state_vec = self._build_state(state, grid) if grid is not None else None

        #store transition in replay buffer
        if self._last_state_vec is not None and next_state_vec is not None:
            self.buffer.push(self._last_state_vec, action, reward, next_state_vec, float(done))

        self.steps_done += 1

        #epsilon decay
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

        #warmup check - don't train until there are enough samples in the buffer
        if(self.steps_done < WARMUP_STEPS or not self.buffer.is_ready(BATCH_SIZE)):
            return

        #gradient descent update
        self._train_step()

        #update target network periodically
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _train_step(self):
        """
        One gradient update using a random mini-batch.

        This implements the DQN loss:
            y = r + γ · max_a' Q_target(s', a')  [if not done]
            y = r                                  [if done]
            loss = SmoothL1(Q_policy(s, a), y)

        The Bellman equation from A1 is exactly this, just minimising the TD error.
        """
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        states_t = torch.FloatTensor(states).to(self.device) # shape (batch_size, state_dim)
        actions_t = torch.LongTensor(actions).to(self.device) # shape (batch_size, 1)
        rewards_t = torch.FloatTensor(rewards).to(self.device) # shape (batch_size, 1)
        next_states_t = torch.FloatTensor(next_states).to(self.device) # shape (batch_size, state_dim)
        dones_t = torch.FloatTensor(dones).to(self.device) # shape (batch_size, 1)

        # Compute Q(s,a) for the actions taken
        q_current = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1) # shape (batch_size,)

        #Belmman Target
        # Compute max_a' Q_target(s', a') for the next states
        with torch.no_grad():
            q_next_max = self.target_net(next_states_t).max(dim=1)[0] # shape (batch_size,)

        q_target = rewards_t + (self.gamma * q_next_max * (1 - dones_t)) # shape (batch_size,)

        # Compute loss
        loss = self.loss_fn(q_current, q_target)

        # Gradient descent step
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        return loss.item()
    
    def save(self, filepath: str):
        """
        Save the policy network weights to a file.
        """
        torch.save({'policy_net': self.policy_net.state_dict(),
                    'target_net': self.target_net.state_dict(),
                    'epsilon': self.epsilon,
                    'steps_done': self.steps_done,
                    'update_count': self.update_count
                     }, filepath)
        print(f"DQNAgent: Saved model to {filepath}")

    def load(self, filepath: str):
        """
        Load the policy network weights from a file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])

        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.update_count = checkpoint['update_count']
        print(f"DQNAgent: Loaded model from {filepath}")


