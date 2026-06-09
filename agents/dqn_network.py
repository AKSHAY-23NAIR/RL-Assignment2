"""
Q-Network for DQN Agent
A simple MLP that approximates the action-value function Q(s,a) for the DQN algorithm.

In Assignment 1 we stored Q-values in a table:
Q_table[state] = [q_up, q_down, q_left, q_right]

In Assignment 2 we use a neural network to approximate this function:
Q_network(state) = [q_up, q_down, q_left, q_right]

The network generalizes, can make Q-value estimates for states it has never visited before,
by interpolating from similar states it has seen during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    """
    NN that approximates Q(s,a) for DQN.
    Input: state vector (STATE_DIM floats)
    Output: one Q-value for each action (ACTION_DIM floats)

    Architecture: state -> FC(128) -> ReLU -> FC(128) -> ReLU -> FC(actions)

    Using ReLU activation to aviod vanishing gradients and allow for non-linear function approximation.
    Matches the PPO actor/critic architecture to make comparison fair
    """
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int=128):
        """
        Args:
            state_dim: no. of input features (state representation size)
            n_actions: no. of output actions (action space size), discrete
            hidden_dim: size of each hidden layer (default 128, matches PPO)"""
       
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            state: tensor of shape (batch_size, state_dim)
        Returns:
            q_values: tensor of shape (batch_size, n_actions) with Q-values for each action
            q_values[i][j] = Q(state[i], action_j)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x) # output layer, no activation (Q-values can be any real number)
        return q_values