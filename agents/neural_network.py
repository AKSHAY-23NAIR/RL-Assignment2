"""Neural Network Components for PPO.

Actor (policy) and Critic (value) networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """Policy network that outputs action probabilities."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        """Initialize the actor network.
        
        Args:
            state_dim: Dimension of the state space.
            action_dim: Number of possible actions.
            hidden_size: Size of hidden layers.
        """
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            state: Input state tensor.
            
        Returns:
            Action logits.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits


class CriticNetwork(nn.Module):
    """Value network that outputs state values."""
    
    def __init__(self, state_dim: int, hidden_size: int = 128):
        """Initialize the critic network.
        
        Args:
            state_dim: Dimension of the state space.
            hidden_size: Size of hidden layers.
        """
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            state: Input state tensor.
            
        Returns:
            State value.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value