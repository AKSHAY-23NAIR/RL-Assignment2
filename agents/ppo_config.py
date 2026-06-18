"""
Hyperparameters and settings for Proximal Policy Optimization.
"""

# Learning rates
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 5e-4

# PPO specific
EPSILON_CLIP = 0.3  # Clipping parameter for PPO objective
EPOCHS_PER_UPDATE = 5  # Number of epochs for policy and value updates
BATCH_SIZE = 128  # Batch size for updates

# Trajectory collection
TRAJECTORY_LENGTH = 1024  # Collect this many steps before updating
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.90  # Lambda for GAE

# Network architecture
HIDDEN_SIZE = 128  # Hidden layer size for both actor and critic
STATE_DIM = 7
ACTION_DIM = 4  # 4 possible actions: up, down, left, right

# Exploration
ENTROPY_COEFF = 0.02 # Encourages exploration
NOVELTY_BONUS = 0.02  # Training-only reward for first visit to a cell per episode
USE_ACTION_MASKING = True

# Training
MAX_EPISODES = 100
MAX_EPISODE_STEPS = 500
