"""PPO Configuration.

Hyperparameters and settings for Proximal Policy Optimization.
"""

# Learning rates
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3

# PPO specific
EPSILON_CLIP = 0.2  # Clipping parameter for PPO objective
EPOCHS_PER_UPDATE = 10  # Number of epochs for policy and value updates
BATCH_SIZE = 64  # Batch size for updates

# Trajectory collection
TRAJECTORY_LENGTH = 4096    # Collect this many steps before updating, first we had 2048 but it was too small for larger grids(A1 grid)
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # Lambda for Generalized Advantage Estimation

# Network architecture
HIDDEN_SIZE = 128  # Hidden layer size for both actor and critic
STATE_DIM = 11  # The state space  
ACTION_DIM = 4  # 4 possible actions: up, down, left, right

# Entropy coefficient

ENTROPY_COEFF = 0.01

# Training
MAX_EPISODES = 100
MAX_EPISODE_STEPS = 500
