"""
DQN Configuration file (Hyperparameter for DQN)

These mirror the PPO hyperparameters to allow a fair comparison.
"""

# Network Architecture
# Same state representation for a fair comparison with PPO
HIDDEN_SIZE = 128  #hidden layer width 
STATE_DIM = 11   # matching match_build_state()
ACTION_DIM = 4  # 0:up, 1:down, 2:left, 3:right

# Training Parameters
LEARNING_RATE = 1e-3 #Adam learning rate 
                     # PPO uses 3e-4 for actor, DQN can be slightly higher because updates are more frequent but smaller 

GAMMA = 0.99

# Epsilon-greedy strategy for action selection
EPSILON_START = 1.0 # start fully random to fully explore
EPSILON_END = 0.05 # min exploraiton rate 5%
EPSILON_DECAY = 0.995 

BUFFER_SIZE = 50_000 #max transactions stored in replay buffer
                    # old ones are discarded when full
BATCH_SIZE = 64 # mini-batch size for training
WARMUP_STEPS = 1_000 # number of steps to populate the replay buffer before training starts (diversity)

TARGET_UPDATE_FREQ = 500 # how often to update the target network (in steps)

MAX_EPISODE_STEPS = 500 # max steps per episode to prevent infinite loops