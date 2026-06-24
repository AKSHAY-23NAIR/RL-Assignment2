Reinforcement Learning - Assignment 2

This repository contains a Reinforcement Learning implementation of restaurant delivery setup using a robot. This includes a DQN baseline and a PPO agent on top of it. This README provides a comprehensive description on how to run the various experiments.

## File structure


```text
README.md
requirements.txt

agents/
  __init__.py
  base_agent.py
  dqn_agent.py
  dqn_config.py
  dqn_network.py
  dqn_replay_buffer.py
  ppo_agent.py
  ppo_config.py
  neural_network.py
  null_agent.py
  random_agent.py

grid_configs/
  A1_grid.npy
  restaurant_delivery_grid.npy
  example_grid.npy
  large_grid.npy
  small_grid.npy
  super_hard.npy

world/
  __init__.py
  environment.py
  grid.py
  gui.py
  grid_creator.py
  helpers.py
  path_visualizer.py
  static/
    style.css
  templates/
    editor.html
    grid.html

train.py
train_dqn.py
train_ppo.py
run_dqn_experiments.py
run_ppo_experiments.py
plot_learning_curve.py
ppo_plots.py
reward_functions.py

out/
  dqn_experiments/
  ppo_experiments/
  smoke_dqn/
  smoke_ppo/

results/
```

## Setup
Create a virtual environment and install the requirements 
```bash
pip install -r requirements.txt
```

## Training Runs
Command to train DQN:
```bash
python train_dqn.py grid_configs/A1_grid.npy --no_gui --iter 500000 --sigma 0.1 --random_seed 0 --eval_episodes 50
```

Command to train PPO:
```bash
python train_ppo.py grid_configs/A1_grid.npy --no_gui --iter 500000 --sigma 0.1 --random_seed 0 --eval_episodes 50
```

## Commands for DQN Experiments
For the A1 grid:
```bash
python run_dqn_experiments.py --grids grid_configs/A1_grid.npy --config_names full no_novelty --seeds 0 1 2 --iter 500000 --sigma 0.1 --eval_episodes 50
```

For our restaurant setup grid:
```bash
python run_dqn_experiments.py --grids grid_configs/restaurant_delivery_grid.npy --config_names full no_novelty --seeds 0 1 2 --iter 200000 --sigma 0.1 --eval_episodes 50
```
Some more commands for trying out different hyperparameters:
```bash
python run_dqn_experiments.py --grids grid_configs/restaurant_delivery_grid.npy --group hidden --seeds 0 1 2 --iter 100000 --sigma 0.02 --eval_episodes 50

python run_dqn_experiments.py --grids grid_configs/restaurant_delivery_grid.npy --group discount --seeds 0 1 2 --iter 100000 --sigma 0.02 --eval_episodes 50

python run_dqn_experiments.py --grids grid_configs/restaurant_delivery_grid.npy --group lr --seeds 0 1 2 --iter 100000 --sigma 0.02 --eval_episodes 50

python run_dqn_experiments.py --grids grid_configs/A1_grid.npy --group lr --seeds 0 1 2 --iter 500000 --sigma 0.02 --eval_episodes 50
```

DQN outputs are saved to:
```text
out/dqn_experiments/
```

## Commands for PPO Experiments

For the A1 grid:
```bash
python run_ppo_experiments.py --grids grid_configs/A1_grid.npy --config_names full no_action_mask no_novelty long_rollout --seeds 0 1 2 --iter 500000 --sigma 0.1 --eval_episodes 50
```

For our restaurant setup grid:
```bash
python run_ppo_experiments.py --grids grid_configs/restaurant_delivery_grid.npy --config_names full no_action_mask no_novelty long_rollout --seeds 0 1 2 --iter 200000 --sigma 0.1 --eval_episodes 50
```
Some more commands for trying out different hyperparameters:
```bash
python run_ppo_experiments.py --grids grid_configs/restaurant_delivery_grid.npy --group hidden --seeds 0 1 2 --iter 100000 --sigma 0.02 --eval_episodes 50

python run_ppo_experiments.py --grids grid_configs/restaurant_delivery_grid.npy --group discount --seeds 0 1 2 --iter 100000 --sigma 0.02 --eval_episodes 50

python run_ppo_experiments.py --grids grid_configs/restaurant_delivery_grid.npy --group lr --seeds 0 1 2 --iter 100000 --sigma 0.02 --eval_episodes 50

python run_ppo_experiments.py --group entropy --seeds 0 1 2 --iter 100000 --sigma 0.02 --eval_episodes 50

python run_ppo_experiments.py --group gae --seeds 0 1 2 --iter 100000 --sigma 0.02 --eval_episodes 50

python run_ppo_experiments.py --group clip --seeds 0 1 2 --iter 100000 --sigma 0.02 --eval_episodes 50
```
PPO outputs are saved to:
```text
out/ppo_experiments/
```

## Plots

To plot learning curves from the saved checkpoints in this project use the following command (in this case for PPO on the A1 grid, using 3 seeds):
```bash
python plot_learning_curve.py out/ppo_experiments/ppo_full_A1_grid_seed0.pt out/ppo_experiments/ppo_full_A1_grid_seed1.pt out/ppo_experiments/ppo_full_A1_grid_seed2.pt
```

## Outputs
Training and experiment scripts save checkpoints, summaries, and plots under:
```text
out/
results/
```



