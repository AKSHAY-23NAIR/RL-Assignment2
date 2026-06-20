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

To plot learning curves from the saved checkpoints in this project use the following command (for PPO in A1 grid here in this case over 3 seeds):
```bash
python plot_learning_curve.py out/ppo_experiments/ppo_full_A1_grid_seed0.pt out/ppo_experiments/ppo_full_A1_grid_seed1.pt out/ppo_experiments/ppo_full_A1_grid_seed2.pt
```

## Outputs
Training and experiment scripts save checkpoints, summaries, and plots under:
```text
out/
results/
```



Welcome to Data Intelligence Challenge-2AMC15!
This is the repository containing the challenge environment code.

## Quickstart

1. Create a virtual environment for this course with Python >= 3.10. Using conda, you can do: `conda create -n dic2025 python=3.11`. Use `conda activate dic2025` to activate it `conda deactivate` to deactivate it.
2. Clone this repository into the local directory you prefer `git clone https://github.com/RL-In-Practice/2AMC15-2026.git`.
3. Install the required packages `pip install -r requirements.txt`. Now, you are ready to use the simulation environment! :partying_face:	
4. Run `$ python train.py grid_configs/example_grid.npy` to start training!

`train.py` is just an example training script. Inside this file, initialize the agent you want to train and evaluate. Feel free to modify it as necessary. Its usage is:

```bash
usage: train.py [-h] [--no_gui] [--sigma SIGMA] [--fps FPS] [--iter ITER]
                [--random_seed RANDOM_SEED] [--start_pos START_POS]
                GRID [GRID ...]

DIC Reinforcement Learning Trainer.

positional arguments:
  GRID                  Paths to the grid file to use. There can be more than
                        one.
options:
  -h, --help                 show this help message and exit
  --no_gui                   Disables rendering to train faster (boolean)
  --sigma SIGMA              Sigma value for the stochasticity of the environment. (float, default=0.1, should be in [0, 1])
  --fps FPS                  Frames per second to render at. Only used if no_gui is not set. (int, default=30)
  --iter ITER                Number of iterations to go through. Should be integer. (int, default=1000)
  --random_seed RANDOM_SEED  Random seed value for the environment. (int, default=0)
  --start_pos START_POS      Agent start position as col,row (e.g. 2,3). If not set, the GUI lets you click to place it. In no_gui mode, defaults to random placement.
```

## Code guide

The code is made up of 2 modules: 

1. `agent`
2. `world`

### The `agent` module

The `agent` module contains the `BaseAgent` class as well as some benchmark agents you may want to test against.

The `BaseAgent` is an abstract class and all RL agents for DIC must inherit from/implement it.
If you know/understand class inheritence, skip the following section:

#### `BaseAgent` as an abstract class
Here you can find an explanation about abstract classes [Geeks for Geeks](https://www.geeksforgeeks.org/abstract-classes-in-python/).

Think of this like how all models in PyTorch start like 

```python
class NewModel(nn.Module):
    def __init__(self):
        super().__init__()
    ...
```

In this case, `NewModel` inherits from `nn.Module`, which gives it the ability to do back propagation, store parameters, etc. without you having to manually code that every time.
It also ensures that every class that inherits from `nn.Module` contains _at least_ the `forward()` method, which allows a forward pass to actually happen.

In the case of your RL agent, inheriting from `BaseAgent` guarantees that your agent implements `update()` and `take_action()`.
This ensures that no matter what RL agent you make and however you code it, the environment and training code can always interact with it in the same way.
Check out the benchmark agents to see examples.

### The `world` module

The world module contains:
1. `grid_creator.py`
2. `environment.py`
3. `grid.py`
4. `gui.py`

#### Grid creator
Run this file to create new grids.

```bash
$ python grid_creator.py
```

This will start up a web server where you create new grids, of different sizes with various elements arrangements.
To view the grid creator itself, go to `127.0.0.1:5000`.
All levels will be saved to the `grid_configs/` directory.


#### The Environment

The `Environment` is very important because it contains everything we hold dear, including ourselves [^1].
It is also the name of the class which our RL agent will act within. Most of the action happens in there.

The main interaction with `Environment` is through the methods:

- `Environment()` to initialize the environment
- `reset()` to reset the environment
- `step()` to actually take a time step with the environment
- `Environment().evaluate_agent()` to evaluate the agent after training.

[^1]: In case you missed it, this sentence is a joke. Please do not write all your code in the `Environment` class.

#### The Grid

The `Grid` class is the the actual representation of the world on which the agent moves. It is a 2D Numpy array.

#### The GUI

The Graphical User Interface provides a way for you to actually see what the RL agent is doing.
While performant and written using PyGame, it is still about 1300x slower than not running a GUI.
Because of this, we recommend using it only while testing/debugging and not while training.
