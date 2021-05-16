
# BARK-ML - Machine Learning for Autonomous Driving

[![CI Build](https://github.com/bark-simulator/bark-ml/workflows/CI/badge.svg)](https://github.com/bark-simulator/bark-ml/actions)
[![Github Contributors](https://img.shields.io/github/contributors/bark-simulator/bark-ml)](https://github.com/bark-simulator/bark-ml/graphs/contributors)
[![Downloads](https://img.shields.io/pypi/dm/bark-ml)](https://pypi.org/project/bark-ml/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/6e2c0ba1291249de9c54cb73c697b62c)](https://www.codacy.com/gh/bark-simulator/bark-ml/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=bark-simulator/bark-ml&amp;utm_campaign=Badge_Grade)
[![Environments](https://img.shields.io/badge/Environments-3-informational)](#gym-environments)

<p align="center">
<img src="https://github.com/bark-simulator/bark-ml/raw/master/docs/images/bark_ml_logo.png" width="65%" alt="BARK-ML" />
</p>

## [Try it on Google Colab! ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jA5QPEHadvIU6GsCy4cFdAv3giS7QvDQ?usp=sharing)

Discrete and continuous environments for autonomous driving —
 ranging from highway, over merging, to intersection scenarios.

The BARK-ML package can be installed using `pip install bark-ml`.

## Gym Environments

### Highway Scenario

```python
env = gym.make("highway-v0")
```

The highway scenario is a curved road with four lanes.
A potential-based reward signal for the desired velocity is used and the episode is terminal once the maximum number of steps (`200`) has been reached or a collision (`reward -= 1`) has occured or the drivable area (`reward -= 1`) has been left.
The other vehicles in the scenario are controlled by the intelligent driver model (IDM).

The highway scenario can use discrete or continuous actions:
*   `highway-v0`: Continuous highway environment
*   `highway-v1`: Discrete highway environment

<p align="center">
<img src="https://github.com/bark-simulator/bark-ml/raw/master/docs/images/bark_ml_highway_round.gif" alt="BARK-ML Highway Scenario" /><br/>
<em>The highway-v0 environment.</em>
</p>

### Merging Scenario

```python
env = gym.make("merging-v0")
```

In the merging scenario, the ego agent's goal is a `StateLimitsGoal` on the left lane that is reached once its states are in a pre-defined range (velocity range of `[5m/s, 15m/s]`, polygonal area on the left lane, and theta range of `[-0.15rad, 0.15rad]`).
A positive reward (`+1`) is given for reaching the goal and a negative reward for having a collision or leaving the drivable area (`-1`).
The other vehicles on the left lane are controlled by the intelligent driver model (IDM) and the ones on the right by the MOBIL model.

The merging scenario can use discrete or continuous actions:
*   `merging-v0`: Continuous merging environment
*   `merging-v1`: Discrete merging environment

<p align="center">
<img src="https://github.com/bark-simulator/bark-ml/raw/master/docs/images/bark_ml_merging.gif" alt="BARK-ML Merging Scenario" /><br/>
<em>The merging-v0 environment.</em>
</p>

### Unprotected Left Turn

```python
env = gym.make("intersection-v0")
```

In the unprotected left turn scenario, the ego agent's goal is a `StateLimitsGoal` placed on the top-left lane.
A positive reward (`+1`) is given for reaching the goal lane and a negative reward for having a collision or leaving the drivable area (`-1`).

The unprotected left turn scenario can use discrete or continuous actions:
*   `intersection-v0`: Continuous intersection environment
*   `intersection-v1`: Discrete intersection environment

## Getting Started

A complete example using the [OpenAi-Gym](https://github.com/openai/gym) interface can be found [here](https://github.com/bark-simulator/bark-ml/blob/master/examples/continuous_env.py):
```python
import gym
import numpy as np
# registers bark-ml environments
import bark_ml.environments.gym  # NOLINT

env = gym.make("merging-v0")

initial_state = env.reset()
done = False
while done is False:
  # action = np.array([0., 0.]) # acceleration and steering-rate
  action = np.random.uniform(low=np.array([-0.5, -0.1]), high=np.array([0.5, 0.1]), size=(2, ))
  observed_state, reward, done, info = env.step(action)
  print(f"Observed state: {observed_state}, Action: {action}, Reward: {reward}, Done: {done}")

```

## Building From Source

BARK-ML can alternatively also be built from source.
Clone the repository using `git clone https://github.com/bark-simulator/bark-ml`, install the virtual python environment and activate it afterwards using:

```bash
bash utils/install.sh
source utils/dev_into.sh
```

Now - once in the virtual python environment - you can build any of the libraries or execute binaries within BARK-ML using [Bazel](https://bazel.build/).
To run the getting started example from above, use the following command: `bazel run //examples:continuous_env`.

## Documentation

Read the [documentation online](https://bark-simulator.github.io/tutorials/bark_ml_getting_started/).

## Graph Neural Network Soft Actor-Critic

You can visualize  (`--mode=visualize`) or train (`--mode=train`) the graph neural network soft actor-critic architecture proposed in the paper "[Graph Neural Networks and Reinforcement Learning for Behavior Generation in Semantic Environments](https://arxiv.org/abs/2006.12576)" using:

```bash
bazel run //experiments:experiment_runner -- --exp_json=/ABSOLUTE_PATH/bark-ml/experiments/configs/phd/01_hyperparams/gnns/merging_large_embedding.json --mode=visualize
```

Make sure to replace `ABSOLUTE_PATH` with your BARK-ML base directory!

<p align="center">
<img src="https://github.com/bark-simulator/bark-ml/raw/master/docs/images/graph_neural_network.gif" alt="Actor-Critic Graph Neural Network Architecture" />
</p>

The merging scenario above is visualized using [BARKSCAPE](https://github.com/bark-simulator/barkscape/).
If you are interested in using a 3D-visualization, have a look at [this](https://github.com/bark-simulator/barkscape/blob/master/examples/bark_ml_runner_example.py)  example.

If your work builds upon the graph neural network architecture, please cite the following [paper](https://arxiv.org/abs/2006.12576):

```bibtex
@inproceedings{Hart2020,
    title = {Graph Neural Networks and Reinforcement Learning for Behavior Generation in Semantic Environments},
    author = {Patrick Hart and Alois Knoll},
    booktitle = {2020 IEEE Intelligent Vehicles Symposium (IV)},
    url = {https://ieeexplore.ieee.org/document/9304738},
    year = {2020}
}
```

## Publications

*   [Graph Neural Networks and Reinforcement Learning for Behavior Generation in Semantic Environments](https://arxiv.org/abs/2006.12576) (IV 2020)
*   [BARK: Open Behavior Benchmarking in Multi-Agent Environments](https://arxiv.org/abs/2003.02604) (IROS 2020)
*   [Counterfactual Policy Evaluation for Decision-Making in Autonomous Driving](https://arxiv.org/abs/2003.11919) (IROS 2020,  PLC Workshop)

## License

BARK-ML code is distributed under MIT License.
