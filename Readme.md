# Gym Environments and Agents for Autonomous Driving

[![CI Build](https://github.com/bark-simulator/bark-ml/workflows/CI/badge.svg)](https://github.com/bark-simulator/bark-ml/actions)
[![Github Contributors](https://img.shields.io/github/contributors/bark-simulator/bark-ml)](https://github.com/bark-simulator/bark-ml/graphs/contributors)
[![Downloads](https://img.shields.io/pypi/dm/bark-ml)](https://pypi.org/project/bark-ml/)
[![Python Versions](https://img.shields.io/pypi/pyversions/bark-ml)](https://pypi.org/project/bark-ml/)
[![Package Versions](https://img.shields.io/pypi/v/bark-ml)](https://pypi.org/project/bark-ml/)
[![Package Versions](https://img.shields.io/github/license/bark-simulator/bark-ml)](LICENSE)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/d1353db9e995441f87ab1a098226cf5f)](https://www.codacy.com/gh/bark-simulator/bark-ml/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=bark-simulator/bark-ml&amp;utm_campaign=Badge_Grade)
[![Environments](https://img.shields.io/badge/Environments-3-informational)](#gym-environments)

<p align="center">
<img src="https://github.com/bark-simulator/bark-ml/raw/master/docs/images/bark_ml_logo.png" width="65%" alt="BARK-ML" />
</p>

## [Try it on Google Colab! ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jA5QPEHadvIU6GsCy4cFdAv3giS7QvDQ?usp=sharing)

BARK-ML offers various OpenAI-Gym environments and reinforcement learning agents for autonomous driving.

Install BARK-ML using `pip install bark-ml`.

## Gym Environments

### Highway Scenario

```python
env = gym.make("highway-v0")
```

The highway scenario is a curved road with four lanes where all vehicles are being controlled by the intelligent driver model (IDM).
For more details have a look [here](https://bark-simulator.github.io/tutorials/bark_ml_environments/#highway).

Available environments:
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
In the merging scenario, the ego vehicle is placed on the right and its goal is placed on the left lane.
All other vehicles are controlled by the MOBIL model.
For more details have a look [here](https://bark-simulator.github.io/tutorials/bark_ml_environments/#merging).

Available environments:
*   `merging-v0`: Continuous merging environment
*   `merging-v1`: Discrete merging environment

<p align="center">
<img src="https://github.com/bark-simulator/bark-ml/raw/master/docs/images/bark_ml_merging.gif" alt="BARK-ML Merging Scenario" /><br/>
<em>The merging-v0 environment.</em>
</p>

### Intersection / Unprotected Left Turn

```python
env = gym.make("intersection-v0")
```

In the intersection scenario, the ego vehicle starts on the bottom-right lane and its goal is set on the top-left lane (unprotected left turn).
For more details have a look [here](https://bark-simulator.github.io/tutorials/bark_ml_environments/#intersection).

Available environments:
*   `intersection-v0`: Continuous intersection environment
*   `intersection-v1`: Discrete intersection environment

<p align="center">
<img src="https://github.com/bark-simulator/bark-ml/raw/master/docs/images/bark_ml_intersection.png" alt="BARK-ML Intersection Scenario" /><br/>
<em>The intersection-v0 environment.</em>
</p>

## Getting Started

An example using the [OpenAi-Gym](https://github.com/openai/gym) interface can be found [here](https://github.com/bark-simulator/bark-ml/blob/master/examples/continuous_env.py):
```python
import gym
import numpy as np
# registers bark-ml environments
import bark_ml.environments.gym  # pylint: disable=unused-import

env = gym.make("merging-v0")

initial_state = env.reset()
done = False
while done is False:
  action = np.array([0., 0.]) # acceleration and steering-rate
  observed_state, reward, done, info = env.step(action)
  print(f"Observed state: {observed_state}, "
        f"Action: {action}, Reward: {reward}, Done: {done}")

```

## Building From Source

Clone the repository using `git clone https://github.com/bark-simulator/bark-ml`, install the virtual python environment and activate it afterwards using:

```bash
bash utils/install.sh
source utils/dev_into.sh
```

Now - once in the virtual python environment - you can build any of the libraries or execute binaries within BARK-ML using [Bazel](https://bazel.build/).
To run the getting started example from above, use the following command: `bazel run //examples:continuous_env`.

## Documentation

Read the [documentation online](https://bark-simulator.github.io/tutorials/bark_ml_getting_started/).

## Publications

*   [Graph Neural Networks and Reinforcement Learning for Behavior Generation in Semantic Environments](https://arxiv.org/abs/2006.12576) (IV 2020)
*   [BARK: Open Behavior Benchmarking in Multi-Agent Environments](https://arxiv.org/abs/2003.02604) (IROS 2020)
*   [Counterfactual Policy Evaluation for Decision-Making in Autonomous Driving](https://arxiv.org/abs/2003.11919) (IROS 2020,  PLC Workshop)

## License

BARK-ML code is distributed under MIT License.
