
<p align="center">
<img src="https://github.com/bark-simulator/bark-ml/raw/master/docs/images/bark_ml_logo.png" width="65%" alt="BARK-ML" />
</p>

# BARK-ML - Machine Learning for Autonomous Driving

![CI Build](https://github.com/bark-simulator/bark-ml/workflows/CI/badge.svg)


Discrete and continuous environments for autonomous driving —
 ranging from highway, over merging, to intersection scenarios.


## Gym Environments

Install the BARK-ML package using `pip install bark-ml`.



### Highway Scenario

```
env = gym.make("highway-v0")
```

In the highway scenario, the ego agent's goal is a `StateLimitGoal` on the left lane that is reached once the states are in a pre-defined range (velocity range of `[12.5m/s, 17.5m/s]`, polygonal area on the left lane, and theta range of `[-0.1rad, 0.1rad]`).
A positive reward (`+1`) is given for reaching the goal and a negative reward for having a collision or leaving the drivable area (`-1`).

The highway scenario can use discrete or continuous actions:
* `highway-v0`: Continuous highway environment
* `highway-v1`: Discrete highway environment


<p align="center">
<img src="https://github.com/bark-simulator/bark-ml/raw/master/docs/images/bark_ml_highway.gif" alt="BARK-ML Highway" />
</p>



### Merging Scenario

```
env = gym.make("merging-v0")
```

In the merging scenario, the ego agent's goal is a `StateLimitGoal` on the left lane that is reached once the states are in a pre-defined range (velocity range of `[5m/s, 15m/s]`, polygonal area on the left lane, and theta range of `[-0.15rad, 0.15rad]`).
A positive reward (`+1`) is given for reaching the goal and a negative reward for having a collision or leaving the drivable area (`-1`).

The merging scenario can use discrete or continuous actions:
* `merging-v0`: Continuous merging environment
* `merging-v1`: Discrete merging environment


<p align="center">
<img src="https://github.com/bark-simulator/bark-ml/raw/master/docs/images/bark-ml.gif" alt="BARK-ML Highway" />
</p>



### Unprotected Left Turn

```
env = gym.make("intersection-v0")
```

In the unprotected left turn scenario, the ego agent's goal is a `StateLimitGoal` placed on the top-left lane.
A positive reward (`+1`) is given for reaching the goal lane and a negative reward for having a collision or leaving the drivable area (`-1`).

The unprotected left turn scenario can use discrete or continuous actions:
* `intersection-v0`: Continuous intersection environment
* `intersection-v1`: Discrete intersection environment


<!-- <p align="center">
<img src="https://github.com/bark-simulator/bark-ml/raw/master/docs/images/bark_ml_highway.gif" alt="BARK-ML Highway" />
</p> -->


## Getting Started

A complete example using the [OpenAi-Gym](https://github.com/openai/gym) inteface can be found [here](https://github.com/bark-simulator/bark-ml/blob/master/examples/continuous_env.py):
```
import gym
import numpy as np
# registers bark-ml environments
import bark_ml.environments.gym

env = gym.make("merging-v0")

initial_state = env.reset()
done = False
while done is False:
  action = np.random.uniform(low=np.array([-0.5, -0.1]), high=np.array([0.5, 0.1]), size=(2, ))
  observed_state, reward, done, info = env.step(action)
  print(f"Observed state: {observed_state}, Action: {action}, Reward: {reward}, Done: {done}")

```

## Graph Neural Network Actor-Critic

To run the graph neural network actor-critic architecture proposed in the paper "Graph Neural Networks and Reinforcement Learning for Behavior Generation in Semantic Environments", you first need to clone the repository `git clone https://github.com/bark-simulator/bark-ml`.

Next, install and enter the virtual environment using:

```
bash utils/install.sh
source utils/dev_into.sh
```

Once you are in the virtual environment, you can run the graph soft-actor-critic using the [Bazel](https://bazel.build/) build tool: 

```
bazel run //experiments:experiment_runner -- --exp_json=/ABSOLUTE_PATH/bark-ml/experiments/configs/phd/01_hyperparams/gnns/merging_large_embedding.json
```

The graph soft-actor agent can be trained using the following command:

```
bazel run //experiments:experiment_runner -- --exp_json=/ABSOLUTE_PATH/bark-ml/experiments/configs/phd/01_hyperparams/gnns/merging_large_embedding.json --mode=train
```

Make sure to replace `ABSOLUTE_PATH` with your BARK-ML base directory.  
If you use BARK-ML and build upon the graph neural network architecture, please cite the following [paper](https://arxiv.org/abs/2006.12576):


```
@inproceedings{Hart2020,
    title = {Graph Neural Networks and Reinforcement Learning for Behavior Generation in Semantic Environments},
    author = {Patrick Hart and Alois Knoll},
    booktitle = {2020 IEEE Intelligent Vehicles Symposium (IV)},
    url = {https://arxiv.org/abs/2006.12576},
    year = {2020}
}
```


## License

BARK-ML specific code is distributed under MIT License.
