
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

In the highway scenario, the ego agent's goal is a `StateLimitGoal` on the left lane that is reached once the states are in a pre-defined range.
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

In the merging scenario, the ego agent's goal is a `StateLimitGoal` on the left lane that is reached once the states are in a pre-defined range.
A positive reward (`+1`) is given for reaching the goal and a negative reward for having a collision or leaving the drivable area (`-1`).

The highway scenario can use discrete or continuous actions:
* `merging-v0`: Continuous merging environment
* `merging-v1`: Discrete merging environment


<p align="center">
<img src="https://github.com/bark-simulator/bark-ml/raw/master/docs/images/bark-ml.gif" alt="BARK-ML Highway" />
</p>



### Unprotected Left Turn

```
env = gym.make("intersection-v0")
```

In this scenario, the ego agent's goal is a `StateLimitGoal` on the right lane that it has to achieve.
Positive reward (`+1`) is given for reaching the goal and negative reward for having a collision or leaving the drivable area (`-1`).

The highway scenario can use discrete or continuous actions:
* `intersection-v0`: Continuous intersection environment
* `intersection-v1`: Discrete intersection environment


<!-- <p align="center">
<img src="https://github.com/bark-simulator/bark-ml/raw/master/docs/images/bark_ml_highway.gif" alt="BARK-ML Highway" />
</p> -->

## Getting Started

A complete example of using the [OpenAi-Gym](https://github.com/openai/gym) inteface can be found [here](https://github.com/bark-simulator/bark-ml/blob/master/examples/continuous_env.py):
```
import gym
import numpy as np
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

The graph neural network actor-critic architecture proposed in the paper "Graph Neural Networks and Reinforcement Learning for Behavior Generation in Semantic Environments" can be visualized using

```
bazel run //experiments:experiment_runner -- --exp_json=/ABSOLUTE_PATH/bark-ml/experiments/configs/phd/01_hyperparams/dnns/merging_large_network.json
```

and trained using

```
bazel run //experiments:experiment_runner -- --exp_json=/ABSOLUTE_PATH/bark-ml/experiments/configs/phd/01_hyperparams/dnns/merging_large_network.json --mode=train
```


If you use BARK-ML and build upon the graph neural network architecture please cite the following [paper](https://arxiv.org/abs/2006.12576):

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
