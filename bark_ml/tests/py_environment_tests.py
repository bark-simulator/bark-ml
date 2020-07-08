# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np
import os
import gym
import matplotlib
# matplotlib.use('PS')
import time

from bark.runtime.commons.parameters import ParameterServer
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  DiscreteHighwayBlueprint, ContinuousMergingBlueprint, DiscreteMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
import bark_ml.environments.gym

class PyEnvironmentTests(unittest.TestCase):
  def test_envs_cont_rl(self):
    params = ParameterServer()
    cont_blueprints = []
    cont_blueprints.append(ContinuousHighwayBlueprint(params))
    cont_blueprints.append(ContinuousMergingBlueprint(params))

    for bp in cont_blueprints:
      env = SingleAgentRuntime(blueprint=bp, render=False)
      env.reset()
      for _ in range(0, 10):
        action = np.random.uniform(low=-0.1, high=0.1, size=(2, ))
        observed_next_state, reward, done, info = env.step(action)
        # print(f"Reward: {reward}, Done: {done}")

  def test_envs_discrete_rl(self):
    params = ParameterServer()
    discrete_blueprints = []
    discrete_blueprints.append(DiscreteHighwayBlueprint(params))
    discrete_blueprints.append(DiscreteMergingBlueprint(params))

    for bp in discrete_blueprints:
      env = SingleAgentRuntime(blueprint=bp, render=False)
      env.reset()
      for _ in range(0, 10):
        action = np.random.randint(low=0, high=3)
        observed_next_state, reward, done, info = env.step(action)
        # print(f"Reward: {reward}, Done: {done}")

  def test_gym_wrapping(self):
    # highway-v0: continuous
    # highway-v1: discrete
    # merging-v0: continuous
    # merging-v1: discrete
    # are registered here: import bark_ml.environments.gym

    cont_envs = [gym.make("highway-v0"), gym.make("merging-v0")]
    for env in cont_envs:
      env.reset()
      for _ in range(0, 10):
        action = np.random.uniform(low=-0.1, high=0.1, size=(2, ))
        observed_next_state, reward, done, info = env.step(action)
        print(f"Observed state: {observed_next_state}, Reward: {reward}, Done: {done}")

    cont_envs = [gym.make("highway-v1"), gym.make("merging-v1")]
    for env in cont_envs:
      env.reset()
      for _ in range(0, 10):
        action = np.random.randint(low=0, high=3)
        observed_next_state, reward, done, info = env.step(action)
        print(f"Observed state: {observed_next_state}, Reward: {reward}, Done: {done}")

if __name__ == '__main__':
  unittest.main()