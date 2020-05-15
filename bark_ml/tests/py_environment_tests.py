# Copyright (c) 2019 Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np
import os
import matplotlib
# matplotlib.use('PS')
import time

from modules.runtime.commons.parameters import ParameterServer
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, DiscreteHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime


class PyEnvironmentTests(unittest.TestCase):
  def test_env_cont_rl(self):
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params)
    env = SingleAgentRuntime(blueprint=bp, render=True)
    env.reset()
    for _ in range(0, 20):
      print(env.step(np.array([1., 0.1])))

  def test_env_discrete_rl(self):
    params = ParameterServer()
    bp = DiscreteHighwayBlueprint(params)
    env = SingleAgentRuntime(blueprint=bp, render=True)
    env.reset()
    for _ in range(0, 20):
      print(env.step(0))

if __name__ == '__main__':
  unittest.main()