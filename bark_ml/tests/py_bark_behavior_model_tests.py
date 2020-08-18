# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np
import os
import matplotlib
import time
import gym
import matplotlib.pyplot as plt


# BARK
from bark.runtime.commons.parameters import ParameterServer
from bark.core.models.dynamic import SingleTrackModel
from bark.core.world import World, MakeTestWorldHighway
from bark.runtime.viewer.matplotlib_viewer import MPViewer

# BARK-ML
from bark_ml.library_wrappers.lib_tf_agents.agents.sac_agent import BehaviorSACAgent
import bark_ml.environments.gym


class PyBarkBehaviorModelTests(unittest.TestCase):
  def test_sac_agent(self):
    params = ParameterServer()
    viewer = MPViewer(
      params=params,
      x_range=[-35, 35],
      y_range=[-35, 35],
      follow_agent_id=True)
  
    env = gym.make("highway-v0")
    sac_agent = BehaviorSACAgent(environment=env, params=params)
    sac_agent._training = False
    env.ml_behavior = sac_agent


    env.reset()
    for agent_id, agent in env._world.agents.items():
      print(agent_id, agent.behavior_model)
      print(agent_id, agent.dynamic_model)
    
    for _ in range(0, 5):
      viewer.clear()
      env._world.Step(0.2)
      viewer.drawWorld(env._world)
      plt.pause(2.)

    # agent_id = list(env._world.agents.keys())[0]
    # observed_world = env._world.Observe([agent_id])[0]
    # print(sac_agent.Plan(0.2, observed_world))


if __name__ == '__main__':
  unittest.main()