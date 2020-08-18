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
from bark_ml.library_wrappers.lib_tf_agents.agents.ppo_agent import BehaviorPPOAgent
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, ContinuousMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorGraphSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner
from bark_ml.observers.graph_observer import GraphObserver
import bark_ml.environments.gym


class PyBarkBehaviorModelTests(unittest.TestCase):
  def test_sac_agent(self):
    params = ParameterServer()
    env = gym.make("highway-v0")
    sac_agent = BehaviorSACAgent(environment=env, params=params)
    ppo_agent = BehaviorPPOAgent(environment=env, params=params)

    behaviors = [ppo_agent, sac_agent]

    for ml_agent in behaviors:
      env.ml_behavior = ml_agent
      env.reset()
      for agent_id, agent in env._world.agents.items():
        print(agent_id, agent.behavior_model)
        print(agent_id, agent.dynamic_model)
      for _ in range(0, 5):
        env._world.Step(0.2)
  
  def test_sac_graph_agent(self):
    params = ParameterServer()
    bp = ContinuousMergingBlueprint(params,
                                number_of_senarios=2500,
                                random_seed=0)
    observer = GraphObserver(params=params)
    env = SingleAgentRuntime(
      blueprint=bp,
      observer=observer,
      render=False)
    sac_agent = BehaviorGraphSACAgent(environment=env,
                                      observer=observer,
                                      params=params)
    env.ml_behavior = sac_agent
    env.reset()
    for agent_id, agent in env._world.agents.items():
      print(agent_id, agent.behavior_model)
      print(agent_id, agent.dynamic_model)
    
    for _ in range(0, 5):
      env._world.Step(0.2)

    # agent_id = list(env._world.agents.keys())[0]
    # observed_world = env._world.Observe([agent_id])[0]
    # print(sac_agent.Plan(0.2, observed_world))
    # sac_graph_agent = BehaviorGraphSACAgent(environment=env, params=params)


if __name__ == '__main__':
  unittest.main()