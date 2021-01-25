# Copyright (c) 2020 fortiss GmbH
#
# Authors: Julian Bernhard, Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

try:
    import debug_settings
except:
    pass

import unittest
import numpy as np
import os
import gym
import matplotlib
import time
from gym import spaces

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.environments.blueprints import \
  DiscreteHighwayBlueprint, DiscreteMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
import bark_ml.environments.gym
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import ImitationAgent

observation_length = 5 
num_actions = 4

class TestObserver():
  @property
  def observation_space(self):
    # TODO(@hart): use from spaces.py
    return spaces.Box(
      low=np.zeros(observation_length),
      high = np.ones(observation_length))

class TestActionWrapper():
  @property
  def action_space(self):
    return spaces.Discrete(num_actions)


def action_values_at_state(state):
  envelope_costs = []
  collision_costs = []
  return_values = []
  for idx in range(1, num_actions+1):
    envelope_costs.append(state[0]* 1.0/idx + state[1]* 1.0/idx)
    collision_costs.append(state[1]* 1.0/idx*0.001 + state[3]* 1.0/idx*0.001)
    collision_costs.append(state[3]* 1.0/idx*0.2 + state[4]* 1.0/idx*0.1)
  action_values = []
  action_values.extend(envelope_costs)
  action_values.extend(collision_costs)
  action_values.extend(return_values)
  return [state, action_values]

def create_data(num):
  observations = np.random.rand(num, observation_length)
  action_values_data = np.apply_along_axis(action_values_at_state, 1, observations)
  return action_values_data

class EvaluationTests(unittest.TestCase):
  # make sure the agent works
  def test_agent_wrapping(self):
    params = ParameterServer()
    env = gym.make("highway-v1", params=params)
    env._observer = TestObserver()
    env._ml_behavior = TestActionWrapper()
    params["ML"]["BaseAgent"]["MaxEpisodeSteps"] = 2
    params["ML"]["BaseAgent"]["NumEvalEpisodes"] = 2
    data = create_data(10000)
    demo_train = data[0:7000]
    demo_test = data[7001:]
    agent = ImitationAgent(agent_save_dir="./save_dir", demonstrations_train=demo_train,
                           demonstrations_test=demo_test,
                           env=env, params=params)
    agent.run(1000000)


if __name__ == '__main__':
  unittest.main()
