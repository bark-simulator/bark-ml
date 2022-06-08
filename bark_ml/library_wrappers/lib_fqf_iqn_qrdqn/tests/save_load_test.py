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
from gym import spaces
import numpy as np

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import FQFAgent

observation_length = 5 
num_actions = 4

class TestObserver:
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

class DummyEnv:
  def __init__(self):
    self._observer = TestObserver()
    self._ml_behavior =  TestActionWrapper()

  def reset(self):
    pass

  def step(self):
    pass

class BaseAgentTests(unittest.TestCase):
  def test_agents(self):
    params = ParameterServer()

    fqf_agent = FQFAgent(env = DummyEnv(), agent_save_dir="./save_dir", params=params)

    fqf_agent.save(checkpoint_type="best")
    fqf_agent.save(checkpoint_type="last")

    loaded_agent = FQFAgent(env = DummyEnv(), agent_save_dir="./save_dir", checkpoint_load="best")

    self.assertEqual(loaded_agent.ml_behavior.action_space.n, fqf_agent.ml_behavior.action_space.n)
    self.assertEqual(loaded_agent.ent_coef, fqf_agent.ent_coef)


if __name__ == '__main__':
  unittest.main()
