# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import os
import matplotlib
matplotlib.use('PS')
import time
import numpy as np
from modules.runtime.scenario.scenario_generation.deterministic \
  import DeterministicScenarioGeneration
from modules.runtime.scenario.scenario_generation.scenario_generation \
  import ScenarioGeneration
from bark.world.goal_definition import GoalDefinition, GoalDefinitionPolygon
from bark.geometry import *
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.runtime import Runtime
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from modules.runtime.viewer.pygame_viewer import PygameViewer



class PyRuntimeTests(unittest.TestCase):
  """Tests to verify the functionality of BARK
  """
  def test_runtime(self):
    """Asserts the runtime to make sure the basic
       functionality is given by the current state of BARK.
    """
    param_server = ParameterServer(
      filename="tests/data/deterministic_scenario_test.json")
    scenario_generation = DeterministicScenarioGeneration(num_scenarios=3,
                                                          random_seed=0,
                                                          params=param_server)

    param_server["Visualization"]["Agents"]["DrawRoute",
      "Draw the local map of each agent",
      True]
    viewer = MPViewer(params=param_server,
                      use_world_bounds=True)
    env = Runtime(0.2,
                  viewer,
                  scenario_generation,
                  render=True)

    env.reset()
    agent_ids = []
    agent_states = []
    centers_of_driving_corridor= []
    for key, agent in env._world.agents.items():
      agent_ids.append(agent.id)
      agent_states.append(agent.state)
        
    for i in range(0, 5):
      print("Scenario {}:".format(
        str(env._scenario_generator._current_scenario_idx)))
      # assert scenario ids
      self.assertEqual(env._scenario_generator._current_scenario_idx, (i % 3) + 1)
      for _ in range(0, 35):
        # assert ids
        states_before = []
        for i, (key, agent) in enumerate(env._world.agents.items()):
          self.assertEqual(key, agent.id)
          self.assertEqual(agent_ids[i], agent.id)
          states_before.append(agent.state)
          # TODO(@hart): why does this not work
          print(key, agent.goal_definition.goal_shape)
        env.step()
        # assert state has been changed by the step() function
        for i, (key, agent) in enumerate(env._world.agents.items()):
          np.testing.assert_equal(np.any(np.not_equal(states_before[i],
                                         agent.state)), True)

      # check whether the reset works     
      env.reset()
      for i, (key, agent) in enumerate(env._world.agents.items()):
        self.assertEqual(key, agent.id)
        self.assertEqual(agent_ids[i], agent.id)
        np.testing.assert_array_equal(agent_states[i], agent.state)

if __name__ == '__main__':
  unittest.main()