# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import os
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


class ScenarioGenerationTests(unittest.TestCase):
  def test_visualization(self):
    param_server = ParameterServer(
      filename="data/deterministic_scenario.json")
    scenario_generation = DeterministicScenarioGeneration(num_scenarios=3,
                                                          random_seed=0,
                                                          params=param_server)

    param_server["Visualization"]["Agents"]["DrawRoute", "Draw Route of each agent", True]
    viewer = MPViewer(params=param_server,
                      x_range=[-50, 50],
                      y_range=[-50, 50],
                      follow_agent_id=101,
                      screen_dims=[500, 500],
                      use_world_bounds=False)
    env = Runtime(0.2,
                  viewer,
                  scenario_generation,
                  render=True)

    # TODO(@hart): assert agent's ids
    # TODO(@hart): make sure the reset places the agents on the same position
    # TODO(@hart): make sure the evaluators work
    # TODO(@hart): assert _current_scenario_idx and make sure the scenario alternation works
    # TODO(@hart): make sure a step moves the agents as expected (kinematic state and time)
    # TODO(@hart): visualize the lane the agent drives on (local map)
    # TODO(@hart): make sure the routing works on the given map with the given configurations
    env.reset()
    for _ in range(0, 5):
      print("Scenario {}:".format(str(env._scenario_generator._current_scenario_idx)))
      for _ in range(0, 35):
        for key, agent in env._world.agents.items():
          print(agent.id)
        env.step()
      env.reset()

if __name__ == '__main__':
  unittest.main()