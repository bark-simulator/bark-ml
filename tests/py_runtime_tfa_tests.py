# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import os
import matplotlib
matplotlib.use('PS')
from modules.runtime.scenario.scenario_generation.uniform_vehicle_distribution \
  import UniformVehicleDistribution
from modules.runtime.scenario.scenario_generation.deterministic \
  import DeterministicScenarioGeneration
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer


from src.rl_runtime import RuntimeRL
from src.observers.nearest_state_observer import ClosestAgentsObserver
from src.wrappers.dynamic_model import DynamicModel
from src.wrappers.tfa_wrapper import TFAWrapper
from src.evaluators.goal_reached import GoalReached

from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils

class PyRuntimeTFARLTests(unittest.TestCase):
  @staticmethod
  def test_tfa_runtime():
    params = ParameterServer(
      filename="tests/data/deterministic_scenario_test.json")
    scenario_generation = DeterministicScenarioGeneration(num_scenarios=3,
                                                          random_seed=0,
                                                          params=params)
    state_observer = ClosestAgentsObserver(params=params)
    action_wrapper = DynamicModel(params=params)
    evaluator = GoalReached(params=params)
    viewer = MPViewer(params=params,
                      x_range=[-30, 30],
                      y_range=[-20, 40],
                      follow_agent_id=True) # use_world_bounds=True

    runtimerl = RuntimeRL(action_wrapper=action_wrapper,
                          observer=state_observer,
                          evaluator=evaluator,
                          step_time=0.05,
                          viewer=viewer,
                          scenario_generator=scenario_generation)

    tfa_env = TFAWrapper(runtimerl)
    _ = tfa_env.reset()


if __name__ == '__main__':
  unittest.main()