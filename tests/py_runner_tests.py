# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import matplotlib
matplotlib.use('PS')
import unittest
import tensorflow as tf
from tf_agents.environments import tf_py_environment

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
from src.agents.sac_agent import SACAgent
from src.runners.tfa_runner import TFARunner

tf.compat.v1.enable_v2_behavior()

class PyRunnerTests(unittest.TestCase):
  @staticmethod
  def test_runner():
    params = ParameterServer(
      filename="tests/data/deterministic_scenario_test.json")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    params["BaseDir"] = base_dir
    scenario_generation = DeterministicScenarioGeneration(num_scenarios=3,
                                                          random_seed=0,
                                                          params=params)
    state_observer = ClosestAgentsObserver(params=params)
    action_wrapper = DynamicModel(params=params)
    evaluator = GoalReached(params=params)
    viewer = MPViewer(params=params,
                      x_range=[-30,30],
                      y_range=[-20,40],
                      follow_agent_id=True)
    runtimerl = RuntimeRL(action_wrapper=action_wrapper,
                          observer=state_observer,
                          evaluator=evaluator,
                          step_time=0.2,
                          viewer=viewer,
                          scenario_generator=scenario_generation,
                          render=False)
    tfa_env = tf_py_environment.TFPyEnvironment(TFAWrapper(runtimerl))
    sac_agent = SACAgent(tfa_env,
                         params=params)
    tfa_runner = TFARunner(tfa_env,
                           sac_agent,
                           params=params,
                           unwrapped_runtime=runtimerl)
    tfa_runner.collect_initial_episodes()
    
    # main functionalities
    tfa_runner.train()
    tfa_runner.visualize()
    tfa_runner.evaluate()


if __name__ == '__main__':
  unittest.main()