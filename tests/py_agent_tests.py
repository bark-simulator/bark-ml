# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
from tf_agents.environments import tf_py_environment

from modules.runtime.scenario.scenario_generation.uniform_vehicle_distribution \
  import UniformVehicleDistribution
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer


from src.rl_runtime import RuntimeRL
from src.observers.nearest_state_observer import StateConcatenation
from src.wrappers.dynamic_model import DynamicModel
from src.wrappers.tfa_wrapper import TFAWrapper
from src.evaluators.goal_reached import GoalReached
from src.agents.sac_agent import SACAgent

class AgentTests(unittest.TestCase):
  @staticmethod
  def test_agent():
    params = ParameterServer(filename="data/highway_merging.json")
    scenario_generation = UniformVehicleDistribution(num_scenarios=3,
                                                     random_seed=0,
                                                     params=params)
    state_observer = StateConcatenation(params=params)
    action_wrapper = DynamicModel(params=params)
    evaluator = GoalReached(params=params)
    viewer = MPViewer(params=params,
                      x_range=[-30,30],
                      y_range=[-20,40],
                      follow_agent_id=True) # use_world_bounds=True

    runtimerl = RuntimeRL(action_wrapper=action_wrapper,
                          observer=state_observer,
                          evaluator=evaluator,
                          step_time=0.05,
                          viewer=viewer,
                          scenario_generator=scenario_generation)

    tfa_env = tf_py_environment.TFPyEnvironment(TFAWrapper(runtimerl))
    sac_agent = SACAgent(tfa_env)
    sac_agent.reset()


if __name__ == '__main__':
    unittest.main()