# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import tensorflow as tf
import numpy as np
tf.compat.v1.enable_v2_behavior()

from tf_agents.environments import tf_py_environment
from modules.runtime.scenario.scenario_generation.deterministic \
  import DeterministicScenarioGeneration
from modules.runtime.scenario.scenario_generation.uniform_vehicle_distribution \
  import UniformVehicleDistribution
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from bark.models.dynamic import SingleTrackModel
from modules.runtime.runtime import Runtime

from src.rl_runtime import RuntimeRL
from src.observers.nearest_state_observer import ClosestAgentsObserver
from src.wrappers.dynamic_model import DynamicModel
from src.wrappers.tfa_wrapper import TFAWrapper
from src.evaluators.goal_reached import GoalReached
from src.agents.sac_agent import SACAgent

from configurations.sac_highway.configuration_lib import SACHighwayConfiguration
from configurations.bark_agent import BARKMLBehaviorModel

class PyBarkAgentTests(unittest.TestCase):
  def test_bark_agent(self):
    params = ParameterServer(
      filename="configurations/sac_highway/config.json")
    configuration = SACHighwayConfiguration(params)
    scenario_generator = configuration._scenario_generator
    scenario, idx = scenario_generator.get_next_scenario()
    world = scenario.get_world_state()
    dynamic_model = SingleTrackModel(params)
    bark_agent = BARKMLBehaviorModel(
      configuration)

    # test whether an action can be set
    bark_agent.SetLastAction(np.array([1., 1.]))
    np.testing.assert_array_equal(
      bark_agent.GetLastAction(),
      np.array([1., 1.]))
    bark_agent.SetLastAction(np.array([2., 1.]))

    # check whether cloning works
    new_agent = bark_agent.Clone()
    np.testing.assert_array_equal(
      new_agent.GetLastAction(),
      np.array([2., 1.]))

    # test plan step
    observed_world = world.Observe([100])[0]
    bark_agent.Plan(0.2, observed_world)
  
  def test_bark_agent_in_world(self):
    params = ParameterServer(
      filename="configurations/sac_highway/config.json")
    configuration = SACHighwayConfiguration(params)
    scenario_generator = configuration._scenario_generator
    scenario, idx = scenario_generator.get_next_scenario()
    world = scenario.get_world_state()

    # bark agent
    dynamic_model = world.agents[scenario._eval_agent_ids[0]].dynamic_model
    bark_agent = BARKMLBehaviorModel(
      configuration)
    
    world.agents[100].behavior_model = bark_agent
    for _ in range(0, 10):
      configuration._viewer.drawWorld(world)
      world.Step(0.2)
      # print(world.agents[scenario._eval_agent_ids[0]].state)

  def test_bark_agent_in_runtime(self):
    # check whether the bark agent really does work
    params = ParameterServer(
      filename="configurations/sac_highway/config.json")
    configuration = SACHighwayConfiguration(params)
    scenario_generator = configuration._scenario_generator
    
    viewer = MPViewer(params=params,
                      x_range=[-50, 50],
                      y_range=[-50, 50],
                      follow_agent_id=100,
                      screen_dims=[500, 500],
                      use_world_bounds=False)
    env = Runtime(0.2,
                  viewer,
                  scenario_generator,
                  render=True)

    env.reset()
    dynamic_model = env._world.agents[env._scenario._eval_agent_ids[0]].dynamic_model
    bark_agent = BARKMLBehaviorModel(configuration)
    env._world.agents[env._scenario._eval_agent_ids[0]].behavior_model = bark_agent

    for _ in range(0, 10):
      env.step()


if __name__ == '__main__':
  unittest.main()