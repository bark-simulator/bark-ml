# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from tf_agents.environments import tf_py_environment
from modules.runtime.scenario.scenario_generation.deterministic \
  import DeterministicScenarioGeneration
from modules.runtime.scenario.scenario_generation.uniform_vehicle_distribution \
  import UniformVehicleDistribution
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from bark.models.dynamic import SingleTrackModel


from src.rl_runtime import RuntimeRL
from src.observers.nearest_state_observer import ClosestAgentsObserver
from src.wrappers.dynamic_model import DynamicModel
from src.wrappers.tfa_wrapper import TFAWrapper
from src.evaluators.goal_reached import GoalReached
from src.agents.sac_agent import SACAgent

from configurations.sac_highway_uniform.configuration import SACHighwayConfiguration
from configurations.bark_agent import BARKMLBehaviorModel

class PyBarkAgentTests(unittest.TestCase):
  @unittest.skip("..")
  def test_bark_agent(self):
    params = ParameterServer(
      filename="configurations/sac_highway_uniform/config.json")
    configuration = SACHighwayConfiguration(params)
    scenario_generator = configuration._scenario_generator
    scenario, idx = scenario_generator.get_next_scenario()
    world = scenario.get_world_state()
    dynamic_model = SingleTrackModel(params)
    bark_agent = BARKMLBehaviorModel(configuration, dynamic_model, scenario._eval_agent_ids)
    bark_agent.plan(0.2, world)
    bark_agent.plan(0.2, world)
    bark_agent.plan(0.2, world)
    new_agent = bark_agent.clone()
    new_agent.plan(0.2, world)
    new_agent.plan(0.2, world)
    new_agent.plan(0.2, world)

  def test_bark_agent_in_world(self):
    params = ParameterServer(
      filename="configurations/sac_highway_uniform/config.json")
    configuration = SACHighwayConfiguration(params)
    scenario_generator = configuration._scenario_generator
    scenario, idx = scenario_generator.get_next_scenario()
    world = scenario.get_world_state()

    # bark agent
    dynamic_model = world.agents[scenario._eval_agent_ids[0]].dynamic_model
    bark_agent = BARKMLBehaviorModel(configuration, dynamic_model, scenario._eval_agent_ids)
    world.agents[scenario._eval_agent_ids[0]].behavior_model = bark_agent


    for _ in range(0, 10):
      configuration._viewer.drawWorld(world)
      world.step(0.2)
      print(world.agents[scenario._eval_agent_ids[0]].state)
    


if __name__ == '__main__':
  unittest.main()