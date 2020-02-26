# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import pickle
import os
import copy
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

from configurations.highway.configuration_lib import HighwayConfiguration
from configurations.bark_agent import BARKMLBehaviorModel

class PyBarkAgentTests(unittest.TestCase):
  def test_bark_agent_in_runtime(self):
    # check whether the bark agent really does work
    params = ParameterServer(
      filename="configurations/highway/config.json")
    configuration = HighwayConfiguration(params)
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

    env.step()

    f = open(os.path.join("./ml_behavior_model.pickle"), "wb")
    pickle.dump(bark_agent, f)

    f = open(os.path.join("./ml_behavior_model.pickle"), "rb")
    bark_agent = pickle.load(f)

    ba_copy = copy.deepcopy(bark_agent)
    env._world.agents[env._scenario._eval_agent_ids[0]].behavior_model = ba_copy
    env.step()

if __name__ == '__main__':
  unittest.main()