# Copyright (c) 2021 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import gym
import numpy as np
import os

import bark
from bark.core.world import World
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.core.world.map import MapInterface
from bark.core.models.behavior import BehaviorIDMClassic
from bark_ml.environments.external_runtime import ExternalRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents.sac_agent import BehaviorSACAgent
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML  # pylint: disable=unused-import
from bark.runtime.scenario.scenario import Scenario

viewer = MPViewer(params=ParameterServer(),
                  x_range=[-50, 50],
                  y_range=[-50, 50],
                  follow_agent_id=True)

class PyEnvironmentTests(unittest.TestCase):
  def setUp(self):
    params = ParameterServer()
    self.params = params
    csvfile = os.path.join(os.path.dirname(__file__), "../environments/blueprints/single_lane/base_map_lanes_guerickestr_short_assymetric_48.csv")
    print(csvfile)
    self.map_interface = MapInterface()
    self.map_interface.SetCsvMap(csvfile, 692000, 5.339e+06)

  def test_create_environment(self):
    map_interface = self.map_interface
    observer = NearestAgentsObserver()
    env = ExternalRuntime(
      map_interface=map_interface, observer=observer, params=self.params)
    self.assertTrue(isinstance(env.observation_space, gym.spaces.box.Box))

  def create_runtime_and_setup_empty_world(self, params):
    map_interface = self.map_interface
    observer = NearestAgentsObserver()
    env = ExternalRuntime(
      map_interface=map_interface, observer=observer, params=params,
      viewer=viewer)
    env.setupWorld()
    return env

  def test_setup_world(self):
    params = ParameterServer()
    env = self.create_runtime_and_setup_empty_world(params)
    self.assertTrue(isinstance(env._world, World))

  def test_add_ego_agent(self):
    params = ParameterServer()
    env = self.create_runtime_and_setup_empty_world(params)
    state = np.array([0, 0, 0, 0, 0])
    env.addEgoAgent(state)
    self.assertTrue(np.array_equal(env.ego_agent.state, state))

  def test_add_obstacle(self):
    params = ParameterServer()
    env = self.create_runtime_and_setup_empty_world(params)
    l = 4
    w = 2
    traj = np.array([[0, 0, 0, 0, 0]])
    obst_id = env.addObstacle(traj, l, w)
    self.assertTrue(np.array_equal(env._world.agents[obst_id].state, traj[0]))
    env.clearAgents()
    self.assertEqual(len(env._world.agents), 0)

  def test_create_sac_agent(self):
    params = ParameterServer()
    map_interface = self.map_interface
    observer = NearestAgentsObserver()
    env = ExternalRuntime(
      map_interface=map_interface, observer=observer, params=params)
    env.ml_behavior = BehaviorContinuousML(params)
    sac_agent = BehaviorSACAgent(environment=env, params=params)
    env.ml_behavior = sac_agent
    self.assertTrue(isinstance(env.ml_behavior, BehaviorSACAgent))

  def test_generate_ego_trajectory(self):
    params = ParameterServer()
    env = self.create_runtime_and_setup_empty_world(params)
    sac_agent = BehaviorSACAgent(environment=env, params=params)
    env.ml_behavior = sac_agent

    # add ego
    state = np.array([0, 0, 0, 0, 0])
    env.addEgoAgent(state)

    N = 10
    state_traj, action_traj = env.generateTrajectory(0.2, N)
    env._viewer.drawTrajectory(state_traj)
    env.render()
    self.assertEqual(len(state_traj), N)

  def test_generate_ego_trajectory_with_IDM(self):
    params = ParameterServer()
    env = self.create_runtime_and_setup_empty_world(params)
    env.ml_behavior = BehaviorIDMClassic(params)

    # add ego
    state = np.array([0, 0, 0, 0, 0])
    env.addEgoAgent(state)

    N = 10
    state_traj, action_traj = env.generateTrajectory(0.2, N)
    env._viewer.drawTrajectory(state_traj)
    env.render()
    self.assertEqual(len(state_traj), N)

  def test_append_to_scenario_history(self):
    params = ParameterServer()
    env = self.create_runtime_and_setup_empty_world(params)

    # behavior_model = BehaviorSACAgent(environment=env, params=params)
    behavior_model = bark.core.models.behavior.BehaviorConstantAcceleration(
      params)

    env.ml_behavior = behavior_model
    state = np.array([0, 0, 0, 0, 0])
    env.addEgoAgent(state)
    scenario = env.getScenarioForSerialization()
    self.assertTrue(isinstance(scenario, Scenario))

if __name__ == '__main__':
  unittest.main()