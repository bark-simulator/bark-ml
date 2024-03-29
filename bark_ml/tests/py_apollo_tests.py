# Copyright (c) 2021 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# try:
#     import debug_settings
# except:
#     pass

import unittest
import gym
import numpy as np
import os

import bark
from bark.core.world import World
from bark.core.geometry import Line2d, Point2d, Distance
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
    state = np.array([0, 0, 0, 0, 0, 0])
    goal_line = Line2d(np.array([[0., 0.], [1., 1.]]))
    env.addEgoAgent(state, goal_line)
    self.assertTrue(np.array_equal(env.ego_agent.state, state))

  def test_create_ROI(self):
    params = ParameterServer()
    if params["World"]["enable_roi","",True]:
      env = self.create_runtime_and_setup_empty_world(params)
    self.assertTrue(env._rect_around_ego is not None)
    self.assertTrue(env._rect_around_ego.Valid())
    a_rect = env._rect_around_ego.CalculateArea()
    self.assertEqual(a_rect, 480)
    self.assertTrue(env._roi is None)
    # Intersection of road corridor and ego rectangle
    state = np.array([0, 970.157, 46.99, 0, 0, 0])
    goal_line = Line2d(np.array([[970.157, 46.99], [971.157, 47.99]]))
    env.addEgoAgent(state, goal_line)
    env.createROI4EgoAgent()
    self.assertTrue(env._roi is not None)
    # # ego_pose = np.array([970.157, 46.99, 0])
    # curr_rect_ego = env._rect_around_ego.Transform(env._ego_pose)
    # print("ego pose:",env._ego_pose)
    self.assertTrue(env._roi.Valid())
    # self.assertTrue(curr_rect_ego.Valid())
    a_roi = env._roi.CalculateArea()
    self.assertEqual(a_roi, a_rect)
    # env._viewer.drawPolygon2d(env._rect_around_ego,'r',0.2)
    # env._viewer.drawPolygon2d(curr_rect_ego,'r',0.2)
    # env._viewer.drawPolygon2d(env._curr_road_corridor.polygon,'r',0.2)
    env._viewer.axes.axis('equal')
    env._viewer.drawPoint2d(Point2d(969.685, 46.5325), 'b',1.0)
    # env.render()
    env._viewer.show(block=False)

    # No Intersection of road corridor and ego rectangle:
    # using ego rectangle
    env.clearAgents()
    state = np.array([0, 0, 0, 0, 0, 0])
    goal_line = Line2d(np.array([[0, 0], [1, 1]]))
    env.addEgoAgent(state, goal_line)
    env.createROI4EgoAgent()
    a_roi_new = env._roi.CalculateArea()
    self.assertEqual(a_roi_new, a_rect)
    
    # env._viewer.drawPolygon2d(env._roi,'r',0.2)
    # env.render()
    # env._viewer.show()
  
  def test_add_obstacle(self):
    params = ParameterServer()
    if params["World"]["enable_roi","",True]:
      env = self.create_runtime_and_setup_empty_world(params)
    l = 4
    w = 2
    state = np.array([0, 1008, 28, 0, 0, 0])
    goal_line = Line2d(np.array([[1008., 28.], [1009., 29.]]))
    env.addEgoAgent(state, goal_line)
    env.createROI4EgoAgent()
    # add obstacle in ROI
    traj = np.array([[0, 1007, 27, 0, 0]])
    obst_id = env.addObstacle(traj, l, w)
    self.assertNotEqual(obst_id, -1)
    self.assertTrue(np.array_equal(env._world.agents[obst_id].state, traj[0]))
    # try adding obstacle outside of ROI
    traj2 = np.array([[0, 100000, 0, 0, 0]])
    obst_id2 = env.addObstacle(traj2, l, w)
    self.assertEqual(obst_id2, -1)
    self.assertEqual(len(env._world.agents), 2)
    env.clearAgents()

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
    state = np.array([0, 0, 0, 0, 0, 0])
    goal_line = Line2d(np.array([[0., 0.], [1., 1.]]))
    env.addEgoAgent(state, goal_line)

    N = 10
    state_traj, action_traj = env.generateTrajectory(0.2, N)
    # env._viewer.drawTrajectory(state_traj)
    # env.render()
    self.assertEqual(len(state_traj), N)

  def test_generate_ego_trajectory_with_IDM(self):
    params = ParameterServer()
    env = self.create_runtime_and_setup_empty_world(params)
    env.ml_behavior = BehaviorIDMClassic(params)

    # add ego
    state = np.array([0, 0, 0, 0, 0])
    goal_line = Line2d(np.array([[0., 0.], [1., 1.]]))
    env.addEgoAgent(state, goal_line)

    N = 10
    state_traj, action_traj = env.generateTrajectory(0.2, N)
    # env._viewer.drawTrajectory(state_traj)
    # env.render()
    self.assertEqual(len(state_traj), N)

  def test_append_to_scenario_history(self):
    params = ParameterServer()
    env = self.create_runtime_and_setup_empty_world(params)

    # behavior_model = BehaviorSACAgent(environment=env, params=params)
    behavior_model = bark.core.models.behavior.BehaviorConstantAcceleration(
      params)

    env.ml_behavior = behavior_model
    state = np.array([0, 0, 0, 0, 0])
    goal_line = Line2d(np.array([[0., 0.], [1., 1.]]))
    env.addEgoAgent(state, goal_line)
    scenario = env.getScenarioForSerialization()
    self.assertTrue(isinstance(scenario, Scenario))

if __name__ == '__main__':
  unittest.main()