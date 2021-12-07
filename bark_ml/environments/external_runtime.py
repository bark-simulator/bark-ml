# Copyright (c) 2021 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

from bark.core.world.agent import *
from bark.core.models.behavior import *
from bark.core.world import *
from bark.core.world.goal_definition import *
from bark.core.models.dynamic import *
from bark.core.models.execution import *
from bark.core.geometry import *
from bark.core.geometry.standard_shapes import *
from bark.runtime.scenario.scenario import Scenario
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML  # pylint: disable=unused-import
from bark.core.world.goal_definition import GoalDefinitionStateLimitsFrenet

class ExternalRuntime:
  """External runtime.

  Can either be initialized using a blueprint or by passing the
  `evaluator`, `observer`, `scenario_generation`, `step_time`, `viewer`
  and ml_behavior.
  """

  def __init__(self,
               map_interface,
               observer,
               params,
               viewer=None,
               render=False):

    self._map_interface = map_interface
    self._observer = observer
    self._viewer = viewer
    self._render = render
    self._world = None
    self._ego_id = 0
    self._params = params
    self._json_params= params.ConvertToDict()
    self._ml_behavior = BehaviorContinuousML(self._params)

  def _step(self, step_time):
    # step and observe
    self._world.Step(step_time)

    # render
    if self._render:
      self.render()

    state, action = self.ego_agent.history[-1]
    return state, action

  def generateTrajectory(self, step_time, num_steps):
    state_traj = []
    action_traj = []
    for _ in range(0, num_steps):
      s, a = self._step(step_time)
      state_traj.append(s)
      action_traj.append(a)
    return np.array(state_traj), np.array(action_traj)

  def setupWorld(self):
    world = World(self._params)
    world.SetMap(self._map_interface)
    self._world = world

  def addEgoAgent(self, state):
    agent = self._createAgent(state, self._ml_behavior, ego_vehicle=True)
    self._world.AddAgent(agent)
    self._ego_id = agent.id
    return agent.id

  def ConvertShapeParameters(self, length, width):
    crad = width/2.0 # collision circle radius
    wb = length - 2*crad # wheelbase
    return (crad, wb)

  def addObstacle(self, prediction, length, width):
    behavior = BehaviorStaticTrajectory(
      self._params,
      prediction)
    (crad, wb) = self.ConvertShapeParameters(length=length, width=width)
    agent = self._createAgent(
      prediction[0], behavior, wb=wb, crad=crad)
    self._world.AddAgent(agent)
    return agent.id

  def _createAgent(self, state, behavior, wb=2., crad=1., ego_vehicle=False):
    agent_behavior = behavior
    agent_dyn = SingleTrackModel(self._params)
    if ego_vehicle:
      agent_dyn = SingleTrackSteeringrateModel(self._params)
    agent_exec = ExecutionModelInterpolate(self._params)
    agent_polygon = GenerateCarRectangle(wb, crad)
    agent_params = self._params.AddChild("agent")

    # HACK: fake goal
    points = np.array([[0., 0.], [1., 1.]])
    new_line = Line2d(points)
    agent_goal = GoalDefinitionStateLimitsFrenet(new_line, (2.5, 2.),
      (0.15, 0.15), (3., 7.))

    new_agent = Agent(
      state,
      agent_behavior,
      agent_dyn,
      agent_exec,
      agent_polygon,
      agent_params,
      agent_goal,
      self._map_interface)
    return new_agent

  def clearAgents(self):
    self._world.ClearAgents()

  def render(self):
    self._viewer.drawWorld(
      self._world,
      [self._ego_id])
    self._viewer.clear()

  def getScenarioForSerialization(self):
    self._world.agents[self._ego_id].behavior_model = BehaviorContinuousML(
      self._params)
    scenario = Scenario(agent_list=list(self._world.agents.values()),
                        map_interface=self._map_interface,
                        eval_agent_ids=[self._ego_id],
                        json_params=self._json_params)
    new_scenario = scenario.copy()
    self._world.agents[self._ego_id].behavior_model = self._ml_behavior
    return new_scenario

  @property
  def action_space(self):
    """Action space of the agent."""
    return self._ml_behavior.action_space

  @property
  def observation_space(self):
    """Observation space of the agent."""
    return self._observer.observation_space

  @property
  def ml_behavior(self):
    return self._ml_behavior

  @ml_behavior.setter
  def ml_behavior(self, ml_behavior):
    self._ml_behavior = ml_behavior

  @property
  def ego_agent(self):
    """Action space of the agent."""
    return self._world.agents[self._ego_id]