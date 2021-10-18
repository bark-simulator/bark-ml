# Copyright (c) 2021 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from bark.core.world.agent import *
from bark.core.models.behavior import *
from bark.core.world import *
from bark.core.world.goal_definition import *
from bark.core.models.dynamic import *
from bark.core.models.execution import *
from bark.core.geometry import *
from bark.core.geometry.standard_shapes import *
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML  # pylint: disable=unused-import


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
    self._ml_behavior = BehaviorContinuousML(self._params)

  def _step(self, step_time):
    # step and observe
    self._world.Step(step_time)

    # render
    if self._render:
      self.render()

    (action, state) = self.ego_agent.history[-1]
    return (action, state)

  def generateTrajectory(self, step_time, num_steps):
    traj = []
    for i in range(0, num_steps):
      (a, s) = self._step(step_time)
      traj.append(s)
    return traj

  def setupWorld(self):
    world = World(self._params)
    world.SetMap(self._map_interface)
    self._world = world

  def addEgoAgent(self, state):
    agent = self._createAgent(state, self._ml_behavior)
    self._world.AddAgent(agent)
    self._ego_id = agent.id
    return agent.id

  def addObstacle(self, prediction, length, width):
    behavior = None
    # TODO: create BehaviorStaticTrajectory model and add it to world
    agent = self._createAgent(prediction[0], behavior, wb=length, crad=width)
    # TODO fill agent
    self._world.AddAgent(agent)
    return agent.id

  def _createAgent(self, state, behavior, wb=2., crad=1.):
    agent_behavior = behavior
    agent_dyn = SingleTrackModel(self._params)
    agent_exec = ExecutionModelInterpolate(self._params)
    agent_polygon = GenerateCarRectangle(wb, crad)
    agent_params = self._params.AddChild("agent")
    # agent_goal = GoalDefinitionPolygon()
    new_agent = Agent(
      state,
      agent_behavior,
      agent_dyn,
      agent_exec,
      agent_polygon,
      agent_params,
      None,
      self._map_interface)
    return new_agent

  def clearAgents(self):
    # TODO: implement in BARK
    self._world.ClearAgents()

  def render(self):
    # TODO: call matplotviewer
    # self._viewer.drawWorld()
    pass

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