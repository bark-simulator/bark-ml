# Copyright (c) 2021 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

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

    self._map_interace = map_interface
    self._observer = observer
    self._viewer = viewer
    self._render = render
    self._world = None
    self._ego_id = 0
    self._params = params
    self._ml_behavior = BehaviorContinuousML(self._params)

  def __step(self, step_time):
    # step and observe
    self._world.Step(self.step_time)

    # render
    if self._render:
      self.__render()

    (action, state) = self.ego_agent.history[-1]
    return (action, state)

  def generateTrajectory(self, step_time, num_steps):
    traj = []
    for i in range(0, num_steps):
      (a, s) = self.__step(step_time)
      traj.append(s)

  def setupWorld(self):
    # TODO: create world from map interface
    pass

  def addEgoAgent(self, state):
    # TODO: create ego agent
    self.__createAgent()
    pass

  def addObstacle(self, length, width, prediction):
    # TODO: create BehaviorStaticTrajectory model and add it to world
    self.__createAgent()
    # TODO fill agent
    agent_id = 101
    return agent_id

  def __createAgent(self):
    pass

  def clearAgents(self):
    # TODO: delete all agents
    pass

  def __render(self):
    # TODO: call matplotviewer
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