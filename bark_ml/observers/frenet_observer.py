# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from gym import spaces
import numpy as np
from bark.runtime.commons.parameters import ParameterServer
from bark.core.models.dynamic import StateDefinition
from bark.core.geometry import *

# bark-ml
from bark_ml.observers.observer import BaseObserver


class FrenetObserver(BaseObserver):
  """Concatenates the n-nearest states of vehicles."""

  def __init__(self, params=ParameterServer()):
    BaseObserver.__init__(self, params)
    self._num_other_agents = \
      self._params["ML"]["FrenetObserver"]["NumOtherAgents",
      "Number of other agents",
      1]
    self._road_norm_width = \
      self._params["ML"]["FrenetObserver"]["RoadNormWidth",
      "Road width.",
      4.8]
    self._max_vel = \
      self._params["ML"]["FrenetObserver"]["MaxVel",
      "Max. velocity",
      15.]

  def getLaneCorr(self, observed_world):
    rc = observed_world.ego_agent.road_corridor
    return rc.lane_corridors[0]

  def getD(self, observed_world, current_state):
    lane_corr = self.getLaneCorr(observed_world)
    center_line = lane_corr.center_line
    d = Distance(
      center_line, Point2d(current_state[1], current_state[2]))
    return d

  def getS(self, observed_world, current_state):
    lane_corr = self.getLaneCorr(observed_world)
    center_line = lane_corr.center_line
    s = GetNearestS(
      center_line, Point2d(current_state[1], current_state[2]))
    return s

  def transformAgentToFrenet(self, observed_world, current_state):
    d = self.getD(observed_world, current_state)
    s = self.getS(observed_world, current_state)
    lc = self.getLaneCorr(observed_world)
    s = self._norm_to_range(s, [0, lc.center_line.Length()])
    d = self._norm_to_range(
      d, [-self._road_norm_width, self._road_norm_width])
    v = self._norm_to_range(
      current_state[int(StateDefinition.VEL_POSITION)], [0, self._max_vel])
    theta = self._norm_to_range(
      current_state[int(StateDefinition.THETA_POSITION)], [0, 6.3])
    return [s, d, theta, v]

  def getNearbyAgents(self, observed_world, state):
    ego_position = Point2d(
      state[int(StateDefinition.X_POSITION)],
      state[int(StateDefinition.Y_POSITION)])
    return observed_world.GetNearestAgents(
      ego_position, self._num_other_agents + 1)

  @staticmethod
  def flatten(t):
    return [item for sublist in t for item in sublist]

  def Observe(self, observed_world):
    """See base class."""
    current_state = observed_world.ego_agent.state
    observed_states = []
    ego_frenet = self.transformAgentToFrenet(
      observed_world, current_state)
    observed_states.append(ego_frenet)
    nearby_agents = self.getNearbyAgents(observed_world, current_state)
    for agent_id, agent in nearby_agents.items():
      if agent_id != observed_world.ego_agent.id:
        other_state = self.transformAgentToFrenet(observed_world, agent.state)
        observed_states.append(other_state)
    flat_observation = self.flatten(observed_states)
    state = np.array(flat_observation, dtype=np.float32)
    return state

  @property
  def observation_space(self):
    return spaces.Box(
      low=np.zeros(4*(self._num_other_agents+1)),
      high = np.ones(4*(self._num_other_agents+1))
    )

  @staticmethod
  def _norm_to_range(value, range):
    return (value - range[0])/(range[1]-range[0])