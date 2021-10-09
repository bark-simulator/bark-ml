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
from bark.core.geometry import *

# bark-ml
from bark_ml.observers.observer import BaseObserver


class FrenetObserver(BaseObserver):
  """Concatenates the n-nearest states of vehicles."""

  def __init__(self, params=ParameterServer()):
    BaseObserver.__init__(self, params)

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

  def Observe(self, observed_world):
    """See base class."""
    current_state = observed_world.ego_agent.state
    d = self.getD(observed_world, current_state)
    s = self.getS(observed_world, current_state)
    state = np.array([d, s, current_state[4]])
    return state

  @property
  def observation_space(self):
    return spaces.Box(
      low=np.zeros(3),
      high = np.ones(3))

  @staticmethod
  def _norm_to_range(value, range):
    return (value - range[0])/(range[1]-range[0])