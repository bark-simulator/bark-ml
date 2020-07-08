# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from gym import spaces
import numpy as np
import math
import operator

from bark.core.models.dynamic import StateDefinition
from bark.core.world import ObservedWorld
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.observer import StateObserver


class SimpleObserver(StateObserver):
  def __init__(self,
               normalize_observations=True,
               params=ParameterServer()):
    StateObserver.__init__(self, params)
    self._state_definition = [int(StateDefinition.X_POSITION),
                              int(StateDefinition.Y_POSITION),
                              int(StateDefinition.THETA_POSITION),
                              int(StateDefinition.VEL_POSITION)]
    self._observation_len = \
      self._max_num_vehicles*self._len_state
    self._normalize_observations = normalize_observations

  def Observe(self, world):
    """see base class
    """
    concatenated_state = np.zeros(self._observation_len, dtype=np.float32)
    for i, (_, agent) in enumerate(world.agents.items()):
      state = agent.state
      if self._normalize_observations:
        state = self._normalize(state)
      reduced_state = self._select_state_by_index(state)
      # use x,y,vx,vy
      new_state = np.copy(reduced_state)
      new_state[2] = reduced_state[3]*np.cos(reduced_state[2])
      new_state[3] = reduced_state[3]*np.sin(reduced_state[2])
      starts_id = i*self._len_state
      concatenated_state[starts_id:starts_id+self._len_state] = new_state
      if i >= self._max_num_vehicles:
        break
    return concatenated_state

  def _norm(self, agent_state, position, range):
    agent_state[int(position)] = \
      (agent_state[int(position)] - range[0])/(range[1]-range[0])
    return agent_state

  def _normalize(self, agent_state):
    agent_state = \
      self._norm(agent_state,
                 StateDefinition.X_POSITION,
                 self._world_x_range)
    agent_state = \
      self._norm(agent_state,
                 StateDefinition.Y_POSITION,
                 self._world_y_range)
    agent_state = \
      self._norm(agent_state,
                 StateDefinition.THETA_POSITION,
                 self._ThetaRange)
    agent_state = \
      self._norm(agent_state,
                 StateDefinition.VEL_POSITION,
                 self._VelocityRange)
    return agent_state

  def Reset(self, world, controlled_agent_ids):
    return world

  @property
  def observation_space(self):
    return spaces.Box(
      low=np.zeros(self._observation_len),
      high=np.ones(self._observation_len))

  @property
  def _len_state(self):
    return len(self._state_definition)


