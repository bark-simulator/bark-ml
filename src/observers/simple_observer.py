
from gym import spaces
import numpy as np
from bark.models.dynamic import StateDefinition
from modules.runtime.commons.parameters import ParameterServer
import math
import operator
from src.commons.spaces import BoundedContinuous, Discrete

from src.observers.observer import StateObserver


class SimpleObserver(StateObserver):
  def __init__(self,
               params=ParameterServer()):
    StateObserver.__init__(self, params)
    self._state_definition = [int(StateDefinition.X_POSITION),
                              int(StateDefinition.Y_POSITION),
                              int(StateDefinition.THETA_POSITION),
                              int(StateDefinition.VEL_POSITION)]
    self._observation_len = \
      self._max_num_vehicles*self._len_state

  def observe(self, world, agents_to_observe):
    """see base class
    """

    concatenated_state = np.zeros(self._observation_len, dtype=np.float32)
    for i, (_, agent) in enumerate(world.agents.items()):
      normalized_state = self._normalize(agent.state)
      reduced_state = self._select_state_by_index(normalized_state)
      starts_id = i*self._len_state
      concatenated_state[starts_id:starts_id+self._len_state] = reduced_state
      
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
                 self._theta_range)
    agent_state = \
      self._norm(agent_state,
                 StateDefinition.VEL_POSITION,
                 self._velocity_range)
    return agent_state

  def reset(self, world, agents_to_observe):
    super(SimpleObserver, self).reset(world, agents_to_observe)
    return world

  @property
  def observation_space(self):
    return spaces.Box(
      low=np.zeros(self._observation_len),
      high=np.ones(self._observation_len))

  @property
  def _len_state(self):
    return len(self._state_definition)


