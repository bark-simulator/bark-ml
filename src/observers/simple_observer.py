
import numpy as np
from bark.models.dynamic import StateDefinition
from modules.runtime.commons.parameters import ParameterServer
import math
import operator
from src.commons.spaces import BoundedContinuous, Discrete

from src.observers.observer import StateObserver


class SimpleObserver(StateObserver):
  def __init__(self,
               params=ParameterServer(),
               max_number_of_vehicles=4):
    StateObserver.__init__(self, params)
    self._state_definition = [int(StateDefinition.X_POSITION),
                              int(StateDefinition.Y_POSITION),
                              int(StateDefinition.THETA_POSITION),
                              int(StateDefinition.VEL_POSITION)]
    self._max_number_of_vehicles = max_number_of_vehicles
    self._observation_len = \
      self._max_number_of_vehicles*self._len_state


  def observe(self, world, agents_to_observe):
    """see base class
    """
    concatenated_state = np.zeros(self._observation_len, dtype=np.float32)
    for i, (key, agent) in enumerate(world.agents.items()):
      reduced_state = self._select_state_by_index(agent.state)
      starts_id = i*self._len_state
      concatenated_state[starts_id:starts_id+self._len_state] = reduced_state
      if i >= self._max_number_of_vehicles:
        break
    return concatenated_state

  @property
  def observation_space(self):
    return BoundedContinuous(16,
                             low=-100000000.0,
                             high=100000000.0)

  @property
  def _len_state(self):
    return len(self._state_definition)


