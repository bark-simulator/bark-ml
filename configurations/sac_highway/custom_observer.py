import numpy as np
from bark.models.dynamic import StateDefinition
from modules.runtime.commons.parameters import ParameterServer
import math
import operator
from src.commons.spaces import BoundedContinuous, Discrete

from src.observers.simple_observer import SimpleObserver


class CustomObserver(SimpleObserver):
  def __init__(self, params=ParameterServer()):
    SimpleObserver.__init__(self, params)
    self._perform_lane_change = False

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


  def reset(self, world, agents_to_observe):
    params = world.get_params()
    self._perform_lane_change = \
      params["ML"]["Maneuver"]["lane_change",
        "Whether a lane change should be performed or not.",
        True]
    return world