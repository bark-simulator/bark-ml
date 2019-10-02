import numpy as np
from bark.models.dynamic import StateDefinition
from modules.runtime.commons.parameters import ParameterServer
import math
import operator
from src.commons.spaces import BoundedContinuous, Discrete

from src.observers.simple_observer import SimpleObserver

class CustomObserver(SimpleObserver):
  def __init__(self, params=ParameterServer()):
    SimpleObserver.__init__(self,
                            params)
    self._perform_lane_change = False
    # TODO(@hart): HACK
    self._state_definition = [6,  # x
                              9,  # y
                              3,  # theta
                              4]  # v

    
  def observe(self, world, agents_to_observe):
    return super(CustomObserver, self).observe(world, agents_to_observe)

  # TODO(@hart): HACK
  def _normalize(self, agent_state):
    agent_state = \
      self._norm(agent_state,
                 6,  # x
                 self._world_x_range)
    agent_state = \
      self._norm(agent_state,
                 9,  # y
                 self._world_y_range)
    agent_state = \
      self._norm(agent_state,
                 3,  # theta
                 [-np.pi, np.pi])
    agent_state = \
      self._norm(agent_state,
                 4,  # v
                 self._velocity_range)
    return agent_state

  # TODO(@hart): HACK
  def reset(self, world, agents_to_observe):
    self._world_x_range = [-100., 100.]
    self._world_y_range = [-100., 100.]
    return world