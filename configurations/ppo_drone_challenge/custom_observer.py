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
                              7,
                              9,  # y
                              10,
                              12,  # z
                              13] 
    self._observation_len = \
      self._max_num_vehicles*self._len_state
      
  # TODO(@hart): implement relative observer
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
                 12,  # z
                 [-100, 100])
    agent_state = \
      self._norm(agent_state,
                 7,  # vx
                 [-25, 25])
    agent_state = \
      self._norm(agent_state,
                 10,  # vy
                 [-25, 25])
    agent_state = \
      self._norm(agent_state,
                 13,  # vz
                 [-25, 25])
    return agent_state

  # TODO(@hart): HACK
  def reset(self, world, agents_to_observe):
    self._world_x_range = [-100., 100.]
    self._world_y_range = [-100., 100.]
    return world
  
