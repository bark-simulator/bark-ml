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
                              13,
                              5] 
    self._observation_len = \
      self._max_num_vehicles*self._len_state

  def _norm_value(self, value, range):
    value = (value - range[0])/(range[1]-range[0])
    return value

  # TODO(@hart): implement relative observer
  def observe(self, world, agents_to_observe):
    # agent = world.agents[agents_to_observe[0]]
    # next_goal = agent.goal_definition. \
    #   GetNextGoal(agent)
    # agent_state = agent.state    
    # center_pt = next_goal.goal_shape.center
    # dx = center_pt[0] - agent_state[6]
    # dy = center_pt[1] - agent_state[9]
    # # dz = center_pt[2] - agent_state[12]
    concatenated_state = super(CustomObserver, self).observe(world, agents_to_observe)
    # # OVERWRITE X, Y, Z
    # concatenated_state[0] = self._norm_value(dx, [-90., 90.])
    # concatenated_state[2] = self._norm_value(dy, [-90., 90.])
    # concatenated_state[6] = self._norm_value(center_pt[2], [-2.*3.14, 2.*3.14])
    return concatenated_state

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
                 [-2000, 2000])
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
                 [-250., 250.])
    return agent_state

  # TODO(@hart): HACK
  def reset(self, world, agents_to_observe):
    self._world_x_range = [-100., 100.]
    self._world_y_range = [-100., 100.]
    return world
  
