
from gym import spaces
import numpy as np
import math
import operator

from bark.models.dynamic import StateDefinition
from bark.world import ObservedWorld
from modules.runtime.commons.parameters import ParameterServer
from src.observers.observer import StateObserver


class GraphObserver(StateObserver):
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

  def observe(self, world):
    """see base class
    """
    graph = {
      'nodes': {},
      'edges': []
    }

    ego_agent = world.ego_agent

    for (_, agent) in world.agents.items():
      state = self._normalize(agent.state) if self._normalize_observations else agent.state
      graph['nodes'][agent.id] = self._create_node(agent)

    return graph, world

  def _create_node(self, agent):
    return {
      'x_pos': agent.state[int(StateDefinition.X_POSITION)],
      'y_pos': agent.state[int(StateDefinition.Y_POSITION)],
      'theta_pos': agent.state[int(StateDefinition.THETA_POSITION)],
      'vel_pos': agent.state[int(StateDefinition.VEL_POSITION)]
    }

  def _create_edge(self, agent1, agent2):
    return {
      'node1': agent1.id,
      'node2': agent2.id
    }

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

  def reset(self, world):
    return world

  @property
  def observation_space(self):
    return spaces.Box(
      low=np.zeros(self._observation_len),
      high=np.ones(self._observation_len))

  @property
  def _len_state(self):
    return len(self._state_definition)


