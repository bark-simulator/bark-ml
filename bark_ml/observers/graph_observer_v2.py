# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from gym import spaces
import numpy as np
from bark.core.models.dynamic import StateDefinition
from bark.core.world import World, ObservedWorld
from bark.runtime.commons.parameters import ParameterServer
import math
import operator

from bark.core.geometry import Distance, Point2d
from bark_ml.observers.observer import StateObserver


class GraphObserverV2(StateObserver):
  def __init__(self, params=ParameterServer()):
    StateObserver.__init__(self, params)
    self._filter_values = [int(StateDefinition.X_POSITION),
                           int(StateDefinition.Y_POSITION),
                           int(StateDefinition.THETA_POSITION),
                            int(StateDefinition.VEL_POSITION)]

    self._self_con = True

  def _position(self, agent) -> Point2d:
    return Point2d(
      agent.state[int(StateDefinition.X_POSITION)],
      agent.state[int(StateDefinition.Y_POSITION)]
    )
    
  def _agents_sorted_by_distance(self, ego_agent, agents):
    """
    Returns the given list of `agents`, sorted in ascending
    order by their relative distance to the `ego_agent`.
    """
    def distance(agent):
      return Distance(
        self._position(ego_agent), 
        self._position(agent))
    if len(agents) > 2:
      agents.sort(key=distance)
    return agents
  
  def GetNearbyAgents(self, center_agent, agents, radius: float):
    center_agent_pos = self._position(center_agent)
    other_agents = filter(lambda a: a[1].id != center_agent.id, agents)
    nearby_agents = []
    for index, agent in other_agents:
      agent_pos = self._position(agent)
      distance = Distance(center_agent_pos, agent_pos)
      if distance <= radius:
        nearby_agents.append((index, agent))
    return self._agents_sorted_by_distance(center_agent, agents)

  def GenSortedList(self, observed_world, radius=150):
    agent_by_node_pos = {}
    nearby_agents = self.GetNearbyAgents(
      observed_world.ego_agent,
      observed_world.other_agents,
      radius)
    agent_by_node_pos[0] = observed_world.ego_agent
    insertion_idx = 1
    for agent in nearby_agents:
      agent_by_node_pos[insertion_idx] = agent
    return agent_by_node_pos
    
  def QueryNodePos(self, agent_map, agent):
    return list(agent_map.keys())[list(agent_map.values()).index(agent)]]
  
  def CalcEdgeValue(self, other_agent, agent):
    return self._norm(agent.state) - self._norm(other_agent.state)
  
  def Observe(self, observed_world):
    """see base class
    """
    pos_agent = self.GenSortedList(observed_world)
    node_vals = [], edge_vals = [], edge_index = []
    for node_pos, agent in pos_agent.items():
      node_vals.append(self._norm(agent.state))
      nearby_agents = self.GetNearbyAgents(agent, pos_agent, 150)
      for nearby_agent in nearby_agents:
        nearby_agent_node_pos = self.QueryNodePos(pos_agent, nearby_agent)
        if self._self_con == False and nearby_agent_node_pos == node_pos:
          continue
        edge_value = self.CalcEdgeValue(nearby_agent, agent)
        edge_vals.append(edge_value)
        edge_ids = [nearby_agent_node_pos, node_pos]
        edge_index.append(edge_index)
        
    # TODO: concat to single vector
    observation = [len_nodes, len_edges, node_vals, edge_vals, edge_index]
    return observation
  
  @classmethod
  def graph(cls, observations, graph_dims, dense=False):
    len_nodes = observations[:, 0] # nodes sizes
    len_edges = observations[:, 1] # edge sizes
    obs = observations
    
    curr_idx = len_nodes*4
    node_vals = obs[:, 1:curr_idx]
    next_idx = len_edges*4
    edge_vals = obs[:, curr_idx:curr_idx+next_idx]
    edge_idx = obs[:, curr_idx+next_idx:curr_idx+2*2*next_idx]
    edge_idx = tf.reshape(edge_idx, [-1, 2])
    return node_vals, edge_idx, edge_vals
    
  @property
  def observation_space(self):
    return spaces.Box(
      low=np.zeros(self._len_ego_state + \
        self._max_num_vehicles*self._len_relative_agent_state),
      high = np.ones(self._len_ego_state + \
        self._max_num_vehicles*self._len_relative_agent_state))

  def _norm(self, agent_state):
    if not self._normalization_enabled:
        return agent_state
    agent_state[int(StateDefinition.X_POSITION)] = \
      self._norm_to_range(agent_state[int(StateDefinition.X_POSITION)],
                          self._world_x_range)
    agent_state[int(StateDefinition.Y_POSITION)] = \
      self._norm_to_range(agent_state[int(StateDefinition.Y_POSITION)],
                          self._world_y_range)
    agent_state[int(StateDefinition.THETA_POSITION)] = \
      self._norm_to_range(agent_state[int(StateDefinition.THETA_POSITION)],
                          self._theta_range)
    agent_state[int(StateDefinition.VEL_POSITION)] = \
      self._norm_to_range(agent_state[int(StateDefinition.VEL_POSITION)],
                          self._velocity_range)
    return agent_state

  def _norm_to_range(self, value, range):
    return (value - range[0])/(range[1]-range[0])

  @property
  def _len_relative_agent_state(self):
    return len(self._state_definition)

  @property
  def _len_ego_state(self):
    return len(self._state_definition)