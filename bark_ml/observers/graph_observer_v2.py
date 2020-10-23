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
import functools
import tensorflow as tf

from bark.core.geometry import Distance, Point2d
from bark_ml.observers.observer import StateObserver


class GraphObserverV2(StateObserver):
  def __init__(self, params=ParameterServer()):
    StateObserver.__init__(self, params)
    self._filter_values = [int(StateDefinition.X_POSITION),
                           int(StateDefinition.Y_POSITION),
                           int(StateDefinition.THETA_POSITION),
                            int(StateDefinition.VEL_POSITION)]

    self._self_con = False

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
  
  def GetNearbyAgents(self, center_agent, agents, radius: float, include_ego=False):
    center_agent_pos = self._position(center_agent)
    if include_ego == False:
      other_agents = filter(lambda a: a.id != center_agent.id, agents.values())
    else:
      other_agents = agents.values()
    nearby_agents = []
    for agent in other_agents:
      agent_pos = self._position(agent)
      distance = Distance(center_agent_pos, agent_pos)
      if distance <= radius:
        nearby_agents.append(agent)
    return self._agents_sorted_by_distance(center_agent, nearby_agents)

  def GenSortedList(self, observed_world, radius=150, max_agents=5):
    agent_by_node_pos = {}
    nearby_agents = self.GetNearbyAgents(
      observed_world.ego_agent,
      observed_world.other_agents,
      radius)
    agent_by_node_pos[0] = observed_world.ego_agent
    insertion_idx = 1
    for agent in nearby_agents:
      agent_by_node_pos[insertion_idx] = agent
      insertion_idx += 1
      if insertion_idx == max_agents:
        break
    return agent_by_node_pos
    
  def QueryNodePos(self, agent_map, agent):
    return list(agent_map.keys())[list(agent_map.values()).index(agent)]
  
  def CalcEdgeValue(self, other_agent, agent):
    return self._norm(agent.state) - self._norm(other_agent.state)
  
  def Observe(self, observed_world):
    """see base class
    """
    pos_agent = self.GenSortedList(observed_world)
    node_vals = []
    edge_vals = []
    edge_index = []
    for node_pos, agent in pos_agent.items():
      node_vals.append(self._norm(agent.state)[1:])
      nearby_agents = self.GetNearbyAgents(agent, pos_agent, 150, include_ego=True)
      for nearby_agent in nearby_agents:
        nearby_agent_node_pos = self.QueryNodePos(pos_agent, nearby_agent)
        if self._self_con == False and nearby_agent_node_pos == node_pos:
          continue
        edge_value = self.CalcEdgeValue(nearby_agent, agent)[1:]
        edge_vals.append(edge_value)
        edge_ids = [nearby_agent_node_pos, node_pos]
        edge_index.append(edge_ids)
          
    node_vals = tf.reshape(
      tf.convert_to_tensor(node_vals, dtype=tf.float32), [1, -1])
    edge_vals = tf.reshape(
      tf.convert_to_tensor(edge_vals, dtype=tf.float32), [1, -1])
    edge_index = tf.reshape(
      tf.convert_to_tensor(edge_index, dtype=tf.float32), [1, -1])
    
    observation = tf.concat([
      tf.reshape(tf.cast(tf.shape(node_vals)[1], dtype=tf.float32), [1, 1]),
      tf.reshape(tf.cast(tf.shape(edge_vals)[1], dtype=tf.float32), [1, 1]),
      node_vals, edge_vals, edge_index], axis=1)
    return observation
  
  @classmethod
  def graph(cls, observations, dense=False):
    # len_nodes = tf.cast(observations[:, 0], dtype=tf.int32) # nodes sizes
    # len_edges = tf.cast(observations[:, 1], dtype=tf.int32) # edge sizes
    batch_size = tf.shape(observations)[0]
    obs = observations
    
    # row_ids = tf.reshape(tf.range(batch_size), [-1, 1])
    # col_ids = tf.reshape(tf.range(2, 25), [1, -1])
    
    node_vals = []
    edge_vals = []
    edge_indices = []
    edge_idx_start = 0
    for obs in observations:
      len_nodes = tf.cast(obs[0], dtype=tf.int32)
      len_edges = tf.cast(obs[1], dtype=tf.int32)
      node_val = tf.reshape(obs[2:len_nodes+2], [-1, 4])
      edge_index = tf.reshape(obs[len_nodes+2:len_nodes+2+len_edges], [-1, 4])
      edge_val = tf.cast(
        tf.reshape(
          obs[len_nodes+2+len_edges:len_nodes+2+2*len_edges], [-1, 2]), dtype=tf.int32) + edge_idx_start
      # print(node_val, edge_index, edge_val)
      node_vals.append(node_val)
      edge_indices.append(edge_index)
      edge_vals.append(edge_val)
      edge_idx_start += tf.shape(node_val)[0]
      
    node_vals = tf.concat(node_vals, axis=0)
    edge_indices = tf.concat(edge_indices, axis=0)
    edge_vals = tf.concat(edge_vals, axis=0)
    return node_vals, edge_indices, edge_vals
    
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