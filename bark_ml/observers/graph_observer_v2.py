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
    self._state_definition = 250
    self.graph_dimensions = None

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
    for node_pos, agent in sorted(pos_agent.items()):
      # TODO: make dynamic
      node_vals.append(self._norm(agent.state)[1:])
      # TODO: make dynamic
      nearby_agents = self.GetNearbyAgents(agent, pos_agent, 1000, include_ego=True)
      for nearby_agent in nearby_agents:
        nearby_agent_node_pos = self.QueryNodePos(pos_agent, nearby_agent)
        if self._self_con == False and nearby_agent_node_pos == node_pos:
          continue
        # TODO: make dynamic
        edge_value = self.CalcEdgeValue(nearby_agent, agent)[1:]
        edge_vals.append(edge_value)
        edge_ids = [nearby_agent_node_pos, node_pos]
        edge_index.append(edge_ids)
    
    node_vals = np.reshape(np.vstack(node_vals), (1, -1))
    edge_vals = np.reshape(np.vstack(edge_vals), (1, -1))
    edge_index = np.reshape(np.vstack(edge_index), (1, -1))
    observation = np.concatenate([
      np.array([[np.shape(node_vals)[1]]], dtype=np.float32),
      np.array([[np.shape(edge_vals)[1]]], dtype=np.float32),
      node_vals, edge_vals, edge_index], axis=1)
    
    # equality transf
    obs = np.zeros(shape=(1, self._len_ego_state))
    obs[0, :np.shape(observation)[1]] = observation
    
    obs = tf.convert_to_tensor(obs, dtype=tf.float32)
    return tf.reshape(obs, [-1])
  
  @classmethod
  def graph(cls, observations, graph_dims=None, dense=False):
    batch_size = tf.shape(observations)[0]
    node_vals = []
    edge_vals = []
    edge_indices = []
    node_lens = []
    edge_lens = []
    globals = []
    edge_idx_start = 0
    for obs in observations:
      len_nodes = tf.cast(obs[0], dtype=tf.int32)
      len_edges = tf.cast(obs[1], dtype=tf.int32)
      # TODO: make dynamic
      node_val = tf.reshape(obs[2:len_nodes+2], [-1, 4])
      # TODO: make dynamic
      edge_val = tf.reshape(obs[len_nodes+2:len_nodes+2+len_edges], [-1, 4])
      # print(len_edges)
      edge_index = tf.cast(
        tf.reshape(
          obs[len_nodes+2+len_edges:(len_nodes+2+len_edges + tf.cast(len_edges/2, dtype=tf.int32))], [-1, 2]), dtype=tf.int32) + edge_idx_start
      node_vals.append(node_val)
      edge_indices.append(edge_index)
      edge_vals.append(edge_val)
      edge_idx_start += tf.shape(node_val)[0]
      node_lens.append([tf.shape(node_val)[0]])
      edge_lens.append([tf.shape(edge_val)[0]])
      # globals.append(node_val[0, :])
    
    node_vals = tf.concat(node_vals, axis=0)
    edge_indices = tf.concat(edge_indices, axis=0)
    edge_vals = tf.concat(edge_vals, axis=0)
    node_lens = tf.cast(tf.concat(node_lens, axis=0), dtype=tf.int32)
    edge_lens = tf.cast(tf.concat(edge_lens, axis=0), dtype=tf.int32)
    globals = tf.stack(globals, axis=0)
    return node_vals, edge_indices, node_lens, edge_lens, globals, edge_vals
    
  @property
  def observation_space(self):
    return spaces.Box(
      low=np.zeros(self._len_ego_state),
      high=np.ones(self._len_ego_state))

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
    print(agent_state)
    return agent_state

  def _norm_to_range(self, value, range):
    return (value - range[0])/(range[1]-range[0])

  @property
  def _len_ego_state(self):
    return self._state_definition