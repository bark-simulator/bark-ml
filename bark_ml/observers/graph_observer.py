import os, time, json, pickle
from gym import spaces
import numpy as np
import math
import operator
import networkx as nx
import tensorflow as tf
from typing import Dict
from collections import OrderedDict
from itertools import islice

from bark.models.dynamic import StateDefinition
from bark.world import ObservedWorld
from bark.geometry import Distance, Point2d
from modules.runtime.commons.parameters import ParameterServer
from bark_ml.observers.observer import StateObserver

class GraphObserver(StateObserver):
  
  def __init__(self,
               normalize_observations=True,
               use_edge_attributes=True,
               params=ParameterServer()):
    StateObserver.__init__(self, params)

    self._normalize_observations = normalize_observations
    self._use_edge_attributes = use_edge_attributes

    # the number of features of a node in the graph
    self.feature_len = 11

    # the maximum number of agents that can be observed
    self.agent_limit = 50

     # the radius an agent can 'see' in meters
    self._visible_distance = 50

  @classmethod
  def attribute_keys(cls):
    return ["x", "y", "theta", "vel", "goal_d", "goal_theta"]

  def Observe(self, world):
    """see base class"""
    graph = nx.OrderedGraph(normalization_ref=self.normalization_data)
    actions = OrderedDict() # generated for now (steering, acceleration)
    agents = self._preprocess_agents(world)
    
    # add nodes
    for (index, agent) in agents:
      # create node
      features = self._extract_features(agent)
      graph.add_node(index, **features)

      # generate actions
      actions[index] = self._generate_actions(features)
      
    # Second loop for edges necessary -> otherwise order of graph_nodes is disrupted
    for (index, agent) in agents:
      # create edges to all other agents
      nearby_agents = self._nearby_agents(agent, agents, self._visible_distance)
      for (nearby_agent_index, _) in nearby_agents:
        graph.add_edge(index, nearby_agent_index)
    
    observation = self._observation_from_graph(graph)
    
    # keep this for now, TODO: move into test
    reconstructed_graph = GraphObserver.graph_from_observation(observation)
    rec_obs = self._observation_from_graph(reconstructed_graph)
    assert observation == rec_obs, "Failure in graph en/decoding!"

    return tf.convert_to_tensor(observation, dtype=tf.float32, name='observation')

  def _observation_from_graph(self, graph):
    """ Encodes the given graph into a bounded array with fixed size.

    The returned array 'a' has the following contents:
    a[0]:                            (int) the maximum number of possibly contained nodes
    a[1]:                            (int) the actual number of contained nodes
    a[2]:                            (int) the number of features per node
    a[3: a[1] * a[2]]:               (floats) the node feature values
    a[3 + a[1] * a[2]: a[0] * a[2]]: (int) all entries have value -1
    a[-a[0] ** 2:]:                  (0 or 1) an adjacency matrix in vector form

    :type graph: A nx.Graph object.
    :param graph:
    
    :rtype: list
    """
    num_nodes = len(graph.nodes)
    obs = [self.agent_limit, num_nodes, self.feature_len]
    
    # append node features
    for (node_id, attributes) in graph.nodes.data():
      obs.extend(list(attributes.values()))

    # fill empty spots (difference between existing and max agents) with -1
    obs.extend(np.full((self.agent_limit - num_nodes) * self.feature_len, -1))

    # build adjacency list
    adjacency_list = np.zeros(self.agent_limit ** 2)
    
    for source, target in graph.edges:
      edge_idx = source * self.agent_limit + target
      adjacency_list[edge_idx] = 1

    obs.extend(adjacency_list)

    assert len(obs) == self._len_state, f'Observation has invalid length ({len(obs)}, expected: {self._len_state})'
    
    return obs

  @classmethod
  def graph_from_observation(cls, observation):
    graph = nx.OrderedGraph()

    node_limit = int(observation[0])
    num_nodes = int(observation[1])
    num_features = int(observation[2])

    obs = observation[3:]

    for node_id in range(num_nodes):
      start_idx = node_id * num_features
      end_idx = start_idx + num_features
      features = obs[start_idx:end_idx]

      attributes = dict(zip(GraphObserver.attribute_keys(), features))
      graph.add_node(node_id, **attributes)
    
    adj_start_idx = node_limit * num_features
    adj_lists = np.array_split(obs[adj_start_idx:], node_limit + 1)
    
    for (node_id, adj_list) in enumerate(adj_lists):
      for target_node_id in np.flatnonzero(adj_list):
        graph.add_edge(node_id, target_node_id)

    return graph

  def _preprocess_agents(self, world):
    """ 
    Returns a list of tuples, consisting
    of an index and an agent object element.
    """
    # make ego_agent the first element, sort others by id
    # there should be a more elegant way to do this
    ego_agent = world.ego_agent
    agents = list(world.agents.values())
    agents.remove(ego_agent)
    agents.sort(key=lambda agent: agent.id)
    agents.insert(0, ego_agent)
    return list(enumerate(agents))[:self.agent_limit]

  def _nearby_agents(self, center_agent, agents, radius: float):
    """
    Returns all elements from 'agents' within the specified 
    radius of the 'center_agent's position.
    """
    center_agent_pos = self._position(center_agent)
    other_agents = filter(lambda a: a[1].id != center_agent.id, agents)
    nearby_agents = []

    for (index, agent) in other_agents:
      agent_pos = self._position(agent)
      distance = Distance(center_agent_pos, agent_pos)

      if distance <= radius:
        nearby_agents.append((index, agent))

    return nearby_agents

  def _extract_features(self, agent) -> Dict[str, float]:
    """Returns dict containing all features of the agent"""
    res = OrderedDict()
    
    # get basic information
    state = agent.state
    res["x"] = state[int(StateDefinition.X_POSITION)] # maybe not needed
    res["y"] = state[int(StateDefinition.Y_POSITION)] # maybe not needed
    res["theta"] = state[int(StateDefinition.THETA_POSITION)]
    res["vel"] = state[int(StateDefinition.VEL_POSITION)]

    # get information related to goal
    goal_center = agent.goal_definition.goal_shape.center[0:2]
    res["goal_x"] = goal_center[0] # goal position in x
    res["goal_y"] = goal_center[1] # goal position in y
    goal_dx = goal_center[0] - res["x"] # distance to goal in x coord
    res["goal_dx"] = goal_dx
    goal_dy = goal_center[1] - res["y"] # distance to goal in y coord
    res["goal_dy"] = goal_dy
    goal_theta = np.arctan2(goal_dy, goal_dx) # theta for straight line to goal
    res["goal_theta"] = goal_theta
    goal_d = np.sqrt(goal_dx**2 + goal_dy**2) # distance to goal
    res["goal_d"] = goal_d
    
    goal_velocity = np.mean(agent.goal_definition.velocity_range)
    res["goal_vel"] = goal_velocity

    if self._normalize_observations:
      n = self.normalization_data

      for k in ["x", "y", "theta", "vel"]:
        res[k] = self._normalize_value(res[k], n[k])
      res["goal_x"] = self._normalize_value(res["goal_x"], n["x"])
      res["goal_y"] = self._normalize_value(res["goal_y"], n["y"])
      res["goal_dx"] = self._normalize_value(res["goal_dx"], n["dx"])
      res["goal_dy"] = self._normalize_value(res["goal_dy"], n["dy"])
      res["goal_d"] = self._normalize_value(res["goal_d"], n["distance"])
      res["goal_theta"] = self._normalize_value(res["goal_theta"], n["theta"])
      res["goal_vel"] = self._normalize_value(res["goal_vel"], n["vel"])
    
    #####################################################
    #    If you change the number of features,          #
    #    please adapt self.feature_len accordingly.     #
    #####################################################
    return res

  def _generate_actions(self, features: Dict[str, float]) -> Dict[str, float]:
    actions = OrderedDict()
    steering = features["goal_theta"] - features["theta"]
    
    v_0 = features["vel"]
    dv = features["goal_vel"] - v_0
    acc = (1./features["goal_d"])*dv*(dv/2+v_0)
    
    if self._normalize_observations:
      range_steering = [-0.1, 0.1]
      range_acc = [-0.6, 0.6]
      steering = self._normalize_value(steering, range_steering)
      acc = self._normalize_value(acc, range_acc)

    actions["steering"] = steering
    actions["acceleration"] = acc

    return actions

  def _normalize_value(self, value, range):
    """norms to [-1, 1] with
    value <= range[0] -> returns -1
    value >= range[1] -> returns 1"""
    normed = 2 * (value - range[0]) / (range[1] - range[0]) - 1
    normed = max(-1, normed) # values lower -1 clipped
    normed = min(1, normed) # values bigger 1 clipped
    return normed

  def reset(self, world):
    return world

  def _position(self, agent) -> Point2d:
    return Point2d(
      agent.state[int(StateDefinition.X_POSITION)],
      agent.state[int(StateDefinition.Y_POSITION)]
    )

  @property
  def normalization_data(self) -> Dict[str, list]:
    """
    The reference ranges of how certain attributes are normalized.
    Use this info to scale normalized values back to real values.
    
    E.g. the value for key 'x' returns the possible range of 
    the x-element of positions. A normalized value of 1.0 
    corresponds to the maximum of this range (and vice versa).

    Note: This dictionary does not include a range for each
    possible attribute, but rather for each _kind_ of attribute,
    like distances, velocities, angles, etc.
    E.g. all distances (between agents, between objects, etc.)
    are scaled relative to the 'distance' range. 
    """
    d = OrderedDict()
    x_range = self._world_x_range[1] - self._world_x_range[0]
    y_range = self._world_y_range[1] - self._world_y_range[0]
    max_dist = np.linalg.norm([x_range, y_range])
    d['x'] = self._world_x_range
    d['y'] = self._world_y_range
    d['theta'] = self._ThetaRange
    d['vel'] = self._VelocityRange
    d['distance'] = [0, max_dist]
    d['dx'] = [-x_range, x_range]
    d['dy'] = [-y_range, y_range]
    return d

  @property
  def observation_space(self):
    #  0 ... 100 for the indices of num_agents and num_features
    # -1 ... 1   for all agent attributes
    #  0 ... 1   for 1 for the adjacency vector
    return spaces.Box(
      low=np.concatenate((
        np.zeros(3),
        np.full(self.agent_limit * self.feature_len, -1),
        np.zeros(self.agent_limit ** 2))),
      high=np.concatenate((
        np.array([100, 100, 100]), 
        np.ones(self._len_state - 3)
      )))

  @property
  def _len_state(self):
    return 3 + (self.agent_limit * self.feature_len) + (self.agent_limit ** 2)