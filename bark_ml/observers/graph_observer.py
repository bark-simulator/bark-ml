import os, time, json, pickle
from gym import spaces
import numpy as np
import math
import operator
import networkx as nx
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

    self.feature_len = 6
    self.agent_limit = 5

    self._normalize_observations = normalize_observations
    self._use_edge_attributes = use_edge_attributes

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
      
      # create edges to all other agents
      nearby_agents = self._nearby_agents(agent, agents, self._visible_distance)
      for (nearby_agent_index, _) in nearby_agents:
        graph.add_edge(index, nearby_agent_index)
    
    observation = self._observation_from_graph(graph)
    
    # keep this for now, TODO: move into test
    # reconstructed_graph = GraphObserver.graph_from_observation(observation)
    # rec_obs = self._observation_from_graph(reconstructed_graph)
    # assert observation == rec_obs, "Failure in graph en/decoding!"

    return observation #, actions

  def _observation_from_graph(self, graph):
    obs = np.zeros(self._len_state)
    num_nodes = len(graph.nodes)

    obs[0] = self.agent_limit
    obs[1] = num_nodes
    obs[2] = self.feature_len
    
    # node attributes
    for (node_id, attributes) in graph.nodes.data():
      start_idx = 3 + node_id * self.feature_len
      end_idx = start_idx + self.feature_len
      obs[start_idx:end_idx] = list(attributes.values())

    adj_start_idx = 3 + self.agent_limit * self.feature_len

    # adjacency list
    for source, target in graph.edges:
      edge_idx = adj_start_idx + source * num_nodes + target
      obs[edge_idx] = 1

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
    return list(enumerate(agents))

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
    goal_dx = goal_center[0] - res["x"] # distance to goal in x coord
    goal_dy = goal_center[1] - res["y"] # distance to goal in y coord
    goal_theta = np.arctan2(goal_dy, goal_dx) # theta for straight line to goal
    goal_d = np.sqrt(goal_dx**2 + goal_dy**2) # distance to goal
    res["goal_d"] = goal_d
    res["goal_theta"] = goal_theta

    if self._NormalizationEnabled:
      n = self.normalization_data

      for k in ["x", "y", "theta", "vel"]:
        res[k] = self._normalize_value(res[k], n[k])
      
      res["goal_d"] = self._normalize_value(res["goal_d"], n["distance"])
      res["goal_theta"] = self._normalize_value(res["goal_theta"], n["theta"])
    
    return res

  def _generate_actions(self, features: Dict[str, float]) -> Dict[str, float]:
    actions = OrderedDict()
    steering = features["goal_theta"] - features["theta"]
    actions["steering"] = steering
    
    dt = 2 # time constant in s for planning horizon
    acc = 2. * (features["goal_d"] - features["vel"] * dt) / (dt**2)
    actions["acceleration"] = acc
    return actions

  def _normalize_value(self, value, range):
    return 2 * (value - range[0]) / (range[1] - range[0]) - 1

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
    d['x'] = self._world_x_range
    d['y'] = self._world_y_range
    d['theta'] = self._ThetaRange
    d['vel'] = self._VelocityRange
    d['distance'] = [0, \
      np.linalg.norm([self._world_x_range[1], self._world_y_range[1]])]
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
        np.ones(self._len_state - 2)
      )))

  @property
  def _len_state(self):
    return 2 + (self.agent_limit * self.feature_len) + (self.agent_limit ** 2)