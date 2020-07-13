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

from bark.core.models.dynamic import StateDefinition
from bark.core.world import ObservedWorld
from bark.core.geometry import Distance, Point2d
from bark.runtime.commons.parameters import ParameterServer

from bark_ml.observers.observer import StateObserver

class GraphObserver(StateObserver):
  
  def __init__(self,
               normalize_observations=True,
               output_supervised_data = False,
               params=ParameterServer()):
    StateObserver.__init__(self, params)

    self._normalize_observations = normalize_observations
    self._output_supervised_data = output_supervised_data

    # the number of features of a node in the graph
    self.feature_len = len(GraphObserver.attribute_keys())

    # the maximum number of agents that can be observed
    self._agent_limit = \
      params["ML"]["GraphObserver"]["AgentLimit", "", 12]

     # the radius an agent can 'see' in meters
    self._visibility_radius = \
      params["ML"]["GraphObserver"]["VisibilityRadius", "", 50]

    self.observe_times = []

  @classmethod
  def attribute_keys(cls):
    return ["x", "y", "theta", "vel", "goal_x", "goal_y", "goal_dx", "goal_dy", "goal_theta", "goal_d",
            "goal_vel"] #, "d_ditch_left", "d_ditch_right"]

  def Observe(self, world):
    """see base class"""
    t0 = time.time()
    agents = self._preprocess_agents(world)
    num_agents = len(agents)
    obs = [self._agent_limit, num_agents, self.feature_len]

    # features
    for _, agent in agents:
      obs.extend(list(self._extract_features(agent).values()))

    # fill empty spots (difference between existing and max agents) with -1
    obs.extend(np.full((self._agent_limit - num_agents) * self.feature_len, -1))

    # edges
    # Second loop for edges necessary
    # -> otherwise order of graph_nodes is disrupted
    edges = []
    for index, agent in agents:
      # create edges to all visible agents
      nearby_agents = self._nearby_agents(
        center_agent=agent, 
        agents=agents, 
        radius=self._visibility_radius)

      for target_index, _ in nearby_agents:
        if (index, target_index) not in edges and\
           (target_index, index) not in edges:
          edges.append((index, target_index))
    
    # build adjacency matrix and convert to list
    adjacency_matrix = np.zeros((self._agent_limit, self._agent_limit))
    for source, target in edges:
      adjacency_matrix[source, target] = 1
    
    adjacency_list = adjacency_matrix.reshape(-1)
    obs.extend(adjacency_list)

    assert len(obs) == self._len_state, f'Observation \
      has invalid length ({len(obs)}, expected: {self._len_state})'
    
    obs = tf.convert_to_tensor(
      obs, 
      dtype=tf.float32, 
      name='observation_v2')

    self.observe_times.append(time.time() - t0)
    return obs

  @classmethod
  def gnn_input(cls, observation):
    node_limit = int(observation[0])
    num_nodes = int(observation[1])
    num_features = int(observation[2])

    obs = observation[3:]

    # features
    features = []
    for node_id in range(num_nodes):
      start_idx = node_id * num_features
      end_idx = start_idx + num_features
      features.append(obs[start_idx:end_idx].numpy())

    # adjacency list
    adj_start_idx = node_limit * num_features
    adj_list = obs[adj_start_idx:]
    adj_matrix = np.reshape(adj_list, (node_limit, -1))
    
    edges = []
    for (source_id, source_edges) in enumerate(adj_matrix):
      for target_id in np.flatnonzero(source_edges):
        edges.append((source_id, target_id))

    return features, edges

  def _preprocess_agents(self, world):
    """ 
    Returns a list of tuples, consisting
    of an index and an agent object element.

    The first element always represents the ego agent.
    The remaining elements resemble other agents, up
    to the limit defined by `self._agent_limit`,
    sorted in ascending order with respect to the agents'
    distance in the world to the ego agent.
    """
    ego_agent = world.ego_agent
    agents = list(world.agents.values())
    agents.remove(ego_agent)
    agents = self._agents_sorted_by_distance(ego_agent, agents)
    agents.insert(0, ego_agent)
    return list(enumerate(agents))[:self._agent_limit]

  def _agents_sorted_by_distance(self, ego_agent, agents):
    def distance(agent):
      return Distance(
        self._position(ego_agent), 
        self._position(agent))
    
    agents.sort(key=distance)
    return agents

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

    # Init data (to keep ordering always equal for reading and writing!!)
    for label in self.attribute_keys():
      res[label] = "inf"
    
    # Update information with real data
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

    # get information related to road
    #agent_posi = self._position(agent)   
    # Get all possible lane corridors
    #agent_lane = agent.road_corridor.GetCurrentLaneCorridor(agent_posi)
    #lane_corridors = agent.road_corridor.GetLeftRightLaneCorridor(agent_posi) # corridors left and right of agents' corridor, None if not existent
    #lanes = [lane_corridors[0], agent_lane, lane_corridors[1]]
    #lanes = list(filter(None, lanes)) #filter out non existing lanes (right or left of agent)

    # Calculate Distance to left and right road bounds
    # road_bound_left = lanes[0].left_boundary
    # d_ditch_left = Distance(road_bound_left, agent_posi)
    # road_bound_right = lanes[-1].right_boundary
    # d_ditch_right = Distance(road_bound_right, agent_posi)
    #res["d_ditch_left"] = d_ditch_left
    #res["d_ditch_right"] = d_ditch_right

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
      #res["d_ditch_left"] = self._normalize_value(res["d_ditch_left"], n["road"])
      #res["d_ditch_right"] = self._normalize_value(res["d_ditch_right"], n["road"])
    
    #####################################################
    #   If you change the number/names of features,     #
    #   please adapt self.attributes_keys accordingly.  #
    #####################################################
    assert list(res.keys()) == self.attribute_keys()

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
    d['road'] = [0, 15] # may need adjustment for if 3 lanes are broader than 15 m
    return d

  def sample(self):
    raise NotImplementedError

  @property
  def observation_space(self):
    #  0 ... 100 for the indices of num_agents and num_features
    # -1 ... 1   for all agent attributes
    #  0 ... 1   for 1 for the adjacency vector
    return spaces.Box(
      low=np.concatenate((
        np.zeros(3),
        np.full(self._agent_limit * self.feature_len, -1),
        np.zeros(self._agent_limit ** 2))),
      high=np.concatenate((
        np.array([100, 100, 100]), 
        np.ones(self._len_state - 3)
      )))

  @property
  def _len_state(self):
    return 3 + (self._agent_limit * self.feature_len) + (self._agent_limit ** 2)
