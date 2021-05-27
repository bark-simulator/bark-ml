# Copyright (c) 2020 fortiss GmbH
#
# Authors: Marco Oliva, Silvan Wimmer
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import logging
import tensorflow as tf
from gym import spaces
from typing import Dict

from bark.core.models.dynamic import StateDefinition
from bark.core.geometry import Distance, Point2d
from bark.runtime.commons.parameters import ParameterServer

from bark_ml.observers.observer import BaseObserver

class GraphObserver(BaseObserver):
  """Graph observer

  This observer converts an `ObservsedWorld` instance into
  a graph-structured observation, consisting of nodes (with
  features), an adjacency matrix expressing the (directed)
  connections between nodes in the graph, and edges (with
  features), expressing the relative information between
  the agents that constitute these connections.

  Graph generation used in the paper "Graph Neural Networks and Reinforcement
  Learning for Behavior Generation in Semantic Environments"
  (https://arxiv.org/abs/2006.12576)
  """

  def __init__(self, params=ParameterServer()):
    """
    Creates an instance of `GraphObserver`.

    Args:
      params: A `ParameterServer` instance that enables further
        configuration of the observer. Defaults to a new instance.
    """
    BaseObserver.__init__(self, params)
    self._logger = logging.getLogger()

    self._normalize_observations = \
      self._params["ML"]["GraphObserver"]["NormalizationEnabled",
      "Whether normalization of features should be performed.",
      True]

    self._num_agents =\
      params["ML"]["GraphObserver"]["AgentLimit",
      "The maximum number of agents that is observed.",
      4]
    self._visibility_radius =\
      params["ML"]["GraphObserver"]["VisibilityRadius",
      "The radius in which agent can 'see', i.e. detect other agents.",
      550]
    self._add_self_loops =\
      params["ML"]["GraphObserver"]["SelfLoops",
      "Whether nodes have self connections.",
      True]
    requested_node_attribute_keys =\
      params["ML"]["GraphObserver"]["EnabledNodeFeatures",
      "The list of available node features, given by their string key that \
       the observer should extract from the world and insert into the \
       observation. For a list of available features, refer to the list \
       returned by `GraphObserver.available_node_attributes`.",
      self.available_node_attributes()
    ]
    self._enabled_node_attribute_keys = self._filter_requested_attributes(
      requested_keys=requested_node_attribute_keys,
      available_keys=self.available_node_attributes(),
      context="node")
    self._logger.info(
      "GraphObserver configured with node attributes: " +
      f"{self._enabled_node_attribute_keys}")
    requested_edge_attribute_keys =\
      params["ML"]["GraphObserver"]["EnabledEdgeFeatures",
      "The list of available edge features, given by their string key that \
       the observer should extract from the world and insert into the \
       observation. For a list of available features, refer to the list \
       returned by `GraphObserver.available_edge_attributes`.",
      self.available_edge_attributes()
    ]
    self._enabled_edge_attribute_keys = self._filter_requested_attributes(
      requested_keys=requested_edge_attribute_keys,
      available_keys=self.available_edge_attributes(),
      context="edge")
    self._logger.info(
      "GraphObserver configured with edge attributes: " +
      f"{self._enabled_edge_attribute_keys}")

    # the number of features of a node in the graph
    self.feature_len = len(self._enabled_node_attribute_keys)
    # the number of features of an edge between two nodes
    self.edge_feature_len = len(self._enabled_edge_attribute_keys)

  def Observe(self, observed_world):
    """See base class."""
    agents = self._preprocess_agents(observed_world)

    # placeholder for the output observation
    obs = np.zeros(self._len_state)

    # insert node features for each agent
    for i, agent in agents:
      start_index = i * self.feature_len
      end_index = start_index + self.feature_len
      obs[start_index:end_index] = self._extract_node_features(agent)

    edge_features = np.zeros((self._num_agents, self._num_agents, self.edge_feature_len))
    adjacency_matrix = np.zeros((self._num_agents, self._num_agents))

    # add edges to all visible agents
    for index, agent in agents:
      nearby_agents = self._nearby_agents(
        center_agent=agent,
        agents=agents,
        radius=self._visibility_radius)

      if self._add_self_loops:
        adjacency_matrix[index, index] = 1

      for target_index, nearby_agent in nearby_agents:
        edge_features[index, target_index, :] = self._extract_edge_features(agent, nearby_agent)
        adjacency_matrix[index, target_index] = 1
        adjacency_matrix[target_index, index] = 1

    # insert adjacency list
    adj_start_index = self._num_agents * self.feature_len
    adj_end_index = adj_start_index + self._num_agents ** 2
    obs[adj_start_index:adj_end_index] = adjacency_matrix.reshape(-1)

    # insert edge features
    obs[adj_end_index:] = edge_features.reshape(-1)

    return tf.convert_to_tensor(obs, dtype=tf.float32, name='observation')

  @classmethod
  def graph(cls, observations, graph_dims, dense=False):
    """
    Maps the given batch of observations into a
    graph representation.

    Args:
    observations: The batch of observations as a tf.Tensor
      of shape (batch_size, observation_size).
    graph_dims: A tuple containing the dimensions of the
      graph as (num_nodes, num_features, num_edge_features).
      If `dense` is set to True, num_edge_features is ignored.
    dense: Specifies the format of the returned graph representation.
      If set to `True`, the edges are returned as a list of pairs
      of nodes indices (relative to the flattened batch) and an additional
      mapping of each node to a graph is returned. If set to
      False (default), edges are returned as sparse adjacency
      matrix and an edge feature matrix is additionally returned.

    Returns:
      X: Node features of (batch_size, num_nodes, num_features)

      If `dense` is True:
      A: Dense representation of edges as a list of node index pairs,
        shape (num_total_edges, 2).
      node_to_graph_map: A 1d list, where each element is the mapping
      of the node (indexed relative to the batch) to a graph.

      If `dense`is False:
      A: Spare binary adjacency matrix of shape
        (batch_size num_nodes, num_nodes).
      E: Edge features of shape
        (batch_size, num_nodes, num_edge_features, num_edge_features).
    """
    obs = observations # for brevity

    if not tf.is_tensor(obs):
      obs = tf.convert_to_tensor(obs)

    n_nodes, n_features = graph_dims[0:2]
    batch_size = tf.shape(observations)[0]

    # extract node features F
    F = tf.reshape(obs[:, :n_nodes * n_features], [batch_size, n_nodes, n_features])

    # extract adjacency matrix A
    adj_start_idx = n_nodes * n_features
    adj_end_idx = adj_start_idx + n_nodes ** 2
    A = tf.reshape(obs[:, adj_start_idx:adj_end_idx], [batch_size, n_nodes, n_nodes])


    if dense:
      # in dense mode, the nodes of all graphs are
      # concatenated into one list that contains all
      # feature vectors of the whole batch.
      # the assignment to graphs is handled by the
      # node_to_graph_map.
      F = tf.reshape(F, [batch_size * n_nodes, n_features])

      # find non-zero elements in the adjacency matrix (edges)
      # and collect their indices
      A = tf.where(tf.greater(A, 0))

      # we need the indices of the source and target nodes to
      # be represented as their indices in the whole batch,
      # in other words: each node index must be the index
      # of the graph in the batch plus the index of the node
      # in the graph. E.g. if each graph has 5 nodes, the
      # node indices are: graph 0: 0-4, graph 1: 5-9, etc.

      # compute a tensor where each element
      # is the graph index of the node that is represented
      # by the same index in A
      graph_indices = tf.reshape(A[:, 0], [1, -1])
      graph_indices = tf.scalar_mul(n_nodes, graph_indices)
      graph_indices = tf.transpose(tf.tile(graph_indices, [2, 1]))

      A = A[:, 1:] + graph_indices
      A = tf.cast(A, tf.int32)

      # since the node feature vectors are all stored in a 2D array,
      # we need to map them to the graphs that they belong to.

      # construct a mapping from the node feature vectors in F
      # to the graphs in the batch, i.e a list where each element
      # contains the index of the graph that the feature vector at
      # the same index in F belongs to, e.g. if each graph contains
      # two nodes, then the node_to_graph_map is [0, 0, 1, 1, 2, 2, ...]
      node_to_graph_map = tf.reshape(tf.range(batch_size), [-1, 1])
      node_to_graph_map = tf.tile(node_to_graph_map, [1, n_nodes])
      node_to_graph_map = tf.reshape(node_to_graph_map, [-1])

      # extract edge features
      n_edge_features = graph_dims[2]
      E_shape = [-1, n_edge_features]
      E = tf.reshape(obs[:, adj_end_idx:], E_shape)
      return F, A, node_to_graph_map, E

    # extract edge features
    n_edge_features = graph_dims[2]
    E_shape = [batch_size, n_nodes, n_nodes, n_edge_features]
    E = tf.reshape(obs[:, adj_end_idx:], E_shape)
    return F, A, E

  def _preprocess_agents(self, world):
    """
    Preproccesses the agents for constructing an
    observation by resorting.

    Args:
      world: A `bark.core.ObservedWorld` instance.

    Returns:
      A list of tuples, consisting of an index and an
      agent object element.

      The first element always represents the ego agent.
      The remaining elements resemble other agents, up
      to the limit defined by `self._num_agents`, sorted
      in ascending order with respect to the agents'
      distance in the world to the ego agent.
    """
    ego_agent = world.ego_agent
    agents = list(world.agents.values())
    agents.remove(ego_agent)
    agents = self._agents_sorted_by_distance(ego_agent, agents)
    agents.insert(0, ego_agent)
    return list(enumerate(agents))[:self._num_agents]

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

  def _nearby_agents(self, center_agent, agents, radius: float):
    """
    Returns all elements from 'agents' whose position, defined
    in x and y coordinates is within the specified `radius` of
    the 'center_agent's position.
    """
    center_agent_pos = self._position(center_agent)
    other_agents = filter(lambda a: a[1].id != center_agent.id, agents)
    nearby_agents = []
    for index, agent in other_agents:
      agent_pos = self._position(agent)
      distance = Distance(center_agent_pos, agent_pos)
      if distance <= radius:
        nearby_agents.append((index, agent))
    return nearby_agents

  def _extract_node_features(self, agent, as_dict=False):
    res = {}

    # Init data (to keep ordering always equal for reading and writing)
    for label in self._enabled_node_attribute_keys:
      res[label] = "inf"

    n = self.normalization_data
    def add_feature(key, value, norm_range):
      if self._normalize_observations:
        value = self._normalize_value(value, norm_range)
      res[key] = value

    try:
      state = agent.state
      if "x" in res:
        add_feature("x", state[int(StateDefinition.X_POSITION)], n["x"])
      if "y" in res:
        add_feature("y", state[int(StateDefinition.Y_POSITION)], n["y"])
      if "theta" in res:
        add_feature("theta", state[int(StateDefinition.THETA_POSITION)], n["theta"])
      if "vel" in res:
        add_feature("vel", state[int(StateDefinition.VEL_POSITION)], n["vel"])

      # get information related to goal
      goal_center = agent.goal_definition.goal_shape.center[0:2]
      goal_dx = goal_center[0] - state[int(StateDefinition.X_POSITION)]
      goal_dy = goal_center[1] - state[int(StateDefinition.Y_POSITION)]

      if "goal_x" in res:
        add_feature("goal_x", goal_center[0], n["x"])
      if "goal_y" in res:
        add_feature("goal_y", goal_center[1], n["y"])
      if "goal_dx" in res:
        add_feature("goal_dx", goal_dx, n["dx"])
      if "goal_dy" in res:
        add_feature("goal_dy", goal_dy, n["dy"])
      if "goal_theta" in res:
        add_feature("goal_theta", np.arctan2(goal_dy, goal_dx), n["theta"])
      if "goal_d" in res:
        add_feature("goal_d", np.sqrt(goal_dx**2 + goal_dy**2), n["distance"])
      if "goal_vel" in res:
        goal_velocity = np.mean(agent.goal_definition.velocity_range)
        add_feature("goal_vel", goal_velocity, n["vel"])
    except:
      raise AttributeError(
        "A problem occured during node feature extraction. Possibly " +
        "a node feature that's specifed in the 'EnabledNodeFeatures' " +
        "parameter is not supported by the current BARK-ML environment.")

    # remove disabled attributes
    res = {key: res[key] for key in self._enabled_node_attribute_keys}
    assert list(res.keys()) == self._enabled_node_attribute_keys

    if not as_dict:
      res = list(res.values())

    return res

  def _extract_edge_features(self, source_agent, target_agent):
    """
    Encodes the relation between the given agents as the
    features of a directed edge in the graph.

    Args:
      source_agent: the agent that defines the source of the edge.
      target_agent: the agent that defines the target of the edge.

    Returns:
      An np.array containing the differences in the agents'
      x and y position, velocities and orientations.
    """
    source_features =\
      self._extract_node_features(source_agent, as_dict=True)
    target_features =\
      self._extract_node_features(target_agent, as_dict=True)

    features = {
      "dx": source_features["x"] - target_features["x"],
      "dy": source_features["y"] - target_features["y"],
      "dvel": source_features["vel"] - target_features["vel"],
      "dtheta": source_features["theta"] - target_features["theta"]
    }

    features = {key: features[key] for key in self._enabled_edge_attribute_keys}
    features = list(features.values())
    assert len(features) == len(self._enabled_edge_attribute_keys)

    return features

  def _filter_requested_attributes(self,
                                   requested_keys,
                                   available_keys,
                                   context=""):
    """
    Filters the requested node/edge attributes by the available
    attributes and returns the intersection of the requested
    and available attributes.
    """
    if not isinstance(requested_keys, list):
      raise ValueError(
        f"Requested {context} feature list must be a list of " +
        f"strings, not {type(requested_keys)}.")

    if len(requested_keys) == 0:
      self._logger.warning(
        "The `GraphObserver` received an empty list of requested " +
        f"{context} attributes.")
      return []

    valid_keys = []
    invalid_keys = []

    for key in requested_keys:
      if key in available_keys:
        valid_keys.append(key)
      else:
        invalid_keys.append(key)

    if len(invalid_keys) > 0:
      self._logger.warning(
        f"The following {context} attributes requested from the GraphObserver " +
        f"are not supported and will be ignored: {invalid_keys}")

    return valid_keys

  def _normalize_value(self, value, vrange):
    """
    Normalizes the `value` with can take on values in
    into the range of [-1, 1].
    If the `value` is outside the given range, it's clamped
    to the bound of [-1, 1]
    """
    normed = (value - vrange[0]) / (vrange[1] - vrange[0])
    # normed = max(-1, normed) # values lower -1 clipped
    # normed = min(1, normed) # values bigger 1 clipped
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
    x_range = self._world_x_range[1] - self._world_x_range[0]
    y_range = self._world_y_range[1] - self._world_y_range[0]
    max_dist = np.linalg.norm([x_range, y_range])

    d = {}
    d['x'] = self._world_x_range
    d['y'] = self._world_y_range
    d['theta'] = self._theta_range
    d['vel'] = self._velocity_range
    d['distance'] = [0, max_dist]
    d['dx'] = [-x_range, x_range]
    d['dy'] = [-y_range, y_range]
    return d

  def sample(self):
    raise NotImplementedError

  @classmethod
  def available_node_attributes(cls, with_descriptions=False):
    attributes = {
      "x": "The x-components of the agent's position.",
      "y": "The y-components of the agent's position.",
      "theta": "The current heading angle of tha agent.",
      "vel": "The current velocity of the agent.",
      "goal_x": "The x-component of the goal's position.",
      "goal_y": "The y-component of the goal's position.",
      "goal_dx": "The difference in the x-component of the agent's and the goal's position.",
      "goal_dy": "The difference in the y-component of the agent's and the goal's position.",
      "goal_theta": "The goal heading angle.",
      "goal_d": "The euclidian distance of the agent to the goal.",
      "goal_vel": "The goal velocity."
    }

    if not with_descriptions:
      attributes = list(attributes.keys())

    return attributes

  @classmethod
  def available_edge_attributes(cls, with_descriptions=False):
    attributes = {
      "dx": "The difference in the x-position of the two agents.",
      "dy": "The difference in the y-position of the two agents.",
      "dvel": "The difference in the velocity of the two agents.",
      "dtheta": "The difference in the heading angle of the two agents."
    }

    if not with_descriptions:
      attributes = list(attributes.keys())

    return attributes

  @property
  def observation_space(self):
    # 0 ... 1   for all node attributes
    # 0 ... 1   for the adjacency list
    # 0 ... 1   for the edge attributes
    return spaces.Box(
      low=np.concatenate((
        np.zeros(self._num_agents*self.feature_len),
        np.zeros(self._num_agents**2),
        np.zeros(self._num_agents**2*self.edge_feature_len))),
      high=np.ones(self._len_state))

  @property
  def _len_state(self):
    len_node_features = self._num_agents*self.feature_len
    len_adjacency = self._num_agents**2
    len_edge_features = len_adjacency*self.edge_feature_len
    return len_node_features + len_adjacency + len_edge_features

  @property
  def graph_dimensions(self):
    """
    Returns a three-element tuple that defines the
    dimensions of the graph as returned by the
    `Observe` method, consisting of
    - the number of nodes
    - the number of node features
    - the number of edge features
    """
    return (
      self._num_agents,
      self.feature_len,
      self.edge_feature_len
    )