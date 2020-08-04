from gym import spaces
import numpy as np
import math
import operator
import tensorflow as tf
from typing import Dict
from itertools import islice
from collections import OrderedDict

from bark.core.models.dynamic import StateDefinition
from bark.core.world import ObservedWorld
from bark.core.geometry import Distance, Point2d
from bark.runtime.commons.parameters import ParameterServer

from bark_ml.observers.observer import StateObserver

class GraphObserver(StateObserver):
  
  def __init__(self,
               normalize_observations=True,
               params=ParameterServer()):
    """
    Creates an instance of `GraphObserver`.

    This observer converts an `ObservsedWorld` instance into 
    a graph-structured observation, consisting of nodes (with 
    features), an adjacency matrix expressing the (directed) 
    connections between nodes in the graph, and edges (with 
    features), expressing the relative information between 
    the agents that constitute these connections.

    Args:
      normalize_observations: A boolean value indicating whether
        the observations returned by the `Observe` function
        should be normalized into the range of [-1, 1].
      params: A `ParameterServer` instance that enables further
        configuration of the observer. Defaults to a new instance.
    """
    StateObserver.__init__(self, params)
    self._normalize_observations = normalize_observations

    # the number of features of a node in the graph
    self.feature_len = len(GraphObserver.node_attribute_keys())

    # the number of features of an edge between two nodes
    self.edge_feature_len = 4

    # the maximum number of agents that can be observed
    self._agent_limit = \
      params["ML"]["GraphObserver"]["AgentLimit", "", 4]

    # the radius an agent can 'see' in meters
    self._visibility_radius = \
      params["ML"]["GraphObserver"]["VisibilityRadius", "", 50]

    # whether each node has an edge pointing to itself 
    self._add_self_loops = \
      params["ML"]["GraphObserver"]["SelfLoops", "", True]

  def Observe(self, world):
    """see base class"""
    agents = self._preprocess_agents(world)
    num_agents = len(agents)
    obs = []

    # append node features
    [obs.extend(self._extract_features(agent)) for _, agent in agents]

    # fill empty spots with zeros (difference between present and max number of agents)
    obs.extend(np.zeros(max(0, self._agent_limit - num_agents) * self.feature_len))

    edge_features = np.zeros((self._agent_limit, self._agent_limit, self.edge_feature_len))
    adjacency_matrix = np.zeros((self._agent_limit, self._agent_limit))

    # add edges to all visible agents
    for index, agent in agents:
      nearby_agents = self._nearby_agents(
        center_agent=agent, 
        agents=agents, 
        radius=self._visibility_radius)

      for target_index, nearby_agent in nearby_agents:
        edge_features[index, target_index, :] = self._extract_edge_features(agent, nearby_agent)

        adjacency_matrix[index, target_index] = 1
        adjacency_matrix[target_index, index] = 1
    
    # if self._add_self_loops:
    #   adjacency_matrix += np.eye(self._agent_limit ** 2)

    adjacency_list = adjacency_matrix.reshape(-1)
    edge_features = edge_features.reshape(-1)
    obs.extend(adjacency_list)
    obs.extend(edge_features)

    # validate the shape of te constructed observation
    assert len(obs) == self._len_state, f'Observation \
      has invalid length ({len(obs)}, expected: {self._len_state})'
    
    return tf.convert_to_tensor(obs, dtype=tf.float32, name='observation')

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
    source_features = self._extract_features(source_agent, with_keys=True)
    target_features = self._extract_features(target_agent, with_keys=True)

    d_x = source_features["x"] - target_features["x"]
    d_y = source_features["y"] - target_features["y"]
    d_vel = source_features["vel"] - target_features["vel"]
    d_theta = source_features["theta"] - target_features["theta"]
    return np.array([d_x, d_y, d_vel, d_theta])

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
      dense: Specifies the returned graph representation. If 
        set to True, the edges are returned as a list of nodes 
        indices (relative to the flattened batch) and an additional
        mapping of each node to a graph is returned. If set to 
        False (default), edges are returned as an adjacency matrix
        and an edge feature matrix is additionally returned.

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
    batch_size = observations.shape[0]
        
    # extract node features F
    F = tf.reshape(obs[:, :n_nodes * n_features], [batch_size, n_nodes, n_features])

    # extract adjacency matrix A
    adj_start_idx = n_nodes * n_features
    adj_end_idx = adj_start_idx + n_nodes ** 2
    A = tf.reshape(obs[:, adj_start_idx:adj_end_idx], [batch_size, n_nodes, n_nodes])

    if dense:
      # in dense mode, the nodes of all graphs are 
      # concatenated in one list of feature vectors and
      # the assignement to graphs is handled by the
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

      # compute a tensor which where each element
      # is the graph index of the node that is represented
      # by the same index A
      graph_indices = tf.reshape(A[:, 0], [1, -1])
      graph_indices = tf.scalar_mul(n_nodes, graph_indices)
      graph_indices = tf.transpose(tf.tile(graph_indices, [2, 1]))
      
      A = A[:,1:] + graph_indices
      A = tf.cast(A, tf.int32)

      # construct a list where each element represents the
      # assignment of a node to a graph via the graph's index
      node_to_graph_map = tf.reshape(tf.range(batch_size), [-1, 1])
      node_to_graph_map = tf.tile(node_to_graph_map, [1, n_nodes])
      node_to_graph_map = tf.reshape(node_to_graph_map, [-1])

      return F, A, node_to_graph_map

    # extract edge features
    n_edge_features = graph_dims[2]
    E_shape = [batch_size, n_nodes, n_nodes, n_edge_features]
    E = tf.reshape(obs[:, adj_end_idx:], E_shape)

    return F, A, E

  def _preprocess_agents(self, world):
    """
    Preproccesses the agents for constructing an
    observation by resorting and pruning the 
    original list of agents.

    Args:
      world: A `bark.core.world` instance.

    Returns:
      A list of tuples, consisting of an index and an 
      agent object element.

      The first element always represents the ego agent.
      The remaining elements resemble other agents, up 
      to the limit defined by `self._agent_limit`, sorted 
      in ascending order with respect to the agents'
      distance in the world to the ego agent.
    """
    ego_agent = world.ego_agent
    agents = list(world.agents.values())
    agents.remove(ego_agent)
    agents = self._agents_sorted_by_distance(ego_agent, agents)
    agents.insert(0, ego_agent)
    return list(enumerate(agents))[:self._agent_limit]

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

    return other_agents

    # TODO: Add this back!
    nearby_agents = []

    for index, agent in other_agents:
      agent_pos = self._position(agent)
      distance = Distance(center_agent_pos, agent_pos)

      if distance <= radius:
        nearby_agents.append((index, agent))

    return nearby_agents

  def _extract_features(self, agent, with_keys=False):
    """Returns dict containing all features of the agent"""
    res = OrderedDict()

    # Init data (to keep ordering always equal for reading and writing)
    for label in self.node_attribute_keys():
      res[label] = "inf"
    
    state = agent.state
    res["x"] = state[int(StateDefinition.X_POSITION)]
    res["y"] = state[int(StateDefinition.Y_POSITION)]
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
    #   If you change the number/names of features,     #
    #   please adapt self.attributes_keys accordingly.  #
    #####################################################
    assert list(res.keys()) == self.node_attribute_keys()

    if not with_keys:
      res = list(res.values())

    return res

  def _normalize_value(self, value, range):
    """
    Normalizes the `value` with can take on values in
    into the range of [-1, 1].
    If the `value` is outside the given range, it's clamped 
    to the bound of [-1, 1]
    """
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

  @classmethod
  def node_attribute_keys(cls):
    """
    The keys corresponding to the value of the feature
    vctor for each node at the corresponding index.
    """
    return ["x", "y", "theta", "vel", "goal_x", "goal_y", 
    "goal_dx", "goal_dy", "goal_theta", "goal_d", "goal_vel"]

  @property
  def observation_space(self):    
    # -1 ... 1   for all node attributes
    #  0 ... 1   for the adjacency list
    # -1 ... 1   for the edge attributes
    return spaces.Box(
      low=np.concatenate((
        np.full(self._agent_limit * self.feature_len, -1),
        np.zeros(self._agent_limit ** 2),
        np.full((self._agent_limit ** 2) * self.edge_feature_len, -1))),
      high=np.ones(self._len_state))

  @property
  def _len_state(self):
    len_node_features = self._agent_limit * self.feature_len
    len_adjacency = self._agent_limit ** 2
    len_edge_features = len_adjacency * self.edge_feature_len
    return len_node_features + len_adjacency + len_edge_features
