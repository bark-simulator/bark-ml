# Copyright (c) 2020 fortiss GmbH
#
# Authors: Marco Oliva, Silvan Wimmer
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import unittest
import tensorflow as tf

# BARK imports
from bark.runtime.commons.parameters import ParameterServer
from bark.core.geometry import Distance, Point2d

# BARK-ML imports
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint

class PyGraphObserverTests(unittest.TestCase):
  """Observer tests"""

  def _get_observation(self, observer, world, eval_id):
    observed_world = world.Observe([eval_id])[0]
    observation = observer.Observe(observed_world)
    return observation, observed_world

  def setUp(self):
    """Setting up the test-case."""
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params, random_seed=0)
    self.env = SingleAgentRuntime(blueprint=bp, render=False)
    self.env.reset()
    self.world = self.env._world
    self.observer = GraphObserver(params)
    self.eval_id = self.env._scenario._eval_agent_ids[0]


  def test_parameter_server_usage(self):
    expected_num_agents = 15
    expected_visibility_radius = 100

    params = ParameterServer()
    params["ML"]["GraphObserver"]["AgentLimit"] = expected_num_agents
    params["ML"]["GraphObserver"]["VisibilityRadius"] = expected_visibility_radius
    params["ML"]["GraphObserver"]["NormalizationEnabled"] = True
    observer = GraphObserver(params=params)

    self.assertEqual(observer._num_agents, expected_num_agents)
    self.assertEqual(observer._visibility_radius, expected_visibility_radius)
    # self.assertTrue(observer._add_self_loops)
    self.assertTrue(observer._normalize_observations)

  def test_request_subset_of_available_node_features(self):
    params = ParameterServer()

    requested_features = GraphObserver.available_node_attributes()[0:5]
    params["ML"]["GraphObserver"]["EnabledNodeFeatures"] = requested_features
    observer = GraphObserver(params=params)

    self.assertEqual(
      observer._enabled_node_attribute_keys,
      requested_features)

  def test_request_subset_of_available_edge_features(self):
    params = ParameterServer()

    requested_features = GraphObserver.available_edge_attributes()[0:2]
    params["ML"]["GraphObserver"]["EnabledEdgeFeatures"] = requested_features
    observer = GraphObserver(params=params)

    self.assertEqual(
      observer._enabled_edge_attribute_keys,
      requested_features)

  def test_request_partially_invalid_node_features(self):
    params = ParameterServer()

    requested_features =\
      GraphObserver.available_node_attributes()[0:5] + ['invalid']
    params["ML"]["GraphObserver"]["EnabledNodeFeatures"] = requested_features
    observer = GraphObserver(params=params)

    # remove invalid feature from expected list
    requested_features.pop(-1)

    self.assertEqual(
      observer._enabled_node_attribute_keys,
      requested_features)

  def test_request_partially_invalid_edge_features(self):
    params = ParameterServer()

    requested_features =\
      GraphObserver.available_edge_attributes()[0:2] + ['invalid']
    params["ML"]["GraphObserver"]["EnabledEdgeFeatures"] = requested_features
    observer = GraphObserver(params=params)

    # remove invalid feature from expected list
    requested_features.pop(-1)

    self.assertEqual(
      observer._enabled_edge_attribute_keys,
      requested_features)

  def test_observe_with_self_loops(self):
    num_agents = 4
    params = ParameterServer()
    params["ML"]["GraphObserver"]["AgentLimit"] = num_agents
    params["ML"]["GraphObserver"]["SelfLoops"] = True
    observer = GraphObserver(params=params)
    obs, _ = self._get_observation(observer, self.world, self.eval_id)
    obs = tf.expand_dims(obs, 0) # add a batch dimension

    _, adjacency, _ = GraphObserver.graph(obs, graph_dims=observer.graph_dimensions)
    adjacency_list_diagonal = (tf.linalg.tensor_diag_part(adjacency[0]))

    # assert ones on the diagonal of the adjacency matrix
    tf.assert_equal(adjacency_list_diagonal, tf.ones(num_agents))

  def test_observe_without_self_loops(self):
    num_agents = 4
    params = ParameterServer()
    params["ML"]["GraphObserver"]["AgentLimit"] = num_agents
    params["ML"]["GraphObserver"]["SelfLoops"] = False
    observer = GraphObserver(params=params)
    obs, _ = self._get_observation(observer, self.world, self.eval_id)
    obs = tf.expand_dims(obs, 0) # add a batch dimension

    _, adjacency, _ = GraphObserver.graph(obs, graph_dims=observer.graph_dimensions)
    adjacency_list_diagonal = (tf.linalg.tensor_diag_part(adjacency[0]))

    # assert zeros on the diagonal of the adjacency matrix
    tf.assert_equal(adjacency_list_diagonal, tf.zeros(num_agents))

  def test_observation_conforms_to_spec(self):
    """
    Verify that the observation returned by the observer
    is valid with respect to its defined observation space.
    """
    num_agents = 4
    params = ParameterServer()
    params["ML"]["GraphObserver"]["AgentLimit"] = num_agents
    observer = GraphObserver(params=params)
    obs, _ = self._get_observation(observer, self.world, self.eval_id)

    self.assertTrue(observer.observation_space.contains(obs))

    # additionally check that the adjacency list is binary, since
    # this can't be enforced by the observation space currently
    adj_start_idx = num_agents * observer.feature_len
    adj_end_idx = adj_start_idx + num_agents ** 2
    adj_list = obs[adj_start_idx : adj_end_idx]

    for element in adj_list: self.assertIn(element, [0, 1])

  def test_observed_agents_selection(self):
    agent_limit = 10
    params = ParameterServer()
    params["ML"]["GraphObserver"]["AgentLimit"] = agent_limit
    observer = GraphObserver(params=params)

    obs, obs_world = self._get_observation(
      observer=observer,
      world=self.world,
      eval_id=self.eval_id)

    obs = tf.expand_dims(obs, 0) # add a batch dimension

    nodes, _, _ = GraphObserver.graph(obs, graph_dims=observer.graph_dimensions)
    nodes = nodes[0] # remove batch dim

    ego_node = nodes[0]
    ego_node_pos = Point2d(
      ego_node[0].numpy(), # x coordinate
      ego_node[1].numpy()) # y coordinate

    # verify that the nodes are ordered by
    # ascending distance to the ego node
    max_distance_to_ego = 0
    for node in nodes:
      pos = Point2d(
        node[0].numpy(), # x coordinate
        node[1].numpy()) # y coordinate
      distance_to_ego = Distance(pos, ego_node_pos)

      self.assertGreaterEqual(distance_to_ego, max_distance_to_ego,
        msg='Nodes are not sorted by distance relative to '\
          + 'the ego node in ascending order.')

      max_distance_to_ego = distance_to_ego

  def test_observation_to_graph_conversion(self):
    params = ParameterServer()
    params["ML"]["GraphObserver"]["SelfLoops"] = False
    graph_observer = GraphObserver(params=params)

    num_nodes = 5
    num_features = 5
    num_edge_features = 4

    node_features = np.random.random_sample((num_nodes, num_features))
    edge_features = np.random.random_sample((num_nodes, num_nodes, num_edge_features))

    # note that edges are bidirectional, the
    # the matrix is symmetric
    adjacency_list = [
      [0, 1, 1, 1, 0], # 1 connects with 2, 3, 4
      [1, 0, 1, 1, 0], # 2 connects with 3, 4
      [1, 1, 0, 1, 0], # 3 connects with 4
      [1, 1, 1, 0, 0], # 4 has no links
      [0, 0, 0, 0, 0]  # empty slot -> all zeros
    ]

    observation = np.array(node_features)
    observation = np.append(observation, adjacency_list)
    observation = np.append(observation, edge_features)
    observation = observation.reshape(-1)
    observations = np.array([observation, observation])

    self.assertEqual(observations.shape, (2, 150))

    expected_nodes = tf.constant([node_features, node_features])
    expected_edge_features = tf.constant([edge_features, edge_features])

    graph_dims = (num_nodes, num_features, num_edge_features)
    nodes, edges, edge_features = graph_observer.graph(observations, graph_dims)

    self.assertTrue(tf.reduce_all(tf.equal(nodes, expected_nodes)))
    self.assertTrue(tf.reduce_all(tf.equal(edge_features, expected_edge_features)))

    observations = np.array([observation, observation, observation])

    # in dense mode, the nodes of all graphs are in a single list
    expected_nodes = tf.constant([node_features, node_features, node_features])
    expected_nodes = tf.reshape(expected_nodes, [-1, num_features])

    # the edges encoded in the adjacency list above
    expected_dense_edges = tf.constant([
      # graph 1
      [0, 1], [0, 2], [0, 3],
      [1, 0], [1, 2], [1, 3],
      [2, 0], [2, 1], [2, 3],
      [3, 0], [3, 1], [3, 2],
      # graph 2
      [5, 6], [5, 7], [5, 8],
      [6, 5], [6, 7], [6, 8],
      [7, 5], [7, 6], [7, 8],
      [8, 5], [8, 6], [8, 7],
      # graph 3
      [10, 11], [10, 12], [10, 13],
      [11, 10], [11, 12], [11, 13],
      [12, 10], [12, 11], [12, 13],
      [13, 10], [13, 11], [13, 12]
    ], dtype=tf.int32)

    expected_node_to_graph_map = tf.constant([
      0, 0, 0, 0, 0,
      1, 1, 1, 1, 1,
      2, 2, 2, 2, 2
    ])

    observations = tf.convert_to_tensor(observations)
    print(observations)
    nodes, edges, node_to_graph_map, E =\
      GraphObserver.graph(observations, graph_dims, dense=True)

    self.assertTrue(tf.reduce_all(tf.equal(nodes, expected_nodes)))
    self.assertTrue(tf.reduce_all(tf.equal(edges, expected_dense_edges)))
    # self.assertTrue(tf.reduce_all(
    #   tf.equal(node_to_graph_map, expected_node_to_graph_map)))

  def test_agent_pruning(self):
    """
    Verify that the observer correctly handles the case where
    there are less agents in the world than set as the limit.
    tl;dr: check that all entries of the node features,
    adjacency matrix, and edge features not corresponding to
    actually existing agents are zeros.
    """
    num_agents = 25
    params = ParameterServer()
    params["ML"]["GraphObserver"]["AgentLimit"] = num_agents
    observer = GraphObserver(params=params)
    obs, world = self._get_observation(observer, self.world, self.eval_id)
    obs = tf.expand_dims(obs, 0) # add a batch dimension

    nodes, adjacency_matrix, edge_features = GraphObserver.graph(
      observations=obs,
      graph_dims=observer.graph_dimensions)

    self.assertEqual(nodes.shape, [1, num_agents, observer.feature_len])

    expected_num_agents = len(world.agents)

    # nodes that do not represent agents, but are contained
    # to fill up the required observation space.
    expected_n_fill_up_nodes = num_agents - expected_num_agents
    fill_up_nodes = nodes[0, expected_num_agents:]

    self.assertEqual(
      fill_up_nodes.shape,
      [expected_n_fill_up_nodes, observer.feature_len])

    # verify that entries for non-existing agents are all zeros
    self.assertEqual(tf.reduce_sum(fill_up_nodes), 0)

    # the equivalent for edges: verify that for each zero entry
    # in the adjacency matrix, the corresponding edge feature
    # vector is a zero vector of correct length.
    zero_indices = tf.where(tf.equal(adjacency_matrix, 0))
    fill_up_edge_features = tf.gather_nd(edge_features, zero_indices)
    edge_feature_len = observer.graph_dimensions[2]
    zero_edge_feature_vectors = tf.zeros(
      [zero_indices.shape[0], edge_feature_len])

    self.assertTrue(tf.reduce_all(tf.equal(
      fill_up_edge_features,
      zero_edge_feature_vectors)))



if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(PyGraphObserverTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
