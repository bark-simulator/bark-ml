# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

#import os, time, json, pickle
import numpy as np
import unittest
import tensorflow as tf
import networkx as nx

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint

class PyGraphObserverTests(unittest.TestCase):
  def setUp(self):
    """Setting up the test-case"""
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params, random_seed=0)
    self.env = SingleAgentRuntime(blueprint=bp, render=True)
    self.env.reset()
    self.world = self.env._world
    self.observer = GraphObserver(params)
    self.eval_id = self.env._scenario._eval_agent_ids[0]

    # Get observation from observer
    observed_world = self.world.Observe([self.eval_id])[0]
    observed_state = self.observer.Observe(observed_world)
    self.observed_state = observed_state

    # Reconstruct the graph
    reconstructed_graph = GraphObserver.graph_from_observation(observed_state)
    self.reconstructed_graph = reconstructed_graph

    # Convert to numpy array and split into subparts for easier handling
    observation = observed_state.numpy()
    self.observation = observation
    n_nodes_max = int(observation[0])
    n_nodes = int(observation[1])
    n_features = int(observation[2])
    # Preprocess metadata and data for later comparison
    metadata = dict()
    metadata["n_nodes_max"] = n_nodes_max
    metadata["n_nodes"] = n_nodes
    metadata["n_features"] = n_features
    metadata["predicted_len"] = 3 + n_nodes_max*n_features + n_nodes_max**2
    self.metadata = metadata
    
    data = dict()
    data["node_features"] = observation[3:n_nodes_max*n_features+3]
    data["adj_matrix"] = observation[3+n_nodes_max*n_features:].reshape((n_nodes_max,-1))
    self.data = data
  
  def test_observation_formalities(self):
    # Convert to numpy array and split into subparts for easier handling
    observed_state = self.observed_state
    predicted_len = self.metadata["predicted_len"]    

    # Check if formal parameter of observed_state are correct
    self.assertTrue(tf.is_tensor(observed_state))
    #assert tf.is_tensor(observed_state)
    self.assertTrue((observed_state.dtype == tf.float32) or (observed_state.dtype == tf.float64))
    #assert (observed_state.dtype == tf.float32) or (observed_state.dtype == tf.float64)
    #assert observed_state.shape == (predicted_len,)
    self.assertEqual(observed_state.shape, (predicted_len,))


  def test_reconstruction_to_graph(self):
    # Check reconstruction to an OrderedGraph
    observed_state = self.observed_state
    observation = self.observation
    reconstructed_graph = self.reconstructed_graph
    n_nodes = self.metadata["n_nodes"]

    # Check if parameter of graph are correct
    self.assertEqual(reconstructed_graph.__class__, nx.OrderedGraph)
    #assert reconstructed_graph.__class__ == nx.OrderedGraph
    self.assertEqual(len(reconstructed_graph.nodes), n_nodes)
    #assert len(reconstructed_graph.nodes) == n_nodes

  def test_reconstruction_to_observation(self):
    # Get graph at first
    observed_state = self.observed_state
    observation = self.observation
    reconstructed_graph = self.reconstructed_graph

    # Reconstruct the observation
    rec_obs = self.observer._observation_from_graph(reconstructed_graph)
    self.assertEqual(observed_state.__class__, rec_obs.__class__)
    self.assertTrue(tf.reduce_all(tf.equal(observed_state, rec_obs)))
    #assert observed_state.__class__ == rec_obs.__class__
    #assert tf.reduce_all(tf.equal(observed_state, rec_obs))

  def test_norming(self):
    observer = GraphObserver()
    range_ = [-10, 10]
    eps = 1*10-6
    self.assertTrue(abs(1 - observer._normalize_value(10, range=range_))< eps)
    self.assertTrue(abs(-1 - observer._normalize_value(-10, range=range_)) < eps)
    self.assertTrue(abs(1 - observer._normalize_value(100, range=range_)) < eps)
    self.assertTrue(abs(0.1 - observer._normalize_value(1, range=range_)) < eps)
        


  def test_parameter_server_usage(self):
    expected_agent_limit = 15
    expected_visibility_radius = 100

    params = ParameterServer()
    params["ML"]["GraphObserver"]["AgentLimit"] = expected_agent_limit
    params["ML"]["GraphObserver"]["VisibilityRadius"] = expected_visibility_radius
    observer = GraphObserver(normalize_observations=True, params=params)

    self.assertEqual(observer._agent_limit, expected_agent_limit)
    self.assertEqual(observer._visibility_radius, expected_visibility_radius)
    #assert observer._normalize_observations #unclear statement

  def test_considered_agents_selection(self):
    agent_limit = 10
    params = ParameterServer()
    params["ML"]["GraphObserver"]["AgentLimit"] = agent_limit
    observer = GraphObserver(params=params)

    obs, obs_world = self._get_observation(
      observer=observer,
      world=self.world,
      eval_id=self.eval_id)

    graph = GraphObserver.graph_from_observation(obs)
    
    num_nodes = len(graph.nodes)
    expected_num_nodes = min(len(obs_world.agents), agent_limit)
    self.assertEqual(num_nodes, expected_num_nodes,
      msg=f'Expected {expected_num_nodes}, got {num_nodes}')
    
    ego_node = graph.nodes[0]
    ego_node_pos = Point2d(
      ego_node['x'].numpy(), 
      ego_node['y'].numpy())

    # verify that the nodes are ordered by
    # ascending distance to the ego node
    max_distance_to_ego = 0
    for _, attributes in graph.nodes.data():
      pos = Point2d(
        attributes['x'].numpy(), 
        attributes['y'].numpy())
      distance_to_ego = Distance(pos, ego_node_pos)

      self.assertGreaterEqual(distance_to_ego, max_distance_to_ego, 
        msg='Nodes are not sorted by distance to the ego node in ascending order.')
      
      max_distance_to_ego = distance_to_ego

  def test_features_from_observation(self):
    params = ParameterServer()
    params["ML"]["GraphObserver"]["AgentLimit"] = 5
    observer = GraphObserver(params=params)

    observation, _ = self._get_observation(
      observer=observer,
      world=self.world,
      eval_id=self.eval_id)

    graph = GraphObserver.graph_from_observation(observation)
    features, edges = GraphObserver.gnn_input(observation)

    graph_features = []
    for _, attributes in graph.nodes.data():
      a = list(attributes.values())
      b = list(map(lambda x: x.numpy(), a))
      graph_features.append(b)
    
    self.assertTrue(np.array_equal(features, graph_features))
    self.assertTrue(np.array_equal(edges, graph.edges))
    
    
if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(PyGraphObserverTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
