# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

#import os, time, json, pickle
import numpy as np
import unittest
from tf_agents.environments import tf_py_environment
import tensorflow as tf
import networkx as nx

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer

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
    assert tf.is_tensor(observed_state)
    assert (observed_state.dtype == tf.float32) or (observed_state.dtype == tf.float64)
    assert observed_state.shape == (predicted_len,)


  def test_reconstruction_to_graph(self):
    # Check reconstruction to an OrderedGraph
    observed_state = self.observed_state
    observation = self.observation
    reconstructed_graph = self.reconstructed_graph
    n_nodes = self.metadata["n_nodes"]

    # Check if parameter of graph are correct
    assert reconstructed_graph.__class__ == nx.OrderedGraph
    assert len(reconstructed_graph.nodes) == n_nodes

  def test_reconstruction_to_observation(self):
    # Get graph at first
    observed_state = self.observed_state
    observation = self.observation
    reconstructed_graph = self.reconstructed_graph

    # Reconstruct the observation
    rec_obs = self.observer._observation_from_graph(reconstructed_graph)
    assert (observation == rec_obs).all()

  def test_norming(self):
    observer = GraphObserver()
    range_ = [-10, 10]
    eps = 1*10-6

    assert abs(1 - observer._normalize_value(10, range=range_))< eps
    assert abs(-1 - observer._normalize_value(-10, range=range_)) < eps
    assert abs(1 - observer._normalize_value(100, range=range_)) < eps
    assert abs(0.1 - observer._normalize_value(1, range=range_)) < eps
        
if __name__ == '__main__':
  unittest.main()