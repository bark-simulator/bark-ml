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
from bark.runtime.commons.parameters import ParameterServer
from bark.core.geometry import Distance, Point2d
from bark.core.models.dynamic import StateDefinition

# BARK-ML imports
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint

class PyGraphObserverTests(unittest.TestCase):

  def _get_observation(self, observer, world, eval_id):
    observed_world = world.Observe([eval_id])[0]
    observation = observer.Observe(observed_world)
    return observation, observed_world

  def _position(self, agent):
    return Point2d(
        agent.state[int(StateDefinition.X_POSITION)],
        agent.state[int(StateDefinition.Y_POSITION)]
      )

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
    self.observed_state, _ = self._get_observation(
      observer=self.observer,
      world=self.world,
      eval_id=self.eval_id)

    # Reconstruct the graph
    reconstructed_graph = GraphObserver\
      .graph_from_observation(self.observed_state)
    self.reconstructed_graph = reconstructed_graph

    # Convert to numpy array and split into subparts for easier handling
    observation = self.observed_state.numpy()
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
    data["adj_matrix"] = observation[3+n_nodes_max*n_features:]\
      .reshape((n_nodes_max,-1))
    self.data = data
  
  def test_observation_formalities(self):
    # Convert to numpy array and split into subparts for easier handling
    observed_state = self.observed_state
    predicted_len = self.metadata["predicted_len"]    

    # Check if formal parameter of observed_state are correct
    assert tf.is_tensor(observed_state)
    assert (observed_state.dtype == tf.float32) \
      or (observed_state.dtype == tf.float64)
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
    assert observed_state.__class__ == rec_obs.__class__
    assert tf.reduce_all(tf.equal(observed_state, rec_obs))

  def test_norming(self):
    observer = GraphObserver()
    range_ = [-10, 10]
    eps = 1*10-6

    assert abs(1 - observer._normalize_value(10, range=range_)) < eps
    assert abs(-1 - observer._normalize_value(-10, range=range_)) < eps
    assert abs(1 - observer._normalize_value(100, range=range_)) < eps
    assert abs(0.1 - observer._normalize_value(1, range=range_)) < eps

  def test_parameter_server_usage(self):
    expected_agent_limit = 15
    expected_visibility_radius = 100

    params = ParameterServer()
    params["ML"]["GraphObserver"]["AgentLimit"] = expected_agent_limit
    params["ML"]["GraphObserver"]["VisibilityRadius"] = expected_visibility_radius
    observer = GraphObserver(normalize_observations=True, params=params)

    assert observer._agent_limit == expected_agent_limit
    assert observer._visibility_radius == expected_visibility_radius
    assert observer._normalize_observations

  def test_considered_agents_selection(self):
    agent_limit = 10
    visibility_radius = 10
    params = ParameterServer()
    params["ML"]["GraphObserver"]["AgentLimit"] = agent_limit
    observer = GraphObserver(params=params)

    obs, obs_world = self._get_observation(
      observer=self.observer,
      world=self.world,
      eval_id=self.eval_id)

    graph = GraphObserver.graph_from_observation(obs)
    
    num_nodes = len(graph.nodes)
    expected_num_nodes = min(len(obs_world.agents), agent_limit)
    assert num_nodes == expected_num_nodes, \
      f'Expected {expected_num_nodes}, got {num_nodes}'
    
    ego_position = self._position(obs_world.ego_agent)

    ego_node = graph.nodes[0]
    ego_node_pos = Point2d(
      ego_node['x'].numpy(), 
      ego_node['y'].numpy())

    # verify that the nodes are ordered by
    # ascending distance to the ego node
    max_distance_to_ego = 0
    for node_id, attributes in graph.nodes.data():
      pos = Point2d(
        attributes['x'].numpy(), 
        attributes['y'].numpy())
      distance_to_ego = Distance(pos, ego_node_pos)

      assert distance_to_ego >= max_distance_to_ego, 'Nodes are \
        not sorted \by distance to the ego node in ascending order.'
      
      max_distance_to_ego = distance_to_ego
  
if __name__ == '__main__':
  unittest.main()