# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np
import os
import matplotlib
import time
import tensorflow as tf

# Bark imports
from bark.runtime.commons.parameters import ParameterServer

# Bark-ml imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, DiscreteHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.observers.graph_observer_v2 import GraphObserverV2
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.core.observers import NearestObserver

from graph_nets import utils_np


class PyObserverTests(unittest.TestCase):

  def test_nearest_observer(self):
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params)
    env = SingleAgentRuntime(blueprint=bp, render=True)
    env.reset()
    world = env._world

    # under test
    observer = NearestAgentsObserver(params)

    eval_id = env._scenario._eval_agent_ids[0]
    observed_world = world.Observe([eval_id])[0]
    start_time = time.time()
    observed_state = observer.Observe(observed_world)
    end_time = time.time()
    print(f"It took {end_time-start_time} seconds.")
    print(observed_state, observer.observation_space.shape)
    
  def test_nearest_observer_cpp(self):
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params)
    env = SingleAgentRuntime(blueprint=bp, render=True)
    env.reset()
    world = env._world

    # under test
    observer = NearestObserver(params)
    observer.Reset(world)
    
    eval_id = env._scenario._eval_agent_ids[0]
    observed_world = world.Observe([eval_id])[0]
    start_time = time.time()
    observed_state = observer.Observe(observed_world)
    end_time = time.time()
    print(f"It took {end_time-start_time} seconds.")
    print(observed_state, observer.observation_space.shape)


  def test_graph_observer_v2(self):
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params)
    env = SingleAgentRuntime(blueprint=bp, render=True)
    env.reset()
    world = env._world
    params["ML"]["GraphObserver"]["AgentLimit"] = 5
    
    # observers
    observer = GraphObserver(params)
    reference_observer = GraphObserverV2(params)
    reference_observer.Reset(world)
    observer.Reset(world)
    
    
    eval_id = env._scenario._eval_agent_ids[0]
    observed_world = world.Observe([eval_id])[0]
    start_time = time.time()
    observed_state = observer.Observe(observed_world)
    observed_ref_state = reference_observer.Observe(observed_world)
    end_time = time.time()
    print(f"It took {end_time-start_time} seconds.")

    # batch
    batch_observation = tf.stack([observed_state, observed_state], axis=0)
    graph = GraphObserver.graph(
      batch_observation, graph_dims=observer.graph_dimensions, dense=True)
    
    batch_observation_reference = tf.stack(
      [observed_ref_state, observed_ref_state], axis=0)
    graph_ref = GraphObserverV2.graph(
      batch_observation_reference, dense=True)
    
    # unpack values
    ref_node_vals, ref_edge_indices, ref_node_lens, ref_edge_lens, ref_globals, ref_edge_vals = graph_ref
    node_vals, edge_indices, _, edge_vals = graph
    
    # TODO: assert this
    print(ref_node_vals, node_vals)
    # print(ref_edge_vals, edge_vals)
    print(ref_edge_indices, ref_edge_vals)
    print(edge_indices, edge_vals)


if __name__ == '__main__':
  unittest.main()