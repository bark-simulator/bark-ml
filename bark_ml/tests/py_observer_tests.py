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
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteML
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, DiscreteHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.observers.graph_observer_v2 import GraphObserverV2
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

    # under test
    observer = GraphObserverV2(params)
    observer.Reset(world)
    
    eval_id = env._scenario._eval_agent_ids[0]
    observed_world = world.Observe([eval_id])[0]
    start_time = time.time()
    observed_state = observer.Observe(observed_world)
    # print(observed_state)
    end_time = time.time()
    print(f"It took {end_time-start_time} seconds.")
    
    # non-batch
    ob = GraphObserverV2.graph(observed_state)
    # print(ob)
    # batch
    batch_observation = tf.concat([observed_state, observed_state], axis=0)
    node_vals, edge_indices, node_lens, edge_lens, globals, edge_vals = GraphObserverV2.graph(batch_observation)
    
    data_dict_0 = {
      "globals": globals,
      "nodes": node_vals,
      "edges": edge_indices,
      "senders": edge_indices[:, 0],
      "receivers": edge_indices[:, 1]
    }
    
    data_dict_list = [data_dict_0]
    graphs_tuple = utils_np.data_dicts_to_graphs_tuple(data_dict_list)
    print(graphs_tuple)
    
    
if __name__ == '__main__':
  unittest.main()