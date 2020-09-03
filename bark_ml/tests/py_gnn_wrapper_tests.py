# Copyright (c) 2020 fortiss GmbH
#
# Authors: Marco Oliva, Silvan Wimmer
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import unittest
import os
import time
import tensorflow as tf
import numpy as np

from bark.runtime.commons.parameters import ParameterServer
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.library_wrappers.lib_tf_agents.networks import GNNWrapper
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorGraphSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner

class PyGNNWrapperTests(unittest.TestCase):
  def test_gnn_parameters(self):
    params = ParameterServer()
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["NumMpLayers"] = 4
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["MpLayerNumUnits"] = 64
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["message_calculation_class"] = "gnn_edge_mlp"
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["global_exchange_mode"] = "mean"
    
    gnn_library = GNNWrapper.SupportedLibrary.spektral
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["Library"] = gnn_library

    
    bp = ContinuousHighwayBlueprint(params, number_of_senarios=2500, random_seed=0)
    observer = GraphObserver(params=params)
    env = SingleAgentRuntime(blueprint=bp, observer=observer, render=False)
    sac_agent = BehaviorGraphSACAgent(environment=env, observer=observer, params=params)

    actor_gnn = sac_agent._agent._actor_network._gnn
    critic_gnn = sac_agent._agent._critic_network_1._gnn

    for gnn in [actor_gnn, critic_gnn]:
      self.assertEqual(gnn._params["NumMpLayers"], 4)
      self.assertEqual(gnn._params["MpLayerNumUnits"], 64)
      self.assertEqual(gnn._params["message_calculation_class"], "gnn_edge_mlp")
      self.assertEqual(gnn._params["global_exchange_mode"], "mean")
      self.assertEqual(gnn._params["Library"], gnn_library)


  def _test_graph_dim_validation_accepts_observer_dims(self):
    observer = GraphObserver()
    gnn = GNNWrapper(graph_dims=observer.graph_dimensions)

    # verify no exception is raised and the dims are applied
    self.assertEqual(gnn._graph_dims, observer.graph_dimensions)

  def test_graph_dim_validation_fails_for_invalid_values(self):
    # all these should raise an exception
    invalid_dim_sets = [(-1, 4, 10), (1, 2), (1, 2, 3, 4), [], None]

    for dims in invalid_dim_sets:
      with self.assertRaises(ValueError):
        GNNWrapper(graph_dims=dims)
      
  def test_gnn_library_validation(self):
    params = ParameterServer()
    graph_dims = [5, 5, 5] # some valid mock value

    with self.assertRaises(ValueError):
      params["Library"] = "cool_but_unsupported_lib"
      GNNWrapper(params, graph_dims)

    # assert no exception
    params["Library"] = GNNWrapper.SupportedLibrary.spektral
    GNNWrapper(params, graph_dims)

    # assert no exception
    params["Library"] = GNNWrapper.SupportedLibrary.tf2_gnn
    GNNWrapper(params, graph_dims)

    # assert no exception when using the string value
    params["Library"] = "spektral"
    GNNWrapper(params, graph_dims)

    # assert no exception
    params["Library"] = "tf2_gnn"
    GNNWrapper(params, graph_dims)

if __name__ == '__main__':
  unittest.main()
