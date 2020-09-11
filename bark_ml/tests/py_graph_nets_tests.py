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
from graph_nets import utils_tf

from bark_ml.library_wrappers.lib_tf_agents.networks.gnn_wrapper import MLPGraphNetwork, MLPGraphIndependent

class PyGraphNetsTests(unittest.TestCase):
  def test_gnn(self):
    # Global features for graph 0.
    globals_0 = [1., 2., 3.]

    # Node features for graph 0.
    nodes_0 = [[10., 20., 30.],  # Node 0
              [11., 21., 31.],  # Node 1
              [12., 22., 32.],  # Node 2
              [13., 23., 33.],  # Node 3
              [14., 24., 34.]]  # Node 4

    # Edge features for graph 0.
    edges_0 = [[100., 200.],  # Edge 0
              [101., 201.],  # Edge 1
              [102., 202.],  # Edge 2
              [103., 203.],  # Edge 3
              [104., 204.],  # Edge 4
              [105., 205.]]  # Edge 5

    # The sender and receiver nodes associated with each edge for graph 0.
    senders_0 = [0,  # Index of the sender node for edge 0
                1,  # Index of the sender node for edge 1
                1,  # Index of the sender node for edge 2
                2,  # Index of the sender node for edge 3
                2,  # Index of the sender node for edge 4
                3]  # Index of the sender node for edge 5
    receivers_0 = [1,  # Index of the receiver node for edge 0
                  2,  # Index of the receiver node for edge 1
                  3,  # Index of the receiver node for edge 2
                  0,  # Index of the receiver node for edge 3
                  3,  # Index of the receiver node for edge 4
                  4]  # Index of the receiver node for edge 5

    data_dict_0 = {
      "globals": globals_0,
      "nodes": nodes_0,
      "edges": edges_0,
      "senders": senders_0,
      "receivers": receivers_0
    }
    
    input_graph = utils_tf.data_dicts_to_graphs_tuple(
      [data_dict_0, data_dict_0])
    
    tf.print(input_graph, summarize=1000)
    gnn = MLPGraphNetwork()
    result = gnn(input_graph)
    tf.print(result.nodes)


if __name__ == '__main__':
  unittest.main()