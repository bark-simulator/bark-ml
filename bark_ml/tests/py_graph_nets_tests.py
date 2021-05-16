# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np
from graph_nets import utils_tf
from bark.runtime.commons.parameters import ParameterServer

from bark_ml.observers.graph_observer import GraphObserver


class PyGraphNetsTests(unittest.TestCase):
  def test_gnn(self):
    # Node features for graph 0.
    nodes_0 = [[10.1, 20., 30.],  # Node 0
              [11., 21., 31.],  # Node 1
              [12., 22., 32.],  # Node 2
              [13., 23., 33.],  # Node 3
              [14., 24., 34.]]  # Node 4

    # Edge features for graph 0.
    edges_0 = [[100., 200.],  # Edge 0
              [101.2, 201.],  # Edge 1
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
      "globals": [],
      "nodes": nodes_0,
      "edges": edges_0,
      "senders": senders_0,
      "receivers": receivers_0
    }

    input_graph = utils_tf.data_dicts_to_graphs_tuple(
      [data_dict_0, data_dict_0])

    num_nodes = len(nodes_0)
    num_features = 3
    num_edge_features = len(edges_0)
    graph_dims = (num_nodes, num_features, num_edge_features)

    # 6 edges x 2
    # 5 nodes x 3
    # adj matrix 5x5
    obs = np.zeros(shape=(1, 52))
    # NOTE: use dense
    params = ParameterServer()
    graph_observer = GraphObserver(params)
    graph_observer.feature_len = 2
    graph_observer.edge_feature_len = 3

    _, _, _ = graph_observer.graph(obs, graph_dims=graph_dims)

    print(input_graph)
    # gnn = MLPGraphNetwork()
    # input_graph_1 = GraphsTuple(
    #   nodes=tf.convert_to_tensor(nodes_0, dtype=tf.float32),
    #   edges=tf.convert_to_tensor(edges_0, dtype=tf.float32),
    #   globals=tf.cast(tf.tile([[0]], [1, 1]), tf.float32),
    #   receivers=tf.convert_to_tensor(receivers_0, dtype=tf.int32),
    #   senders=tf.convert_to_tensor(senders_0, dtype=tf.int32),
    #   n_node=tf.tile([5], [1]),
    #   n_edge=tf.tile([6], [1]))

    # tf.print(input_graph_1)
    # result = gnn(input_graph_1)
    # tf.print(result)


if __name__ == '__main__':
  unittest.main()