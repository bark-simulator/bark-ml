# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import logging
import tensorflow as tf
from enum import Enum
from spektral.layers import GraphConv

# bark-ml
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.library_wrappers.lib_tf_agents.networks.gnn_wrapper import GNNWrapper


class GCNWrapper(GNNWrapper):
  """
  Implements a GCN.
  """

  def __init__(self,
               params=ParameterServer(),
               graph_dims=None,
               name='GNN',
               output_dtype=tf.float32):
    """
    Initializes a GNNWrapper instance.

    Args:
    params: A `ParameterServer` instance containing the parameters
      to configure the GNN.
    graph_dims: A tuple containing the three elements
      (num_nodes, len_node_features, len_edge_features) of the input graph.
      Needed to properly convert observations back into a graph structure 
      that can be processed by the GNN.
    name: Name of the instance.
    output_dtype: The dtype to which the GNN output is casted.
    """
    self._num_message_passing_layers = params["GCN"][
      "NumMessagePassingLayers", "Number of message passing layers", 3]
    self._embedding_size = params["GCN"][
      "EmbeddingSize", "Embedding size of nodes", 128]
    self._activation_func = params["GCN"][
      "Activation", "Activation function", "relu"]
    self._layers = []
    super(GCNWrapper, self).__init__(
      params=params,
      graph_dims=graph_dims,
      name=name,
      output_dtype=output_dtype)
    
  def _init_network(self):
    for _ in range(self._num_message_passing_layers):
      layer = spektral.layers.GraphConv(
        self._embedding_size, activation=self._activation_func)
      self._layers.append(layer)

  @tf.function
  def _init_call_func(self, observations, training=False):
    embeddings, adj_matrix, edge_features = GraphObserver.graph(
      observations=observations, 
      graph_dims=self._graph_dims)
    for layer in self._layers: 
      embeddings = layer([embeddings, adj_matrix])
    return embeddings

