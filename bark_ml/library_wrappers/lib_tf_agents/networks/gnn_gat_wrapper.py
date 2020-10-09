# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import logging
import tensorflow as tf
from enum import Enum
from spektral.layers import GraphAttention

# bark-ml
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.library_wrappers.lib_tf_agents.networks.gnn_wrapper import GNNWrapper


class GATWrapper(GNNWrapper):
  """
  Implements a GAT.
  """

  def __init__(self,
               params=ParameterServer(),
               name='GAT',
               output_dtype=tf.float32):
    """
    Initializes a GATWrapper instance.

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
    super(GATWrapper, self).__init__(
      params=params,
      name=name,
      output_dtype=output_dtype)
    self._num_message_passing_layers = params["GAT"][
      "NumMessagePassingLayers", "Number of message passing layers", 2]
    self._embedding_size = params[
      "EmbeddingSize", "Embedding size of nodes", 128]
    self._activation_func = params["GAT"][
      "Activation", "Activation function", "elu"]
    self._num_attn_heads = params["GAT"][
      "NumAttnHeads", "Number of attention heads to be used", 4]
    self._dropout_rate = params[
      "DropoutRate", "", 0.]
    self._layers = []
    # initialize network & call func
    self._init_network()
    self._call_func = self._init_call_func
    
  def _init_network(self):
    for _ in range(self._num_message_passing_layers):
      layer = GraphAttention(
        self._embedding_size,
        attn_heads=self._num_attn_heads,
        dropout_rate=self._dropout_rate,
        activation=self._activation_func)
      self._layers.append(layer)
    layer = GraphAttention(
      self._embedding_size,
      attn_heads=1,
      dropout_rate=self._dropout_rate,
      activation=self._activation_func)
    self._layers.append(layer)

  @tf.function
  def _init_call_func(self, observations, training=False):
    embeddings, adj_matrix, edge_features = GraphObserver.graph(
      observations=observations, 
      graph_dims=self._graph_dims)
    for layer in self._layers: 
      embeddings = layer([embeddings, adj_matrix])
    return embeddings

