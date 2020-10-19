# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import logging
import tensorflow as tf
from enum import Enum
from spektral.layers import EdgeConditionedConv

# bark-ml
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.library_wrappers.lib_tf_agents.networks.gnn_wrapper import GNNWrapper


class GEdgeCondWrapper(GNNWrapper):
  """
  Implements a Edge Conditioned Graph Neural Network (ECGNN)
  """

  def __init__(self,
               params=ParameterServer(),
               name='EdgeCond',
               output_dtype=tf.float32):
    """
    Initializes a GNNWrapper instance.

    Args:
    params: A `ParameterServer` instance containing the parameters
      to configure the GNN.
    name: Name of the instance.
    output_dtype: The dtype to which the GNN output is casted.
    """
    super(GEdgeCondWrapper, self).__init__(
      params=params,
      name=name,
      output_dtype=output_dtype)
    self._num_message_passing_layers = params["ML"]["EdgeCond"][
      "NumMessagePassingLayers", "Number of message passing layers", 3]
    self._embedding_size = params["ML"]["EdgeCond"][
      "EmbeddingSize", "Embedding size of nodes", 80]
    self._activation_func = params["ML"]["EdgeCond"][
      "Activation", "Activation function", "relu"]
    self._layers = []
    # initialize network & call func
    self._init_network()
    self._call_func = self._init_call_func

  def _init_network(self):
    for _ in range(self._num_message_passing_layers):
      layer = EdgeConditionedConv(
        channels=self._embedding_size,
        kernel_network=self._params["ML"]["EdgeCond"][
          "EdgeMLPLayers", "edge mpl", [128, 64]],
        activation=self._params["ML"][
          "MPLActivationFunction", "", "relu"])
      self._layers.append(layer)

  @tf.function
  def _init_call_func(self, observations, training=False):
    embeddings, adj_matrix, edge_features = GraphObserver.graph(
      observations=observations, 
      graph_dims=self._graph_dims)
    for layer in self._layers: 
      embeddings = layer([embeddings, adj_matrix, edge_features])
    return embeddings

