# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import tensorflow as tf
from graph_nets import modules
from graph_nets.graphs import GraphsTuple

# bark-ml
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.library_wrappers.lib_tf_agents.networks.gnns.graph_network import GraphNetwork


def make_mlp(name, layer_size=150, embedding_size=80):
  return tf.keras.Sequential([
    tf.keras.layers.Dense(
      layer_size, activation='relu',
      bias_initializer=tf.constant_initializer(0.001),
      name=name+"_0"),
    tf.keras.layers.Dense(
      embedding_size, activation='relu',
      bias_initializer=tf.constant_initializer(0.001),
      name=name+"_1"),
    # tf.keras.layers.LayerNormalization()
  ])

class InteractionWrapper(GraphNetwork):

  """
  Implements a graph lib.
  """

  def __init__(self,
               params=ParameterServer(),
               name='InteractionNetwork',
               output_dtype=tf.float32):
    """
    Initializes a InteractionNetwork-Wrapper instance.

    Args:
    params: A `ParameterServer` instance containing the parameters
      to configure the GNN.
    name: Name of the instance.
    output_dtype: The dtype to which the GNN output is casted.
    """
    super(InteractionWrapper, self).__init__(
      params=params,
      name=name,
      output_dtype=output_dtype)
    self._num_message_passing_layers = params["ML"]["InteractionNetwork"][
      "NumMessagePassingLayers", "Number of message passing layers", 2]
    self._embedding_size = params["ML"]["InteractionNetwork"][
      "EmbeddingSize", "Embedding size of nodes", 20]

    self._node_mlps = []
    self._edge_mlps = []
    for i in range(0, self._num_message_passing_layers):
      self._node_mlps.append(
        make_mlp(name+"_node_"+str(i), embedding_size=self._embedding_size))
      self._edge_mlps.append(
        make_mlp(name+"_edge_"+str(i), embedding_size=self._embedding_size))

    self._latent_trace = None
    # initialize network & call func
    self._init_network(name)
    self._call_func = self._init_call_func

  def _init_network(self, name=None):
    self._graph_blocks = []
    for i in range(0, self._num_message_passing_layers):
      self._graph_blocks.append(
        modules.InteractionNetwork(
          edge_model_fn=lambda: self._edge_mlps[i],
          node_model_fn=lambda: self._node_mlps[i]))

  # @tf.function
  def _init_call_func(self, observations, training=False):
    """Graph nets implementation."""
    node_vals, edge_indices, node_to_graph, edge_vals = GraphObserver.graph(
      observations=observations,
      graph_dims=self._graph_dims,
      dense=True)
    batch_size = tf.shape(observations)[0]
    node_counts = tf.unique_with_counts(node_to_graph)[2]
    edge_counts = tf.math.square(node_counts)

    input_graph = GraphsTuple(
      nodes=tf.cast(node_vals, tf.float32),
      edges=tf.cast(edge_vals, tf.float32),
      globals=tf.tile([[0.]], [batch_size, 1]),
      receivers=tf.cast(edge_indices[:, 1], tf.int32),
      senders=tf.cast(edge_indices[:, 0], tf.int32),
      n_node=node_counts,
      n_edge=edge_counts)

    self._latent_trace = []
    latent = input_graph
    for gb in self._graph_blocks:
      latent = gb(latent)
      self._latent_trace.append(latent)
    node_values = tf.reshape(latent.nodes, [batch_size, -1, self._embedding_size])
    return node_values

