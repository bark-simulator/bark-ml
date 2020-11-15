# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import logging
import tensorflow as tf
from enum import Enum
from graph_nets import modules
from graph_nets import utils_tf
from graph_nets import utils_np
from graph_nets.graphs import GraphsTuple
import sonnet as snt
import tensorflow_addons as tfa

# bark-ml
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.observers.graph_observer_v2 import GraphObserverV2
from bark_ml.library_wrappers.lib_tf_agents.networks.gnn_wrapper import GNNWrapper


def make_mlp(name):
  return tf.keras.Sequential([
    tf.keras.layers.Dense(
      80, activation='relu',
      bias_initializer=tf.constant_initializer(0.001),
      name=name+"_0"),
    tf.keras.layers.Dense(
      80, activation='tanh',
      bias_initializer=tf.constant_initializer(0.001),
      name=name+"_1"),
    tf.keras.layers.LayerNormalization()
  ])



class GSNTWrapper(GNNWrapper):
  """
  Implements a graph lib.
  """

  def __init__(self,
               params=ParameterServer(),
               name='GNST',
               output_dtype=tf.float32):
    """
    Initializes a GSNTWrapper instance.

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
    super(GSNTWrapper, self).__init__(
      params=params,
      name=name,
      output_dtype=output_dtype)
    self._num_message_passing_layers = params["ML"]["GSNT"][
      "NumMessagePassingLayers", "Number of message passing layers", 2]
    self._embedding_size = params["ML"]["GSNT"][
      "EmbeddingSize", "Embedding size of nodes", 40]
    self._layers = []
    self._node_mlp = make_mlp(name+"_node")
    self._edge_mlp = make_mlp(name+"_edge")
    # initialize network & call func
    self._init_network(name)
    self._call_func = self._init_call_func
    
  def _init_network(self, name=None):
    # TODO: this might not be clonable
    self._gnn_core_0 = modules.InteractionNetwork(
      edge_model_fn=lambda: self._node_mlp,
      node_model_fn=lambda: self._edge_mlp)
    # self._gnn_core_1 = modules.InteractionNetwork(
    #   edge_model_fn=make_mlp,
    #   node_model_fn=make_mlp)
  
  @tf.function
  def _init_call_func(self, observations, training=False):
    """Graph nets implementation"""
    
    node_vals, edge_indices, node_to_graph, edge_vals = GraphObserver.graph(
      observations=observations, 
      graph_dims=self._graph_dims,
      dense=True)
    batch_size = tf.shape(observations)[0]
    _, _, node_counts = tf.unique_with_counts(node_to_graph)
    edge_counts = tf.math.square(node_counts)

    # reshape node vals and edge vals
    # TODO: change
    # node_vals = tf.reshape(node_vals, [-1, 1, 7])
    # edge_vals = tf.reshape(edge_vals, [-1, 1, 4])
    # tf.print(tf.shape(node_vals))
    
    # tf.print(edge_indices)
    input_graph = GraphsTuple(
      nodes=tf.cast(node_vals, tf.float32),
      edges=tf.cast(edge_vals, tf.float32),
      globals=tf.tile([[0.]], [batch_size, 1]),
      receivers=tf.cast(edge_indices[:, 1], tf.int32),
      senders=tf.cast(edge_indices[:, 0], tf.int32),
      n_node=node_counts,
      n_edge=edge_counts) 
    
    latent = self._gnn_core_0(input_graph)
    # latent = self._gnn_core_1(latent)
    
    node_values = tf.reshape(latent.nodes, [batch_size, -1, self._embedding_size])
    return node_values

