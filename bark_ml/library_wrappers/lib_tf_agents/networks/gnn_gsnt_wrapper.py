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

NUM_LAYERS = 2  # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE = 16  # Hard-code latent layer sizes for demos.

def make_mlp():
  # return tf.keras.Sequential([
  #   tf.keras.layers.Dense(
  #     150, activation='relu', kernel_initializer='glorot_uniform',
  #     bias_initializer=tf.keras.initializers.Constant(value=0.)),
  #   tf.keras.layers.Dense(
  #     150, activation='relu', kernel_initializer='glorot_uniform',
  #     bias_initializer=tf.keras.initializers.Constant(value=0.)),
  # ])
  return snt.Sequential([
      snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True),
      snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
  ])

  # kernel_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2'

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
      "EmbeddingSize", "Embedding size of nodes", 16]
    self._layers = []
    # initialize network & call func
    self._init_network()
    self._call_func = self._init_call_func
    
  def _init_network(self):
    self._gnn_core_0 = modules.InteractionNetwork(
      edge_model_fn=make_mlp,
      node_model_fn=make_mlp)
    # self._gnn_core_1 = modules.InteractionNetwork(
    #   edge_model_fn=make_mlp,
    #   node_model_fn=make_mlp)
  
  # @tf.function
  def _init_call_func(self, observations, training=False):
    """Graph nets implementation"""
    
    node_vals, edge_indices, node_to_graph, edge_vals = GraphObserver.graph(
      observations=observations, 
      graph_dims=self._graph_dims,
      dense=True)
    batch_size = tf.shape(observations)[0]
    _, _, node_counts = tf.unique_with_counts(node_to_graph)
    edge_counts = tf.math.square(node_counts)

    # tf.print(edge_indices)
    input_graph = GraphsTuple(
      nodes=tf.cast(node_vals, tf.float32),  # validate
      edges=tf.cast(edge_vals, tf.float32),  # validate
      globals=tf.tile([[0.]], [batch_size, 1]),
      receivers=tf.cast(edge_indices[:, 1], tf.int32),  # validate
      senders=tf.cast(edge_indices[:, 0], tf.int32),  # validate
      n_node=node_counts,  # change
      n_edge=edge_counts)  
    
    latent = self._gnn_core_0(input_graph)
    # latent = self._gnn_core_1(latent)
    
    node_values = tf.reshape(latent.nodes, [batch_size, -1, self._embedding_size])
    return node_values

