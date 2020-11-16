# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import logging
import tensorflow as tf
from enum import Enum
from tf2_gnn.layers import GNN, GNNInput
import sonnet as snt
import tensorflow_addons as tfa

# bark-ml
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.observers.graph_observer_v2 import GraphObserverV2
from bark_ml.library_wrappers.lib_tf_agents.networks.graph_network import GraphNetwork


def make_mlp():
  return tf.keras.Sequential([
    tf.keras.layers.Dense(
      150, activation='relu', kernel_initializer='glorot_uniform',
      bias_initializer=tf.keras.initializers.Constant(value=0.),
      kernel_regularizer='l2', activity_regularizer='l2'),
    tf.keras.layers.Dense(
      150, activation='relu', kernel_initializer='glorot_uniform',
      bias_initializer=tf.keras.initializers.Constant(value=0.),
      kernel_regularizer='l2', activity_regularizer='l2'),
    tf.keras.layers.Dense(
      80, activation='tanh', kernel_initializer='glorot_uniform',
      bias_initializer=tf.keras.initializers.Constant(value=0.),
      kernel_regularizer='l2', activity_regularizer='l2')
  ])
  # return snt.Sequential([
  #     snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True),
  #     snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
  # ])
  # 

class TF2GNNEdgeMLPWrapper(GraphNetwork):
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
    super(TF2GNNEdgeMLPWrapper, self).__init__(
      params=params,
      name=name,
      output_dtype=output_dtype)
    self._num_message_passing_layers = params["ML"]["TF2Wrapper"][
      "NumMessagePassingLayers", "Number of message passing layers", 2]
    self._embedding_size = params["ML"]["TF2Wrapper"][
      "EmbeddingSize", "Embedding size of nodes", 80]
    self._layers = []
    # initialize network & call func
    self._init_network()
    self._call_func = self._init_call_func
    
  def _init_network(self):
    # TODO: to make this work we need to create and store the layers here
    gnn_params = GNN.get_default_hyperparameters()
    self._gnn = GNN(gnn_params)
  
  # @tf.function
  def _init_call_func(self, observations, training=False):
    """Graph nets implementation"""
    batch_size = tf.constant(observations.shape[0])

    embeddings, adj_list, node_to_graph_map = GraphObserver.graph(
      observations=observations, 
      graph_dims=self._graph_dims, 
      dense=True)

    gnn_input = GNNInput(
      node_features=embeddings,
      adjacency_lists=(adj_list,),
      node_to_graph_map=node_to_graph_map,
      num_graphs=batch_size,
    )

    # tf2_gnn outputs a flattened node embeddings vector, so we 
    # reshape it to have the embeddings of each node seperately.
    flat_output = self._gnn(gnn_input, training=training)
    output = tf.reshape(flat_output, [batch_size, -1, self.num_units])

    return output