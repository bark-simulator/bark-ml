# Copyright (c) 2020 fortiss GmbH
#
# Authors: Marco Oliva
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import logging
import tensorflow as tf

from enum import Enum
from tf2_gnn.layers import GNN, GNNInput
from spektral.layers import EdgeConditionedConv

from bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.graph_observer import GraphObserver

class GNNWrapper(tf.keras.Model):
  """
  Implements a graph neural network.

  This class serves as a wrapper over the specific implementation
  of the configured GNN library.
  """

  class SupportedLibrary(str, Enum):
    """
    Enumeration of the currently supported GNN libraries.
    """
    spektral = "spektral"
    tf2_gnn  = "tf2_gnn"

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
    super(GNNWrapper, self).__init__(name=name)

    self.output_dtype = output_dtype
    self._params = params
    self._graph_dims = self._validated_graph_dims(graph_dims)

    self.num_units = params[
      "MpLayersHiddenDim", 
      "Hidden dim of the message passing layers. This will be the\
       size of the output node embeddings. If using spektral, this\
       specifies the number of channels in the convolution layer.", 
      256]
    
    self._num_message_passing_layers = params[
      "NumMpLayers",
      "The number of message passing layers in the GNN.",
      1]
    
    lib = params[
      "Library", 
      "Either 'spektral' or 'tf2_gnn'. Specifies with which of the two\
       libraries the GNN is initialized. Note that depending on the\
       library, a different set of params may apply.", 
      GNNWrapper.SupportedLibrary.spektral]

    logging.info(
      f'"{name}" configured with `{lib}` for input graphs with ' +
      f'{graph_dims[0]} nodes, {graph_dims[1]} node features, ' + 
      f'and {graph_dims[2]} edge features.')
    
    if lib in [GNNWrapper.SupportedLibrary.spektral, "spektral"]:
      self._init_spektral_layers(params.ConvertToDict())
      self._call_func = self._call_spektral
    elif lib in [GNNWrapper.SupportedLibrary.tf2_gnn, "tf2_gnn"]:
      self._init_tf2_gnn_layers(params.ConvertToDict())
      self._call_func = self._call_tf2_gnn
    else:
      raise ValueError(
        f"Invalid GNN library '{lib}'. Use 'spektral' or 'tf2_gnn'")

  def _validated_graph_dims(self, graph_dims):
    if graph_dims is None:
      raise ValueError('Graph dimensions must not be `None`.')
    if len(graph_dims) != 3:
      raise ValueError('Graph dimensions must be of length 3.')

    int_dims = list(map(int, graph_dims))
    
    if min(int_dims) < 0:
      raise ValueError('Graph dimensions must be positive.')

    return int_dims

  def _init_spektral_layers(self, params):
    self._convolutions = []
    
    for _ in range(self._num_message_passing_layers):
      self._convolutions.append(EdgeConditionedConv(
        channels=self.num_units,
        kernel_network=params.get("EdgeFcLayerParams", [256]),
        activation=params.get("MPLayerActivation", "relu"))
      )

  def _init_tf2_gnn_layers(self, params):
    # map bark-ml parameter keys to tf2_gnn parameter keys
    mapped_params = {}
    mapped_params["hidden_dim"] = self.num_units
    mapped_params["num_layers"] = self._num_message_passing_layers
    mapped_params["global_exchange_mode"] =\
      params.get("global_exchange_mode", "gru")
    mapped_params["message_calculation_class"] =\
      params.get("message_calculation_class", "rgcn")

    mp_style = mapped_params["message_calculation_class"]
    gnn_params = GNN.get_default_hyperparameters(mp_style)
    gnn_params.update(mapped_params)

    self._gnn = GNN(gnn_params)

  @tf.function(experimental_relax_shapes=True)
  # relax shapes due to varying batch sizes
  def call(self, observations, training=False):
    if observations.shape[0] == 0:
      return tf.random.normal(shape=(0, self.num_units))

    output = self._call_func(observations, training=training)
    return tf.cast(output, self.output_dtype)

  @tf.function
  def _call_spektral(self, observations, training=False):
    embeddings, adj_matrix, edge_features = GraphObserver.graph(
      observations=observations, 
      graph_dims=self._graph_dims)

    for conv in self._convolutions: 
      embeddings = conv([embeddings, adj_matrix, edge_features])

    return embeddings

  @tf.function 
  def _call_tf2_gnn(self, observations, training=False):
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