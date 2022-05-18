# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Marco Oliva
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import tensorflow as tf

from bark.runtime.commons.parameters import ParameterServer


class GraphNetwork(tf.keras.Model):
  """
  Implements a graph neural network.

  This class serves as a wrapper over the specific implementation
  of the configured GNN library.
  """

  def __init__(self,
               params=ParameterServer(),
               name='GNN',
               output_dtype=tf.float32):
    """
    Initializes a GraphNetwork instance.

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
    super(GraphNetwork, self).__init__(name=name)
    self.output_dtype = output_dtype
    self._params = params
    try:
      self._graph_dims = self._validated_graph_dims(params["ML"]["GraphDims"])
    except:
      pass

    # logging.info(
    #   f'"{name}" configured with `spektral` for input graphs with ' +
    #   f'{self._graph_dims[0]} nodes, {self._graph_dims[1]} node features, ' +
    #   f'and {self._graph_dims[2]} edge features.')


  def _validated_graph_dims(self, graph_dims):
    if graph_dims is None:
      raise ValueError('Graph dimensions must not be `None`.')
    if len(graph_dims) != 3:
      raise ValueError('Graph dimensions must be of length 3.')
    int_dims = list(map(int, graph_dims))
    if min(int_dims) < 0:
      raise ValueError('Graph dimensions must be positive.')
    return int_dims

  def _init_network(self, name=None):
    pass

  # @tf.function
  def _init_call_func(self, observations, training=False):
    pass

  # relax shapes due to varying batch sizes
  # @tf.function(experimental_relax_shapes=True)
  def call(self, observations, training=False):
    if tf.shape(observations)[0] == 0:
      return tf.random.normal(shape=(0, self._embedding_size))
    output = self._call_func(observations, training=training)
    return tf.cast(output, self.output_dtype)
