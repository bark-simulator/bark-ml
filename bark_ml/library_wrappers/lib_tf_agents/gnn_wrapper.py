import time
import tensorflow as tf
# from gnn.gnn import GNN
from tf2_gnn.layers import GNN, GNNInput
from tf_agents.utils import common
from bark.source.commons import select_col, find_value_ids, select_row, select_range


class GNNWrapper(tf.keras.Model):
  def __init__(self,
               node_layers_def,
               edge_layers_def,
               h0_dim,
               e0_dim,
               name='GNN',
               **kwargs):
    super(GNNWrapper, self).__init__(name=name)
    self._h0_dim = h0_dim
    self._e0_dim = e0_dim
    self._zero_var = None
    # self._number_of_edges = None

    self._node_layers_def = node_layers_def
    # self._old_gnn = GNN(node_layers_def,
    #                     edge_layers_def,
    #                     name=name)
    self._dense_proj = tf.keras.layers.Dense(node_layers_def[-1]["units"],
                                             activation='relu',
                                             trainable=True)
    # self._gnn = common.function(self._gnn)
    params = GNN.get_default_hyperparameters()
    params["hidden_dim"] = node_layers_def[0]["units"]
    # TODO(@hart): modify the size of the network
    self._gnn = GNN(params)

  # @tf.function
  def call(self, graph, training=False):
    """Call function for the CGNN
    
    Arguments:
        graph {np.array} -- Graph definitoin
    
    Returns:
        (np.array, np.array) -- Return edges and values
    """
    number_of_edges = tf.cast(find_value_ids(select_col(graph, 0), -1.)[0], tf.int32)
    number_of_nodes = tf.cast(find_value_ids(select_col(graph, 2), -1.)[0], tf.int32)

    edge_index = tf.cast(graph[:number_of_edges, 0:2], tf.int32)
    values = tf.reshape(graph[:number_of_nodes, 2:(2+self._h0_dim)], [-1, self._h0_dim])
    edges = tf.reshape(graph[:number_of_edges, (2+self._h0_dim):(2+self._h0_dim+self._e0_dim)], [-1, self._e0_dim])

    # step through layers
    # : HERE WE CAN CALL MSFT
    # : always is a single graph
    # : use dense layer to reduce graph output
    # tf.print(edge_index)

    layer_input = GNNInput(
      node_features = values,
      adjacency_lists = (
        edge_index,
      ),
      node_to_graph_map = tf.fill(dims=(number_of_nodes,), value=0),
      num_graphs = 1)
    gnn_output = self._gnn(layer_input, training=training)
    out =  self._dense_proj(gnn_output)
    return out
    # return self._old_gnn(edge_index, values, edges)

  def batch_call(self, graph, training=False):
    """Calls the network multiple times
    
    Arguments:
        graph {np.array} -- Graph representation
    
    Returns:
        np.array -- Batch of values
    """
    if graph.shape[0] == 0:
      if self._zero_var is None:
        self._zero_var = tf.zeros(shape=(0, 1, self._node_layers_def[-1]["units"]))
      return self._zero_var
    if len(graph.shape) == 3:
      return tf.map_fn(lambda g: self.call(g)[0], graph)
    if len(graph.shape) == 4:
      return tf.map_fn(lambda g: self.batch_call(g), graph)
    else:
      return self.call(graph)[0]
  
  def reset(self):
    self._gnn.reset()