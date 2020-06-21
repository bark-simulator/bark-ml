import time
import tensorflow as tf
from tf2_gnn.layers import GNN, GNNInput
from tf_agents.utils import common
from bark_ml.observers.graph_observer import GraphObserver
import networkx as nx
from typing import OrderedDict


class GNNWrapper(tf.keras.Model):

  def __init__(self,
               node_layers_def,
               name='GNN',
               **kwargs):
    super(GNNWrapper, self).__init__(name=name)
    self._node_layers_def = node_layers_def

    params = GNN.get_default_hyperparameters()
    params["hidden_dim"] = node_layers_def[0]["units"]
    self._gnn = GNN(params)

  # @tf.function
  def call(self, observation, training=False):
    """Call function for the GNN"""
    graph = GraphObserver.graph_from_observation(observation)

    features = []
    for (node_id, attributes) in graph.nodes.data():
      features.append(list(attributes.values()))

    gnn_input = GNNInput(
      node_features = tf.convert_to_tensor(features),
      adjacency_lists = (
        tf.constant(list(map(list, graph.edges)), dtype=tf.int32),
      ),
      node_to_graph_map = tf.fill(dims=(len(graph.nodes),), value=0),
      num_graphs = 1,
    )

    return self._gnn(gnn_input)

  def batch_call(self, graph, training=False):
    """Calls the network multiple times
    
    Arguments:
        graph {np.array} -- Graph representation
    
    Returns:
        np.array -- Batch of values
    """
    if graph.shape[0] == 0:
      return tf.zeros(shape=(1, self._node_layers_def[-1]["units"]))
    if len(graph.shape) == 1:
      return self.call(graph)
    if len(graph.shape) == 2:
      if graph.shape[0] == 1:
        return self.call(tf.reshape(graph, [-1]))
      else:
        return tf.map_fn(lambda g: self.call(g), graph)
    if len(graph.shape) == 3:
      return tf.map_fn(lambda g: self.call(g), graph)
    if len(graph.shape) == 4:
      return tf.map_fn(lambda g: self.batch_call(g), graph)
    else:
      raise ValueError(f'Graph has invalid shape {graph.shape}')
      
  def reset(self):
    self._gnn.reset()