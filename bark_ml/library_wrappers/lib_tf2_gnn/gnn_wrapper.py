import time
import tensorflow as tf
from tf2_gnn.layers import GNN, GNNInput
from tf2_gnn.layers.message_passing import GGNN, GNN_FiLM
from tf_agents.utils import common
from bark_ml.observers.graph_observer import GraphObserver
import networkx as nx
from typing import OrderedDict


class GNNWrapper(tf.keras.Model):

  def __init__(self,
               num_layers=4,
               num_units=16,
               name='GNN',
               **kwargs):
    super(GNNWrapper, self).__init__(name=name)

    self.num_layers = num_layers
    self.num_units = num_units

    params = GNN.get_default_hyperparameters()
    params.update(GNN_FiLM.get_default_hyperparameters())
    #params["global_exchange_mode"] = "mlp"
    params["num_layers"] = num_layers
    params["hidden_dim"] = num_units
    params["message_calculation_class"] = "gnn_film"
    self._gnn = GNN(params)

  #@tf.function
  def call(self, observation, training=False):
    """Call function for the GNN"""
    graph = GraphObserver.graph_from_observation(observation)

    features = []
    for _, attributes in graph.nodes.data():
      features.append(list(attributes.values()))

    gnn_input = GNNInput(
      node_features = tf.convert_to_tensor(features),
      adjacency_lists = (
        tf.constant(list(map(list, graph.edges)), dtype=tf.int32),
      ),
      node_to_graph_map = tf.fill(dims=(len(graph.nodes),), value=0),
      num_graphs = 1,
    )

    return self._gnn(gnn_input, training=training)

  def batch_call(self, graph, training=False):
    """Calls the network multiple times
    
    Arguments:
        graph {np.array} -- Graph representation
    
    Returns:
        np.array -- Batch of values
    """
    if graph.shape[0] == 0:
      return tf.random.normal(shape=(0, self.num_units))
    if len(graph.shape) == 1:
      return self.call(graph, training=training)
    if len(graph.shape) == 2:
      if graph.shape[0] == 1:
        return self.call(tf.reshape(graph, [-1]), training=training)
      else:
        return tf.map_fn(lambda g: self.call(g, training=training), graph)
    if len(graph.shape) == 3:
      return tf.map_fn(lambda g: self.call(g, training=training), graph)
    if len(graph.shape) == 4:
      return tf.map_fn(lambda g: self.batch_call(g, training=training), graph)
    else:
      raise ValueError(f'Graph has invalid shape {graph.shape}')
      
  def reset(self):
    self._gnn.reset()