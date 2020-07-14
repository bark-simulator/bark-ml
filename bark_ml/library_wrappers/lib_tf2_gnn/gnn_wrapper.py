import time
import numpy as np
import tensorflow as tf
from tf2_gnn.layers import GNN, GNNInput, \
  NodesToGraphRepresentationInput, WeightedSumGraphRepresentation
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
    params["global_exchange_mode"] = "mlp"
    params["num_layers"] = num_layers
    params["hidden_dim"] = num_units
    params["message_calculation_class"] = "ggnn"
    self._gnn = GNN(params)

    self.graph_conversion_times = []
    self.gnn_call_times = []

  #@tf.function
  def call(self, observation, training=False):
    """Call function for the GNN"""
    t0 = time.time()
    features, edges = GraphObserver.gnn_input(observation)

    gnn_input = GNNInput(
      node_features = tf.convert_to_tensor(features),
      adjacency_lists = (
        tf.constant(list(map(list, edges)), dtype=tf.int32),
      ),
      node_to_graph_map = tf.fill(dims=(len(features),), value=0),
      num_graphs = 1,
    )
    self.graph_conversion_times.append(time.time() - t0)

    t0 = time.time()
    output = self._gnn(gnn_input, training=training)
    self.gnn_call_times.append(time.time() - t0)
    return output

  def call_fast(self, observations, training=False):
    t0 = time.time()
    features, edges = [], []
    node_to_graph_map = []
    num_graphs = len(observations)

    edge_index_offset = 0
    for i, sample in enumerate(observations):
      f, e = GraphObserver.gnn_input(sample)
      features.extend(f)
      num_nodes = len(f)
      e = np.asarray(e) + edge_index_offset
      edges.extend(e)
      node_to_graph_map.extend(np.full(num_nodes, i))
      edge_index_offset += num_nodes

    gnn_input = GNNInput(
      node_features=tf.convert_to_tensor(features),
      adjacency_lists=(
        tf.convert_to_tensor(list(map(list, edges)), dtype=tf.int32),
      ),
      node_to_graph_map=tf.convert_to_tensor(node_to_graph_map),
      num_graphs=num_graphs,
    )

    self.graph_conversion_times.append(time.time() - t0)

    t0 = time.time()
    output = self._gnn(gnn_input, training=training)
    self.gnn_call_times.append(time.time() - t0)
    return output

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
        return self.call_fast(graph, training=training)
        #return tf.map_fn(lambda g: self.call(g, training=training), graph)
    if len(graph.shape) == 3:
      return tf.map_fn(lambda g: self.call(g, training=training), graph)
    else:
      raise ValueError(f'Graph has invalid shape {graph.shape}')
      
  def reset(self):
    self._gnn.reset()