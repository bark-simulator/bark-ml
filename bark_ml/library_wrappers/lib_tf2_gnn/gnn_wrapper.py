import time
import numpy as np
import tensorflow as tf
from tf2_gnn.layers import GNN, GNNInput
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.graph_observer import GraphObserver
from tensorflow.keras.layers import Dense
from spektral.layers import EdgeConditionedConv, GlobalAttnSumPool, GraphAttention
from spektral.utils import numpy_to_batch


class GNNWrapper(tf.keras.Model):

  def __init__(self,
               params,
               name='GNN',
               **kwargs):
    super(GNNWrapper, self).__init__(name=name)

    mp_style = params.get("message_calculation_class", "rgcn")
    gnn_params = GNN.get_default_hyperparameters(mp_style)    
    gnn_params.update(params)

    self._gnn = GNN(gnn_params)
    self.num_units = gnn_params["hidden_dim"]
  
    self._conv1 = EdgeConditionedConv(128, activation='relu')
    self._conv2 = EdgeConditionedConv(128, activation='relu')
    self._pool1 = GlobalAttnSumPool()
    self._dense1 = Dense(self.num_units, activation='tanh')

    self.use_spektral = True
    self.graph_conversion_times = []
    self.gnn_call_times = []

  @tf.function
  def call(self, observations, training=False):
    if observations.shape[0] == 0:
      return tf.random.normal(shape=(0, self.num_units))
    #if len(observations.shape) > 3:
    #  raise ValueError(f'Graph has invalid shape {observations.shape}')
    
    #t0 = time.time()
    
    if self.use_spektral:
      return self.call_spektral(observations, training)
    else:
      return self.call_tf2_gnn(observations, training)
    
    #self.gnn_call_times.append(time.time() - t0)

  @tf.function
  def call_spektral(self, observations, training=False):
    batch_size = observations.shape[0]
    n_nodes = 4
    n_features = 4
    n_edge_features = 4

    obs = observations[:, 3:] # removes first three elements of each sample
    adj_start_idx = 4 * 4
    adj_end_idx = adj_start_idx + 4 ** 2

    X = tf.reshape(obs[:, :n_nodes * n_nodes], [batch_size, n_nodes, n_features])
    A = tf.reshape(obs[:, adj_start_idx:adj_end_idx], [batch_size, n_nodes, n_nodes])
    E = tf.reshape(obs[:, adj_end_idx:], [batch_size, n_nodes, n_nodes, n_edge_features])

    X = self._conv1([X, A, E])
    X = self._pool1(X)
    X = self._dense1(X)
    
    return X

  def call_tf2_gnn(self, observations, training=False):
    t0 = time.time()
    features, edges = [], []
    node_to_graph_map = []
    num_graphs = tf.constant(len(observations))

    edge_index_offset = 0
    for i, sample in enumerate(observations):
      f, e = GraphObserver.graph(sample)
      features.extend(f)
      num_nodes = len(f)
      e = np.asarray(e) + edge_index_offset
      edges.extend(e)
      node_to_graph_map.extend(np.full(num_nodes, i))
      edge_index_offset += num_nodes

    gnn_input = GNNInput(
      node_features=tf.convert_to_tensor(features),
      adjacency_lists=(
        tf.convert_to_tensor(edges, dtype=tf.int32),
      ),
      node_to_graph_map=tf.convert_to_tensor(node_to_graph_map),
      num_graphs=num_graphs,
    )

    self.graph_conversion_times.append(time.time() - t0)

    t0 = time.time()
    output = self._gnn(gnn_input, training=training)
    self.gnn_call_times.append(time.time() - t0)
    return output
      
  def reset(self):
    self._gnn.reset()