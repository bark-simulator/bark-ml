import time
import numpy as np
import tensorflow as tf
from tf2_gnn.layers import GNN, GNNInput
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.graph_observer import GraphObserver
from tensorflow.keras.layers import Dense
from spektral.layers import EdgeConditionedConv, GlobalAttnSumPool, GraphAttention

class GNNWrapper(tf.keras.Model):

  def __init__(self,
               params,
               name='GNN',
               **kwargs):
    super(GNNWrapper, self).__init__(name=name)

    self.num_units = params["FcLayerParams", "", 256] 

    # TODO: properly inject
    self._graph_dims = params["GraphDimensions", "", (4, 4, 4)] 
    self.use_spektral = False
    
    if self.use_spektral:
      self._init_spektral_layers(params)
    else:
      self._init_tf2_gnn_layers(params)

  def _init_spektral_layers(self, params):
    self._conv1 = EdgeConditionedConv(
      params["MPChannels", "", 128], 
      kernel_network=params["KernelNetUnits", "", [256, 256]],
      activation=params["MPLayerActivation", "", "relu"])

    self._pool1 = GlobalAttnSumPool()
    self._dense1 = Dense(
      units=self.num_units,
      activation=params["DenseActivation", "", "tanh"])

  def _init_tf2_gnn_layers(self, params):
    # map bark-ml parameter keys to tf2_gnn parameter keys
    mapped_params = {}
    mapped_params["hidden_dim"] = self.num_units
    mapped_params["num_layers"] = params["NumLayers", "", 1]
    mapped_params["global_exchange_mode"] =\
      params["global_exchange_mode", "", "mean"]
    mapped_params["message_calculation_class"] =\
      params["message_calculation_class", "", "gnn_edge_mlp"]

    mp_style = mapped_params["message_calculation_class"]
    gnn_params = GNN.get_default_hyperparameters(mp_style)    
    gnn_params.update(mapped_params)

    self._gnn = GNN(gnn_params)

  #@tf.function
  def call(self, observations, training=False):
    if observations.shape[0] == 0:
      return tf.random.normal(shape=(0, self.num_units))

    if self.use_spektral:
      return self.call_spektral(observations, training)
    else:
      return self.call_tf2_gnn(observations, training)

  @tf.function
  def call_spektral(self, observations, training=False):
    X, A, E = GraphObserver.graph(observations, self._graph_dims)

    X = self._conv1([X, A, E])
    X = self._pool1(X)
    X = self._dense1(X)
    
    return X

  def call_tf2_gnn(self, observations, training=False):
    batch_size = tf.constant(observations.shape[0])

    X, A, node_to_graph_map = GraphObserver.graph(
      observations, 
      graph_dims=self._graph_dims, 
      dense=True)

    gnn_input = GNNInput(
      node_features=X,
      adjacency_lists=(A,),
      node_to_graph_map=node_to_graph_map,
      num_graphs=batch_size,
    )

    return self._gnn(gnn_input, training=training)
      
  def reset(self):
    self._gnn.reset()