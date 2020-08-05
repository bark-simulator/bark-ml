import logging
import tensorflow as tf
from tf2_gnn.layers import GNN, GNNInput
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.graph_observer import GraphObserver
from tensorflow.keras.layers import Dense
from spektral.layers import EdgeConditionedConv

class GNNWrapper(tf.keras.Model):

  def __init__(self,
               params,
               name='GNN',
               output_dtype=tf.float32,
               **kwargs):
    super(GNNWrapper, self).__init__(name=name)

    params = params.ConvertToDict()
    self.num_units = params.get("MpLayerNumUnits", 256)
    self.output_dtype = output_dtype
    self._logger = logging.getLogger()

    self._graph_dims = params.get("GraphDimensions")
    
    lib = params.get("library", "tf2_gnn")
    self.use_spektral = lib == "spektral"

    self._logger.info(f'Initializing GNN with library: {lib}')
    
    if self.use_spektral:
      self._init_spektral_layers(params)
    else:
      self._init_tf2_gnn_layers(params)

  def _init_spektral_layers(self, params):
    self._convolutions = []
    
    for i in range(params.get("NumMpLayers", 1)):
      self._convolutions.append(EdgeConditionedConv(
        channels=params.get("MPChannels", 64),
        kernel_network=params.get("KernelNetUnits", [256]),
        activation=params.get("MPLayerActivation", "relu"))
      )

    self._dense = Dense(
      units=self.num_units,
      activation=params.get("DenseActivation", "relu"))

  def _init_tf2_gnn_layers(self, params):
    # map bark-ml parameter keys to tf2_gnn parameter keys
    mapped_params = {}
    mapped_params["hidden_dim"] = self.num_units
    mapped_params["num_layers"] = params.get("NumMpLayers", 1)
    mapped_params["global_exchange_mode"] =\
      params.get("global_exchange_mode", "gru")
    mapped_params["message_calculation_class"] =\
      params.get("message_calculation_class", "gnn_edge_mlp")

    mp_style = mapped_params["message_calculation_class"]
    gnn_params = GNN.get_default_hyperparameters(mp_style)    
    gnn_params.update(mapped_params)

    self._gnn = GNN(gnn_params)

  @tf.function
  def call(self, observations, training=False):
    if observations.shape[0] == 0:
      return tf.random.normal(shape=(0, self.num_units))

    if self.use_spektral:
      output = self.call_spektral(observations, training)
    else:
      output = self.call_tf2_gnn(observations, training)

    return tf.cast(output, self.output_dtype)

  @tf.function
  def call_spektral(self, observations, training=False):
    node_features, adjacency_matrix, edge_features = GraphObserver.graph(
      observations, self._graph_dims)

    for conv in self._convolutions: 
      node_features = conv([node_features, adjacency_matrix, edge_features])

    node_features = self._dense(node_features)
    return node_features

  @tf.function 
  def call_tf2_gnn(self, observations, training=False):
    batch_size = tf.constant(observations.shape[0])

    node_features, adjacency_list, node_to_graph_map = GraphObserver.graph(
      observations, 
      graph_dims=self._graph_dims, 
      dense=True)

    gnn_input = GNNInput(
      node_features=node_features,
      adjacency_lists=(adjacency_list,),
      node_to_graph_map=node_to_graph_map,
      num_graphs=batch_size,
    )

    # tf2_gnn outputs a flattened node embeddings vector, so we 
    # reshape it to have the embeddings of each node seperately.
    flat_output = self._gnn(gnn_input, training=training)
    output = tf.reshape(flat_output, [batch_size, -1, self.num_units])
    return output
      
  def reset(self):
    self._gnn.reset()