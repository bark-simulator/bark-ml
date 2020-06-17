import time
import tensorflow as tf
from tf2_gnn.layers import GNN, GNNInput
from tf_agents.utils import common
from bark_ml.observers.graph_observer import GraphObserver
from tf2_gnn import GraphObserverModel
import networkx as nx
from typing import OrderedDict


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
    self._dense_proj = tf.keras.layers.Dense(node_layers_def[-1]["units"],
                                             activation='relu',
                                             trainable=True)

    params = GNN.get_default_hyperparameters()
    params["hidden_dim"] = node_layers_def[0]["units"]
    
    self._gnn = GraphObserverModel(
      model = "RGCN", 
      task = "NodeLevelRegression",
      max_epochs= 200, 
      patience=30)

  # @tf.function
  def call(self, observation, training=False):
    """Call function for the CGNN
    
    Arguments:
        graph {np.array} -- Graph definition
    
    Returns:
        (np.array, np.array) -- Return edges and values
    """
    #return tf.constant([0, 0])

    #this is a networkx.OrderGraph object
    graph = GraphObserver.graph_from_observation(observation)
    num_nodes = len(graph.nodes)
    #random actions generated when we trying to predict actions from observation
    #later, when we receive reward as true actions, we replace random actions by true actions here
    list_random_actions = []
    for agent_id in range(num_nodes): 
      list_random_actions.append({
        'steering': 0.0,
        'acceleration': 0.0
      })
    

    graph_dict = nx.node_link_data(graph)

    raw_data = {}
    raw_data['graph'] = graph_dict
    raw_data['actions'] = list_random_actions
    #print(f'entire graph model: {raw_data}')

    # how to input the graph to get a prediction?
    predicted_output, _ = self._gnn([raw_data]) #here true_output just random output
    
    return predicted_output

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
      return tf.map_fn(lambda g: self.call(g), graph)
    if len(graph.shape) == 4:
      return tf.map_fn(lambda g: self.batch_call(g), graph)
    else:
      return self.call(graph)
  
  def reset(self):
    self._gnn.reset()