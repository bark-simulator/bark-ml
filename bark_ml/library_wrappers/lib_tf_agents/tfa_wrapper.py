import numpy as np
import tensorflow as tf

# tfa specific
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()


class TFAWrapper(py_environment.PyEnvironment):
  """Wrapper for TensorFlow Agents (https://github.com/tensorflow/agents)

  Arguments:
    py_environment -- Base class for environment from tf_agents
  """
  def __init__(self, env):
    self._env = env
    self._action_spec = array_spec.BoundedArraySpec(
      shape=self._env.action_space.shape,
      dtype=np.float32,
      minimum=self._env.action_space.low,
      maximum=self._env.action_space.high,
      name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
      shape=self._env.observation_space.shape,
      dtype=np.float32,
      minimum=self._env.observation_space.low,
      maximum=self._env.observation_space.high,
      name='observation')
    self._state = np.zeros(shape=self._env.observation_space.shape,
      dtype=np.float32)
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def render(self):
    """Renders the enviornment
    """
    return self._env.render()

  def _reset(self):
    """Resets the wrapper
    """
    self._state = np.array(self._env.reset(), dtype=np.float32)
    self._episode_ended = False
    return ts.restart(self._state)

  def _step(self, action):
    """Steps the world for a given dt
    """
    if self._episode_ended:
      return self.reset()
    state, reward, self._episode_ended, _ = self._env.step(action)
    self._state = np.array(state, dtype=np.float32)
    if self._episode_ended:
      return ts.termination(self._state, reward=reward)
    else:
      return ts.transition(self._state, reward=reward, discount=0.99)




# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Actor network to use with DDPG agents.
Note: This network scales actions to fit the given spec by using `tanh`. Due to
the nature of the `tanh` function, actions near the spec bounds cannot be
returned.
"""

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import common

#from bark_ml.library_wrappers.lib_tf_agents.gnn import GNNWrapper

@gin.configurable
class GNNActorNetwork(network.Network):
  """Creates an actor network."""

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               fc_layer_params=None,
               dropout_layer_params=None,
               conv_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               name='ActorNetwork'):
    """Creates an instance of `ActorNetwork`.
    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        inputs.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the outputs.
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      dropout_layer_params: Optional list of dropout layer parameters, each item
        is the fraction of input units to drop or a dictionary of parameters
        according to the keras.Dropout documentation. The additional parameter
        `permanent', if set to True, allows to apply dropout at inference for
        approximated Bayesian inference. The dropout layers are interleaved with
        the fully connected layers; there is a dropout layer after each fully
        connected layer, except if the entry in the list is None. This list must
        have the same length of fc_layer_params, or be None.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      name: A string representing name of the network.
    Raises:
      ValueError: If `input_tensor_spec` or `action_spec` contains more than one
        item, or if the action data type is not `float`.
    """

    super(GNNActorNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)
    self._gnn = GNNWrapper(
      node_layers_def=[
        {"units" : 80, "activation": "relu", "dropout_rate": 0.0, "type": "DenseLayer"},
        {"units" : 80, "activation": "relu", "dropout_rate": 0.0, "type": "DenseLayer"},
        {"units" : 80, "activation": "relu", "dropout_rate": 0.0, "type": "DenseLayer"},
      ],
      edge_layers_def=[
        {"units" : 80, "activation": "relu", "dropout_rate": 0.0, "type": "DenseLayer"},
        {"units" : 80, "activation": "relu", "dropout_rate": 0.0, "type": "DenseLayer"},
        {"units" : 80, "activation": "relu", "dropout_rate": 0.0, "type": "DenseLayer"},
      ],
      h0_dim=4,
      e0_dim=2)
    if len(tf.nest.flatten(input_tensor_spec)) > 1:
      raise ValueError('Only a single observation is supported by this network')

    flat_action_spec = tf.nest.flatten(output_tensor_spec)
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action is supported by this network')
    self._single_action_spec = flat_action_spec[0]

    if self._single_action_spec.dtype not in [tf.float32, tf.float64]:
      raise ValueError('Only float actions are supported by this network.')

    # TODO(kbanoop): Replace mlp_layers with encoding networks.
    self._mlp_layers = utils.mlp_layers(
        conv_layer_params,
        fc_layer_params,
        dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform'),
        name='input_mlp')

    self._mlp_layers.append(
        tf.keras.layers.Dense(
            flat_action_spec[0].shape.num_elements(),
            activation=tf.keras.activations.tanh,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003),
            name='action'))

    self._output_tensor_spec = output_tensor_spec

  def call(self, observations, step_type=(), network_state=(), training=False):
    del step_type  # unused.
    # graph transform
    observations = self._gnn.batch_call(observations)
    #observations = tf.nest.flatten(observations)
    output = tf.cast(observations, tf.float32)

    # for layer in self._mlp_layers:
    #   output = layer(output, training=training)

    actions = common.scale_to_spec(output, self._single_action_spec)
    output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec,
                                              [actions])

    return output_actions, network_state

############################################################
############################################################

import time
import tensorflow as tf
# from gnn.gnn import GNN
from tf2_gnn.layers import GNN, GNNInput
from tf_agents.utils import common
#from bark.source.commons import select_col, find_value_ids, select_row, select_range
from bark_ml.observers.graph_observer import GraphObserver
from tf2_gnn import GraphObserverModel


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
      model="RGCN", 
      task="NodeLevelRegression",
      max_epochs= 300, 
      patience=30
    )

  # @tf.function
  def call(self, observation, training=False):
    """Call function for the CGNN
    
    Arguments:
        graph {np.array} -- Graph definition
    
    Returns:
        (np.array, np.array) -- Return edges and values
    """
    graph = GraphObserver.graph_from_observation(observation)
    
    output = self._gnn(graph)
    print(f'output: {output}')

    # print("Decoding graph...")
    # print(graph.nodes)
    # print(graph.edges)

    # step through layers
    # : HERE WE CAN CALL MSFT
    # : always is a single graph
    # : use dense layer to reduce graph output
    # tf.print(edge_index)

    # print("\n--------------------------------------------\n")
    # node_features = list(map(lambda n: graph.nodes[n].values(), graph.nodes))
    # print(node_features)
    # print("\n--------------------------------------------\n")

    # adjacency_list = tf.constant(graph.edges, dtype=tf.int32)

    # layer_input = GNNInput(
    #   node_features = tf.random.normal(shape=(4, 6)),
    #   adjacency_lists = adjacency_list,
    #   node_to_graph_map = tf.fill(dims=(len(graph.nodes),), value=0),
    #   num_graphs = 1
    # )

    # layer_input = GNNInput(
    # node_features = tf.random.normal(shape=(5, 3)),
    #   adjacency_lists = (
    #       tf.constant([[0, 1], [1, 2], [3, 4]], dtype=tf.int32),
    #       tf.constant([[1, 2], [3, 4]], dtype=tf.int32),
    #       tf.constant([[2, 0]], dtype=tf.int32)
    #   ),
    #   node_to_graph_map = tf.fill(dims=(5,), value=0),
    #   num_graphs = 1
    # )
    
    # gnn_output = self._gnn(layer_input, training=training)
    # out = self._dense_proj(gnn_output)
    return np.array([[0.5, 0.0]])

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