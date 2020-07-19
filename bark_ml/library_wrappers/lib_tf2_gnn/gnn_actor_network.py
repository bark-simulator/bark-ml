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

import time
import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import numpy as np

from tf_agents.agents.sac import sac_agent
from tf_agents.networks import network, normal_projection_network, utils, encoding_network
from tf_agents.utils import common, nest_utils

from bark_ml.library_wrappers.lib_tf2_gnn import GNNWrapper

def projection_net(spec):
  return normal_projection_network.NormalProjectionNetwork(
    spec,
    mean_transform=None,
    state_dependent_std=True,
    init_means_output_factor=0.1,
    std_transform=sac_agent.std_clip_transform,
    scale_distribution=True)


@gin.configurable
class GNNActorNetwork(network.Network):
  """Creates an actor GNN."""

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               gnn_params,
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

    if len(tf.nest.flatten(input_tensor_spec)) > 1:
      raise ValueError('Only a single observation is supported by this network')
    
    flat_action_spec = tf.nest.flatten(output_tensor_spec)
    
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action is supported by this network')
    
    self._single_action_spec = flat_action_spec[0]
    if self._single_action_spec.dtype not in [tf.float32, tf.float64]:
      raise ValueError('Only float actions are supported by this network.')
    
    self._gnn = GNNWrapper(params=gnn_params)

    self._encoder = encoding_network.EncodingNetwork(
      input_tensor_spec=tf.TensorSpec([None, self._gnn.num_units]),
      preprocessing_layers=None,
      preprocessing_combiner=None,
      conv_layer_params=conv_layer_params,
      fc_layer_params=fc_layer_params,
      dropout_layer_params=dropout_layer_params,
      activation_fn=activation_fn,
      kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
      batch_squash=False,
      dtype=tf.float32)

    self._projection_nets = tf.nest.map_structure(projection_net, output_tensor_spec)
    self._output_tensor_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                        self._projection_nets)

    self.call_times = []
    self.gnn_call_times = []

  @property  
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self, observations, step_type=(), network_state=(), training=False):
    if len(observations.shape) == 1:
      observations = tf.expand_dims(observations, axis=0)

    batch_size, feature_len = observations.shape
    
    t0 = time.time()
    output = self._gnn.batch_call(observations, training=training)
    self.gnn_call_times.append(time.time() - t0)
    output = tf.cast(output, tf.float32)

    # extract ego state (node 0)
    if len(output.shape) == 2 and output.shape[0] != 0:
      output = tf.reshape(output, [batch_size, -1, self._gnn.num_units])
      output = tf.gather(output, 0, axis=1)
    elif len(output.shape) == 3:
      output = tf.gather(output, 0, axis=1)

    tf.summary.histogram("actor_gnn_output", output)
    
    output, network_state = self._encoder(
      output,
      step_type=step_type,
      network_state=network_state,
      training=training)
      
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)

    def call_projection_net(proj_net):
      distribution, _ = proj_net(
          output, outer_rank, training=training)
      return distribution

    output_actions = tf.nest.map_structure(
        call_projection_net, self._projection_nets)

    try:
        tf.summary.histogram("actor_output_actions", output_actions)
    except Exception as e:
        pass

    self.call_times.append(time.time() - t0)
    return output_actions, network_state
