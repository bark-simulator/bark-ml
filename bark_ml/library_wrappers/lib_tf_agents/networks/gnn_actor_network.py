# Copyright (c) 2020 fortiss GmbH
#
# Authors: Marco Oliva
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gin
import tensorflow as tf # pylint: disable=unused-import

from tf_agents.agents.sac import sac_agent
from tf_agents.networks import network, normal_projection_network, encoding_network
from tf_agents.utils import nest_utils
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.library_wrappers.lib_tf_agents.networks.gnns.interaction_wrapper import InteractionWrapper # pylint: disable=unused-import


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
               gnn,
               fc_layer_params=None,
               dropout_layer_params=None,
               conv_layer_params=None,
               activation_fn=tf.nn.relu,
               name='ActorNetwork',
               params=ParameterServer()):
    """
    Creates an instance of `ActorNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        inputs.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the outputs.
      gnn: The function that initializes a graph neural network that
        accepts the input observations and computes node embeddings.
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

    if flat_action_spec[0].dtype not in [tf.float32, tf.float64]:
      raise ValueError('Only float actions are supported by this network.')

    if gnn is None:
      raise ValueError('`gnn` must not be `None`.')

    self._gnn = gnn(name=name, params=params)
    self._latent_trace = None
    self._encoder = encoding_network.EncodingNetwork(
      input_tensor_spec=tf.TensorSpec([None, self._gnn._embedding_size]),
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

  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self, observations, step_type=(), network_state=(), training=False):
    del step_type # unused.

    if len(observations.shape) == 1:
      observations = tf.expand_dims(observations, axis=0)
    batch_size = tf.shape(observations)[0]
    embeddings = self._gnn(observations, training=training)
    self._latent_trace = self._gnn._latent_trace

    # extract ego state (node 0)
    if batch_size > 0:
      embeddings = embeddings[:, 0]

    # with tf.name_scope("GNNActorNetwork"):
    #   tf.summary.histogram("actor_gnn_output", embeddings)

    output, network_state = self._encoder(embeddings, training=training)
    output = embeddings

    outer_rank = nest_utils.get_outer_rank(
      observations,
      self.input_tensor_spec)

    def call_projection_net(net):
      distribution, _ = net(output, outer_rank, training=training)
      return distribution

    output_actions = tf.nest.map_structure(
      call_projection_net, self._projection_nets)

    return output_actions, network_state
