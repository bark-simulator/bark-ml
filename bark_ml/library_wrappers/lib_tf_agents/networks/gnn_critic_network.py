# Copyright (c) 2020 fortiss GmbH
#
# Authors: Marco Oliva
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gin
import tensorflow as tf  # pylint: disable=unused-import

from tf_agents.networks import network, utils
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.library_wrappers.lib_tf_agents.networks.gnns.interaction_wrapper import InteractionWrapper # pylint: disable=unused-import


@gin.configurable
class GNNCriticNetwork(network.Network):
  """Creates a critic network."""

  def __init__(self,
               input_tensor_spec,
               gnn,
               observation_fc_layer_params=None,
               observation_dropout_layer_params=None,
               observation_conv_layer_params=None,
               observation_activation_fn=tf.nn.relu,
               action_fc_layer_params=None,
               action_dropout_layer_params=None,
               action_conv_layer_params=None,
               action_activation_fn=tf.nn.relu,
               joint_fc_layer_params=None,
               joint_dropout_layer_params=None,
               joint_activation_fn=tf.nn.relu,
               output_activation_fn=None,
               name='CriticNetwork',
               params=ParameterServer()):
    """
    Creates an instance of `GNNCriticNetwork`.

    Args:
      input_tensor_spec: A tuple of (observation, action) each a nest of
        `tensor_spec.TensorSpec` representing the inputs.
      gnn: The function that initializes a graph neural network that
        accepts the input observations and computes node embeddings.
      observation_fc_layer_params: Optional list of fully connected parameters
        for observations, where each item is the number of units in the layer.
      observation_dropout_layer_params: Optional list of dropout layer
        parameters, each item is the fraction of input units to drop or a
        dictionary of parameters according to the keras.Dropout documentation.
        The additional parameter `permanent', if set to True, allows to apply
        dropout at inference for approximated Bayesian inference. The dropout
        layers are interleaved with the fully connected layers; there is a
        dropout layer after each fully connected layer, except if the entry in
        the list is None. This list must have the same length of
        observation_fc_layer_params, or be None.
      observation_conv_layer_params: Optional list of convolution layer
        parameters for observations, where each item is a length-three tuple
        indicating (embedding_size, kernel_size, stride).
      observation_activation_fn: Activation function applied to the observation
        layers, e.g. tf.nn.relu, slim.leaky_relu, ...
      action_fc_layer_params: Optional list of fully connected parameters for
        actions, where each item is the number of units in the layer.
      action_dropout_layer_params: Optional list of dropout layer parameters,
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent', if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of action_fc_layer_params, or
        be None.
      action_conv_layer_params: Optional list of convolution layer
        parameters for actions, where each item is a length-three tuple
        indicating (embedding_size, kernel_size, stride).
      action_activation_fn: Activation function applied to the action layers,
        e.g. tf.nn.relu, slim.leaky_relu, ...
      joint_fc_layer_params: Optional list of fully connected parameters after
        merging observations and actions, where each item is the number of units
        in the layer.
      joint_dropout_layer_params: Optional list of dropout layer parameters,
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent', if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of joint_fc_layer_params, or
        be None.
      joint_activation_fn: Activation function applied to the joint layers,
        e.g. tf.nn.relu, slim.leaky_relu, ...
      output_activation_fn: Activation function for the last layer. This can be
        used to restrict the range of the output. For example, one can pass
        tf.keras.activations.sigmoid here to restrict the output to be bounded
        between 0 and 1.
      name: A string representing name of the network.

    Raises:
      ValueError: If `observation_spec` or `action_spec` contains more than one
        observation.
    """
    super(GNNCriticNetwork, self).__init__(
      input_tensor_spec=input_tensor_spec,
      state_spec=(),
      name=name)

    observation_spec, action_spec = input_tensor_spec

    if len(tf.nest.flatten(observation_spec)) > 1:
      raise ValueError('Only a single observation is supported by this network')

    if len(tf.nest.flatten(action_spec)) > 1:
      raise ValueError('Only a single action is supported by this network')

    if gnn is None:
      raise ValueError('`gnn` must not be `None`.')

    self._gnn = gnn(name=name, params=params)

    self._observation_layers = utils.mlp_layers(
      observation_conv_layer_params,
      observation_fc_layer_params,
      observation_dropout_layer_params,
      activation_fn=observation_activation_fn,
      kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
        scale=1./3., mode='fan_in', distribution='uniform'),
      name='observation_encoding')

    self._action_layers = utils.mlp_layers(
      action_conv_layer_params,
      action_fc_layer_params,
      action_dropout_layer_params,
      activation_fn=action_activation_fn,
      kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
        scale=1./3., mode='fan_in', distribution='uniform'),
      name='action_encoding')

    self._joint_layers = utils.mlp_layers(
        None,
        joint_fc_layer_params,
        joint_dropout_layer_params,
        activation_fn=joint_activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
          scale=1./3., mode='fan_in', distribution='uniform'),
        name='joint_mlp')

    self._joint_layers.append(
      tf.keras.layers.Dense(
        units=1,
        activation=output_activation_fn,
        kernel_initializer=tf.keras.initializers.RandomUniform(
          minval=-0.003, maxval=0.003),
        name='value'))

  def call(self, inputs, step_type=(), network_state=(), training=False):
    del step_type # unused.

    observations, actions = inputs
    batch_size = tf.shape(observations)[0]
    embeddings = self._gnn(observations, training=training)

    if batch_size > 0:
      embeddings = embeddings[:, 0] # extract ego state
      actions = tf.reshape(actions, [batch_size, -1])

    # with tf.name_scope("GNNCriticEmbeddings"):
    #   tf.summary.histogram("critic_gnn_output", embeddings)

    embeddings = tf.cast(tf.nest.flatten(embeddings)[0], tf.float32)
    for layer in self._observation_layers:
      embeddings = layer(embeddings, training=training)

    actions = tf.cast(tf.nest.flatten(actions)[0], tf.float32)
    for layer in self._action_layers:
      actions = layer(actions, training=training)

    joint = tf.concat([embeddings, actions], 1)
    for layer in self._joint_layers:
      joint = layer(joint, training=training)

    # with tf.name_scope("GNNCriticNetwork"):
    #   tf.summary.histogram("critic_output_value", joint)

    return tf.reshape(joint, [-1]), network_state
