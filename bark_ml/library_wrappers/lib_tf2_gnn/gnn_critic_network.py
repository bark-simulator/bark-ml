import time
import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import numpy as np

from tf_agents.networks import network, utils, encoding_network
from bark_ml.library_wrappers.lib_tf2_gnn import GNNWrapper

@gin.configurable
class GNNCriticNetwork(network.Network):
  """Creates a critic GNN."""

  def __init__(self,
               input_tensor_spec,
               gnn_params,
               action_fc_layer_params=None,
               action_dropout_layer_params=None,
               dropout_layer_params=None,
               conv_layer_params=None,
               joint_fc_layer_params=None,
               joint_dropout_layer_params=None,
               activation_fn=tf.nn.relu,
               output_activation_fn=None,
               name='CriticNetwork'):
    """Creates an instance of `CriticNetwork`.

    Args:
      input_tensor_spec: A tuple of (observation, action) each a nest of
        `tensor_spec.TensorSpec` representing the inputs.
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
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
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

    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action is supported by this network')
    self._single_action_spec = flat_action_spec[0]

    self._gnn = GNNWrapper(params=gnn_params)

    self._encoder = encoding_network.EncodingNetwork(
      input_tensor_spec=tf.TensorSpec([None, self._gnn.num_units]),
      preprocessing_layers=None,
      preprocessing_combiner=None,
      conv_layer_params=conv_layer_params,
      fc_layer_params=action_fc_layer_params,
      dropout_layer_params=dropout_layer_params,
      activation_fn=activation_fn,
      kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
      batch_squash=False,
      dtype=tf.float32)

    self._joint_layers = utils.mlp_layers(
        None,
        joint_fc_layer_params,
        joint_dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform'),
        name='joint_mlp')

    self._joint_layers.append(
      tf.keras.layers.Dense(
        1,
        activation=output_activation_fn,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.003, maxval=0.003),
        name='value'))

    self.call_times = []
    self.gnn_call_times = []
    self.encoder_call_times = []
    self.joint_call_times = []

  def call(self, inputs, step_type=(), network_state=(), training=False):
    t1 = time.time()
    observations, actions = inputs
    batch_size = observations.shape[0]
     
    t0 = time.time()
    node_embeddings = self._gnn.batch_call(observations, training=training)
    self.gnn_call_times.append(time.time() - t0)
    actions = tf.cast(actions, tf.float32)
    node_embeddings = tf.cast(node_embeddings, tf.float32)

    if batch_size > 0:
      node_embeddings = tf.reshape(node_embeddings, [batch_size, -1, self._gnn.num_units])
      node_embeddings = tf.gather(node_embeddings, 0, axis=1) # extract ego state
      actions = tf.reshape(actions, [batch_size, -1])
    else:
      actions = tf.zeros([0, actions.shape[-1]])

    t0 = time.time()
    actions, network_state = self._encoder(
      actions,
      step_type=step_type,
      network_state=network_state,
      training=training)
    self.encoder_call_times.append(time.time() - t0)

    t0 = time.time()
    joint = tf.concat([node_embeddings, actions], 1)
    for layer in self._joint_layers:
      joint = layer(joint, training=training)
    self.joint_call_times.append(time.time() - t0)
    
    output = tf.transpose(joint)

    self.call_times.append(time.time() - t1)
    return output, network_state