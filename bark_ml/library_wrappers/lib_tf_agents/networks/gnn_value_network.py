# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gin
import tensorflow as tf  # pylint: disable=unused-import

from tf_agents.networks import network, encoding_network
from bark.runtime.commons.parameters import ParameterServer


@gin.configurable
class GNNValueNetwork(network.Network):
  """Feed Forward value network. Reduces to 1 value output per batch item."""

  def __init__(self,
               input_tensor_spec,
               gnn,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=(75, 40),
               dropout_layer_params=None,
               activation_fn=tf.nn.relu,
               kernel_initializer=None,
               batch_squash=False,
               dtype=tf.float32,
               name='ValueNetwork',
               params=ParameterServer()):
    """
    Creates an instance of `ValueNetwork`.
    Network supports calls with shape outer_rank + observation_spec.shape. Note
    outer_rank must be at least 1.
    Args:
      input_tensor_spec: A `tensor_spec.TensorSpec` or a tuple of specs
        representing the input observations.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
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
      activation_fn: Activation function, e.g. tf.keras.activations.relu,.
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default variance_scaling_initializer
      batch_squash: If True the outer_ranks of the observation are squashed into
        the batch dimension. This allow encoding networks to be used with
        observations with shape [BxTx...].
      dtype: The dtype to use by the convolution and fully connected layers.
      name: A string representing name of the network.
    Raises:
      ValueError: If input_tensor_spec is not an instance of network.InputSpec.
    """
    super(GNNValueNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)

    if not kernel_initializer:
      kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform()

    if gnn is None:
      raise ValueError('`gnn` must not be `None`.')

    self._gnn = gnn(name=name + "_GNN", params=params)

    self._encoder = encoding_network.EncodingNetwork(
      input_tensor_spec=tf.TensorSpec([None, self._gnn._embedding_size]),
      preprocessing_layers=None,
      preprocessing_combiner=None,
      conv_layer_params=conv_layer_params,
      fc_layer_params=fc_layer_params,
      dropout_layer_params=dropout_layer_params,
      activation_fn=tf.keras.activations.relu,
      kernel_initializer=kernel_initializer,
      batch_squash=False,
      dtype=tf.float32)

    self._postprocessing_layers = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.compat.v1.initializers.random_uniform(
            minval=-0.03, maxval=0.03))

  def call(self, observations, step_type=None, network_state=(), training=False):
    # print(observations)
    if len(tf.shape(observations)) == 3:
      observations = tf.squeeze(observations, axis=0)

    embeddings = self._gnn(observations, training=training)

    if tf.shape(embeddings)[0] > 0:
      embeddings = embeddings[:, 0] # extract ego state

    with tf.name_scope("PPOCriticNetwork"):
      tf.summary.histogram("embedding", embeddings)

    state, network_state = self._encoder(
      embeddings,
      step_type=step_type,
      network_state=network_state,
      training=training)

    value = self._postprocessing_layers(state, training=training)
    value = tf.expand_dims(value, axis=0)
    return tf.squeeze(value, -1), network_state