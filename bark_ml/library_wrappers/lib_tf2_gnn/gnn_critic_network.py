import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import network
from tf_agents.networks import utils
from bark_ml.library_wrappers.lib_tf2_gnn import GNNWrapper

@gin.configurable
class GNNCriticNetwork(network.Network):
  """Creates a critic GNN."""

  def __init__(self,
               input_tensor_spec,
               action_fc_layer_params=None,
               action_dropout_layer_params=None,
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

    self._gnn = GNNWrapper(num_layers=2, num_units=80)

    # TODO(kbanoop): Replace mlp_layers with encoding networks.
    self._action_layers = utils.mlp_layers(
      fc_layer_params=action_fc_layer_params
    )

    self._joint_layers = []

    self._joint_layers.append(
        tf.keras.layers.Dense(
            1,
            input_shape=(82,),
            activation=output_activation_fn,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003),
            name='value'))

  def call(self, inputs, step_type=(), network_state=(), training=False):
    del step_type # unused.

    observations, actions = inputs
    batch_size = observations.shape[0]

    actions = tf.cast(actions, tf.float32)
    observations = tf.cast(observations, tf.float32)
    observations = self._gnn.batch_call(observations, training=training)

    if batch_size > 0:
      observations = tf.gather(observations, 0, axis=1) # extract ego state
      observations = tf.reshape(observations, [batch_size, -1])
      actions = tf.reshape(actions, [batch_size, -1])
    else:
      actions = tf.zeros([0, actions.shape[-1]])

    for layer in self._action_layers:
      actions = layer(actions, training=training)

    joint = tf.concat([observations, actions], 1)
    for layer in self._joint_layers:
      joint = layer(joint, training=training)
    
    output = tf.transpose(joint)
    return output, network_state
