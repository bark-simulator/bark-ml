import sys
import logging
import time
import tensorflow as tf
import numpy as np
tf.compat.v1.enable_v2_behavior()

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

# tf agent imports
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf_agents.runners.tfa_runner import TFARunner


class CDQNRunner(TFARunner):
  def __init__(self,
               environment=None,
               agent=None,
               params=None):
    TFARunner.__init__(self,
                       environment=environment,
                       agent=agent,
                       params=params)

  @tf.function
  def _inference(self, input):
    """Q network of the categorical DQN returns a value distribution"""
    """This function calculates the Expectation what is the q-value"""
    q_logits, _ = self.q_model.call(input)
    q_probabilities = tf.nn.softmax(q_logits)
    q_values = tf.reduce_sum(self._agent._agent._support * q_probabilities, axis=-1)
    return q_values

  def _train(self):
    self.q_model = self._agent._agent._q_network
    num_state_dims = np.shape(self._wrapped_env._observation_spec)[0]
    inference = self._inference.get_concrete_function(input=tf.TensorSpec([1, num_state_dims], tf.float32))

    iterator = iter(self._agent._dataset)
    for _ in range(0, self._params["ML"]["CDQNRunner"]["NumberOfCollections", "", 10000]):
      self._agent._training = True
      global_iteration = self._agent._agent._train_step_counter.numpy()
      self._collection_driver.run()
      experience, _ = next(iterator)
      self._agent._agent.train(experience)
      if global_iteration % self._params["ML"]["CDQNRunner"]["EvaluateEveryNSteps", "", 100] == 0:
        self.Evaluate()
        tf.saved_model.save(self.q_model, self._params["ML"]["TFARunner"]["ModelPath"], signatures=inference)
        self._agent.Save()