import sys
import logging
import time
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

# BARK imports
from bark.core.models.behavior import BehaviorModel

# tf2rl imports
import tf2rl

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.tf2rl_wrapper import TF2RLWrapper # needed?


class BehaviorTF2RLAgent:
  """Base class for agents based on the tf2rl library."""

  def __init__(self,
                environment=None,
                params=None):
    self._params = params
    self._environment = environment

    # TODO: Copies from BehaviorTFAAgent class needed?
    # self._wrapped_env = tf_py_environment.TFPyEnvironment(
    # TFAWrapper(self._environment))
    # self._ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64))
    # self._agent = self.GetAgent(self._wrapped_env, params)
    # self._ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
    #                                 agent=self._agent)
    # self._ckpt_manager = self.GetCheckpointer()
    # self._logger = logging.getLogger()
    self._training = False
    pass


  def Reset(self):
    """agent specific implemetation"""
    pass


  def Act(self, state):
    """agent specific implemetation"""
    pass


  def Plan(self, observed_world, dt):
    """agent specific implemetation"""
    pass


  def Save(self):
    """TODO: Consider implementation after talking with Feri
    """
    # save_path = self._ckpt_manager.save(
    # global_step=self._agent._train_step_counter)
    # self._logger.info("Saved checkpoint for step {}.".format(
    # int(self._agent._train_step_counter.numpy())))
    pass


  def Load(self):
    """Never called in SAC either so ommited
    """
    pass




