import sys
import logging
import time
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

# tf2rl imports
import tf2rl


class BehaviorTF2RLAgent:
  """Base class for agents based on the tf2rl library."""

  def __init__(self,
                environment=None,
                params=None):
      self._params = params
      self._environment = environment

    # TODO
    # not sure whether these are needed or not:
    # these are methods and variables of the BehaviorTFAAgent class
    # can be, that these things are only needed when tf_agents implementation is used.
    """
    self._wrapped_env = tf_py_environment.TFPyEnvironment(
    TFAWrapper(self._environment))
    self._ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64))
    self._agent = self.GetAgent(self._wrapped_env, params)
    self._ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                    agent=self._agent)
    self._ckpt_manager = self.GetCheckpointer()
    self._logger = logging.getLogger()
    self._training = False
    """


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
    """Save agent I guess. Has to be implemented here
    Not sure if necessary or not.
    """
    pass


  def Load(self):
    """Load agent I guess.  Has to be implemented here
    Not sure if necessary or not.
    """
    pass




