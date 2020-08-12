# Copyright (c) 2020 fortiss GmbH
#
# Authors: Silvan Wimmer
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import time
import tensorflow as tf

class RandomActorNet:
  """RandomActor which returns uniformely sampled random action when called.
 
  Args:
    low: int, float, double or array, lower bound of random actions.
    high: int, float, double or array, upper bound of random actions.
  """
  def __init__(self, low=-0.4, high=0.4):
    self.low = low
    self.high = high
          
  def __call__(self, inputs, **args):
    """Depending on input shape (batch vs. single call) returning random
    actions.
    
    Args:
      inputs: observation or batch of observations.
      args: (not used, just to be compatible with other classes call methods).
    
    Returns:
      predictions: tf.Tensor of predictions or batch of predictions (same outer
                   shape as inputs).
    """
    del args
    size = (inputs.shape[0], 2)
    predictions = np.random.uniform(low=self.low, high=self.high, size=size)
    predictions = tf.constant(predictions, dtype=tf.float32)
    return predictions

class ConstantActorNet:
  """ConstantActor which returns always two constants.
  
  These constants can either be the means of a dataset, then a dataset should
  be given during initialization. Or manually set constants, then these should
  be given. If None or both are given, an Exception is raised.
  
  Args:
    dataset: a tf.data.Dataset for which the mean should be returned.
    constants: a numpy array of size 2 which should be returned
  """
  def __init__(self, dataset=None, constants=None):
    """Initializes ConstantActorNet"""   
    if dataset is None and constants is not None:
      self.constants = constants
    
    elif dataset is not None and constants is None:
      # Calculate mean of dataset
      num_batches = len(list(dataset.as_numpy_iterator()))
      first = True
      for inputs, labels in dataset:
        if first:
          means = tf.math.reduce_mean(labels, axis=0)
          first = False
        else:
          means += tf.math.reduce_mean(labels, axis=0)
      means = (means / num_batches).numpy()
      self.constants = means

  def __call__(self, inputs, **args):
    """Depending on input shape (batch vs. single call) returning random
    actions.
    
    Args:
      inputs: observation or batch of observations.
      args: (not used, just to be compatible with other classes call methods).
    
    Returns:
      predictions: tf.Tensor of predictions or batch of predictions (same outer
                   shape as inputs).
    """
    del args
    size = (inputs.shape[0], 2)
    predictions = np.resize(self.constants, size)
    predictions = tf.constant(predictions, dtype=tf.float32)
    return predictions