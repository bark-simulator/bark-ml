# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import gym
from gym import Space
import numpy as np


class Discrete(gym.spaces.Discrete):

    """
    Gym discrete spaces.
    """

    def __init__(self, n):
        self._n = n
        super(Discrete, self).__init__( n)

    def sample(self):
        return self.np_random.randint(self._n)

    @property
    def low(self):
        return 0

    @property
    def high(self):
        return self._n - 1

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and \
         (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self._n

    def __repr__(self):
        return "Discrete(%d)" % self._n

    def __eq__(self, other):
        return isinstance(other, Discrete) and self._n == other._n


class BoundedContinuous(Space):

  """
  Gym continuous spaces.
  """

  def __init__(self,
               n,
               low=None,
               high=None):
    self._n = n
    self._low = low
    self._high = high
    Space.__init__(self, shape=(n,))

  def sample(self):
    if len(self._low) > 1 and len(self._high) > 1:
      sample_vec = []
      for mi, ma in zip(self._low, self._high):
        sample_vec.append(self.np_random.uniform(mi, ma))
      return np.hstack(sample_vec)
    return self.np_random.uniform(size=(self._n,))

  @property
  def low(self):
    return self._low

  @property
  def high(self):
    return self._high

  @property
  def n(self):
    return self._n

  def __repr__(self):
      return "BoundedContinuous(%d)" % self._n

  def __eq__(self, other):
    return isinstance(other, BoundedContinuous) and self._n == other._n