# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

from bark.models.behavior import BehaviorModel, BehaviorMPContinuousActions, \
  PrimitiveConstAccStayLane, PrimitiveConstAccChangeToLeft, PrimitiveConstAccChangeToRight
from bark.models.dynamic import SingleTrackModel
from bark_ml.commons.py_spaces import Discrete


class BehaviorDiscreteML(BehaviorMPContinuousActions):
  def __init__(self,
               params=None):
    BehaviorMPContinuousActions.__init__(
      self,
      params)
    self._params = params

    # add motion primitives
    for acc in np.linspace(-3., 3., num=8):
      for steering_rate in np.linspace(-.2, .2, num=7):
        super().AddMotionPrimitive(
          np.array([acc, steering_rate], dtype=np.float32))

  @property
  def action_space(self):
    return Discrete(self.GetNumMotionPrimitives(None))
