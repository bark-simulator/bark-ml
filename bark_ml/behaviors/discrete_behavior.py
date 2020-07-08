# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

from bark.core.models.behavior import BehaviorModel, BehaviorMPContinuousActions
from bark.core.models.dynamic import SingleTrackModel
from bark_ml.commons.py_spaces import Discrete


class BehaviorDiscreteML(BehaviorMPContinuousActions):
  def __init__(self,
               params=None):
    BehaviorMPContinuousActions.__init__(
      self,
      params)
    self._min_max_acc = params["ML"]["BehaviorDiscreteML"][
      "MinMaxAcc", "", [-3., 3.]]
    self._acc_d_steps = params["ML"]["BehaviorDiscreteML"][
      "AccDiscretizationSteps", "", 10]
    self._min_max_steer = params["ML"]["BehaviorDiscreteML"][
      "MinMaxSteeringRate", "", [-.2, .2]]
    self._steer_d_steps = params["ML"]["BehaviorDiscreteML"][
      "SteeringRateDiscretizationSteps", "", 5]

    # add motion primitives
    for acc in np.linspace(
      self._min_max_acc[0], self._min_max_acc[1], self._acc_d_steps):
      for steering_rate in np.linspace(
        self._min_max_steer[0], self._min_max_steer[1], self._steer_d_steps):
        super().AddMotionPrimitive(
          np.array([acc, steering_rate], dtype=np.float32))

  @property
  def action_space(self):
    return Discrete(self.GetNumMotionPrimitives(None))
