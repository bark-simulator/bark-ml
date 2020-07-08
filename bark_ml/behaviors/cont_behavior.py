# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

from bark.core.models.behavior import BehaviorModel, BehaviorDynamicModel
from bark_ml.commons.py_spaces import BoundedContinuous


class BehaviorContinuousML(BehaviorDynamicModel):
  def __init__(self,
               params=None):
    BehaviorDynamicModel.__init__(self, params)
    self._lower_bounds = params["ML"]["BehaviorContinuousML"][
      "ActionsLowerBound",
      "Lower-bound for actions.",
      [-5.0, -0.2]]
    self._upper_bounds = params["ML"]["BehaviorContinuousML"][
      "ActionsUpperBound",
      "Upper-bound for actions.",
      [4.0, 0.2]]

  @property
  def action_space(self):
    return BoundedContinuous(
      2,  # acceleration and steering-rate
      low=np.array(self._lower_bounds, dtype=np.float32),
      high=np.array(self._upper_bounds, dtype=np.float32))