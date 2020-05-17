# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

from bark.models.behavior import BehaviorModel, BehaviorDynamicModel
from bark_ml.commons.py_spaces import BoundedContinuous


class ContinuousMLBehavior(BehaviorDynamicModel):
  def __init__(self,
               params=None):
    BehaviorDynamicModel.__init__(self, params)
    self._lower_bounds = params["ContinuousMLBehavior"][
      "actions_lower_bound",
      "Lower-bound for actions.",
      [-0.5, -0.01]]
    self._upper_bounds = params["ContinuousMLBehavior"][
      "actions_upper_bound",
      "Upper-bound for actions.",
      [0.5, 0.01]]

  @property
  def action_space(self):
    return BoundedContinuous(
      2,  # acceleration and steering-rate
      low=np.array(self._lower_bounds, dtype=np.float32),
      high=np.array(self._upper_bounds, dtype=np.float32))