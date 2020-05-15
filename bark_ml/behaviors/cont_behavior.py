# Copyright (c) 2019 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

from bark.models.behavior import BehaviorModel, BehaviorDynamicModel
from bark_ml.commons.py_spaces import BoundedContinuous
from bark_ml.behaviors.ml_behavior import MLBehavior


class ContinuousMLBehavior(BehaviorModel):
  def __init__(self,
               params=None):
    BehaviorModel.__init__(self, params)
    self._behavior = BehaviorDynamicModel(params)
    self._lower_bounds = params["ContinuousMLBehavior"][
      "actions_lower_bound",
      "Lower-bound for actions.",
      [-0.5, -0.01]]
    self._upper_bounds = params["ContinuousMLBehavior"][
      "actions_upper_bound",
      "Upper-bound for actions.",
      [0.5, 0.01]]
    
  def ActionToBehavior(self, action):
    self._behavior.ActionToBehavior(action)

  def Plan(self, observed_world, dt):
    return self._behavior.Plan(observed_world, dt)

  def Clone(self):
    return self

  @property
  def action_space(self):
    return BoundedContinuous(
      self._control_inputs*action_num,
      low=np.array(self._lower_bounds, dtype=np.float32),
      high=np.array(self._upper_bounds, dtype=np.float32))