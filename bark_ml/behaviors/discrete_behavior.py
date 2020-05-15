# Copyright (c) 2019 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

from bark_project.behavior import BehaviorModel, BehaviorMotionPrimtives
from bark_ml.behaviors import MLBehavior
from bark_ml.commons.py_spaces import Discrete


class DiscreteMLBehavior(MLBehavior):
  def __init__(self,
               params=None)
    super().__init__(self, params)
    self._behavior = BehaviorMotionPrimtives(self._params)

  def ActionToBehavior(self, action):
    self._behavior.ActionToBehavior(action)

  def Plan(self, observed_world, dt):
    return self._behavior.Plan(observed_world, dt)

  def Reset(self):
    control_inputs = params["DiscreteMLBehavior"]["MotionPrimitives",
      "Motion primitives available as discrete actions", \
      [[4.,0.], [2.,0.],[-0.5,0.],[-1.,0.]]]
    self._behavior.AddMotionPrimitive(np.array(control_input))

  @property
  def action_space(self):
    return Discrete(self._behavior.GetNumMotionPrimitives(None))