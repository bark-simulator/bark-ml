# Copyright (c) 2019 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

from bark.models.behavior import BehaviorModel, BehaviorMPMacroActions
from bark.models.dynamic import SingleTrackModel
from bark_ml.commons.py_spaces import Discrete


class DiscreteMLBehavior(BehaviorMPMacroActions):
  def __init__(self,
               dynamic_model=None,
               params=None):
    BehaviorMPMacroActions.__init__(
      self,
      dynamic_model,
      params)
    self._params = params
  
  def Reset(self):
    control_inputs =self._params["DiscreteMLBehavior"]["MotionPrimitives",
      "Motion primitives available as discrete actions", \
      [[4.,0.], [2.,0.],[-0.5,0.],[-1.,0.]]]
    for control_input in control_inputs:
      self.AddMotionPrimitive(np.array(control_input))

  @property
  def action_space(self):
    return Discrete(self._behavior.GetNumMotionPrimitives(None))