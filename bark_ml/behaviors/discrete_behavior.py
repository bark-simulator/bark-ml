# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

from bark.models.behavior import BehaviorModel, BehaviorMPMacroActions, \
  PrimitiveConstAccStayLane, PrimitiveConstAccChangeToLeft, PrimitiveConstAccChangeToRight
from bark.models.dynamic import SingleTrackModel
from bark_ml.commons.py_spaces import Discrete


class BehaviorDiscreteML(BehaviorMPMacroActions):
  def __init__(self,
               dynamic_model=None,
               params=None):
    BehaviorMPMacroActions.__init__(
      self,
      dynamic_model,
      params)
    self._params = params
    self._dynamic_model = dynamic_model

    # TODO(@hart): make generic and configurable
    # add motion primitives
    motion_primitives = []
    motion_primitives.append(
      PrimitiveConstAccStayLane(self._params, self._dynamic_model, 0, 0.1))
    motion_primitives.append(
      PrimitiveConstAccChangeToLeft(self._params, self._dynamic_model, 0.1))
    motion_primitives.append(
      PrimitiveConstAccChangeToRight(self._params, self._dynamic_model, 0.1))
    for mp in motion_primitives:
      super().AddMotionPrimitive(mp)

  @property
  def action_space(self):
    return Discrete(self.GetNumMotionPrimitives(None))