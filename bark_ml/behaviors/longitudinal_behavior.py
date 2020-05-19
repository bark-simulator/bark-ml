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


class BehaviorLongitudinalML(BehaviorMPMacroActions):
  def __init__(self,
               dynamic_model=None,
               params=None):
    BehaviorMPMacroActions.__init__(
      self,
      dynamic_model,
      params)
    self._params = params
    self._dynamic_model = dynamic_model

    # add motion primitives
    motion_primitives = []
    for acc in np.linspace(-4., 3., num=10):
      # stay on lane; acc = 0
      super().AddMotionPrimitive(
        PrimitiveConstAccStayLane(self._params, self._dynamic_model, acc, 2.5))

  @property
  def action_space(self):
    return Discrete(super().GetNumMotionPrimitives(None))