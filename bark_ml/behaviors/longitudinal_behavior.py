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
               params=None):
    BehaviorMPMacroActions.__init__(
      self,
      params)
    self._params = params

    # add motion primitives
    motion_primitives = []
    for acc in np.linspace(-4., 3., num=10):
      # stay on lane
      super().AddMotionPrimitive(
        PrimitiveConstAccStayLane(self._params, acc))

  @property
  def action_space(self):
    return Discrete(super().GetNumMotionPrimitives(None))