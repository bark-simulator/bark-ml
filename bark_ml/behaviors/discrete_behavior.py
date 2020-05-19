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
               params=None):
    BehaviorMPMacroActions.__init__(
      self,
      params)
    self._params = params

    # TODO(@hart): make generic and configurable
    # add motion primitives
    motion_primitives = []
    # stay on lane; acc = 0
    motion_primitives.append(
      PrimitiveConstAccStayLane(params, 0))
    # stay on lane; acc = -1.
    motion_primitives.append(
      PrimitiveConstAccStayLane(params, -1.))
    # stay on lane; acc = +1.
    motion_primitives.append(
      PrimitiveConstAccStayLane(params, 1.))
    # change to the left lane
    motion_primitives.append(
      PrimitiveConstAccChangeToLeft(params))
    # change to the right lane
    motion_primitives.append(
      PrimitiveConstAccChangeToRight(params))
    for mp in motion_primitives:
      super().AddMotionPrimitive(mp)

  @property
  def action_space(self):
    return Discrete(super().GetNumMotionPrimitives(None))