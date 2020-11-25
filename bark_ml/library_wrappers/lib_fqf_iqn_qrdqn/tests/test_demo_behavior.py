# Copyright (c) 2020 fortiss GmbH
#
# Authors: Julian Bernhard, Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
from bark.core.models.behavior import BehaviorModel

class TestDemoBehavior(BehaviorModel):
  """Dummy Python behavior model
  """
  def __init__(self,
               dynamic_model = None,
               params = None):
    BehaviorModel.__init__(self, params)
    self._dynamic_model = dynamic_model
    self._params = params

  def Plan(self, delta_time, world):
    super(TestDemoBehavior, self).ActionToBehavior(
      np.array([2., 1.], dtype=np.float32))
    trajectory = np.array([[0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.]], dtype=np.float32)
    super(TestDemoBehavior, self).SetLastTrajectory(trajectory)
    return trajectory

  def Clone(self):
    return self

  def GetLastMacroAction(self):
    return 22

  def __setstate__(self, d):
    pass

  def __getstate__(self):
    return {}