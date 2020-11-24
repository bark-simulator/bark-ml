# Copyright (c) 2020 fortiss GmbH
#
# Authors: Julian Bernhard, Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from bark.world.tests.python_behavior_model import PythonDistanceBehavior

class TestDemoBehavior(PythonDistanceBehavior):
  def __init__(self, params):
    super().__init__(params)

  def GetLastMacroAction(self):
    return 22

  def __setstate__(self, d):
    pass

  def __getstate__(self):
    return {}

  def Clone(self):
    return self