# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np

from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML
from bark.runtime.commons.parameters import ParameterServer


class PyBehaviorTests(unittest.TestCase):
  def test_discrete_behavior(self):
    params = ParameterServer()
    discrete_behavior = BehaviorDiscreteMacroActionsML(params)
    # sets 0-th motion primitive active
    discrete_behavior.ActionToBehavior(0)
    print(discrete_behavior.action_space)

  def test_cont_behavior(self):
    params = ParameterServer()
    cont_behavior = BehaviorContinuousML(params)
    # sets numpy array as next action
    cont_behavior.ActionToBehavior(np.array([0., 0.]))
    print(cont_behavior.action_space)


if __name__ == '__main__':
  unittest.main()