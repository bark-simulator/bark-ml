# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np
import os
import matplotlib
import time

from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteML
from bark_project.modules.runtime.commons.parameters import ParameterServer
from bark.models.dynamic import SingleTrackModel
from bark.world import World, MakeTestWorldHighway


class PyBehaviorTests(unittest.TestCase):
  def test_discrete_behavior(self):
    params = ParameterServer()
    dynamic_model = SingleTrackModel(params)
    discrete_behavior = BehaviorDiscreteML(dynamic_model, params)
    # sets 0-th motion primitive active
    discrete_behavior.ActionToBehavior(0)
    
  def test_cont_behavior(self):
    params = ParameterServer()
    cont_behavior = BehaviorContinuousML(params)
    # sets numpy array as next action
    cont_behavior.ActionToBehavior(np.array([0., 0.]))


if __name__ == '__main__':
  unittest.main()