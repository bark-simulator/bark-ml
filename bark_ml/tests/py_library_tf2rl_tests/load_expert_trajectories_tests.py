# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import math
import joblib
import unittest
import numpy as np
from gym.spaces.box import Box
from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories import *
from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import *
from bark_project.bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver


class LoadExpertTrajectoriesTest(unittest.TestCase):
  """
  Tests for the tf2rl utils.
  """

  def test_assert_file_exists(self):
    """
    Test: Assert that a non existent file raises an error.
    """
    with self.assertRaises(AssertionError):
      load_expert_trajectories('/tmp/')

  def test_file_contains_expert_trajectories(self):
    """
    Test: Assert that error is thrown if the file does not
            contain expert trajectories.
    """
    file_path = os.path.join(
        os.path.dirname(__file__),
        'data', 'invalid_file.jblb')
    with self.assertRaises(KeyError):
      load_expert_trajectories(os.path.dirname(file_path))


if __name__ == '__main__':
  unittest.main()
