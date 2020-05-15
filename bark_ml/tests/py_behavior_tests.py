# Copyright (c) 2019 Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np
import os
import matplotlib
matplotlib.use('PS')
import time

from modules.runtime.commons.parameters import ParameterServer


class PyBehaviorTests(unittest.TestCase):
  @unittest.skip
  def test_runtime_rl(self):
    params = ParameterServer(
      filename="tests/data/deterministic_scenario_test.json")
    pass

if __name__ == '__main__':
  unittest.main()