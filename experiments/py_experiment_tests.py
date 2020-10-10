# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np
import os
import matplotlib
import time
import tensorflow as tf
from graph_nets import utils_tf
from graph_nets.graphs import GraphsTuple
from bark.runtime.commons.parameters import ParameterServer

from experiments.example_experiment.experiment import Experiment


class PyExperimentTests(unittest.TestCase):
  def test_experiment(self):
    experiment = Experiment(
      json_file="experiments/example_experiment/config.json")
    
if __name__ == '__main__':
  unittest.main()