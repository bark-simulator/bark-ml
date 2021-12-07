# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import os
from bark_ml.experiment.experiment import Experiment
from bark_ml.experiment.experiment_runner import ExperimentRunner


class PyExperimentTests(unittest.TestCase):
  def test_experiment(self):
    experiment = Experiment(
      os.path.join(os.path.dirname(__file__),
      "data/highway_gnn.json"))

  def test_experiment_runner(self):
    exp_runner = ExperimentRunner(
      json_file=os.path.join(os.path.dirname(__file__),
      "data/highway_gnn.json"), mode="print", random_seed=0)


if __name__ == '__main__':
  unittest.main()