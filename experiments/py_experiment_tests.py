# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
from experiments.experiment import Experiment


class PyExperimentTests(unittest.TestCase):
  def test_experiment_class(self):
    experiment = Experiment("experiments/configs/highway_gnn.json")
    # visualize/evaluate
    # experiment.runner.Run(num_episodes=1, render=False)
    # experiment.params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = \
    #   "/Users/hart/Development/bark-ml/checkpoints_merge_spektral_att3/"
    # experiment.params["ML"]["TFARunner"]["SummaryPath"] = \
    #   "/Users/hart/Development/bark-ml/checkpoints_merge_spektral_att3/"
    # experiment.runner.SetupSummaryWriter()
    # experiment.runner.Train()

if __name__ == '__main__':
  unittest.main()