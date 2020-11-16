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
import importlib
import matplotlib
import time
import tensorflow as tf
from graph_nets import utils_tf
from graph_nets.graphs import GraphsTuple
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.environments.blueprints.configurable.configurable_scenario import ConfigurableScenarioBlueprint
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.runners.sac_runner import SACRunner
from bark_ml.library_wrappers.lib_tf_agents.agents.graph_sac_agent import BehaviorGraphSACAgent
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from experiments.experiment import Experiment


class PyExperimentTests(unittest.TestCase):
  def test_experiment_class(self):
    experiment = Experiment("experiments/configs/highway_interaction_network.json")
    # visualize/evaluate
    experiment.runner.Run(num_episodes=2, render=False)
    # experiment.params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = \
    #   "/Users/hart/Development/bark-ml/checkpoints_merge_spektral_att3/"
    # experiment.params["ML"]["TFARunner"]["SummaryPath"] = \
    #   "/Users/hart/Development/bark-ml/checkpoints_merge_spektral_att3/"
    # experiment.runner.SetupSummaryWriter()
    # experiment.runner.Train()

if __name__ == '__main__':
  unittest.main()