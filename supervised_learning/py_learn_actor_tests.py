# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import pickle
import logging
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import time
import networkx as nx
import tensorflow as tf
import datetime

# BARK imports
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent, BehaviorPPOAgent, BehaviorGraphSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner, PPORunner
from bark_ml.observers.graph_observer import GraphObserver
from supervised_learning.data_generation import DataGenerator

# Supervised imports
from supervised_learning.actor_nets import ConstantActorNet, RandomActorNet, \
    get_GNN_SAC_actor_net, get_SAC_actor_net
from supervised_learning.data_generation import DataGenerator
from supervised_learning.data_handler import Dataset
from supervised_learning.learner import Learner

class PyGNNActorTests(unittest.TestCase):
    def setUp(self):
        ######################
        #    Parameter       #
        self.log_dir = "/home/silvan/working_bark/supervised_learning/logs/"
        self.epochs = 200
        self.batch_size = 32
        self.train_split = 0.8
        self.data_path = "/home/silvan/working_bark/supervised_learning/data_new/"
        self.num_scenarios = 200
        ######################

        """Setting up the test case"""
        self.params = ParameterServer(filename="examples/example_params/tfa_params.json")
        self.params["ML"]["BehaviorTFAAgents"]["NumCheckpointsToKeep"] = None
        self.params["ML"]["SACRunner"]["EvaluateEveryNSteps"] = 50
        self.params["ML"]["BehaviorSACAgent"]["BatchSize"] = 32
        self.params["ML"]["GraphObserver"]["AgentLimit"] = 8
        self.params["ML"]["BehaviorGraphSACAgent"]["NumLayersGNN"] = 4
        self.params["ML"]["BehaviorGraphSACAgent"]["NumUnitsGNN"] = 256
        self.params["World"]["remove_agents_out_of_map"] = False
        
        # Get dataset
        self.observer = GraphObserver(params=self.params)
        dataset = Dataset(self.data_path, self.observer, self.params, batch_size=self.batch_size,
                         train_split=self.train_split, num_scenarios=self.num_scenarios)
        dataset.get_datasets()
        self.train_dataset = dataset.train_dataset
        self.test_dataset = dataset.test_dataset

        # Calculate some values about dataset
        num_batches = len(list(self.train_dataset.as_numpy_iterator()))
        first = True
        for inputs, labels in self.train_dataset:
            if first==True:
                means = tf.math.reduce_mean(labels, axis=0)
                mins = tf.math.reduce_min(labels, axis=0)
                maxs = tf.math.reduce_max(labels, axis=0)
                first = False
            else:
                means += tf.math.reduce_mean(labels, axis=0)
                mins = tf.math.minimum(mins, tf.math.reduce_min(labels, axis=0))
                maxs = tf.math.maximum(maxs, tf.math.reduce_max(labels, axis=0))
        means = (means / num_batches).numpy()

        # Get agent
        #self.actor_net = get_GNN_SAC_actor_net(self.num_scenarios, self.params, self.observer)
        #self.actor_net = get_SAC_actor_net(self.num_scenarios, self.params, self.observer)
        self.actor_net = ConstantActorNet(constants=means)
        #self.actor_net = RandomActorNet()

        # Get supervised learner
        self.supervised_learner = Learner(self.actor_net, self.train_dataset, self.test_dataset, self.log_dir)
    
    def test_actor_network(self):
        # Evaluates actor net formalia
        actor_net = self.actor_net
        self.assertIsNotNone(actor_net)
    
    def test_training(self):
        logging.info(self.actor_net.__class__==ConstantActorNet)
        if (self.actor_net.__class__ == ConstantActorNet) or (self.actor_net.__class__==RandomActorNet):
            mode = "Number"
            only_test = True
        else:
            mode = "Distribution"
            only_test = False

        self.supervised_learner.train(epochs=self.epochs, only_test=only_test, mode=mode)
        title1 = str(self.actor_net.__class__) + " TRAIN DATASET"
        self.supervised_learner.visualize_predictions(self.train_dataset, title=title1, mode=mode)
        title2 = str(self.actor_net.__class__) + " TEST DATASET"
        self.supervised_learner.visualize_predictions(self.test_dataset, title=title2, mode=mode)
if __name__ == '__main__':
    unittest.main()