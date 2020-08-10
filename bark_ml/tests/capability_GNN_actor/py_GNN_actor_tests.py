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
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent,\
  BehaviorPPOAgent, BehaviorGraphSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner, PPORunner
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.library_wrappers.lib_tf_agents.networks.gnn_wrapper import GNNWrapper 

# Supervised specific imports
from bark_ml.tests.capability_GNN_actor.data_generation import DataGenerator
from bark_ml.tests.capability_GNN_actor.actor_nets import ConstantActorNet,\
  RandomActorNet, get_GNN_SAC_actor_net, get_SAC_actor_net
from bark_ml.tests.capability_GNN_actor.data_generation import DataGenerator
from bark_ml.tests.capability_GNN_actor.data_handler import Dataset
from bark_ml.tests.capability_GNN_actor.learner import Learner

class PyGNNActorTests(unittest.TestCase):
  def setUp(self):
    ######################
    #    Parameter       #
    self.log_dir = "/home/silvan/working_bark/supervised/logs"
    self.epochs = 5
    self.batch_size = 32
    self.train_split = 0.8
    self.data_path = "/home/silvan/working_bark/supervised/data"
    self.num_scenarios = 3
    ######################

    """Setting up the test case"""
    params = ParameterServer(filename="examples/example_params/tfa_sac_gnn_example_params.json")
    params["ML"]["SACRunner"]["NumberOfCollections"] = int(1e6)
    params["ML"]["GraphObserver"]["AgentLimit"] = 4
    params["ML"]["BehaviorGraphSACAgent"]["DebugSummaries"] = False
    params["ML"]["BehaviorGraphSACAgent"]["BatchSize"] = 128
    params["ML"]["BehaviorGraphSACAgent"]["CriticJointFcLayerParams"] = [128, 128]
    params["ML"]["BehaviorGraphSACAgent"]["CriticObservationFcLayerParams"] = [128, 128]
    params["ML"]["BehaviorGraphSACAgent"]["ActorFcLayerParams"] = [128, 64]

    # GNN parameters
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["NumMpLayers"] = 1
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["MpLayersHiddenDim"] = 128

    gnn_library = GNNWrapper.SupportedLibrary.spektral # "tf2_gnn" or "spektral"
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["library"] = gnn_library 

    # (n_nodes, n_features, n_edge_features)
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["GraphDimensions"] = (4, 11, 4)

    if gnn_library == GNNWrapper.SupportedLibrary.spektral:
      params["ML"]["BehaviorGraphSACAgent"]["GNN"]["KernelNetUnits"] = [512, 256]
      params["ML"]["BehaviorGraphSACAgent"]["GNN"]["MPLayerActivation"] = "relu"
      params["ML"]["GraphObserver"]["SelfLoops"] = True # add self-targeted edges to graph nodes
    elif gnn_library == GNNWrapper.SupportedLibrary.tf2_gnn:
      # NOTE: when using the ggnn mp class, MPLayerUnits must match n_features!
      params["ML"]["BehaviorGraphSACAgent"]["GNN"]["message_calculation_class"] = "rgcn"
      params["ML"]["BehaviorGraphSACAgent"]["GNN"]["global_exchange_mode"] = "gru"
      params["ML"]["BehaviorGraphSACAgent"]["GNN"]["dense_every_num_layers"] = -1 # no dense layers between mp layers
      params["ML"]["BehaviorGraphSACAgent"]["GNN"]["global_exchange_every_num_layers"] = 1
      params["ML"]["GraphObserver"]["SelfLoops"] = False
    params["World"]["remove_agents_out_of_map"] = False
    self.params = params
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
      if first:
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
    self.actor_net = get_GNN_SAC_actor_net(self.num_scenarios, self.params, self.observer)
    #self.actor_net = get_SAC_actor_net(self.num_scenarios, self.params, self.observer)
    #self.actor_net = ConstantActorNet(constants=means)
    #self.actor_net = RandomActorNet(low=[0, -0.4], high=[0.1, 0.4])

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
