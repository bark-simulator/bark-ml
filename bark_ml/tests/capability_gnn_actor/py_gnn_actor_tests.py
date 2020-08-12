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
from bark_ml.tests.capability_gnn_actor.data_generation import DataGenerator
from bark_ml.tests.capability_gnn_actor.actor_nets import ConstantActorNet,\
  RandomActorNet
from bark_ml.tests.capability_gnn_actor.data_generation import DataGenerator
from bark_ml.tests.capability_gnn_actor.data_handler import SupervisedData
from bark_ml.tests.capability_gnn_actor.learner import Learner

class PyGNNActorTests(unittest.TestCase):
  def setUp(self):
    """Setting up the test cases"""
    ######################
    #    Parameter       #
    self._log_dir = "/home/silvan/working_bark/supervised/logs"
    #self._epochs = 500
    self._batch_size = 32
    self._train_split = 0.8
    #self._data_path = "/home/silvan/working_bark/supervised/data"
    self._num_scenarios = 2
    ######################
    
    #filename = "examples/example_params/tfa_sac_gnn_spektral_default.json"
    filename = "examples/example_params/tfa_sac_gnn_tf2_gnn_default.json"

    self._params = ParameterServer(filename=filename)
    self._observer = GraphObserver(params=self._params)
    bp = ContinuousHighwayBlueprint(self._params,
                                    number_of_senarios=2,
                                    random_seed=0)
    env = SingleAgentRuntime(blueprint=bp, observer=self._observer,
                             render=False)
    # Get GNN SAC actor net
    sac_agent = BehaviorGraphSACAgent(environment=env, observer=self._observer,
                                      params=self._params)
    self.gnn_actor = sac_agent._agent._actor_network

    # Get SAC actor net
    #sac_agent = BehaviorSACAgent(environment=env, params=self._params)
    #self.sac_actor = sac_agent._agent._actor_network
  
  def test_actor_network(self):
    """Evaluates existence of gnn_actor"""
    self.assertIsNotNone(self.gnn_actor)

  def test_actor_overfitting(self):
    """Checks if GNN_actor is capable of overfitting anything by comparing
    it on a very small dataset with a random actor and an actor outputting
    the mean of the dataset.
    
    This test trains an GNN SAC agent for 500 epochs and considers the average
    of train losses per epoch for the last 20 epochs for comparison. This
    average is then compared with the average train losses of Random and 
    Constant actors."""
    
    # Build very small dataset with 2 scenarios
    dataset = SupervisedData(self._observer, self._params,
                             data_path=None,
                             batch_size=self._batch_size,
                             train_split=0.8,
                             num_scenarios=5)
    self._train_dataset = dataset._train_dataset
    self._test_dataset = dataset._test_dataset

    # Get benchmark data (run Constant Actor)
    constant_actor = ConstantActorNet(dataset=self._train_dataset)
    learner1 = Learner(constant_actor, self._train_dataset, self._test_dataset)
    losses_constant = learner1.train(epochs=50, only_test=True,
                                     mode="Number")
    # Get benchmark data (run RandomActor)
    random_actor = RandomActorNet(low=[0, -0.4], high=[0.1, 0.4])
    learner2 = Learner(random_actor, self._train_dataset, self._test_dataset)
    losses_random = learner2.train(epochs=50, only_test=True,
                                   mode="Number")
    # Train GNN_Actor
    gnn_actor = self.gnn_actor
    learner3 = Learner(gnn_actor, self._train_dataset, self._test_dataset,
                       log_dir=self._log_dir)
    losses_gnn = learner3.train(epochs=500, only_test=False,
                                mode="Distribution")

    # Compare the losses (means) for training (first list in losses!)
    avg_train_loss_constant = np.mean(np.array(losses_constant[0]))
    avg_train_loss_random = np.mean(np.array(losses_random[0]))
    # Consider only last 10 epochs of GNN for comparison (before: training phase)
    avg_train_loss_gnn = np.mean(np.array(losses_gnn[0][-20:]))
    print("avg_train_loss_constant:", avg_train_loss_constant)
    print("avg_train_loss_random:", avg_train_loss_random)
    print("avg_train_loss_gnn:", avg_train_loss_gnn)
    self.assertLess(avg_train_loss_gnn, avg_train_loss_random)
    self.assertLess(avg_train_loss_gnn, avg_train_loss_constant)


  """def test_benchmark_with_normal_SAC(self):
    dataset = SupervisedData(self._observer, self._params,
                             data_path=self._data_path,
                             batch_size=self._batch_size,
                             train_split=self._train_split,
                             num_scenarios=self._num_scenarios)
    self._train_dataset = dataset._train_dataset
    self._test_dataset = dataset._test_dataset

    gnn_actor = self.gnn_actor
    sac_actor = self.sac_actor
    random_actor = RandomActorNet(low=[0, -0.4], high=[0.1, 0.4])
    constant_actor = ConstantActorNet(dataset=self._train_dataset)

    actor_nets = [gnn_actor, sac_actor, random_actor, constant_actor]
    for actor_net in actor_nets:
      time.sleep(1)
      learner = Learner(actor_net, self._train_dataset, self._test_dataset,
                        log_dir=self._log_dir)
      if (actor_net.__class__ == ConstantActorNet) or \
            (actor_net.__class__==RandomActorNet):
        mode = "Number"
        only_test = True
      else:
        mode = "Distribution"
        only_test = False

      learner.train(epochs=self._epochs, only_test=only_test, mode=mode)
      #title1 = str(self.actor_net.__class__) + " TRAIN DATASET"
      #learner.visualize_predictions(self._train_dataset, title=title1, mode=mode)
      #title2 = str(self.actor_net.__class__) + " TEST DATASET"
      #learner.visualize_predictions(self._test_dataset, title=title2, mode=mode)"""
    
                                        
if __name__ == '__main__':
    unittest.main()
