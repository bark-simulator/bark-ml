# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest, pickle
import numpy as np
import os
import matplotlib
import time
import networkx as nx
import tensorflow as tf

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
from examples.data_generation import DataGenerator

class PyGNNActorTests(unittest.TestCase):

  def setUp(self):
      """Setting up the test case"""
      params = ParameterServer()
      self.observer = GraphObserver(params)

      # Generate new data (necessary when changes on observer are made)
      #graph_generator = DataGenerator(num_scenarios=100, dump_dir='/home/silvan/working_bark/supervised_learning/data/')
      #graph_generator.run_scenarios()

      # Load raw data
      data_collection = list()
      data_path = "/home/silvan/working_bark/supervised_learning/data/"
      scenarios = os.listdir(data_path)
      for scenario in scenarios:
          scenario_path = data_path + "/" + scenario
          with open(scenario_path, 'rb') as f:
              data = pickle.load(f)
          data_collection.append(data)
      print("Raw data loading completed")

      # Transform raw data to supervised dataset
      self.Y = list()
      self.X = list()
      for data in data_collection:
          for data_point in data:
              # Transform raw data to nx.Graph
              graph_data = data_point["graph"]
              actions = data_point["actions"]
              graph = nx.node_link_graph(graph_data)
              # Transform graph to observation
              observation = self.observer._observation_from_graph(graph)

              # Save in training data variables
              self.X.append(observation)
              self.Y.append(actions)
      
  def test_train_data(self):
      # Evaluate shapes of X and Y
      assert len(self.X) == len(self.Y)
      assert self.X[0].__class__ == tf.python.framework.ops.EagerTensor

  def test_actor_network(self):
      params = ParameterServer(filename="examples/example_params/tfa_params.json")
      params["ML"]["BehaviorTFAAgents"]["NumCheckpointsToKeep"] = None
      params["ML"]["SACRunner"]["EvaluateEveryNSteps"] = 50
      params["ML"]["BehaviorSACAgent"]["BatchSize"] = 32
      params["World"]["remove_agents_out_of_map"] = False
      
      bp = ContinuousHighwayBlueprint(params, number_of_senarios=2500, random_seed=0)
      env = SingleAgentRuntime(blueprint=bp, observer=self.observer, render=False)
      sac_agent = BehaviorGraphSACAgent(environment=env, params=params)
      env.ml_behavior = sac_agent
      tf_agent = sac_agent.GetAgent(sac_agent.env, params)
      actor_network = tf_agent._actor_network

    
      self.assertIsNotNone(actor_network)

if __name__ == '__main__':
  unittest.main()