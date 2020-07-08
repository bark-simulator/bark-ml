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

class PyGNNActorTests(unittest.TestCase):

  def setUp(self):
      """Setting up the test case"""
      params = ParameterServer()
      self.observer = GraphObserver(params)

      # Load raw data
      self.data_collection = list()
      data_path = "/home/silvan/working_bark/supervised_learning/data/"
      scenarios = os.listdir(data_path)
      for scenario in scenarios:
          scenario_path = data_path + "/" + scenario
          with open(scenario_path, 'rb') as f:
              data = pickle.load(f)
          self.data_collection.append(data)
      print("Raw data loading completed")

      # Transform raw data into supervised dataset
      self.Y = list()
      self.X = list()
      for data in self.data_collection:
          for data_point in data:
              # Transform raw data to nx.Graph
              graph = data_point["graph"]
              print(graph)
              graph = nx.node_link_graph(graph)
              # Transform graph to observation
              observation = self.observer._observation_from_graph(graph)
              print(observation)
      # Bring data in pipeline
      # load network
      # Train network

  def test_data(self):
      print(len(self.data_collection))
      print((self.data_collection[0]))

if __name__ == '__main__':
  unittest.main()