# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np
import os
import matplotlib
import time
import networkx as nx

from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteML
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, DiscreteHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.library_wrappers.lib_tf2_gnn import GNNWrapper

class PyGNNWrapperTests(unittest.TestCase):

  def test_forward_pass(self):
    gnn = GNNWrapper()
    observation = GraphObserver().sample()


    assert False, "Test"

if __name__ == '__main__':
  unittest.main()