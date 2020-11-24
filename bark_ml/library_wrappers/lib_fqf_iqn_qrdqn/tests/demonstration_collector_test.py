# Copyright (c) 2020 fortiss GmbH
#
# Authors: Julian Bernhard, Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

try:
    import debug_settings
except:
    pass


import unittest
import numpy as np
import os
import gym
import matplotlib
import time

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.environments.blueprints import \
  DiscreteHighwayBlueprint, DiscreteMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
import bark_ml.environments.gym
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.demonstrations import DemonstrationCollector
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML

from bark.world.tests.python_behavior_model import PythonDistanceBehavior

class TestDemoBehavior(PythonDistanceBehavior):
  def GetLastMacroAction(self):
    return 22

class DemonstrationCollectorTests(unittest.TestCase):
  def test_collect_demonstrations(self):
    bp = DiscreteHighwayBlueprint(params, number_of_senarios=10, random_seed=0)
    env = SingleAgentRuntime(blueprint=bp, render=False)
    env._observer = NearestAgentsObserver(params)
    env._action_wrapper = BehaviorDiscreteMacroActionsML(params)
    
    collector = DemonstrationCollector()
    collector.CollectDemonstrations(env, demo_behavior, 2, 10, "./test_demo_creation")

if __name__ == '__main__':
  unittest.main()
