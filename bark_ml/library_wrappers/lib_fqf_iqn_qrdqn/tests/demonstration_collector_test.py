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
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent.demonstrations import DemonstrationCollector
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML

import bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.tests.test_demo_behavior
from bark_ml.evaluators.evaluator import StateEvaluator

class TestEvaluator(StateEvaluator):
  reach_goal = True
  def __init__(self,
               params=ParameterServer()):
    StateEvaluator.__init__(self, params)
    self.step = 0

  def _evaluate(self, observed_world, eval_results, action):
    """Returns information about the current world state
    """
    done = False
    reward = 0.0
    info = {"goal_r1" : False}
    if self.step > 2:        
      done = True
      if self.reach_goal:
        reward = 0.1
        info = {"goal_r1" : True}
    self.step += 1
    return reward, done, info
    
  def Reset(self, world):
    self._step = 0
    #every second scenario goal is not reached
    TestEvaluator.reach_goal = not TestEvaluator.reach_goal 


class DemonstrationCollectorTests(unittest.TestCase):
  def test_collect_demonstrations(self):
    params = ParameterServer()
    bp = DiscreteHighwayBlueprint(params, number_of_senarios=10, random_seed=0)
    env = SingleAgentRuntime(blueprint=bp, render=False)
    env._observer = NearestAgentsObserver(params)
    env._action_wrapper = BehaviorDiscreteMacroActionsML(params)
    env._evaluator = TestEvaluator()
    
    demo_behavior = bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.\
            tests.test_demo_behavior.TestDemoBehavior(params)
    collector = DemonstrationCollector()
    collection_result = collector.CollectDemonstrations(env, demo_behavior, 4, "./test_demo_collected", \
           use_mp_runner=False, runner_init_params={"deepcopy" : False})
    self.assertTrue(os.path.exists("./test_demo_collected/collection_result"))
    print(collection_result.get_data_frame().to_string())

    experiences = collector.ProcessCollectionResult(eval_criteria = {"goal_r1" : lambda x : x})
    # expected length = 2 scenarios (only every second reaches goal) x 3 steps (4 executed, but first not counted)
    self.assertEqual(len(experiences), 2*3) 

    collector.dump("./final_collections")

    loaded_collector = DemonstrationCollector.load("./final_collections")
    experiences_loaded = loaded_collector.GetDemonstrationExperiences()
    print(experiences_loaded)
    self.assertEqual(len(experiences_loaded), 2*3) 


if __name__ == '__main__':
  unittest.main()
