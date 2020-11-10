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
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import FQFAgent
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML


class BaseAgentTests(unittest.TestCase):
  def test_agents(self):
    params = ParameterServer()
    params["ML"]["BaseAgent"]["NumSteps"] = 2
    params["ML"]["BaseAgent"]["MaxEpisodeSteps"] = 2

    bp = DiscreteHighwayBlueprint(params, number_of_senarios=10, random_seed=0)
    env = SingleAgentRuntime(blueprint=bp, render=False)
    env._observer = NearestAgentsObserver(params)
    env._action_wrapper = BehaviorDiscreteMacroActionsML(params)

    fqf_agent = FQFAgent(env=env, params=params)
    fqf_agent.train_episode()

    fqf_agent.save("./save_dir")

    loaded_agent = FQFAgent(save_dir="./save_dir")
    
    loaded_agent_with_env = FQFAgent(env=env, save_dir="./save_dir")
    loaded_agent_with_env.train_episode()

    self.assertEqual(loaded_agent.ml_behavior.action_space.n, fqf_agent.ml_behavior.action_space.n)
    self.assertEqual(loaded_agent.ent_coef, fqf_agent.ent_coef)


if __name__ == '__main__':
  unittest.main()
