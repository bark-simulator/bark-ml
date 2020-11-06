# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

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
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import BaseAgent


class BaseAgentTests(unittest.TestCase):
  def test_agents(self):
    params = ParameterServer()
    params["ML"]["BaseAgent"]["NumSteps"] = 2
    params["ML"]["BaseAgent"]["MaxEpisodeSteps"] = 2

    bp = DiscreteHighwayBlueprint(params, number_of_senarios=10, random_seed=0)
    env = SingleAgentRuntime(blueprint=bp, render=False)

    base_agent = BaseAgent(env=env, params=params)



if __name__ == '__main__':
  unittest.main()
