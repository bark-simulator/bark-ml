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
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import IQNAgent, FQFAgent, QRDQNAgent

from libs.evaluation.training_benchmark_database import TrainingBenchmarkDatabase

class EvaluationTests(unittest.TestCase):
  # make sure the agent works
  def test_agent_wrapping(self):
    params = ParameterServer()
    env = gym.make("highway-v1", params=params)
    env.reset()
    params["ML"]["BaseAgent"]["MaxEpisodeSteps"] = 2
    params["ML"]["BaseAgent"]["NumEvalEpisodes"] = 2
    train_bench = TrainingBenchmarkDatabase()
    agent = FQFAgent(env=env, params=params, training_benchmark=train_bench)
    agent.train_episode()
    agent.evaluate()


if __name__ == '__main__':
  unittest.main()
