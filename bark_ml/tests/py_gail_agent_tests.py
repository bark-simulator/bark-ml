import unittest
import gym
import os

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.agents.gail_agent import BehaviorGAILAgent

# TF2RL imports
from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.gail import GAIL
from tf2rl.experiments.utils import restore_latest_n_traj

class PyLibraryWrappersGAILAgentTests(unittest.TestCase):

  def test_agent_wrapping(self):
    """tests __init__() method of BehaviorGAILAgent class"""

    params = ParameterServer(filename="bark_ml/tests/gail_data/params/gail_params_open-ai.json")

    # create environment
    env_name = "Pendulum-v0"
    env = gym.make(env_name) 

    # create agent:
    agent = BehaviorGAILAgent(environment=env, params=params)


if __name__ == '__main__':
  unittest.main()