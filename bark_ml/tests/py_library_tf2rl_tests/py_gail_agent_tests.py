import unittest
import gym
import os

# BARK imports
from bark_project.bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.agents.gail_agent import BehaviorGAILAgent

# TF2RL imports
from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.gail import GAIL
from tf2rl.experiments.utils import restore_latest_n_traj


class PyLibraryWrappersGAILAgentTests(unittest.TestCase):
  """
  Tests of some gail agent methods
  """

  def setUp(self):
    """
    setup
    """
    self.params = ParameterServer(filename=os.path.join(os.path.dirname(
        __file__), "gail_data/params/gail_params_bark.json"))

    # create environment
    env_name = "Pendulum-v0"
    self.env = gym.make(env_name)

    # create agent:
    self.agent = BehaviorGAILAgent(environment=self.env, params=self.params)

  def test_agent_wrapping(self):
    """
    tests generator and discriminator instances in __init__() method of BehaviorGAILAgent class
    """
    self.assertIsInstance(self.agent.generator, DDPG)
    self.assertIsInstance(self.agent.discriminator, GAIL)

  def test_agent_parameters(self):
    """
    tests some passed parameters of __init__() method of BehaviorGAILAgent class
    """
    self.assertIsInstance(self.env.observation_space.shape, tuple)
    self.assertGreater(self.env.observation_space.shape[0], 0)
    self.assertIsInstance(self.env.action_space.high.size, int)
    self.assertGreater(self.env.action_space.high.size, 0)
    self.assertGreater(self.agent.generator.n_warmup, 0)
    self.assertGreater(self.agent.generator.batch_size, 0)

    self.assertGreater(self.agent.discriminator.batch_size, 0)


if __name__ == '__main__':
  unittest.main()
