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
    # TODO docstring
    """
    """

    def setUp(self):
        """
        setup
        """
        self.params = ParameterServer(filename=os.path.join(os.path.dirname(
            __file__), "gail_data/params/gail_params_open-ai.json"))

        # create environment
        env_name = "Pendulum-v0"
        self.env = gym.make(env_name)

        # create agent:
        self.agent = BehaviorGAILAgent(environment=self.env, params=self.params)

    def test_agent_wrapping(self):
        """
        tests __init__() method of BehaviorGAILAgent class
        """
        # TODO Write real tests
        # Just running the __init__ method is a good sanity check if everything compiles
        # and runs as expected, but this is not a test. A test runs some methods, gets the result and
        # checks if the result is expected, by for example hardcoding the expected values in the test,
        # then running the method to get some actual values and then check for equality with self.assertEqual(...)
        raise NotImplementedError("Intended to fail!\nOpen bark_ml/tests/py_library_tf2rl_tests/py_gail_agent_tests.py and see comments.")
        pass


if __name__ == '__main__':
    unittest.main()
