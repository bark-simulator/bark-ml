import unittest
import os
import numpy as np

# BARK imports
from bark_project.bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.tf2rl_wrapper import TF2RLWrapper
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner 
from bark_ml.library_wrappers.lib_tf2rl.agents.gail_agent import BehaviorGAILAgent
from bark_ml.environments.blueprints import ContinuousMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime

# TF2RL imports:
from tf2rl.experiments.utils import restore_latest_n_traj

# other:
from bark_ml.tests.py_library_tf2rl_tests.py_gail_runner_tests import dir_check


class PyTrainingGymTests(unittest.TestCase):

    def setUp(self):
        """
        Setup
        """
        params = ParameterServer(filename=os.path.join(os.path.dirname(__file__), "gail_data/params/gail_params_bark.json"))

        # creating the dirs for logging if they are not present already:
        dir_check(self.params)

        # create environment
        bp = ContinuousMergingBlueprint(params,
                                        number_of_senarios=500,
                                        random_seed=0)
        env = SingleAgentRuntime(blueprint=bp,
                                render=False)

        # wrapped environment for compatibility with batk-ml
        wrapped_env = TF2RLWrapper(env)
        
        # Dummy expert trajectories:
        expert_trajs = {
            'obses': np.zeros((1000, 16), dtype=np.float32),
            'next_obses': np.zeros((1000, 16), dtype=np.float32),
            'acts': np.zeros((1000, 2), dtype=np.float32)
        }

        # create agent and runner:
        agent = BehaviorGAILAgent(
            environment=wrapped_env,
            params=params
        )
        env.ml_behavior = agent
        runner = GAILRunner(
            environment=wrapped_env,
            agent=agent,
            params=params,
            expert_trajs=expert_trajs)

    def test_initialization(self):
        """
        tests the __init__() method of the GAILRunner and BehaviorGAILAgent classes.
        """
        # I moved all code to the setup function, as there was no real testing happening.
        # Please add code that uses the unittest assertion methods like self.assertEqual(...)
        # To test actual functionality
        # 
        # Just running the __init__ method is a good sanity check if everything compiles 
        # and runs as expected, but this is not a test. A test runs some methods, gets the result and 
        # checks if the result is expected, by for example hardcoding the expected values in the test,
        # then running the method to get some actual values and then check for equality with self.assertEqual(...)
        raise NotImplementedError("Intended to fail!\nOpen bark_ml/tests/py_library_tf2rl_tests/py_gail_bark_training_tests.py and see comments.")


    def test_training(self):
        """
        tests the Train() method of the GAILRunner class with an Open-AI gym environment.
        """
        # I moved all code to the setup function, as there was no real testing happening.
        # Please add code that uses the unittest assertion methods like self.assertEqual(...)
        # To test actual functionality
        # 
        # Just running the train method for some iterations is a good sanity check if everything compiles 
        # and runs as expected, but this is not a test. A test runs some methods, gets the result and 
        # checks if the result is expected, by for example hardcoding the expected values in the test,
        # then running the method to get some actual values and then check for equality with self.assertEqual(...)
        self.runner.Train()
        raise NotImplementedError("Intended to fail!\nOpen bark_ml/tests/py_library_tf2rl_tests/py_gail_bark_training_tests.py and see comments.")


if __name__ == '__main__':
    unittest.main()