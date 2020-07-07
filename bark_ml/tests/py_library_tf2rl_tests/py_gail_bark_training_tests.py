import unittest
import os
import numpy as np
from gym.spaces.box import Box

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


class PyTrainingGymTests(unittest.TestCase):

    def setUp(self):
        """
        Setup
        """
        self.params = ParameterServer(filename=os.path.join(os.path.dirname(__file__), "gail_data/params/gail_params_bark.json"))

        # creating the dirs for logging if they are not present already:
        for key in ['logdir', 'model_dir', 'expert_path_dir']:
            if not os.path.exists(self.params["ML"]["GAILRunner"]["tf2rl"][key]):
                os.makedirs(self.params["ML"]["GAILRunner"]["tf2rl"][key])

        # create environment
        self.bp = ContinuousMergingBlueprint(self.params,
                                        number_of_senarios=500,
                                        random_seed=0)
        self.env = SingleAgentRuntime(blueprint=self.bp,
                                render=False)

        # wrapped environment for compatibility with tf2rl
        self.wrapped_env = TF2RLWrapper(self.env)
        
        # Dummy expert trajectories:
        self.expert_trajs = {
            'obses': np.zeros((1000, 16), dtype=np.float32),
            'next_obses': np.zeros((1000, 16), dtype=np.float32),
            'acts': np.zeros((1000, 2), dtype=np.float32)
        }

        # create agent and runner:
        self.agent = BehaviorGAILAgent(
            environment=self.wrapped_env,
            params=self.params
        )
        self.env.ml_behavior = self.agent
        self.runner = GAILRunner(
            environment=self.wrapped_env,
            agent=self.agent,
            params=self.params,
            expert_trajs=self.expert_trajs)

    def test_TF2RLWrapper(self):
        """
        tests the wrapper class.
        """
        self.assertIsInstance(self.wrapped_env.action_space, Box)
        self.assertIsInstance(self.wrapped_env.observation_space, Box)
        self.assertTrue((self.wrapped_env.action_space.high == self.env.action_space.high).all())
        self.assertTrue((self.wrapped_env.action_space.low == self.env.action_space.low).all())
        


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
        #self.runner.Train()
        #raise NotImplementedError("Intended to fail!\nOpen bark_ml/tests/py_library_tf2rl_tests/py_gail_bark_training_tests.py and see comments.")


if __name__ == '__main__':
    unittest.main()