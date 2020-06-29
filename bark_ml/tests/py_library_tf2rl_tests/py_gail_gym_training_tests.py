import unittest
import gym
import os

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner 
from bark_ml.library_wrappers.lib_tf2rl.agents.gail_agent import BehaviorGAILAgent

# TF2RL imports:
from tf2rl.experiments.utils import restore_latest_n_traj


class PyTrainingGymTests(unittest.TestCase):

    def setUp(self):
        """
        setup
        """
        self.params = ParameterServer(filename=os.path.join(os.path.dirname(__file__), "gail_data/params/gail_params_open-ai.json"))

        # creating the dirs for logging if they are not present already:
        if not os.path.exists(self.params["ML"]["GAILRunner"]["tf2rl"]["logdir"]):
            os.makedirs(self.params["ML"]["GAILRunner"]["tf2rl"]["logdir"])
        if not os.path.exists(self.params["ML"]["GAILRunner"]["tf2rl"]["model_dir"]):
            os.makedirs(self.params["ML"]["GAILRunner"]["tf2rl"]["model_dir"])
        if not os.path.exists(self.params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"]):
            os.makedirs(self.params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"])

        if len(os.listdir(self.params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"])) == 0:
            print("No expert trajectories found, plaese generate demonstrations first")
            print("python tf2rl/examples/run_sac.py --env-name=Pendulum-v0 --save-test-path --test-interval=50000")
            print("After that, save expert trajectories into bark_ml/tests/gail_data/expert_data")
            exit()

        # create example environment:
        env_name = "Pendulum-v0"
        self.env = gym.make(env_name) 

        # getting the expert trajectories from the .pkl file:
        self.expert_trajs = restore_latest_n_traj(dirname=self.params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"],
                                            max_steps=self.params["ML"]["GAILRunner"]["tf2rl"]["max_steps"])

        # create angent and runner:
        self.agent = BehaviorGAILAgent(
            environment=self.env,
            params=self.params
        )
        
        self.runner = GAILRunner(
            environment=self.env,
            agent=self.agent,
            params=self.params,
            expert_trajs=self.expert_trajs)

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
        raise NotImplementedError("Intended to fail!\nOpen bark_ml/tests/py_library_tf2rl_tests/py_gail_runner_tests.py and see comments.")


    def test_training(self):
        """
        tests the Train() method of the GAILRunner class with an Open-AI gym environment.
        """
        self.runner.Train()

        # I moved most code to the setup function, as there was no real testing happening.
        # Please add code that uses the unittest assertion methods like self.assertEqual(...)
        # To test actual functionality
        # 
        # Just running the Train method is a good sanity check if everything compiles 
        # and runs as expected, but this is not a test. A test runs some methods, gets the result and 
        # checks if the result is expected, by for example hardcoding the expected values in the test,
        # then running the method to get some actual values and then check for equality with self.assertEqual(...)
        raise NotImplementedError("Intended to fail!\nOpen bark_ml/tests/py_library_tf2rl_tests/py_gail_gym_training_tests.py and see comments.")


if __name__ == '__main__':
    unittest.main()