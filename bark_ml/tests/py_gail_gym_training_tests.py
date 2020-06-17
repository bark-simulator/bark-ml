import unittest
import gym
import os

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner 
from bark_ml.library_wrappers.lib_tf2rl.agents.gail_agent import BehaviorGAILAgent


class PyTrainingGymTests(unittest.TestCase):

    def test_initialization(self):
        """tests the __init__() method of the GAILRunner and BehaviorGAILAgent classes."""

        params = ParameterServer(filename="bark_ml/tests/gail_data/params/gail_params.json")

        if len(os.listdir(params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"])) == 0:
            print("No expert trajectories found, plaese generate demonstrations first")
            print("python tf2rl/examples/run_sac.py --env-name=Pendulum-v0 --save-test-path --test-interval=50000")
            print("After that, save expert trajectories into bark_ml/tests/gail_data/expert_data")
            exit()

        env_name = "Pendulum-v0"
        env = gym.make(env_name) 

        agent = BehaviorGAILAgent(
            environment=env,
            params=params
        )
        
        runner = GAILRunner(
            environment=env,
            agent=agent,
            params=params)


    def test_training(self):
        """tests the Train() method of the GAILRunner class with an Open-AI gym environment."""

        params = ParameterServer(filename="bark_ml/tests/gail_data/params/gail_params.json")

        if len(os.listdir(params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"])) == 0:
            print("No expert trajectories found, plaese generate demonstrations first")
            print("python tf2rl/examples/run_sac.py --env-name=Pendulum-v0 --save-test-path --test-interval=50000")
            print("After that, save expert trajectories into bark_ml/tests/gail_data/expert_data")
            exit()

        env_name = "Pendulum-v0"
        env = gym.make(env_name) 

        agent = BehaviorGAILAgent(
            environment=env,
            params=params
        )
        
        runner = GAILRunner(
            environment=env,
            agent=agent,
            params=params)

        runner.Train()


if __name__ == '__main__':
    unittest.main()