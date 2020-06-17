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

    def test_initialization(self):
        """tests the __init__() method of the GAILRunner and BehaviorGAILAgent classes."""

        params = ParameterServer(filename="bark_ml/tests/gail_data/params/gail_params_open-ai.json")

        if len(os.listdir(params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"])) == 0:
            print("No expert trajectories found, plaese generate demonstrations first")
            print("python tf2rl/examples/run_sac.py --env-name=Pendulum-v0 --save-test-path --test-interval=50000")
            print("After that, save expert trajectories into bark_ml/tests/gail_data/expert_data")
            exit()

        # create example environment:
        env_name = "Pendulum-v0"
        env = gym.make(env_name) 

        # getting the expert trajectories from the .pkl file:
        expert_trajs = restore_latest_n_traj(dirname=params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"],
                                            max_steps=params["ML"]["GAILRunner"]["tf2rl"]["max_steps"])

        # create angent and runner:
        agent = BehaviorGAILAgent(
            environment=env,
            params=params
        )
        
        runner = GAILRunner(
            environment=env,
            agent=agent,
            params=params,
            expert_trajs=expert_trajs)


    def test_training(self):
        """tests the Train() method of the GAILRunner class with an Open-AI gym environment."""

        params = ParameterServer(filename="bark_ml/tests/gail_data/params/gail_params_open-ai.json")

        if len(os.listdir(params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"])) == 0:
            print("No expert trajectories found, plaese generate demonstrations first")
            print("python tf2rl/examples/run_sac.py --env-name=Pendulum-v0 --save-test-path --test-interval=50000")
            print("After that, save expert trajectories into bark_ml/tests/gail_data/expert_data")
            exit()

        # create environment:
        env_name = "Pendulum-v0"
        env = gym.make(env_name) 

        # getting the expert trajectories from the .pkl file:
        expert_trajs = restore_latest_n_traj(dirname=params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"],
                                            max_steps=params["ML"]["GAILRunner"]["tf2rl"]["max_steps"])

        # create agent and runner:
        agent = BehaviorGAILAgent(
            environment=env,
            params=params
        )
        
        runner = GAILRunner(
            environment=env,
            agent=agent,
            params=params,
            expert_trajs=expert_trajs)

        runner.Train()


if __name__ == '__main__':
    unittest.main()