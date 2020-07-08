import unittest
import gym
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

# TF2RL imports
from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.gail import GAIL
from tf2rl.experiments.irl_trainer import IRLTrainer
from tf2rl.experiments.utils import restore_latest_n_traj

class sample_agent():
    """dummy class just to test the runner. Has got the same values as a normal GAIL agent."""
    def __init__(self, generator, discriminator):
        """initialize tht sample agent."""
        self.generator = generator
        self.discriminator = discriminator


class PyGAILRunnerTests(unittest.TestCase):
    # TODO docstring
    
    def setUp(self):
        """
        setup
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
            'obses': np.zeros((10, 16)),
            'next_obses': np.ones((10, 16)),
            'acts': np.ones((10, 2)) * 2
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
        """tests the wrapper class."""
        self.assertIsInstance(self.wrapped_env.action_space, Box)
        self.assertIsInstance(self.wrapped_env.observation_space, Box)
        self.assertTrue((self.wrapped_env.action_space.high == self.env.action_space.high).all())
        self.assertTrue((self.wrapped_env.action_space.low == self.env.action_space.low).all())

    
    def test_get_trainer(self):
        """get_trainer"""
        trainer = self.runner.GetTrainer()
        # assertions:
        self.assertIsInstance(trainer, IRLTrainer)
        self.assertIsInstance(trainer._irl, GAIL)
        self.assertIsInstance(trainer._policy, DDPG)
        self.assertEqual(trainer._irl, self.agent.discriminator)
        self.assertEqual(trainer._policy, self.agent.generator)
        self.assertTrue((trainer._expert_obs == self.expert_trajs["obses"]).all())
        self.assertTrue((trainer._expert_next_obs == self.expert_trajs["next_obses"]).all())
        self.assertTrue((trainer._expert_act == self.expert_trajs["acts"]).all())

if __name__ == '__main__':
    unittest.main()