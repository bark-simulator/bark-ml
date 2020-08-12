import unittest
import gym
import os
import numpy as np
from gym.spaces.box import Box
from pathlib import Path

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


class PyGAILRunnerTests(unittest.TestCase):
  """Tests the GAILRunner class.
  It tests the followings:
      - TF2RLWrapper wraps the BARK runtime correctly.
      - __init__ method of the GAILRunner
      - get_trainer method of the GAILRunner
  """

  def setUp(self):
    """
    setup
    """
    self.params = ParameterServer(
        filename=os.path.join(
            os.path.dirname(__file__),
            "gail_data/params/gail_params_bark.json"))

    # creating the dirs for logging if they are not present already:
    for key in ['logdir', 'model_dir', 'expert_path_dir']:
      self.params["ML"]["GAILRunner"]["tf2rl"][key] = os.path.join(
        Path.home(), self.params["ML"]["GAILRunner"]["tf2rl"][key])
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
        'acts': 2 * np.ones((10, 2))
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

  def test_runner_init(self):
    """tests the init function of the runner."""
    self.assertEqual(self.runner._expert_trajs, self.expert_trajs)
    self.assertEqual(self.runner._agent, self.agent)
    self.assertEqual(self.runner._environment, self.wrapped_env)
    self.assertEqual(self.runner._params, self.params)

  def test_get_trainer(self):
    """get_trainer"""
    trainer = self.runner.GetTrainer()

    self.assertIsInstance(trainer, IRLTrainer)
    self.assertIsInstance(trainer._irl, GAIL)
    self.assertIsInstance(trainer._policy, DDPG)
    self.assertEqual(trainer._irl, self.agent.discriminator)
    self.assertEqual(trainer._policy, self.agent.generator)
    self.assertEqual(trainer._env, self.wrapped_env)

    self.assertTrue((trainer._expert_obs == self.expert_trajs["obses"]).all())
    self.assertTrue((trainer._expert_next_obs ==
                     self.expert_trajs["next_obses"]).all())
    self.assertTrue((trainer._expert_act == self.expert_trajs["acts"]).all())
    self.assertEqual(
        trainer._env.observation_space.shape[0],
        self.expert_trajs["obses"].shape[1])
    self.assertEqual(
        trainer._env.observation_space.shape[0],
        self.expert_trajs["next_obses"].shape[1])
    self.assertEqual(
        trainer._env.action_space.shape[0],
        self.expert_trajs["acts"].shape[1])


if __name__ == '__main__':
  unittest.main()
