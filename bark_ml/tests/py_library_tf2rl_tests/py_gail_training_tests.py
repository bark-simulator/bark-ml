import unittest
import os
import numpy as np
from gym.spaces.box import Box
from pathlib import Path

# BARK imports
from bark_project.bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner
from bark_ml.library_wrappers.lib_tf2rl.agents.gail_agent import BehaviorGAILAgent

# TF2RL imports:
from tf2rl.experiments.utils import restore_latest_n_traj


class PyTrainingBARKTests(unittest.TestCase):
  """This test aims to test whether the agent learns or not.
  The aim is to learn that the expert action is always the multiplicative
  of the observation by 2. The expert trajectories are generated accordingly.
  The environment also gives back a reward 1 if the action is nearly the double
  of the observation and -1 otherways. (The reward of the environment during training does not
  matter actually, but it is easier to see this way, wheteher the agent learns or not.)
  The agent is trained for a while, and the test passes if the accuracy of the trained agent on 
  some test observations is significantly better than it was before training.
  """

  def setUp(self):
    """
    Setup
    """
    self.params = ParameterServer(
        filename=os.path.join(
            os.path.dirname(__file__),
            "gail_data/params/gail_params_test_env.json"))

    # creating the dirs for logging if they are not present already:
    for key in ['logdir', 'model_dir', 'expert_path_dir']:
      self.params["ML"]["GAILRunner"]["tf2rl"][key] = os.path.join(
        Path.home(), self.params["ML"]["GAILRunner"]["tf2rl"][key])
      if not os.path.exists(self.params["ML"]["GAILRunner"]["tf2rl"][key]):
        os.makedirs(self.params["ML"]["GAILRunner"]["tf2rl"][key])

    # create environment
    self.env = test_env()

    # Dummy expert trajectories:
    random_obses = np.random.uniform(
      low=-1, high=1, size=(1001, 2)).astype(np.float32)
    self.expert_trajs = {
        'obses': random_obses[:-1, :],
        'next_obses': random_obses[1:, :],
        'acts': random_obses[:-1, :] * 2
    }

    # create agent and runner:
    self.agent = BehaviorGAILAgent(
        environment=self.env,
        params=self.params
    )
    self.runner = GAILRunner(
        environment=self.env,
        agent=self.agent,
        params=self.params,
        expert_trajs=self.expert_trajs)

  def test_training(self):
    """
    tests the Train() method of the GAILRunner class with the test environment.
    """
    test_obses = np.random.uniform(
      low=-1, high=1, size=(100, 2)).astype(np.float32)
    # evaluation before training:
    actions = self.runner._agent.Act(test_obses)
    accuracy_before = np.sum(
        np.isclose(actions / test_obses, 2., rtol=5e-2, atol=1e-1)) / (
        test_obses.shape[0] * test_obses.shape[1])

    self.runner.Train()

    # evaluation after training:
    actions = self.runner._agent.Act(test_obses)
    accuracy_after = np.sum(
        np.isclose(actions / test_obses, 2., rtol=5e-2, atol=1e-1)) / (
        test_obses.shape[0] * test_obses.shape[1])

    # assert that the performance after training is much better:
    if accuracy_before == 0:
      self.assertGreater(accuracy_after, 0.4)
    else:
      self.assertGreater(accuracy_after, accuracy_before * 10)


class test_env():
  """simple environment to check whether the gail agent learns or not.
  The environment gives back the reward 1 if:
      action ~= obs * 2
  The reward is -1 in every other case.
  Every episode consists of 1 step.
  """

  def __init__(self):
    """initializes the test environment."""
    self.action_space = Box(low=np.array([-2, -2]), high=np.array([2, 2]))
    self.observation_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]))

    self.reset()

  def step(self, action):
    """step function of the environment"""
    if np.allclose(2., action / self.obs, rtol=5e-2, atol=1e-1):
      reward = 1
      next_obs = self.obs
      done = True
    else:
      reward = -1
      next_obs = self.obs
      done = True

    return next_obs, reward, done, None

  def reset(self):
    """resets the agent"""
    self.obs = np.random.uniform(low=-1, high=1, size=(2,)).astype(np.float32)
    return self.obs


if __name__ == '__main__':
  unittest.main()
