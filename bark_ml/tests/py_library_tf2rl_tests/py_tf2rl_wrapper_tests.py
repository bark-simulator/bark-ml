import unittest
import gym
import numpy as np
from gym.spaces.box import Box

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.tf2rl_wrapper import TF2RLWrapper
from bark_ml.library_wrappers.lib_tf2rl.normalization_utils import normalize


class PyTF2RLWrapperTests(unittest.TestCase):
  """For testing the TF2RL wrapper, which wraps the BARK runtime
  for compatibility with the tf2rl library.
  """

  def setUp(self):
    """Initial setup for the tests."""
    self.obs_low = np.array([-5., -8.])
    self.obs_high = np.array([3., 7.])
    self.act_low = np.array([-0.5, -1.2])
    self.act_high = np.array([0.3, 1.4])
    self.observation_space = Box(low=self.obs_low, high=self.obs_high)
    self.action_space = Box(low=self.act_low, high=self.act_high)

    self.env_orig = test_env(observation_space=self.observation_space,
                             action_space=self.action_space)

    self.wrapped_env = TF2RLWrapper(self.env_orig)
    self.wrapped_env_norm = TF2RLWrapper(
      self.env_orig, normalize_features=True)

  def test_init(self):
    """tests the init method of the TF2RLWrapper."""
    # without normalization:
    self.assertEqual(self.wrapped_env._env, self.env_orig)
    self.assertIsInstance(self.wrapped_env.action_space, Box)
    self.assertIsInstance(self.wrapped_env.observation_space, Box)
    self.assertTrue((self.wrapped_env.action_space.high ==
                     self.env_orig.action_space.high).all())
    self.assertTrue((self.wrapped_env.action_space.low ==
                     self.env_orig.action_space.low).all())
    self.assertTrue((self.wrapped_env.observation_space.high ==
                     self.env_orig.observation_space.high).all())
    self.assertTrue((self.wrapped_env.observation_space.low ==
                     self.env_orig.observation_space.low).all())

    # with normalization:
    self.assertEqual(self.wrapped_env_norm._env, self.env_orig)
    self.assertIsInstance(self.wrapped_env.action_space, Box)
    self.assertIsInstance(self.wrapped_env.observation_space, Box)
    self.assertTrue((self.wrapped_env_norm.action_space.high ==
                     np.ones_like(self.env_orig.action_space.high)).all())
    self.assertTrue(
      (self.wrapped_env_norm.action_space.low == -np.ones_like(self.env_orig.action_space.low)).all())
    self.assertTrue(
        (self.wrapped_env_norm.observation_space.high == np.ones_like(
            self.env_orig.observation_space.high)).all())
    self.assertTrue(
        (self.wrapped_env_norm.observation_space.low == -np.ones_like(
            self.env_orig.observation_space.low)).all())

  def test_reset(self):
    """tests the reset function of the TF2RLWrapper"""
    for _ in range(100):
      wrapped_obs = self.wrapped_env.reset()
      self.assertTrue((wrapped_obs == self.env_orig.obs).all())

    for _ in range(100):
      wrapped_obs = self.wrapped_env_norm.reset()
      self.assertTrue(((wrapped_obs <= 1.) & (wrapped_obs >= -1.)).all())
      self.assertTrue((wrapped_obs == normalize(
        self.env_orig.obs, self.env_orig.observation_space)).all())

  def test_step(self):
    """tests the step function of the TF2RLWrapper"""
    for _ in range(100):
      init_obs = np.random.uniform(low=self.obs_low, high=self.obs_high)
      action = np.random.uniform(low=self.act_low, high=self.act_high)

      self.env_orig.obs = init_obs.copy()
      etalon_next_obs, _, _, _ = self.env_orig.step(action)
      # wrapped without normalization
      self.wrapped_env._env.obs = init_obs.copy()
      wrapped_next_obs, _, _, _ = self.wrapped_env.step(action)
      self.assertTrue((etalon_next_obs == wrapped_next_obs).all())
      self.assertTrue((etalon_next_obs == self.wrapped_env._env.obs).all())
      # wrapped with normalization
      self.wrapped_env_norm._env.obs = init_obs.copy()
      action = normalize(action, self.env_orig.action_space)
      wrapped_norm_next_obs, _, _, _ = self.wrapped_env_norm.step(action)
      self.assertTrue(np.allclose(
          normalize(
              etalon_next_obs,
              self.env_orig.observation_space),
          wrapped_norm_next_obs))
      self.assertTrue(np.allclose(
        etalon_next_obs, self.wrapped_env_norm._env.obs))

  def normalize_action(self, action):
    """normalizes an action to be between -1 and 1"""
    action -= self.act_low
    action /= (self.act_high - self.act_low)
    action = action * 2. - 1.
    return action


class test_env():
  """dummy environment for testing."""

  def __init__(self, observation_space, action_space):
    """initializing a dummy environment with specified
    observation and action spaces.
    """
    self.observation_space = observation_space
    self.action_space = action_space
    self.reset()

  def reset(self):
    """reset function of the environment."""
    self.obs = np.random.uniform(
      low=self.observation_space.low, high=self.observation_space.high)
    return self.obs

  def step(self, action):
    """step function of the environment."""
    self.obs += action
    self.obs = np.clip(self.obs, self.observation_space.low,
                       self.observation_space.high)
    return self.obs, None, None, None


if __name__ == '__main__':
  unittest.main()
