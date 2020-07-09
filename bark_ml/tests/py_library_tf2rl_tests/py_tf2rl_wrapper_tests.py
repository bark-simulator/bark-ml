import unittest
import gym
import numpy as np
from gym.spaces.box import Box

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.tf2rl_wrapper import TF2RLWrapper


class PyTF2RLWrapperTests(unittest.TestCase):
    """For testing the TF2RL wrapper, which wraps the BARK runtime
    for compatibility with the tf2rl library.
    """

    def setUp(self):
        """Initial setup for the tests."""
        obs_low = np.array([-5., -8.])
        obs_high = np.array([3., 7.])
        act_low = np.array([-0.5, -1.2])
        act_high = np.array([0.3, 1.4])
        self.observation_space = Box(low=obs_low, high=obs_high)
        self.action_space = Box(low=act_low, high=act_high)

        self.env_orig = test_env(observation_space=self.observation_space,
            action_space=self.action_space)
        self.env_test = test_env(observation_space=self.observation_space,
            action_space=self.action_space)
        self.env_normalized = test_env(observation_space=self.observation_space,
            action_space=self.action_space,)
        
        self.wrapped_env = TF2RLWrapper(self.env_test)
        self.wrapped_env_norm = TF2RLWrapper(self.env_normalized, normalize_features=True)


    def test_init(self):
        """tests the init method of the TF2RLWrapper."""
        # without normalization:
        self.assertEqual(self.wrapped_env._env, self.env_test)
        self.assertIsInstance(self.wrapped_env.action_space, Box)
        self.assertIsInstance(self.wrapped_env.observation_space, Box)
        self.assertTrue((self.wrapped_env.action_space.high == self.env_test.action_space.high).all())
        self.assertTrue((self.wrapped_env.action_space.low == self.env_test.action_space.low).all())
        self.assertTrue((self.wrapped_env.observation_space.high == self.env_test.observation_space.high).all())
        self.assertTrue((self.wrapped_env.observation_space.low == self.env_test.observation_space.low).all())

        # with normalization:
        self.assertEqual(self.wrapped_env_norm._env, self.env_normalized)
        self.assertIsInstance(self.wrapped_env.action_space, Box)
        self.assertIsInstance(self.wrapped_env.observation_space, Box)
        self.assertTrue((self.wrapped_env_norm.action_space.high == np.ones_like(self.env_normalized.action_space.high)).all())
        self.assertTrue((self.wrapped_env_norm.action_space.low == -np.ones_like(self.env_normalized.action_space.low)).all())
        self.assertTrue((self.wrapped_env_norm.observation_space.high == np.ones_like(self.env_normalized.observation_space.high)).all())
        self.assertTrue((self.wrapped_env_norm.observation_space.low == -np.ones_like(self.env_normalized.observation_space.low)).all())


    def test_rescale_action(self):
        """tests the _rescale_action method of the TF2RLWrapper"""
        rescale_action = self.wrapped_env_norm._rescale_action
        self.assertTrue(np.allclose(self.env_orig.action_space.high, rescale_action(1.)))
        self.assertTrue(np.allclose(self.env_orig.action_space.low, rescale_action(-1.)))
        mean_actions = (self.env_orig.action_space.high + self.env_orig.action_space.low) / 2.
        self.assertTrue(np.allclose(mean_actions, rescale_action(0.)))


    def test_normalize_observation(self):
        """tests the _normalize observation method of the TF2RLWrapper"""
        pass



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
        self.obs = np.random.uniform(low=self.observation_space.low, high=self.observation_space.high)
        return self.obs


    def step(self, action):
        """step function of the environment."""
        self.obs += action
        self.obs = np.clip(self.obs, self.observation_space.low, self.observation_space.high)
        return self.obs, None, None, None


if __name__ == '__main__':
    unittest.main()