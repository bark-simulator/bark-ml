# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import joblib
import unittest
from gym.spaces.box import Box
from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories import *
from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import *

class LoadExpertTrajectoriesTest(unittest.TestCase):
    """
    Tests for the tf2rl utils.
    """

    def setUp(self):
        """
        setup
        """
        self.params = ParameterServer(filename=os.path.join(os.path.dirname(__file__), "gail_data/params/gail_params_bark.json"))
        self.expert_trajectories_directory = os.path.join(os.path.dirname(__file__), 'data', 'expert_trajectories')
        
        self.expert_trajectories = load_expert_trajectories(
            self.expert_trajectories_directory)

        env = test_env(self.params)
        self.expert_trajectories_norm = load_expert_trajectories(
            self.expert_trajectories_directory,
            normalize_features=True,
            env=env)
        assert self.expert_trajectories

    def test_assert_file_exists(self):
        """
        Test: Assert that a non existent file raises an error.
        """
        with self.assertRaises(AssertionError):
            load_expert_trajectories('/tmp/')

    def test_file_contains_expert_trajectories(self):
        """
        Test: Assert that error is thrown if the file does not
                contain expert trajectories.
        """
        file_path = os.path.join(os.path.dirname(__file__), 'data', 'invalid_file.jblb')
        with self.assertRaises(KeyError):
            load_expert_trajectories(os.path.dirname(file_path))

    def test_normalized_trajectories(self):
        """Tests whether all the expert trajectories are in the range of -1 and 1"""


class test_env():
    """dummy environment for testing."""
    def __init__(self, params):
        """init method"""
        self.observation_space = Box(low=np.zeros((16,)), high=np.ones((16,)))
        self.action_space = 

if __name__ == '__main__':
    unittest.main()
