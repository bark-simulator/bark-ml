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
from bark_project.bark.runtime.commons.parameters import ParameterServer

class LoadExpertTrajectoriesTest(unittest.TestCase):
    """
    Tests for the tf2rl utils.
    """

    def setUp(self):
        """
        setup
        """
        self.params = ParameterServer(filename=os.path.join(os.path.dirname(__file__), "gail_data/params/gail_params_bark.json"))
        self.expert_trajectories_directory = os.path.join(os.path.dirname(__file__), 'data', 'expert_trajectories', 'sac')
        
        self.expert_trajectories, self.avg_trajectory_length, self.num_trajectories = load_expert_trajectories(self.expert_trajectories_directory)

        env = test_env(self.params)
        self.expert_trajectories_norm, self.avg_trajectory_length_norm, self.num_trajectories_norm = load_expert_trajectories(
            self.expert_trajectories_directory,
            normalize_features=True,
            env=env)

        assert self.expert_trajectories
        assert self.expert_trajectories_norm

        self.assertEqual(self.avg_trajectory_length, 14.4) 
        self.assertEqual(self.avg_trajectory_length, self.avg_trajectory_length_norm) 

        self.assertEqual(self.num_trajectories, 5)
        self.assertEqual(self.num_trajectories, self.num_trajectories_norm)

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
        self.assertTrue(((self.expert_trajectories_norm['obses'] >= -1) &\
            (self.expert_trajectories_norm['obses'] <= 1)).all())
        self.assertTrue(((self.expert_trajectories_norm['next_obses'] >= -1) &\
            (self.expert_trajectories_norm['next_obses'] <= 1)).all())
        self.assertTrue(((self.expert_trajectories_norm['acts'] >= -1) &\
            (self.expert_trajectories_norm['acts'] <= 1)).all())


class test_env():
    """dummy environment for testing."""

    def __init__(self, params):
        """init method"""

        num_max_vehicles = params["ML"]["StateObserver"]["MaxNumAgents"]

        if params["ML"]["StateObserver"]["NormalizationEnabled"]:
            self.observation_space = Box(low=np.zeros(((num_max_vehicles + 1) * 4,)),
                high=np.ones(((num_max_vehicles + 1) * 4,)))
        else:
            x_range, y_range = [-10000, 10000], [-10000, 10000]
            theta_range = params["ML"]["StateObserver"]["ThetaRange"]
            velocity_range = params["ML"]["StateObserver"]["VelocityRange"]
            self.observation_space = Box(low=np.array([x_range[0], y_range[0],\
                theta_range[0], velocity_range[0]] * (num_max_vehicles + 1)))

        act_low = params["ML"]["BehaviorContinuousML"]["ActionsLowerBound"]
        act_high = params["ML"]["BehaviorContinuousML"]["ActionsUpperBound"]
        self.action_space = Box(low=np.array(act_low), high=np.array(act_high))


if __name__ == '__main__':
    unittest.main()
