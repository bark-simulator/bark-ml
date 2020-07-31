# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import joblib
import unittest
import numpy as np
from gym.spaces.box import Box
from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories import *
from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import *
from bark_project.bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver

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

    def test_sample_subset(self):
        """Test: Correct subset size sampled"""
        np.random.seed(0)
        subset, avg_length, num_trajectories = load_expert_trajectories(self.expert_trajectories_directory, subset_size=3)

        self.assertEqual(avg_length, 15)
        self.assertEqual(num_trajectories, 3)

        self.assertEqual(subset['obses'].shape, (45, 16))
        self.assertEqual(subset['next_obses'].shape, (45, 16))
        self.assertEqual(subset['acts'].shape, (45, 2))

class test_env():
    """dummy environment for testing."""

    def __init__(self, params):
        """init method"""

        num_max_vehicles = params["ML"]["StateObserver"]["MaxNumAgents"]
        x_range, y_range = [-10000, 10000], [-10000, 10000]
        theta_range = params["ML"]["StateObserver"]["ThetaRange"]
        velocity_range = params["ML"]["StateObserver"]["VelocityRange"]
        
        self._observer = NearestAgentsObserver(params)
        # #  = {
        # '_world_x_range': x_range,
        # '_world_y_range': y_range,
        # '_ThetaRange': theta_range,
        # '_VelocityRange': velocity_range
        # }
        self.observation_space = Box(
            low=np.array([x_range[0], y_range[0],\
            theta_range[0], velocity_range[0]] * (num_max_vehicles + 1)),\
            high=np.array([x_range[1], y_range[1],\
            theta_range[1], velocity_range[1]] * (num_max_vehicles + 1))
            )

        act_low = params["ML"]["BehaviorContinuousML"]["ActionsLowerBound"]
        act_high = params["ML"]["BehaviorContinuousML"]["ActionsUpperBound"]
        self.action_space = Box(low=np.array(act_low), high=np.array(act_high))
    
    def reset(self):
        pass

if __name__ == '__main__':
    unittest.main()
