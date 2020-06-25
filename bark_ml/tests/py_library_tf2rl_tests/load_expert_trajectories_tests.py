# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import pickle
import unittest
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
        self.expert_trajectories_directory = os.path.join(os.path.dirname(__file__), "data")
        self.pickle_files = list_files_in_dir(self.expert_trajectories_directory, '*.pkl')

        self.expert_trajectories_per_file = load_expert_trajectory_files(
            self.expert_trajectories_directory)
        assert self.expert_trajectories_per_file    
        
        self.expert_trajectories, self.dt = load_expert_trajectories(
            self.expert_trajectories_directory)
        assert self.expert_trajectories
        assert (self.dt - 100) < 10e5 

    def test_assert_file_exists(self):
        """
        Test: Assert that a non existent file raises an error.
        """
        with self.assertRaises(ValueError):
            load_expert_trajectory_file("non_existent_file.pkl")

    def test_file_contains_expert_trajectories(self):
        """
        Test: Assert that error is thrown if the file does not
                contain expert trajectories.
        """
        errornous_content = {
            "foo": "bar"
        }

        file_path = 'test.pkl'
        with open(file_path, "wb") as pickle_file:
            pickle.dump(errornous_content, pickle_file)
            file_path = pickle_file.name

        no_expert_trajectories_contained = load_expert_trajectory_file(file_path)
        assert not no_expert_trajectories_contained
        os.remove(file_path)

if __name__ == '__main__':
    unittest.main()
