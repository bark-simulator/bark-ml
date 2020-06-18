# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import pickle
import unittest
from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories \
     import load_expert_trajectory_file


class LoadExpertTrajectoriesTest(unittest.TestCase):
    """
    Tests for the tf2rl utils.
    """

    def setUp(self):
        """
        setup
        """
        self.expert_trajectories = {}
        for agent_id in range(5):
            self.expert_trajectories[agent_id] = {
                'obs': [],
                'act': [],
                'done': [],
                'time': [],
                'merge': []
            }

        self.file_path = "expert_trajectories.pkl"
        with open(self.file_path, "wb") as output_file:
            self.file_path = os.path.abspath(output_file.name)
            pickle.dump(self.expert_trajectories, output_file)

    def tearDown(self):
        """
        tear down
        """
        os.remove(self.file_path)

    def test_load_pkl_file(self):
        """
        Test: Load the pkl file containing the expert trajectories.
        """
        expert_trajectories_loaded = load_expert_trajectory_file(
            self.file_path)
        self.assertEqual(self.expert_trajectories, expert_trajectories_loaded)

    def test_assert_file_exists(self):
        """
        Test: Assert that a non existent file raises an error.
        """
        with self.assertRaises(ValueError):
            load_expert_trajectory_file("non_existent_file.pkl")

    def test_assert_is_pkl_file(self):
        """
        Test: Assert that a non pkl file is not loaded.
        """
        import shutil
        new_file_name = self.file_path.replace('.pkl', '.txt')
        shutil.copyfile(self.file_path, new_file_name)

        with self.assertRaises(ValueError):
            load_expert_trajectory_file(new_file_name)
        os.remove(new_file_name)

    def test_file_contains_expert_trajectories(self):
        """
        Test: Assert that error is thrown if the file does not
                contain expert trajectories.
        """
        load_expert_trajectory_file(self.file_path)

        errornous_content = {
            "foo": "bar"
        }

        with open(self.file_path, "wb") as pickle_file:
            pickle.dump(errornous_content, pickle_file)

        with self.assertRaises(ValueError):
            load_expert_trajectory_file(self.file_path)


if __name__ == '__main__':
    unittest.main()
