# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import pickle
import unittest
from bark_ml.library_wrappers.lib_tf2rl.utils import load_expert_trajectory_file


class PyLibraryWrappersTF2RLUtilsTests(unittest.TestCase):
    """
    Tests for the tf2rl utils.
    """

    def test_load_pkl_file(self):
        """
        Test: Load the pkl file containing the expert trajectories.
        """
        expert_trajectories = {}
        for agent_id in range(5):
            expert_trajectories[agent_id] = {
                'obs': [],
                'act': [],
                'done': [],
                'time': [],
                'merge': []
            }

        file_path = "expert_trajectoriesectories.pkl"
        with open(file_path, "wb") as output_file:
            file_path = os.path.abspath(output_file.name)
            pickle.dump(expert_trajectories, output_file)

        expert_trajectories_loaded = load_expert_trajectoriesectory_file(file_path)

        self.assertEqual(expert_trajectories, expert_trajectories_loaded)
        os.remove(file_path)

if __name__ == '__main__':
    unittest.main()
