import os
import pickle
import unittest
from bark_ml.library_wrappers.lib_tf2rl.generate_expert_trajectories \
     import *

# Please add a new test case class for every function you test.
# Modularise the tests as much as possible, but on a reasonable scale.

class CalculateActionTests(unittest.TestCase):
    """
    Tests for the calculate action function.
    """

    def test_calculate_action(self):
        """
        Test: Calculate the action based on two consecutive observations.
        """
        # TODO Add more working examples as just this single
        observations = [[0] * 12, [1000] * 12]
        timestamps = [1000, 2000]

        expected_action = [1.2160906747839564, 1.0]

        self.assertEqual(
            expected_action,
            calculate_action(observations[1], observations[0], timestamps[1], timestamps[0], 2.7))

    def test_calculate_action_timestamps_are_equal(self):
        """
        Test: Calculate the action based on two consecutive observations.
        """
        observations = [[0] * 12, [1000] * 12]
        timestamps = [1000, 1000]

        expected_action = [0.0, 0.0]

        self.assertEqual(
            expected_action,
            calculate_action(observations[1], observations[0], timestamps[1], timestamps[0], 2.7))


class GetMapAndTrackFilesTests(unittest.TestCase):
    """
    Tests: get_track_files and get_map_files
    """

    def test_get_track_files(self):
        """
        Test: get_track_files
        """
        interaction_data_set_mock_path = os.path.join(os.path.dirname(__file__), 'data/interaction_data_set_mock')
        track_files = get_track_files(interaction_data_set_mock_path)

        self.assertIs(len(track_files), 1)
        self.assertIs(track_files[0].endswith('bark_ml/tests/py_library_tf2rl_tests/data/interaction_data_set_mock/DR_DEU_Merging_MT/tracks/vehicle_tracks_013.csv'), True)

    def test_load_map_files(self):
        """
        Test: get_map_files
        """
        interaction_data_set_mock_path = os.path.join(os.path.dirname(__file__), 'data/interaction_data_set_mock')
        map_files = get_map_files(interaction_data_set_mock_path)

        self.assertIs(len(map_files), 1)
        self.assertIs(map_files[0].endswith('bark_ml/tests/py_library_tf2rl_tests/data/interaction_data_set_mock/DR_DEU_Merging_MT/map/DR_DEU_Merging_MT_v01_shifted.xodr'), True)
    
    def test_interaction_data_set_path_invalid(self):
        """
        Test: The given path is invalid
        """
        map_files = get_map_files(os.path.dirname(__file__))
        print(map_files)


if __name__ == '__main__':
    unittest.main()
