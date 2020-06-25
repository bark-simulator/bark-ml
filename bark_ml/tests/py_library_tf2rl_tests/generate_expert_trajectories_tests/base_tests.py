import os
import pickle
import unittest
import numpy as np
import shutil
from bark_ml.library_wrappers.lib_tf2rl.generate_expert_trajectories \
    import *

# Please add a new test case class for every function you test.
# Modularise the tests as much as possible, but on a reasonable scale.

interaction_data_set_mock_path = os.path.join(
    os.path.dirname(__file__), 'data/interaction_data_set_mock')
known_key = ('DR_DEU_Merging_MT/DR_DEU_Merging_MT_v01_shifted',
             'DR_DEU_Merging_MT/vehicle_tracks_013')


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
        track_files = get_track_files(interaction_data_set_mock_path)

        self.assertIs(len(track_files), 1)
        self.assertIs(track_files[0].endswith(
            'bark_ml/tests/py_library_tf2rl_tests/generate_expert_trajectories_tests/data/interaction_data_set_mock/DR_DEU_Merging_MT/tracks/vehicle_tracks_013.csv'), True)

    def test_load_map_files(self):
        """
        Test: get_map_files
        """
        map_files = get_map_files(interaction_data_set_mock_path)

        self.assertIs(len(map_files), 1)
        self.assertIs(map_files[0].endswith(
            'bark_ml/tests/py_library_tf2rl_tests/generate_expert_trajectories_tests/data/interaction_data_set_mock/DR_DEU_Merging_MT/map/DR_DEU_Merging_MT_v01_shifted.xodr'), True)

    def test_interaction_data_set_path_invalid(self):
        """
        Test: The given path is not an interaction dataset path
        """
        with self.assertRaises(ValueError):
            map_files = get_map_files(os.path.dirname(__file__))


class CreateParameterServersForScenariosTests(unittest.TestCase):
    """
    Tests: create_parameter_servers_for_scenarios
    """

    def test_create_parameter_servers_for_scenarios(self):
        """
        Test: Valid parameter server
        """

        param_servers = create_parameter_servers_for_scenarios(
            interaction_data_set_mock_path)
        self.assertIn(known_key, param_servers)

        param_server = param_servers[known_key]
        assert param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["MapFilename"].endswith(
            'bark_ml/tests/py_library_tf2rl_tests/generate_expert_trajectories_tests/data/interaction_data_set_mock/DR_DEU_Merging_MT/map/DR_DEU_Merging_MT_v01_shifted.xodr')
        assert param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["TrackFilename"].endswith(
            'bark_ml/tests/py_library_tf2rl_tests/generate_expert_trajectories_tests/data/interaction_data_set_mock/DR_DEU_Merging_MT/tracks/vehicle_tracks_013.csv')

        track_ids = [i for i in range(1, 87) if i != 18]
        self.assertEqual(param_server["Scenario"]["Generation"]
                         ["InteractionDatasetScenarioGeneration"]["TrackIds"], track_ids)

        self.assertEqual(param_server["Scenario"]["Generation"]
                         ["InteractionDatasetScenarioGeneration"]["StartTs"], 100)
        self.assertEqual(param_server["Scenario"]["Generation"]
                         ["InteractionDatasetScenarioGeneration"]["EndTs"], 327300)

        param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["EgoTrackId"] = 1


class CreateScenarioTests(unittest.TestCase):
    """
    Tests: create_scenario
    """

    def test_create_scenario(self):
        """
        Test: Valid scenario
        """
        param_servers = create_parameter_servers_for_scenarios(
            interaction_data_set_mock_path)
        self.assertIn(known_key, param_servers)

        scenario, start_ts, end_ts = create_scenario(param_servers[known_key])
        self.assertEqual(start_ts, 100)
        self.assertEqual(end_ts, 327300)

        assert scenario.map_file_name.endswith(
            'bark_ml/tests/py_library_tf2rl_tests/generate_expert_trajectories_tests/data/interaction_data_set_mock/DR_DEU_Merging_MT/map/DR_DEU_Merging_MT_v01_shifted.xodr')
        self.assertEqual(len(scenario._agent_list), 86)


if __name__ == '__main__':
    unittest.main()