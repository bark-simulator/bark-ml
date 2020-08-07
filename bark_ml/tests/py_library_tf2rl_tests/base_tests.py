import os
import math
import pickle
import unittest
import numpy as np
import shutil
from bark_ml.library_wrappers.lib_tf2rl.generate_expert_trajectories \
    import *

# Please add a new test case class for every function you test.
# Modularise the tests as much as possible, but on a reasonable scale.

tracks_folder = os.path.join(
    os.path.dirname(__file__), 'data/interaction_data_set_mock/DR_DEU_Merging_MT/tracks')
map_file = os.path.join(
    os.path.dirname(__file__), 'data/interaction_data_set_mock/DR_DEU_Merging_MT/map/DR_DEU_Merging_MT_v01_shifted.xodr')
known_key = ('DR_DEU_Merging_MT_v01_shifted',
             'vehicle_tracks_013')


class CalculateActionTests(unittest.TestCase):
    """
    Tests for the calculate action function.
    """

    def test_calculate_action_simple_zero(self):
        """
        Test: Calculate the action based on two consecutive observations with an initially stopped car.
        """
        observations = [[0] * 4, [1000] * 4]
        time_step = 1000

        expected_action = [1.0, 0.0]

        self.assertEquals(
            expected_action,
            calculate_action(observations, time_step, 2.7))

    def test_calculate_action_simple_zero(self):
        """
        Test: Calculate the action based on two consecutive observations with an initially stopped car with noisy data.
        """
        observations = [[1e-3] * 4, [1000 + 1e-3] * 4]
        time_step = 1000

        expected_action = [1.0, 0.5 * math.pi]
        calculated_action = calculate_action(observations, time_step, 2.7)
        self.list_almost_equal(expected_action, calculated_action)

    def test_calculate_action_simple_straight(self):
        """
        Test: Calculate the action based on two consecutive observations in straight movement.
        """
        observations = [[0, 0, 0, 500],[0, 0, 0, 1000]]
        time_step = 500

        expected_action = [1.0, 0.0]

        self.assertEquals(
            expected_action,
            calculate_action(observations, time_step, 2.7))

    def test_calculate_action_simple_curve(self):
        """
        Test: Calculate the action based on two consecutive observations with a curve.
        """
        observations = [[50] * 4, [100] * 4]
        time_step = 10

        expected_action = [5.0, math.atan(0.27)]
        action = calculate_action(observations, time_step, 2.7)

        self.list_almost_equal(expected_action, action)

    def test_calculate_action_2nd_degree(self):
        """
        Test: Calculate the action based on two consecutive observations.
        """
        observations = [[0] * 4, [1000] * 4, [0] * 4]
        time_step = 1000

        expected_action = [0.0, 0.0]
        action = calculate_action(observations[0:3], time_step, 2.7)
        
        steering_error = abs(action[0] - expected_action[0])
        acceleration_error = abs(action[1] - expected_action[1])

        self.assertLessEqual(steering_error, 1e-8)
        self.assertLessEqual(acceleration_error, 1e-8)

    def test_calculate_action_collinear(self):
        """
        Test: Calculate the action based on two consecutive observations.
        """
        observations = [[0] * 4, [1000] * 4, [2000] * 4]
        time_step = 1000

        expected_action = [1.0, 0.002699993439028698]
        action = calculate_action(observations[0:3], time_step, 2.7)

        steering_error = abs(action[0] - expected_action[0])
        acceleration_error = abs(action[1] - expected_action[1])

        self.assertLessEqual(steering_error, 1e-8)
        self.assertLessEqual(acceleration_error, 1e-8)

    def test_calculate_action_zero_time_step(self):
        """
        Test: Calculate the action based on two consecutive observations.
        """
        observations = [[0] * 4, [1000] * 4]
        time_step = 0

        expected_action = [0.0, 0.0]

        self.assertEquals(
            expected_action,
            calculate_action(observations[0:2], time_step))

    def list_almost_equal(self, first, second):
        """Checks if two lists are almost equal
        """
        self.assertEqual(len(first), len(second))
        for i, a in enumerate(first):
            self.assertAlmostEqual(a, second[i], places=3)

    def compare_calculate_to_expected(self, observations, expected_actions, time_step):
        """For the given observations calculate all first and second degree actions and compare them to the expected result. Assumes a wheel_base of 1.

        Args:
            observations (list): The observations
            expected_actions (list): The action
            time_step (float): The timestep
        """
        for i, observation in enumerate(observations[:-1]):
            current_observations = [observation, observations[i+1]]

            calculated_action = calculate_action(current_observations, time_step, wheel_base=1)
            self.list_almost_equal(expected_actions[i], calculated_action)

            rotation_change = observations[i + 1][2] - observation[2]
            if observation[3] != 0:
                self.assertAlmostEqual(rotation_change, math.tan(calculated_action[1]) * observation[3])

        # for i, observation in enumerate(observations[1:-1]):
        #     current_observations = [observations[i], observation, observations[i+2]]

        #     calculated_action = calculate_action(current_observations, time_step, wheel_base=1)
        #     self.list_almost_equal(expected_actions[i + 1], calculated_action)

        #     rotation_change = observation[2] - observations[i][2] 
        #     self.assertAlmostEqual(rotation_change, math.tan(calculated_action[0]))

    def test_calculate_action_circle_zero_acceleration(self):
        """
        Test: Calculate the action based on observations around a circle with constant velocity.
        """
        number_observations = 10
        rotation_change = 0.5 * math.pi
        observations = [[0, 0, i * rotation_change, 1] for i in range(number_observations)]
        time_step = 1
        expected_actions = [[0.0, math.atan(rotation_change)]] * number_observations
        self.compare_calculate_to_expected(observations, expected_actions, time_step)

    def test_calculate_action_non_linear_rotations(self):
        """
        Test: Calculate the action based on observations around a circle with constant velocity.
        """
        observations = [[0, 0, 0, 1], [0, 0, 0.5 * math.pi, 1], [0, 0, 0, 1]]
        time_step = 1
        expected_actions = [[0.0, math.atan(0.5 * math.pi)], [0.0, -math.atan(0.5 * math.pi)]]
        self.compare_calculate_to_expected(observations, expected_actions, time_step)

    def test_calculate_action_circle_zero_start_velocity(self):
        """
        Test: Calculate the action based on observations around a circle with constant acceleration and zero start velocity.
        """
        number_observations = 10
        rotation_change = math.pi
        observations = [[0, 0, i * rotation_change, i] for i in range(number_observations)]
        time_step = 1
        expected_actions = [[1.0, math.atan(rotation_change)]]
        expected_actions.extend([[1.0, math.atan(rotation_change / i)] for i in range(1, number_observations - 1)])
        self.compare_calculate_to_expected(observations, expected_actions, time_step)

    def test_calculate_action_half_circle_zero_start_velocity_zero_acceleration(self):
        """
        Test: Calculate the action based on two consecutive observations.
        """
        observations = [[0, 0, 0, 0], [1, 1, math.pi, 0]]
        time_step = 1

        expected_action = [0.0, 0.0]
        actual_action = calculate_action(observations, time_step, wheel_base=1)
        self.list_almost_equal(expected_action, actual_action)

    def test_calculate_action_full_circle(self):
        """
        Test: Calculate the action based on two consecutive observations.
        """
        observations = [[0, 0, 0, 1], [1, 1, math.pi, 1], [1, 1, 2 * math.pi, 1]]
        time_step = 1

        expected_action = [0.0, math.atan(math.pi)]
        calculated_action = calculate_action(observations[:2], time_step, wheel_base=1)
        self.assertAlmostEqual(expected_action, calculated_action)

        calculated_action = calculate_action(observations[1:], time_step, wheel_base=1)
        self.assertAlmostEqual(expected_action, calculated_action)

    def test_calculate_action_sin(self):
        """
        Test: Calculate the action based on consecutive observations with sinusoid angle changes.
        """
        velocity = 30.0
        time_step = 1
        wheel_base = 2.7
        num_samples = 500

        observations = []
        for i in range(num_samples):
            angle = i * math.pi / num_samples
            entry = [0, 0, math.sin(angle), velocity]
            observations.append(entry)

        for i in range(1, num_samples-1):
            action = calculate_action(observations[i-1:i+2], time_step, wheel_base)

            d_theta = math.cos(i * math.pi / num_samples) * math.pi / num_samples
            steering_angle = math.atan2(wheel_base * d_theta, velocity)
            expected_action = [0.0, steering_angle]

            steering_error = abs(action[1] - expected_action[1])
            acceleration_error = abs(action[0] - expected_action[0])

            # Higher threshold for the steering angle due to the sine derivative approximation
            self.assertLessEqual(steering_error, 1e-5)
            self.assertLessEqual(acceleration_error, 1e-8)


class GetMapAndTrackFilesTests(unittest.TestCase):
    """
    Tests: get_track_files and get_map_files
    """

    def test_get_track_files(self):
        """
        Test: get_track_files
        """
        track_files = get_track_files(tracks_folder)

        self.assertIs(len(track_files), 1)
        self.assertIs(track_files[0].endswith(
            'bark_ml/tests/py_library_tf2rl_tests/data/interaction_data_set_mock/DR_DEU_Merging_MT/tracks/vehicle_tracks_013.csv'), True)

    def test_interaction_data_set_path_invalid(self):
        """
        Test: The given path is not an interaction dataset path
        """
        with self.assertRaises(NotADirectoryError):
            track_files = get_track_files(__file__)


class CreateParameterServersForScenariosTests(unittest.TestCase):
    """
    Tests: create_parameter_servers_for_scenarios
    """

    def test_create_parameter_servers_for_scenarios(self):
        """
        Test: Valid parameter server
        """

        param_servers = create_parameter_servers_for_scenarios(
            map_file, tracks_folder)
        self.assertIn(known_key, param_servers)

        param_server = param_servers[known_key]
        assert param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["MapFilename"].endswith(
            'bark_ml/tests/py_library_tf2rl_tests/data/interaction_data_set_mock/DR_DEU_Merging_MT/map/DR_DEU_Merging_MT_v01_shifted.xodr')
        assert param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["TrackFilename"].endswith(
            'bark_ml/tests/py_library_tf2rl_tests/data/interaction_data_set_mock/DR_DEU_Merging_MT/tracks/vehicle_tracks_013.csv')

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
            map_file, tracks_folder)
        self.assertIn(known_key, param_servers)

        scenario, start_ts, end_ts = create_scenario(param_servers[known_key])
        self.assertEqual(start_ts, 100)
        self.assertEqual(end_ts, 327300)

        assert scenario.map_file_name.endswith(
            'bark_ml/tests/py_library_tf2rl_tests/data/interaction_data_set_mock/DR_DEU_Merging_MT/map/DR_DEU_Merging_MT_v01_shifted.xodr')
        self.assertEqual(len(scenario._agent_list), 86)

class GetViewerTests(unittest.TestCase):
    """Tests: get_viewer
    """

    def test_correct_viewer_given(self):
        """Test: Is the returned viewer of the correct type.
        """
        from bark.runtime.viewer.pygame_viewer import PygameViewer
        from bark.runtime.viewer.matplotlib_viewer import MPViewer

        param_servers = create_parameter_servers_for_scenarios(
            map_file, tracks_folder)
        self.assertIn(known_key, param_servers)

        viewer = get_viewer(param_servers[known_key], "pygame")
        self.assertEqual(type(viewer), PygameViewer)
        viewer = get_viewer(param_servers[known_key], "matplotlib")
        self.assertEqual(type(viewer), MPViewer)

if __name__ == '__main__':
    unittest.main()
