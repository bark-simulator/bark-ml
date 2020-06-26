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

    def test_calculate_action_simple(self):
        """
        Test: Calculate the action based on two consecutive observations.
        """
        # TODO Add more working examples as just this single
        observations = [[0] * 12, [1000] * 12]
        time_step = 1000

        expected_action = [1.2160906747839564, 1.0]

        self.assertEquals(
            expected_action,
            calculate_action(observations[0:2], time_step, 2.7))

    def test_calculate_action_2nd_degree(self):
        """
        Test: Calculate the action based on two consecutive observations.
        """
        # TODO Add more working examples as just this single
        observations = [[0] * 12, [1000] * 12, [0] * 12]
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
        # TODO Add more working examples as just this single
        observations = [[0] * 12, [1000] * 12, [2000] * 12]
        time_step = 1000

        expected_action = [1.2160906747839564, 1.0]

        self.assertEquals(
            expected_action,
            calculate_action(observations[0:3], time_step, 2.7))

    def test_calculate_action_timestamps_are_equal(self):
        """
        Test: Calculate the action based on two consecutive observations.
        """
        observations = [[0] * 12, [1000] * 12]
        time_step = 0

        expected_action = [0.0, 0.0]

        self.assertEquals(
            expected_action,
            calculate_action(observations[0:2], time_step))

if __name__ == '__main__':
    unittest.main()
