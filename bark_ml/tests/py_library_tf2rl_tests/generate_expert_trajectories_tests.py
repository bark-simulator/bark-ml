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

if __name__ == '__main__':
    unittest.main()
