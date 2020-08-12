import math
import unittest
from bark_ml.library_wrappers.lib_tf2rl.compare_trajectories import *


class CompareTrajectoriesTest(unittest.TestCase):
  """Tests for the compare trajectories functions.
  """

  def test_key_not_in_trajectory(self):
    """Test: Expection risen when a key is not in the trajectory.
    """
    first = {
        'obses': [[0] * 16] * 10,
        'acts': [[2] * 2] * 10,
    }
    second = {
        'obses': [[1] * 16] * 10,
        'next_obses': [[2] * 16] * 10,
    }
    with self.assertRaises(ValueError):
      compare_trajectories(first, second)

  def test_compare_trajectories_same_dimensions(self):
    """Test: Calculates the norm between two trajectories of the same length.
    """
    first = {
        'obses': [[1] * 16] * 10,
        'next_obses': [[2] * 16] * 10,
        'acts': [[4] * 2] * 10,
    }
    second = {
        'obses': [[0] * 16] * 10,
        'next_obses': [[1] * 16] * 10,
        'acts': [[2] * 2] * 10,
    }
    distances = compare_trajectories(first, second)
    self.assertAlmostEqual(distances['obses'], 4.0)
    self.assertAlmostEqual(distances['next_obses'], 4.0)
    self.assertAlmostEqual(distances['acts'], math.sqrt(8))

  def test_compare_trajectories_different_dimensions(self):
    """Test: Calculates the norm between two trajectories of different lengths.
    """
    first = {
        'obses': [[1] * 16] * 10,
        'next_obses': [[2] * 16] * 10,
        'acts': [[4] * 2] * 10,
    }
    second = {
        'obses': [[0] * 16] * 8,
        'next_obses': [[1] * 16] * 8,
        'acts': [[2] * 2] * 8,
    }
    distances = [
        compare_trajectories(first, second),
        compare_trajectories(second, first)]
    self.assertAlmostEqual(distances[0], distances[1])
    self.assertAlmostEqual(distances[0]['obses'], 8.0)
    self.assertAlmostEqual(distances[0]['next_obses'], 12.0)
    self.assertAlmostEqual(distances[0]['acts'], math.sqrt(32) + math.sqrt(8))

  def test_compare_same_trajectories(self):
    """Test: Calculates the norm between two equal trajectories.
    """
    first = {
        'obses': [[1] * 16] * 10,
        'next_obses': [[2] * 16] * 10,
        'acts': [[4] * 2] * 10,
    }
    second = {
        'obses': [[1] * 16] * 10,
        'next_obses': [[2] * 16] * 10,
        'acts': [[4] * 2] * 10,
    }
    distances = compare_trajectories(first, second)
    self.assertAlmostEqual(distances['obses'], 0.0)
    self.assertAlmostEqual(distances['next_obses'], 0.0)
    self.assertAlmostEqual(distances['acts'], 0)

  def test_compare_same_trajectories(self):
    """Test: Calculates the norm between two equal trajectories except one is longer.
    """
    first = {
        'obses': [[1] * 16] * 10,
        'next_obses': [[2] * 16] * 10,
        'acts': [[4] * 2] * 10,
    }
    second = {
        'obses': [[1] * 16] * 8,
        'next_obses': [[2] * 16] * 8,
        'acts': [[4] * 2] * 8,
    }
    distances = compare_trajectories(first, second)
    self.assertAlmostEqual(distances['obses'], 4.0)
    self.assertAlmostEqual(distances['next_obses'], 8.0)
    self.assertAlmostEqual(distances['acts'], math.sqrt(32))


class CalculateMeanActionTests(unittest.TestCase):
  """Tests for the calculate_mean_action function
  """

  def test_calculate_mean_action(self):
    """Test: Calculates the mean for a valid trajectory.
    """
    trajectories = {
        'obses': [[1] * 16] * 16,
        'next_obses': [[2] * 16] * 16,
        'acts': [[1] * 2] * 16,
    }
    mean_action = calculate_mean_action(trajectories)
    self.assertAlmostEqual(mean_action[0], 1.0)
    self.assertAlmostEqual(mean_action[1], 1.0)

  def test_calculate_mean_action_invalid(self):
    """Test: Exception risen on invalid trajectories.
    """
    trajectories = {
        'obses': [[1] * 16] * 16,
        'next_obses': [[2] * 16] * 16,
    }
    with self.assertRaises(ValueError):
      calculate_mean_action(trajectories)


if __name__ == '__main__':
  unittest.main()
