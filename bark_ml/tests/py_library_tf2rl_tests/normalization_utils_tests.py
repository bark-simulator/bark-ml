import unittest
import numpy as np
from gym.spaces.box import Box

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.normalization_utils import rescale, normalize


class NormalizationUtilsTests(unittest.TestCase):
  """Testing the normalizing and rescaling functions."""

  def setUp(self):
    """setting up some variables for testing."""
    self.low = np.array([-0.5, -1.2])
    self.high = np.array([0.3, 1.4])
    self.mean = (self.high + self.low) / 2.
    self.space = Box(low=self.low, high=self.high)

  def test_rescale(self):
    """tests the rescale function"""
    self.assertTrue(np.allclose(self.space.high, rescale(1., self.space)))
    self.assertTrue(np.allclose(self.space.low, rescale(-1., self.space)))
    self.assertTrue(np.allclose(self.mean, rescale(0., self.space)))

  def test_normalize(self):
    """tests the normalize function"""
    self.assertTrue(np.allclose(normalize(self.space.high, self.space), 1.))
    self.assertTrue(np.allclose(normalize(self.space.low, self.space), -1.))
    self.assertTrue(np.allclose(
        normalize(self.mean, self.space),
        0., atol=1e-06))


if __name__ == '__main__':
  unittest.main()
