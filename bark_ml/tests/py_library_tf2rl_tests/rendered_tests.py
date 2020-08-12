import os
import pickle
import unittest
import numpy as np
import shutil
from bark_ml.library_wrappers.lib_tf2rl.generate_expert_trajectories \
    import *
from simulation_based_tests import *


class RenderedSimulateScenarioTests(SimulateScenarioTests):
  """
  Tests: simulate_scenario
  """

  def test_simulate_scenario_pygame(self):
    """
    Test: Replay scenario with pygame renderer
    """
    self.expert_trajectories = simulate_scenario(
        self.param_server, sim_time_step=self.sim_time_step, renderer="pygame")

  def test_simulate_scenario_matplotlib(self):
    """
    Test: Replay scenario with matplotlib renderer
    """
    self.expert_trajectories = simulate_scenario(
        self.param_server, sim_time_step=self.sim_time_step,
        renderer="matplotlib")


if __name__ == '__main__':
  unittest.main()
