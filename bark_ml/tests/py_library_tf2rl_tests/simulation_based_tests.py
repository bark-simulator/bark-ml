import os
import unittest
import numpy as np
import shutil
from bark_ml.library_wrappers.lib_tf2rl.generate_expert_trajectories \
    import *
from base_tests import tracks_folder, map_file, known_key
from simulation_based_tests import *


class SimulationBasedTests(unittest.TestCase):
  """
  Base class for simulation based tests
  """

  def setUp(self):
    """
    Setup
    """
    param_servers = create_parameter_servers_for_scenarios(
        map_file, tracks_folder)
    self.assertIn(known_key, param_servers)

    param_server = param_servers[known_key]
    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["TrackIds"] = [
        63, 64, 65, 66, 67, 68]
    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["StartTs"] = 232000
    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["EndTs"] = 259000
    self.test_agent_id = 65
    param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["EgoTrackId"] = self.test_agent_id
    self.param_server = param_server
    self.expert_trajectories = {}
    self.sim_time_step = 100


class SimulateScenarioTests(SimulationBasedTests):
  """
  Tests: simulate_scenario
  """

  def test_simulate_scenario_no_renderer(self):
    """
    Test: Replay scenario with no renderer
    """
    self.expert_trajectories = simulate_scenario(
        self.param_server, sim_time_step=self.sim_time_step)

  def tearDown(self):
    """
    Tear down
    """
    for agent_id in range(63, 69):
      self.assertIn(agent_id, self.expert_trajectories)
      agent_expert_trajectories = self.expert_trajectories[agent_id]
      for key in ['obs', 'time', 'wheelbase']:
        self.assertIn(key, agent_expert_trajectories)
        assert len(agent_expert_trajectories[key]) == 269

      for wheelbase in agent_expert_trajectories['wheelbase']:
        assert wheelbase == 2.7

      times = agent_expert_trajectories['time']
      for i in range(1, len(times)):
        self.assertAlmostEqual(
            times[i] - times[i - 1],
            self.sim_time_step / 1000, delta=1e-5)


class MeasureWorldTests(SimulationBasedTests):
  """
  Tests: measure_world
  """

  def setUp(self):
    """
    Setup
    """
    super().setUp()
    scenario, start_ts, end_ts = create_scenario(self.param_server)
    self.sim_time_step = 100
    self.sim_steps = int((end_ts - start_ts) / (self.sim_time_step))
    self.sim_time_step_seconds = self.sim_time_step / 1000

    self.observer = NearestAgentsObserver(self.param_server)

    self.observer_not_normalized = NearestAgentsObserver(self.param_server)
    self.observer_not_normalized._NormalizationEnabled = False

    self.world_state = scenario.GetWorldState()

  def test_measure_world(self):
    """
    Test: measure_world
    """
    current_measurement = np.array([0.0] * 12)
    previous_measurement = np.zeros_like(current_measurement)
    max_values = np.zeros_like(previous_measurement)
    min_values = np.zeros_like(previous_measurement)

    for _ in range(0, self.sim_steps):
      self.world_state.Step(self.sim_time_step_seconds)

      previous_measurement = current_measurement
      all_measurements = measure_world(
        self.world_state, self.observer, self.observer_not_normalized)

      if self.test_agent_id in all_measurements:
        current_measurement = all_measurements[self.test_agent_id]['obs']

        difference = current_measurement - previous_measurement
        self.set_min_max_values(difference, min_values, max_values)

    expected_max_values = np.array([
        996.13000488, 1004.25897217, 3.06100011, 1.68449903,
        998.4039917, 1006.92498779, 3.08599997, 2.19838786,
        990.78497314, 1007.76800537, 3.07599998, 2.25579476])
    expected_min_values = np.array([
        -3.55957031e-01, -1.70288086e-02, -4.99963760e-03, -7.10122585e-02,
        -2.02299194e+01, -5.71960449e-01, -1.70185804e-01, -8.40493798e-01,
        -1.93980713e+01, -6.87011719e-01, -1.70184374e-01, -4.82387066e-01])

    for i in range(expected_max_values.shape[0]):
      self.assertAlmostEqual(expected_max_values[i], max_values[i])
      self.assertAlmostEqual(expected_min_values[i], min_values[i])

  @staticmethod
  def set_min_max_values(difference, min_values, max_values):
    """Updates the current min and max values with the calculated differences. 

    Args:
        difference (list): The calculated differences
        min_values (list): The current min values
        max_values (list): The current max values
    """
    for i in range(difference.shape[0]):
      if max_values[i] < difference[i]:
        max_values[i] = difference[i]
      if min_values[i] > difference[i]:
        min_values[i] = difference[i]


class GenerateExpertTrajectoriesForScenarioTests(SimulationBasedTests):
  """
  Tests: generate_expert_trajectories_for_scenario
  """

  def test_generate_expert_trajectories_for_scenario(self):
    """
    Test: generate_expert_trajectories_for_scenario
    """
    self.expert_trajectories = generate_expert_trajectories_for_scenario(
        self.param_server, self.sim_time_step)

    for agent_id in self.expert_trajectories:
      actions = self.expert_trajectories[agent_id]['act']
      assert len(actions) == len(self.expert_trajectories[agent_id]['obs'])
      assert len(actions[0]) == 2

      dones = self.expert_trajectories[agent_id]['done']
      for i in range(len(dones) - 1):
        dones[i] == 0

      dones[-1] = 1

    print('end')


class GenerateAndStoreExpertTrajectories(SimulationBasedTests):
  """
  Tests: generate_and_store_expert_trajectories
  """

  def assertListAlmostEqual(self, a: list, b: list):
    """Asserts that the two given lists are almost equal
    """
    self.assertEqual(len(a), len(b))
    for i in range(len(a)):
      self.assertAlmostEqual(a[i], b[i])

  def assert_file_equal(self, expected_path: str, generated_path: str):
    """Asserts that the given files contain the same expert trajectories.

    Args:
        expected_path (str): The path to the expected trajectories.
        generated_path (str): The path to the generated trajectories.
    """
    loaded_expert_trajectories = joblib.load(expected_path)
    generated_expert_trajectories = joblib.load(generated_path)

    for key in ['obs', 'act']:
      self.assertIn(key, loaded_expert_trajectories.keys())
      self.assertIn(key, generated_expert_trajectories.keys())

    self.assertEqual(
        len(loaded_expert_trajectories['obs']),
        len(generated_expert_trajectories['obs']))

    for i in range(len(loaded_expert_trajectories['obs'])):
      self.assertListAlmostEqual(
          list(loaded_expert_trajectories['obs'][i]),
          list(generated_expert_trajectories['obs'][i]))
      self.assertListAlmostEqual(
          list(loaded_expert_trajectories['act'][i]),
          list(generated_expert_trajectories['act'][i]))

  def test_generate_and_store_expert_trajectories(self):
    """
    Test: generate_and_store_expert_trajectories
    """
    store_path = os.path.join(
        os.path.dirname(__file__),
        'test_generate_and_store_expert_trajectories')
    filenames = generate_and_store_expert_trajectories(
        known_key[0],
        known_key[1],
        store_path, self.param_server, sim_time_step=self.sim_time_step)

    expected_paths = [os.path.join(
        os.path.dirname(__file__),
        'data', 'expert_trajectories', f.split('/')[-1])
        for f in filenames]

    for i in range(len(expected_paths)):
      self.assert_file_equal(expected_paths[i], filenames[i])

    shutil.rmtree(store_path)

  def assert_equals(self, actual_values, expected_values):
    """Asserts that the given actual values are equal to the expected values.

    Args:
        actual_values (list): The calculated actual values
        expected_values (list): The expected values
    """
    for i in range(len(actual_values)):
      if type(actual_values[i]) == list:
        self.assertListAlmostEqual(actual_values[i], expected_values[i])
      elif type(actual_values[i]) == np.ndarray:
        assert (actual_values[i] == expected_values[i]).all()
      else:
        self.assertEqual(actual_values[i], expected_values[i])


if __name__ == '__main__':
  unittest.main()
