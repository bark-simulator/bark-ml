import os
import math
import pickle
import unittest
import numpy as np
import shutil
from bark_ml.library_wrappers.lib_tf2rl.generate_expert_trajectories \
    import *

tracks_folder = os.path.join(
    os.path.dirname(__file__),
    'data/interaction_data_set_mock/DR_DEU_Merging_MT/tracks')
map_file = os.path.join(
    os.path.dirname(__file__),
    'data/interaction_data_set_mock/DR_DEU_Merging_MT/map/DR_DEU_Merging_MT_v01_shifted.xodr')
known_key = ('DR_DEU_Merging_MT_v01_shifted',
             'vehicle_tracks_013')


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
    self.assertIs(
        track_files[0].endswith(
            'bark_ml/tests/py_library_tf2rl_tests/data/interaction_data_set_mock/DR_DEU_Merging_MT/tracks/vehicle_tracks_013.csv'),
        True)

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
    self.assertEqual(
        param_server["Scenario"]["Generation"]
        ["InteractionDatasetScenarioGeneration"]["TrackIds"],
        track_ids)

    self.assertEqual(param_server["Scenario"]["Generation"]
                     ["InteractionDatasetScenarioGeneration"]["StartTs"], 100)
    self.assertEqual(param_server["Scenario"]["Generation"]
                     ["InteractionDatasetScenarioGeneration"]["EndTs"], 327300)

    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["EgoTrackId"] = 1


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
