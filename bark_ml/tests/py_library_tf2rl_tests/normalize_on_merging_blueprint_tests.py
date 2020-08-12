# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import joblib
import unittest

# BARK imports
from bark_project.bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.tf2rl_wrapper import TF2RLWrapper
from bark_ml.library_wrappers.lib_tf2rl.agents.gail_agent import BehaviorGAILAgent
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner
from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories import load_expert_trajectories
from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import list_files_in_dir
from bark_ml.tests.py_library_tf2rl_tests.generate_sac_trajectory_tests_base import GenerateSACTrajectoryTestsBase


class NormalizeOnMergingBlueprintTest(GenerateSACTrajectoryTestsBase):
  """Tests for the normalization of expert trajectories on the merging blueprint.
  """

  def setUp(self):
    """Setup
    """
    super().setUp()
    self.wrapped_env = TF2RLWrapper(self.env,
                                    normalize_features=True)

  def test_normalization_of_sac_expert_trajectories_on_merging_blueprint_with_generate(self):
    """Tests if expert trajectories generated from the SAC agent on the merging blueprint can be normalized correctly.
    """
    dirname = self.generate_test_trajectory()
    joblib_files = list_files_in_dir(dirname, file_ending='.jblb')

    # Load raw
    raw_trajectories = joblib.load(joblib_files[0])
    for i in range(len(raw_trajectories['obs_norm'])):
      raw_trajectories['obs_norm'][i] = raw_trajectories['obs_norm'][i] * 2. - 1.

    # Load normalized
    expert_trajectories, avg_trajectory_length, num_trajectories = load_expert_trajectories(
      dirname, normalize_features=True, env=self.env)

    # Compare
    self.compare_trajectories(raw_trajectories, expert_trajectories)
    self.remove(dirname)


if __name__ == '__main__':
  unittest.main()
