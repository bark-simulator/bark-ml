# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import shutil
import joblib
import unittest
from pathlib import Path
from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories import load_expert_trajectories
from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import list_files_in_dir

# BARK imports
from bark_project.bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents.sac_agent import BehaviorSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners.sac_runner import SACRunnerGenerator


class GenerateSACTrajectoryTestsBase(unittest.TestCase):
  """Baseclass for tests that need a SAC trajectory to be generated.
  """

  def setUp(self):
    """Setup
    """
    self.params = ParameterServer(
      filename="bark_ml/tests/py_library_tf2rl_tests/data/params.json")
    bp = ContinuousMergingBlueprint(self.params,
                                    number_of_senarios=10,
                                    random_seed=0)
    self.env = SingleAgentRuntime(blueprint=bp,
                                  render=False)

  def generate_test_trajectory(self):
    """Generates a test trajectory from a SAC agent
    """
    sac_agent = BehaviorSACAgent(environment=self.env,
                                 params=self.params)
    self.env.ml_behavior = sac_agent
    runner = SACRunnerGenerator(params=self.params,
                                environment=self.env,
                                agent=sac_agent)
    expert_trajectories = runner.GenerateExpertTrajectories(
        num_trajectories=1, render=False)
    dirname = "test_expert_trajectories"
    Path(dirname).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(dirname, "test_expert_trajectory.jblb")
    joblib.dump(expert_trajectories[0], filename)
    return dirname

  def compare_trajectories(self, raw_trajectories, expert_trajectories):
    """Compares two trajectories
    """
    values = ['X', 'Y', 'Theta', 'Vel']
    for i, raw in enumerate(raw_trajectories['obs_norm']):
      if i >= len(expert_trajectories['obses']):
        break
      for j, value in enumerate(raw):
        print(values[j % 4], ':')
        print(value)
        print(expert_trajectories['obses'][i][j])
        print('*' * 80)
        self.assertAlmostEqual(
            value, expert_trajectories['obses'][i][j],
            places=2)

  def remove(self, directory):
    """Teardown
    """
    shutil.rmtree(directory, ignore_errors=True)
