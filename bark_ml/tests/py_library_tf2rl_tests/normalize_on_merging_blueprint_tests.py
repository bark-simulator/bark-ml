# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import joblib
import unittest
import numpy as np
from gym.spaces.box import Box
from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories import *
from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import *
from bark_project.bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
import os
from pathlib import Path

from absl import app
from absl import flags

# BARK imports
from bark_project.bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf2rl.tf2rl_wrapper import TF2RLWrapper
from bark_ml.library_wrappers.lib_tf2rl.agents.gail_agent import BehaviorGAILAgent
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner
from bark_ml.library_wrappers.lib_tf_agents.agents.sac_agent import BehaviorSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners.sac_runner import SACRunnerGenerator

class NormalizeOnMergingBlueprintTest(unittest.TestCase):
    """Tests for the normalization of expert trajectories on the merging blueprint.
    """

    def setUp(self):
        """Setup
        """
        self.params = ParameterServer(filename="bark_ml/tests/py_library_tf2rl_tests/data/params.json")
        bp = ContinuousMergingBlueprint(self.params,
                                        number_of_senarios=10,
                                        random_seed=0)
        self.env = SingleAgentRuntime(blueprint=bp,
                                render=False)
        self.wrapped_env = TF2RLWrapper(self.env, 
                        normalize_features=True)

    def generate_test_trajectory(self):
        """Generates a test trajectory from a SAC agent
        """
        sac_agent = BehaviorSACAgent(environment=self.env,
                                    params=self.params)
        self.env.ml_behavior = sac_agent
        runner = SACRunnerGenerator(params=self.params,
                            environment=self.env,
                            agent=sac_agent)
        expert_trajectories = runner.GenerateExpertTrajectories(num_trajectories=1, render=False)
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
                self.assertAlmostEqual(value, expert_trajectories['obses'][i][j], places=2)

    def test_normalization_of_sac_expert_trajectories_on_merging_blueprint_with_generate(self):
        """Tests if expert trajectories generated from the SAC agent on the merging blueprint can be normalized correctly.
        """
        # Generate
        dirname = self.generate_test_trajectory()
        joblib_files = list_files_in_dir(dirname, file_ending='.jblb')
        
        # Load raw
        raw_trajectories = joblib.load(joblib_files[0])
        for i in range(len(raw_trajectories['obs_norm'])):
            raw_trajectories['obs_norm'][i] = raw_trajectories['obs_norm'][i] * 2. - 1.

        # Load normalized
        expert_trajectories, avg_trajectory_length, num_trajectories = load_expert_trajectories(dirname,
            normalize_features=True,
            env=self.env
            ) 

        # Compare
        self.compare_trajectories(raw_trajectories, expert_trajectories)
        
        import shutil
        shutil.rmtree(dirname, ignore_errors=True)

if __name__ == '__main__':
    unittest.main()
