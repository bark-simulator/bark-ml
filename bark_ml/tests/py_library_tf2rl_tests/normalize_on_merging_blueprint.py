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

class NormalizeOnMergingBlueprintTest(unittest.TestCase):

    def test_normalization_of_sac_expert_trajectories_on_merging_blueprint(self):
        params = ParameterServer(filename="bark_ml/tests/py_library_tf2rl_tests/data/gail_params.json")
        bp = ContinuousMergingBlueprint(params,
                                        number_of_senarios=10,
                                        random_seed=0)

        env = SingleAgentRuntime(blueprint=bp,
                                render=False)

        # wrapped environment for compatibility with tf2rl
        wrapped_env = TF2RLWrapper(env, 
        normalize_features=True)

        expert_path_dir = "bark_ml/tests/py_library_tf2rl_tests/data/expert_trajectories/sac"
        
        seed = 0
        np.random.seed(seed=seed)
        joblib_files = list_files_in_dir(expert_path_dir, file_ending='.jblb')
        
        raw_trajectories = joblib.load(joblib_files[0])
        for i in range(len(raw_trajectories['obs_norm'])):
            raw_trajectories['obs_norm'][i] = raw_trajectories['obs_norm'][i] * 2. - 1.

        np.random.seed(seed=seed)
        expert_trajectories, avg_trajectory_length, num_trajectories = load_expert_trajectories(expert_path_dir,
            normalize_features=True,
            env=env, # the unwrapped env has to be used, since that contains the unnormalized spaces.
            ) 

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


if __name__ == '__main__':
    unittest.main()
