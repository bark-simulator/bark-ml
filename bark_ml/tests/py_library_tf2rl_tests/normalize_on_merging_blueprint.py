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

    def setUp(self):
        params = ParameterServer(filename="bark_ml/tests/py_library_tf2rl_tests/data/gail_params.json")
        bp = ContinuousMergingBlueprint(params,
                                        number_of_senarios=10,
                                        random_seed=0)

        env = SingleAgentRuntime(blueprint=bp,
                                render=False)

        # wrapped environment for compatibility with tf2rl
        wrapped_env = TF2RLWrapper(env, 
        normalize_features=True)

        # GAIL-agent
        gail_agent = BehaviorGAILAgent(environment=wrapped_env, params=params)
        expert_path_dir = "../com_github_gail_4_bark_large_data_store/expert_trajectories/sac/merging"
        subset_size=1
        
        np.random.seed(0)
        joblib_files = list_files_in_dir(expert_path_dir, file_ending='.jblb')
        indices = np.random.choice(len(joblib_files), subset_size, replace=False)
        joblib_files = np.array(joblib_files)[indices]

        raw_trajectories = joblib.load(joblib_files[0])

        np.random.seed(0)
        expert_trajectories, avg_trajectory_length, num_trajectories = load_expert_trajectories(expert_path_dir,
            normalize_features=True,
            env=env, # the unwrapped env has to be used, since that contains the unnormalized spaces.
            subset_size=subset_size
            ) 

        runner = GAILRunner(params=params,
                            environment=wrapped_env,
                            agent=gail_agent,
                            expert_trajs=expert_trajectories)

    def test_test(self):
        print('lol')

if __name__ == '__main__':
    unittest.main()
