import os
import unittest
import numpy as np
from pathlib import Path

import gym

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from modules.runtime.viewer.video_renderer import VideoRenderer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf2rl.tf2rl_wrapper import TF2RLWrapper
from bark_ml.library_wrappers.lib_tf2rl.agents.gail_agent import BehaviorGAILAgent
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner


class PyTrainingBARKTests(unittest.TestCase):
    def test_training(self):
        """tests the Train() method of the GAILRunner class with a bark gym environment."""

        params = ParameterServer(filename="bark_ml/tests/gail_data/params/gail_params_bark.json")

        """if len(os.listdir(params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"])) == 0:
            print("No expert trajectories found, plaese generate demonstrations first")
            exit()"""

        # create environment
        bp = ContinuousMergingBlueprint(params,
                                        number_of_senarios=500,
                                        random_seed=0)
        env = SingleAgentRuntime(blueprint=bp,
                                render=False)

        # wrapped environment for compatibility with batk-ml
        wrapped_env = TF2RLWrapper(env)
        
        # TODO: loading the expert trajectories:
        expert_trajs = {
            'obses': np.zeros((1000, 16), dtype=np.float32),
            'next_obses': np.zeros((1000, 16), dtype=np.float32),
            'acts': np.zeros((1000, 2), dtype=np.float32)
        }

        # create agent and runner:
        agent = BehaviorGAILAgent(
            environment=wrapped_env,
            params=params
        )
        env.ml_behavior = agent
        runner = GAILRunner(
            environment=wrapped_env,
            agent=agent,
            params=params,
            expert_trajs=expert_trajs)

        runner.Train()


if __name__ == '__main__':
    unittest.main()
