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
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint, GailMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf2rl.tf2rl_wrapper import TF2RLWrapper
from bark_ml.library_wrappers.lib_tf2rl.agents.gail_agent import BehaviorGAILAgent
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner
from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories import load_expert_trajectories
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate"],
                  "Mode the configuration should be executed in.")

def run_configuration(argv):
  params = ParameterServer(filename="examples/example_params/gail_params.json")

  # create environment
  blueprint = params['World']['blueprint']
  if blueprint == 'merging':
    bp = ContinuousMergingBlueprint(params,
                                    number_of_senarios=2500,
                                    random_seed=0)
  elif blueprint == 'intersection':
    bp = ContinuousIntersectionBlueprint(params,
                                    number_of_senarios=2500,
                                    random_seed=0)
  elif blueprint == 'highway':
    bp = ContinuousHighwayBlueprint(params,
                                    number_of_senarios=2500,
                                    random_seed=0)
  else:
    raise ValueError(f'{FLAGS.blueprint} is no valid blueprint. See help.')
  
  env = SingleAgentRuntime(blueprint=bp,
                          render=False)

  # wrapped environment for compatibility with tf2rl
  wrapped_env = TF2RLWrapper(env, 
    normalize_features=params["ML"]["Settings"]["NormalizeFeatures"])

  # GAIL-agent
  gail_agent = BehaviorGAILAgent(environment=wrapped_env,
                               params=params)

  np.random.seed(123456789)
  if FLAGS.mode != 'visualize':
    expert_trajectories, avg_trajectory_length, num_trajectories = load_expert_trajectories(params['ML']['ExpertTrajectories']['expert_path_dir'],
      normalize_features=params["ML"]["Settings"]["NormalizeFeatures"],
      env=env, # the unwrapped env has to be used, since that contains the unnormalized spaces.
      subset_size=params['ML']['ExpertTrajectories']['subset_size']
      ) 
  else:
    expert_trajectories = {
      "obses": np.empty([0, 16]),
      "next_obses": np.empty([0, 16]),
      "acts": np.empty([0, 2])
    }

  runner = GAILRunner(params=params,
                     environment=wrapped_env,
                     agent=gail_agent,
                     expert_trajs=expert_trajectories)

  if FLAGS.mode == "train":
    runner.Train()
  elif FLAGS.mode == "visualize": 
    runner.Visualize(2000)
  elif FLAGS.mode == "evaluate":
    runner.Evaluate(expert_trajectories, avg_trajectory_length, num_trajectories)
  

if __name__ == '__main__':
  app.run(run_configuration)