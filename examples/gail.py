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
from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories import load_expert_trajectories
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate"],
                  "Mode the configuration should be executed in.")

def run_configuration(argv):
  params = ParameterServer(filename="examples/example_params/gail_params.json")

  # Uncomment these to use the pretrained agents from https://github.com/GAIL-4-BARK/large_data_store
  # The agents are automatically integrated using bazel together with the expert trajectories
  # params["ML"]["GAILRunner"]["tf2rl"]["logdir"] = "../com_github_gail_4_bark_large_data_store/pretrained_agents/gail/merging"
  # params["ML"]["GAILRunner"]["tf2rl"]["model_dir"] = "../com_github_gail_4_bark_large_data_store/pretrained_agents/gail/merging"

  # When training a gail agent we add a suffix to the specified model and log dir to distinguish between training runs.
  # If you want to visualize or evaluate using your locally trained gail agent, you have to specify the run to use.
  # Therefore look into the directory specified in params["ML"]["GAILRunner"]["tf2rl"]["logdir"] and 
  # pick one of your runs with the naming scheme '<timestamp>_DDPG_GAIL'
  # Add the name of the folder with the run to your:
  # params["ML"]["GAILRunner"]["tf2rl"]["logdir"] and params["ML"]["GAILRunner"]["tf2rl"]["model_dir"]
  # So if your model_dir and logdir were 'examples/gail_training' it becomes 'examples/gail_training/20200807T121018.454776_DDPG_GAIL' in the gail_params.json
  # Alternatively append it as in the following lines:
  # params["ML"]["GAILRunner"]["tf2rl"]["logdir"] = os.path.join(params["ML"]["GAILRunner"]["tf2rl"]["logdir"], "20200807T121018.454776_DDPG_GAIL")
  # params["ML"]["GAILRunner"]["tf2rl"]["model_dir"] = os.path.join(params["ML"]["GAILRunner"]["tf2rl"]["model_dir"], "20200807T121018.454776_DDPG_GAIL")

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
  gail_agent = BehaviorGAILAgent(environment=wrapped_env, params=params)

  # np.random.seed(123456789)
  expert_trajectories = None
  if FLAGS.mode != 'visualize':
    expert_trajectories, avg_trajectory_length, num_trajectories = load_expert_trajectories(params['ML']['ExpertTrajectories']['expert_path_dir'],
      normalize_features=params["ML"]["Settings"]["NormalizeFeatures"],
      env=env, # the unwrapped env has to be used, since that contains the unnormalized spaces.
      subset_size=params['ML']['ExpertTrajectories']['subset_size']
      ) 

  runner = GAILRunner(params=params,
                     environment=wrapped_env,
                     agent=gail_agent,
                     expert_trajs=expert_trajectories)

  if FLAGS.mode == "train":
    runner.Train()
  elif FLAGS.mode == "visualize": 
    runner.Visualize(params["Visualization"]["NumberOfScenarios"])
  elif FLAGS.mode == "evaluate":
    runner.Evaluate(expert_trajectories, avg_trajectory_length, num_trajectories)
  

if __name__ == '__main__':
  app.run(run_configuration)