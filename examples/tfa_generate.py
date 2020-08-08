# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# TensorFlow Agents (https://github.com/tensorflow/agents) example
import os
import sys
from pathlib import Path
import joblib

import gym
from absl import app
from absl import flags

# this will disable all BARK log messages
import os
os.environ['GLOG_minloglevel'] = '3' 

# BARK imports
from bark_project.bark.runtime.commons.parameters import ParameterServer
from bark_project.bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark_project.bark.runtime.viewer.video_renderer import VideoRenderer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent, BehaviorPPOAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunnerGenerator

FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "generate",
                  ["train", "visualize", "generate"],
                  "Mode the configuration should be executed in.")

def save_expert_trajectories(output_dir: str, expert_trajectories: dict):
  """Saves the given expert trajectories.

  Args:
      output_dir (str): The output folder.
      expert_trajectories (dict): The expert trajectories.
  """
  Path(output_dir).mkdir(parents=True, exist_ok=True)

  for scenario_id, expert_trajectories in expert_trajectories.items():
    filename = os.path.join(output_dir, f'{scenario_id}.jblb')
    joblib.dump(expert_trajectories, filename)

def run_configuration(argv):
  """ Main """
  params = ParameterServer(filename="examples/example_params/tfa_params.json")
  # params = ParameterServer()
  output_dir = params["GenerateExpertTrajectories"]["OutputDirectory"]

  # create environment
  blueprint = params["World"]["Blueprint"]
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
    raise ValueError(f'{blueprint} is no valid blueprint. See help.')

  env = SingleAgentRuntime(blueprint=bp,
                          render=False)


  sac_agent = BehaviorSACAgent(environment=env,
                              params=params)
  env.ml_behavior = sac_agent
  runner = SACRunnerGenerator(params=params,
                    environment=env,
                    agent=sac_agent)

  if FLAGS.mode == "train":
    runner.SetupSummaryWriter()
    runner.Train()
  elif FLAGS.mode == "visualize":
    runner.Visualize(params["Visualization"]["NumberOfEpisodes"])
  elif FLAGS.mode == "generate":
    expert_trajectories = runner.GenerateExpertTrajectories(num_trajectories=params["GenerateExpertTrajectories"]["NumberOfTrajectories"], render=params["World"]["render"])
    save_expert_trajectories(output_dir=output_dir, expert_trajectories=expert_trajectories)

  # store all used params of the training
  # params.Save(os.path.join(Path.home(), "examples/example_params/tfa_params.json"))
  sys.exit(0)

if __name__ == '__main__':
  app.run(run_configuration)
