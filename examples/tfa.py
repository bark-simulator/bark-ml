# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# TensorFlow Agents (https://github.com/tensorflow/agents) example
import gym
from absl import app
from absl import flags

# this will disable all BARK log messages
import os
os.environ['GLOG_minloglevel'] = '3'

# BARK imports
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner


# for training: bazel run //examples:tfa -- --mode=train
FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate"],
                  "Mode the configuration should be executed in.")


def run_configuration(argv):
  params = ParameterServer(filename="examples/example_params/tfa_params.json")
  # params = ParameterServer()
  # NOTE: Modify these paths in order to save the checkpoints and summaries
  # params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = "/Users/hart/Development/bark-ml/checkpoints_merging_nn/"
  # params["ML"]["TFARunner"]["SummaryPath"] = "/Users/hart/Development/bark-ml/checkpoints_merging_nn/"
  params["Visualization"]["Agents"]["Alpha"]["Other"] = 0.2
  params["Visualization"]["Agents"]["Alpha"]["Controlled"] = 0.2
  params["Visualization"]["Agents"]["Alpha"]["Controlled"] = 0.2
  params["ML"]["VisualizeCfWorlds"] = False
  params["ML"]["VisualizeCfHeatmap"] = True
  params["World"]["remove_agents_out_of_map"] = False

  viewer = MPViewer(
   params=params,
   x_range=[-35, 35],
   y_range=[-35, 35],
   follow_agent_id=True)
  # viewer = VideoRenderer(
  #  renderer=viewer,
  #  world_step_time=0.2,
  #  fig_path="/Users/hart/Development/bark-ml/videos/normal")

  # create environment
  bp = ContinuousMergingBlueprint(params,
                                  num_scenarios=10000,
                                  random_seed=0)
  env = SingleAgentRuntime(blueprint=bp,
                           render=False,
                           viewer=viewer)

  # PPO-agent
  # ppo_agent = BehaviorPPOAgent(environment=env,
  #                              params=params)
  # env.ml_behavior = ppo_agent
  # runner = PPORunner(params=params,
  #                    environment=env,
  #                    agent=ppo_agent)

  # SAC-agent
  sac_agent = BehaviorSACAgent(environment=env,
                               params=params)
  env.ml_behavior = sac_agent
  runner = SACRunner(params=params,
                     environment=env,
                     agent=sac_agent)
  if FLAGS.mode == "train":
    runner.SetupSummaryWriter()
    runner.Train()
  elif FLAGS.mode == "visualize":
    runner.Run(num_episodes=50, render=True)
  elif FLAGS.mode == "evaluate":
    runner.Run(num_episodes=100, render=False)

  # store all used params of the training
  # params.Save("YOUR_PATH")


if __name__ == '__main__':
  app.run(run_configuration)