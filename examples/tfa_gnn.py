# Copyright (c) 2020 fortiss GmbH
#
# Authors: Marco Oliva, Silvan Wimmer
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# TensorFlow Agents (https://github.com/tensorflow/agents) example
import gym # pylint: disable=unused-import
from absl import app
from absl import flags

# this will disable all BARK log messages
# import os
# os.environ['GLOG_minloglevel'] = '3'

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorGraphSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner
from bark_ml.observers.graph_observer import GraphObserver

# for training: bazel run //examples:tfa -- --mode=train
FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate"],
                  "Mode the configuration should be executed in.")

def run_configuration(argv):
  params = ParameterServer()

  # NOTE: Modify these paths to specify your preferred path for checkpoints and summaries
  # params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = "/Users/hart/Development/bark-ml/checkpoints_merge_spektral_att2/"
  # params["ML"]["TFARunner"]["SummaryPath"] = "/Users/hart/Development/bark-ml/checkpoints_merge_spektral_att2/"

  #viewer = MPViewer(
  #  params=params,
  #  x_range=[-35, 35],
  #  y_range=[-35, 35],
  #  follow_agent_id=True)
  #viewer = VideoRenderer(
  #  renderer=viewer,
  #  world_step_time=0.2,
  #  fig_path="/your_path_here/training/video/")

  # create environment
  bp = ContinuousMergingBlueprint(params,
                                  num_scenarios=2500,
                                  random_seed=0)

  observer = GraphObserver(params=params)

  env = SingleAgentRuntime(
    blueprint=bp,
    observer=observer,
    render=False)
  sac_agent = BehaviorGraphSACAgent(environment=env,
                                    observer=observer,
                                    params=params,
                                    init_gnn='init_interaction_network')
  env.ml_behavior = sac_agent
  runner = SACRunner(params=params,
                     environment=env,
                     agent=sac_agent)

  if FLAGS.mode == "train":
    runner.SetupSummaryWriter()
    runner.Train()
  elif FLAGS.mode == "visualize":
    runner.Run(num_episodes=10, render=True)
  elif FLAGS.mode == "evaluate":
    runner.Run(num_episodes=250, render=False)

  # store all used params of the training
  # params.Save("your_path_here/tfa_sac_gnn_params.json")

if __name__ == '__main__':
  app.run(run_configuration)
