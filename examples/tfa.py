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
# import os
# os.environ['GLOG_minloglevel'] = '3' 

# BARK imports
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer

# BARK-ML imports
from bark_ml.environments.blueprints import DiscreteHighwayBlueprint, ContinuousHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent, BehaviorPPOAgent, BehaviorGraphSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner, PPORunner


# for training: bazel run //examples:tfa -- --mode=train
FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate"],
                  "Mode the configuration should be executed in.")

def run_configuration(argv):
  #params = ParameterServer(filename="examples/example_params/tfa_params.json")
  params = ParameterServer()
  # NOTE: Modify these paths in order to save the checkpoints and summaries
  params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = "/home/silvan/working_bark/training_sac/checkpoints"
  params["ML"]["TFARunner"]["SummaryPath"] = "/home/silvan/working_bark/training_sac/summary"
  params["ML"]["BehaviorSACAgent"]["DebugSummaries"] = True
  params["World"]["remove_agents_out_of_map"] = False
  params["ML"]["SACRunner"]["NumberOfCollections"] = 20000
  params["ML"]["SACRunner"]["EvaluateEveryNSteps"] = 50
  params["ML"]["BehaviorSACAgent"]["BatchSize"] = 32


  # viewer = MPViewer(
  #   params=params,
  #   x_range=[-35, 35],
  #   y_range=[-35, 35],
  #   follow_agent_id=True)
  
  # viewer = VideoRenderer(
  #   renderer=viewer,
  #   world_step_time=0.2,
  #   fig_path="/Users/marco.oliva/2020/bark-ml/video/")

  # create environment
  bp = ContinuousHighwayBlueprint(params,
                                  number_of_senarios=2500,
                                  random_seed=0)

  env = SingleAgentRuntime(blueprint=bp,
                           render=False)

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
    runner.Visualize(5)
  elif FLAGS.mode == "evaluate":
    runner.Evaluate()
  
  # store all used params of the training
  # params.Save("/home/hart/Dokumente/2020/bark-ml/examples/example_params/tfa_params.json")


if __name__ == '__main__':
  app.run(run_configuration)
