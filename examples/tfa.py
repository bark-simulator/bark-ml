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
from bark.runtime.viewer.video_renderer import VideoRenderer
from bark.core.models.behavior import BehaviorConstantAcceleration

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.environments.counterfactual_runtime import CounterfactualRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent, BehaviorPPOAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner, PPORunner


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
  params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = "/Users/hart/Development/bark-ml/nn_checkpoints/"
  params["ML"]["TFARunner"]["SummaryPath"] = "/Users/hart/Development/bark-ml/nn_checkpoints/"
  params["Visualization"]["Agents"]["Alpha"]["Other"] = 0.2
  params["Visualization"]["Agents"]["Alpha"]["Controlled"] = 0.2
  params["Visualization"]["Agents"]["Alpha"]["Controlled"] = 0.2
  params["ML"]["VisualizeCfWorlds"] = False
  params["ML"]["VisualizeCfHeatmap"] = True
  params["World"]["remove_agents_out_of_map"] = False
  params["ML"]["ResultsFolder"] = "/Users/hart/Development/bark-ml/results/data/"
  
  # create environment
  bp = ContinuousHighwayBlueprint(params,
                                  number_of_senarios=10000,
                                  random_seed=0)
  env = SingleAgentRuntime(blueprint=bp,
                           render=False)

  # behavior_model_pool = []
  # for count, a in enumerate([-2., 0., 2.]):
  #   local_params = params.AddChild("local_"+str(count))
  #   local_params["BehaviorConstantAcceleration"]["ConstAcceleration"] = a
  #   behavior = BehaviorConstantAcceleration(local_params)
  #   behavior_model_pool.append(behavior)
  # env = CounterfactualRuntime(
  #   blueprint=bp,
  #   render=False,
  #   params=params,
  #   behavior_model_pool=behavior_model_pool)
  
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
    runner.Run(num_episodes=10, render=True)
  elif FLAGS.mode == "evaluate":
    runner.Run(num_episodes=100, render=False)
  
  # store all used params of the training
  # params.Save("YOUR_PATH")


if __name__ == '__main__':
  app.run(run_configuration)