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

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint, DiscreteMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent, BehaviorPPOAgent, BehaviorCDQNAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner, PPORunner, CDQNRunner


# for training: bazel run //examples:tfa -- --mode=train
FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate"],
                  "Mode the configuration should be executed in.")


def run_configuration(argv):
  # params = ParameterServer(filename="examples/example_params/tfa_params_discrete.json")
  params = ParameterServer()
  # NOTE: Modify these paths in order to save the checkpoints and summaries
  params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = "/Users/hart/Development/bark-ml/checkpoints/"
  params["ML"]["TFARunner"]["SummaryPath"] = "/Users/hart/Development/bark-ml/summaries/"
  params["ML"]["TFARunner"]["ModelPath"] = "/Users/hart/Development/bark-ml/model/"
  params["World"]["remove_agents_out_of_map"] = True

  # create environment
  # discrete environment (to use with CDQN)
  bp = DiscreteMergingBlueprint(params,
                                number_of_senarios=2500,
                                random_seed=0)

  # continuous environment
  #bp = ContinuousMergingBlueprint(params,
  #                                number_of_senarios=2500,
  #                                random_seed=0)

  #env = SingleAgentRuntime(blueprint=bp,
  #                         render=False)

  # CDQN-agent
  cdqn_agent = BehaviorCDQNAgent(environment=env,
                                params=params)
  env.ml_behavior = cdqn_agent
  runner = CDQNRunner(params=params,
                      environment=env,
                      agent=cdqn_agent)
  
  # PPO-agent
  # ppo_agent = BehaviorPPOAgent(environment=env,
  #                              params=params)
  # env.ml_behavior = ppo_agent
  # runner = PPORunner(params=params,
  #                    environment=env,
  #                    agent=ppo_agent)

  # SAC-agent
  #sac_agent = BehaviorSACAgent(environment=env,
  #                             params=params)
  #env.ml_behavior = sac_agent
  #runner = SACRunner(params=params,
  #                   environment=env,
  #                   agent=sac_agent)
  if FLAGS.mode == "train":
    runner.SetupSummaryWriter()
    runner.Train()
  elif FLAGS.mode == "visualize":
    runner.Visualize(5)
  
  # store all used params of the training
  # params.Save("/home/hart/Dokumente/2020/bark-ml/examples/example_params/tfa_params_discrete.json")


if __name__ == '__main__':
  app.run(run_configuration)