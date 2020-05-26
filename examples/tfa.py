# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# TensorFlow Agents (https://github.com/tensorflow/agents) example
import os
from pathlib import Path

import gym
from absl import app
from absl import flags

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from modules.runtime.viewer.video_renderer import VideoRenderer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent, BehaviorPPOAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner, PPORunner


FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate"],
                  "Mode the configuration should be executed in.")


def run_configuration(argv):
  params = ParameterServer(filename="examples/example_params/tfa_params.json")
  # params = ParameterServer()
  params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = os.path.join(Path.home(), "checkpoints/")
  params["ML"]["TFARunner"]["SummaryPath"] = os.path.join(Path.home(), "checkpoints/")
  params["World"]["remove_agents_out_of_map"] = True

  # create environment
  bp = ContinuousMergingBlueprint(params,
                                  number_of_senarios=500,
                                  random_seed=0)
  env = SingleAgentRuntime(blueprint=bp,
                           render=False)

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
    runner.Train()
  elif FLAGS.mode == "visualize":
    runner.Visualize(5)
  
  # store all used params of the training
  # params.Save(os.path.join(Path.home(), "examples/example_params/tfa_params.json"))


if __name__ == '__main__':
  app.run(run_configuration)
  print('********************************************** Finished **********************************************')