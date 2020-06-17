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
from bark_ml.library_wrappers.lib_tf2rl.agents.gail_agent import BehaviorGAILAgent
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner


FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate"],
                  "Mode the configuration should be executed in.")

flags.DEFINE_string("train_out",
                  help="The absolute path to where the checkpoints and summaries are saved during training.",
                  # default=os.path.join(Path.home(), ".bark-ml/gail")
                  default=os.path.join(Path.home(), "")
                  )

flags.DEFINE_string("test_env",
                  help="Example environment in accord with tf2rl to test our code.",
                  default="Pendulum-v0"
                  )

flags.DEFINE_string("gpu",
                  help="-1 for CPU, 0 for GPU",
                  default=0
                  )


def run_configuration(argv):
  params = ParameterServer(filename="examples/example_params/gail_params.json")
  # params = ParameterServer()
  # changing the logging directories if not the default is used. (Which would be the same as it is in the json file.)
  params["ML"]["GAILRunner"]["tf2rl"]["logdir"] = os.path.join(FLAGS.train_out, "logs", "bark")
  params["ML"]["GAILRunner"]["tf2rl"]["model_dir"] = os.path.join(FLAGS.train_out, "models", "bark")

  params["World"]["remove_agents_out_of_map"] = True
  params["ML"]["Settings"]["GPUUse"] = FLAGS.gpu

  # create environment
  bp = ContinuousMergingBlueprint(params,
                                  number_of_senarios=500,
                                  random_seed=0)
  env = SingleAgentRuntime(blueprint=bp,
                           render=False)

  # Only for test purposes
  # env = gym.make(FLAGS.test_env)

  # SAC-agent
  # sac_agent = BehaviorSACAgent(environment=env,
  #                              params=params)
  # env.ml_behavior = sac_agent
  # runner = SACRunner(params=params,
  #                    environment=env,
  #                    agent=sac_agent)

  # GAIL-agent
  gail_agent = BehaviorGAILAgent(environment=env,
                               params=params)
  env.ml_behavior = gail_agent
  runner = GAILRunner(params=params,
                     environment=env,
                     agent=gail_agent)

  if FLAGS.mode == "train":
    runner.Train()
  elif FLAGS.mode == "visualize":
    runner.Visualize(5)
  
  # store all used params of the training
  params.Save(os.path.join(FLAGS.train_out, "examples/example_params/gail_params.json"))


if __name__ == '__main__':
  app.run(run_configuration)