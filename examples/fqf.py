# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import gym
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import FQFAgent
import bark_ml.environments.gym
from bark.runtime.commons.parameters import ParameterServer
from absl import app
from absl import flags
# this will disable all BARK log messages
import os
os.environ['GLOG_minloglevel'] = '3'


# for training: bazel run //examples:fqf -- --mode=train
FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate"],
                  "Mode the configuration should be executed in.")

flags.DEFINE_enum("env",
                  "highway-v1",
                  ["highway-v1", "merging-v1", "intersection-v1"],
                  "Environment the agent should interact in.")

flags.DEFINE_bool("load", False, "Load weights from checkpoint path.")

def run_configuration(argv):

  params = ParameterServer(filename="examples/example_params/fqf_params.json")
  params["ML"]["BaseAgent"]["SummaryPath"] = "/home/mansoor/Study/Werkstudent/fortiss/code/bark-ml/summaries"
  params["ML"]["BaseAgent"]["CheckpointPath"] = "/home/mansoor/Study/Werkstudent/fortiss/code/bark-ml/checkpoints"
  
  env = gym.make(FLAGS.env, params=params)
  agent = FQFAgent(env=env, test_env=env, params = params)

  if FLAGS.load and params["ML"]["BaseAgent"]["CheckpointPath"]:
    agent.load_models(os.path.join(params["ML"]["BaseAgent"]["CheckpointPath"],"best"))

  if FLAGS.mode == "train": 
    agent.run()

  elif FLAGS.mode == "visualize":
    agent.visualize()

  elif FLAGS.mode == "evaluate":
    # writes evaluaion data using summary writer in summary path
    agent.evaluate() 

  else:
    raise Exception("Invalid argument for --mode")

if __name__ == '__main__':
  app.run(run_configuration)
