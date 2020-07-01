# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# TensorFlow Agents (https://github.com/tensorflow/agents) example
import os
import sys
from pathlib import Path
import pickle

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
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent, BehaviorPPOAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner, PPORunner


# for training: bazel run //examples:tfa -- --mode=train
FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate", "generate"],
                  "Mode the configuration should be executed in.")
flags.DEFINE_integer("num_episodes",
                  default=5,
                  help="The number of episodes to run a simulation. Defaults to 5. (Ignored when training.)") 
flags.DEFINE_integer("num_trajectories",
                  default=1000,
                  help="The minimal number of expert trajectories that have to be generated. Defaults to 1000. (Only used when generating.)") 
flags.DEFINE_boolean("render",
                  default=True,
                  help="Render during generation of expert trajectories.") 

default_output_file: str = os.path.join(os.path.dirname(__file__), "expert_trajectories.pkl")
flags.DEFINE_string("output_file",
                  default=default_output_file,
                  help="The minimal number of expert trajectories that have to be generated. Defaults to " + default_output_file)  

def save_expert_trajectories(output_file: str, expert_trajectories: dict):
  _output_file = os.path.expanduser(output_file)
  Path(os.path.dirname(_output_file)).mkdir(parents=True, exist_ok=True)

  with open(_output_file, 'wb') as handle:
      pickle.dump(expert_trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_configuration(argv):
  params = ParameterServer(filename="examples/example_params/tfa_params.json")
  # params = ParameterServer()
  params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = os.path.join(Path.home(), "checkpoints/")
  params["ML"]["TFARunner"]["SummaryPath"] = os.path.join(Path.home(), "checkpoints/")
  params["World"]["remove_agents_out_of_map"] = True

  # create environment
  bp = ContinuousMergingBlueprint(params,
                                  number_of_senarios=2500,
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
    runner.SetupSummaryWriter()
    runner.Train()
  elif FLAGS.mode == "visualize":
    runner.Visualize(FLAGS.num_episodes)
  elif FLAGS.mode == "generate":
    expert_trajectories = runner.GenerateExpertTrajectories(num_trajectories=FLAGS.num_trajectories, render=FLAGS.render)
    save_expert_trajectories(output_file=FLAGS.output_file, expert_trajectories=expert_trajectories)
  # store all used params of the training
  # params.Save(os.path.join(Path.home(), "examples/example_params/tfa_params.json"))
  sys.exit(0)

if __name__ == '__main__':
  app.run(run_configuration)
