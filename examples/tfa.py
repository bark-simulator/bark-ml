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
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner, PPORunner


# for training: bazel run //examples:tfa -- --mode=train
FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate", "generate"],
                  "Mode the configuration should be executed in.")
flags.DEFINE_enum("agent",
                  "sac",
                  ["sac", "ppo"],
                  "The tfa agent type.")
flags.DEFINE_enum("blueprint",
                  "merging",
                  ["intersection", "highway", "merging"],
                  "The tfa agent type.")
flags.DEFINE_integer("num_episodes",
                  default=5,
                  help="The number of episodes to run a simulation. Defaults to 5. (Only used when visualizing.)") 
flags.DEFINE_integer("num_trajectories",
                  default=1000,
                  help="The minimal number of expert trajectories that have to be generated. Defaults to 1000. (Only used when generating.)") 
flags.DEFINE_boolean("render",
                  default=False,
                  help="Render during generation of expert trajectories.") 

default_output_dir: str = os.path.join(os.path.expanduser('~/'), 'checkpoints')
flags.DEFINE_string("output_dir",
                  default=default_output_dir,
                  help="The output directory. Defaults to " + default_output_dir)  

def save_expert_trajectories(output_dir: str, expert_trajectories: dict):
  """Saves the given expert trajectories.

  Args:
      output_dir (str): The output folder.
      expert_trajectories (dict): The expert trajectories.
  """
  _output_dir = os.path.join(os.path.expanduser(output_dir), "expert_trajectories")
  Path(_output_dir).mkdir(parents=True, exist_ok=True)

  for scenario_id, expert_trajectories in expert_trajectories.items():
    filename = os.path.join(_output_dir, f'{scenario_id}.jblb')
    joblib.dump(expert_trajectories, filename)

def run_configuration(argv):
  """Main

  Args:
      argv: The commandline argumends

  Raises:
      ValueError: If the given agent is not sac or ppo
  """
  output_dir = os.path.expanduser(FLAGS.output_dir)
  if output_dir == default_output_dir:
    output_dir = os.path.join(os.path.expanduser(FLAGS.output_dir), FLAGS.agent, FLAGS.blueprint)

  params = ParameterServer(filename="examples/example_params/tfa_params.json")
  # params = ParameterServer()
  params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = output_dir
  params["ML"]["TFARunner"]["SummaryPath"] = output_dir
  params["World"]["remove_agents_out_of_map"] = True


  # create environment
  if FLAGS.blueprint == 'merging':
    bp = ContinuousMergingBlueprint(params,
                                    number_of_senarios=2500,
                                    random_seed=0)
  elif FLAGS.blueprint == 'intersection':
    bp = ContinuousIntersectionBlueprint(params,
                                    number_of_senarios=2500,
                                    random_seed=0)
  elif FLAGS.blueprint == 'highway':
    bp = ContinuousHighwayBlueprint(params,
                                    number_of_senarios=2500,
                                    random_seed=0)
  else:
    raise ValueError(f'{FLAGS.blueprint} is no valid blueprint. See help.')

  env = SingleAgentRuntime(blueprint=bp,
                          render=False)


  if FLAGS.agent == 'ppo':
    ppo_agent = BehaviorPPOAgent(environment=env,
                                 params=params)
    env.ml_behavior = ppo_agent
    runner = PPORunner(params=params,
                       environment=env,
                       agent=ppo_agent)
  elif FLAGS.agent == 'sac':
    sac_agent = BehaviorSACAgent(environment=env,
                                params=params)
    env.ml_behavior = sac_agent
    runner = SACRunner(params=params,
                      environment=env,
                      agent=sac_agent)
  else:
    raise ValueError(f'{FLAGS.agent} is no valid agent. See help.')


  if FLAGS.mode == "train":
    runner.SetupSummaryWriter()
    runner.Train()
  elif FLAGS.mode == "visualize":
    runner.Visualize(FLAGS.num_episodes)
  elif FLAGS.mode == "generate":
    expert_trajectories = runner.GenerateExpertTrajectories(num_trajectories=FLAGS.num_trajectories, render=FLAGS.render)
    save_expert_trajectories(output_dir=output_dir, expert_trajectories=expert_trajectories)

  # store all used params of the training
  # params.Save(os.path.join(Path.home(), "examples/example_params/tfa_params.json"))
  sys.exit(0)

if __name__ == '__main__':
  app.run(run_configuration)
