# Copyright (c) 2020 fortiss GmbH
#
# Authors: Marco Oliva, Silvan Wimmer
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# TensorFlow Agents (https://github.com/tensorflow/agents) example

import gym
import numpy as np
from absl import app
from absl import flags

# this will disable all BARK log messages
# import os
# os.environ['GLOG_minloglevel'] = '3'

# BARK imports
from bark.runtime.commons.parameters import ParameterServer
from bark.core.models.behavior import BehaviorConstantAcceleration

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousMergingBlueprint
from bark_ml.environments.counterfactual_runtime import CounterfactualRuntime
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
  # params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = "/Users/hart/Development/bark-ml/checkpoints/"
  # params["ML"]["TFARunner"]["SummaryPath"] = "/Users/hart/Development/bark-ml/checkpoints/"
  params["Visualization"]["Agents"]["Alpha"]["Other"] = 0.2
  params["Visualization"]["Agents"]["Alpha"]["Controlled"] = 0.2
  params["Visualization"]["Agents"]["Alpha"]["Controlled"] = 0.2
  params["ML"]["VisualizeCfWorlds"] = False
  params["ML"]["VisualizeCfHeatmap"] = False
  # params["ML"]["ResultsFolder"] = "/Users/hart/Development/bark-ml/results/data/"

  # viewer = MPViewer(
  #   params=params,
  #   x_range=[-35, 35],
  #   y_range=[-35, 35],
  #   follow_agent_id=True)


  # create environment
  bp = ContinuousMergingBlueprint(params,
                                  num_scenarios=2500,
                                  random_seed=0)

  observer = GraphObserver(params=params)

  behavior_model_pool = []
  for count, a in enumerate([-5., 0., 5.]):
    local_params = params.AddChild("local_"+str(count))
    local_params["BehaviorConstantAcceleration"]["ConstAcceleration"] = a
    behavior = BehaviorConstantAcceleration(local_params)
    behavior_model_pool.append(behavior)

  env = CounterfactualRuntime(
    blueprint=bp,
    observer=observer,
    render=False,
    params=params,
    behavior_model_pool=behavior_model_pool)
  sac_agent = BehaviorGraphSACAgent(environment=env,
                                    observer=observer,
                                    params=params)
  env.ml_behavior = sac_agent
  runner = SACRunner(params=params,
                     environment=env,
                     agent=sac_agent)

  if FLAGS.mode == "train":
    runner.SetupSummaryWriter()
    runner.Train()
  elif FLAGS.mode == "visualize":
    runner._environment._max_col_rate = 0.
    runner.Run(num_episodes=1, render=True)
  elif FLAGS.mode == "evaluate":
    for cr in np.arange(0, 1, 0.1):
      runner._environment._max_col_rate = cr
      runner.Run(num_episodes=250, render=False, max_col_rate=cr)
    runner._environment._tracer.Save(
      params["ML"]["ResultsFolder"] + "evaluation_results_runtime.pckl")
    goal_reached = runner._tracer.success_rate
    runner._tracer.Save(
      params["ML"]["ResultsFolder"] + "evaluation_results_runner.pckl")

  # store all used params of the training
  # params.Save("your_path_here/tfa_sac_gnn_params.json")

if __name__ == '__main__':
  app.run(run_configuration)
