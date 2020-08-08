# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# TensorFlow Agents (https://github.com/tensorflow/agents) example
import gym
from absl import app
from absl import flags
import tensorflow as tf

# this will disable all BARK log messages
# import os
# os.environ['GLOG_minloglevel'] = '3' 

# BARK imports
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent, BehaviorPPOAgent, BehaviorGraphSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner, PPORunner
from bark_ml.observers.graph_observer import GraphObserver


# for training: bazel run //examples:tfa -- --mode=train
FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate"],
                  "Mode the configuration should be executed in.")

def run_configuration(argv):
  params = ParameterServer(filename="examples/example_params/tfa_sac_gnn_example_params.json")
  params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = '/Users/marco.oliva/Development/bark-ml_logs/checkpoints/2'
  params["ML"]["TFARunner"]["SummaryPath"] = '/Users/marco.oliva/Development/bark-ml_logs/summaries/1'
  params["ML"]["SACRunner"]["NumberOfCollections"] = int(1e6)
  params["ML"]["GraphObserver"]["AgentLimit"] = 4
  params["ML"]["BehaviorGraphSACAgent"]["DebugSummaries"] = False
  params["ML"]["BehaviorGraphSACAgent"]["BatchSize"] = 256
  params["ML"]["BehaviorGraphSACAgent"]["CriticJointFcLayerParams"] = [128, 128]
  params["ML"]["BehaviorGraphSACAgent"]["CriticObservationFcLayerParams"] = [128]
  params["ML"]["BehaviorGraphSACAgent"]["ActorFcLayerParams"] = [128, 64]
  params["ML"]["BehaviorGraphSACAgent"]["GNN"]["NumMpLayers"] = 1
  params["ML"]["BehaviorGraphSACAgent"]["GNN"]["MpLayerNumUnits"] = 128
  params["ML"]["BehaviorGraphSACAgent"]["GNN"]["library"] = "spektral" # "tf2_gnn" or "spektral"

  # (n_nodes, n_features, n_edge_features)
  params["ML"]["BehaviorGraphSACAgent"]["GNN"]["GraphDimensions"] = (4, 11, 4)

  # tf2_gnn
  # NOTE: when using the ggnn mp class, MPLayerUnits must match n_features!
  params["ML"]["BehaviorGraphSACAgent"]["GNN"]["message_calculation_class"] = "rgcn"
  params["ML"]["BehaviorGraphSACAgent"]["GNN"]["global_exchange_mode"] = "gru"
  params["ML"]["BehaviorGraphSACAgent"]["GNN"]["dense_every_num_layers"] = 1
  params["ML"]["BehaviorGraphSACAgent"]["GNN"]["global_exchange_every_num_layers"] = 1
  
  # spektral
  params["ML"]["BehaviorGraphSACAgent"]["GNN"]["MPChannels"] = 128
  params["ML"]["BehaviorGraphSACAgent"]["GNN"]["KernelNetUnits"] = [512, 256]
  params["ML"]["BehaviorGraphSACAgent"]["GNN"]["MPLayerActivation"] = "relu"
  params["ML"]["BehaviorGraphSACAgent"]["GNN"]["DenseActication"] = "relu"


    # viewer = MPViewer(
  #   params=params,
  #   x_range=[-35, 35],

  #   y_range=[-35, 35],
  #   follow_agent_id=True)
  
  # viewer = VideoRenderer(
  #   renderer=viewer,
  #   world_step_time=0.2,
  #   fig_path="/Users/marco.oliva/2020/bark-ml/video/")

  #tf.summary.trace_on(graph=True, profiler=True)

  # create environment
  bp = ContinuousHighwayBlueprint(params,
                                  number_of_senarios=2500,
                                  random_seed=0)

  observer = GraphObserver(params=params)
  
  env = SingleAgentRuntime(
    blueprint=bp,
    observer=observer,
    render=False)

  sac_agent = BehaviorGraphSACAgent(environment=env,
                                    params=params)
  env.ml_behavior = sac_agent
  runner = SACRunner(params=params,
                     environment=env,
                     agent=sac_agent)
  runner.SetupSummaryWriter()

  if FLAGS.mode == "train":
    runner.Train()
  elif FLAGS.mode == "visualize":
    runner.Visualize(5)
  elif FLAGS.mode == "evaluate":
    runner.Evaluate()
  
  # store all used params of the training
  # params.Save("/home/hart/Dokumente/2020/bark-ml/examples/example_params/tfa_params.json")

if __name__ == '__main__':
  app.run(run_configuration)
