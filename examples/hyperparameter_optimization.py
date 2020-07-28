#!/usr/bin/env python3 

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
from random import random
import time



params = ParameterServer(filename="examples/example_params/tfa_params.json")
#params = ParameterServer()
# NOTE: Modify these paths in order to save the checkpoints and summaries
from config import tfa_gnn_checkpoint_path, tfa_gnn_summary_path
params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = tfa_gnn_checkpoint_path
params["World"]["remove_agents_out_of_map"] = False
params["ML"]["BehaviorSACAgent"]["DebugSummaries"] = True
params["ML"]["SACRunner"]["EvaluateEveryNSteps"] = 100
params["ML"]["BehaviorSACAgent"]["BatchSize"] = 128
params["ML"]["GraphObserver"]["AgentLimit"] = 4
params["ML"]["BehaviorSACAgent"]["CriticJointFcLayerParams"] = [256, 128]
params["ML"]["BehaviorGraphSACAgent"]["ActorFcLayerParams"] = [256, 256]
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["MpLayerNumUnits"] = 11
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["library"] = "tf2_gnn" # "tf2_gnn" or "spektral"
params["ML"]["SACRunner"]["NumberOfCollections"] = int(1e6)
# tf2_gnn
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["message_calculation_class"] = "ggnn"
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["global_exchange_mode"] = "gru"

# spektral
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["MPChannels"] = 128
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["KernelNetUnits"] = [256, 256]
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["MPLayerActivation"] = "relu"
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["DenseActication"] = "tanh"



layers_gnn = 1 + int(random() * 4)
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["NumLayers"] = layers_gnn
#units_gnn = 2**(5 + int(random() * 3))


summary_path_hyperparameter_run = tfa_gnn_summary_path + f'/{int(time.time())}__ \
        layers_{layers_gnn}'
params["ML"]["TFARunner"]["SummaryPath"] = summary_path_hyperparameter_run



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

runner.Train()

if __name__ == '__main__':
  app.run(run_configuration)
