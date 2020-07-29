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
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["library"] = "tf2_gnn" # "tf2_gnn" or "spektral"
params["ML"]["SACRunner"]["NumberOfCollections"] = int(1e6)
# tf2_gnn
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["message_calculation_class"] = "ggnn"
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["global_exchange_mode"] = "gru"

# spektral
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["MPLayerActivation"] = "relu"
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["DenseActication"] = "tanh"



layers_gnn = 1 + int(random() * 4)
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["NumLayers"] = layers_gnn
#units_gnn = 2**(5 + int(random() * 3))

mp_layers = 5 + int(random() * 10)
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["MpLayerNumUnits"] = mp_layers

mp_channels = 2**(5 + int(random() * 4))
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["MPChannels"] = mp_channels

critic_fc_1 = 2**(6 + int(random() * 4)) 
critic_fc_2 = 2**(5 + int(random() * 4))  
params["ML"]["BehaviorSACAgent"]["CriticJointFcLayerParams"] = [critic_fc_1, critic_fc_2]


actor_fc_1 = 2**(6 + int(random() * 4)) 
actor_fc_2 = 2**(6 + int(random() * 4))  
params["ML"]["BehaviorGraphSACAgent"]["ActorFcLayerParams"] = [actor_fc_1, actor_fc_2]

spek_k_1 = 2**(6 + int(random() * 4)) 
spek_k_2 = 2**(6 + int(random() * 4))  
params["ML"]["BehaviorGraphSACAgent"]["GNN"]["KernelNetUnits"] = [spek_k_1, spek_k_2]

summary_path_hyperparameter_run = tfa_gnn_summary_path + f'/{int(time.time())}__ \
        layers_{layers_gnn}__mp_layers_{mp_layers}__mp_channels_{mp_channels}__ \
        critic_fc_{critic_fc_1}_{critic_fc_2}__actor_fc_{actor_fc_1}_{actor_fc_2}__ \
        spek_k_{spek_k_1}_{spek_k_2}'

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
