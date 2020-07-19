# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import numpy as np
import time
import tensorflow as tf

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent, BehaviorGraphSACAgent
from bark_ml.observers.graph_observer import GraphObserver

class RandomActorNet:
    def __init__(self, low=-0.4, high=0.4):
        self.low = low
        self.high = high
        
    def __call__(self, inputs, **args):
        #logging.info(inputs.shape)
        size = (inputs.shape[0], 2)
        predictions = np.random.uniform(low=self.low, high=self.high, size=size)
        return tf.constant(predictions, dtype=tf.float32)

class ConstantActorNet:
    def __init__(self, constants=np.array([0.025, -0.066])):
        self.constants = constants

    def __call__(self, inputs, **args):
        #logging.info(inputs.shape)
        size = (inputs.shape[0], 2)
        predictions = np.resize(self.constants, size)
        return tf.constant(predictions, dtype=tf.float32)

def get_GNN_SAC_actor_net(num_scenarios, params, observer):
    """Function that returns GNN SAC agent's actor net"""
    bp = ContinuousHighwayBlueprint(params, number_of_senarios=num_scenarios, random_seed=0)
    env = SingleAgentRuntime(blueprint=bp, observer=observer, render=False)
    sac_agent = BehaviorGraphSACAgent(environment=env, params=params)
    actor_net = sac_agent._agent._actor_network
    return actor_net
    
def get_SAC_actor_net(num_scenarios, params, observer):
    """Function that returns normal SAC agent's actor net """
    bp = ContinuousHighwayBlueprint(params, number_of_senarios=num_scenarios, random_seed=0)
    env = SingleAgentRuntime(blueprint=bp, observer=observer, render=False)
    sac_agent = BehaviorSACAgent(environment=env, params=params)
    actor_net = sac_agent._agent._actor_network
    return actor_net