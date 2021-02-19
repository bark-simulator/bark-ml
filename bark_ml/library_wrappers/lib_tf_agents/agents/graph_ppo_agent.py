# Copyright (c) 2020 fortiss GmbH
#
# Authors: Marco Oliva
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# bark
from bark.core.models.behavior import BehaviorModel, BehaviorDynamicModel

import tensorflow as tf

from tf_agents.policies import greedy_policy
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils.common import Checkpointer
from tf_agents.trajectories import time_step as ts

from bark_ml.library_wrappers.lib_tf_agents.networks import GNNActorDistributionNetwork, GNNValueNetwork
from bark_ml.library_wrappers.lib_tf_agents.agents.tfa_agent import BehaviorTFAAgent
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.library_wrappers.lib_tf_agents.agents.gnn_initializers import init_interaction_network


class BehaviorGraphPPOAgent(BehaviorTFAAgent):
  """
  ppo-Agent with graph neural networks.
  This agent is based on the tf-agents library.
  """
  
  def __init__(self,
               environment=None,
               observer=None,
               params=None,
               init_gnn='init_gat'):
    """
    Initializes a `BehaviorGraphppoAgent` instance.

    Args:
    environment:
    observer: The `GraphObserver` instance that generates the observations.
    params: A `ParameterServer` instance containing parameters to configure
      the agent.
    """
    # the super init calls 'GetAgent', so assign the observer before
    self._gnn_ppo_params = params["ML"]["BehaviorGraphPPOAgent"]
    self._init_gnn = eval(init_gnn)
    BehaviorTFAAgent.__init__(self,
                              environment=environment,
                              params=params,
                              observer=observer)
    self._replay_buffer = self.GetReplayBuffer()
    self._collect_policy = self.GetCollectionPolicy()
    self._eval_policy = self.GetEvalPolicy()

  def GetAgent(self, env, params):
    self._params["ML"]["GraphDims"] = self._observer.graph_dimensions
    # actor network
    actor_net = GNNActorDistributionNetwork(
      input_tensor_spec=env.observation_spec(),
      output_tensor_spec=env.action_spec(),
      gnn=self._init_gnn,
      fc_layer_params=self._gnn_ppo_params[
        "ActorFcLayerParams", "", [512, 256, 256]],
      params=params
    )

    # critic network
    value_net = GNNValueNetwork(
      env.observation_spec(),
      gnn=self._init_gnn,
      fc_layer_params=tuple(self._gnn_ppo_params[
        "CriticFcLayerParams", "", [512, 256, 256]]),
      params=params
    )
    
    # agent
    tf_agent = ppo_agent.PPOAgent(
      env.time_step_spec(),
      env.action_spec(),
      optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=3e-4),
      actor_net=actor_net,
      value_net=value_net,
      normalize_observations=False,
      normalize_rewards=False,
      use_gae=False,
      debug_summaries=True,
      summarize_grads_and_vars=False,
      train_step_counter=self._ckpt.step)
    
    tf_agent.initialize()
    return tf_agent

  def GetReplayBuffer(self):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=self._agent.collect_data_spec,
      batch_size=self._wrapped_env.batch_size,
      max_length=self._gnn_ppo_params["ReplayBufferCapacity", "", 10000])

  def GetCollectionPolicy(self):
    return self._agent.collect_policy

  def GetEvalPolicy(self):
    return self._agent.policy

  def Reset(self):
    pass

  @property
  def collect_policy(self):
    return self._collect_policy

  @property
  def eval_policy(self):
    return self._eval_policy