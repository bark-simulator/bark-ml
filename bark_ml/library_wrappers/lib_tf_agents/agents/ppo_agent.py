# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import tensorflow as tf

# tfa
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network

from tf_agents.agents.ppo import ppo_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from bark_ml.library_wrappers.lib_tf_agents.agents.tfa_agent import BehaviorTFAAgent


class BehaviorPPOAgent(BehaviorTFAAgent):
  """BARK PPO agent."""

  def __init__(self,
               environment=None,
               params=None,
               observer=None):
    self._ppo_params = params["ML"]["BehaviorPPOAgent"]
    BehaviorTFAAgent.__init__(
      self,
      environment=environment,
      params=params,
      observer=observer)
    self._replay_buffer = self.GetReplayBuffer()
    self._collect_policy = self.GetCollectionPolicy()
    self._eval_policy = self.GetEvalPolicy()

  def GetAgent(self, env, params):
    actor_net = actor_distribution_network.ActorDistributionNetwork(
      env.observation_spec(),
      env.action_spec(),
      fc_layer_params=tuple(self._ppo_params[
        "ActorFcLayerParams", "", [512, 256, 256]]))
    value_net = value_network.ValueNetwork(
      env.observation_spec(),
      fc_layer_params=tuple(self._ppo_params[
        "CriticFcLayerParams", "", [512, 256, 256]]))

    tf_agent = ppo_agent.PPOAgent(
      env.time_step_spec(),
      env.action_spec(),
      actor_net=actor_net,
      value_net=value_net,
      normalize_observations=self._ppo_params[
        "NormalizeObservations", "", False],
      normalize_rewards=self._ppo_params["NormalizeRewards", "", False],
      optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=self._ppo_params["LearningRate", "", 3e-4]),
      train_step_counter=self._ckpt.step,
      num_epochs=self._ppo_params["NumEpochs", "", 25],
      name=self._ppo_params["AgentName", "", "ppo_agent"],
      debug_summaries=self._ppo_params["DebugSummaries", "", False])
    tf_agent.initialize()
    return tf_agent

  def GetReplayBuffer(self):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=self._agent.collect_data_spec,
      batch_size=self._ppo_params["NumParallelEnvironments", "", 1],
      max_length=self._ppo_params["ReplayBufferCapacity", "", 1000])

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