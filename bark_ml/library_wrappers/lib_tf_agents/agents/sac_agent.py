# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import tensorflow as tf

# BARK
from bark.core.models.behavior import BehaviorModel

# tfa
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.policies import greedy_policy

from tf_agents.agents.sac import sac_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from bark_ml.library_wrappers.lib_tf_agents.agents.tfa_agent import BehaviorTFAAgent


class BehaviorSACAgent(BehaviorTFAAgent):
  """SAC-Agent

  Agent based on the tf-agents library.
  """

  def __init__(self,
               environment=None,
               params=None,
               observer=None):
    self._sac_params = params["ML"]["BehaviorSACAgent"]
    BehaviorTFAAgent.__init__(
      self,
      environment=environment,
      params=params,
      observer=observer)
    BehaviorModel.__init__(self, params)
    self._replay_buffer = self.GetReplayBuffer()
    self._dataset = self.GetDataset()
    self._collect_policy = self.GetCollectionPolicy()
    self._eval_policy = self.GetEvalPolicy()

  def GetAgent(self, env, params):
    def _normal_projection_net(action_spec, init_means_output_factor=0.1):
      return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=sac_agent.std_clip_transform,
        scale_distribution=True)

    # actor network
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=tuple(
          self._sac_params["ActorFcLayerParams", "", [512, 256, 256]]),
        continuous_projection_net=_normal_projection_net)

    # critic network
    critic_net = critic_network.CriticNetwork(
      (env.observation_spec(), env.action_spec()),
      observation_fc_layer_params=None,
      action_fc_layer_params=None,
      joint_fc_layer_params=tuple(
        self._sac_params["CriticJointFcLayerParams", "", [512, 256, 256]]))

    # agent
    tf_agent = sac_agent.SacAgent(
      env.time_step_spec(),
      env.action_spec(),
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=self._sac_params["ActorLearningRate", "", 3e-4]),
      critic_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=self._sac_params["CriticLearningRate", "", 3e-4]),
      alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=self._sac_params["AlphaLearningRate", "", 3e-4]),
      target_update_tau=self._sac_params["TargetUpdateTau", "", 0.05],
      target_update_period=self._sac_params["TargetUpdatePeriod", "", 3],
      td_errors_loss_fn=tf.math.squared_difference,
      gamma=self._sac_params["Gamma", "", 0.995],
      reward_scale_factor=self._sac_params["RewardScaleFactor", "", 1.],
      train_step_counter=self._ckpt.step,
      name=self._sac_params["AgentName", "", "sac_agent"],
      debug_summaries=self._sac_params["DebugSummaries", "", False])

    tf_agent.initialize()
    return tf_agent

  def GetReplayBuffer(self):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=self._agent.collect_data_spec,
      batch_size=self._wrapped_env.batch_size,
      max_length=self._sac_params["ReplayBufferCapacity", "", 100000])

  def GetDataset(self):
    dataset = self._replay_buffer.as_dataset(
      num_parallel_calls=self._sac_params["ParallelBufferCalls", "", 1],
      sample_batch_size=self._sac_params["BatchSize", "", 512],
      num_steps=self._sac_params["BufferNumSteps", "", 2]) \
        .prefetch(self._sac_params["BufferPrefetch", "", 3])
    return dataset

  def GetCollectionPolicy(self):
    return self._agent.collect_policy

  def GetEvalPolicy(self):
    return greedy_policy.GreedyPolicy(self._agent.policy)

  def Reset(self):
    pass

  @property
  def collect_policy(self):
    return self._collect_policy

  @property
  def eval_policy(self):
    return self._eval_policy

