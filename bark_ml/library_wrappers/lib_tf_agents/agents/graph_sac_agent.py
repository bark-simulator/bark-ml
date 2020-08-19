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
from tf_agents.agents.sac import sac_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils.common import Checkpointer
from tf_agents.trajectories import time_step as ts

from bark_ml.library_wrappers.lib_tf_agents.networks import GNNActorNetwork, GNNCriticNetwork
from bark_ml.library_wrappers.lib_tf_agents.agents.tfa_agent import BehaviorTFAAgent
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.library_wrappers.lib_tf_agents.networks.gnn_wrapper import GNNWrapper


class BehaviorGraphSACAgent(BehaviorTFAAgent):
  """
  SAC-Agent with graph neural networks.
  This agent is based on the tf-agents library.
  """
  
  def __init__(self,
               environment=None,
               observer=None,
               params=None):
    """
    Initializes a `BehaviorGraphSACAgent` instance.

    Args:
    environment:
    observer: The `GraphObserver` instance that generates the observations.
    params: A `ParameterServer` instance containing parameters to configure
      the agent.
    """
    # the super init calls 'GetAgent', so assign the observer before
    self._gnn_sac_params = params["ML"]["BehaviorGraphSACAgent"]
    self._observer = observer
    BehaviorTFAAgent.__init__(self,
                              environment=environment,
                              params=params)
    self._replay_buffer = self.GetReplayBuffer()
    self._dataset = self.GetDataset()
    self._collect_policy = self.GetCollectionPolicy()
    self._eval_policy = self.GetEvalPolicy()

  def GetAgent(self, env, params):
    

    def init_gnn(name):
      """
      Returns a new `GNNWrapper`instance with the given `name`.
      We need this function to be able to prefix the variable
      names with the names of the parent actor or critic network,
      by passing in this function and initializing the instance in 
      the parent network.
      """
      return GNNWrapper(
        params=self._gnn_sac_params["GNN"], 
        graph_dims=self._observer.graph_dimensions,
        name=name)

    # actor network
    actor_net = GNNActorNetwork(
      input_tensor_spec=env.observation_spec(),
      output_tensor_spec=env.action_spec(),
      gnn=init_gnn,
      fc_layer_params=self._gnn_sac_params[
        "ActorFcLayerParams", "", [128, 64]]
    )

    # critic network
    critic_net = GNNCriticNetwork(
      (env.observation_spec(), env.action_spec()),
      gnn=init_gnn,
      observation_fc_layer_params=self._gnn_sac_params[
        "CriticObservationFcLayerParams", "", [128]],
      action_fc_layer_params=self._gnn_sac_params[
        "CriticActionFcLayerParams", "", None],
      joint_fc_layer_params=self._gnn_sac_params[
        "CriticJointFcLayerParams", "", [128, 128]]
    )
    
    # agent
    tf_agent = sac_agent.SacAgent(
      env.time_step_spec(),
      env.action_spec(),
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=self._gnn_sac_params["ActorLearningRate", "", 3e-4]),
      critic_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=self._gnn_sac_params["CriticLearningRate", "", 3e-4]),
      alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=self._gnn_sac_params["AlphaLearningRate", "", 3e-4]),
      target_update_tau=self._gnn_sac_params["TargetUpdateTau", "", 0.05],
      target_update_period=self._gnn_sac_params["TargetUpdatePeriod", "", 3],
      td_errors_loss_fn=tf.math.squared_difference,
      gamma=self._gnn_sac_params["Gamma", "", 0.995],
      reward_scale_factor=self._gnn_sac_params["RewardScaleFactor", "", 1.],
      train_step_counter=self._ckpt.step,
      name=self._gnn_sac_params["AgentName", "", "gnn_sac_agent"],
      debug_summaries=self._gnn_sac_params["DebugSummaries", "", False])
    
    tf_agent.initialize()
    return tf_agent

  def GetReplayBuffer(self):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=self._agent.collect_data_spec,
      batch_size=self._wrapped_env.batch_size,
      max_length=self._gnn_sac_params["ReplayBufferCapacity", "", 10000])

  def GetDataset(self):
    dataset = self._replay_buffer.as_dataset(
      num_parallel_calls=self._gnn_sac_params["ParallelBufferCalls", "", 1],
      sample_batch_size=self._gnn_sac_params["BatchSize", "", 512],
      num_steps=self._gnn_sac_params["BufferNumSteps", "", 2]) \
        .prefetch(self._gnn_sac_params["BufferPrefetch", "", 3])
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