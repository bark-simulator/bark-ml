import tensorflow as tf

# BARK imports
from bark.core.models.behavior import BehaviorModel, BehaviorDynamicModel

# tfa
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.policies import greedy_policy

from tf_agents.agents.ppo import ppo_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import time_step as ts

from bark_ml.library_wrappers.lib_tf_agents.agents.tfa_agent import BehaviorTFAAgent
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.commons.py_spaces import BoundedContinuous


class BehaviorPPOAgent(BehaviorTFAAgent, BehaviorContinuousML):
  def __init__(self,
               environment=None,
               params=None):
    BehaviorTFAAgent.__init__(self,
                      environment=environment,
                      params=params)
    BehaviorContinuousML.__init__(self, params)
    self._replay_buffer = self.GetReplayBuffer()
    self._collect_policy = self.GetCollectionPolicy()
    self._eval_policy = self.GetEvalPolicy()

  def GetAgent(self, env, params):
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=tuple(
          self._params["ML"]["BehaviorPPOAgent"][
            "ActorFcLayerParams", "", [512, 256, 256]]))
    value_net = value_network.ValueNetwork(
      env.observation_spec(),
      fc_layer_params=tuple(
        self._params["ML"]["BehaviorPPOAgent"][
          "CriticFcLayerParams", "", [512, 256, 256]]))

    tf_agent = ppo_agent.PPOAgent(
      env.time_step_spec(),
      env.action_spec(),
      actor_net=actor_net,
      value_net=value_net,
      normalize_observations=self._params["ML"]["BehaviorPPOAgent"]
      ["NormalizeObservations", "", False],
      normalize_rewards=self._params["ML"]["BehaviorPPOAgent"][
        "NormalizeRewards", "", False],
      optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=self._params["ML"]["BehaviorPPOAgent"][
            "LearningRate", "", 3e-4]),
      train_step_counter=self._ckpt.step,
      num_epochs=self._params["ML"]["BehaviorPPOAgent"]["NumEpochs", "", 30],
      name=self._params["ML"]["BehaviorPPOAgent"][
        "AgentName", "", "ppo_agent"],
      debug_summaries=self._params["ML"]["BehaviorPPOAgent"][
        "DebugSummaries", "", False])
    tf_agent.initialize()
    return tf_agent

  def GetReplayBuffer(self):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=self._agent.collect_data_spec,
      batch_size=self._params["ML"]["BehaviorPPOAgent"][
        "NumParallelEnvironments", "", 1],
      max_length=self._params["ML"]["BehaviorPPOAgent"][
        "ReplayBufferCapacity", "", 1000])

  def GetCollectionPolicy(self):
    return self._agent.collect_policy

  def GetEvalPolicy(self):
    return self._agent.policy

  def Reset(self):
    pass

  def Clone(self):
    pass

  @property
  def collect_policy(self):
    return self._collect_policy

  @property
  def eval_policy(self):
    return self._eval_policy

  def Act(self, state):
    action_step = self.eval_policy.action(
      ts.transition(state, reward=0.0, discount=1.0))
    return action_step.action.numpy()

  def Plan(self, observed_world, dt):
    if self._training == True:
      observed_state = self._environment._observer.Observe(observed_world)
      action = self.Act(observed_state)
      super().ActionToBehavior(action)
    return super().Plan(observed_world, dt)