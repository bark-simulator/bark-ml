import tensorflow as tf

# tfa
from tf_agents.networks import categorical_q_network

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils.common import Checkpointer
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

from bark_ml.library_wrappers.lib_tf_agents.agents.tfa_agent import BehaviorTFAAgent
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteML


class BehaviorCDQNAgent(BehaviorTFAAgent, BehaviorDiscreteML):
  """SAC-Agent
     This agent is based on the tf-agents library.
  """
  def __init__(self,
               environment=None,
               params=None):
    BehaviorTFAAgent.__init__(self,
                              environment=environment,
                              params=params)
    BehaviorDiscreteML.__init__(self, params)
    self._replay_buffer = self.GetReplayBuffer()
    self._dataset = self.GetDataset()
    self._collect_policy = self.GetCollectionPolicy()
    self._eval_policy = self.GetEvalPolicy()

  def GetAgent(self, env, params):
    # categorical q network
    categorical_q_net = categorical_q_network.CategoricalQNetwork(
      env.observation_spec(),
      env.action_spec(),
      num_atoms=self._params["ML"]["BehaviorCDQNAgent"]["NumAtoms", "", 51],
      fc_layer_params=tuple(
        self._params["ML"]["BehaviorCDQNAgent"]["CategoricalFcLayerParams", "", [300, 300, 300]]))
    
    # agent
    tf_agent = categorical_dqn_agent.CategoricalDqnAgent(
      env.time_step_spec(),
      env.action_spec(),
      categorical_q_network=categorical_q_net,
      optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=self._params["ML"]["BehaviorCDQNAgent"]["LearningRate", "", 5e-5]),
      min_q_value=self._params["ML"]["BehaviorCDQNAgent"]["MinQValue", "", -10],
      max_q_value=self._params["ML"]["BehaviorCDQNAgent"]["MaxQValue", "", 10],
      epsilon_greedy=self._params["ML"]["BehaviorCDQNAgent"]["EpsilonGreedy", "", 0.3],
      n_step_update=self._params["ML"]["BehaviorCDQNAgent"]["nStepUpdate", "", 2],
      target_update_tau=self._params["ML"]["BehaviorCDQNAgent"]["TargetUpdateTau", "", 0.01],
      target_update_period=self._params["ML"]["BehaviorCDQNAgent"]["TargetUpdatePeriod", "", 1],
      td_errors_loss_fn=common.element_wise_squared_loss,
      gamma=self._params["ML"]["BehaviorCDQNAgent"]["Gamma", "", 0.995],
      reward_scale_factor=self._params["ML"]["BehaviorCDQNAgent"]["RewardScaleFactor", "", 1.],
      train_step_counter=self._ckpt.step,
      name=self._params["ML"]["BehaviorCDQNAgent"]["AgentName", "", "cdqn_agent"],
      debug_summaries=self._params["ML"]["BehaviorCDQNAgent"]["DebugSummaries", "", False])
    
    tf_agent.initialize()
    return tf_agent

  def GetReplayBuffer(self):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=self._agent.collect_data_spec,
      batch_size=self._wrapped_env.batch_size,
      max_length=self._params["ML"]["BehaviorCDQNAgent"]["ReplayBufferCapacity", "", 10000])

  def GetDataset(self):
    dataset = self._replay_buffer.as_dataset(
      num_parallel_calls=self._params["ML"]["BehaviorCDQNAgent"]["ParallelBufferCalls", "", 1],
      sample_batch_size=self._params["ML"]["BehaviorCDQNAgent"]["BatchSize", "", 512],
      num_steps=self._params["ML"]["BehaviorCDQNAgent"]["BufferNumSteps", "", 2]+1) \
        .prefetch(self._params["ML"]["BehaviorCDQNAgent"]["BufferPrefetch", "", 2])
    return dataset

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