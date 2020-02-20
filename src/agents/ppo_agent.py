import tensorflow as tf

# tfa
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.policies import greedy_policy

from tf_agents.agents.ppo import ppo_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils.common import Checkpointer
from tf_agents.trajectories import time_step as ts

from src.agents.tfa_agent import TFAAgent


class PPOAgent(TFAAgent):
  """PPO-Agent
     This agent is based on the tf-agents library.
  """
  def __init__(self,
               environment=None,
               replay_buffer=None,
               checkpointer=None,
               dataset=None,
               params=None,
               use_rnns=False):
    self._use_rnns = use_rnns
    TFAAgent.__init__(self,
                      environment=environment,
                      params=params)
    self._replay_buffer = self.get_replay_buffer()
    # self._dataset = self.get_dataset()
    self._collect_policy = self.get_collect_policy()
    self._eval_policy = self.get_eval_policy()

  def get_agent(self, env, params):
    """Returns a TensorFlow PPO-Agent
    
    Arguments:
        env {TFAPyEnvironment} -- Tensorflow-Agents PyEnvironment
        params {ParameterServer} -- ParameterServer from BARK
    
    Returns:
        agent -- tf-agent
    """

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=tuple(
          self._params["ML"]["Agent"]["actor_fc_layer_params"]))
    value_net = value_network.ValueNetwork(
      env.observation_spec(),
      fc_layer_params=tuple(
        self._params["ML"]["Agent"]["critic_fc_layer_params"]))

    # agent
    tf_agent = ppo_agent.PPOAgent(
      env.time_step_spec(),
      env.action_spec(),
      actor_net=actor_net,
      value_net=value_net,
      normalize_observations=self._params["ML"]["Agent"]["normalize_observations", "", False],
      normalize_rewards=self._params["ML"]["Agent"]["normalize_rewards", "", False],
      optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=self._params["ML"]["Agent"]["learning_rate", "", 3e-4]),
      train_step_counter=self._ckpt.step,
      num_epochs=self._params["ML"]["Agent"]["num_epochs", "", 30],
      name=self._params["ML"]["Agent"]["agent_name", "", "ppo_agent"],
      debug_summaries=self._params["ML"]["Agent"]["debug_summaries", "", False])
    tf_agent.initialize()
    return tf_agent

  def get_replay_buffer(self):
    """Replay buffer
    
    Returns:
        ReplayBuffer -- tf-agents replay buffer
    """
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=self._agent.collect_data_spec,
      batch_size=self._params["ML"]["Agent"]["num_parallel_environments"],
      max_length=self._params["ML"]["Agent"]["replay_buffer_capacity"])

  def get_collect_policy(self):
    """Returns the collection policy of the agent
    
    Returns:
        CollectPolicy -- Samples from the agent's distribution
    """
    return self._agent.collect_policy

  def get_eval_policy(self):
    """Returns the greedy policy of the agent
    
    Returns:
        GreedyPolicy -- Always returns best suitable action
    """
    return self._agent.policy

  def reset(self):
    pass

  @property
  def collect_policy(self):
    return self._collect_policy

  @property
  def eval_policy(self):
    return self._eval_policy

  def act(self, state):
    """ see base class
    """
    action_step = self.eval_policy.action(
      ts.transition(state, reward=0.0, discount=1.0))
    return action_step.action.numpy()