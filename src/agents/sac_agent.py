import tensorflow as tf

# tfa
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.policies import greedy_policy

from tf_agents.agents.sac import sac_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils.common import Checkpointer
from tf_agents.trajectories import time_step as ts

from src.agents.tfa_agent import TFAAgent

class SACAgent(TFAAgent):
  """SAC-Agent
     This agent is based on the tf-agents library.
  """
  def __init__(self,
               environment=None,
               replay_buffer=None,
               checkpointer=None,
               dataset=None,
               params=None):
    TFAAgent.__init__(self,
                      environment=environment,
                      params=params)
    self._replay_buffer = self.get_replay_buffer()
    self._dataset = self.get_dataset()
    self._collect_policy = self.get_collect_policy()
    self._eval_policy = self.get_eval_policy()

  def get_agent(self, env, params):
    """Returns a TensorFlow SAC-Agent
    
    Arguments:
        env {TFAPyEnvironment} -- Tensorflow-Agents PyEnvironment
        params {ParameterServer} -- ParameterServer from BARK
    
    Returns:
        agent -- tf-agent
    """
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
          self._params["ML"]["Agent"]["actor_fc_layer_params", "", [512, 256, 256]]),
        continuous_projection_net=_normal_projection_net)

    # critic network
    critic_net = critic_network.CriticNetwork(
      (env.observation_spec(), env.action_spec()),
      observation_fc_layer_params=None,
      action_fc_layer_params=None,
      joint_fc_layer_params=tuple(
        self._params["ML"]["Agent"]["critic_joint_fc_layer_params", "", [512, 256, 256]]))
    
    # agent
    tf_agent = sac_agent.SacAgent(
      env.time_step_spec(),
      env.action_spec(),
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=self._params["ML"]["Agent"]["actor_learning_rate", "", 3e-4]),
      critic_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=self._params["ML"]["Agent"]["critic_learning_rate", "", 3e-4]),
      alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=self._params["ML"]["Agent"]["alpha_learning_rate", "", 3e-4]),
      target_update_tau=self._params["ML"]["Agent"]["target_update_tau", "", 0.005],
      target_update_period=self._params["ML"]["Agent"]["target_update_period", "", 3],
      td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
      gamma=self._params["ML"]["Agent"]["gamma", "", 0.995],
      reward_scale_factor=self._params["ML"]["Agent"]["reward_scale_factor", "", 1.],
      gradient_clipping=self._params["ML"]["Agent"]["gradient_clipping"],
      train_step_counter=self._ckpt.step,
      name=self._params["ML"]["Agent"]["agent_name"],
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
      batch_size=self._env.batch_size,
      max_length=self._params["ML"]["Agent"]["replay_buffer_capacity", "", 10000])

  def get_dataset(self):
    """Dataset generated of the replay buffer
    
    Returns:
        dataset -- subset of experiences
    """
    dataset = self._replay_buffer.as_dataset(
      num_parallel_calls=self._params["ML"]["Agent"]["parallel_buffer_calls", "", 1],
      sample_batch_size=self._params["ML"]["Agent"]["batch_size", "", 256],
      num_steps=self._params["ML"]["Agent"]["buffer_num_steps", "", 1]) \
        .prefetch(self._params["ML"]["Agent"]["buffer_prefetch", "", 2])
    return dataset

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
    return greedy_policy.GreedyPolicy(self._agent.policy)

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
