import tensorflow as tf

# tfa
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.policies import greedy_policy

from tf_agents.agents.sac import sac_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils.common import Checkpointer

from src.agents.base_agent import BaseAgent

# TODO(@hart): create a TFABaseAgent with all checkpointing, saving, loading etc. functionality
class SACAgent(BaseAgent):
  def __init__(self,
               environment=None,
               replay_buffer=None,
               checkpointer=None,
               dataset=None,
               params=None):
    BaseAgent.__init__(self,
                       params=params)
    self._ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64))
    self._agent = self.get_agent(environment, params)
    self._env = environment
    self._replay_buffer = self.get_replay_buffer()
    self._ckpt_manager  = self.get_checkpointer()
    self._dataset = self.get_dataset()
    self._collect_policy = self.get_collect_policy()
    self._eval_policy = self.get_eval_policy()
    self._env = environment
    self._ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                     agent=self._agent)

  def get_agent(self, env, params):
    """Returns a tensorflow SAC-Agent
    
    Arguments:
        env {TFAPyEnvironment} -- Tensorflow-Agents PyEnvironment
        params {ParameterServer} -- ParameterServer from BARK
    
    Returns:
        agent -- tf-agent
    """
    # TODO(@hart): put in params
    actor_fc_layer_params = (512, 512, 128) # 4 layer net right now # changes 14 from (1024, 512, 512, 256)
    critic_joint_fc_layer_params = (256, 256, 128) # 2 layer critic net
    actor_learning_rate = 2e-4  # @param
    critic_learning_rate = 1e-3  # @param changes 14 from 3e-4
    alpha_learning_rate = 3e-4 # @param
    target_update_tau = 0.05 # @param # changed in changes 14 from 0.005
    target_update_period = 3 #@param changes in changes 14 from 1
    reward_scale_factor = 1.0 #@param
    gamma = 0.995 #@param
    gradient_clipping = None # @param
    agent_name = "SAC_agent"

    def _normal_projection_net(action_spec, init_means_output_factor=0.1):
      return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=sac_agent.std_clip_transform,
        scale_distribution=True)

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=actor_fc_layer_params,
        continuous_projection_net=_normal_projection_net)

    critic_net = critic_network.CriticNetwork(
      (env.observation_spec(), env.action_spec()),
      observation_fc_layer_params=None,
      action_fc_layer_params=None,
      joint_fc_layer_params=critic_joint_fc_layer_params)

    # SAC agent
    tf_agent = sac_agent.SacAgent(
      env.time_step_spec(),
      env.action_spec(),
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=actor_learning_rate),
      critic_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=critic_learning_rate),
      alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=alpha_learning_rate),
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      gradient_clipping=gradient_clipping,
      train_step_counter=self._ckpt.step,
      name=agent_name)
    tf_agent.initialize()
    return tf_agent

  def get_replay_buffer(self):
    # TODO(@hart): put in params
    replay_buffer_capacity = 256
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=self._agent.collect_data_spec,
      batch_size=self._env.batch_size,
      max_length=replay_buffer_capacity)

  def get_checkpointer(self, log_path="/"):
    # TODO(@hart): put in params
    checkpointer = Checkpointer(log_path,
      global_step=self._ckpt.step,
      tf_agent=self._agent,
      max_to_keep=2)
    checkpointer.initialize_or_restore()
    return checkpointer

  def get_dataset(self):
    # TODO(@hart): put in params
    batch_size = 256
    dataset = self._replay_buffer.as_dataset(
      num_parallel_calls=3,
      sample_batch_size=batch_size,
      num_steps=2).prefetch(3)
    return dataset

  def get_collect_policy(self):
    return self._agent.collect_policy

  def get_eval_policy(self):
    return greedy_policy.GreedyPolicy(self._agent.policy)

  def reset(self):
    pass

  @property
  def collect_policy(self):
    return self._collect_policy

  @property
  def eval_policy(self):
    return self._eval_policy

  def save(self):
    """Saves an agent
    """
    save_path = self._ckpt_manager.save()
    print("Saved checkpoint for step {}: {}".format(int(self._ckpt.step) + 1,
                                                    save_path))

  def load(self):
    """Loads an agent
    """
    try:
      self._ckpt.restore(self._ckpt_manager.latest_checkpoint)
    except:
      return RuntimeError("Could not load agent.")
    if self._ckpt_manager.latest_checkpoint:
      print("Restored agent from {}".format(
        self._ckpt_manager.latest_checkpoint))
    else:
      print("Initializing agent from scratch.")