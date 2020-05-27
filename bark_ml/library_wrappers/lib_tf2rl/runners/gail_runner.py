import sys
import logging
import time
import tensorflow as tf
import numpy as np

import tf2rl
from tf2rl.experiments.irl_trainer import IRLTrainer
from tf2rl.experiments.utils import restore_latest_n_traj

tf.compat.v1.enable_v2_behavior()

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer

from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts

from src.runners.base_runner import BaseRunner

logger = logging.getLogger()
# NOTE(@hart): this will print all statements
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class GAILRunner(BaseRunner):
  """Runner that takes the runtime and agent
     and runs the training and evaluation as specified.
  """
  def __init__(self,
               runtime=None,
               agent=None,
               params=ParameterServer(),
               unwrapped_runtime=None):
    super().__init__(self,
                       runtime=runtime,
                       agent=agent,
                       params=params)
    
    self._unwrapped_runtime = unwrapped_runtime

  def train(self):
    """Instantiates an IRLtrainer instance from the tf2rl library
    and trains the GAIL agent with it.
    """
    if self._unwrapped_runtime is not None:
      trainer = self.get_trainer()

    trainer()

  
  def get_trainer(self):
    """Creates an IRLtrainer instance."""
    policy = self._agent._generator   # the agent's generator network, so in our case the DDPG agent
    irl = self._agent._discriminator  # the agent's discriminator network so in our case the GAIL network

    # creating args from the ParameterServer which can be given to the IRLtrainer:
    args = self.get_args_from_params()

    # getting the expert trajectories from the .pkl file:
    expert_trajs = restore_latest_n_traj(args.expert_path_dir,
                                         n_path=args.n_path, max_steps=args.max_steps)
    
    trainer=IRLtrainer(policy=policy,
                       env=self._unwrapped_runtime,
                       args=args,
                       irl=irl,
                       expert_obs=expert_trajs["obses"],
                       expert_next_obs=expert_trajs["next_obses"],
                       expert_act=expert_trajs["acts"])

    return trainer

  
  def evaluate(self):
    """Evaluates the agent
       Need to overwrite the class of the base function as the metric class somehow does
       not work.
    """
    ##################################################################
    #global_iteration = self._agent._agent._train_step_counter.numpy()
    ##################################################################
    logger.info("Evaluating the agent's performance in {} episodes."
      .format(str(self._params["ML"]["Runner"]["evaluation_steps"])))
    # Ticket (https://github.com/tensorflow/agents/issues/59) recommends
    # to do the rendering in the original environment
    rewards = []
    steps = []
    if self._unwrapped_runtime is not None:
      for _ in range(0, self._params["ML"]["Runner"]["evaluation_steps"]):
        obs = self._unwrapped_runtime.reset()
        is_terminal = False

        while not is_terminal:
          action = self._agent._generator.get_action(obs)
          obs, reward, is_terminal, _ = self._unwrapped_runtime.step(action)
          rewards.append(reward)
          steps.append(1)

    mean_reward = np.sum(np.array(rewards))/self._params["ML"]["Runner"]["evaluation_steps"]
    mean_steps = np.sum(np.array(steps))/self._params["ML"]["Runner"]["evaluation_steps"]
    tf.summary.scalar("mean_reward",
                      mean_reward)
    tf.summary.scalar("mean_steps",
                      mean_steps)

    #########################################
    #tf.summary.scalar("mean_reward",
    #                  mean_reward,
    #                  step=global_iteration)
    #tf.summary.scalar("mean_steps",
    #                  mean_steps,
    #                  step=global_iteration)
    #########################################
    logger.info(
      "The agent achieved on average {} reward and {} steps in \
      {} episodes." \
      .format(str(mean_reward),
              str(mean_steps),
              str(self._params["ML"]["Runner"]["evaluation_steps"])))


  def visualize(self):
    """Implements the visualization. See base class."""

    if self._unwrapped_runtime is not None:
        for _ in range(0, num_episodes):
          obs = self._unwrapped_runtime.reset()
          is_terminal = False
          while not is_terminal:
            print(obs)
            action = self._agent._generator.get_action(obs)
            # TODO(@hart); make generic for multi agent planning
            obs, reward, is_terminal, _ = self._unwrapped_runtime.step(action)
            print(reward)
            self._unwrapped_runtime.render()


  def get_args_from_params(self):
      """creates an args object from the ParameterServer object, that
      can be given to the IRLtrainer.
      """
      
      # experiment settings
      args.max_steps
      args.episode_max_steps 
      args.n_experiments
      args.show_progress
      args.save_model_interval
      args.save_summary_interval
      args.normalize_obs
      args.logdir
      args.model_dir
      # replay buffer
      args.use_prioritized_rb
      args.use_nstep_rb
      args.n_step
      # test settings
      args.test_interval
      args.show_test_progress
      args.test_episodes
      args.save_test_path
      args.save_test_movie
      args.show_test_images

      return args