from bark_ml.library_wrappers.lib_tf2rl.runners.tf2rl_runner import TF2RLRunner
from tf2rl.experiments.utils import restore_latest_n_traj
from tf2rl.experiments.irl_trainer import IRLTrainer
import tf2rl
import sys
import logging
import time
import argparse
from pathlib import Path
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()


class GAILRunner(TF2RLRunner):
  """GAIL runner implementation based on tf2rl library."""

  def __init__(self,
               environment=None,
               agent=None,
               params=None,
               expert_trajs=None):
    """initializing a TF2RL GAIL agent.
    Args:
      - environment: BARK runtime or open-ai gym type environment.
      - agent: TF2RL GAIL agent
      - params: BARK ParameterServer
      - expert_trajs: Expert trajectories.
    """

    self._expert_trajs = expert_trajs
    if not self._expert_trajs:
      import numpy as np
      self._expert_trajs = {
          'obses': np.empty([0, 0]),
          'next_obses': np.empty([0, 0]),
          'acts': np.empty([0, 0])}

    TF2RLRunner.__init__(self,
                         environment=environment,
                         agent=agent,
                         params=params)

  def _train(self):
    """Traines the GAIL agent with the IRLTrainer which was
    instantiated by self.GetTrainer.
    """
    self._trainer()

  def GetTrainer(self):
    """Creates an IRLtrainer instance."""
    policy = self._agent.generator   # the agent's generator network, so in our case the DDPG agent
    # the agent's discriminator network so in our case the GAIL network
    irl = self._agent.discriminator

    # creating args from the ParameterServer which can be given to the IRLtrainer:
    args = self._get_args_from_params()

    trainer = IRLTrainer(policy=policy,
                         env=self._environment,
                         args=args,
                         irl=irl,
                         expert_obs=self._expert_trajs["obses"],
                         expert_next_obs=self._expert_trajs["next_obses"],
                         expert_act=self._expert_trajs["acts"])

    return trainer

  def _get_args_from_params(self):
    """creates an args object from the ParameterServer object, that
    can be given to the IRLtrainer.
    Args:
      EXPERIMENT SETTINGS:
        - max_steps:              int, Maximum number steps to interact with env.
        - episode_max_steps:      int, Maximum steps in an episode
        - n_experiments:          int, Number of experiments
        - show_progress:          bool, Call `render` in training process
        - save_model_interval:    int, Interval to save model
        - save_summary_interval:  int, Interval to save summary
        - dir_suffix:             str, Suffix for directory that contains results
        - normalize_obs:          bool, Normalize observation
        - logdir:                 str, Output directory
        - logging_level           str, logging level, choices=['DEBUG', 'INFO', 'WARNING']
        - model_dir:              str, Directory to restore model
      REPLAY BUFFER:
        - expert_path_dir:        str, Directory that contains expert trajectories
        - use_propritized_rb:     bool, Flag to use prioritized experience replay
        - use_nstep_rb:           bool, Flag to use nstep experience replay
        - n_step:                 int, Number of steps to look over
      TEST SETTINGS:
        - evaluaate:              bool, evaluate trained agent.
        - test_interval:          int, Interval to evaluate trained model
        - show_test_progress:     bool, Call `render` in evaluation process
        - test_episodes:          int, Number of episodes to evaluate at once
        - save_test_path:         str, Save trajectories of evaluation
        - save_test_movie:        bool, Save rendering results
        - show_test_images:       bool, Show input images to neural networks when an episode finishes
      OTHER:
        - gpu:                    int, name of gpu device

    """
    tf2rl_params = self._params['ML']['GAILRunner']['tf2rl']
    args = {}

    # experiment settings
    args['max_steps'] = tf2rl_params['max_steps']
    args['episode_max_steps'] = tf2rl_params['episode_max_steps']
    args['n_experiments'] = tf2rl_params['n_experiments']
    args['show_progress'] = tf2rl_params['show_progress']
    args['save_model_interval'] = tf2rl_params['save_model_interval']
    args['save_summary_interval'] = tf2rl_params['save_summary_interval']
    args['dir_suffix'] = tf2rl_params['dir_suffix']
    args['normalize_obs'] = tf2rl_params['normalize_obs']
    args['logdir'] = tf2rl_params['logdir']
    args['logging_level'] = tf2rl_params['logging_level']
    args['model_dir'] = tf2rl_params['model_dir']

    # replay buffer
    args['use_prioritized_rb'] = tf2rl_params['use_prioritized_rb']
    args['use_nstep_rb'] = tf2rl_params['use_nstep_rb']
    args['n_step'] = tf2rl_params['n_step']

    # test settings
    args['evaluate'] = tf2rl_params['evaluate']
    args['test_interval'] = tf2rl_params['test_interval']
    args['show_test_progress'] = tf2rl_params['show_test_progress']
    args['test_episodes'] = tf2rl_params['test_episodes']
    args['save_test_path'] = tf2rl_params['save_test_path']
    args['save_test_movie'] = tf2rl_params['save_test_movie']
    args['show_test_images'] = tf2rl_params['show_test_images']

    # other:
    args['gpu'] = self._params["ML"]["Settings"]["GPUUse", "", 0]

    Path(args['model_dir']).mkdir(parents=True, exist_ok=True)
    Path(args['logdir']).mkdir(parents=True, exist_ok=True)

    return argparse.Namespace(**args)
