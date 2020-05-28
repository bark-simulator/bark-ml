import sys
import logging
import time
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

# BARK imports
#from bark_project.modules.runtime.commons.parameters import ParameterServer

# tf2rl imports
import tf2rl
from tf2rl.experiments.irl_trainer import IRLTrainer
from tf2rl.experiments.utils import restore_latest_n_traj

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.runners.tf2rl_runner import TF2RLRunner

class GAILRunner(TF2RLRunner):
  """GAIL runner implementation based on tf2rl library."""

  def __init__(self,
              environment=None,
              agent=None,
              params=None):
      
    super().__init__(self,
                    environment=environment,
                    agent=agent,
                    params=params)
    

  def _train(self):
    """Traines the GAIL agent with the IRLtrainer which was
    instantiated by self.GetTrainer.
    """
    self._trainer()

  
  def GetTrainer(self):
    """Creates an IRLtrainer instance."""
    policy = self._agent._generator   # the agent's generator network, so in our case the DDPG agent
    irl = self._agent._discriminator  # the agent's discriminator network so in our case the GAIL network

    # creating args from the ParameterServer which can be given to the IRLtrainer:
    args = self._get_args_from_params()

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


  def get_args_from_params(self):
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
        - normalize_obs:          bool, Normalize observation
        - logdir:                 str, Output directory
        - model_dir:              str, Directory to restore model
      REPLAY BUFFER:
        - expert_path_dir:        str, Directory that contains expert trajectories
        - use_propritized_rb:     bool, Flag to use prioritized experience replay
        - use_nstep_rb:           bool, Flag to use nstep experience replay
        - n_step:                 int, Number of steps to look over
      TEST SETTINGS:
        - test_interval:          int, Interval to evaluate trained model
        - show_test_progress:     bool, Call `render` in evaluation process
        - test_episodes:          int, Number of episodes to evaluate at once
        - save_test_path:         str, Save trajectories of evaluation
        - save_test_movie:        bool, Save rendering results
        - show_test_images:       bool, Show input images to neural networks when an episode finishes

    """
    
    # experiment settings
    args.max_steps = params['tf2rl']['runner']['max_steps']
    args.episode_max_steps = params['tf2rl']['runner']['episode_max_steps']
    args.n_experiments = params['tf2rl']['runner']['n_experiments']
    args.show_progress = params['tf2rl']['runner']['show_progress']
    args.save_model_interval = params['tf2rl']['runner']['save_model_interval]
    args.save_summary_interval = params['tf2rl']['runner']['save_summary_interval']
    args.normalize_obs = params['tf2rl']['runner']['normalize_obs']
    args.logdir = params['tf2rl']['runner']['logdir']
    args.model_dir = params['tf2rl']['runner']['model_dir']
    # replay buffer
    args.expert_path_dir = params['tf2rl']['runner']['expert_path_dir']
    args.use_prioritized_rb = params['tf2rl']['runner']['use_prioritized_rb']
    args.use_nstep_rb = params['tf2rl']['runner']['use_nstep_rb']
    args.n_step = params['tf2rl']['runner']['n_step']
    # test settings
    args.test_interval = params['tf2rl']['runner']['test_interval']
    args.show_test_progress = params['tf2rl']['runner']['show_test_progress']
    args.test_episodes = params['tf2rl']['runner']['test_episodes']
    args.save_test_path = params['tf2rl']['runner']['save_test_path']
    args.save_test_movie = params['tf2rl']['runner']['save_test_movie']
    args.show_test_images = params['tf2rl']['runner']['show_test_images']

    return args