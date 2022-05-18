# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import logging
import tensorflow as tf
import numpy as np
tf.compat.v1.enable_v2_behavior()

# BARK imports
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.buffered_viewer import BufferedViewer
from bark.core.geometry import *
from bark.core.world.renderer import *

# tf agent imports
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.trajectories import time_step as ts

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf_agents.py_bark_environment import PyBARKEnvironment
from bark_ml.commons.tracer import Tracer

def get_index(episode_log, key, idx):
  return episode_log[idx][key]

def calculate_mean(episode_log, key):
  mean = 0.
  for log in episode_log:
    mean += log[key]
  return mean/len(episode_log)

def check_if_any(episode_log, key, val):
  for log in episode_log:
    if log[key] == val:
      return True
  return False

class TFARunner:
  """Used to train, evaluate and visualize a BARK-ML agent."""

  def __init__(self,
               environment=None,
               agent=None,
               tracer=None,
               params=None):
    self._params = params or ParameterServer()
    self._eval_metrics = [
      tf_metrics.AverageReturnMetric(
        buffer_size=self._params["ML"]["TFARunner"]["EvaluationSteps", "", 25]),
      tf_metrics.AverageEpisodeLengthMetric(
        buffer_size=self._params["ML"]["TFARunner"]["EvaluationSteps", "", 25])
    ]
    self._agent = agent
    self._agent.set_action_externally = True
    self._summary_writer = None
    self._environment = environment
    self._wrapped_env = tf_py_environment.TFPyEnvironment(
      PyBARKEnvironment(self._environment))
    self.GetInitialCollectionDriver()
    self.GetCollectionDriver()
    self._logger = logging.getLogger()
    self._tracer = tracer or Tracer()
    self._colliding_scenario_ids = []
    self._max_success_rate = -1.0
    self._max_reward = -10000.0

  def SetupSummaryWriter(self):
    if self._params["ML"]["TFARunner"]["SummaryPath"] is not None:
      try:
        self._summary_writer = tf.summary.create_file_writer(
          self._params["ML"]["TFARunner"]["SummaryPath"])
      except:
        pass
    self.GetInitialCollectionDriver()
    self.GetCollectionDriver()

  def GetInitialCollectionDriver(self):
    self._initial_collection_driver = \
      dynamic_episode_driver.DynamicEpisodeDriver(
        env=self._wrapped_env,
        policy=self._agent._agent.collect_policy,
        observers=[self._agent._replay_buffer.add_batch],
        num_episodes=self._params["ML"]["TFARunner"]["InitialCollectionEpisodes", "", 50])

  def GetCollectionDriver(self):
    self._collection_driver = dynamic_episode_driver.DynamicEpisodeDriver(
      env=self._wrapped_env,
      policy=self._agent._agent.collect_policy,
      observers=[self._agent._replay_buffer.add_batch],
      num_episodes=self._params["ML"]["TFARunner"]["CollectionEpisodesPerStep", "", 1])

  def CollectInitialEpisodes(self):
    self._initial_collection_driver.run()

  def Train(self):
    self.CollectInitialEpisodes()
    if self._summary_writer is not None:
      with self._summary_writer.as_default():
        self._train()
    else:
      self._train()

  def _train(self):
    """Agent specific."""
    pass

  def ReshapeActionIfRequired(self, action_step):
    action_shape = action_step.action.shape
    expected_shape = self._agent._eval_policy.action_spec.shape
    action = action_step.action.numpy()
    if action_shape != expected_shape:
      action = np.reshape(action, expected_shape)
    return action

  def RunEpisode(self, render=True):
    episode_log = []
    state = self._environment.reset()
    is_terminal = False
    if render:
      self._environment.render()
    info = {}
    reward = 0
    while not is_terminal:
      action_step = self._agent._eval_policy.action(
        ts.transition(state, reward=0.0, discount=1.0))
      action = self.ReshapeActionIfRequired(action_step)

      state, reward, is_terminal, info = self._environment.step(action)
      episode_log.append({
        "state": state, "action" : action, "reward": reward,
        "is_terminal": is_terminal, **info})
      if render:
        self._environment.render()
    episode_log.append({
        "state": state, "action" : None, "reward": reward,
        "is_terminal": is_terminal, **info})
    return episode_log

  def Run(
    self, num_episodes=10, render=False, mode="not_training", **kwargs):
    episode_logs = {}
    collision, success, steps, reward = 0, 0, 0., 0.
    for i in range(0, num_episodes):
      if render:
        self._logger.info(f"Simulating episode {i}.")

      episode_log = self.RunEpisode(render=render)
      if mode == "evaluate":
        episode_logs[i] = episode_log
      steps += get_index(episode_log, "step_count", -1)
      reward += calculate_mean(episode_log, "reward")
      collision += check_if_any(episode_log, "collision", True) or \
        check_if_any(episode_log, "drivable_area", True)
      # rethink success: goal_reached without collision and expiring drivable_area
      success += check_if_any(episode_log, "goal_reached", True) and \
        check_if_any(episode_log, "collision", False) and \
        check_if_any(episode_log, "drivable_area", False)

    mean_steps = steps / num_episodes
    mean_reward = reward / num_episodes
    col_rate = collision / num_episodes
    success_rate = success / num_episodes

    print(
      f"The agent achieved an average reward of {mean_reward:.3f}," +
      f" collision-rate of {col_rate:.5f}, took on average" +
      f" {mean_steps:.3f} steps, and reached a success-rate of " +
      f" {success_rate:.3f} (evaluated over {num_episodes} episodes).")

    if mode == "training":
      best_ckpt_folder=self._agent._best_ckpt_manager._manager._directory
      if success_rate > self._max_success_rate or \
        (success_rate == self._max_success_rate and mean_reward > self._max_reward):
        self._agent.SaveCheckpoint()
        with open(best_ckpt_folder + 'info.txt', 'w') as f:
          f.write(f"Success-rate {success_rate:.3f}, collision-rate: {col_rate:.5f}"
                  f", reward {mean_reward:.3f}, steps: {mean_steps:.3f}.")
        self._max_success_rate = success_rate
        self._max_reward = mean_reward

      global_iteration = self._agent._agent._train_step_counter.numpy()
      tf.summary.scalar("mean_reward", mean_reward, step=global_iteration)
      tf.summary.scalar("mean_steps", mean_steps, step=global_iteration)
      tf.summary.scalar("collision_rate", col_rate, step=global_iteration)
      tf.summary.scalar("goal_rate", success_rate, step=global_iteration)

      # TODO: specify what should be logged in tensorboard apart from the base values
      #   res = {}
      #   for state in self._tracer._states:
      #     for key, val in state.items():
      #       if key not in res:
      #         res[key] = 0.
      #       res[key] += val

      #   for key, val in res.items():
      #     if key not in ["state", "goal_reached", "step_count", "num_episode", "reward"]:
      #       tf.summary.scalar(f"auto_{key}", val, step=global_iteration)

    if mode == "evaluate":
      return episode_logs

