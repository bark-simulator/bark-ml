# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import sys
import logging
import time
import tensorflow as tf
import numpy as np
tf.compat.v1.enable_v2_behavior()

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

# tf agent imports
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf_agents.tfa_wrapper import TFAWrapper
from bark_ml.commons.tracer import Tracer


class TFARunner:
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
      TFAWrapper(self._environment))
    self.GetInitialCollectionDriver()
    self.GetCollectionDriver()
    self._logger = logging.getLogger()
    self._tracer = tracer or Tracer()

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
    """Agent specific
    """
    pass

  def ReshapeActionIfRequired(self, action_step):
    action_shape = action_step.action.shape
    expected_shape = self._agent._eval_policy.action_spec.shape
    action = action_step.action.numpy()
    if action_shape != expected_shape:
      # logging.warning("Action shape" + str(action_shape) + \
      #   " does not match with expected shape " + str(expected_shape) +\
      #   " -> reshaping is tried")
      action = np.reshape(action, expected_shape)
      # logging.info(action)
    return action

  def RunEpisode(self, render=True, **kwargs):
    state = self._environment.reset()
    is_terminal = False
    while not is_terminal:
      action_step = self._agent._eval_policy.action(
        ts.transition(state, reward=0.0, discount=1.0))
      action = self.ReshapeActionIfRequired(action_step)
      env_data = self._environment.step(action)
      self._tracer.Trace(env_data, **kwargs)
      state, is_terminal = env_data[0], env_data[2]
      if render:
        self._environment.render()

  def Run(self, num_episodes=10, render=False, mode="not_training", **kwargs):
    for i in range(0, num_episodes):
      trajectory = self.RunEpisode(
        render=render, **kwargs, num_episode=i)
    # average collision, reward, and step count
    mean_col_rate = self._tracer.Query(
      key="collision", group_by="num_episode", agg_type="ANY_TRUE").mean()
    mean_col_rate += self._tracer.Query(
      key="drivable_area", group_by="num_episode", agg_type="ANY_TRUE").mean()
    goal_reached = self._tracer.Query(
      key="goal_reached", group_by="num_episode", agg_type="ANY_TRUE").mean()
    mean_reward = self._tracer.Query(
      key="reward", group_by="num_episode", agg_type="SUM").mean()
    mean_steps = self._tracer.Query(
      key="step_count", group_by="num_episode", agg_type="LAST_VALUE").mean()
    if mode == "training":
      global_iteration = self._agent._agent._train_step_counter.numpy()
      tf.summary.scalar("mean_reward", mean_reward, step=global_iteration)
      tf.summary.scalar("mean_steps", mean_steps, step=global_iteration)
      tf.summary.scalar(
        "mean_collision_rate", mean_col_rate, step=global_iteration)
    self._logger.info(
      f"The agent achieved an average reward of {mean_reward:.3f}," +
      f" collision-rate of {mean_col_rate:.5f}, took on average" +
      f" {mean_steps:.3f} steps, and reached the goal " + 
      f" {goal_reached:.3f} (evaluated over {num_episodes} episodes).")
    # reset tracer
    self._tracer.Reset()