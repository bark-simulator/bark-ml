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
from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf_agents.runners.tfa_runner import TFARunner


class SACRunner(TFARunner):
  def __init__(self,
               environment=None,
               agent=None,
               params=None):
    TFARunner.__init__(self,
                       environment=environment,
                       agent=agent,
                       params=params)
    
    self._number_of_collections =\
      self._params["ML"]["SACRunner"]["NumberOfCollections", "", 40000]
    self._evaluation_interval =\
      self._params["ML"]["SACRunner"]["EvaluateEveryNSteps", "", 100] 

  def _train(self):
    iterator = iter(self._agent._dataset)
    iteration_start_time = time.time()

    for i in range(1, self._number_of_collections):
      global_iteration = self._agent._agent._train_step_counter.numpy()
      tf.summary.experimental.set_step(global_iteration)

      t0 = time.time()
      self._collection_driver.run()
      self._log_collection_duration(start_time=t0, iteration=global_iteration)
 
      experience, _ = next(iterator)

      t0 = time.time()      
      self._agent._agent.train(experience)
      self._log_training_duration(start_time=t0, iteration=global_iteration)
            
      if global_iteration % self._evaluation_interval == 0:
        self._log_evaluation_interval_duration(
          start_time=iteration_start_time, 
          iteration=global_iteration)
        iteration_start_time = time.time()

        self.Run(
          num_episodes=self._params["ML"]["TFARunner"]["EvaluationSteps", "", 20],
          mode="training")
        self._agent.Save()

  def _log_collection_duration(self, start_time, iteration):
    with tf.name_scope("Durations"):
      tf.summary.scalar(
        "episode_collection_duration", time.time() - start_time, iteration)

  def _log_training_duration(self, start_time, iteration):
    with tf.name_scope("Durations"):
      tf.summary.scalar("training_turation", time.time() - start_time, iteration)

  def _log_evaluation_interval_duration(self, start_time, iteration):
    if iteration == 0: 
      return

    iterations = f'{iteration-self._evaluation_interval}-{iteration}'

    total_duration = time.time() - start_time
    mean_step_duration = total_duration / self._evaluation_interval
    self._logger.info(
      f'Training iterations {iterations} took {total_duration:.3f} seconds' +
      f' (avg. {mean_step_duration:.3f}s / iteration).')

    with tf.name_scope("Durations"):
      tf.summary.scalar("mean_step_duration",
                        mean_step_duration,
                        step=iteration)