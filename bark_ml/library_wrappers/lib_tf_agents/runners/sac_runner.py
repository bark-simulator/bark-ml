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
      self._params["ML"]["SACRunner"]["NumberOfCollections", "", 2000]
    self._evaluation_interval =\
      self._params["ML"]["SACRunner"]["EvaluateEveryNSteps", "", 100] 

  def _train(self):
    iterator = iter(self._agent._dataset)
    t = time.time()

    for i in range(0, self._number_of_collections):
      self._agent._training = True
      
      global_iteration = self._agent._agent._train_step_counter.numpy()
      self._collection_driver.run()
      experience, _ = next(iterator)
      self._agent._agent.train(experience)
      
      if global_iteration % self._evaluation_interval == 0:
        iterations = f'{global_iteration-self._evaluation_interval}-{global_iteration}'

        iteration_duration = (time.time() - t) / self._evaluation_interval
        self._logger.info(f'Training iterations {iterations} took {iteration_duration:.3f} seconds \
          (avg. {iteration_duration:.3f}s / iteration).')
        t = time.time()

        self.Evaluate()
        self._agent.Save()

