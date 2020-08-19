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


class TFARunner:
  def __init__(self,
               environment=None,
               agent=None,
               params=None):

    self._params = params
    self._eval_metrics = [
      tf_metrics.AverageReturnMetric(
        buffer_size=self._params["ML"]["TFARunner"]["EvaluationSteps", "", 25]),
      tf_metrics.AverageEpisodeLengthMetric(
        buffer_size=self._params["ML"]["TFARunner"]["EvaluationSteps", "", 25])
    ]
    self._agent = agent
    self._agent.set_action_externally = True
    self._summary_writer = None
    self._params = params or ParameterServer()
    self._environment = environment
    self._wrapped_env = tf_py_environment.TFPyEnvironment(
      TFAWrapper(self._environment))
    self.GetInitialCollectionDriver()
    self.GetCollectionDriver()
    self._logger = logging.getLogger()

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
      logging.warning("Action shape" + str(action_shape) + \
        " does not match with expected shape " + str(expected_shape) +\
        " -> reshaping is tried")
      action = np.reshape(action, expected_shape)
      logging.info(action)
    return action

  def RunEpisode(self, visualize=True):
    state = self._environment.reset()
    is_terminal = False
    trajectory = []
    while not is_terminal:
      action_step = self._agent._eval_policy.action(
        ts.transition(state, reward=0.0, discount=1.0))
      action = self.ReshapeActionIfRequired(action_step)
      observations = self._environment.step(action)
      state, is_terminal = observations[0], observations[2]
      trajectory.append([observations])
      if visualize:
        self._environment.render()
      return trajectory

  def Run(self, num_episodes=10, visualize=True, **kwargs):
    for i in range(0, num_episodes):
      trajectory = self.RunEpisode(visualize=visualize)
      # NOTE: log stuff 
      # tracer.Trace(trajectory, num_episode=i) -> extract and log
    # NOTE: Evaluate some stuff

  # NOTE: combine Evaluate(..) with Visualize(..) in Run(..)
  # Run(episode_number, visualize)
  def Evaluate(self):
    global_iteration = self._agent._agent._train_step_counter.numpy()
    self._logger.info("Evaluating the agent's performance in {} episodes."
      .format(str(self._params["ML"]["TFARunner"]["EvaluationSteps", "", 20])))
    metric_utils.eager_compute(
      self._eval_metrics,
      self._wrapped_env,
      self._agent._agent.policy,
      num_episodes=self._params["ML"]["TFARunner"]["EvaluationSteps", "", 20],
      use_function=False)
    metric_utils.log_metrics(self._eval_metrics)
    tf.summary.scalar("mean_reward",
                      self._eval_metrics[0].result().numpy(),
                      step=global_iteration)
    tf.summary.scalar("mean_steps",
                      self._eval_metrics[1].result().numpy(),
                      step=global_iteration)
    
    self._logger.info(
      "The agent achieved on average {} reward and {} steps in {} episodes." \
      .format(str(self._eval_metrics[0].result().numpy()),
              str(self._eval_metrics[1].result().numpy()),
              str(self._params["ML"]["TFARunner"]["EvaluationSteps", "", 20])))


  def Visualize(self, num_episodes=1):
    for _ in range(0, num_episodes):
      state = self._environment.reset()
      is_terminal = False
      while not is_terminal:
        action_step = self._agent._eval_policy.action(ts.transition(state, reward=0.0, discount=1.0))
        action_shape = action_step.action.shape
        expected_shape = self._agent._eval_policy.action_spec.shape
        action = action_step.action.numpy()
        if action_shape != expected_shape:
          logging.warning("Action shape" + str(action_shape) + \
            " does not match with expected shape " + str(expected_shape) +\
            " -> reshaping is tried")
          action = np.reshape(action, expected_shape)
          logging.info(action)
        state, reward, is_terminal, _ = self._environment.step(action)
        self._environment.render()
