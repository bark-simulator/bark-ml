import sys
import logging
import time
import tensorflow as tf
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
    self.get_initial_collection_driver()
    self.get_collection_driver()

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

  def Evaluate(self):
    self._agent._training = False
    global_iteration = self._agent._agent._train_step_counter.numpy()
    self._logger.info("Evaluating the agent's performance in {} episodes."
      .format(str(self._params["ML"]["TFARunner"]["EvaluationSteps", "", 20])))
    metric_utils.eager_compute(
      self._eval_metrics,
      self._wrapped_env,
      self._agent._agent.policy,
      num_episodes=self._params["ML"]["TFARunner"]["EvaluationSteps", "", 20])
    metric_utils.log_metrics(self._eval_metrics)
    tf.summary.scalar("mean_reward",
                      self._eval_metrics[0].result().numpy(),
                      step=global_iteration)
    tf.summary.scalar("mean_steps",
                      self._eval_metrics[1].result().numpy(),
                      step=global_iteration)
    self._logger.info(
      "The agent achieved on average {} reward and {} steps in \
      {} episodes." \
      .format(str(self._eval_metrics[0].result().numpy()),
              str(self._eval_metrics[1].result().numpy()),
              str(self._params["ML"]["TFARunner"]["EvaluationSteps", "", 20])))


  def Visualize(self, num_episodes=1):
    self._agent._training = False
    for _ in range(0, num_episodes):
      state = self._environment.reset()
      is_terminal = False
      while not is_terminal:
        action_step = self._agent._eval_policy.action(ts.transition(state, reward=0.0, discount=1.0))
        state, reward, is_terminal, _ = self._environment.step(action_step.action.numpy())
        self._environment.render()