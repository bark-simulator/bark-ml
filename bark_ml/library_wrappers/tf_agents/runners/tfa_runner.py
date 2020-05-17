import sys
import logging
import time
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from bark_project.modules.runtime.commons.parameters import ParameterServer

from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts

from src.runners.base_runner import BaseRunner


class TFARunner:
  def __init__(self,
               runtime=None,
               agent=None,
               params=None):
    self._eval_metrics = [
      tf_metrics.AverageReturnMetric(
        buffer_size=self._params["ML"]["Runner"]["evaluation_steps"]),
      tf_metrics.AverageEpisodeLengthMetric(
        buffer_size=self._params["ML"]["Runner"]["evaluation_steps"])
    ]
    self._agent = agent
    self._summary_writer = None
    self._params = params or ParameterServer()
    self._unwrapped_runtime = unwrapped_runtime
    self.get_initial_collection_driver()
    self.get_collection_driver()
    self._logger = logging.getLogger()
    # TODO(@hart): eval. runtime
    # TODO(@hart): wrapped env.

  def setup_writer(self):
    if self._params["ML"]["Runner"]["summary_path"] is not None:
      try:
        self._summary_writer = tf.summary.create_file_writer(
          self._params["BaseDir"] + "/" + self._params["ML"]["Runner"]["summary_path"])
      except:
        pass
    self.get_initial_collection_driver()
    self.get_collection_driver()

  def GetInitialCollectionDriver(self):
    self._initial_collection_driver = \
      dynamic_episode_driver.DynamicEpisodeDriver(
        env=self._runtime,
        policy=self._agent._agent.collect_policy,
        observers=[self._agent._replay_buffer.add_batch],
        num_episodes=self._params["ML"]["Runner"]["initial_collection_steps"])

  def GetCollectionDriver(self):
    self._collection_driver = dynamic_episode_driver.DynamicEpisodeDriver(
      env=self._runtime,
      policy=self._agent._agent.collect_policy,
      observers=[self._agent._replay_buffer.add_batch],
      num_episodes=self._params["ML"]["Runner"]["collection_episodes_per_cycle"])

  def CollectInitialEpisodes(self):
    self._initial_collection_driver.run()

  def Train(self):
    self.collect_initial_episodes()
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
    global_iteration = self._agent._agent._train_step_counter.numpy()
    logger.info("Evaluating the agent's performance in {} episodes."
      .format(str(self._params["ML"]["Runner"]["evaluation_steps"])))
    metric_utils.eager_compute(
      self._eval_metrics,
      self._runtime,
      self._agent._agent.policy,
      num_episodes=self._params["ML"]["Runner"]["evaluation_steps"])
    metric_utils.log_metrics(self._eval_metrics)
    tf.summary.scalar("mean_reward",
                      self._eval_metrics[0].result().numpy(),
                      step=global_iteration)
    tf.summary.scalar("mean_steps",
                      self._eval_metrics[1].result().numpy(),
                      step=global_iteration)
    logger.info(
      "The agent achieved on average {} reward and {} steps in \
      {} episodes." \
      .format(str(self._eval_metrics[0].result().numpy()),
              str(self._eval_metrics[1].result().numpy()),
              str(self._params["ML"]["Runner"]["evaluation_steps"])))

  def Visualize(self, num_episodes=1):
    # to do the rendering in the original environment
    if self._unwrapped_runtime is not None:
      for _ in range(0, num_episodes):
        state = self._unwrapped_runtime.reset()
        is_terminal = False
        while not is_terminal:
          print(state)
          action_step = self._agent._eval_policy.action(ts.transition(state, reward=0.0, discount=1.0))
          print(action_step)
          # TODO(@hart); make generic for multi agent planning
          state, reward, is_terminal, _ = self._unwrapped_runtime.step(action_step.action.numpy())
          print(reward)
          self._unwrapped_runtime.render()