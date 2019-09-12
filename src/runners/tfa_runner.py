import sys
import logging
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver

from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
from tf_agents.utils import common
from src.runners.base_runner import BaseRunner


logger = logging.getLogger()
# TODO(@hart): this will print all statements
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class TFARunner(BaseRunner):
  def __init__(self,
               runtime,
               agent,
               number_of_collections=10,
               initial_collection_steps=5,
               collection_steps_per_cycle=5,
               evaluate_every_n_steps=5,
               evaluation_steps=1,
               initial_collection_driver=None,
               collection_driver=None,
               summary_path=None):
    BaseRunner.__init__(self,
                        runtime=runtime,
                        agent=agent,
                        number_of_collections=number_of_collections,
                        initial_collection_steps=initial_collection_steps,
                        collection_steps_per_cycle=collection_steps_per_cycle)
    self._eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=evaluation_steps),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=evaluation_steps)
    ]
    self._summary_writer = None
    if summary_path is not None:
      self._summary_writer = tf.summary.create_file_writer(summary_path)
    self._evaluate_every_n_steps = evaluate_every_n_steps
    self._evaluation_steps = evaluation_steps
    self.get_initial_collection_driver()
    self.get_collection_driver()

    # collect initial episodes
    self.collect_initial_episodes()

  def get_initial_collection_driver(self):
    self._initial_collection_driver = dynamic_episode_driver.DynamicEpisodeDriver(
      env=self._runtime,
      policy=self._agent._agent.collect_policy,
      observers=[self._agent._replay_buffer.add_batch],
      num_episodes=self._initial_collection_steps)
    self._initial_collection_driver.run = common.function(
      self._initial_collection_driver.run)

  def get_collection_driver(self):
    self._collection_driver = dynamic_episode_driver.DynamicEpisodeDriver(
      env=self._runtime,
      policy=self._agent._agent.collect_policy,
      observers=[self._agent._replay_buffer.add_batch],
      num_episodes=self._collection_steps_per_cycle)
    self._collection_driver.run = common.function(self._collection_driver.run)

  def collect_initial_episodes(self):
    self._initial_collection_driver.run()

  def train(self):
    if self._summary_writer is not None:
      with self._summary_writer.as_default():
        self._train()
    else:
      self._train()

  def _train(self):
    iterator = iter(self._agent._dataset)
    for i in range(0, self._number_of_collections):
      logger.info("Iteration: {}".format(str(self._agent._ckpt.step.numpy())))
      self._collection_driver.run()
      experience, _ = next(iterator)
      self._agent._agent.train(experience)
      if i % self._evaluate_every_n_steps == 0:
        self.evaluate()

  def evaluate(self):
    logger.info("Evaluating the agent's performance in {} episodes."
      .format(str(self._evaluation_steps)))
    metric_utils.eager_compute(
      self._eval_metrics,
      self._runtime,
      self._agent._agent.policy,
      num_episodes=self._evaluation_steps)
    metric_utils.log_metrics(self._eval_metrics)
    logger.info(
      "The agent achieved on average {} reward and {} steps in \
      {} episodes." \
      .format(str(self._eval_metrics[0].result().numpy()),
              str(self._eval_metrics[1].result().numpy()),
              str(self._evaluation_steps)))
