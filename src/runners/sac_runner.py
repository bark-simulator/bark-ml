import sys
import logging
import time
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from modules.runtime.commons.parameters import ParameterServer

from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts

from src.runners.tfa_runner import TFARunner


logger = logging.getLogger()
# NOTE(@hart): this will print all statements
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class SACRunner(TFARunner):
  """Runner that takes the runtime and agent
     and runs the training and evaluation as specified.
  """
  def __init__(self,
               runtime=None,
               agent=None,
               params=ParameterServer(),
               unwrapped_runtime=None):
    TFARunner.__init__(self,
                       runtime=runtime,
                       agent=agent,
                       params=params,
                       unwrapped_runtime=unwrapped_runtime)

  def train(self):
    """Wrapper that sets the summary writer.
       This enables a seamingless integration with TensorBoard.
    """
    # collect initial episodes
    self.collect_initial_episodes()
    # main training cycle
    if self._summary_writer is not None:
      with self._summary_writer.as_default():
        self._train()
    else:
      self._train()

  def _train(self):
    """Trains the agent as specified in the parameter file
    """
    iterator = iter(self._agent._dataset)
    for _ in range(0, self._params["ML"]["Runner"]["number_of_collections"]):
      global_iteration = self._agent._agent._train_step_counter.numpy()
      self._collection_driver.run()
      experience, _ = next(iterator)
      self._agent._agent.train(experience)
      if global_iteration % self._params["ML"]["Runner"]["evaluate_every_n_steps"] == 0:
        self.evaluate()
        self._agent.save()

  def visualize(self, num_episodes=1):
    # Ticket (https://github.com/tensorflow/agents/issues/59) recommends
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