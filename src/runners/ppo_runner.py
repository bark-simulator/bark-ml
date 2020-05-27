import sys
import logging
import time
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.policies import actor_policy
from modules.runtime.commons.parameters import ParameterServer

from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts

from src.runners.tfa_runner import TFARunner


logger = logging.getLogger()
# NOTE(@hart): this will print all statements
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class PPORunner(TFARunner):
  """Runner that takes the runtime and agent
     and runs the training and evaluation as specified.
  """
  def __init__(self,
               runtime=None,
               eval_runtime=None,
               agent=None,
               params=ParameterServer(),
               unwrapped_runtime=None):
    TFARunner.__init__(self,
                       runtime=runtime,
                       agent=agent,
                       params=params,
                       unwrapped_runtime=unwrapped_runtime)
    self._eval_runtime = eval_runtime

  def _train(self):
    """Trains the agent as specified in the parameter file
    """
    # iterator = iter(self._agent._dataset)
    for i in range(0, self._params["ML"]["Runner"]["number_of_collections"]):
      global_iteration = self._agent._agent._train_step_counter.numpy()
      self._collection_driver.run()
      trajectories = self._agent._replay_buffer.gather_all()
      self._agent._agent.train(experience=trajectories)
      self._agent._replay_buffer.clear()
      if i % self._params["ML"]["Runner"]["evaluate_every_n_steps"] == 0:
        self.evaluate()
        self._agent.save()

  def evaluate(self, num=None):
    """Evaluates the agent
    """
    if num is None:
      num = self._params["ML"]["Runner"]["evaluation_steps"]
    global_iteration = self._agent._agent._train_step_counter.numpy()
    logger.info("Evaluating the agent's performance in {} episodes."
      .format(str(num)))
    metric_utils.eager_compute(
      self._eval_metrics,
      self._eval_runtime,
      self._agent._agent.policy,
      num_episodes=num)
    metric_utils.log_metrics(self._eval_metrics)
    tf.summary.scalar("mean_reward",
                      self._eval_metrics[0].result().numpy(),
                      step=global_iteration)
    tf.summary.scalar("mean_steps",
                      self._eval_metrics[1].result().numpy(),
                      step=global_iteration)
    logger.error(
      "The agent achieved on average {} reward and {} steps in \
      {} episodes." \
      .format(str(self._eval_metrics[0].result().numpy()),
              str(self._eval_metrics[1].result().numpy()),
              str(self._params["ML"]["Runner"]["evaluation_steps"])))
  
  def visualize(self, num_episodes=1):
    # Ticket (https://github.com/tensorflow/agents/issues/59) recommends
    # to do the rendering in the original environment
    if self._unwrapped_runtime is not None:
      for _ in range(0, num_episodes):
        state = self._unwrapped_runtime.reset()
        is_terminal = False
        # time_step_spec = ts.time_step_spec(self._runtime.observation_spec)
        # initial_state = actor_policy.ActorPolicy(
        #   time_step_spec,
        #   self._runtime.action_spec,
        #   self._agent._agent._actor_net).get_initial_state(1)

        while not is_terminal:
          time_step = ts.transition(state, reward=0.0, discount=1.0)
          action_step = self._agent._eval_policy.action(time_step)
          print("action: ", action_step.action.numpy())
          # TODO(@hart); make generic for multi agent planning
          state, reward, is_terminal, _ = self._unwrapped_runtime.step(action_step.action.numpy())
          print("state: ", state, "reward: ", reward, "is_terminal", is_terminal)
          # print("reward: ", reward, "is_terminal", is_terminal)
          self._unwrapped_runtime.render()