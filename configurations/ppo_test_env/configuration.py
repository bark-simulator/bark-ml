import coloredlogs, logging
from absl import app
from absl import flags
import tensorflow as tf
import gym

from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from modules.runtime.commons.parameters import ParameterServer


from src.wrappers.tfa_wrapper import TFAWrapper
from src.test_env import TestEnvironment
from src.agents.ppo_agent import PPOAgent
from src.runners.ppo_runner import PPORunner
from configurations.base_configuration import BaseConfiguration

# configuration specific evaluator
coloredlogs.install()
logger = logging.getLogger()

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode',
                  'visualize',
                  ['train', 'visualize', 'evaluate'],
                  'Mode the configuration should be executed in.')

class PPOTestEnv(BaseConfiguration):
  """Hermetic and reproducible configuration class
  """
  def __init__(self,
               params):
    BaseConfiguration.__init__(
      self,
      params)

  def _build_configuration(self):
    """Builds a configuration using an PPO agent
    """
    # self._runtime = RuntimeRL(action_wrapper=self._behavior_model,
    #                           observer=self._observer,
    #                           evaluator=self._evaluator,
    #                           step_time=0.2,
    #                           viewer=self._viewer,
    #                           scenario_generator=self._scenario_generator)
    self._runtime = TestEnvironment(4)

    # tfa_env = tf_py_environment.TFPyEnvironment(TFAWrapper(self._runtime))
    tfa_env = tf_py_environment.TFPyEnvironment(
      parallel_py_environment.ParallelPyEnvironment(
        [lambda: TFAWrapper(self._runtime)] * self._params["ML"]["Agent"]["num_parallel_environments"]))
    self._agent = PPOAgent(tfa_env, params=self._params, use_rnns=False)
    self._runner = PPORunner(tfa_env,
                             self._agent,
                             params=self._params,
                             unwrapped_runtime=self._runtime)

def run_configuration(argv):
  params = ParameterServer(filename="configurations/ppo_test_env/config.json")
  configuration = PPOTestEnv(params)
  
  if FLAGS.mode == 'train':
    configuration.train()
  elif FLAGS.mode == 'visualize':
    params["ML"]["Agent"]["num_parallel_environments"] = 1
    logger.setLevel("INFO")
    configuration.visualize(5)
  elif FLAGS.mode == 'evaluate':
    configuration.evaluate()

if __name__ == '__main__':
  app.run(run_configuration)