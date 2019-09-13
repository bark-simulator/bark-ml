from absl import app
from absl import flags
import tensorflow as tf
from tf_agents.environments import tf_py_environment

from modules.runtime.scenario.scenario_generation.uniform_vehicle_distribution \
  import UniformVehicleDistribution
from modules.runtime.scenario.scenario_generation.deterministic \
  import DeterministicScenarioGeneration
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer

from src.rl_runtime import RuntimeRL
from src.observers.nearest_state_observer import ClosestAgentsObserver
from src.wrappers.dynamic_model import DynamicModel
from src.wrappers.tfa_wrapper import TFAWrapper
from src.evaluators.goal_reached import GoalReached
from src.agents.sac_agent import SACAgent
from src.runners.tfa_runner import TFARunner
from configurations.base_experiment import BaseExperiment

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode',
                  'visualize',
                  ['train', 'visualize', 'evaluate'],
                  'Mode the experiment should be executed in.')

class Experiment(BaseExperiment):
  """Hermetic and reproducible experiment class
  """
  def __init__(self):
    BaseExperiment.__init__(self,
                            "configurations/01_sac_highway/config.json")

  def _build_experiment(self):
    """The components will be filled in the configurations
    """
    self._scenario_generator = \
      DeterministicScenarioGeneration(num_scenarios=3,
                                      random_seed=0,
                                      params=self._params)
    self._observer = ClosestAgentsObserver(params=self._params)
    self._agent_model = DynamicModel(params=self._params)
    self._evaluator = GoalReached(params=self._params)
    self._viewer = MPViewer(params=self._params,
                            x_range=[-30,30],
                            y_range=[-20,40],
                            follow_agent_id=True)
    self._runtime = RuntimeRL(action_wrapper=self._agent_model,
                              observer=self._observer,
                              evaluator=self._evaluator,
                              step_time=0.2,
                              viewer=self._viewer,
                              scenario_generator=self._scenario_generator)
    tfa_env = tf_py_environment.TFPyEnvironment(TFAWrapper(self._runtime))
    self._agent = SACAgent(tfa_env, params=self._params)
    self._runner = TFARunner(tfa_env,
                             self._agent,
                             params=self._params,
                             unwrapped_runtime=self._runtime)

def run_experiment(argv):
  experiment = Experiment()
  if FLAGS.mode == 'train':
    experiment.train()
  elif FLAGS.mode == 'visualize':
    experiment.visualize()
  elif FLAGS.mode == 'evaluate':
    experiment.evaluate()

if __name__ == '__main__':
  app.run(run_experiment)