from absl import app
from absl import flags
import tensorflow as tf
from tf_agents.environments import tf_py_environment

from modules.runtime.scenario.scenario_generation.uniform_vehicle_distribution \
  import UniformVehicleDistribution
from modules.runtime.scenario.scenario_generation.deterministic \
  import DeterministicScenarioGeneration
from modules.runtime.scenario.scenario_generation.deterministic_drone_challenge \
  import DeterministicDroneChallengeGeneration
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from modules.runtime.viewer.video_renderer import VideoRenderer


from src.rl_runtime import RuntimeRL
from src.observers.nearest_state_observer import ClosestAgentsObserver
from src.wrappers.dynamic_model import DynamicModel
from src.wrappers.tfa_wrapper import TFAWrapper
from src.evaluators.goal_reached import GoalReached
from src.agents.sac_agent import SACAgent
from src.runners.tfa_runner import TFARunner
from configurations.base_configuration import BaseConfiguration

# configuration specific evaluator
from configurations.sac_drone_challenge.custom_evaluator import CustomEvaluator
from configurations.sac_drone_challenge.custom_observer import CustomObserver

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode',
                  'visualize',
                  ['train', 'visualize', 'evaluate'],
                  'Mode the configuration should be executed in.')

class SACDroneChallenge(BaseConfiguration):
  """Hermetic and reproducible configuration class
  """
  def __init__(self,
               params):
    BaseConfiguration.__init__(
      self,
      params)

  def _build_configuration(self):
    """Builds a configuration using an SAC agent
    """
    self._scenario_generator = \
      DeterministicDroneChallengeGeneration(num_scenarios=3,
                                            random_seed=0,
                                            params=self._params)
    self._observer = CustomObserver(params=self._params)
    self._behavior_model = DynamicModel(model_name="TripleIntegratorModel",
                                        params=self._params)
    self._evaluator = CustomEvaluator(params=self._params)

    viewer = MPViewer(params=self._params,
                      x_range=[-40, 40],
                      y_range=[-40, 40],
                      follow_agent_id=True)
    self._viewer = viewer
    # self._viewer = VideoRenderer(renderer=viewer, world_step_time=0.2)
    self._runtime = RuntimeRL(action_wrapper=self._behavior_model,
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

def run_configuration(argv):
  params = ParameterServer(filename="configurations/sac_drone_challenge/config.json")
  configuration = SACDroneChallenge(params)
  
  if FLAGS.mode == 'train':
    configuration.train()
  elif FLAGS.mode == 'visualize':
    configuration.visualize(5)
    # configuration._viewer.export_video("/home/hart/Dokumente/2019/bark-ml/configurations/sac_drone_challenge/video/lane_merge")
  elif FLAGS.mode == 'evaluate':
    configuration.evaluate()

if __name__ == '__main__':
  app.run(run_configuration)