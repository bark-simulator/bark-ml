import matplotlib as mpl
mpl.use('Agg')

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
from modules.runtime.viewer.video_renderer import VideoRenderer


from src.rl_runtime import RuntimeRL
from src.observers.nearest_state_observer import ClosestAgentsObserver
from src.wrappers.dynamic_model import DynamicModel
from src.wrappers.tfa_wrapper import TFAWrapper
from src.evaluators.goal_reached import GoalReached
from src.agents.sac_agent import SACAgent
from src.runners.sac_runner import SACRunner
from configurations.base_configuration import BaseConfiguration

# configuration specific evaluator
from configurations.sac_highway.custom_evaluator import CustomEvaluator
from configurations.sac_highway.custom_observer import CustomObserver
from configurations.sac_highway.configuration_lib import SACHighwayConfiguration

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode',
                  'train',
                  ['train', 'visualize', 'evaluate'],
                  'Mode the configuration should be executed in.')

def run_configuration(argv):
  params = ParameterServer(filename="/mnt/glusterdata/home/hart/experiment_1/configuration.runfiles/bark_ml/configurations/sac_highway/config.json")
  configuration = SACHighwayConfiguration(params)
  
  if FLAGS.mode == 'train':
    configuration.train()
  elif FLAGS.mode == 'visualize':
    configuration.visualize(10)
    # configuration._viewer.export_video("/home/hart/Dokumente/2019/bark-ml/configurations/sac_highway/video/lane_merge")
  elif FLAGS.mode == 'evaluate':
    configuration.evaluate()

if __name__ == '__main__':
  app.run(run_configuration)