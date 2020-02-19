import os
import matplotlib as mpl
if os.environ.get('DISPLAY') == ':0':
  print('No display found. Using non-interactive Agg backend')
  mpl.use('Agg')

from absl import app
from absl import flags
import tensorflow as tf
from tf_agents.environments import tf_py_environment

from modules.runtime.commons.parameters import ParameterServer
from configurations.base_configuration import BaseConfiguration

# configuration specific evaluator
from configurations.highway.custom_evaluator import CustomEvaluator
from configurations.highway.configuration_lib import HighwayConfiguration

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode',
                  'train',
                  ['train', 'visualize', 'evaluate'],
                  'Mode the configuration should be executed in.')
flags.DEFINE_string('base_dir',
                    os.path.dirname(
                      os.path.dirname(os.path.dirname(__file__))),
                    'Base directory of bark-ml.')


def run_configuration(argv):
  params = ParameterServer(filename=FLAGS.base_dir + "/configurations/highway/config.json")
  scenario_generation = params["Scenario"]["Generation"]["ConfigurableScenarioGeneration"]
  # params = ParameterServer(filename=FLAGS.base_dir + "/configurations/highway/config_easy.json")
  # scenario_generation = params["Scenario"]["Generation"]["DeterministicScenarioGeneration"]
  map_filename = scenario_generation["MapFilename"]
  scenario_generation["MapFilename"] = FLAGS.base_dir + "/" + map_filename
  params["BaseDir"] = FLAGS.base_dir
  configuration = HighwayConfiguration(params)
  
  if FLAGS.mode == 'train':
    configuration.train()
  elif FLAGS.mode == 'visualize':
    configuration.visualize(10)
    # configuration._viewer.export_video("/home/hart/Dokumente/2019/bark-ml/configurations/highway/video/lane_merge")
  elif FLAGS.mode == 'evaluate':
    configuration.evaluate()

if __name__ == '__main__':
  app.run(run_configuration)