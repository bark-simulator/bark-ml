from modules.runtime.commons.parameters import ParameterServer

class BaseExperiment:
  """Hermetic and reproducible experiment class
  """
  def __init__(self,
               config_file=None):
    self._params = ParameterServer(filename=config_file)
    self._scenario_generator = None
    self._observer = None
    self._agent_model = None
    self._evaluator = None
    self._viewer = None
    self._runtime = None
    self._agent = None
    self._runner = None

    # build experiment
    self._build_experiment()

  def _build_experiment(self):
    """The components will be filled in the configurations
    """
    pass

  def train(self):
    """Trains the agent
    """
    self._runner.train()

  def visualize(self):
    """Visualizes the agent
    """
    self._runner.visualize()

  def evaluate(self):
    """Evaluates the agent
    """
    self._runner.evaluate()
