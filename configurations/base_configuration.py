from modules.runtime.commons.parameters import ParameterServer

class BaseConfiguration:
  """Hermetic and reproducible configurationion
  """
  def __init__(self,
               params=None):
    self._params = params
    self._scenario_generator = None
    self._observer = None
    self._behavior_model = None
    self._evaluator = None
    self._viewer = None
    self._runtime = None
    self._agent = None
    self._runner = None

    # build configurationion
    self._build_configuration()

  def _build_configuration(self):
    """Specifies the configuration
    """
    pass

  def train(self):
    """Trains the agent
    """
    self._runner.train()

  def visualize(self, num_episodes=1):
    """Visualizes the agent
    """
    self._runner.visualize(num_episodes)

  def evaluate(self):
    """Evaluates the agent
    """
    self._runner.evaluate()
