from bark_ml.environments.blueprints import *
from bark_ml.environments import *
from bark_ml.behaviors import *
from bark_ml.observers import *
from bark_ml.evaluators import *
from bark_ml.library_wrappers.lib_tf_agents.agents import *
from bark_ml.library_wrappers.lib_tf_agents.runners import *
from bark_ml.core.observers import *
from bark_ml.core.evaluators import *
from bark.runtime.commons.parameters import ParameterServer


def LoadModule(module_name, dict_items):
  """Helper function to load dictionaries.

  Args:
      module_name (string): Name of the module
      dict_items (dict): Configuration items for the module

  Returns:
      Object: Object that has bene requested
  """
  # HACK
  if module_name == "FrenetObserver":
    return FrenetObserver(dict_items['params'])
  if module_name == "StaticObserver":
    return StaticObserver(dict_items['params'])
  if module_name ==  "GeneralEvaluator":
    return GeneralEvaluator(dict_items['params'])
  return eval("{}(**dict_items)".format(module_name))


class Experiment:
  """The Experiment-class contains all entities in order to run,
  train, or evaluate an agent."""

  def __init__(self, json_file, params=None, mode=None):
    """
    Initialized the Experiment using a json file and a ParameterServer()

    Args:
        json_file (json): Configuration of the experiment
        params (ParameterServer, optional): Contains the parameters used.
                                            Defaults to None.
        mode (string, optional): Which mode should be ran. Defaults to None.
    """
    self._params = params
    self._mode = mode
    self.InitJson(json_file)

  def InitJson(self, json_file):
    if self._params is None:
      self._params = ParameterServer(filename=json_file)
    self._exp_params = self._params["Experiment"]
    self._blueprint = self.InitBlueprint()
    self._observer = self.InitObserver()
    self._evaluator = self.InitEvaluator()
    self._runtime = self.InitRuntime()
    self._agent = self.InitAgent()
    self._runner = self.InitRunner()

  def InitBlueprint(self):
    """
    Initialized the scenario blueprint.

    Returns:
        Blueprint: Contains a set of N-scenarios
    """
    module_name = self._exp_params["Blueprint"]["ModuleName"]
    items = self._exp_params["Blueprint"]["Config"].ConvertToDict()
    items["params"] = self._params
    if self._mode == "evaluate":
      items["num_scenarios"] = self._exp_params["NumEvaluationEpisodes"]
    if self._mode == "visualize":
      items["num_scenarios"] = self._exp_params["NumVisualizationEpisodes"]
    blueprint = LoadModule(module_name, items)
    # NOTE: this should be configurable also
    blueprint._ml_behavior = BehaviorContinuousML(params=self._params)
    return blueprint

  def InitObserver(self):
    """
    Creates the observer for the Experiment

    Returns:
        Observer: Converts the environment's state for RL
    """
    module_name = self._exp_params["Observer"]["ModuleName"]
    items = self._exp_params["Observer"]["Config"].ConvertToDict()
    items["params"] = self._params
    return LoadModule(module_name, items)

  def InitEvaluator(self):
    """
    Initialized the Evaluator.

    Returns:
        Evaluator: Computes the reward and whether an episode is terminal
    """
    module_name = self._exp_params["Evaluator"]["ModuleName"]
    items = self._exp_params["Evaluator"]["Config"].ConvertToDict()
    items["params"] = self._params
    return LoadModule(module_name, items)

  def InitAgent(self):
    """
    Initializes the RL-Agent.

    Returns:
        Agent: RL-Agent
    """
    module_name = self._exp_params["Agent"]["ModuleName"]
    items = self._exp_params["Agent"]["Config"].ConvertToDict()
    items["environment"] = self._runtime
    items["observer"] = self._observer
    items["params"] = self._params
    agent = LoadModule(module_name, items)
    self._runtime.ml_behavior = agent
    return agent

  def InitRuntime(self):
    """
    Initializes the Runtime.

    Returns:
        Runtime: Implements the basic OpenAI-Gym interface.
    """
    module_name = self._exp_params["Runtime"]["ModuleName"]
    items = self._exp_params["Runtime"]["Config"].ConvertToDict()
    items["evaluator"] = self._evaluator
    items["observer"] = self._observer
    items["blueprint"] = self._blueprint
    return LoadModule(module_name, items)

  def InitRunner(self):
    """
    The Runner runs the training, visualization, and evaluation.

    Returns:
        Runner: Collects episodes.
    """
    module_name = self._exp_params["Runner"]["ModuleName"]
    items = self._exp_params["Runner"]["Config"].ConvertToDict()
    items["environment"] = self._runtime
    items["params"] = self._params
    items["agent"] = self._agent
    return LoadModule(module_name, items)

  @property
  def agent(self):
    return self._agent

  @property
  def runtime(self):
    return self._runtime

  @property
  def runner(self):
    return self._runner

  @property
  def params(self):
    return self._params