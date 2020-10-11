from bark.runtime.commons.parameters import ParameterServer
from bark_ml.environments.blueprints import *
from bark_ml.environments import *
from bark_ml.behaviors import *
from bark_ml.observers import *
from bark_ml.evaluators import *
from bark_ml.library_wrappers.lib_tf_agents.agents import *
from bark_ml.library_wrappers.lib_tf_agents.runners import *
from bark_ml.core.observers import *
from bark_ml.core.evaluators import *


def LoadModule(module_name, dict_items):
  return eval("{}(**dict_items)".format(module_name))

class Experiment:
  def __init__(self, json_file):
    self.InitJson(json_file)
  
  def InitJson(self, json_file):
    self._params = ParameterServer(filename=json_file)
    self._exp_params = self._params["Experiment"]
    self._blueprint = self.InitBlueprint()
    self._observer = self.InitObserver()
    self._evaluator = self.InitEvaluator()
    self._runtime = self.InitRuntime()
    self._agent = self.InitAgent()
    self._runner = self.InitRunner() 

  def InitBlueprint(self):
    module_name = self._exp_params["Blueprint"]["ModuleName"]
    items = self._exp_params["Blueprint"]["Config"].ConvertToDict()
    items["params"] = self._params
    blueprint = LoadModule(module_name, items)
    # NOTE: this should be configurable also
    blueprint._ml_behavior = BehaviorContinuousML(params=self._params)
    return blueprint
  
  def InitObserver(self):
    module_name = self._exp_params["Observer"]["ModuleName"]
    items = self._exp_params["Observer"]["Config"].ConvertToDict()
    items["params"] = self._params
    return LoadModule(module_name, items)
  
  def InitEvaluator(self):
    module_name = self._exp_params["Evaluator"]["ModuleName"]
    items = self._exp_params["Evaluator"]["Config"].ConvertToDict()
    items["params"] = self._params
    return LoadModule(module_name, items)
  
  def InitAgent(self):
    module_name = self._exp_params["Agent"]["ModuleName"]
    items = self._exp_params["Agent"]["Config"].ConvertToDict()
    items["environment"] = self._runtime
    items["observer"] = self._observer
    items["params"] = self._params
    agent = LoadModule(module_name, items)
    self._runtime.ml_behavior = agent
    return agent
  
  def InitRuntime(self):
    module_name = self._exp_params["Runtime"]["ModuleName"]
    items = self._exp_params["Runtime"]["Config"].ConvertToDict()
    items["evaluator"] = self._evaluator
    items["observer"] = self._observer
    items["blueprint"] = self._blueprint
    return LoadModule(module_name, items)
  
  def InitRunner(self):
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