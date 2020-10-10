from bark.runtime.commons.parameters import ParameterServer
from bark_ml.environments.blueprints import *


class ExperimentTemplate:
  def __init__(self, json_file):
      self.GetJson(json_file)
  
  def GetJson(self, json_file):
    self._params = ParameterServer(filename=json_file)
    self._blueprint = self.GetBlueprint()
    self._observer = self.GetObserver()
    self._evaluator = self.GetEvaluator()
    self._runtime = self.GetRuntime()
    self._agent = self.GetAgent()
    self._runner = self.GetRunner() 

  def GetBlueprint(self):
    pass
  
  def GetObserver(self):
    pass
  
  def GetEvaluator(self):
    pass
  
  def GetAgent(self):
    pass
  
  def GetRuntime(self):
    pass
  
  def GetRunner(self):
    pass