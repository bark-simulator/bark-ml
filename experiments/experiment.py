

class Experiment:
  def __init__(self, json_file):
      self.LoadJson(json_file)
  
  def LoadJson(self, json_file):
    self._params = ParameterServer(filename=json_file)
    # experiment_params = params["Experiment"]
    self._blueprint = self.LoadBlueprint()
    self._observer = self.LoadObserver()
    self._evaluator = self.LoadEvaluator()
    self._runtime = self.LoadRuntime()
    self._agent = self.LoadAgent()
    self._runner = self.LoadRunner() 

  def LoadBlueprint(self):
    # why not register these, create json and dump
    exp_param = self._params["Experiment"]
    exp_param["name"]
    pass
  
  def LoadObserver(self):
    # why not register these, create json and dump
    pass
  
  def LoadEvaluator(self):
    # why not register these, create json and dump
    pass
  
  def LoadAgent(self):
    # why not register these, create json and dump
    pass
  
  def LoadRuntime(self):
    return SingleAgentRuntime(
      blueprint=self._blueprint,
      observer=self._observer,
      observer=self._evaluator,
      render=False)

  def LoadRunner(self):
    # why not register these, create json and dump
    pass