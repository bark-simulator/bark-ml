import numpy as np
from bark.models.behavior import BehaviorModel, DynamicBehaviorModel
from bark.models.dynamic import SingleTrackModel
from modules.runtime.commons.parameters import ParameterServer
from configurations.base_configuration import BaseConfiguration
from bark.world import World, ObservedWorld

# include all configurations
from configurations.highway.configuration import HighwayConfiguration

class BARKMLBehaviorModel(BehaviorModel):
  """This class makes a trained agent available as BehaviorModel
     in BARK.
  """
  def __init__(self,
               configuration=None):
    BehaviorModel.__init__(self, configuration._params)
    self._configuration = configuration
    self._dynamic_behavior_model = DynamicBehaviorModel(
      self._configuration._behavior_model._dynamic_model,
      configuration._params)

  def Plan(self, delta_time, observed_world):
    self._configuration._observer.reset(observed_world, [0])
    observed_state = self._configuration._observer.observe(
      observed_world)
    action = self._configuration._agent.act(observed_state)
    self._dynamic_behavior_model.SetLastAction(action)
    trajectory = self._dynamic_behavior_model.Plan(delta_time, observed_world)
    super(BARKMLBehaviorModel, self).SetLastTrajectory(trajectory)
    return trajectory

  def Clone(self):
    return self

  # def __getstate__(self):
  #   try:
  #     del self.__dict__['_configuration']
  #   except:
  #     pass
  #   try:
  #     del self.__dict__['_dynamic_behavior_model']
  #   except:
  #     pass
  #   odict = self.__dict__.copy()
  #   return odict
  
  # def __setstate__(self, sdict):
  #   # HACK
  #   base_dir = "/home/hart/Dokumente/2020/bark-ml"
  #   params = ParameterServer(filename=base_dir + "/configurations/highway/config.json")
  #   scenario_generation = params["Scenario"]["Generation"]["ConfigurableScenarioGeneration"]
  #   map_filename = scenario_generation["MapFilename"]
  #   scenario_generation["MapFilename"] = base_dir + "/" + map_filename
  #   params["BaseDir"] = base_dir
  #   sdict['_configuration'] = HighwayConfiguration(params)
  #   sdict['_dynamic_behavior_model'] = DynamicBehaviorModel(
  #     sdict['_configuration']._behavior_model._dynamic_model,
  #     sdict['_configuration']._params)
  #   BehaviorModel.__init__(self, sdict['_configuration']._params)
  #   self.__dict__.update(sdict)