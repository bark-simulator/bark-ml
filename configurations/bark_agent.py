import numpy as np
from bark.models.behavior import BehaviorModel, DynamicBehaviorModel
from bark.models.dynamic import SingleTrackModel
from modules.runtime.commons.parameters import ParameterServer
from configurations.base_configuration import BaseConfiguration
from bark.world import World, ObservedWorld

# include all configurations
from configurations.sac_highway.configuration import SACHighwayConfiguration

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
    observed_state = self._configuration._observer.observe(
      world=observed_world,
      agents_to_observe=observed_world.ego_agent.id)
    action = self._configuration._agent.act(observed_state)
    self._dynamic_behavior_model.SetLastAction(action)
    trajectory = self._dynamic_behavior_model.Plan(delta_time, observed_world)
    super(BARKMLBehaviorModel, self).SetLastTrajectory(trajectory)
    return trajectory

  def Clone(self):
    return self
