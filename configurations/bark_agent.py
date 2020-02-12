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
     in BARK. There will be no evaluation from the side of the 
     agent. Thus, a BARKAgent will only use the observer and agent
     from the configuration.
  """
  def __init__(self,
               configuration=None,
               dynamic_model = None,
               agents_to_observe=None):
    BehaviorModel.__init__(self, configuration._params)
    self._configuration = configuration
    self._dynamic_behavior_model = DynamicBehaviorModel(
      dynamic_model, configuration._params)
    self._agents_to_observe = agents_to_observe

  def Plan(self, delta_time, world):
    observed_world = world
    if not isinstance(world, ObservedWorld):
      observed_world = world.Observe(self._agents_to_observe)[0]
    
    observed_state = self._configuration._observer.observe(
      world=observed_world,
      agents_to_observe=self._agents_to_observe)
    action = self._configuration._agent.act(observed_state)
    self._dynamic_behavior_model.SetLastAction(action)
    trajectory = self._dynamic_behavior_model.Plan(delta_time, observed_world)
    super(BARKMLBehaviorModel, self).SetLastTrajectory(trajectory)
    return trajectory

  def Clone(self):
    return self
