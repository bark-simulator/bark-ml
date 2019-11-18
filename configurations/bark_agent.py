import numpy as np
from bark.models.behavior import BehaviorModel, DynamicBehaviorModel
from bark.models.dynamic import SingleTrackModel
from modules.runtime.commons.parameters import ParameterServer
from configurations.base_configuration import BaseConfiguration
from bark.world import World, ObservedWorld

# include all configurations
from configurations.sac_highway.configuration import SACHighwayConfiguration

class BARKMLBehaviorModel(DynamicBehaviorModel):
  """This class makes a trained agent available as BehaviorModel
     in BARK. There will be no evaluation from the side of the 
     agent. Thus, a BARKAgent will only use the observer and agent
     from the configuration.
  """
  def __init__(self,
               configuration=None,
               dynamic_model = None,
               agents_to_observe=None):
    DynamicBehaviorModel.__init__(self, dynamic_model, configuration._params)
    self._configuration = configuration
    self._dynamic_model = dynamic_model
    self._agents_to_observe = agents_to_observe

  def plan(self, delta_time, world):
    # world is a observed world here
    observed_state = self._configuration._observer.observe(
      world=world,
      agents_to_observe=self._agents_to_observe)
    action = self._configuration._agent.act(observed_state)

    # need to pass the action
    super(BARKMLBehaviorModel, self).set_last_action(np.array([1., 2.]))

    observed_world = world
    if not isinstance(observed_world, ObservedWorld):
      observed_world = world.observe(self._agents_to_observe)[0]

    return super(BARKMLBehaviorModel, self).plan(delta_time, observed_world)

  def clone(self):
    return self
