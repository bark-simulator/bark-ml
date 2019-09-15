from modules.models.behavior import BehaviorModel
from modules.runtime.commons.parameters import ParameterServer
from configurations.base_configuration import BaseConfiguration

# include all configurations
from configurations.sac_highway.configuration import SACHighwayConfiguration

class BARKAgent(BehaviorModel):
  """This class makes a trained agent available as BehaviorModel
     in BARK. There will be no evaluation from the side of the 
     agent. Thus, a BARKAgent will only use the observer and agent
     from the configuration.
  """
  def __init__(self,
               params=ParameterServer(),
               configuration=BaseConfiguration()):
    BehaviorModel.__init__(self, params)
    self._configuration = configuration

  def plan(self, delta_time, world):
    # TODO(@hart): per agent there should be only one eval_id!
    observed_state = self._configuration._observer.observe(
      world=world,
      agents_to_observe=self._configuration._runtime._scenario._eval_agent_ids)
    action = self._configuration._agent.act(observed_state)
    world.agents[self._configuration._runtime._scenario._eval_agent_ids] \
      ._behavior_model.set_action(action)

  def clone(self):
    return self