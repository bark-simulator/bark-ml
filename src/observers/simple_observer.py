
import numpy as np
from bark.models.dynamic import StateDefinition
from modules.runtime.commons.parameters import ParameterServer
import math
import operator
from src.commons.spaces import BoundedContinuous, Discrete

from src.observers.observer import StateObserver


class SimpleObserver(StateObserver):
  def __init__(self, params=ParameterServer()):
    StateObserver.__init__(self, params)
    self._state_definition = [int(StateDefinition.X_POSITION),
                              int(StateDefinition.Y_POSITION),
                              int(StateDefinition.THETA_POSITION),
                              int(StateDefinition.VEL_POSITION)]

  def observe(self, world, agents_to_observe):
    """Observes the world
    
    Arguments:
        world {bark.world} -- BARK world
        agents_to_observe {list(int)} -- ids of agents to observe
    
    Returns:
        np.array -- concatenated state array
    """
    # observe world
    observed_worlds =  world.observe(agents_to_observe)
    # concatenated state (zero)
    concatenated_state = np.zeros(len(world.agents)*self._len_state)
    for i, (key, agent) in enumerate(world.agents.items()):
      reduced_state = self._select_state_by_index(agent.state)
      starts_id = i*self._len_state
      concatenated_state[starts_id:starts_id+self._len_state] = reduced_state
    return concatenated_state

  @property
  def observation_space(self):
    return Discrete()

  @property
  def _len_state(self):
    return len(self._state_definition)


