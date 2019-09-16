from abc import ABC, abstractmethod

class StateObserver(ABC):
  def __init__(self,
               params):
    self._params = params

  @abstractmethod
  def observe(self, world, agents_to_observe):
    """Observes the world
    
    Arguments:
        world {bark.world} -- BARK world
        agents_to_observe {list(int)} -- ids of agents to observe
    
    Returns:
        np.array -- concatenated state array
    """
    pass
  
  def _select_state_by_index(self, state):
    """selects a subset of an array using the state definition
    
    Arguments:
        state {np.array} -- full state space
    
    Returns:
        np.array -- reduced state space
    """
    return state[self._state_definition]

  @abstractmethod
  def reset(self, world, agents_to_observe):
    pass # return world

  @property
  def observation_space(self):
    pass

  def reset(self, world, agents_to_observe):
    # TODO(@hart); make generic for multi agent planning
    return world