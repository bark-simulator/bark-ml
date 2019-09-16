from abc import ABC, abstractmethod

class StateObserver(ABC):
  def __init__(self,
               params):
    self._params = params
    self._world_x_range = None
    self._world_y_range = None

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

  def reset(self, world, agents_to_observe):
    bb = world.bounding_box
    self._world_x_range = [bb[0].x(), bb[1].x()]
    self._world_y_range = [bb[0].y(), bb[1].y()]
    return world

  @property
  def observation_space(self):
    pass