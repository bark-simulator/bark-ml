from abc import ABC, abstractmethod

class StateObserver(ABC):
  def __init__(self,
               params):
    self._params = params

  @abstractmethod
  def observe(self, world, agents_to_observe):
    pass

  @abstractmethod
  def reset(self, world, agents_to_observe):
    pass # return world

  @property
  def observation_space(self):
    pass
