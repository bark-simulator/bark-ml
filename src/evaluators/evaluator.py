from abc import ABC, abstractmethod

from modules.runtime.commons.parameters import ParameterServer

class StateEvaluator(ABC):
  def __init__(self,
               params=ParameterServer()):
    self._params = params

  @abstractmethod
  def get_evaluation(self, world):
    return # reward, done, info

  @abstractmethod
  def reset(self, world, agents_to_evaluate):
    return # world