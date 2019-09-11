from abc import ABC, abstractmethod

from modules.runtime.commons.parameters import ParameterServer

class StateEvaluator(ABC):
  """Evaluates the state of the environment
     e.g., if a collision has happend
  """
  def __init__(self,
               params=ParameterServer()):
    self._params = params

  @abstractmethod
  def get_evaluation(self, world):
    """Returns an RL tuple
    
    Arguments:
        world {bark.world} -- World provided by bark
    
    Returns:
        (rewad, done, info) -- RL-tuple
    """
    pass

  @abstractmethod
  def reset(self, world, agents_to_evaluate):
    """Returns a world with evaluators
    """