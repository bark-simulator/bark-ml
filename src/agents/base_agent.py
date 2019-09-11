from abc import ABC, abstractmethod

class BaseAgent(ABC):
  """Base class for bark-ml agents
  """
  def __init__(self,
               agent=None,
               params=None):
    self._agent = agent
    self._params = params


  @abstractmethod
  def reset(self):
    """Resets all the internal states
    """
    pass