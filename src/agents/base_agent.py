from abc import ABC, abstractmethod

class BaseAgent(ABC):
  """Base class for bark-ml agents
  """
  def __init__(self,
               agent):
    self._agent = agent

  @abstractmethod
  def execute(self, state):
    """Returns an action given a state
    """
    pass

  @abstractmethod
  def reset(self):
    """Resets all the internal states
    """
    pass