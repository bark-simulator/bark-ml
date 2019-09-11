from abc import ABC, abstractmethod

class BaseAgent(ABC):
  """Base class for bark-ml agents
  """
  def __init__(self,
               agent):
    self._agent = agent

  @abstractmethod
  def execute(self, state):
    pass

  @abstractmethod
  def reset(self):
    pass