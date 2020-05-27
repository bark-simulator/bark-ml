from abc import ABC, abstractmethod

class BaseAgent(ABC):
  """Base class for bark-ml agents
  """
  def __init__(self,
               environment=None,
               agent=None,
               params=None):
    self._env = environment
    self._agent = agent
    self._params = params

  @abstractmethod
  def reset(self):
    """Resets all the internal states
    """
    pass

  @abstractmethod
  def act(self, state):
    """Returns an action based on a state
    """
    pass

  @abstractmethod
  def save(self):
    """Saves an agent
    """
    pass

  @abstractmethod
  def load(self):
    """Loads an agent
    """
    pass