from abc import ABC, abstractmethod
class BaseRunner(ABC):
  def __init__(self,
               runtime=None,
               agent=None,
               params=None):
    self._params = params
    self._runtime = runtime
    self._agent = agent
    self._train_metric = None
    self._eval_metrics = None
    self._render_evaluation = True

  @abstractmethod
  def collect_initial_episodes(self):
    """Collects initial episodes without prior training
    """
    pass

  @abstractmethod
  def train(self):
    """Trains the agent for a given period
    """

  @abstractmethod
  def evaluate(self):
    """Evaluates the agent for a given number of episodes
    """
    pass

  @abstractmethod
  def visualize(self, num_episodes):
    """Visualizes the agent in the runtime (num_episodes)
    """
    pass

  def reset(self):
    """Resets the runtime and the agent
    """
    self._runtime.reset()
    self._agent.reset()

    