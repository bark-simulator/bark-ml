from abc import ABC, abstractmethod

from modules.runtime.commons.parameters import ParameterServer

class StateEvaluator(ABC):
  """Evaluates the state of the environment
     e.g., if a collision has happend
  """
  def __init__(self,
               params=ParameterServer()):
    self._params = params
    self._evaluators = {}
    self._viewer = None

  def evaluate(self, observed_world, action, observed_state):
    """Evaluates the observed world
    """
    eval_results, reward, done = None, 0., False
    eval_results = observed_world.Evaluate()
    reward, done, eval_results = self._evaluate(
      observed_world, eval_results, action, observed_state)
    return reward, done, eval_results

  def reset(self, world):
    world.ClearEvaluators()
    self._add_evaluators()
    for key, evaluator in self._evaluators.items():
      world.AddEvaluator(key, evaluator)
    return world

  def set_viewer(self, viewer):
    self._viewer = viewer