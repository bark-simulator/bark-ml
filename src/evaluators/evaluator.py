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

  @abstractmethod
  def evaluate(self, world):
    """Returns an RL tuple
    
    Arguments:
        world {bark.world} -- World provided by bark
    
    Returns:
        (rewad, done, info) -- RL-tuple
    """
    pass

  def reset(self, world, agents_to_evaluate):
    if len(agents_to_evaluate) != 1:
      raise ValueError("Invalid number of agents provided for GoalReached \
                        evaluation, number= {}" \
                        .format(len(agents_to_evaluate)))
    self._eval_agent = agents_to_evaluate[0]
    world.clear_evaluators()
    self._add_evaluators()
    for key, evaluator in self._evaluators.items():
      world.add_evaluator(key, evaluator)
    return world