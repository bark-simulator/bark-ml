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

  def evaluate(self, world, action):
    """Evaluates the passed world
    
    Arguments:
        world {bark.world} -- World containing all information
    
    Returns:
        (reward, status, evaluation results) -- Rl-tuple
    """
    eval_results = None
    reward = 0.
    done = False
    # TODO(@hart); make generic for multi agent planning
    if self._eval_agent in world.agents:
      eval_results = world.Evaluate()
      reward, done, eval_results = self._evaluate(world, eval_results, action)
    return reward, done, eval_results

  def reset(self, world, agents_to_evaluate):
    # if len(agents_to_evaluate) != 1:
    #   raise ValueError("Invalid number of agents provided for GoalReached \
    #                     evaluation, number= {}" \
    #                     .format(len(agents_to_evaluate)))
    # TODO(@hart); make generic for multi agent planning
    self._eval_agent = agents_to_evaluate[0]
    world.ClearEvaluators()
    self._add_evaluators()
    for key, evaluator in self._evaluators.items():
      world.AddEvaluator(key, evaluator)
    return world

  def set_viewer(self, viewer):
    self._viewer = viewer