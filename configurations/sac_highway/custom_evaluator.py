import numpy as np
from bark.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionAgents, \
  EvaluatorCollisionDrivingCorridor, EvaluatorStepCount
from modules.runtime.commons.parameters import ParameterServer
from bark.geometry import *

from src.evaluators.goal_reached import GoalReached

class CustomEvaluator(GoalReached):
  """Shows the capability of custom elements inside
     a configuration.
  """
  def __init__(self,
               params=ParameterServer(),
               eval_agent=None):
    GoalReached.__init__(self,
                         params,
                         eval_agent)

  def _add_evaluators(self):
    self._evaluators["goal_reached"] = EvaluatorGoalReached()
    self._evaluators["collision"] = \
      EvaluatorCollisionAgents()
    self._evaluators["step_count"] = EvaluatorStepCount()

  def _evaluate(self, world, eval_results):
    """Returns information about the current world state
    """
    done = False
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    step_count = eval_results["step_count"]
    
    # calculate reward
    reward = collision * self._collision_penalty + \
      success * self._goal_reward
    
    # determine if terminal
    if success or collision or step_count > self._max_steps:
      done = True

    return reward, done, eval_results
    
