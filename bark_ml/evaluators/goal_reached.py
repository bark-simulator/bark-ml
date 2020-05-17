# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from bark.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionEgoAgent, \
  EvaluatorStepCount, EvaluatorDrivableArea
from bark_project.modules.runtime.commons.parameters import ParameterServer

from bark_ml.evaluators.evaluator import StateEvaluator


class GoalReached(StateEvaluator):
  def __init__(self,
               params=ParameterServer(),
               eval_agent=None):
    StateEvaluator.__init__(self, params)
    self._GoalReward = \
      self._params["ML"]["GoalReachedEvaluator"]["GoalReward",
        "Reward for reaching the goal.",
        1.]
    self._CollisionPenalty = \
      self._params["ML"]["GoalReachedEvaluator"]["CollisionPenalty",
        "Reward given for a collisions.",
        -1.]
    self._MaxSteps = \
      self._params["ML"]["GoalReachedEvaluator"]["MaxSteps",
        "Maximum steps per episode.",
        50]
    self._eval_agent = eval_agent

  def _add_evaluators(self):
    self._evaluators["goal_reached"] = EvaluatorGoalReached()
    self._evaluators["collision"] = EvaluatorCollisionEgoAgent()
    self._evaluators["step_count"] = EvaluatorStepCount()
    self._evaluators["drivable_area"] = EvaluatorDrivableArea()

  def _evaluate(self, observed_world, eval_results, action):
    """Returns information about the current world state
    """
    done = False
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    drivable_area = eval_results["drivable_area"]
    step_count = eval_results["step_count"]
    # determine whether the simulation should terminate
    if success or collision or drivable_area or step_count > self._MaxSteps:
      done = True
    # calculate reward
    reward = collision * self._CollisionPenalty + \
      success * self._GoalReward
    return reward, done, eval_results
    
  def Reset(self, world):
    return super(GoalReached, self).Reset(world)