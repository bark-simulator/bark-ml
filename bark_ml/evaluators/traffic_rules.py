# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from typing import Tuple
import numpy as np
from bark.core.world import World, ObservedWorld
from bark.core.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionEgoAgent, \
  EvaluatorStepCount, EvaluatorDrivableArea
from bark.core.world.evaluation.ltl import EvaluatorLTL, SafeDistanceLabelFunction
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.evaluators.evaluator import BaseEvaluator
from bark.core.world.evaluation.ltl import EvaluatorLTL
from bark_ml.evaluators.reward_shaping import RewardShapingEvaluator

class EvaluatorTrafficRules(RewardShapingEvaluator):
  """Sparse reward evaluator returning +1 for reaching the goal,
  -1 for having a collision or leaving the drivable area."""

  def __init__(self,
               params=ParameterServer(),
               eval_agent=None):
    RewardShapingEvaluator.__init__(self, params)
    self._goal_reward = \
      self._params["ML"]["GoalReachedEvaluator"]["GoalReward",
        "Reward for reaching the goal.",
        1.]
    self._col_penalty = \
      self._params["ML"]["GoalReachedEvaluator"]["CollisionPenalty",
        "Reward given for a collisions.",
        -1.]
    self._max_steps = \
      self._params["ML"]["GoalReachedEvaluator"]["MaxSteps",
        "Maximum steps per episode.",
        60]
    self._rules = {
      "ltl_formula": "G sd_front",
      "label_functions": [
        SafeDistanceLabelFunction(
          "sd_front",
          False, 1.0, 1.0, -7.84, -7.84, True, 4, False, 2.0, False)]
      }
    self._eval_agent = eval_agent
    self.traffic_rule_violation_pre = 0
    self.traffic_rule_violation_post = 0
    self.traffic_rule_violations = 0

  #def _add_evaluators(self) -> dict:
  def _add_evaluators(self):
    evaluators = {}
    evaluators["goal_reached"] = EvaluatorGoalReached()
    evaluators["collision"] = EvaluatorCollisionEgoAgent()
    evaluators["step_count"] = EvaluatorStepCount()
    evaluators["drivable_area"] = EvaluatorDrivableArea()
    evaluators["evaluator_ltl"] = EvaluatorLTL(
      agent_id=self._eval_id, **self._rules)
    return evaluators

  def _evaluate(self,
    observed_world: ObservedWorld,
    eval_results: dict,
    action: np.ndarray):
    #action: np.ndarray) -> Tuple(float, bool, dict):
    """Returns information about the current world state."""
    # Todo
    reward, _, _ = super()._evaluate(observed_world, eval_results, action)
    done = False
    success = eval_results["goal_reached"]
    step_count = eval_results["step_count"]
    collision = eval_results["collision"] or \
        eval_results["drivable_area"] or \
        (step_count > self._max_steps)

    # TODO: integrate traffic rule violation
    self.traffic_rule_violation_post = eval_results["evaluator_ltl"]
    if self.traffic_rule_violation_post < self.traffic_rule_violation_pre:
        self.traffic_rule_violation_pre = self.traffic_rule_violation_post
    self.current_traffic_rule_violations = self.traffic_rule_violation_post - self.traffic_rule_violation_pre
    self.traffic_rule_violations = self.traffic_rule_violations + self.current_traffic_rule_violations
    self.traffic_rule_violation_pre = self.traffic_rule_violation_post
    #print('Traffic rule violations so far:\n')
    #print(self.traffic_rule_violations)
    #print('\nTraffic rule violation currently:\n')
    #print(self.current_traffic_rule_violations)

    if success or collision or step_count > self._max_steps:
      done = True
    if collision:
      success = 0
      eval_results["goal_reached"] = 0
    # calculate reward
    print("\n Traffic Rule Violation: ")
    print(self.current_traffic_rule_violations)
    reward -= collision * self._col_penalty + \
      success * self._goal_reward - self.current_traffic_rule_violations
    #print('\nReward:\n')
    #print(reward)
    return reward, done, eval_results

  def Reset(self, world: World, eval_id: int):
    return super(EvaluatorTrafficRules, self).Reset(world, eval_id)
